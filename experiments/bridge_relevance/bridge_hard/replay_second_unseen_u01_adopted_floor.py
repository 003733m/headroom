#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONTROL = Path(__file__).resolve().parents[3]

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

RAW = (
    Path.home()
    / "headroom-second-unseen-adopted-20260905"
    / "results"
    / "U01-ON"
)

WIRE = RAW / "codex-wire"

SELECTION = (
    HERE
    / "second_unseen_adopted_bridge_selection.json"
)

OUT = (
    HERE
    / "second_unseen_u01_adopted_floor_replay_results.json"
)

EXACT_PATH = HERE / "run_exact_wire_v3_replay.py"
MATCHED_PATH = HERE / "run_hard_matched_context_replay.py"


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        die(f"cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# Production code must still equal the frozen algorithm.
cp = subprocess.run(
    [
        "git",
        "diff",
        "--quiet",
        FREEZE,
        "--",
        "headroom",
    ],
    cwd=CONTROL,
)

if cp.returncode != 0:
    die("production tree differs from frozen 89da algorithm")


exact = load_module(
    "second_unseen_exact",
    EXACT_PATH,
)

matched = load_module(
    "second_unseen_matched",
    MATCHED_PATH,
)

# Some historical replay helpers have their original V3 freeze constant.
# The handler itself must use the current frozen production tree.
if hasattr(exact, "FREEZE"):
    exact.FREEZE = FREEZE


import headroom.trajectory_relevance as tr  # noqa: E402
import headroom.proxy.handlers.openai as openai_handler  # noqa: E402
from headroom.providers import OpenAIProvider  # noqa: E402
from headroom.tokenizer import Tokenizer  # noqa: E402


OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "local_shell_call_output",
}

CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "local_shell_call",
}


def body(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(errors="replace"))
    value = d.get("body")

    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, dict):
        die(f"wire body missing in {path.name}")

    return value


def output_text(item: dict[str, Any]) -> str:
    value = item.get("output")

    if isinstance(value, str):
        return value

    return tr._responses_text(value)


def call_map(items: list[Any]) -> dict[str, dict[str, Any]]:
    result = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("type") not in CALL_TYPES:
            continue

        call_id = item.get("call_id")

        if isinstance(call_id, str):
            result[call_id] = item

    return result


def first_live_adopted_event() -> dict[str, Any]:
    seen: set[tuple[str, str]] = set()

    threshold = (
        openai_handler.OpenAIHandlerMixin
        .OPENAI_RESPONSES_ROUTER_MIN_BYTES
    )

    for wire_path in sorted(
        WIRE.glob("*_http_inbound_request.json")
    ):
        payload = body(wire_path)
        items = payload.get("input")

        if not isinstance(items, list):
            continue

        calls = call_map(items)

        for before_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            if item.get("type") not in OUTPUT_TYPES:
                continue

            call_id = item.get("call_id")

            if not isinstance(call_id, str):
                continue

            target = output_text(item)

            identity = (
                call_id,
                hashlib.sha256(
                    target.encode(
                        "utf-8",
                        errors="replace",
                    )
                ).hexdigest(),
            )

            if identity in seen:
                continue

            seen.add(identity)

            command = (
                tr._responses_producing_search_command(
                    items,
                    before_index=before_index,
                )
            )

            if not command:
                continue

            context = (
                tr.build_responses_search_relevance_context(
                    items,
                    before_index=before_index,
                    target_content=target,
                )
            )

            if not context:
                continue

            adopted = (
                tr._responses_adopted_bridge_identifiers(
                    items,
                    before_index=before_index,
                    trajectory_context=context,
                )
            )

            if not adopted:
                continue

            producer = calls.get(call_id, {})
            tool_name = str(
                producer.get("name") or ""
            ).casefold()

            structural = (
                openai_handler
                ._responses_trajectory_search_eligible(
                    exact.make_handler(
                        True,
                        str(
                            payload.get("model")
                            or "gpt-5.6-terra"
                        ),
                    )._content_router,
                    target,
                )
            )

            bash_search = (
                tool_name == "bash"
                and bool(
                    re.search(
                        r"(^|\s)(rg|grep)\s",
                        command,
                        re.I,
                    )
                )
            )

            byte_eligible = (
                len(
                    target.encode(
                        "utf-8",
                        errors="replace",
                    )
                )
                >= threshold
            )

            handler_gate = (
                byte_eligible
                and (
                    structural
                    or bash_search
                )
            )

            if not handler_gate:
                continue

            return {
                "wire_path": wire_path,
                "payload": payload,
                "items": items,
                "before_index": before_index,
                "call_id": call_id,
                "target": target,
                "command": command,
                "context": context,
                "adopted": tuple(adopted),
                "structural": structural,
                "bash_search": bash_search,
                "threshold": threshold,
            }

    die("no production-eligible adopted event found")


event = first_live_adopted_event()

original_helper = (
    tr._responses_adopted_bridge_identifiers
)


def run_floor(enabled: bool):
    if enabled:
        tr._responses_adopted_bridge_identifiers = (
            original_helper
        )
    else:
        tr._responses_adopted_bridge_identifiers = (
            lambda *args, **kwargs: ()
        )

    try:
        payload = copy.deepcopy(event["payload"])

        model = str(
            payload.get("model")
            or "gpt-5.6-terra"
        )

        handler = exact.make_handler(
            True,
            model,
        )

        result = (
            handler._compress_openai_responses_payload(
                payload,
                model=model,
                request_id=(
                    "second_unseen_u01_"
                    + (
                        "floor_on"
                        if enabled
                        else "floor_off"
                    )
                ),
                timing={},
                client="opencode",
                savings_tags={},
            )
        )

        return result

    finally:
        tr._responses_adopted_bridge_identifiers = (
            original_helper
        )


A = run_floor(False)
D = run_floor(True)


def compressed_target(result) -> str | None:
    payload = result[0]
    items = payload.get("input")

    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            continue

        if (
            item.get("call_id")
            == event["call_id"]
            and item.get("type")
            in OUTPUT_TYPES
        ):
            return output_text(item)

    return None


A_target = compressed_target(A)
D_target = compressed_target(D)

if A_target is None or D_target is None:
    die("target output missing after replay")


# Historical oracle is constructed only now, AFTER natural task selection.
selection = json.loads(
    SELECTION.read_text()
)

task = next(
    x
    for x in selection["tasks"]
    if x["task_id"] == "U01"
)

oracle = matched.historical_oracle(task)

records = matched.parse_records(
    event["target"]
)

model_for_tokens = str(
    event["payload"].get("model")
    or "gpt-5.6-terra"
)

token_model = model_for_tokens.split("/")[-1]

provider = OpenAIProvider()
tokenizer = Tokenizer(
    provider.get_token_counter(token_model),
    token_model,
)


rows = []

for rec in records:
    fix_adjacent = matched.is_fix_adjacent(
        rec,
        oracle,
    )

    keep_A = matched.retained(
        A_target,
        rec,
    )

    keep_D = matched.retained(
        D_target,
        rec,
    )

    rows.append({
        "path": rec["path"],
        "line": rec["line"],
        "text": rec["text"],
        "fix_adjacent": fix_adjacent,
        "A_keep": keep_A,
        "D_keep": keep_D,
        "record_tokens": tokenizer.count_text(
            rec["raw"]
        ),
    })


critical = [
    r for r in rows
    if r["fix_adjacent"]
]

rescues = [
    r for r in critical
    if (
        not r["A_keep"]
        and r["D_keep"]
    )
]

regressions = [
    r for r in critical
    if (
        r["A_keep"]
        and not r["D_keep"]
    )
]

newly_kept = [
    r for r in rows
    if (
        not r["A_keep"]
        and r["D_keep"]
    )
]

new_critical = [
    r for r in newly_kept
    if r["fix_adjacent"]
]

new_noncritical = [
    r for r in newly_kept
    if not r["fix_adjacent"]
]


def sha(text: str) -> str:
    return hashlib.sha256(
        text.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()


summary = {
    "experiment": (
        "second_unseen_adopted_bridge_validation"
    ),
    "task": "U01",
    "production_freeze": FREEZE,
    "evaluation": (
        "deterministic exact-request adopted-floor A/D replay"
    ),
    "comparison": {
        "A": (
            "frozen trajectory relevance with "
            "adopted preservation floor disabled"
        ),
        "D": (
            "same frozen trajectory relevance with "
            "production adopted preservation floor enabled"
        )
    },
    "natural_event": {
        "command": event["command"],
        "adopted_identifiers": list(
            event["adopted"]
        ),
        "target_bytes": len(
            event["target"].encode(
                "utf-8",
                errors="replace",
            )
        ),
        "target_sha256": sha(
            event["target"]
        ),
        "structural_search_eligible": (
            event["structural"]
        ),
        "bash_search_eligible": (
            event["bash_search"]
        ),
        "handler_search_gate": True
    },
    "replay": {
        "A_target_sha256": sha(A_target),
        "D_target_sha256": sha(D_target),
        "A_target_bytes": len(
            A_target.encode(
                "utf-8",
                errors="replace",
            )
        ),
        "D_target_bytes": len(
            D_target.encode(
                "utf-8",
                errors="replace",
            )
        ),
        "target_outputs_differ": (
            A_target != D_target
        ),
        "A_tokens_saved_request": int(A[2]),
        "D_tokens_saved_request": int(D[2])
    },
    "retention": {
        "parsed_records": len(rows),
        "historical_fix_adjacent_records": (
            len(critical)
        ),
        "A_critical_kept": sum(
            r["A_keep"]
            for r in critical
        ),
        "D_critical_kept": sum(
            r["D_keep"]
            for r in critical
        ),
        "critical_DROP_to_KEEP": len(
            rescues
        ),
        "critical_KEEP_to_DROP": len(
            regressions
        ),
        "newly_kept_records": len(
            newly_kept
        ),
        "newly_kept_critical_records": len(
            new_critical
        ),
        "newly_kept_noncritical_records": len(
            new_noncritical
        ),
        "additional_critical_record_tokens": sum(
            r["record_tokens"]
            for r in new_critical
        ),
        "additional_noncritical_record_tokens": sum(
            r["record_tokens"]
            for r in new_noncritical
        )
    },
    "rescued_critical_records": [
        {
            "path": r["path"],
            "line": r["line"],
            "text": r["text"],
        }
        for r in rescues
    ],
    "regressed_critical_records": [
        {
            "path": r["path"],
            "line": r["line"],
            "text": r["text"],
        }
        for r in regressions
    ],
    "raw_wire_committed": False
}


if len(critical) == 0:
    result_class = (
        "NO_CRITICAL_OPPORTUNITY"
    )
elif len(rescues) > 0 and len(regressions) == 0:
    result_class = (
        "UNSEEN_CRITICAL_RESCUE"
    )
elif A_target == D_target:
    result_class = (
        "NATURAL_ACTIVATION_NO_INCREMENTAL_RETENTION"
    )
elif len(rescues) == 0:
    result_class = (
        "RETENTION_CHANGED_NO_CRITICAL_RESCUE"
    )
else:
    result_class = (
        "MIXED_CRITICAL_EFFECT"
    )

summary["result_class"] = result_class


OUT.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


print("=" * 78)
print("SECOND UNSEEN U01 — EXACT ADOPTED FLOOR A/D")
print("=" * 78)

print("command:", event["command"])
print(
    "adopted:",
    list(event["adopted"]),
)

print(
    "target bytes:",
    len(
        event["target"].encode(
            "utf-8",
            errors="replace",
        )
    ),
)

print()
print("A bytes:", len(A_target.encode()))
print("D bytes:", len(D_target.encode()))
print("A == D:", A_target == D_target)

print()
r = summary["retention"]

for key, value in r.items():
    print(f"{key}: {value}")

print()
print("RESULT_CLASS:", result_class)

if rescues:
    print()
    print("RESCUED HISTORICAL-CRITICAL RECORDS")

    for row in rescues:
        print(
            row["path"],
            row["line"],
            repr(row["text"]),
        )

print()
print("result:", OUT)
print("raw wire remains local only")
