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

CONTROL = Path.home() / "projects/headroom"
HERE = CONTROL / "experiments/bridge_relevance/bridge_hard"

RUNTIME = Path(
    "/tmp/headroom-second-unseen-adopted-runtime-89da0898"
).resolve()

ROOT = (
    Path.home()
    / "headroom-second-unseen-adopted-20260905"
)

RESULT_ROOT = ROOT / "results"

PROTOCOL = (
    HERE
    / "second_unseen_adopted_bridge_mechanism_protocol.json"
)

SELECTION = (
    HERE
    / "second_unseen_adopted_bridge_selection.json"
)

OUT = (
    HERE
    / "second_unseen_adopted_bridge_mechanism_results.json"
)

EXACT_PATH = HERE / "run_exact_wire_v3_replay.py"

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "local_shell_call_output",
}

RECORD_RE = re.compile(
    r"^(?P<path>.+?)"
    r"(?P<sep>[:-])"
    r"(?P<line>\d+)"
    r"(?P=sep)"
    r"(?P<text>.*)$"
)

HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)"
    r"(?:,(?P<old_count>\d+))?"
    r" \+(?P<new_start>\d+)"
    r"(?:,(?P<new_count>\d+))?"
    r" @@"
)


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def git(*args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=CONTROL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if cp.returncode:
        die(cp.stdout)
    return cp.stdout


def verify_committed_unchanged(path: Path) -> None:
    rel = str(path.relative_to(CONTROL))

    cp = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=CONTROL,
    )
    if cp.returncode:
        die(f"not committed: {rel}")

    status = git(
        "status",
        "--porcelain=v1",
        "--",
        rel,
    ).strip()

    if status:
        die(f"locked file modified: {rel}")


def sha(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def normalize_path(path: str) -> str:
    p = path.replace("\\", "/").strip()

    while p.startswith("./"):
        p = p[2:]

    return p


def text_of(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []

        for item in value:
            t = text_of(item)
            if t:
                parts.append(t)

        return "\n".join(parts)

    if isinstance(value, dict):
        for key in (
            "text",
            "output",
            "content",
            "stdout",
            "stderr",
        ):
            if key in value:
                t = text_of(value[key])
                if t:
                    return t

    return ""


def load_exact():
    spec = importlib.util.spec_from_file_location(
        "second_unseen_exact_helpers",
        EXACT_PATH,
    )

    if spec is None or spec.loader is None:
        die("cannot load exact replay helper")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Reuse helper machinery, but this experiment's
    # production freeze is the later 89da commit.
    if hasattr(module, "FREEZE"):
        module.FREEZE = FREEZE

    return module


exact = load_exact()

# These imports MUST resolve to frozen 89da runtime.
import headroom.trajectory_relevance as tr  # noqa: E402
from headroom.providers import OpenAIProvider  # noqa: E402
from headroom.tokenizer import Tokenizer  # noqa: E402


module_path = Path(tr.__file__).resolve()

if RUNTIME not in module_path.parents:
    die(
        "headroom imported from wrong tree: "
        f"{module_path}"
    )


def body_from_wire(path: Path) -> dict[str, Any]:
    d = json.loads(
        path.read_text(errors="replace")
    )

    body = d.get("body")

    if isinstance(body, str):
        body = json.loads(body)

    if not isinstance(body, dict):
        die(f"invalid body: {path}")

    return body


def target_output(
    payload: dict[str, Any],
    call_id: str,
) -> str:
    items = payload.get("input")

    if not isinstance(items, list):
        return ""

    for item in items:
        if (
            isinstance(item, dict)
            and item.get("type") in OUTPUT_TYPES
            and item.get("call_id") == call_id
        ):
            return text_of(item.get("output"))

    return ""


def historical_oracle(
    source_commit: str,
    production_files: list[str],
) -> dict[str, dict[int, str]]:
    """
    Parent-version non-empty source lines within +/-5 lines
    of the old side of each historical production fix hunk.
    """
    parent = git(
        "rev-parse",
        f"{source_commit}^",
    ).strip()

    result: dict[str, dict[int, str]] = {}

    for path in production_files:
        try:
            parent_text = git(
                "show",
                f"{parent}:{path}",
            )
        except SystemExit:
            continue

        parent_lines = parent_text.splitlines()

        diff = git(
            "diff",
            "--unified=0",
            parent,
            source_commit,
            "--",
            path,
        )

        line_numbers: set[int] = set()

        for raw in diff.splitlines():
            m = HUNK_RE.match(raw)

            if not m:
                continue

            old_start = int(m.group("old_start"))

            old_count = (
                int(m.group("old_count"))
                if m.group("old_count") is not None
                else 1
            )

            width = max(1, old_count)

            start = max(
                1,
                old_start - 5,
            )

            end = min(
                len(parent_lines),
                old_start + width - 1 + 5,
            )

            line_numbers.update(
                range(start, end + 1)
            )

        result[path] = {
            line: parent_lines[line - 1]
            for line in sorted(line_numbers)
            if (
                1 <= line <= len(parent_lines)
                and parent_lines[line - 1].strip()
            )
        }

    return result


def parse_records(target: str) -> list[dict[str, Any]]:
    rows = []

    for raw in target.splitlines():
        m = RECORD_RE.match(raw)

        if not m:
            continue

        rows.append({
            "raw": raw,
            "path": normalize_path(
                m.group("path")
            ),
            "line": int(m.group("line")),
            "text": m.group("text"),
        })

    return rows


def is_fix_adjacent(
    rec: dict[str, Any],
    oracle: dict[str, dict[int, str]],
) -> bool:
    rec_path = normalize_path(rec["path"])
    rec_text = rec["text"].strip()
    rec_line = int(rec["line"])

    if not rec_text:
        return False

    for path, lines in oracle.items():
        norm = normalize_path(path)

        if not (
            rec_path == norm
            or rec_path.endswith("/" + norm)
        ):
            continue

        expected = lines.get(rec_line)

        if (
            expected is not None
            and expected.strip() == rec_text
        ):
            return True

        # Same conservative fallback used by the
        # earlier matched historical replay.
        for line, text in lines.items():
            if (
                abs(line - rec_line) <= 3
                and text.strip() == rec_text
            ):
                return True

    return False


def strip_adopted_marker(context: str) -> str:
    return "\n".join(
        line
        for line in context.splitlines()
        if not line.strip().startswith(
            "Adopted bridge identifiers:"
        )
    ).strip()


def replay(
    payload: dict[str, Any],
    model: str,
    *,
    preservation_enabled: bool,
) -> dict[str, Any]:
    """
    Both A and D retain frozen trajectory relevance.

    The ONLY treatment difference is whether the frozen
    adopted-bridge helper may emit preservation identifiers.
    """
    original = (
        tr._responses_adopted_bridge_identifiers
    )

    try:
        if not preservation_enabled:

            def disabled(*args, **kwargs):
                return ()

            tr._responses_adopted_bridge_identifiers = (
                disabled
            )

        handler = exact.make_handler(
            True,
            model,
        )

        result = (
            handler._compress_openai_responses_payload(
                copy.deepcopy(payload),
                model=model,
                request_id=(
                    "second_unseen_D"
                    if preservation_enabled
                    else "second_unseen_A"
                ),
                timing={},
                client="opencode",
                savings_tags={},
            )
        )

        return {
            "payload": result[0],
            "modified": bool(result[1]),
            "tokens_saved": int(result[2]),
            "transforms": list(result[3]),
            "reason": result[4],
            "input_bytes": int(result[5]),
            "output_bytes": int(result[6]),
            "attempted_input_tokens": int(
                result[7]
            ),
        }

    finally:
        tr._responses_adopted_bridge_identifiers = (
            original
        )


def main() -> None:
    verify_committed_unchanged(PROTOCOL)
    verify_committed_unchanged(SELECTION)

    protocol = json.loads(
        PROTOCOL.read_text()
    )

    if (
        protocol.get("status")
        != "locked_before_second_unseen_retention_outcome"
    ):
        die("mechanism protocol not locked")

    if protocol.get("production_freeze") != FREEZE:
        die("freeze mismatch")

    selected = protocol["selected_task"]

    if selected["task_id"] != "U01":
        die("expected U01")

    source_commit = selected["source_commit"]

    selection = json.loads(
        SELECTION.read_text()
    )

    task = next(
        t
        for t in selection["tasks"]
        if t["task_id"] == "U01"
    )

    on_root = RESULT_ROOT / "U01-ON"

    applicability = json.loads(
        (on_root / "applicability.json").read_text()
    )

    event = applicability[
        "first_eligible_event"
    ]

    if event is None:
        die("ON applicability event missing")

    expected_ids = tuple(
        selected["on_adopted_bridges"]
    )

    if not expected_ids:
        die("no locked ON adopted identifiers")

    if tuple(event["adopted_bridges"]) != expected_ids:
        die("adopted IDs differ from locked protocol")

    wire_files = sorted(
        (on_root / "codex-wire").glob(
            "*_http_inbound_request.json"
        )
    )

    request_index = int(
        event["request_index"]
    )

    if not (
        1 <= request_index <= len(wire_files)
    ):
        die("invalid request index")

    wire_path = wire_files[
        request_index - 1
    ]

    payload = body_from_wire(wire_path)

    items = payload.get("input")

    if not isinstance(items, list):
        die("payload input missing")

    before_index = int(
        event["target_item_index"]
    )

    if not (
        0 <= before_index < len(items)
    ):
        die("invalid target index")

    item = items[before_index]

    if not isinstance(item, dict):
        die("target item invalid")

    call_id = str(
        item.get("call_id") or ""
    )

    if call_id != event["call_id"]:
        die("call_id mismatch")

    target = text_of(
        item.get("output")
    )

    if not target:
        die("empty target")

    # Recompute frozen D context before looking at retention.
    context_d = (
        tr.build_responses_search_relevance_context(
            items,
            before_index=before_index,
            target_content=target,
        )
    )

    adopted_d = (
        tr._responses_adopted_bridge_identifiers(
            items,
            before_index=before_index,
            trajectory_context=context_d,
        )
    )

    if tuple(adopted_d) != expected_ids:
        die(
            "frozen production adoption does not "
            "match locked protocol"
        )

    # Verify A differs only by preservation metadata,
    # not by the relevance scoring context.
    original_adopted = (
        tr._responses_adopted_bridge_identifiers
    )

    try:
        tr._responses_adopted_bridge_identifiers = (
            lambda *args, **kwargs: ()
        )

        context_a = (
            tr.build_responses_search_relevance_context(
                items,
                before_index=before_index,
                target_content=target,
            )
        )

    finally:
        tr._responses_adopted_bridge_identifiers = (
            original_adopted
        )

    if (
        strip_adopted_marker(context_a)
        != strip_adopted_marker(context_d)
    ):
        die(
            "A/D relevance contexts differ beyond "
            "the adopted preservation marker"
        )

    model = str(
        payload.get("model")
        or "gpt-5.6-terra"
    )

    # Deterministic exact captured-request replay.
    A = replay(
        payload,
        model,
        preservation_enabled=False,
    )

    D = replay(
        payload,
        model,
        preservation_enabled=True,
    )

    output_a = target_output(
        A["payload"],
        call_id,
    )

    output_d = target_output(
        D["payload"],
        call_id,
    )

    if not output_a:
        die("A target output missing")

    if not output_d:
        die("D target output missing")

    oracle = historical_oracle(
        source_commit,
        task["production_files"],
    )

    records = parse_records(target)

    tokenizer = Tokenizer(
        OpenAIProvider().get_token_counter(
            "gpt-4o"
        ),
        "gpt-4o",
    )

    evaluated = []

    critical_total = 0
    critical_a = 0
    critical_d = 0

    rescue = 0
    regression = 0

    newly_kept_total = 0
    newly_kept_critical = 0
    newly_kept_noncritical = 0

    added_critical_tokens = 0
    added_noncritical_tokens = 0

    for rec in records:
        variants = exact.variants(
            rec["path"],
            rec["line"],
            rec["text"],
        )

        # Conservative scorable gate:
        # the exact matcher must be able to recognize the
        # record in the original uncompressed target.
        scorable = exact.has(
            target,
            variants,
        )

        if not scorable:
            continue

        critical = is_fix_adjacent(
            rec,
            oracle,
        )

        kept_a = exact.has(
            output_a,
            variants,
        )

        kept_d = exact.has(
            output_d,
            variants,
        )

        record_tokens = tokenizer.count_text(
            rec["raw"]
        )

        if critical:
            critical_total += 1

            if kept_a:
                critical_a += 1

            if kept_d:
                critical_d += 1

            if (not kept_a) and kept_d:
                rescue += 1

            if kept_a and (not kept_d):
                regression += 1

        newly_kept = (
            (not kept_a)
            and kept_d
        )

        if newly_kept:
            newly_kept_total += 1

            if critical:
                newly_kept_critical += 1
                added_critical_tokens += (
                    record_tokens
                )
            else:
                newly_kept_noncritical += 1
                added_noncritical_tokens += (
                    record_tokens
                )

        evaluated.append({
            "path": rec["path"],
            "line": rec["line"],
            "text": rec["text"],
            "fix_adjacent": critical,
            "record_tokens": record_tokens,
            "kept_A": kept_a,
            "kept_D": kept_d,
            "newly_kept_by_D": newly_kept,
        })

    result = {
        "status": "complete_single_locked_unseen_result",
        "experiment": (
            "second_unseen_adopted_bridge_mechanism_evaluation"
        ),
        "classification": (
            "retention-outcome-locked deterministic "
            "mechanism replay on unseen natural trajectory"
        ),
        "production_freeze": FREEZE,
        "task_id": "U01",
        "source_commit": source_commit,
        "command": event["command"],
        "call_id": call_id,
        "adopted_identifiers": list(
            expected_ids
        ),
        "target_sha256": sha(target),
        "target_bytes": len(
            target.encode(
                "utf-8",
                errors="replace",
            )
        ),
        "historical_oracle": {
            "definition": (
                "parent-version production lines within "
                "+/-5 lines of historical fix hunk, "
                "with conservative +/-3 line-shift text match"
            ),
            "production_files": task[
                "production_files"
            ],
        },
        "A": {
            "description": (
                "frozen trajectory relevance with "
                "adopted preservation disabled"
            ),
            "modified": A["modified"],
            "tokens_saved": A["tokens_saved"],
            "transforms": A["transforms"],
            "target_output_sha256": sha(
                output_a
            ),
            "target_output_tokens": (
                tokenizer.count_text(output_a)
            ),
        },
        "D": {
            "description": (
                "same frozen trajectory relevance with "
                "natural adopted preservation enabled"
            ),
            "modified": D["modified"],
            "tokens_saved": D["tokens_saved"],
            "transforms": D["transforms"],
            "target_output_sha256": sha(
                output_d
            ),
            "target_output_tokens": (
                tokenizer.count_text(output_d)
            ),
        },
        "metrics": {
            "parsed_records": len(records),
            "scorable_records": len(
                evaluated
            ),
            "historical_fix_adjacent_scorable": (
                critical_total
            ),
            "critical_kept_A": critical_a,
            "critical_kept_D": critical_d,
            "critical_DROP_to_KEEP": rescue,
            "critical_KEEP_to_DROP": regression,
            "newly_kept_records": newly_kept_total,
            "newly_kept_critical_records": (
                newly_kept_critical
            ),
            "newly_kept_noncritical_records": (
                newly_kept_noncritical
            ),
            "additional_critical_record_tokens": (
                added_critical_tokens
            ),
            "additional_noncritical_record_tokens": (
                added_noncritical_tokens
            ),
            "additional_total_record_tokens": (
                added_critical_tokens
                + added_noncritical_tokens
            ),
        },
        "records": evaluated,
        "raw_wire_not_embedded": True,
        "no_additional_agent_run": True,
    }

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print("=" * 78)
    print(
        "SECOND UNSEEN ADOPTED-BRIDGE "
        "MECHANISM RESULT"
    )
    print("=" * 78)

    print("task:", result["task_id"])
    print(
        "adopted:",
        result["adopted_identifiers"],
    )
    print(
        "target bytes:",
        result["target_bytes"],
    )

    m = result["metrics"]

    print()
    print(
        "scorable historical-critical:",
        m["historical_fix_adjacent_scorable"],
    )
    print(
        "critical A -> D:",
        f"{m['critical_kept_A']} -> "
        f"{m['critical_kept_D']}",
    )
    print(
        "DROP->KEEP rescue:",
        m["critical_DROP_to_KEEP"],
    )
    print(
        "KEEP->DROP regression:",
        m["critical_KEEP_to_DROP"],
    )

    print()
    print(
        "newly kept records:",
        m["newly_kept_records"],
    )
    print(
        "  critical:",
        m["newly_kept_critical_records"],
    )
    print(
        "  noncritical:",
        m["newly_kept_noncritical_records"],
    )
    print(
        "additional critical tokens:",
        m[
            "additional_critical_record_tokens"
        ],
    )
    print(
        "additional noncritical tokens:",
        m[
            "additional_noncritical_record_tokens"
        ],
    )

    print()
    print(
        "A target tokens:",
        result["A"]["target_output_tokens"],
    )
    print(
        "D target tokens:",
        result["D"]["target_output_tokens"],
    )

    print()
    print("result:", OUT)


if __name__ == "__main__":
    main()
