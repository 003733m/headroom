#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import headroom.proxy.handlers.openai as o
import headroom.trajectory_relevance as tr
from headroom.transforms.content_router import ContentRouter

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

RUN_ROOT = (
    Path.home()
    / "headroom-second-unseen-adopted-20260905"
    / "results"
)

OUT = (
    HERE
    / "second_unseen_adopted_bridge_result.json"
)


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def output_text(item: dict[str, Any]) -> str:
    value = item.get("output")
    if isinstance(value, str):
        return value
    return tr._responses_text(value)


def body(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(errors="replace"))
    b = d.get("body")

    if isinstance(b, str):
        b = json.loads(b)

    if not isinstance(b, dict):
        fail(f"missing JSON body: {path}")

    return b


def production_min_bytes() -> int:
    owners = []

    for name, obj in vars(o).items():
        if (
            inspect.isclass(obj)
            and hasattr(
                obj,
                "OPENAI_RESPONSES_ROUTER_MIN_BYTES",
            )
        ):
            owners.append(
                (
                    name,
                    int(
                        getattr(
                            obj,
                            "OPENAI_RESPONSES_ROUTER_MIN_BYTES",
                        )
                    ),
                )
            )

    values = {v for _, v in owners}

    if len(values) != 1:
        fail(f"cannot resolve unique min-byte threshold: {owners}")

    return values.pop()


MIN_BYTES = production_min_bytes()


def analyze_condition(condition: str) -> dict[str, Any]:
    root = RUN_ROOT / condition
    metrics_path = root / "metrics.json"

    if not metrics_path.exists():
        fail(f"missing metrics: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())

    if metrics.get("agent_exit") != 0:
        fail(f"{condition}: invalid agent exit")

    if metrics.get("oracle_exit") != 0:
        fail(f"{condition}: oracle failed")

    router = ContentRouter()

    wire_files = sorted(
        (root / "codex-wire").glob(
            "*_http_inbound_request.json"
        )
    )

    if not wire_files:
        fail(f"{condition}: no inbound wire")

    seen: set[tuple[str, str]] = set()
    events = []

    for request_index, wire in enumerate(wire_files, 1):
        b = body(wire)
        items = b.get("input")

        if not isinstance(items, list):
            continue

        for before_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            if item.get("type") not in {
                "function_call_output",
                "custom_tool_call_output",
                "local_shell_call_output",
            }:
                continue

            target = output_text(item)
            call_id = str(item.get("call_id") or "")

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

            pattern = tr._responses_search_pattern(command)

            if not pattern:
                continue

            target_bytes = len(
                target.encode(
                    "utf-8",
                    errors="replace",
                )
            )

            structural = (
                o._responses_trajectory_search_eligible(
                    router,
                    target,
                )
            )

            # _responses_producing_search_command() only returns
            # a linked rg/grep-producing command. Therefore this
            # reproduces the handler's bash_search_call_ids arm
            # for this particular output.
            bash_search_linked = True

            handler_search_gate = (
                structural
                or bash_search_linked
            )

            context = (
                tr.build_responses_search_relevance_context(
                    items,
                    before_index=before_index,
                    target_content=target,
                )
            )

            bridges = list(
                tr._trajectory_context_bridge_identifiers(
                    context
                )
            )

            adopted = list(
                tr._responses_adopted_bridge_identifiers(
                    items,
                    before_index=before_index,
                    trajectory_context=context,
                )
            )

            byte_eligible = target_bytes >= MIN_BYTES

            complete_applicability = (
                byte_eligible
                and handler_search_gate
                and bool(context)
                and bool(adopted)
            )

            events.append({
                "request_index": request_index,
                "command": command,
                "pattern": pattern,
                "target_bytes": target_bytes,
                "minimum_bytes": MIN_BYTES,
                "byte_eligible": byte_eligible,
                "structural_search_eligible": structural,
                "bash_search_linked": bash_search_linked,
                "handler_search_gate": handler_search_gate,
                "context_nonempty": bool(context),
                "trajectory_bridges": bridges,
                "adopted_bridges": adopted,
                "complete_applicability": (
                    complete_applicability
                ),
            })

    applicable = [
        x for x in events
        if x["complete_applicability"]
    ]

    return {
        "condition": condition,
        "run_valid": True,
        "task_pass": bool(metrics.get("task_pass")),
        "agent_exit": metrics.get("agent_exit"),
        "oracle_exit": metrics.get("oracle_exit"),
        "requests": metrics.get("requests"),
        "rg_capture_count": metrics.get(
            "rg_capture_count"
        ),
        "relevance_split_units": metrics.get(
            "relevance_split_units"
        ),
        "search_relevance_chains": metrics.get(
            "search_relevance_chains"
        ),
        "treatment_activated": metrics.get(
            "treatment_activated"
        ),
        "natural_search_outputs": len(events),
        "complete_applicability_events": len(
            applicable
        ),
        "first_complete_applicability_event": (
            applicable[0]
            if applicable
            else None
        ),
        "events": events,
    }


off = analyze_condition("U01-OFF")
on = analyze_condition("U01-ON")

result = {
    "experiment": (
        "second_unseen_adopted_bridge_validation"
    ),
    "classification": (
        "prospective applicability-conditioned "
        "unseen validation"
    ),
    "production_freeze": FREEZE,
    "task": "U01",
    "conditions": {
        "OFF": off,
        "ON": on,
    },
    "primary_outcome": {
        "off_complete_applicability": (
            off["complete_applicability_events"] > 0
        ),
        "on_complete_applicability": (
            on["complete_applicability_events"] > 0
        ),
        "on_treatment_activated": bool(
            on["treatment_activated"]
        ),
        "causal_retention_estimate_available": bool(
            on["treatment_activated"]
        ),
    },
    "interpretation": (
        "The frozen adopted-bridge applicability mechanism "
        "reproduced on the unseen natural trajectory, including "
        "handler-level search eligibility, non-empty trajectory "
        "context, and explicit adopted identifiers. However, the "
        "fresh ON run recorded no final relevance_split treatment "
        "activation, so the preservation floor did not provide an "
        "unseen causal retention estimate."
    ),
    "raw_wire_committed": False,
}

OUT.write_text(
    json.dumps(result, indent=2) + "\n"
)

print("=" * 80)
print("SECOND UNSEEN FINAL AUDIT")
print("=" * 80)

for name, x in (("OFF", off), ("ON", on)):
    print()
    print(name)
    print("  pass:", x["task_pass"])
    print(
        "  complete applicability events:",
        x["complete_applicability_events"],
    )
    print(
        "  relevance_split_units:",
        x["relevance_split_units"],
    )
    print(
        "  treatment_activated:",
        x["treatment_activated"],
    )

    ev = x["first_complete_applicability_event"]

    if ev:
        print("  command:", ev["command"])
        print(
            "  adopted:",
            ev["adopted_bridges"],
        )
        print(
            "  handler gate:",
            ev["handler_search_gate"],
        )
        print(
            "  structural:",
            ev["structural_search_eligible"],
        )
        print(
            "  bash linked:",
            ev["bash_search_linked"],
        )

print()
print("-" * 80)
print(
    "CAUSAL RETENTION ESTIMATE AVAILABLE:",
    result["primary_outcome"][
        "causal_retention_estimate_available"
    ],
)
print("result:", OUT)
