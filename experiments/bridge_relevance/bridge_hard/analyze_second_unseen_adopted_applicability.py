#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

RUNTIME = Path(
    "/tmp/headroom-second-unseen-adopted-runtime-89da0898"
).resolve()

RESULT_ROOT = (
    Path.home()
    / "headroom-second-unseen-adopted-20260905"
    / "results"
)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


# IMPORTANT:
# This analyzer intentionally never reads:
# - historical source commit / fix patch,
# - historical-critical oracle,
# - baseline KEEP/DROP,
# - adopted-floor retention result,
# - task solution outcome except run validity.
#
# It implements only the applicability rule locked before the run.

import headroom.trajectory_relevance as tr  # noqa: E402


module_path = Path(tr.__file__).resolve()

if RUNTIME not in module_path.parents:
    die(
        "trajectory_relevance imported from wrong tree:\n"
        f"{module_path}\nexpected under {RUNTIME}"
    )


def body_from_wire(path: Path) -> dict:
    d = json.loads(path.read_text(errors="replace"))
    body = d.get("body")

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            die(f"non-JSON body in {path.name}")

    if not isinstance(body, dict):
        die(f"missing body dict in {path.name}")

    return body


def output_text(item: dict) -> str:
    value = item.get("output")

    if isinstance(value, str):
        return value

    # Frozen production helper handles Responses content lists.
    return tr._responses_text(value)


def analyze(condition: str) -> dict:
    out = RESULT_ROOT / condition

    metrics_path = out / "metrics.json"
    if not metrics_path.exists():
        die(f"metrics missing: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())

    if metrics.get("agent_exit") != 0:
        die("invalid run: nonzero agent exit")

    if metrics.get("oracle_exit") != 0:
        die("invalid run: oracle did not pass")

    wire_root = out / "codex-wire"

    wire_files = sorted(
        wire_root.glob("*_http_inbound_request.json")
    )

    if not wire_files:
        die("no inbound wire files")

    seen_outputs: set[tuple[str, str]] = set()
    searches: list[dict] = []

    for request_index, wire_path in enumerate(wire_files, 1):
        body = body_from_wire(wire_path)
        items = body.get("input")

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

            call_id = str(item.get("call_id") or "")
            target = output_text(item)

            # Cumulative Responses requests repeat old outputs.
            # Count each concrete output only on first exposure.
            digest = hashlib.sha256(
                target.encode(errors="replace")
            ).hexdigest()

            identity = (call_id, digest)

            if identity in seen_outputs:
                continue

            seen_outputs.add(identity)

            command = tr._responses_producing_search_command(
                items,
                before_index=before_index,
            )

            # The frozen production parser itself decides whether this
            # target has a linked rg/grep producing search command.
            if not command:
                continue

            pattern = tr._responses_search_pattern(command)

            if not pattern:
                continue

            query_ids = tr._responses_query_identifiers(pattern)

            trajectory_context = (
                tr.build_responses_search_relevance_context(
                    items,
                    before_index=before_index,
                    target_content=target,
                )
            )

            bridges = (
                tr._trajectory_context_bridge_identifiers(
                    trajectory_context
                )
            )

            adopted = (
                tr._responses_adopted_bridge_identifiers(
                    items,
                    before_index=before_index,
                    trajectory_context=trajectory_context,
                )
            )

            searches.append({
                "request_index": request_index,
                "target_item_index": before_index,
                "call_id": call_id,
                "command": command,
                "pattern": pattern,
                "query_identifiers": list(query_ids),
                "trajectory_bridges": list(bridges),
                "adopted_bridges": list(adopted),
                "target_bytes": len(
                    target.encode(errors="replace")
                ),
                # Structural reachability:
                # target output is linked by call_id to a producing
                # rg/grep command recognized by frozen production parser.
                "production_search_reachable": True,
            })

    eligible_events = [
        row
        for row in searches
        if (
            row["production_search_reachable"]
            and row["trajectory_bridges"]
            and row["adopted_bridges"]
        )
    ]

    result = {
        "experiment": (
            "second_unseen_adopted_bridge_validation"
        ),
        "condition": condition,
        "production_freeze": FREEZE,
        "analysis_type": (
            "outcome_blind_natural_applicability"
        ),
        "run_valid": True,
        "wire_requests": len(wire_files),
        "unique_natural_search_outputs": len(searches),
        "eligible": bool(eligible_events),
        "first_eligible_event": (
            eligible_events[0]
            if eligible_events
            else None
        ),
        "searches": searches,
        "forbidden_outcomes_inspected": False,
    }

    local_result = out / "applicability.json"
    local_result.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    return result


def main() -> None:
    if len(sys.argv) != 2:
        die(
            "usage: analyze_second_unseen_adopted_applicability.py "
            "U01-OFF"
        )

    result = analyze(sys.argv[1])

    print("=" * 78)
    print("SECOND UNSEEN — OUTCOME-BLIND APPLICABILITY")
    print("=" * 78)
    print("freeze:", result["production_freeze"])
    print("condition:", result["condition"])
    print("wire requests:", result["wire_requests"])
    print(
        "unique natural search outputs:",
        result["unique_natural_search_outputs"],
    )
    print()

    for i, row in enumerate(result["searches"], 1):
        print(f"SEARCH {i}")
        print("  command :", row["command"])
        print(
            "  query ids:",
            row["query_identifiers"],
        )
        print(
            "  bridges  :",
            row["trajectory_bridges"],
        )
        print(
            "  adopted  :",
            row["adopted_bridges"],
        )
        print(
            "  bytes    :",
            row["target_bytes"],
        )
        print()

    print("-" * 78)
    print("ELIGIBLE:", result["eligible"])

    if result["eligible"]:
        x = result["first_eligible_event"]
        print(
            "FIRST ADOPTED BRIDGE:",
            x["adopted_bridges"],
        )
        print("FIRST COMMAND:", x["command"])
    else:
        print("No frozen adopted-bridge event.")

    print()
    print(
        "No historical-critical or retention outcome "
        "was inspected."
    )


if __name__ == "__main__":
    main()
