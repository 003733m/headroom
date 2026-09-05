#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from headroom.trajectory_relevance import (
    _responses_query_identifiers,
    _responses_search_pattern,
    _trajectory_context_bridge_identifiers,
    build_search_relevance_context,
)

HERE = Path(__file__).resolve().parent
SCREEN_MANIFEST = HERE / "historical_difficulty_screen_manifest.json"

RAW_ROOT = (
    Path.home()
    / "headroom-hard-screen-raw-20260904"
    / "results"
)

OUT = HERE / "adopted_bridge_holdout_selection.json"

ALGORITHM_FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

EXCLUDED = {
    "G04",
    "G07",
    "G09",
    "G10",
    "G19",
    "G21",
}

MAX_SELECTED = 6


def parse_meta(path: Path) -> dict[str, Any]:
    fields: dict[str, str] = {}

    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value

    stem = path.stem

    stdout = path.with_suffix(".stdout")
    stderr = path.with_suffix(".stderr")

    return {
        "stem": stem,
        "meta": str(path),
        "argv": fields.get("argv", ""),
        "started_ns": int(fields.get("started_ns", "0") or 0),
        "rc": int(fields.get("rc", "0") or 0),
        "stdout": (
            stdout.read_text(errors="replace")
            if stdout.exists()
            else ""
        ),
        "stderr": (
            stderr.read_text(errors="replace")
            if stderr.exists()
            else ""
        ),
    }


def decode_argv(raw: str) -> str:
    """
    The capture wrapper stores shell-escaped argv, e.g.
    CodeHandler\\|code_handler\\|compression .
    Reconstruct an rg command sufficiently for the production parser.
    """
    try:
        parts = shlex.split(raw)
    except Exception:
        # Conservative fallback: still let the production parser attempt it.
        return "rg " + raw

    return "rg " + " ".join(shlex.quote(part) for part in parts)


def combined_output(capture: dict[str, Any]) -> str:
    out = capture["stdout"]

    if capture["stderr"]:
        if out and not out.endswith("\n"):
            out += "\n"
        out += capture["stderr"]

    return out


def visible_tool_message(text: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": text,
    }


def analyze_task(task_id: str) -> dict[str, Any]:
    run_root = RAW_ROOT / f"{task_id}-OFF"
    cap_root = run_root / "rg-captures"

    if not cap_root.exists():
        return {
            "task_id": task_id,
            "status": "missing_rg_captures",
            "capture_count": 0,
            "opportunities": [],
        }

    captures = [
        parse_meta(p)
        for p in cap_root.glob("*.meta")
    ]
    captures.sort(
        key=lambda x: (
            x["started_ns"],
            x["stem"],
        )
    )

    prior_messages: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []

    for index, cap in enumerate(captures):
        command = decode_argv(cap["argv"])
        pattern = _responses_search_pattern(command)
        query_ids = list(_responses_query_identifiers(pattern))

        target = combined_output(cap)

        trajectory_context = ""

        if prior_messages and target:
            trajectory_context = build_search_relevance_context(
                prior_messages,
                before_index=len(prior_messages),
                user_context="",
                target_content=target,
                novelty_context="",
            )

        bridges = list(
            _trajectory_context_bridge_identifiers(
                trajectory_context
            )
        )

        bridge_by_cf = {
            bridge.casefold(): bridge
            for bridge in bridges
        }

        adopted = []

        for query_id in query_ids:
            bridge = bridge_by_cf.get(query_id.casefold())
            if bridge is not None:
                adopted.append(bridge)

        adopted = list(dict.fromkeys(adopted))

        if adopted:
            opportunities.append(
                {
                    "capture_index": index,
                    "started_ns": cap["started_ns"],
                    "argv": cap["argv"],
                    "command": command,
                    "pattern": pattern,
                    "query_identifiers": query_ids,
                    "v3_bridges": bridges,
                    "adopted_bridge_identifiers": adopted,
                    "target_bytes": len(
                        target.encode(
                            "utf-8",
                            errors="replace",
                        )
                    ),
                }
            )

        # Causal ordering: current output becomes trajectory state only
        # after its own opportunity has been evaluated.
        if target:
            prior_messages.append(
                visible_tool_message(target)
            )

    return {
        "task_id": task_id,
        "status": "ok",
        "capture_count": len(captures),
        "opportunity_count": len(opportunities),
        "eligible": bool(opportunities),
        "opportunities": opportunities,
    }


manifest = json.loads(SCREEN_MANIFEST.read_text())

tasks = sorted(
    manifest["tasks"],
    key=lambda task: task["task_id"],
)

task_by_id = {
    task["task_id"]: task
    for task in tasks
}

analysis = []

for task in tasks:
    task_id = task["task_id"]

    if task_id in EXCLUDED:
        continue

    row = analyze_task(task_id)
    row["source_commit"] = task["source_commit"]
    row["subject"] = task["subject"]
    row["production_files"] = task["production_files"]
    row["test_files"] = task["test_files"]
    row["failing_nodeids"] = task["failing_nodeids"]

    analysis.append(row)

eligible_ids = sorted(
    row["task_id"]
    for row in analysis
    if row.get("eligible")
)

selected_ids = eligible_ids[:MAX_SELECTED]

selected = []

for task_id in selected_ids:
    task = task_by_id[task_id]
    row = next(
        x for x in analysis
        if x["task_id"] == task_id
    )

    selected.append(
        {
            "task_id": task_id,
            "source_commit": task["source_commit"],
            "subject": task["subject"],
            "production_files": task["production_files"],
            "test_files": task["test_files"],
            "failing_nodeids": task["failing_nodeids"],
            "opportunity_count": row["opportunity_count"],
            "opportunities": row["opportunities"],
        }
    )

payload = {
    "status": "selected_from_pre_treatment_OFF_only_traces",
    "experiment": "adopted_bridge_unseen_holdout",
    "algorithm_freeze": ALGORITHM_FREEZE,
    "selection_data": (
        "historical difficulty-screen OFF-only rg captures"
    ),
    "selection_scope": (
        "rg-observable adopted-bridge opportunities only; "
        "this intentionally under-approximates all production tool-output "
        "opportunities"
    ),
    "excluded_development_tasks": sorted(EXCLUDED),
    "candidate_count": len(analysis),
    "eligibility_rule": (
        "frozen V3 derives an identifier from prior chronological rg output "
        "and a later agent rg query reuses the same conservative structured "
        "identifier"
    ),
    "selection_rule": (
        "task_id ascending; first 6 eligible tasks, "
        "or all if fewer than 6"
    ),
    "eligible_task_ids": eligible_ids,
    "selected_task_ids": selected_ids,
    "all_candidate_analysis": analysis,
    "selected_tasks": selected,
    "forbidden_selection_information": [
        "historical-fix-adjacent retention",
        "DROP-to-KEEP outcome",
        "ON result",
        "treated task success",
        "treated token savings",
    ],
}

OUT.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)

print("=" * 88)
print("ADOPTED-BRIDGE HOLDOUT SELECTION")
print("=" * 88)

print("candidate tasks:", len(analysis))
print("eligible tasks:", len(eligible_ids))
print("eligible IDs:", eligible_ids)
print("selected IDs:", selected_ids)

print()

for row in analysis:
    flag = "ELIGIBLE" if row.get("eligible") else "-"
    print(
        f"{row['task_id']:>3} "
        f"{flag:8} "
        f"captures={row.get('capture_count', 0):2} "
        f"opportunities={row.get('opportunity_count', 0):2}"
    )

    for opp in row.get("opportunities", []):
        print(
            "    adopted=",
            opp["adopted_bridge_identifiers"],
        )
        print(
            "    query=",
            opp["pattern"][:160],
        )

print()
print("JSON:", OUT)
