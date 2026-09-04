#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import re
import shlex
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "run_hard_matched_context_replay.py"
ROOT = Path.home() / "headroom-hard-confirmatory-20260904"
OUT = ROOT / "retrieval_anchored_v4_probe.json"

TASKS = ["G10", "G09", "G07", "G19", "G04", "G21"]

spec = importlib.util.spec_from_file_location("matched_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise SystemExit("Could not load matched replay module")

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

tr = base.tr

OPTS_WITH_VALUE = {
    "-g", "--glob",
    "-C", "--context",
    "-A", "--after-context",
    "-B", "--before-context",
    "-m", "--max-count",
    "-e", "--regexp",
    "-f", "--file",
    "-t", "--type",
    "-T", "--type-not",
    "--encoding",
    "--engine",
    "--ignore-file",
    "--sort",
    "--sortr",
}


def extract_search_pattern(command: str) -> str:
    try:
        args = shlex.split(command)
    except Exception:
        return ""

    start = None
    for i, arg in enumerate(args):
        if arg in {"rg", "grep"}:
            start = i + 1
            break

    if start is None:
        return ""

    explicit = []
    i = start

    while i < len(args):
        arg = args[i]

        if arg in {"-e", "--regexp"}:
            if i + 1 < len(args):
                explicit.append(args[i + 1])
                i += 2
                continue

        if arg.startswith("--regexp="):
            explicit.append(arg.split("=", 1)[1])
            i += 1
            continue

        i += 1

    if explicit:
        return "|".join(explicit)

    i = start

    while i < len(args):
        arg = args[i]

        if arg == "--":
            return args[i + 1] if i + 1 < len(args) else ""

        if arg in OPTS_WITH_VALUE:
            i += 2
            continue

        if arg.startswith("--") and "=" in arg:
            i += 1
            continue

        if arg.startswith("-"):
            i += 1
            continue

        return arg

    return ""


def context_tokens(context: str) -> list[str]:
    marker = "Trajectory bridge identifiers:"
    if marker not in context:
        return []

    tail = context.split(marker, 1)[1].strip()
    return tail.split()


def structured_pattern_hits(pattern: str):
    seen = set()
    result = []

    for hit in tr._find_candidates(pattern):
        key = (hit.token.casefold(), hit.kind)
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)

    return result


def latest_anchor_outputs(
    messages: list[dict[str, Any]],
    pattern: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    latest_user = tr._latest_bridge_user_context(messages)

    chosen_indexes = set()
    anchors = []

    for hit in structured_pattern_hits(pattern):
        # Must be newly trajectory-derived, not simply copied from user prompt.
        if latest_user and tr._contains_candidate(
            latest_user,
            hit.token,
            kind=hit.kind,
        ):
            continue

        found_index = None

        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]

            if message.get("role") != "tool":
                continue

            text = message.get("content")
            if not isinstance(text, str):
                continue

            if tr._contains_candidate(
                text,
                hit.token,
                kind=hit.kind,
            ):
                found_index = i
                break

        if found_index is None:
            continue

        chosen_indexes.add(found_index)

        anchors.append(
            {
                "token": hit.token,
                "kind": hit.kind,
                "message_index": found_index,
            }
        )

    selected = [
        messages[i]
        for i in sorted(chosen_indexes)
    ]

    return selected, anchors


def anchored_context(
    items: list[Any],
    *,
    before_index: int,
    target_content: str,
) -> tuple[str, list[dict[str, str]], str]:
    command_info = base.producing_call(items, before_index)

    if command_info is None:
        return "", [], ""

    _, command = command_info
    pattern = extract_search_pattern(command)

    if not pattern:
        return "", [], ""

    messages = tr.responses_items_to_bridge_messages(
        items,
        before_index=before_index,
    )

    if not messages:
        return "", [], pattern

    selected, anchors = latest_anchor_outputs(
        messages,
        pattern,
    )

    if not selected:
        return "", [], pattern

    novelty = tr._latest_bridge_user_context(messages)

    context = tr.build_search_relevance_context(
        selected,
        before_index=len(selected),
        user_context="",
        target_content=target_content,
        novelty_context=novelty,
    )

    return context, anchors, pattern


results = []
aggregate = {
    "search_units": 0,
    "units_with_anchor": 0,
    "v3_nonempty": 0,
    "v4_nonempty": 0,
    "v3_bridge_identifiers": 0,
    "v4_bridge_identifiers": 0,
    "v3_removed_by_v4": 0,
    "v4_new_after_local_rerank": 0,
}

for task in TASKS:
    rows = base.load_wire(task)
    units = base.first_search_units(rows)

    row_by_req = {
        row["request_id"]: row
        for row in rows
    }

    print()
    print("=" * 72)
    print(task)
    print("=" * 72)

    for unit in units:
        items = row_by_req[unit["request_id"]]["pre"].get("input", [])

        if not isinstance(items, list):
            continue

        global_ctx = tr.build_responses_search_relevance_context(
            items,
            before_index=unit["before_index"],
            target_content=unit["target"],
        )

        local_ctx, anchors, pattern = anchored_context(
            items,
            before_index=unit["before_index"],
            target_content=unit["target"],
        )

        v3 = context_tokens(global_ctx)
        v4 = context_tokens(local_ctx)

        v3_cf = {x.casefold(): x for x in v3}
        v4_cf = {x.casefold(): x for x in v4}

        removed = [
            v3_cf[k]
            for k in v3_cf.keys() - v4_cf.keys()
        ]

        added = [
            v4_cf[k]
            for k in v4_cf.keys() - v3_cf.keys()
        ]

        pattern_cf = pattern.casefold()

        incremental_v3 = [
            x for x in v3
            if x.casefold() not in pattern_cf
        ]

        incremental_v4 = [
            x for x in v4
            if x.casefold() not in pattern_cf
        ]

        aggregate["search_units"] += 1
        aggregate["units_with_anchor"] += int(bool(anchors))
        aggregate["v3_nonempty"] += int(bool(v3))
        aggregate["v4_nonempty"] += int(bool(v4))
        aggregate["v3_bridge_identifiers"] += len(v3)
        aggregate["v4_bridge_identifiers"] += len(v4)
        aggregate["v3_removed_by_v4"] += len(removed)
        aggregate["v4_new_after_local_rerank"] += len(added)

        row = {
            "task": task,
            "request_id": unit["request_id"],
            "call_id": unit["call_id"],
            "command": unit["command"],
            "pattern": pattern,
            "anchors": anchors,
            "v3": v3,
            "v4": v4,
            "removed_from_v3": removed,
            "new_after_local_rerank": added,
            "incremental_v3": incremental_v3,
            "incremental_v4": incremental_v4,
        }

        results.append(row)

        print(
            f"pattern={pattern!r} "
            f"anchors={[x['token'] for x in anchors]}"
        )
        print(f"  V3: {v3}")
        print(f"  V4: {v4}")
        print(f"  removed: {removed}")
        print(f"  new: {added}")


payload = {
    "evaluation_type": "post_hoc_exploratory_mechanism",
    "base_algorithm_freeze": base.FREEZE,
    "algorithm": "retrieval_anchored_local_provenance",
    "aggregate": aggregate,
    "units": results,
}

OUT.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
)

print()
print("=" * 72)
print("AGGREGATE")
print("=" * 72)

for key, value in aggregate.items():
    print(f"{key}: {value}")

if aggregate["v3_bridge_identifiers"]:
    reduction = (
        aggregate["v3_bridge_identifiers"]
        - aggregate["v4_bridge_identifiers"]
    ) / aggregate["v3_bridge_identifiers"]

    print(f"bridge_count_reduction: {reduction:.4f}")

print("JSON:", OUT)
