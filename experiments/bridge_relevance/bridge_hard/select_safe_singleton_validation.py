#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shlex
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import headroom.trajectory_relevance as tr
from headroom.trajectory_relevance import (
    DEFAULT_MAX_TOOL_OUTPUTS,
    TARGET_CORROBORATION_KINDS,
    _context_evidence,
    _find_candidates,
    _responses_query_identifiers,
    _responses_search_pattern,
    _trajectory_context_bridge_identifiers,
    build_search_relevance_context,
)

HERE = Path(__file__).resolve().parent

RAW_ROOT = (
    Path.home()
    / "headroom-hard-screen-raw-20260904"
    / "results"
)

MANIFEST = HERE / "historical_difficulty_screen_manifest.json"
OUT = HERE / "safe_singleton_validation_selection.json"

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

EXCLUDED = {
    "G04",
    "G07",
    "G09",
    "G10",
    "G11",
    "G19",
    "G21",
}

MAX_SELECTED = 3
SAFE_KINDS = frozenset(TARGET_CORROBORATION_KINDS)


# Exact selector-only memoization. This changes cost only, not semantics.
_ORIGINAL_FIND = tr._find_candidates
_ORIGINAL_CONTEXT = tr._context_evidence


@lru_cache(maxsize=32)
def _cached_hits_tuple(text: str) -> tuple[Any, ...]:
    return tuple(_ORIGINAL_FIND(text))


def cached_hits(text: str) -> list[Any]:
    return list(_cached_hits_tuple(text))


@lru_cache(maxsize=262_144)
def cached_context(line: str, token: str) -> tuple[int, int]:
    return _ORIGINAL_CONTEXT(line, token)


tr._find_candidates = cached_hits
tr._context_evidence = cached_context


def parse_meta(path: Path) -> dict[str, Any]:
    fields: dict[str, str] = {}

    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value

    stdout = path.with_suffix(".stdout")
    stderr = path.with_suffix(".stderr")

    return {
        "stem": path.stem,
        "started_ns": int(fields.get("started_ns", "0") or 0),
        "argv": fields.get("argv", ""),
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
    try:
        parts = shlex.split(raw)
    except Exception:
        return "rg " + raw

    return "rg " + " ".join(
        shlex.quote(part)
        for part in parts
    )


def combined_output(cap: dict[str, Any]) -> str:
    text = cap["stdout"]

    if cap["stderr"]:
        if text and not text.endswith("\n"):
            text += "\n"
        text += cap["stderr"]

    return text


def tool_message(text: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": text,
    }


def exact_identifier_pattern(token: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


def prior_singleton_safe_evidence(
    history: list[dict[str, Any]],
    query_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Find safe-kind query IDs relying specifically on singleton admission.

    Causal rule: only history strictly before the current search is inspected.
    """

    evidence: dict[str, dict[str, Any]] = {}

    for query_id in query_ids:
        pattern = exact_identifier_pattern(query_id)

        containing_outputs = 0
        occurrences = 0
        kinds: set[str] = set()
        best_positive = 0
        strongest_negative = 0

        for message in history:
            text = message.get("content", "")

            if not isinstance(text, str) or not text:
                continue

            if not pattern.search(text):
                continue

            output_contains = False
            lines = text.splitlines()

            for hit in _find_candidates(text):
                if hit.token.casefold() != query_id.casefold():
                    continue

                output_contains = True
                occurrences += 1
                kinds.add(hit.kind)

                line = lines[hit.line_no]

                positive, negative = _context_evidence(
                    line,
                    hit.token,
                )

                best_positive = max(
                    best_positive,
                    positive,
                )
                strongest_negative = max(
                    strongest_negative,
                    negative,
                )

            if output_contains:
                containing_outputs += 1

        safe_kinds = sorted(
            kind
            for kind in kinds
            if kind in SAFE_KINDS
        )

        # Specifically isolate the path where the singleton exception matters:
        # one prior output, no positive cue, safe semantic kind.
        if (
            containing_outputs == 1
            and best_positive <= 0
            and safe_kinds
        ):
            evidence[query_id] = {
                "prior_outputs_containing": containing_outputs,
                "occurrences": occurrences,
                "kinds": sorted(kinds),
                "safe_kinds": safe_kinds,
                "best_positive": best_positive,
                "strongest_negative": strongest_negative,
            }

    return evidence


def analyze_task(task_id: str) -> dict[str, Any]:
    root = RAW_ROOT / f"{task_id}-OFF" / "rg-captures"

    if not root.exists():
        return {
            "task_id": task_id,
            "status": "missing_rg_captures",
            "capture_count": 0,
            "opportunities": [],
            "eligible": False,
        }

    captures = [
        parse_meta(p)
        for p in root.glob("*.meta")
    ]

    captures.sort(
        key=lambda row: (
            row["started_ns"],
            row["stem"],
        )
    )

    # Production trajectory horizon is 8 outputs. Selector intentionally
    # observes rg outputs only, so this remains a conservative under-approx.
    prior = deque(
        maxlen=DEFAULT_MAX_TOOL_OUTPUTS
    )

    opportunities: list[dict[str, Any]] = []

    for index, cap in enumerate(captures):
        command = decode_argv(cap["argv"])
        search_pattern = _responses_search_pattern(command)
        query_ids = list(
            _responses_query_identifiers(search_pattern)
        )

        history = list(prior)

        singleton = prior_singleton_safe_evidence(
            history,
            query_ids,
        )

        target = combined_output(cap)

        if singleton and target:
            context = build_search_relevance_context(
                history,
                before_index=len(history),
                user_context="",
                target_content=target,
                novelty_context="",
            )

            bridges = list(
                _trajectory_context_bridge_identifiers(
                    context
                )
            )

            bridge_cf = {
                token.casefold(): token
                for token in bridges
            }

            admitted = []

            for query_id, evidence in singleton.items():
                bridge = bridge_cf.get(
                    query_id.casefold()
                )

                if bridge is None:
                    continue

                admitted.append(
                    {
                        "identifier": bridge,
                        "singleton_evidence": evidence,
                    }
                )

            if admitted:
                opportunities.append(
                    {
                        "capture_index": index,
                        "started_ns": cap["started_ns"],
                        "argv": cap["argv"],
                        "pattern": search_pattern,
                        "query_identifiers": query_ids,
                        "admitted_safe_singletons": admitted,
                        "trajectory_bridges": bridges,
                        "target_bytes": len(
                            target.encode(
                                "utf-8",
                                errors="replace",
                            )
                        ),
                    }
                )

        # Strict causality: current output is only available to later searches.
        if target:
            prior.append(
                tool_message(target)
            )

    return {
        "task_id": task_id,
        "status": "ok",
        "capture_count": len(captures),
        "opportunity_count": len(opportunities),
        "eligible": bool(opportunities),
        "opportunities": opportunities,
    }


manifest = json.loads(
    MANIFEST.read_text()
)

tasks = sorted(
    manifest["tasks"],
    key=lambda task: task["task_id"],
)

analysis: list[dict[str, Any]] = []

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

    print(
        f"scanned {task_id}: "
        f"captures={row.get('capture_count', 0)} "
        f"opportunities={row.get('opportunity_count', 0)} "
        f"eligible={row.get('eligible', False)}",
        flush=True,
    )

eligible_ids = sorted(
    row["task_id"]
    for row in analysis
    if row.get("eligible")
)

selected_ids = eligible_ids[:MAX_SELECTED]

selected_tasks = [
    row
    for row in analysis
    if row["task_id"] in selected_ids
]

payload = {
    "status": "selected_from_treatment_blind_OFF_only_traces",
    "experiment": "safe_singleton_adopted_bridge_validation",
    "algorithm_freeze": FREEZE,
    "safe_singleton_kinds": sorted(SAFE_KINDS),
    "excluded_tasks": sorted(EXCLUDED),
    "candidate_count": len(analysis),
    "eligible_task_ids": eligible_ids,
    "selected_task_ids": selected_ids,
    "selection_rule": (
        "task_id ascending; first 3 eligible tasks, "
        "or all if fewer than 3"
    ),
    "selection_information_forbidden": [
        "ON trajectory",
        "adopted-floor retention outcome",
        "critical DROP-to-KEEP",
        "critical KEEP-to-DROP",
        "treated success",
        "treated token savings",
    ],
    "all_candidate_analysis": analysis,
    "selected_tasks": selected_tasks,
}

OUT.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
)

print()
print("=" * 88)
print("SAFE SINGLETON VALIDATION SELECTION")
print("=" * 88)
print("candidate tasks:", len(analysis))
print("eligible tasks:", len(eligible_ids))
print("eligible IDs:", eligible_ids)
print("selected IDs:", selected_ids)

for row in selected_tasks:
    print()
    print(row["task_id"])

    for opp in row["opportunities"]:
        print("  query:", opp["pattern"][:220])

        for item in opp["admitted_safe_singletons"]:
            print(
                "  singleton:",
                item["identifier"],
                item["singleton_evidence"],
            )

print()
print("JSON:", OUT)
