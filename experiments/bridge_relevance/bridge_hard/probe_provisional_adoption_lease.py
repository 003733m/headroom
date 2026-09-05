#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONTROL = Path(__file__).resolve().parents[3]

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

HARD_ROOT = Path.home() / "headroom-hard-confirmatory-20260904"
MEASURED = HARD_ROOT / "normalized_retention_measurement.json"
BRIDGES = HARD_ROOT / "retrieval_anchored_v41_probe.json"

G11_ROOT = (
    Path.home()
    / "headroom-adopted-bridge-holdout-20260905"
    / "results"
    / "G11-ON"
    / "rg-captures"
)

OUT = HARD_ROOT / "provisional_adoption_lease_probe.json"

TASKS = ["G10", "G09", "G07", "G19", "G04", "G21"]

G11_SOURCE_COMMIT = (
    "6137967083936467c570e8c7f20e94f43ccc13aa"
)
G11_PRODUCTION_FILE = (
    "headroom/pricing/litellm_model_resolution.py"
)

MAX_PRIOR_OUTPUTS = 8

RECORD_RE = re.compile(
    r"^(?P<path>.+?)(?P<sep>[:-])"
    r"(?P<line>\d+)(?P=sep)(?P<text>.*)$"
)

HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)"
    r"(?:,(?P<old_count>\d+))?"
    r" \+(?P<new_start>\d+)"
    r"(?:,(?P<new_count>\d+))? @@"
)


# ---------------------------------------------------------------------
# Reuse the already-existing conservative query identifier helpers.
# ---------------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "query_floor",
    HERE / "probe_query_hit_preservation.py",
)

if spec is None or spec.loader is None:
    raise SystemExit(
        "Could not load query-hit preservation helpers"
    )

qh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qh)


# Production imports. We verify below that headroom/ is unchanged
# relative to the production freeze.
import headroom.trajectory_relevance as tr  # noqa: E402

from headroom.transforms.content_router import ContentRouter  # noqa: E402
from headroom.transforms.relevance_split import (  # noqa: E402
    plan_relevance_split,
)


# Fresh production-equivalent router configuration for the direct
# mechanism replay. _get_relevance_scorer() serves BM25 immediately;
# keep the returned scorer object fixed for the entire deterministic probe.
_DIRECT_ROUTER = ContentRouter()
_DIRECT_SCORER = _DIRECT_ROUTER._get_relevance_scorer()

if _DIRECT_SCORER is None:
    raise SystemExit("Production relevance scorer unavailable")


# Probe-only exact memoization.
#
# This changes runtime only, not extraction/ranking semantics.
# The probe repeatedly inspects the same multi-megabyte historical outputs.
_ORIGINAL_FIND_CANDIDATES = tr._find_candidates
_ORIGINAL_CONTEXT_EVIDENCE = tr._context_evidence


@lru_cache(maxsize=262_144)
def _cached_find_candidates_tuple(text: str):
    return tuple(_ORIGINAL_FIND_CANDIDATES(text))


def _cached_find_candidates(text: str):
    return list(_cached_find_candidates_tuple(text))


@lru_cache(maxsize=262_144)
def _cached_context_evidence(line: str, token: str):
    return _ORIGINAL_CONTEXT_EVIDENCE(line, token)


tr._find_candidates = _cached_find_candidates
tr._context_evidence = _cached_context_evidence


def die(msg: str) -> None:
    raise SystemExit(msg)


def sha(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def normalize_path(path: str) -> str:
    p = path.replace("\\", "/").strip()

    while p.startswith("./"):
        p = p[2:]

    return p


def verify_production_freeze() -> None:
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
        die(
            "Production headroom/ differs from frozen "
            f"algorithm {FREEZE}"
        )

    print("Production freeze: OK", FREEZE)


def git_text(*args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=CONTROL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if cp.returncode:
        die(
            "git command failed:\n"
            + "git "
            + " ".join(args)
            + "\n"
            + cp.stderr
        )

    return cp.stdout


def exact_present(text: str, identifier: str) -> bool:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_])"
        rf"{re.escape(identifier)}"
        rf"(?![A-Za-z0-9_])"
    )

    return bool(pattern.search(text))


def candidate_on_line(
    line: str,
    identifier: str,
) -> tuple[bool, list[str]]:
    kinds: set[str] = set()

    try:
        hits = tr._find_candidates(line)
    except Exception:
        hits = []

    for hit in hits:
        token = str(getattr(hit, "token", ""))

        if token.casefold() != identifier.casefold():
            continue

        kind = str(getattr(hit, "kind", ""))

        if kind:
            kinds.add(kind)

    return bool(kinds), sorted(kinds)


def prior_evidence_for_identifier(
    previous_events: list[dict[str, Any]],
    identifier: str,
) -> dict[str, Any] | None:
    """
    Strictly causal.

    Search only bounded outputs preceding the current retrieval.
    Current target output never self-sources adoption.
    """

    matched_outputs: list[int] = []
    kinds: set[str] = set()
    total_matching_lines = 0

    previous_events = previous_events[-MAX_PRIOR_OUTPUTS:]

    for event in previous_events:
        path = Path(event["stdout_path"])

        if not path.exists():
            continue

        output_matched = False

        with path.open(errors="replace") as f:
            for line in f:
                if not exact_present(line, identifier):
                    continue

                candidate, line_kinds = candidate_on_line(
                    line,
                    identifier,
                )

                if not candidate:
                    continue

                output_matched = True
                total_matching_lines += 1
                kinds.update(line_kinds)

        if output_matched:
            matched_outputs.append(
                int(event["event_index"])
            )

    if not matched_outputs:
        return None

    return {
        "prior_event_indices": matched_outputs,
        "distinct_prior_outputs": len(matched_outputs),
        "matching_lines": total_matching_lines,
        "kinds": sorted(kinds),
    }


def parse_meta(meta: Path) -> dict[str, str]:
    result: dict[str, str] = {}

    for line in meta.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        result[key] = value

    return result


def load_rg_events(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []

    rows = []

    for meta in root.glob("*.meta"):
        fields = parse_meta(meta)

        stdout = meta.with_suffix(".stdout")

        rows.append(
            {
                "started_ns": int(
                    fields.get("started_ns", "0") or 0
                ),
                "stem": meta.stem,
                "argv": fields.get("argv", ""),
                "stdout_path": str(stdout),
            }
        )

    rows.sort(
        key=lambda x: (
            x["started_ns"],
            x["stem"],
        )
    )

    for index, event in enumerate(rows):
        event["event_index"] = index

        command = "rg " + event["argv"]

        pattern = qh.extract_search_pattern(command)

        event["command"] = command
        event["pattern"] = pattern
        event["query_identifiers"] = (
            qh.structured_query_identifiers(pattern)
        )

        stdout_path = Path(event["stdout_path"])

        if stdout_path.exists():
            text = stdout_path.read_text(errors="replace")
            event["target_sha256"] = sha(text)
            event["target_bytes"] = len(
                text.encode(
                    "utf-8",
                    errors="replace",
                )
            )
        else:
            event["target_sha256"] = None
            event["target_bytes"] = 0

    # Derive causal behavioral adoption.
    for index, event in enumerate(rows):
        prior = rows[:index]

        evidence: dict[str, Any] = {}

        for identifier in event["query_identifiers"]:
            found = prior_evidence_for_identifier(
                prior,
                identifier,
            )

            if found is not None:
                evidence[identifier] = found

        event["prior_seen_query_identifiers"] = (
            sorted(evidence)
        )
        event["prior_evidence"] = evidence

    # One-search lease lifecycle.
    active_previous: set[str] = set()
    ever_adopted: set[str] = set()

    for event in rows:
        current = set(
            event["prior_seen_query_identifiers"]
        )

        renewed = active_previous & current
        expired = active_previous - current
        fresh = current - active_previous

        # Counterfactual identifiers that indefinite persistence
        # would still carry although the current query no longer
        # adopts them.
        stale_if_persistent = (
            ever_adopted
            - set(event["query_identifiers"])
        )

        event["lease_fresh"] = sorted(fresh)
        event["lease_renewed"] = sorted(renewed)
        event["lease_expired"] = sorted(expired)
        event["stale_if_persistent"] = sorted(
            stale_if_persistent
        )

        ever_adopted.update(current)
        active_previous = current

    return rows


def build_event_indices(
    task_events: dict[str, list[dict[str, Any]]],
):
    by_sha: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    by_pattern: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for task, events in task_events.items():
        for event in events:
            target_sha = event.get("target_sha256")

            if target_sha:
                by_sha[(task, target_sha)] = event

            by_pattern[
                (task, event["pattern"])
            ].append(event)

    return by_sha, by_pattern


def event_for_unit(
    task: str,
    unit: dict[str, Any],
    by_sha: dict[
        tuple[str, str],
        dict[str, Any],
    ],
    by_pattern: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ],
) -> dict[str, Any] | None:
    target_sha = unit.get("target_sha256")

    if isinstance(target_sha, str):
        hit = by_sha.get((task, target_sha))

        if hit is not None:
            return hit

    pattern = qh.extract_search_pattern(
        unit.get("command", "")
    )

    candidates = by_pattern.get(
        (task, pattern),
        [],
    )

    if len(candidates) == 1:
        return candidates[0]

    return None


def bridge_key(
    unit: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        unit["task"],
        unit["request_id"],
        unit["call_id"],
    )


def floor_ids_for_unit(
    unit: dict[str, Any],
    event: dict[str, Any] | None,
    bridge_map: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
) -> dict[str, list[str]]:
    pattern = qh.extract_search_pattern(
        unit.get("command", "")
    )

    query_ids = qh.structured_query_identifiers(
        pattern
    )

    bridge = bridge_map.get(
        bridge_key(unit),
        {},
    )

    v3 = [
        str(x)
        for x in bridge.get("v3", [])
    ]

    v3_cf = {
        x.casefold(): x
        for x in v3
    }

    adopted = [
        ident
        for ident in query_ids
        if ident.casefold() in v3_cf
    ]

    provisional = []

    if event is not None:
        event_seen = {
            x.casefold()
            for x in event[
                "prior_seen_query_identifiers"
            ]
        }

        provisional = [
            ident
            for ident in query_ids
            if ident.casefold() in event_seen
        ]

    return {
        "B": [],
        "D": adopted,
        "P": provisional,
        "Q": query_ids,
    }


def empty_condition() -> dict[str, Any]:
    return {
        "units_with_floor_ids": 0,
        "critical_kept": 0,
        "critical_DROP_to_KEEP": 0,
        "critical_KEEP_to_DROP": 0,
        "newly_kept_records": 0,
        "newly_kept_critical": 0,
        "newly_kept_nonfix": 0,
        "additional_critical_tokens": 0,
        "additional_nonfix_tokens": 0,
        "additional_total_tokens": 0,
    }


def analyze_hard(
    measured: dict[str, Any],
    bridge_map: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    task_events: dict[
        str,
        list[dict[str, Any]],
    ],
    validity: str | None,
) -> dict[str, Any]:
    if validity is None:
        units = measured["units"]
        name = "all_existing_replay_units"
    else:
        units = [
            u
            for u in measured["units"]
            if bool(u.get(validity))
        ]
        name = validity

    by_sha, by_pattern = build_event_indices(
        task_events
    )

    result: dict[str, Any] = {
        "stratum": name,
        "units": len(units),
        "units_matched_to_natural_rg_capture": 0,
        "critical_total": 0,
        "baseline_critical_kept": 0,
        "conditions": {
            key: empty_condition()
            for key in ("B", "D", "P", "Q")
        },
        "persistent_stale_counterfactual": {
            "units_with_stale_ids": 0,
            "baseline_dropped_records_matching_stale_ids": 0,
            "baseline_dropped_nonfix_records_matching_stale_ids": 0,
            "additional_record_tokens_if_persisted": 0,
        },
        "details": [],
    }

    for unit in units:
        task = unit["task"]

        event = event_for_unit(
            task,
            unit,
            by_sha,
            by_pattern,
        )

        if event is not None:
            result[
                "units_matched_to_natural_rg_capture"
            ] += 1

        ids = floor_ids_for_unit(
            unit,
            event,
            bridge_map,
        )

        for condition in ("D", "P", "Q"):
            if ids[condition]:
                result["conditions"][
                    condition
                ]["units_with_floor_ids"] += 1

        stale_ids = (
            event.get("stale_if_persistent", [])
            if event is not None
            else []
        )

        if stale_ids:
            result[
                "persistent_stale_counterfactual"
            ]["units_with_stale_ids"] += 1

        detail_new = {
            "task": task,
            "request_id": unit["request_id"],
            "call_id": unit["call_id"],
            "command": unit["command"],
            "matched_event_index": (
                event["event_index"]
                if event is not None
                else None
            ),
            "adopted_D": ids["D"],
            "behavioral_lease_P": ids["P"],
            "query_floor_Q": ids["Q"],
            "stale_if_persistent": stale_ids,
            "newly_kept_P": [],
        }

        for rec in unit["records"]:
            if not rec.get("scorable"):
                continue

            critical = bool(
                rec.get("fix_adjacent")
            )

            baseline = bool(
                rec["kept"]["B"]
            )

            tokens = int(
                rec.get("record_tokens") or 0
            )

            if critical:
                result["critical_total"] += 1

                if baseline:
                    result[
                        "baseline_critical_kept"
                    ] += 1

            for condition in ("B", "D", "P", "Q"):
                floor_ids = ids[condition]

                hits = (
                    qh.matching_identifiers(
                        rec.get("text", ""),
                        floor_ids,
                    )
                    if floor_ids
                    else []
                )

                keep = (
                    baseline
                    if condition == "B"
                    else baseline or bool(hits)
                )

                c = result["conditions"][
                    condition
                ]

                if critical and keep:
                    c["critical_kept"] += 1

                if (
                    critical
                    and not baseline
                    and keep
                ):
                    c[
                        "critical_DROP_to_KEEP"
                    ] += 1

                if (
                    critical
                    and baseline
                    and not keep
                ):
                    c[
                        "critical_KEEP_to_DROP"
                    ] += 1

                if (
                    condition != "B"
                    and not baseline
                    and hits
                ):
                    c["newly_kept_records"] += 1
                    c[
                        "additional_total_tokens"
                    ] += tokens

                    if critical:
                        c[
                            "newly_kept_critical"
                        ] += 1
                        c[
                            "additional_critical_tokens"
                        ] += tokens
                    else:
                        c[
                            "newly_kept_nonfix"
                        ] += 1
                        c[
                            "additional_nonfix_tokens"
                        ] += tokens

                    if condition == "P":
                        detail_new[
                            "newly_kept_P"
                        ].append(
                            {
                                "path": rec["path"],
                                "line": rec["line"],
                                "fix_adjacent": critical,
                                "matched_identifiers": hits,
                                "tokens": tokens,
                                "text": rec["text"],
                            }
                        )

            # Persistent-adoption counterfactual:
            #
            # What would an indefinite behavioral-adoption floor
            # retain here after explicit current-query reuse ended?
            if (
                not baseline
                and stale_ids
            ):
                stale_hits = (
                    qh.matching_identifiers(
                        rec.get("text", ""),
                        stale_ids,
                    )
                )

                if stale_hits:
                    s = result[
                        "persistent_stale_counterfactual"
                    ]

                    s[
                        "baseline_dropped_records_matching_stale_ids"
                    ] += 1
                    s[
                        "additional_record_tokens_if_persisted"
                    ] += tokens

                    if not critical:
                        s[
                            "baseline_dropped_nonfix_records_matching_stale_ids"
                        ] += 1

        if (
            detail_new["newly_kept_P"]
            or detail_new["stale_if_persistent"]
            or detail_new["behavioral_lease_P"]
        ):
            result["details"].append(
                detail_new
            )

    denom = result["critical_total"]

    for condition in ("B", "D", "P", "Q"):
        c = result["conditions"][condition]

        c["critical_recall"] = (
            c["critical_kept"] / denom
            if denom
            else None
        )

    q_nonfix = result["conditions"][
        "Q"
    ]["additional_nonfix_tokens"]

    p_nonfix = result["conditions"][
        "P"
    ]["additional_nonfix_tokens"]

    d_nonfix = result["conditions"][
        "D"
    ]["additional_nonfix_tokens"]

    result["P_vs_Q"] = {
        "nonfix_token_reduction": (
            (q_nonfix - p_nonfix) / q_nonfix
            if q_nonfix
            else None
        ),
        "critical_recall_equal": (
            result["conditions"]["P"][
                "critical_kept"
            ]
            == result["conditions"]["Q"][
                "critical_kept"
            ]
        ),
    }

    result["P_vs_D"] = {
        "critical_rescue_delta": (
            result["conditions"]["P"][
                "critical_DROP_to_KEEP"
            ]
            - result["conditions"]["D"][
                "critical_DROP_to_KEEP"
            ]
        ),
        "additional_nonfix_token_delta": (
            p_nonfix - d_nonfix
        ),
    }

    return result


# ---------------------------------------------------------------------
# G11 direct primary-case analysis.
# ---------------------------------------------------------------------

def historical_oracle(
    commit: str,
    path: str,
) -> dict[str, dict[int, str]]:
    diff = git_text(
        "show",
        "--format=",
        "--unified=0",
        commit,
        "--",
        path,
    )

    parent_text = git_text(
        "show",
        f"{commit}^:{path}",
    )

    parent = parent_text.splitlines()

    line_numbers: set[int] = set()

    for line in diff.splitlines():
        match = HUNK_RE.match(line)

        if not match:
            continue

        old_start = int(
            match.group("old_start")
        )

        old_count = int(
            match.group("old_count") or "1"
        )

        width = max(old_count, 1)

        start = max(1, old_start - 5)

        end = min(
            len(parent),
            old_start + width - 1 + 5,
        )

        line_numbers.update(
            range(start, end + 1)
        )

    return {
        path: {
            line: parent[line - 1]
            for line in sorted(line_numbers)
            if (
                1 <= line <= len(parent)
                and parent[line - 1].strip()
            )
        }
    }


def parse_records(
    target: str,
) -> list[dict[str, Any]]:
    rows = []

    for raw in target.splitlines():
        match = RECORD_RE.match(raw)

        if not match:
            continue

        rows.append(
            {
                "raw": raw,
                "path": normalize_path(
                    match.group("path")
                ),
                "line": int(
                    match.group("line")
                ),
                "text": match.group("text"),
            }
        )

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
        norm_path = normalize_path(path)

        if not (
            rec_path == norm_path
            or rec_path.endswith(
                "/" + norm_path
            )
        ):
            continue

        expected = lines.get(rec_line)

        if (
            expected is not None
            and expected.strip() == rec_text
        ):
            return True

        # Same conservative line-shift fallback as the
        # matched hard replay.
        for line, text in lines.items():
            if (
                abs(line - rec_line) <= 3
                and text.strip() == rec_text
            ):
                return True

    return False


def actual_v3_adopted_ids(
    events: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    event = events[index]

    history = []

    for prev in events[
        max(0, index - MAX_PRIOR_OUTPUTS):index
    ]:
        text = Path(
            prev["stdout_path"]
        ).read_text(errors="replace")

        history.append(
            {
                "role": "tool",
                "content": text,
            }
        )

    target = Path(
        event["stdout_path"]
    ).read_text(errors="replace")

    try:
        context = (
            tr.build_search_relevance_context(
                history,
                before_index=len(history),
                user_context="",
                target_content=target,
                novelty_context="",
            )
        )

        bridges = list(
            tr._trajectory_context_bridge_identifiers(
                context
            )
        )
    except Exception as exc:
        return {
            "error": repr(exc),
            "bridges": [],
            "adopted": [],
        }

    bridge_cf = {
        str(x).casefold(): str(x)
        for x in bridges
    }

    adopted = [
        ident
        for ident in event[
            "query_identifiers"
        ]
        if ident.casefold() in bridge_cf
    ]

    return {
        "bridges": bridges,
        "adopted": adopted,
    }


def normalize_plan(
    plan: Any,
) -> list[tuple[bool, str]]:
    result: list[tuple[bool, str]] = []

    if not isinstance(plan, list):
        raise TypeError(
            f"Unexpected plan type: {type(plan)!r}"
        )

    for item in plan:
        if (
            isinstance(item, (tuple, list))
            and len(item) >= 2
        ):
            result.append(
                (
                    bool(item[0]),
                    str(item[1]),
                )
            )
            continue

        keep = getattr(item, "keep", None)

        text = getattr(
            item,
            "content",
            getattr(item, "text", None),
        )

        if (
            keep is not None
            and text is not None
        ):
            result.append(
                (bool(keep), str(text))
            )
            continue

        raise TypeError(
            f"Unexpected plan item: {item!r}"
        )

    return result


def direct_plan(
    content: str,
    query: str,
    identifiers: list[str],
) -> dict[str, Any]:
    try:
        raw = plan_relevance_split(
            content,
            query,
            _DIRECT_SCORER,
            threshold=(
                _DIRECT_ROUTER.config.relevance.relevance_threshold
            ),
            adaptive=(
                _DIRECT_ROUTER.config.relevance_adaptive_threshold
            ),
            max_records=(
                _DIRECT_ROUTER.config.relevance_max_records
            ),
            force_keep_identifiers=tuple(
                identifiers
            ),
        )

        plan = normalize_plan(raw)

    except Exception as exc:
        return {
            "error": repr(exc),
            "identifiers": identifiers,
        }

    kept_segments = [
        text
        for keep, text in plan
        if keep
    ]

    dropped_segments = [
        text
        for keep, text in plan
        if not keep
    ]

    return {
        "identifiers": identifiers,
        "runs": len(plan),
        "kept_runs": len(kept_segments),
        "dropped_runs": len(
            dropped_segments
        ),
        "kept_bytes": sum(
            len(
                x.encode(
                    "utf-8",
                    errors="replace",
                )
            )
            for x in kept_segments
        ),
        "dropped_bytes": sum(
            len(
                x.encode(
                    "utf-8",
                    errors="replace",
                )
            )
            for x in dropped_segments
        ),
        "_kept_segments": kept_segments,
    }


def record_retained_by_plan(
    rec: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    segments = result.get(
        "_kept_segments",
        [],
    )

    raw = rec["raw"]

    return any(
        raw in segment
        for segment in segments
    )


def analyze_g11() -> dict[str, Any]:
    events = load_rg_events(G11_ROOT)

    if len(events) < 2:
        return {
            "error": (
                "Expected at least two fresh "
                f"G11-ON rg captures, got {len(events)}"
            )
        }

    # Primary early-adoption event is the second search.
    event = events[1]

    provisional = list(
        event[
            "prior_seen_query_identifiers"
        ]
    )

    v3 = actual_v3_adopted_ids(
        events,
        1,
    )

    adopted = list(
        v3.get("adopted", [])
    )

    naive = list(
        event["query_identifiers"]
    )

    target = Path(
        event["stdout_path"]
    ).read_text(errors="replace")

    query = event["pattern"]

    conditions = {
        "B": direct_plan(
            target,
            query,
            [],
        ),
        "D": direct_plan(
            target,
            query,
            adopted,
        ),
        "P": direct_plan(
            target,
            query,
            provisional,
        ),
        "Q": direct_plan(
            target,
            query,
            naive,
        ),
    }

    oracle = historical_oracle(
        G11_SOURCE_COMMIT,
        G11_PRODUCTION_FILE,
    )

    records = parse_records(target)

    for rec in records:
        rec["fix_adjacent"] = (
            is_fix_adjacent(
                rec,
                oracle,
            )
        )

    summary: dict[str, Any] = {}

    for name, condition in conditions.items():
        if "error" in condition:
            summary[name] = {
                "error": condition["error"],
            }
            continue

        critical_total = 0
        critical_kept = 0
        all_kept = 0

        for rec in records:
            keep = record_retained_by_plan(
                rec,
                condition,
            )

            if keep:
                all_kept += 1

            if rec["fix_adjacent"]:
                critical_total += 1

                if keep:
                    critical_kept += 1

        summary[name] = {
            "identifiers": condition[
                "identifiers"
            ],
            "runs": condition["runs"],
            "kept_runs": condition[
                "kept_runs"
            ],
            "dropped_runs": condition[
                "dropped_runs"
            ],
            "kept_bytes": condition[
                "kept_bytes"
            ],
            "dropped_bytes": condition[
                "dropped_bytes"
            ],
            "parsed_records": len(records),
            "records_kept": all_kept,
            "critical_total": critical_total,
            "critical_kept": critical_kept,
        }

    baseline_critical = (
        summary.get("B", {}).get(
            "critical_kept"
        )
    )

    for name in ("D", "P", "Q"):
        value = summary.get(
            name,
            {},
        ).get("critical_kept")

        if (
            baseline_critical is not None
            and value is not None
        ):
            summary[name][
                "critical_delta_vs_B"
            ] = value - baseline_critical

    clean_events = []

    for e in events:
        clean_events.append(
            {
                "event_index": e[
                    "event_index"
                ],
                "argv": e["argv"],
                "pattern": e["pattern"],
                "query_identifiers": e[
                    "query_identifiers"
                ],
                "prior_seen_query_identifiers": e[
                    "prior_seen_query_identifiers"
                ],
                "prior_evidence": e[
                    "prior_evidence"
                ],
                "lease_fresh": e[
                    "lease_fresh"
                ],
                "lease_renewed": e[
                    "lease_renewed"
                ],
                "lease_expired": e[
                    "lease_expired"
                ],
                "stale_if_persistent": e[
                    "stale_if_persistent"
                ],
                "target_bytes": e[
                    "target_bytes"
                ],
            }
        )

    return {
        "event_count": len(events),
        "primary_event_index": 1,
        "primary_identifier_expected": (
            "resolve_litellm_model_name"
        ),
        "production_v3": v3,
        "provisional_identifiers": provisional,
        "query_identifiers": naive,
        "conditions": summary,
        "events": clean_events,
    }


def lifecycle_summary(
    task_events: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    result = {
        "search_events": 0,
        "events_with_behavioral_adoption": 0,
        "fresh_lease_activations": 0,
        "lease_renewals": 0,
        "lease_expirations": 0,
        "events_with_stale_persistent_state": 0,
        "tasks": {},
    }

    for task, events in task_events.items():
        task_result = {
            "search_events": len(events),
            "events_with_behavioral_adoption": 0,
            "fresh_lease_activations": 0,
            "lease_renewals": 0,
            "lease_expirations": 0,
            "events_with_stale_persistent_state": 0,
        }

        for event in events:
            result["search_events"] += 1

            if event[
                "prior_seen_query_identifiers"
            ]:
                result[
                    "events_with_behavioral_adoption"
                ] += 1
                task_result[
                    "events_with_behavioral_adoption"
                ] += 1

            fresh = len(
                event["lease_fresh"]
            )

            renewed = len(
                event["lease_renewed"]
            )

            expired = len(
                event["lease_expired"]
            )

            result[
                "fresh_lease_activations"
            ] += fresh
            result[
                "lease_renewals"
            ] += renewed
            result[
                "lease_expirations"
            ] += expired

            task_result[
                "fresh_lease_activations"
            ] += fresh
            task_result[
                "lease_renewals"
            ] += renewed
            task_result[
                "lease_expirations"
            ] += expired

            if event[
                "stale_if_persistent"
            ]:
                result[
                    "events_with_stale_persistent_state"
                ] += 1
                task_result[
                    "events_with_stale_persistent_state"
                ] += 1

        result["tasks"][task] = (
            task_result
        )

    return result


def print_condition(
    label: str,
    data: dict[str, Any],
) -> None:
    print(
        f"{label}: "
        f"critical={data['critical_kept']} "
        f"rescue={data['critical_DROP_to_KEEP']} "
        f"regression={data['critical_KEEP_to_DROP']} "
        f"new={data['newly_kept_records']} "
        f"nonfix_tok={data['additional_nonfix_tokens']} "
        f"total_tok={data['additional_total_tokens']}"
    )


def main() -> None:
    verify_production_freeze()

    if not MEASURED.exists():
        die(f"Missing {MEASURED}")

    if not BRIDGES.exists():
        die(f"Missing {BRIDGES}")

    measured = json.loads(
        MEASURED.read_text()
    )

    bridge_data = json.loads(
        BRIDGES.read_text()
    )

    bridge_map = {
        (
            u["task"],
            u["request_id"],
            u["call_id"],
        ): u
        for u in bridge_data["units"]
    }

    task_events: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    print()
    print("Loading hard natural rg trajectories...")

    for task in TASKS:
        root = (
            HARD_ROOT
            / "results"
            / f"{task}-ON"
            / "rg-captures"
        )

        events = load_rg_events(root)

        task_events[task] = events

        print(
            f"{task}: "
            f"{len(events)} search event(s)"
        )

    hard = {
        "strict": analyze_hard(
            measured,
            bridge_map,
            task_events,
            "strict_hash_valid",
        ),
        "content_valid": analyze_hard(
            measured,
            bridge_map,
            task_events,
            "content_valid",
        ),
        "all": analyze_hard(
            measured,
            bridge_map,
            task_events,
            None,
        ),
    }

    lifecycle = lifecycle_summary(
        task_events
    )

    print()
    print("Analyzing fresh G11-ON...")

    g11 = analyze_g11()

    payload = {
        "evaluation_type": (
            "post_hoc_exploratory_mechanism"
        ),
        "experiment": (
            "provisional_adoption_lease_probe"
        ),
        "production_freeze": FREEZE,
        "policy": {
            "raw_candidate": (
                "no preservation"
            ),
            "provisional": (
                "bounded prior exact structured occurrence "
                "AND explicit current-query reuse; "
                "one retrieval-result lease"
            ),
            "corroborated": (
                "existing frozen V3 bridge AND "
                "explicit current-query reuse"
            ),
            "renewal": (
                "requires explicit reuse in the next "
                "retrieval query"
            ),
            "expiry": (
                "automatic when explicit retrieval-query "
                "reuse stops"
            ),
        },
        "hard_matched_retention": hard,
        "hard_natural_lifecycle": lifecycle,
        "fresh_G11_ON": g11,
        "important_limitations": [
            "Designed after observing the G11 early-adoption failure; not unseen validation.",
            "Hard retention outcomes reuse existing normalized matched-replay baseline decisions.",
            "Persistent-stale numbers are a counterfactual indefinite-adoption burden, not a claim about current production V3 persistence.",
            "Fresh G11 direct relevance-split analysis is a mechanism probe, not a fresh agent treatment-effect estimate."
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

    print()
    print("=" * 88)
    print("HARD STRICT MATCHED RETENTION")
    print("=" * 88)

    strict = hard["strict"]

    print(
        "units:",
        strict["units"],
        "matched natural captures:",
        strict[
            "units_matched_to_natural_rg_capture"
        ],
    )

    for key in ("B", "D", "P", "Q"):
        print_condition(
            key,
            strict["conditions"][key],
        )

    print()
    print(
        "P vs Q nonfix token reduction:",
        strict["P_vs_Q"][
            "nonfix_token_reduction"
        ],
    )

    print(
        "P vs Q same critical recall:",
        strict["P_vs_Q"][
            "critical_recall_equal"
        ],
    )

    print(
        "P vs D critical rescue delta:",
        strict["P_vs_D"][
            "critical_rescue_delta"
        ],
    )

    print(
        "P vs D additional nonfix token delta:",
        strict["P_vs_D"][
            "additional_nonfix_token_delta"
        ],
    )

    print()
    print("Persistent stale counterfactual:")
    print(
        json.dumps(
            strict[
                "persistent_stale_counterfactual"
            ],
            indent=2,
        )
    )

    print()
    print("=" * 88)
    print("LEASE LIFECYCLE")
    print("=" * 88)

    print(
        json.dumps(
            lifecycle,
            indent=2,
        )
    )

    print()
    print("=" * 88)
    print("FRESH G11-ON PRIMARY CASE")
    print("=" * 88)

    if "error" in g11:
        print("G11 error:", g11["error"])
    else:
        print(
            "provisional identifiers:",
            g11[
                "provisional_identifiers"
            ],
        )

        print(
            "production V3 adopted:",
            g11[
                "production_v3"
            ].get("adopted", []),
        )

        for key in ("B", "D", "P", "Q"):
            print(
                key,
                json.dumps(
                    g11["conditions"].get(
                        key,
                        {},
                    ),
                    ensure_ascii=False,
                ),
            )

    print()
    print("JSON:", OUT)


if __name__ == "__main__":
    main()
