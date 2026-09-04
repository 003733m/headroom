#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

ROOT = Path.home() / "headroom-hard-confirmatory-20260904"
INPUT = ROOT / "normalized_retention_measurement.json"
OUT = ROOT / "query_hit_preservation_probe.json"

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

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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

    explicit: list[str] = []
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


def is_camel_case(token: str) -> bool:
    # Require both lower and upper case, e.g. TrafficLearner, MemoryHandler.
    return (
        any(c.islower() for c in token)
        and any(c.isupper() for c in token)
        and token[0].isalpha()
    )


def structured_query_identifiers(pattern: str) -> list[str]:
    """
    Conservative current-query anchors.

    Keep:
      - snake_case / _private_name / ALL_CAPS_WITH_UNDERSCORE
      - CamelCase
      - bare identifiers explicitly following def/class

    Reject generic bare words such as warning, failed, dedup, contribution.
    """
    selected: set[str] = set()

    for m in re.finditer(
        r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        pattern,
    ):
        selected.add(m.group(1))

    for token in IDENT_RE.findall(pattern):
        if len(token) < 3:
            continue

        if "_" in token or token.startswith("_") or is_camel_case(token):
            selected.add(token)

    return sorted(selected, key=lambda x: (x.casefold(), x))


def identifier_present(text: str, identifier: str) -> bool:
    pat = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    )
    return bool(pat.search(text))


def matching_identifiers(
    text: str,
    identifiers: list[str],
) -> list[str]:
    return [
        ident
        for ident in identifiers
        if identifier_present(text, ident)
    ]


def summarize(
    units: list[dict[str, Any]],
    validity_field: str | None,
) -> dict[str, Any]:
    if validity_field is None:
        selected = list(units)
        name = "all_existing_replay_units"
    else:
        selected = [
            u for u in units
            if bool(u.get(validity_field))
        ]
        name = validity_field

    result: dict[str, Any] = {
        "stratum": name,
        "units": len(selected),
        "units_with_structured_query_identifier": 0,
        "units_with_baseline_dropped_query_hit": 0,
        "scorable_record_exposures": 0,
        "historical_fix_adjacent_scorable": 0,
        "baseline_critical_kept": 0,
        "floor_critical_kept": 0,
        "critical_DROP_to_KEEP": 0,
        "critical_KEEP_to_DROP": 0,
        "newly_kept_records": 0,
        "newly_kept_fix_adjacent_records": 0,
        "newly_kept_nonfix_records": 0,
        "additional_fix_adjacent_record_tokens": 0,
        "additional_nonfix_record_tokens": 0,
        "additional_total_record_tokens": 0,
    }

    task_rows: list[dict[str, Any]] = []

    for unit in selected:
        pattern = extract_search_pattern(unit.get("command", ""))
        identifiers = structured_query_identifiers(pattern)

        if identifiers:
            result["units_with_structured_query_identifier"] += 1

        unit_new = []
        unit_has_dropped_hit = False

        for rec in unit["records"]:
            if not rec.get("scorable"):
                continue

            result["scorable_record_exposures"] += 1

            is_critical = bool(rec.get("fix_adjacent"))
            if is_critical:
                result["historical_fix_adjacent_scorable"] += 1

            baseline_keep = bool(rec["kept"]["B"])

            hits = matching_identifiers(
                rec.get("text", ""),
                identifiers,
            )

            floor_keep = baseline_keep or bool(hits)

            if is_critical and baseline_keep:
                result["baseline_critical_kept"] += 1

            if is_critical and floor_keep:
                result["floor_critical_kept"] += 1

            if is_critical and (not baseline_keep) and floor_keep:
                result["critical_DROP_to_KEEP"] += 1

            # Monotonic floor should make this impossible.
            if is_critical and baseline_keep and not floor_keep:
                result["critical_KEEP_to_DROP"] += 1

            if (not baseline_keep) and hits:
                unit_has_dropped_hit = True
                result["newly_kept_records"] += 1

                rtok = int(rec.get("record_tokens") or 0)
                result["additional_total_record_tokens"] += rtok

                if is_critical:
                    result["newly_kept_fix_adjacent_records"] += 1
                    result["additional_fix_adjacent_record_tokens"] += rtok
                else:
                    result["newly_kept_nonfix_records"] += 1
                    result["additional_nonfix_record_tokens"] += rtok

                unit_new.append(
                    {
                        "path": rec["path"],
                        "line": rec["line"],
                        "fix_adjacent": is_critical,
                        "matched_identifiers": hits,
                        "record_tokens": rtok,
                        "text": rec["text"],
                    }
                )

        if unit_has_dropped_hit:
            result["units_with_baseline_dropped_query_hit"] += 1

        if unit_new:
            task_rows.append(
                {
                    "task": unit["task"],
                    "request_id": unit["request_id"],
                    "call_id": unit["call_id"],
                    "command": unit["command"],
                    "pattern": pattern,
                    "structured_query_identifiers": identifiers,
                    "newly_kept": unit_new,
                }
            )

    denom = result["historical_fix_adjacent_scorable"]

    result["baseline_critical_recall"] = (
        result["baseline_critical_kept"] / denom
        if denom
        else None
    )

    result["floor_critical_recall"] = (
        result["floor_critical_kept"] / denom
        if denom
        else None
    )

    result["critical_recall_delta"] = (
        result["floor_critical_recall"]
        - result["baseline_critical_recall"]
        if denom
        else None
    )

    new_total = result["newly_kept_records"]

    result["fix_adjacent_fraction_of_newly_kept"] = (
        result["newly_kept_fix_adjacent_records"] / new_total
        if new_total
        else None
    )

    result["details"] = task_rows

    return result


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing input: {INPUT}")

    d = json.loads(INPUT.read_text())
    units = d["units"]

    summaries = {
        "strict": summarize(units, "strict_hash_valid"),
        "content_valid": summarize(units, "content_valid"),
        "all": summarize(units, None),
    }

    payload = {
        "evaluation_type": "post_hoc_exploratory_mechanism",
        "algorithm": "query_hit_preservation_floor_v1",
        "base_algorithm_freeze": d["algorithm_freeze"],
        "input_measurement": str(INPUT),
        "summaries": summaries,
    }

    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )

    for key in ("strict", "content_valid", "all"):
        s = summaries[key]

        print()
        print("=" * 86)
        print(key.upper())
        print("=" * 86)

        print(
            f"units={s['units']} "
            f"query_id_units={s['units_with_structured_query_identifier']} "
            f"dropped_hit_units={s['units_with_baseline_dropped_query_hit']}"
        )

        print(
            "critical: "
            f"{s['baseline_critical_kept']}/"
            f"{s['historical_fix_adjacent_scorable']} "
            f"-> "
            f"{s['floor_critical_kept']}/"
            f"{s['historical_fix_adjacent_scorable']}"
        )

        print(
            "recall: "
            f"{s['baseline_critical_recall']} "
            f"-> {s['floor_critical_recall']} "
            f"delta={s['critical_recall_delta']}"
        )

        print(
            f"critical rescue={s['critical_DROP_to_KEEP']} "
            f"regression={s['critical_KEEP_to_DROP']}"
        )

        print(
            f"newly_kept={s['newly_kept_records']} "
            f"fix_adjacent={s['newly_kept_fix_adjacent_records']} "
            f"nonfix={s['newly_kept_nonfix_records']}"
        )

        print(
            f"additional tokens: "
            f"critical={s['additional_fix_adjacent_record_tokens']} "
            f"noncritical={s['additional_nonfix_record_tokens']} "
            f"total={s['additional_total_record_tokens']}"
        )

        print(
            "fix-adjacent fraction among additions:",
            s["fix_adjacent_fraction_of_newly_kept"],
        )

        for detail in s["details"]:
            print()
            print(
                f"{detail['task']} "
                f"{detail['command']}"
            )

            for r in detail["newly_kept"]:
                label = "CRITICAL" if r["fix_adjacent"] else "nonfix"
                print(
                    f"  [{label}] "
                    f"{r['path']}:{r['line']} "
                    f"ids={r['matched_identifiers']} "
                    f"tokens={r['record_tokens']}"
                )
                print(
                    "    ",
                    r["text"][:220],
                )

    print()
    print("JSON:", OUT)


if __name__ == "__main__":
    main()
