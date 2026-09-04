#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / "headroom-hard-confirmatory-20260904"

MEASURED = ROOT / "normalized_retention_measurement.json"
BRIDGES = ROOT / "retrieval_anchored_v41_probe.json"
OUT = ROOT / "adopted_bridge_floor_probe.json"

spec = importlib.util.spec_from_file_location(
    "query_floor",
    HERE / "probe_query_hit_preservation.py",
)
if spec is None or spec.loader is None:
    raise SystemExit("Could not load query-hit floor helpers")

qh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qh)


def unit_key(task: str, u: dict[str, Any]):
    return (
        task,
        u["request_id"],
        u["call_id"],
    )


measured = json.loads(MEASURED.read_text())
bridge_data = json.loads(BRIDGES.read_text())

bridge_map = {
    (
        u["task"],
        u["request_id"],
        u["call_id"],
    ): u
    for u in bridge_data["units"]
}


def analyze(validity: str | None):
    if validity is None:
        units = measured["units"]
        name = "all_existing_replay_units"
    else:
        units = [
            u for u in measured["units"]
            if bool(u.get(validity))
        ]
        name = validity

    result = {
        "stratum": name,
        "units": len(units),
        "units_with_adopted_bridge": 0,
        "units_with_new_retention": 0,
        "critical_total": 0,
        "baseline_critical_kept": 0,
        "floor_critical_kept": 0,
        "critical_rescue": 0,
        "critical_regression": 0,
        "newly_kept_records": 0,
        "newly_kept_critical": 0,
        "newly_kept_nonfix": 0,
        "additional_critical_tokens": 0,
        "additional_nonfix_tokens": 0,
        "additional_total_tokens": 0,
        "details": [],
    }

    for u in units:
        task = u["task"]
        key = unit_key(task, u)

        bridge = bridge_map.get(key, {})
        v3 = bridge.get("v3", [])

        pattern = qh.extract_search_pattern(
            u.get("command", "")
        )

        query_ids = qh.structured_query_identifiers(pattern)

        v3_by_casefold = {
            str(x).casefold(): str(x)
            for x in v3
        }

        adopted = [
            ident
            for ident in query_ids
            if ident.casefold() in v3_by_casefold
        ]

        if adopted:
            result["units_with_adopted_bridge"] += 1

        newly_kept = []

        for rec in u["records"]:
            if not rec.get("scorable"):
                continue

            critical = bool(rec.get("fix_adjacent"))
            baseline = bool(rec["kept"]["B"])

            if critical:
                result["critical_total"] += 1
                if baseline:
                    result["baseline_critical_kept"] += 1

            hits = qh.matching_identifiers(
                rec.get("text", ""),
                adopted,
            )

            floor = baseline or bool(hits)

            if critical and floor:
                result["floor_critical_kept"] += 1

            if critical and (not baseline) and floor:
                result["critical_rescue"] += 1

            if critical and baseline and not floor:
                result["critical_regression"] += 1

            if baseline or not hits:
                continue

            tokens = int(rec.get("record_tokens") or 0)

            result["newly_kept_records"] += 1
            result["additional_total_tokens"] += tokens

            if critical:
                result["newly_kept_critical"] += 1
                result["additional_critical_tokens"] += tokens
            else:
                result["newly_kept_nonfix"] += 1
                result["additional_nonfix_tokens"] += tokens

            newly_kept.append({
                "path": rec["path"],
                "line": rec["line"],
                "fix_adjacent": critical,
                "matched_identifiers": hits,
                "tokens": tokens,
                "text": rec["text"],
            })

        if newly_kept:
            result["units_with_new_retention"] += 1

            result["details"].append({
                "task": task,
                "request_id": u["request_id"],
                "call_id": u["call_id"],
                "command": u["command"],
                "current_query_identifiers": query_ids,
                "frozen_v3_bridges": v3,
                "adopted_bridge_identifiers": adopted,
                "newly_kept": newly_kept,
            })

    denom = result["critical_total"]

    result["baseline_critical_recall"] = (
        result["baseline_critical_kept"] / denom
        if denom else None
    )

    result["floor_critical_recall"] = (
        result["floor_critical_kept"] / denom
        if denom else None
    )

    result["critical_recall_delta"] = (
        result["floor_critical_recall"]
        - result["baseline_critical_recall"]
        if denom else None
    )

    new = result["newly_kept_records"]

    result["critical_fraction_of_newly_kept"] = (
        result["newly_kept_critical"] / new
        if new else None
    )

    return result


summaries = {
    "strict": analyze("strict_hash_valid"),
    "content_valid": analyze("content_valid"),
    "all": analyze(None),
}

payload = {
    "evaluation_type": "post_hoc_exploratory_mechanism",
    "algorithm": "adopted_bridge_preservation_floor_v1",
    "summaries": summaries,
}

OUT.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
)

for name in ("strict", "content_valid", "all"):
    s = summaries[name]

    print()
    print("=" * 88)
    print(name.upper())
    print("=" * 88)

    print(
        f"units={s['units']} "
        f"adopted_bridge_units={s['units_with_adopted_bridge']} "
        f"new_retention_units={s['units_with_new_retention']}"
    )

    print(
        f"critical: "
        f"{s['baseline_critical_kept']}/{s['critical_total']} "
        f"-> "
        f"{s['floor_critical_kept']}/{s['critical_total']}"
    )

    print(
        f"recall: "
        f"{s['baseline_critical_recall']} "
        f"-> {s['floor_critical_recall']} "
        f"delta={s['critical_recall_delta']}"
    )

    print(
        f"critical rescue={s['critical_rescue']} "
        f"regression={s['critical_regression']}"
    )

    print(
        f"newly kept: "
        f"total={s['newly_kept_records']} "
        f"critical={s['newly_kept_critical']} "
        f"nonfix={s['newly_kept_nonfix']}"
    )

    print(
        f"additional tokens: "
        f"critical={s['additional_critical_tokens']} "
        f"nonfix={s['additional_nonfix_tokens']} "
        f"total={s['additional_total_tokens']}"
    )

    print(
        "critical fraction among additions:",
        s["critical_fraction_of_newly_kept"],
    )

    for d in s["details"]:
        print()
        print(d["task"], d["command"])
        print("  query ids:", d["current_query_identifiers"])
        print("  V3 bridges:", d["frozen_v3_bridges"])
        print("  adopted:", d["adopted_bridge_identifiers"])

        for r in d["newly_kept"]:
            tag = "CRITICAL" if r["fix_adjacent"] else "nonfix"
            print(
                f"    [{tag}] "
                f"{r['path']}:{r['line']} "
                f"ids={r['matched_identifiers']} "
                f"tokens={r['tokens']}"
            )
            print("      ", r["text"][:200])

print()
print("JSON:", OUT)
