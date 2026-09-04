#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / "headroom-hard-confirmatory-20260904"

HARD = ROOT / "hard_matched_context_replay_results.json"
V41 = ROOT / "retrieval_anchored_v41_matched_replay.json"
OUT = ROOT / "normalized_retention_measurement.json"

TASKS = ["G10", "G09", "G07", "G19", "G04", "G21"]

spec = importlib.util.spec_from_file_location(
    "matched_base",
    HERE / "run_hard_matched_context_replay.py",
)
if spec is None or spec.loader is None:
    raise SystemExit("Cannot load matched replay helpers")

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sha(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def informative(text: str) -> bool:
    # Reject punctuation-only / structurally ambiguous records such as ")".
    return bool(re.search(r"[A-Za-z0-9_]{3,}", text))


def present(output: str | None, record_text: str) -> bool:
    if output is None:
        return False

    needle = norm(record_text)
    if not needle:
        return False

    return needle in norm(output)


def key(unit: dict[str, Any]) -> tuple[str, str, str]:
    return (
        unit["request_id"],
        unit["call_id"],
        unit["target_sha256"],
    )


hard = json.loads(HARD.read_text())
v41 = json.loads(V41.read_text())

manifest, _tasks = base.load_manifest()
model = manifest["model"]
tok = base.exact.ProductionLikeTokenCounter(model)

analyzed = []

for task in TASKS:
    hard_map = {
        key(u): u
        for u in hard["tasks"][task]["units"]
    }

    rows = base.load_wire(task)
    row_map = {
        r["request_id"]: r
        for r in rows
    }

    print()
    print("=" * 90)
    print(task)
    print("=" * 90)

    for vu in v41["tasks"][task]["units"]:
        k = key(vu)

        if k not in hard_map:
            print("MISSING HARD MATCH:", k)
            continue

        hu = hard_map[k]

        req = vu["request_id"]
        cid = vu["call_id"]

        captured = base.exact.output_by_call_id(
            row_map[req]["post"],
            cid,
        )

        a_output = hu["outputs"].get("A")

        outputs = {
            "B": vu["outputs"].get("B"),
            "C": vu["outputs"].get("C"),
            "E": vu["outputs"].get("E"),
        }

        # Original target records. Determine whether line content uniquely
        # identifies one source record inside this target.
        records = vu["records"]

        normalized = [
            norm(r["text"])
            for r in records
        ]
        counts = Counter(
            x for x in normalized
            if x
        )

        measured = []

        captured_vec = []
        a_vec = []

        for rec, ntext in zip(records, normalized):
            scorable = (
                bool(ntext)
                and informative(ntext)
                and counts[ntext] == 1
            )

            if not scorable:
                measured.append({
                    "path": rec["path"],
                    "line": rec["line"],
                    "text": rec["text"],
                    "fix_adjacent": rec["fix_adjacent"],
                    "scorable": False
                })
                continue

            cp = present(captured, rec["text"])
            ap = present(a_output, rec["text"])

            kept = {
                mode: present(out, rec["text"])
                for mode, out in outputs.items()
            }

            captured_vec.append(cp)
            a_vec.append(ap)

            measured.append({
                "path": rec["path"],
                "line": rec["line"],
                "text": rec["text"],
                "fix_adjacent": rec["fix_adjacent"],
                "scorable": True,
                "captured_keep": cp,
                "A_keep": ap,
                "kept": kept,
                "record_tokens": tok.count_text(rec["raw"]),
            })

        strict = (
            captured is not None
            and a_output is not None
            and sha(captured) == sha(a_output)
        )

        content_valid = (
            bool(captured_vec)
            and captured_vec == a_vec
        )

        bc_same_as_previous_run = (
            sha(vu["outputs"].get("B"))
            == sha(hu["outputs"].get("B"))
            and
            sha(vu["outputs"].get("C"))
            == sha(hu["outputs"].get("C"))
        )

        row = {
            "task": task,
            "request_id": req,
            "call_id": cid,
            "command": vu["command"],
            "strict_hash_valid": strict,
            "content_valid": content_valid,
            "BC_replay_stable_across_runs": bc_same_as_previous_run,
            "records": measured,
            "outputs": outputs,
        }

        analyzed.append(row)

        critical = [
            r for r in measured
            if r["fix_adjacent"]
        ]
        critical_scorable = [
            r for r in critical
            if r["scorable"]
        ]

        if critical:
            print()
            print("command:", vu["command"])
            print(
                "strict=", strict,
                "content_valid=", content_valid,
                "critical_total=", len(critical),
                "critical_scorable=", len(critical_scorable),
            )

            for rec in critical:
                if not rec["scorable"]:
                    print(
                        f"  UNSCORABLE {rec['path']}:{rec['line']} "
                        f"{rec['text']!r}"
                    )
                    continue

                print(
                    f"  {rec['path']}:{rec['line']} "
                    f"B={rec['kept']['B']} "
                    f"C={rec['kept']['C']} "
                    f"E={rec['kept']['E']} "
                    f"| {rec['text'][:160]}"
                )


def summarize(name: str, predicate):
    units = [u for u in analyzed if predicate(u)]

    out = {
        "stratum": name,
        "units": len(units),
        "critical_record_exposures_total": 0,
        "critical_record_exposures_scorable": 0,
        "noncritical_record_exposures_scorable": 0,
    }

    for mode in ("B", "C", "E"):
        out[mode] = {
            "critical_kept": 0,
            "noncritical_kept_record_tokens": 0,
            "full_output_tokens": 0,
        }

    contrasts = {
        "B_to_C": {"rescue": 0, "regress": 0},
        "B_to_E": {"rescue": 0, "regress": 0},
        "C_to_E": {"rescue": 0, "regress": 0},
    }

    for unit in units:
        for mode in ("B", "C", "E"):
            output = unit["outputs"].get(mode)
            if output is not None:
                out[mode]["full_output_tokens"] += tok.count_text(output)

        for rec in unit["records"]:
            if rec["fix_adjacent"]:
                out["critical_record_exposures_total"] += 1

            if not rec["scorable"]:
                continue

            if rec["fix_adjacent"]:
                out["critical_record_exposures_scorable"] += 1
            else:
                out["noncritical_record_exposures_scorable"] += 1

            for mode in ("B", "C", "E"):
                if not rec["kept"][mode]:
                    continue

                if rec["fix_adjacent"]:
                    out[mode]["critical_kept"] += 1
                else:
                    out[mode]["noncritical_kept_record_tokens"] += (
                        rec["record_tokens"]
                    )

            if rec["fix_adjacent"]:
                for cname, left, right in (
                    ("B_to_C", "B", "C"),
                    ("B_to_E", "B", "E"),
                    ("C_to_E", "C", "E"),
                ):
                    lk = rec["kept"][left]
                    rk = rec["kept"][right]

                    if not lk and rk:
                        contrasts[cname]["rescue"] += 1
                    if lk and not rk:
                        contrasts[cname]["regress"] += 1

    denom = out["critical_record_exposures_scorable"]

    out["critical_scorable_coverage"] = (
        denom / out["critical_record_exposures_total"]
        if out["critical_record_exposures_total"]
        else None
    )

    for mode in ("B", "C", "E"):
        out[mode]["critical_recall"] = (
            out[mode]["critical_kept"] / denom
            if denom
            else None
        )

        burden = out[mode]["noncritical_kept_record_tokens"]

        out[mode]["evidence_efficiency"] = (
            out[mode]["critical_kept"] / (burden / 1000.0)
            if burden
            else None
        )

    out.update(contrasts)
    return out


all_summary = summarize(
    "all_existing_replay_units",
    lambda _u: True,
)

content_summary = summarize(
    "content_valid",
    lambda u: u["content_valid"],
)

strict_summary = summarize(
    "strict_hash_valid",
    lambda u: u["strict_hash_valid"],
)

stability = {
    "units": len(analyzed),
    "BC_replay_stable_across_runs": sum(
        u["BC_replay_stable_across_runs"]
        for u in analyzed
    ),
}

payload = {
    "measurement": "normalized_unique_record_content_retention",
    "algorithm_freeze": base.FREEZE,
    "no_new_compression_run": True,
    "stability": stability,
    "summaries": {
        "all": all_summary,
        "content_valid": content_summary,
        "strict": strict_summary,
    },
    "units": analyzed,
}

OUT.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
)

print()
print("=" * 90)
print("AGGREGATES")
print("=" * 90)

for summary in (
    all_summary,
    content_summary,
    strict_summary,
):
    print()
    print(summary["stratum"])
    print(
        "units=",
        summary["units"],
        "critical=",
        f"{summary['critical_record_exposures_scorable']}/"
        f"{summary['critical_record_exposures_total']}",
        "coverage=",
        summary["critical_scorable_coverage"],
    )

    for mode in ("B", "C", "E"):
        m = summary[mode]
        print(
            f"  {mode}: "
            f"critical={m['critical_kept']} "
            f"recall={m['critical_recall']} "
            f"noncritical_tokens={m['noncritical_kept_record_tokens']} "
            f"full_output_tokens={m['full_output_tokens']} "
            f"eff={m['evidence_efficiency']}"
        )

    print("  B→C:", summary["B_to_C"])
    print("  B→E:", summary["B_to_E"])
    print("  C→E:", summary["C_to_E"])

print()
print("B/C stable across previous replay:", stability)
print("JSON:", OUT)
