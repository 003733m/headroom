"""Controlled multi-position benchmark for trajectory-aware evidence fidelity.

Measures whether a bridge identifier discovered in prior tool outputs helps
preserve the exact relevant search-result line under the normal coding-profile
lossless-then-lossy compression path.

This is a controlled mechanism benchmark, not a natural coding-task benchmark.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from headroom.trajectory_relevance import (
    build_responses_search_relevance_context,
)
from headroom.transforms.content_router import ContentRouter
from headroom.transforms.kompress_compressor import warm_kompress_model

SEED = 20260901
N_CASES = 20
N_LINES = 500

RESULTS_DIR = Path("experiments/bridge_relevance/results")
CSV_PATH = RESULTS_DIR / "evidence_fidelity_multitarget.csv"
SUMMARY_PATH = RESULTS_DIR / "evidence_fidelity_multitarget_summary.json"


def build_search_output(target_position: int, target_id: str) -> tuple[str, str]:
    lines: list[str] = []

    for i in range(1, N_LINES + 1):
        request_id = f"req-{10000 + i}"

        if i == target_position:
            request_id = target_id

        lines.append(
            f"src/module_{i:05d}.py:{i}: "
            f"worker diagnostic trace_ref='{request_id}'"
        )

    relevant_evidence = (
        f"module_{target_position:05d}.py:{target_position}: "
        f"worker diagnostic trace_ref='{target_id}'"
    )

    return "\n".join(lines) + "\n", relevant_evidence


def build_bridge_context(target_id: str) -> str:
    # Two distinct prior tool results independently converge on the same ID.
    # Positive causal cues intentionally mirror the bridge scorer's intended
    # use case rather than relying on recurrence alone.
    items = [
        {
            "type": "function_call_output",
            "output": (
                f"Worker failure affected request {target_id}. "
                "The diagnostic indicates the affected execution."
            ),
        },
        {
            "type": "function_call_output",
            "output": (
                f"Confirmed root cause for request {target_id}. "
                "Inspect the matching worker diagnostic in the next search."
            ),
        },
    ]

    return build_responses_search_relevance_context(
        items,
        before_index=len(items),
    )


def run_case(case_index: int, position: int) -> dict[str, object]:
    # Unique bridge ID per case so the benchmark does not repeatedly test
    # one memorized/special identifier.
    target_id = f"req-{2000 + case_index:04d}"

    content, relevant_evidence = build_search_output(position, target_id)
    context = build_bridge_context(target_id)

    bridge_correct = target_id in context

    # Fresh routers prevent case-to-case router state from becoming part of
    # the comparison. The process-global Kompress model remains warmed.
    off_router = ContentRouter()
    on_router = ContentRouter()

    # Match the coding/proxy path observed in the live OpenCode experiment:
    # lossless compaction may be followed by whole-output Kompress.
    off_router._lossless_then_lossy = True
    on_router._lossless_then_lossy = True

    off = off_router.compress(
        content,
        context=context,
        trajectory_search_relevance=False,
    )

    on = on_router.compress(
        content,
        context=context,
        trajectory_search_relevance=True,
    )

    # Harness invariants: abort immediately if the benchmark stops
    # exercising the intended OFF/ON paths.
    assert target_id in context, context
    assert "lossless_search" in off.strategy_chain, off.strategy_chain
    assert "relevance_split" in on.strategy_chain, on.strategy_chain
    assert target_id in on.compressed, target_id

    original_chars = len(content)
    off_chars = len(off.compressed)
    on_chars = len(on.compressed)

    off_evidence = relevant_evidence in off.compressed
    on_evidence = relevant_evidence in on.compressed

    off_identifier = target_id in off.compressed
    on_identifier = target_id in on.compressed

    return {
        "case": case_index,
        "position": position,
        "target_id": target_id,
        "bridge_correct": bridge_correct,
        "original_chars": original_chars,
        "off_chars": off_chars,
        "on_chars": on_chars,
        "off_reduction_pct": 100.0 * (1.0 - off_chars / original_chars),
        "on_reduction_pct": 100.0 * (1.0 - on_chars / original_chars),
        "compression_cost_pp": (
            100.0 * (1.0 - off_chars / original_chars)
            - 100.0 * (1.0 - on_chars / original_chars)
        ),
        "on_extra_chars_vs_off": on_chars - off_chars,
        "off_identifier_retained": off_identifier,
        "on_identifier_retained": on_identifier,
        "off_evidence_retained": off_evidence,
        "on_evidence_retained": on_evidence,
        "quality_improved": (not off_evidence) and on_evidence,
        "quality_degraded": off_evidence and (not on_evidence),
        "off_strategy": off.strategy_used.value,
        "on_strategy": on.strategy_used.value,
        "off_chain": " > ".join(off.strategy_chain),
        "on_chain": " > ".join(on.strategy_chain),
    }


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("===== WARM KOMPRESS =====")
    ready = warm_kompress_model(
        device="cpu",
        allow_download=False,
    )
    print("ready:", ready)

    if not ready:
        raise SystemExit("Kompress is not ready; benchmark aborted.")

    rng = random.Random(SEED)

    # Exclude only the first/last four records so every case has surrounding
    # context. Sampling is fixed before any case is evaluated.
    positions = sorted(rng.sample(range(5, N_LINES - 3), N_CASES))

    print("seed:", SEED)
    print("positions:", positions)
    print()

    rows: list[dict[str, object]] = []

    for case_index, position in enumerate(positions, start=1):
        row = run_case(case_index, position)
        rows.append(row)

        print(
            f"[{case_index:02d}/{N_CASES}] "
            f"pos={position:3d} "
            f"id={row['target_id']} "
            f"bridge={row['bridge_correct']} "
            f"exact OFF={row['off_evidence_retained']} "
            f"ON={row['on_evidence_retained']} "
            f"reduction OFF={row['off_reduction_pct']:.2f}% "
            f"ON={row['on_reduction_pct']:.2f}%"
        )

    fieldnames = list(rows[0].keys())

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    off_evidence_count = sum(bool(r["off_evidence_retained"]) for r in rows)
    on_evidence_count = sum(bool(r["on_evidence_retained"]) for r in rows)

    off_id_count = sum(bool(r["off_identifier_retained"]) for r in rows)
    on_id_count = sum(bool(r["on_identifier_retained"]) for r in rows)

    bridge_count = sum(bool(r["bridge_correct"]) for r in rows)

    improved_count = sum(bool(r["quality_improved"]) for r in rows)
    degraded_count = sum(bool(r["quality_degraded"]) for r in rows)

    off_reductions = [float(r["off_reduction_pct"]) for r in rows]
    on_reductions = [float(r["on_reduction_pct"]) for r in rows]
    compression_costs = [float(r["compression_cost_pp"]) for r in rows]
    extra_chars = [int(r["on_extra_chars_vs_off"]) for r in rows]

    summary = {
        "benchmark": "controlled_multi_position_evidence_fidelity",
        "status": "controlled_mechanism_evaluation",
        "seed": SEED,
        "n_cases": N_CASES,
        "n_lines_per_case": N_LINES,
        "positions": positions,
        "kompress_ready": ready,
        "bridge_context_correct": {
            "count": bridge_count,
            "rate": bridge_count / N_CASES,
        },
        "relevant_evidence_retention": {
            "off_count": off_evidence_count,
            "off_rate": off_evidence_count / N_CASES,
            "on_count": on_evidence_count,
            "on_rate": on_evidence_count / N_CASES,
            "absolute_gain": (on_evidence_count - off_evidence_count) / N_CASES,
        },
        "identifier_retention": {
            "off_count": off_id_count,
            "off_rate": off_id_count / N_CASES,
            "on_count": on_id_count,
            "on_rate": on_id_count / N_CASES,
        },
        "paired_quality": {
            "improved_cases": improved_count,
            "degraded_cases": degraded_count,
            "unchanged_cases": N_CASES - improved_count - degraded_count,
        },
        "compression": {
            "off_mean_reduction_pct": statistics.mean(off_reductions),
            "off_median_reduction_pct": statistics.median(off_reductions),
            "on_mean_reduction_pct": statistics.mean(on_reductions),
            "on_median_reduction_pct": statistics.median(on_reductions),
            "mean_compression_cost_pp": statistics.mean(compression_costs),
            "median_compression_cost_pp": statistics.median(compression_costs),
            "mean_on_extra_chars_vs_off": statistics.mean(extra_chars),
            "median_on_extra_chars_vs_off": statistics.median(extra_chars),
        },
        "strategy_chains": {
            "off": dict(Counter(str(r["off_chain"]) for r in rows)),
            "on": dict(Counter(str(r["on_chain"]) for r in rows)),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== SUMMARY =====")
    print(
        f"bridge context correct: "
        f"{bridge_count}/{N_CASES} ({pct(bridge_count, N_CASES):.1f}%)"
    )
    print(
        f"exact evidence OFF: "
        f"{off_evidence_count}/{N_CASES} ({pct(off_evidence_count, N_CASES):.1f}%)"
    )
    print(
        f"exact evidence ON:  "
        f"{on_evidence_count}/{N_CASES} ({pct(on_evidence_count, N_CASES):.1f}%)"
    )
    print(
        f"paired improvements: {improved_count}; "
        f"degradations: {degraded_count}"
    )

    print()
    print(
        f"OFF mean/median reduction: "
        f"{statistics.mean(off_reductions):.2f}% / "
        f"{statistics.median(off_reductions):.2f}%"
    )
    print(
        f"ON  mean/median reduction: "
        f"{statistics.mean(on_reductions):.2f}% / "
        f"{statistics.median(on_reductions):.2f}%"
    )
    print(
        f"mean/median compression cost: "
        f"{statistics.mean(compression_costs):.2f} / "
        f"{statistics.median(compression_costs):.2f} percentage points"
    )

    print()
    print("OFF chains:", summary["strategy_chains"]["off"])
    print("ON chains:", summary["strategy_chains"]["on"])

    print()
    print("CSV:", CSV_PATH)
    print("summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
