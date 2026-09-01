"""Harder calibration benchmark for trajectory bridge scoring.

V2 differs from the first calibration set in two important ways:

1. It contains matched scenarios where one scoring signal is the only
   useful discriminator.
2. Identifier values and textual templates vary independently so lexical
   tie-breaking cannot systematically reveal the ground truth.

This remains calibration data. The final weights selected here must be
frozen before evaluation on a separately generated held-out benchmark.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

from experiments.bridge_relevance.calibrate_bridge_weights import (
    WeightConfig,
    evaluate,
    extract_raw_candidates,
    ranking_key,
)

SEED = 20260901
N_PER_KIND = 16

KINDS = (
    "request_id",
    "test_name",
    "exception",
    "file_path",
    "config_key",
    "function_name",
    "class_name",
)

SCENARIOS = (
    "recency_isolated",
    "occurrence_isolated",
    "speculation_isolated",
    "causal_isolated",
    "cross_vs_weak_causal",
    "causal_vs_cross",
    "false_recurring",
    "hypothesis_shift",
    "neutral_bridge_threshold",
    "abstain_speculative",
    "abstain_single_tool",
)


@dataclass(frozen=True)
class Case:
    scenario: str
    kind: str
    tool_outputs: tuple[str, ...]
    truth: str | None


def token_for(kind: str, number: int) -> str:
    if kind == "request_id":
        return f"req-{number:04d}"

    if kind == "test_name":
        return f"test_checkout_bridge_{number:04d}"

    if kind == "exception":
        return f"CheckoutService{number:04d}Error"

    if kind == "file_path":
        return f"src/checkout/module_{number:04d}.py"

    if kind == "config_key":
        return f"CHECKOUT_BRIDGE_{number:04d}"

    if kind == "function_name":
        return f"resolve_checkout_bridge_{number:04d}"

    if kind == "class_name":
        return f"CheckoutBridgeNode{number:04d}"

    raise ValueError(kind)


def mention(token: str, kind: str) -> str:
    # The production function-name pattern expects a call-like occurrence.
    if kind == "function_name":
        return f"{token}()"

    return token


def neutral(
    token: str,
    kind: str,
    rng: random.Random,
) -> str:
    value = mention(token, kind)

    templates = (
        "Observed reference {value}",
        "Diagnostic output contains {value}",
        "Located symbol {value}",
        "Seen during inspection: {value}",
        "Recorded reference {value}",
    )

    return rng.choice(templates).format(
        value=value
    )


def strong_positive(
    token: str,
    kind: str,
    rng: random.Random,
) -> str:
    value = mention(token, kind)

    templates = (
        "Root cause identified: {value}",
        "Affected component confirmed: {value}",
        "Failure identified at {value}",
    )

    return rng.choice(templates).format(
        value=value
    )


def weak_positive(
    token: str,
    kind: str,
    rng: random.Random,
) -> str:
    value = mention(token, kind)

    # "expected" and "actual" are deliberately weaker positive cues
    # than "root cause" / "affected".
    templates = (
        "Expected evidence points to {value}",
        "Actual value references {value}",
    )

    return rng.choice(templates).format(
        value=value
    )


def speculative(
    token: str,
    kind: str,
    rng: random.Random,
) -> str:
    value = mention(token, kind)

    templates = (
        "Initial suspicion: {value}",
        "Current hypothesis: {value}",
        "Maybe related to {value}",
        "Possible candidate: {value}",
    )

    return rng.choice(templates).format(
        value=value
    )


def token_set(
    kind: str,
    rng: random.Random,
) -> tuple[str, str, str]:
    # Random assignment prevents token lexical ordering from consistently
    # favoring either truth or distractor when scores tie.
    numbers = rng.sample(
        range(100, 9900),
        3,
    )

    return (
        token_for(kind, numbers[0]),
        token_for(kind, numbers[1]),
        token_for(kind, numbers[2]),
    )


def build_case(
    scenario: str,
    kind: str,
    rng: random.Random,
) -> Case:
    truth, wrong, other = token_set(
        kind,
        rng,
    )

    if scenario == "recency_isolated":
        # Equal:
        #   causal
        #   speculation
        #   distinct-tool recurrence
        #   total occurrence
        #
        # Only last-seen recency differs.
        outputs = (
            neutral(wrong, kind, rng),
            neutral(truth, kind, rng),
            neutral(wrong, kind, rng),
            neutral(truth, kind, rng),
        )

    elif scenario == "occurrence_isolated":
        # Both candidates:
        #   occur in exactly two tools
        #   have identical recency
        #   have no causal/speculative cues
        #
        # Truth occurs three times; wrong occurs twice.
        outputs = (
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    neutral(truth, kind, rng),
                    neutral(wrong, kind, rng),
                )
            ),
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    neutral(wrong, kind, rng),
                )
            ),
        )

    elif scenario == "speculation_isolated":
        # Both candidates have equal:
        #   occurrence
        #   cross-tool recurrence
        #   recency
        #
        # Only the distractor carries speculative language.
        outputs = (
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    speculative(wrong, kind, rng),
                )
            ),
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    speculative(wrong, kind, rng),
                )
            ),
        )

    elif scenario == "causal_isolated":
        # Equal recurrence, occurrence and recency.
        # Only truth receives explicit causal evidence.
        outputs = (
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    neutral(wrong, kind, rng),
                )
            ),
            "\n".join(
                (
                    strong_positive(
                        truth,
                        kind,
                        rng,
                    ),
                    neutral(wrong, kind, rng),
                )
            ),
        )

    elif scenario == "cross_vs_weak_causal":
        # Truth is a neutral bridge across distinct tools.
        # Distractor appears only once but receives weak positive evidence.
        #
        # Cross-tool evidence must have enough influence to prevent every
        # causal-looking single observation from dominating.
        outputs = (
            neutral(truth, kind, rng),
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    weak_positive(
                        wrong,
                        kind,
                        rng,
                    ),
                )
            ),
        )

    elif scenario == "causal_vs_cross":
        # Opposite pressure:
        #
        # Wrong is a neutral cross-tool bridge.
        # Truth is a single-tool but explicit root-cause observation.
        #
        # Strong causal evidence must be capable of defeating recurrence.
        outputs = (
            neutral(wrong, kind, rng),
            "\n".join(
                (
                    neutral(wrong, kind, rng),
                    strong_positive(
                        truth,
                        kind,
                        rng,
                    ),
                )
            ),
        )

    elif scenario == "false_recurring":
        # A stale speculative hypothesis recurs, while later explicit
        # evidence names the actual root cause.
        outputs = (
            speculative(wrong, kind, rng),
            speculative(wrong, kind, rng),
            strong_positive(
                truth,
                kind,
                rng,
            ),
        )

    elif scenario == "hypothesis_shift":
        # Longer version of stale-hypothesis pressure.
        outputs = (
            speculative(wrong, kind, rng),
            speculative(wrong, kind, rng),
            neutral(wrong, kind, rng),
            strong_positive(
                truth,
                kind,
                rng,
            ),
        )

    elif scenario == "neutral_bridge_threshold":
        # A legitimate bridge can contain no explicit causal vocabulary.
        # The confidence threshold therefore cannot be so high that it
        # rejects every neutral cross-tool bridge.
        outputs = (
            neutral(truth, kind, rng),
            neutral(truth, kind, rng),
        )

    elif scenario == "abstain_speculative":
        # There is no known truth. Two identifiers recur, but only as
        # speculative hypotheses. Correct behavior is no bridge injection.
        outputs = (
            "\n".join(
                (
                    speculative(wrong, kind, rng),
                    speculative(other, kind, rng),
                )
            ),
            "\n".join(
                (
                    speculative(wrong, kind, rng),
                    speculative(other, kind, rng),
                )
            ),
        )

        return Case(
            scenario=scenario,
            kind=kind,
            tool_outputs=outputs,
            truth=None,
        )

    elif scenario == "abstain_single_tool":
        # A lone neutral identifier does not constitute trajectory
        # corroboration and should be filtered by production eligibility.
        outputs = (
            neutral(wrong, kind, rng),
        )

        return Case(
            scenario=scenario,
            kind=kind,
            tool_outputs=outputs,
            truth=None,
        )

    else:
        raise ValueError(scenario)

    return Case(
        scenario=scenario,
        kind=kind,
        tool_outputs=outputs,
        truth=truth,
    )


def build_cases() -> list[Case]:
    rng = random.Random(SEED)

    cases: list[Case] = []

    for scenario in SCENARIOS:
        for kind in KINDS:
            for _ in range(N_PER_KIND):
                cases.append(
                    build_case(
                        scenario,
                        kind,
                        rng,
                    )
                )

    rng.shuffle(cases)

    return cases


def config_grid():
    causal_values = (
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
    )

    cross_values = (
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
    )

    occurrence_values = (
        0.0,
        1.0,
        2.0,
        3.0,
    )

    recency_values = (
        0.0,
        1.0,
        2.0,
        3.0,
    )

    speculation_values = (
        2.0,
        4.0,
        6.0,
        8.0,
    )

    thresholds = (
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        12.0,
    )

    for (
        causal,
        cross_tool,
        occurrence,
        recency,
        speculation,
    ) in itertools.product(
        causal_values,
        cross_values,
        occurrence_values,
        recency_values,
        speculation_values,
    ):
        # Structural prior:
        # maximum explicit causal evidence should not be assigned less
        # weight than maximum cross-tool recurrence.
        if causal < cross_tool:
            continue

        for threshold in thresholds:
            yield WeightConfig(
                causal=causal,
                cross_tool=cross_tool,
                occurrence=occurrence,
                recency=recency,
                speculation=speculation,
                threshold=threshold,
            )


def is_perfect(metrics: dict) -> bool:
    return (
        metrics["macro_accuracy"] == 1.0
        and metrics["positive_accuracy"] == 1.0
        and metrics["mrr"] == 1.0
        and metrics["abstain_accuracy"] == 1.0
        and metrics["false_promotion_rate"] == 0.0
    )


def main() -> None:
    cases = build_cases()

    print(
        f"Calibration V2 cases: {len(cases)}"
    )

    raw_cases = [
        extract_raw_candidates(case)
        for case in cases
    ]

    missing_truth = 0

    for case, raw in zip(
        cases,
        raw_cases,
        strict=True,
    ):
        if case.truth is None:
            continue

        candidates = {
            candidate.token
            for candidate in raw
        }

        if case.truth not in candidates:
            missing_truth += 1

    print(
        "Positive truths missing from candidate set: "
        f"{missing_truth}"
    )

    if missing_truth:
        raise SystemExit(
            "Calibration V2 invalid: candidate extraction "
            "missed positive ground truth."
        )

    results = []

    for cfg in config_grid():
        metrics = evaluate(
            cases,
            raw_cases,
            cfg,
        )

        results.append(
            {
                "config": cfg,
                "metrics": metrics,
            }
        )

    results.sort(
        key=ranking_key,
        reverse=True,
    )

    perfect_count = sum(
        is_perfect(row["metrics"])
        for row in results
    )

    best = results[0]

    current_defaults = WeightConfig(
        causal=10.0,
        cross_tool=12.0,
        occurrence=3.0,
        recency=2.0,
        speculation=5.0,
        threshold=-1_000_000.0,
    )

    current_metrics = evaluate(
        cases,
        raw_cases,
        current_defaults,
    )

    out_dir = Path(
        "experiments/bridge_relevance/results"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / "bridge_weight_calibration_v2_top50.csv"
    )

    fieldnames = (
        "rank",
        "causal",
        "cross_tool",
        "occurrence",
        "recency",
        "speculation",
        "threshold",
        "macro_accuracy",
        "positive_accuracy",
        "mrr",
        "abstain_accuracy",
        "false_promotion_rate",
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank, row in enumerate(
            results[:50],
            start=1,
        ):
            cfg = row["config"]
            metrics = row["metrics"]

            writer.writerow(
                {
                    "rank": rank,
                    "causal": cfg.causal,
                    "cross_tool": cfg.cross_tool,
                    "occurrence": cfg.occurrence,
                    "recency": cfg.recency,
                    "speculation": cfg.speculation,
                    "threshold": cfg.threshold,
                    "macro_accuracy": (
                        metrics["macro_accuracy"]
                    ),
                    "positive_accuracy": (
                        metrics["positive_accuracy"]
                    ),
                    "mrr": metrics["mrr"],
                    "abstain_accuracy": (
                        metrics["abstain_accuracy"]
                    ),
                    "false_promotion_rate": (
                        metrics[
                            "false_promotion_rate"
                        ]
                    ),
                }
            )

    best_cfg = best["config"]
    best_metrics = best["metrics"]

    summary = {
        "seed": SEED,
        "n_cases": len(cases),
        "n_configs_evaluated": len(results),
        "n_perfect_configs": perfect_count,
        "development_defaults": {
            "config": {
                "causal": current_defaults.causal,
                "cross_tool": (
                    current_defaults.cross_tool
                ),
                "occurrence": (
                    current_defaults.occurrence
                ),
                "recency": (
                    current_defaults.recency
                ),
                "speculation": (
                    current_defaults.speculation
                ),
                "threshold": None,
            },
            "metrics": current_metrics,
        },
        "best_calibration_config": {
            "causal": best_cfg.causal,
            "cross_tool": best_cfg.cross_tool,
            "occurrence": best_cfg.occurrence,
            "recency": best_cfg.recency,
            "speculation": best_cfg.speculation,
            "threshold": best_cfg.threshold,
        },
        "best_metrics": best_metrics,
        "top50_csv": str(csv_path),
    }

    json_path = (
        out_dir
        / "bridge_weight_calibration_v2_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
