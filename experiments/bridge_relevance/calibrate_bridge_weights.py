"""Calibrate trajectory-bridge scoring on disjoint synthetic trajectories.

This is a CALIBRATION benchmark, not the final test set.

The production candidate extractor is used once to obtain raw trajectory
features. Candidate scores are then recomputed cheaply over a constrained
weight grid.

The final chosen configuration must later be frozen and evaluated on a
separate held-out benchmark and real coding-agent tasks.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from headroom.trajectory_relevance import rank_bridge_candidates

SEED = 91021
N_PER_CATEGORY = 12

KINDS = (
    "request_id",
    "test_name",
    "exception",
    "file_path",
    "config_key",
    "function_name",
    "class_name",
)

POSITIVE_SCENARIOS = (
    "neutral_cross_tool",
    "false_recurring",
    "hypothesis_shift",
    "causal_single_tool",
    "multiple_bridges",
    "late_corroboration",
    "noisy_repetition",
)

ABSTAIN_SCENARIO = "abstain_speculative"


@dataclass(frozen=True)
class Case:
    scenario: str
    kind: str
    tool_outputs: tuple[str, ...]
    truth: str | None


@dataclass(frozen=True)
class RawCandidate:
    token: str
    causal: float
    cross_tool: float
    occurrence: float
    recency: float
    speculation: float


@dataclass(frozen=True)
class WeightConfig:
    causal: float
    cross_tool: float
    occurrence: float
    recency: float
    speculation: float
    threshold: float


def token_for(kind: str, n: int) -> str:
    if kind == "request_id":
        return f"req-{n:04d}"

    if kind == "test_name":
        return f"test_checkout_bridge_{n:04d}"

    if kind == "exception":
        return f"CheckoutService{n:04d}Error"

    if kind == "file_path":
        return f"src/checkout/module_{n:04d}.py"

    if kind == "config_key":
        return f"CHECKOUT_BRIDGE_{n:04d}"

    if kind == "function_name":
        return f"resolve_checkout_bridge_{n:04d}"

    if kind == "class_name":
        return f"CheckoutBridge{n:04d}"

    raise ValueError(kind)


def mention(token: str, kind: str) -> str:
    # Production function-name candidate regex requires a following "(".
    if kind == "function_name":
        return f"{token}()"

    return token


def neutral(token: str, kind: str) -> str:
    return f"Observed reference {mention(token, kind)}"


def positive(token: str, kind: str) -> str:
    return f"Root cause identified: {mention(token, kind)}"


def affected(token: str, kind: str) -> str:
    return f"Affected component: {mention(token, kind)}"


def speculative(token: str, kind: str) -> str:
    return f"Initial suspicion maybe {mention(token, kind)}"


def hypothesis(token: str, kind: str) -> str:
    return f"Current hypothesis {mention(token, kind)}"


def make_case(
    scenario: str,
    kind: str,
    base: int,
) -> Case:
    truth = token_for(kind, base)

    d1 = token_for(kind, base + 1000)
    d2 = token_for(kind, base + 2000)
    d3 = token_for(kind, base + 3000)

    if scenario == "neutral_cross_tool":
        # Same total occurrence can appear in a distractor, but the true
        # candidate is distributed across distinct tools.
        outputs = (
            "\n".join(
                [
                    neutral(truth, kind),
                    neutral(d1, kind),
                    neutral(d1, kind),
                    neutral(d3, kind),
                ]
            ),
            "\n".join(
                [
                    neutral(truth, kind),
                    neutral(d2, kind),
                    neutral(d2, kind),
                ]
            ),
        )

    elif scenario == "false_recurring":
        # Wrong hypothesis recurs across tools, while a later single-tool
        # observation contains explicit causal evidence for the truth.
        outputs = (
            speculative(d1, kind),
            hypothesis(d1, kind),
            positive(truth, kind),
        )

    elif scenario == "hypothesis_shift":
        # Stronger stale-hypothesis case: the wrong candidate persists
        # across several earlier tools before the final root-cause result.
        outputs = (
            speculative(d1, kind),
            hypothesis(d1, kind),
            f"Continuing investigation of {mention(d1, kind)}",
            positive(truth, kind),
        )

    elif scenario == "causal_single_tool":
        # Explicit causal evidence must sometimes beat neutral recurrence.
        outputs = (
            neutral(d1, kind),
            neutral(d1, kind),
            positive(truth, kind),
        )

    elif scenario == "multiple_bridges":
        # Multiple candidates recur across tools. Local causal evidence
        # should break the tie.
        outputs = (
            "\n".join(
                [
                    neutral(truth, kind),
                    neutral(d1, kind),
                    neutral(d2, kind),
                ]
            ),
            "\n".join(
                [
                    affected(truth, kind),
                    neutral(d1, kind),
                    neutral(d2, kind),
                ]
            ),
        )

    elif scenario == "late_corroboration":
        # Both candidates have equal distinct-tool recurrence, but the true
        # candidate is corroborated later in the trajectory.
        outputs = (
            neutral(d1, kind),
            "\n".join(
                [
                    neutral(d1, kind),
                    neutral(truth, kind),
                ]
            ),
            neutral(truth, kind),
        )

    elif scenario == "noisy_repetition":
        # Raw repetition inside one tool should not beat a bridge spread
        # across distinct tools.
        outputs = (
            "\n".join(
                [
                    neutral(d1, kind),
                    neutral(d1, kind),
                    neutral(d1, kind),
                    neutral(truth, kind),
                ]
            ),
            neutral(truth, kind),
        )

    elif scenario == ABSTAIN_SCENARIO:
        # Recurring but explicitly speculative evidence. There is no known
        # bridge to inject; a sufficiently confident system should abstain.
        outputs = (
            "\n".join(
                [
                    speculative(d1, kind),
                    speculative(d2, kind),
                ]
            ),
            "\n".join(
                [
                    hypothesis(d1, kind),
                    speculative(d2, kind),
                ]
            ),
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

    scenarios = (
        *POSITIVE_SCENARIOS,
        ABSTAIN_SCENARIO,
    )

    for scenario in scenarios:
        for kind in KINDS:
            for _ in range(N_PER_CATEGORY):
                base = rng.randint(1, 6000)

                cases.append(
                    make_case(
                        scenario,
                        kind,
                        base,
                    )
                )

    rng.shuffle(cases)

    return cases


def extract_raw_candidates(
    case: Case,
) -> list[RawCandidate]:
    # Use the actual production extractor and eligibility logic.
    #
    # Large top_k ensures calibration does not lose eligible candidates
    # merely because of the current development weights.
    ranked = rank_bridge_candidates(
        list(case.tool_outputs),
        top_k=200,
    )

    raw: list[RawCandidate] = []

    for candidate in ranked:
        causal_signal = (
            candidate.positive_evidence / 10.0
        )

        cross_steps = min(
            max(candidate.distinct_tools - 1, 0),
            2,
        )

        cross_signal = cross_steps / 2.0

        occurrence_signal = (
            min(candidate.occurrences, 3) / 3.0
        )

        speculation_signal = (
            candidate.negative_evidence / 5.0
        )

        raw.append(
            RawCandidate(
                token=candidate.token,
                causal=causal_signal,
                cross_tool=cross_signal,
                occurrence=occurrence_signal,
                recency=candidate.recency,
                speculation=speculation_signal,
            )
        )

    return raw


def score_candidate(
    candidate: RawCandidate,
    cfg: WeightConfig,
) -> float:
    return (
        cfg.causal * candidate.causal
        + cfg.cross_tool * candidate.cross_tool
        + cfg.occurrence * candidate.occurrence
        + cfg.recency * candidate.recency
        - cfg.speculation * candidate.speculation
    )


def evaluate(
    cases: list[Case],
    raw_cases: list[list[RawCandidate]],
    cfg: WeightConfig,
) -> dict:
    per_scenario_correct: dict[str, list[int]] = defaultdict(list)

    reciprocal_ranks: list[float] = []

    positive_correct = 0
    positive_total = 0

    abstain_correct = 0
    abstain_total = 0

    for case, raw in zip(cases, raw_cases, strict=True):
        scored = [
            (
                score_candidate(candidate, cfg),
                candidate.token,
            )
            for candidate in raw
        ]

        scored = [
            item
            for item in scored
            if item[0] >= cfg.threshold
        ]

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        tokens = [
            token
            for _, token in scored
        ]

        if case.truth is None:
            correct = int(len(tokens) == 0)

            abstain_total += 1
            abstain_correct += correct

            per_scenario_correct[
                case.scenario
            ].append(correct)

            continue

        positive_total += 1

        top1 = int(
            bool(tokens)
            and tokens[0] == case.truth
        )

        positive_correct += top1

        per_scenario_correct[
            case.scenario
        ].append(top1)

        if case.truth in tokens:
            rank = tokens.index(case.truth) + 1
            reciprocal_ranks.append(
                1.0 / rank
            )
        else:
            reciprocal_ranks.append(0.0)

    scenario_accuracy = {
        scenario: (
            sum(values) / len(values)
        )
        for scenario, values
        in per_scenario_correct.items()
    }

    macro_accuracy = (
        sum(scenario_accuracy.values())
        / len(scenario_accuracy)
    )

    positive_accuracy = (
        positive_correct / positive_total
    )

    abstain_accuracy = (
        abstain_correct / abstain_total
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    return {
        "macro_accuracy": macro_accuracy,
        "positive_accuracy": positive_accuracy,
        "mrr": mrr,
        "abstain_accuracy": abstain_accuracy,
        "false_promotion_rate": (
            1.0 - abstain_accuracy
        ),
        "scenario_accuracy": scenario_accuracy,
    }


def config_grid():
    causal_values = (6.0, 8.0, 10.0, 12.0)
    cross_values = (4.0, 6.0, 8.0, 10.0, 12.0)
    occurrence_values = (0.0, 1.0, 2.0)
    recency_values = (0.0, 1.0, 2.0, 3.0)
    speculation_values = (2.0, 4.0, 6.0, 8.0)

    thresholds = (
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
    )

    for (
        causal,
        cross,
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
        # Interpretable structural constraint:
        # explicit causal evidence should never receive less maximum
        # weight than cross-tool recurrence.
        if causal < cross:
            continue

        for threshold in thresholds:
            yield WeightConfig(
                causal=causal,
                cross_tool=cross,
                occurrence=occurrence,
                recency=recency,
                speculation=speculation,
                threshold=threshold,
            )


def ranking_key(row: dict):
    cfg: WeightConfig = row["config"]
    metrics = row["metrics"]

    # Primary: scenario-balanced correctness.
    # Secondary: positive ranking quality.
    # Tertiary: avoid speculative false promotions.
    # Final tie-break: prefer smaller/simpler total weight.
    total_weight = (
        cfg.causal
        + cfg.cross_tool
        + cfg.occurrence
        + cfg.recency
        + cfg.speculation
    )

    return (
        metrics["macro_accuracy"],
        metrics["mrr"],
        metrics["abstain_accuracy"],
        -metrics["false_promotion_rate"],
        -total_weight,
    )


def main():
    cases = build_cases()

    print(
        f"Calibration cases: {len(cases)}"
    )

    raw_cases = [
        extract_raw_candidates(case)
        for case in cases
    ]

    # Sanity check: all positive truths should be reachable by the
    # production candidate extractor before weight calibration.
    missing_truth = 0

    for case, raw in zip(
        cases,
        raw_cases,
        strict=True,
    ):
        if case.truth is None:
            continue

        tokens = {
            candidate.token
            for candidate in raw
        }

        if case.truth not in tokens:
            missing_truth += 1

    print(
        f"Positive truths missing from candidate set: "
        f"{missing_truth}"
    )

    if missing_truth:
        raise SystemExit(
            "Calibration invalid: production candidate extraction "
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

    best = results[0]

    # Evaluate the current development defaults separately.
    current_defaults = WeightConfig(
        causal=10.0,
        cross_tool=12.0,
        occurrence=3.0,
        recency=2.0,
        speculation=5.0,
        # Existing production default has no explicit score threshold.
        # A very low value approximates that behavior after eligibility.
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
        / "bridge_weight_calibration_top20.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
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
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank, row in enumerate(
            results[:20],
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

    best_cfg: WeightConfig = best["config"]
    best_metrics = best["metrics"]

    summary = {
        "seed": SEED,
        "n_cases": len(cases),
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
        "top20_csv": str(csv_path),
    }

    json_path = (
        out_dir
        / "bridge_weight_calibration_summary.json"
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
