"""Held-out synthetic evaluation for frozen trajectory-bridge scoring.

IMPORTANT:
- The scoring weights were frozen BEFORE this benchmark was run.
- Do not retune weights or thresholds using these results.
- This benchmark uses new seeds, templates, evidence orders, and scenario
  compositions that were not used by the calibration scripts.
- Real coding-agent evaluation remains the primary practical benchmark.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from headroom.trajectory_relevance import (
    BridgeScoringConfig,
    rank_bridge_candidates,
)

SEED = 771903
N_PER_KIND = 20

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
    "two_hop_join",
    "diagnostic_fork",
    "stale_then_confirmed",
    "late_single_cause",
    "bridge_with_noise",
    "competing_bridges",
    "three_tool_chain",
    "mixed_evidence",
)

NEGATIVE_SCENARIOS = (
    "speculative_loop",
    "isolated_observations",
    "ambiguous_competitors",
)


@dataclass(frozen=True)
class Case:
    scenario: str
    kind: str
    outputs: tuple[str, ...]
    truth: str | None


def token_for(kind: str, number: int) -> str:
    if kind == "request_id":
        return f"req-{number:04d}"

    if kind == "test_name":
        return f"test_payment_recovery_{number:04d}"

    if kind == "exception":
        return f"PaymentRecovery{number:04d}Error"

    if kind == "file_path":
        return f"services/payment/recovery_{number:04d}.py"

    if kind == "config_key":
        return f"PAYMENT_RECOVERY_{number:04d}"

    if kind == "function_name":
        return f"recover_payment_state_{number:04d}"

    if kind == "class_name":
        return f"PaymentRecoveryNode{number:04d}"

    raise ValueError(kind)


def mention(token: str, kind: str) -> str:
    if kind == "function_name":
        return f"{token}()"

    return token


def choose_tokens(
    kind: str,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    values = rng.sample(
        range(100, 9800),
        4,
    )

    return tuple(
        token_for(kind, value)
        for value in values
    )


def neutral(
    token: str,
    kind: str,
    rng: random.Random,
) -> str:
    value = mention(token, kind)

    templates = (
        "Inspection recorded {value}",
        "Trace output includes {value}",
        "Reference observed: {value}",
        "Tool result mentions {value}",
        "Diagnostic record contains {value}",
        "Found during repository inspection: {value}",
    )

    return rng.choice(templates).format(
        value=value
    )


def causal(
    token: str,
    kind: str,
    rng: random.Random,
) -> str:
    value = mention(token, kind)

    templates = (
        "Root cause identified at {value}",
        "Failure identified around {value}",
        "Affected component confirmed: {value}",
        "Confirmed failure evidence: {value}",
    )

    return rng.choice(templates).format(
        value=value
    )


def weak_causal(
    token: str,
    kind: str,
    rng: random.Random,
) -> str:
    value = mention(token, kind)

    templates = (
        "Expected value references {value}",
        "Actual diagnostic value contains {value}",
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
        "Possible explanation: {value}",
        "Working hypothesis: {value}",
        "Current suspicion concerns {value}",
        "Maybe related to {value}",
        "Candidate under investigation: {value}",
    )

    return rng.choice(templates).format(
        value=value
    )


def build_case(
    scenario: str,
    kind: str,
    rng: random.Random,
) -> Case:
    truth, wrong, other, extra = choose_tokens(
        kind,
        rng,
    )

    if scenario == "two_hop_join":
        # Truth appears neutrally in two distinct tool outputs separated
        # by unrelated noise.
        outputs = (
            neutral(truth, kind, rng),
            "\n".join(
                (
                    neutral(other, kind, rng),
                    neutral(extra, kind, rng),
                )
            ),
            neutral(truth, kind, rng),
        )

    elif scenario == "diagnostic_fork":
        # Two paths are explored. Wrong is repeated locally in one tool,
        # while truth connects two independent tool results.
        outputs = (
            "\n".join(
                (
                    neutral(wrong, kind, rng),
                    neutral(wrong, kind, rng),
                    neutral(truth, kind, rng),
                )
            ),
            "\n".join(
                (
                    neutral(other, kind, rng),
                    neutral(truth, kind, rng),
                )
            ),
        )

    elif scenario == "stale_then_confirmed":
        # An early hypothesis recurs, then explicit later evidence changes
        # the diagnosis.
        outputs = (
            speculative(wrong, kind, rng),
            speculative(wrong, kind, rng),
            neutral(other, kind, rng),
            causal(truth, kind, rng),
        )

    elif scenario == "late_single_cause":
        # Several neutral recurring identifiers exist, but a late
        # single-tool causal observation identifies the real target.
        outputs = (
            neutral(wrong, kind, rng),
            neutral(wrong, kind, rng),
            neutral(other, kind, rng),
            causal(truth, kind, rng),
        )

    elif scenario == "bridge_with_noise":
        # Many unrelated identifiers surround a legitimate bridge.
        outputs = (
            "\n".join(
                (
                    neutral(other, kind, rng),
                    neutral(truth, kind, rng),
                    neutral(extra, kind, rng),
                )
            ),
            "\n".join(
                (
                    neutral(wrong, kind, rng),
                    neutral(extra, kind, rng),
                )
            ),
            neutral(truth, kind, rng),
        )

    elif scenario == "competing_bridges":
        # Both truth and wrong span multiple tools; stronger local causal
        # evidence should resolve the competition.
        outputs = (
            "\n".join(
                (
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
            causal(truth, kind, rng),
        )

    elif scenario == "three_tool_chain":
        # Truth is independently observed across three tool results.
        # Distractor repetition is concentrated earlier.
        outputs = (
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    neutral(wrong, kind, rng),
                    neutral(wrong, kind, rng),
                )
            ),
            neutral(truth, kind, rng),
            "\n".join(
                (
                    neutral(other, kind, rng),
                    neutral(truth, kind, rng),
                )
            ),
        )

    elif scenario == "mixed_evidence":
        # Truth receives weak causal evidence plus cross-tool support;
        # wrong receives speculative recurrence.
        outputs = (
            "\n".join(
                (
                    neutral(truth, kind, rng),
                    speculative(wrong, kind, rng),
                )
            ),
            "\n".join(
                (
                    weak_causal(truth, kind, rng),
                    speculative(wrong, kind, rng),
                )
            ),
        )

    elif scenario == "speculative_loop":
        # No confirmed bridge exists. Recurrence alone should not force a
        # speculative hypothesis into relevance context.
        outputs = (
            speculative(wrong, kind, rng),
            "\n".join(
                (
                    speculative(wrong, kind, rng),
                    speculative(other, kind, rng),
                )
            ),
            speculative(other, kind, rng),
        )

        return Case(
            scenario=scenario,
            kind=kind,
            outputs=outputs,
            truth=None,
        )

    elif scenario == "isolated_observations":
        # Several unrelated single-tool observations but no bridge.
        outputs = (
            neutral(wrong, kind, rng),
            neutral(other, kind, rng),
            neutral(extra, kind, rng),
        )

        return Case(
            scenario=scenario,
            kind=kind,
            outputs=outputs,
            truth=None,
        )

    elif scenario == "ambiguous_competitors":
        # Multiple weak speculative candidates recur similarly. There is
        # no justified winner, so abstention is preferable.
        outputs = (
            "\n".join(
                (
                    speculative(wrong, kind, rng),
                    speculative(other, kind, rng),
                )
            ),
            "\n".join(
                (
                    speculative(other, kind, rng),
                    speculative(wrong, kind, rng),
                )
            ),
        )

        return Case(
            scenario=scenario,
            kind=kind,
            outputs=outputs,
            truth=None,
        )

    else:
        raise ValueError(scenario)

    return Case(
        scenario=scenario,
        kind=kind,
        outputs=outputs,
        truth=truth,
    )


def build_cases() -> list[Case]:
    rng = random.Random(SEED)

    cases = []

    for scenario in (
        *POSITIVE_SCENARIOS,
        *NEGATIVE_SCENARIOS,
    ):
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


def frozen_config() -> BridgeScoringConfig:
    return BridgeScoringConfig(
        causal_weight=4.0,
        cross_tool_weight=2.0,
        occurrence_weight=2.0,
        recency_weight=1.0,
        speculation_weight=2.0,
        min_score=2.0,
    )


def legacy_development_config() -> BridgeScoringConfig:
    # Pre-calibration V3 defaults, fixed before this held-out run.
    return BridgeScoringConfig(
        causal_weight=10.0,
        cross_tool_weight=12.0,
        occurrence_weight=3.0,
        recency_weight=2.0,
        speculation_weight=5.0,
        min_score=None,
    )


def evaluate_config(
    cases: list[Case],
    scoring: BridgeScoringConfig,
) -> tuple[dict, list[dict]]:
    scenario_correct = defaultdict(list)
    kind_correct = defaultdict(list)

    positive_total = 0
    positive_top1 = 0
    positive_top3 = 0
    reciprocal_rank_sum = 0.0

    negative_total = 0
    abstain_correct = 0

    rows = []

    for index, case in enumerate(cases):
        ranked = rank_bridge_candidates(
            list(case.outputs),
            top_k=6,
            scoring=scoring,
        )

        tokens = [
            candidate.token
            for candidate in ranked
        ]

        if case.truth is None:
            negative_total += 1

            correct = int(
                len(tokens) == 0
            )

            abstain_correct += correct

            scenario_correct[
                case.scenario
            ].append(correct)

            kind_correct[
                case.kind
            ].append(correct)

            rows.append(
                {
                    "case_index": index,
                    "scenario": case.scenario,
                    "kind": case.kind,
                    "truth": "",
                    "top1": (
                        tokens[0]
                        if tokens
                        else ""
                    ),
                    "top3": "|".join(
                        tokens[:3]
                    ),
                    "correct": correct,
                    "rank": "",
                }
            )

            continue

        positive_total += 1

        top1 = int(
            bool(tokens)
            and tokens[0] == case.truth
        )

        top3 = int(
            case.truth in tokens[:3]
        )

        positive_top1 += top1
        positive_top3 += top3

        if case.truth in tokens:
            rank = (
                tokens.index(case.truth)
                + 1
            )
            reciprocal_rank_sum += (
                1.0 / rank
            )
        else:
            rank = None

        scenario_correct[
            case.scenario
        ].append(top1)

        kind_correct[
            case.kind
        ].append(top1)

        rows.append(
            {
                "case_index": index,
                "scenario": case.scenario,
                "kind": case.kind,
                "truth": case.truth,
                "top1": (
                    tokens[0]
                    if tokens
                    else ""
                ),
                "top3": "|".join(
                    tokens[:3]
                ),
                "correct": top1,
                "rank": (
                    rank
                    if rank is not None
                    else ""
                ),
            }
        )

    scenario_accuracy = {
        key: (
            sum(values)
            / len(values)
        )
        for key, values
        in sorted(
            scenario_correct.items()
        )
    }

    kind_accuracy = {
        key: (
            sum(values)
            / len(values)
        )
        for key, values
        in sorted(
            kind_correct.items()
        )
    }

    macro_scenario_accuracy = (
        sum(
            scenario_accuracy.values()
        )
        / len(scenario_accuracy)
    )

    metrics = {
        "n_cases": len(cases),
        "positive_cases": positive_total,
        "negative_cases": negative_total,
        "positive_top1_accuracy": (
            positive_top1
            / positive_total
        ),
        "positive_top3_accuracy": (
            positive_top3
            / positive_total
        ),
        "mrr": (
            reciprocal_rank_sum
            / positive_total
        ),
        "abstain_accuracy": (
            abstain_correct
            / negative_total
        ),
        "false_promotion_rate": (
            1.0
            - abstain_correct
            / negative_total
        ),
        "macro_scenario_accuracy": (
            macro_scenario_accuracy
        ),
        "scenario_accuracy": (
            scenario_accuracy
        ),
        "kind_accuracy": kind_accuracy,
    }

    return metrics, rows


def write_rows(
    path: Path,
    rows: list[dict],
) -> None:
    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cases = build_cases()

    print(
        f"Held-out cases: {len(cases)}"
    )

    legacy_metrics, legacy_rows = (
        evaluate_config(
            cases,
            legacy_development_config(),
        )
    )

    frozen_metrics, frozen_rows = (
        evaluate_config(
            cases,
            frozen_config(),
        )
    )

    out_dir = Path(
        "experiments/bridge_relevance/results"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    legacy_csv = (
        out_dir
        / "held_out_legacy_development.csv"
    )

    frozen_csv = (
        out_dir
        / "held_out_frozen_scoring.csv"
    )

    write_rows(
        legacy_csv,
        legacy_rows,
    )

    write_rows(
        frozen_csv,
        frozen_rows,
    )

    summary = {
        "evaluation_status": (
            "held_out_after_weight_freeze"
        ),
        "seed": SEED,
        "n_cases": len(cases),
        "legacy_development_scoring": {
            "config": {
                "causal": 10.0,
                "cross_tool": 12.0,
                "occurrence": 3.0,
                "recency": 2.0,
                "speculation": 5.0,
                "threshold": None,
            },
            "metrics": legacy_metrics,
        },
        "frozen_calibrated_scoring": {
            "config": {
                "causal": 4.0,
                "cross_tool": 2.0,
                "occurrence": 2.0,
                "recency": 1.0,
                "speculation": 2.0,
                "threshold": 2.0,
            },
            "metrics": frozen_metrics,
        },
        "raw_results": {
            "legacy": str(
                legacy_csv
            ),
            "frozen": str(
                frozen_csv
            ),
        },
        "warning": (
            "Do not retune scoring from these held-out results."
        ),
    }

    summary_path = (
        out_dir
        / "held_out_bridge_summary.json"
    )

    summary_path.write_text(
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
