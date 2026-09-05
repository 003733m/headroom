import csv
import random
from collections import defaultdict
from pathlib import Path

from automatic_bridge_probe import (
    extract_bridge_candidates,
)

from bridge_extractor_v2 import (
    extract_bridge_candidates_v2,
)

from bridge_extractor_v3 import (
    extract_bridge_candidates_v3,
)

from adversarial_ranking_benchmark import (
    CATEGORIES,
    N_CASES,
    RANK_K,
    SCENARIOS,
    SCENARIO_BUILDERS,
    SEED,
    evaluate_rank,
    rank_expected,
)


OUTPUT_PATH = Path(
    "experiments/bridge_relevance/"
    "adversarial_ranking_v3_results.csv"
)


def main():
    rng = random.Random(SEED)

    extractors = [
        "v1",
        "v2",
        "v3",
    ]

    aggregate = defaultdict(
        lambda: {
            "n": 0,
            "top1": 0,
            "top3": 0,
            "rr": 0.0,
            "covered": 0,
            "rank_sum": 0,
            "rank_count": 0,
        }
    )

    rows = []

    for scenario in SCENARIOS:
        builder = SCENARIO_BUILDERS[
            scenario
        ]

        for category in CATEGORIES:
            needle_positions = rng.sample(
                list(range(50, 451)),
                N_CASES,
            )

            for case_id, position in enumerate(
                needle_positions
            ):
                tool_outputs, expected = builder(
                    category,
                    position,
                    rng,
                )

                combined = "\n".join(
                    tool_outputs
                )

                rankings = {
                    "v1": (
                        extract_bridge_candidates(
                            combined,
                            top_k=RANK_K,
                        )
                    ),
                    "v2": (
                        extract_bridge_candidates_v2(
                            tool_outputs,
                            top_k=RANK_K,
                        )
                    ),
                    "v3": (
                        extract_bridge_candidates_v3(
                            tool_outputs,
                            top_k=RANK_K,
                        )
                    ),
                }

                for extractor in extractors:
                    ranking = rankings[
                        extractor
                    ]

                    rank = rank_expected(
                        ranking,
                        expected,
                    )

                    metrics = evaluate_rank(
                        rank
                    )

                    key = (
                        scenario,
                        extractor,
                    )

                    agg = aggregate[key]

                    agg["n"] += 1

                    agg["top1"] += (
                        metrics["top1"]
                    )

                    agg["top3"] += (
                        metrics["top3"]
                    )

                    agg["rr"] += (
                        metrics["rr"]
                    )

                    agg["covered"] += (
                        metrics["covered"]
                    )

                    if rank is not None:
                        agg["rank_sum"] += rank
                        agg["rank_count"] += 1

                    rows.append(
                        {
                            "scenario": scenario,
                            "category": category,
                            "case_id": case_id,
                            "expected": expected,
                            "extractor": extractor,
                            "rank": (
                                rank
                                if rank is not None
                                else ""
                            ),
                            "top1": metrics["top1"],
                            "top3": metrics["top3"],
                            "reciprocal_rank": (
                                metrics["rr"]
                            ),
                            "covered": (
                                metrics["covered"]
                            ),
                            "ranking": " ".join(
                                ranking
                            ),
                        }
                    )

    print()
    print("=" * 84)
    print(
        "ADVERSARIAL RANKING — V1 vs V2 vs V3"
    )
    print("=" * 84)

    for scenario in SCENARIOS:
        print()
        print("#" * 84)
        print(
            "SCENARIO:",
            scenario,
        )
        print("#" * 84)

        for extractor in extractors:
            agg = aggregate[
                (scenario, extractor)
            ]

            n = agg["n"]

            top1 = (
                agg["top1"] / n
            )

            top3 = (
                agg["top3"] / n
            )

            mrr = (
                agg["rr"] / n
            )

            coverage = (
                agg["covered"] / n
            )

            mean_rank = (
                agg["rank_sum"]
                / agg["rank_count"]
                if agg["rank_count"]
                else float("nan")
            )

            print(
                f"{extractor:4s}  "
                f"N={n:4d}  "
                f"TOP1={top1:6.1%}  "
                f"TOP3={top3:6.1%}  "
                f"MRR={mrr:6.3f}  "
                f"COVERAGE="
                f"{coverage:6.1%}  "
                f"MEAN_RANK="
                f"{mean_rank:5.2f}"
            )

    print()
    print("=" * 84)
    print("OVERALL")
    print("=" * 84)

    for extractor in extractors:
        relevant = [
            aggregate[
                (scenario, extractor)
            ]
            for scenario in SCENARIOS
        ]

        n = sum(
            x["n"]
            for x in relevant
        )

        top1 = (
            sum(
                x["top1"]
                for x in relevant
            )
            / n
        )

        top3 = (
            sum(
                x["top3"]
                for x in relevant
            )
            / n
        )

        mrr = (
            sum(
                x["rr"]
                for x in relevant
            )
            / n
        )

        coverage = (
            sum(
                x["covered"]
                for x in relevant
            )
            / n
        )

        rank_count = sum(
            x["rank_count"]
            for x in relevant
        )

        rank_sum = sum(
            x["rank_sum"]
            for x in relevant
        )

        mean_rank = (
            rank_sum / rank_count
            if rank_count
            else float("nan")
        )

        print(
            f"{extractor:4s}  "
            f"N={n:4d}  "
            f"TOP1={top1:6.1%}  "
            f"TOP3={top3:6.1%}  "
            f"MRR={mrr:6.3f}  "
            f"COVERAGE={coverage:6.1%}  "
            f"MEAN_RANK={mean_rank:5.2f}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "category",
                "case_id",
                "expected",
                "extractor",
                "rank",
                "top1",
                "top3",
                "reciprocal_rank",
                "covered",
                "ranking",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "Raw results written to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
