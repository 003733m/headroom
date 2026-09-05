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


N_CASES = 100
RANK_K = 50
SEED = 3030

OUTPUT_PATH = Path(
    "experiments/bridge_relevance/"
    "adversarial_ranking_results.csv"
)


# Only types that BOTH V1 and V2 can extract.
# Function/class coverage was already tested separately.
CATEGORIES = [
    "request_id",
    "test_name",
    "exception",
    "file_path",
    "config_key",
]

SCENARIOS = [
    "true_bridge",
    "false_recurring",
    "multiple_bridges",
    "hypothesis_shift",
    "late_corroboration",
]


def make_token(category, i):
    if category == "request_id":
        return f"req-{i:04d}"

    if category == "test_name":
        return f"test_case_{i:04d}"

    if category == "exception":
        return f"Service{i:04d}Error"

    if category == "file_path":
        return f"src/module_{i:04d}.py"

    if category == "config_key":
        return f"CONFIG_KEY_{i:04d}"

    raise ValueError(category)


def sample_other_positions(
    needle_position,
    n,
    rng,
):
    available = [
        i
        for i in range(50, 451)
        if i != needle_position
    ]

    return rng.sample(
        available,
        n,
    )


def shuffled(lines, rng):
    lines = list(lines)
    rng.shuffle(lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Scenario A
# ---------------------------------------------------------------------

def make_true_bridge(
    category,
    needle_position,
    rng,
):
    """
    TRUE bridge:
        needle appears once in Tool A
        needle appears once in Tool B

    Distractors:
        repeated twice, but only inside ONE tool.

    Thus:
        needle occurrences = 2
        distractor occurrences = 2

    Only distinct-tool recurrence separates them.
    """

    needle = make_token(
        category,
        needle_position,
    )

    positions = sample_other_positions(
        needle_position,
        16,
        rng,
    )

    a_positions = positions[:8]
    b_positions = positions[8:]

    tool_a = []
    tool_b = []

    for pos in a_positions:
        token = make_token(category, pos)

        tool_a.extend([
            f"Observed candidate {token}",
            f"Follow-up observation {token}",
        ])

    for pos in b_positions:
        token = make_token(category, pos)

        tool_b.extend([
            f"Observed candidate {token}",
            f"Follow-up observation {token}",
        ])

    tool_a.append(
        f"Current observation {needle}"
    )

    tool_b.append(
        f"Current observation {needle}"
    )

    return [
        shuffled(tool_a, rng),
        shuffled(tool_b, rng),
    ], needle


# ---------------------------------------------------------------------
# Scenario B
# ---------------------------------------------------------------------

def make_false_recurring(
    category,
    needle_position,
    rng,
):
    """
    FALSE recurring hypothesis:
        a wrong candidate repeats across tools.

    TRUE needle:
        appears once,
        but has explicit root-cause evidence.

    Tests whether V2 blindly trusts recurrence.
    """

    needle = make_token(
        category,
        needle_position,
    )

    positions = sample_other_positions(
        needle_position,
        10,
        rng,
    )

    false_bridge = make_token(
        category,
        positions[0],
    )

    noise = [
        make_token(category, p)
        for p in positions[1:]
    ]

    tool_a = [
        f"Initial suspicion {false_bridge}",
        f"Root cause identified: {needle}",
    ]

    tool_b = [
        f"Follow-up suspicion {false_bridge}",
    ]

    for token in noise[:4]:
        tool_a.append(
            f"Routine observation {token}"
        )

    for token in noise[4:]:
        tool_b.append(
            f"Routine observation {token}"
        )

    return [
        shuffled(tool_a, rng),
        shuffled(tool_b, rng),
    ], needle


# ---------------------------------------------------------------------
# Scenario C
# ---------------------------------------------------------------------

def make_multiple_bridges(
    category,
    needle_position,
    rng,
):
    """
    Three identifiers all repeat across Tool A and Tool B.

    Only the true needle has explicit causal evidence.

    Tests whether contextual evidence breaks
    cross-tool recurrence ties.
    """

    needle = make_token(
        category,
        needle_position,
    )

    positions = sample_other_positions(
        needle_position,
        8,
        rng,
    )

    false_a = make_token(
        category,
        positions[0],
    )

    false_b = make_token(
        category,
        positions[1],
    )

    noise = [
        make_token(category, p)
        for p in positions[2:]
    ]

    tool_a = [
        f"Observed {false_a}",
        f"Observed {false_b}",
        f"Observed {needle}",
    ]

    tool_b = [
        f"Observed {false_a}",
        f"Observed {false_b}",
        f"Affected component: {needle}",
    ]

    for token in noise[:3]:
        tool_a.append(
            f"Routine observation {token}"
        )

    for token in noise[3:]:
        tool_b.append(
            f"Routine observation {token}"
        )

    return [
        shuffled(tool_a, rng),
        shuffled(tool_b, rng),
    ], needle


# ---------------------------------------------------------------------
# Scenario D
# ---------------------------------------------------------------------

def make_hypothesis_shift(
    category,
    needle_position,
    rng,
):
    """
    Early tools repeat a WRONG hypothesis.

    A later tool discovers the actual root cause.

    Tests confirmation-bias / stale-hypothesis risk.
    """

    needle = make_token(
        category,
        needle_position,
    )

    positions = sample_other_positions(
        needle_position,
        10,
        rng,
    )

    stale_hypothesis = make_token(
        category,
        positions[0],
    )

    noise = [
        make_token(category, p)
        for p in positions[1:]
    ]

    tool_a = [
        f"Initial suspicion {stale_hypothesis}",
    ]

    tool_b = [
        f"Follow-up investigation {stale_hypothesis}",
    ]

    tool_c = [
        f"Root cause identified: {needle}",
    ]

    for token in noise[:3]:
        tool_a.append(
            f"Routine observation {token}"
        )

    for token in noise[3:6]:
        tool_b.append(
            f"Routine observation {token}"
        )

    for token in noise[6:]:
        tool_c.append(
            f"Routine observation {token}"
        )

    return [
        shuffled(tool_a, rng),
        shuffled(tool_b, rng),
        shuffled(tool_c, rng),
    ], needle


# ---------------------------------------------------------------------
# Scenario E
# ---------------------------------------------------------------------

def make_late_corroboration(
    category,
    needle_position,
    rng,
):
    """
    Both candidates have equal cross-tool recurrence.

    Wrong candidate:
        Tool A + Tool B

    True candidate:
        Tool B + Tool C

    No strong error/root-cause cue.

    The true candidate is simply the more recent
    cross-tool bridge.

    Tests V2's small recency signal.
    """

    needle = make_token(
        category,
        needle_position,
    )

    positions = sample_other_positions(
        needle_position,
        10,
        rng,
    )

    old_bridge = make_token(
        category,
        positions[0],
    )

    noise = [
        make_token(category, p)
        for p in positions[1:]
    ]

    tool_a = [
        f"Observed connection {old_bridge}",
    ]

    tool_b = [
        f"Observed connection {old_bridge}",
        f"Observed connection {needle}",
    ]

    tool_c = [
        f"Observed connection {needle}",
    ]

    for token in noise[:3]:
        tool_a.append(
            f"Routine observation {token}"
        )

    for token in noise[3:6]:
        tool_b.append(
            f"Routine observation {token}"
        )

    for token in noise[6:]:
        tool_c.append(
            f"Routine observation {token}"
        )

    return [
        shuffled(tool_a, rng),
        shuffled(tool_b, rng),
        shuffled(tool_c, rng),
    ], needle


SCENARIO_BUILDERS = {
    "true_bridge": make_true_bridge,
    "false_recurring": make_false_recurring,
    "multiple_bridges": make_multiple_bridges,
    "hypothesis_shift": make_hypothesis_shift,
    "late_corroboration": make_late_corroboration,
}


def rank_expected(ranking, expected):
    try:
        return ranking.index(expected) + 1
    except ValueError:
        return None


def evaluate_rank(rank):
    if rank is None:
        return {
            "top1": 0,
            "top3": 0,
            "rr": 0.0,
            "covered": 0,
        }

    return {
        "top1": int(rank == 1),
        "top3": int(rank <= 3),
        "rr": 1.0 / rank,
        "covered": 1,
    }


def main():
    rng = random.Random(SEED)

    extractors = [
        "v1",
        "v2",
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
                    "v1": extract_bridge_candidates(
                        combined,
                        top_k=RANK_K,
                    ),
                    "v2": (
                        extract_bridge_candidates_v2(
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
                    agg["top1"] += metrics[
                        "top1"
                    ]
                    agg["top3"] += metrics[
                        "top3"
                    ]
                    agg["rr"] += metrics[
                        "rr"
                    ]
                    agg["covered"] += metrics[
                        "covered"
                    ]

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
    print("=" * 82)
    print("ADVERSARIAL RANKING BENCHMARK")
    print("=" * 82)

    for scenario in SCENARIOS:
        print()
        print("#" * 82)
        print("SCENARIO:", scenario)
        print("#" * 82)

        for extractor in extractors:
            agg = aggregate[
                (scenario, extractor)
            ]

            n = agg["n"]

            top1 = agg["top1"] / n
            top3 = agg["top3"] / n
            mrr = agg["rr"] / n
            coverage = (
                agg["covered"] / n
            )

            if agg["rank_count"]:
                mean_rank = (
                    agg["rank_sum"]
                    / agg["rank_count"]
                )
            else:
                mean_rank = float("nan")

            print(
                f"{extractor:4s}  "
                f"N={n:4d}  "
                f"TOP1={top1:6.1%}  "
                f"TOP3={top3:6.1%}  "
                f"MRR={mrr:6.3f}  "
                f"COVERAGE={coverage:6.1%}  "
                f"MEAN_RANK={mean_rank:5.2f}"
            )

    print()
    print("=" * 82)
    print("OVERALL")
    print("=" * 82)

    for extractor in extractors:
        relevant = [
            aggregate[(scenario, extractor)]
            for scenario in SCENARIOS
        ]

        n = sum(x["n"] for x in relevant)

        top1 = (
            sum(x["top1"] for x in relevant)
            / n
        )

        top3 = (
            sum(x["top3"] for x in relevant)
            / n
        )

        mrr = (
            sum(x["rr"] for x in relevant)
            / n
        )

        coverage = (
            sum(x["covered"] for x in relevant)
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

