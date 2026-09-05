import csv
import random
from pathlib import Path

from headroom.transforms import (
    SearchCompressor,
    SearchCompressorConfig,
)

from automatic_bridge_probe import (
    extract_bridge_candidates,
)

from bridge_extractor_v2 import (
    extract_all_identifiers_from_tools,
    extract_bridge_candidates_v2,
)


N_LINES = 500
N_CASES = 100
BUDGETS = [10, 20, 30]

DISTRACTORS_PER_TOOL = 10
TOP_K = 6
SEED = 2027

USER_QUERY = (
    "Investigate why the application "
    "sometimes behaves incorrectly."
)

OUTPUT_PATH = Path(
    "experiments/bridge_relevance/"
    "distinct_tool_results.csv"
)


CATEGORIES = [
    "request_id",
    "test_name",
    "exception",
    "file_path",
    "config_key",
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


def make_search_output(category):
    return "\n".join(
        (
            f"src/runtime.py:{100 + i}:"
            f"evidence = '{make_token(category, i)}'"
        )
        for i in range(1, N_LINES + 1)
    )


def make_tool_outputs(category, needle_position, rng):
    """
    Critical control:

    True bridge:
        appears once in Tool A
        appears once in Tool B
        total occurrences = 2
        distinct tools = 2

    Every distractor:
        appears twice, but only inside ONE tool
        total occurrences = 2
        distinct tools = 1

    Therefore ordinary occurrence count alone
    cannot identify the true bridge.
    """

    needle = make_token(
        category,
        needle_position,
    )

    available = [
        i
        for i in range(50, 451)
        if i != needle_position
    ]

    positions = rng.sample(
        available,
        DISTRACTORS_PER_TOOL * 2,
    )

    a_positions = positions[:DISTRACTORS_PER_TOOL]
    b_positions = positions[DISTRACTORS_PER_TOOL:]

    tool_a_lines = []
    tool_b_lines = []

    for pos in a_positions:
        token = make_token(category, pos)

        # Same-tool repetition.
        tool_a_lines.append(
            f"Observed candidate {token}"
        )
        tool_a_lines.append(
            f"Follow-up observation {token}"
        )

    for pos in b_positions:
        token = make_token(category, pos)

        # Same-tool repetition.
        tool_b_lines.append(
            f"Observed candidate {token}"
        )
        tool_b_lines.append(
            f"Follow-up observation {token}"
        )

    # True bridge: only one appearance in each tool.
    insert_a = rng.randrange(
        len(tool_a_lines) + 1
    )
    insert_b = rng.randrange(
        len(tool_b_lines) + 1
    )

    tool_a_lines.insert(
        insert_a,
        f"Current observation {needle}",
    )

    tool_b_lines.insert(
        insert_b,
        f"Current observation {needle}",
    )

    return (
        "\n".join(tool_a_lines),
        "\n".join(tool_b_lines),
    )


def compress_search(
    search_output,
    context,
    budget,
):
    config = SearchCompressorConfig(
        max_total_matches=budget,
        max_matches_per_file=budget,
        max_files=1,
        boost_errors=False,
    )

    compressor = SearchCompressor(config)

    result = compressor.compress(
        search_output,
        context=context,
    )

    return result.compressed


def make_context(
    condition,
    tool_outputs,
    needle,
):
    if condition == "user_only":
        additions = []

    elif condition == "naive":
        additions = (
            extract_all_identifiers_from_tools(
                tool_outputs
            )
        )

    elif condition == "v1":
        combined = "\n".join(tool_outputs)

        additions = extract_bridge_candidates(
            combined,
            top_k=TOP_K,
        )

    elif condition == "v2":
        additions = (
            extract_bridge_candidates_v2(
                tool_outputs,
                top_k=TOP_K,
            )
        )

    elif condition == "oracle":
        additions = [needle]

    else:
        raise ValueError(condition)

    context = USER_QUERY

    if additions:
        context += " " + " ".join(additions)

    return context, additions


def main():
    rng = random.Random(SEED)

    positions = rng.sample(
        list(range(50, 451)),
        N_CASES,
    )

    conditions = [
        "user_only",
        "naive",
        "v1",
        "v2",
        "oracle",
    ]

    rows = []

    for category in CATEGORIES:
        search_output = make_search_output(
            category
        )

        print()
        print("#" * 78)
        print("CATEGORY:", category)
        print("#" * 78)

        for budget in BUDGETS:
            stats = {
                condition: {
                    "hits": 0,
                    "selected": 0,
                    "candidate_count": 0,
                }
                for condition in conditions
            }

            for position in positions:
                needle = make_token(
                    category,
                    position,
                )

                tool_a, tool_b = (
                    make_tool_outputs(
                        category,
                        position,
                        rng,
                    )
                )

                tool_outputs = [
                    tool_a,
                    tool_b,
                ]

                for condition in conditions:
                    context, additions = (
                        make_context(
                            condition,
                            tool_outputs,
                            needle,
                        )
                    )

                    compressed = (
                        compress_search(
                            search_output,
                            context,
                            budget,
                        )
                    )

                    preserved = (
                        needle in compressed
                    )

                    selected = (
                        needle in additions
                    )

                    stats[
                        condition
                    ]["hits"] += int(
                        preserved
                    )

                    stats[
                        condition
                    ]["selected"] += int(
                        selected
                    )

                    stats[
                        condition
                    ][
                        "candidate_count"
                    ] += len(additions)

                    rows.append(
                        {
                            "category": category,
                            "budget": budget,
                            "position": position,
                            "needle": needle,
                            "condition": condition,
                            "preserved": preserved,
                            "needle_selected": selected,
                            "candidate_count": (
                                len(additions)
                            ),
                            "selected_candidates": (
                                " ".join(additions)
                            ),
                        }
                    )

            print()
            print(
                f"BUDGET {budget}/{N_LINES}"
            )

            for condition in conditions:
                s = stats[condition]

                recall = (
                    s["hits"] / N_CASES
                )

                selected_rate = (
                    s["selected"] / N_CASES
                )

                avg_candidates = (
                    s["candidate_count"]
                    / N_CASES
                )

                print(
                    f"{condition:10s} "
                    f"RECALL={recall:6.1%}  "
                    f"SELECTED="
                    f"{selected_rate:6.1%}  "
                    f"AVG_CAND="
                    f"{avg_candidates:5.1f}"
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
                "category",
                "budget",
                "position",
                "needle",
                "condition",
                "preserved",
                "needle_selected",
                "candidate_count",
                "selected_candidates",
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
