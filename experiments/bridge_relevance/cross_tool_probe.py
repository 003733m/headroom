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
N_CASES = 50

BUDGETS = [
    10,
    30,
    100,
]

DISTRACTORS_PER_TOOL = 10

SEED = 2026

TOP_K = 6

USER_QUERY = (
    "Investigate why the application "
    "sometimes behaves incorrectly."
)

OUTPUT_PATH = Path(
    "experiments/bridge_relevance/"
    "cross_tool_results.csv"
)


CATEGORIES = [
    "request_id",
    "test_name",
    "exception",
    "file_path",
    "config_key",
    "function_name",
    "class_name",
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

    if category == "function_name":
        return f"process_item_{i:04d}"

    if category == "class_name":
        return f"WorkerClass{i:04d}"

    raise ValueError(category)


def make_search_output(category):
    """
    Tool C:
    a large search result containing 500 possible
    identifiers.

    The compressor must decide which ones survive.
    """

    lines = []

    for i in range(1, N_LINES + 1):
        token = make_token(
            category,
            i,
        )

        lines.append(
            f"src/runtime.py:{100 + i}:"
            f"evidence = '{token}'"
        )

    return "\n".join(lines)


def choose_distractors(
    needle_position,
    rng,
):
    available = [
        i
        for i in range(50, 451)
        if i != needle_position
    ]

    chosen = rng.sample(
        available,
        DISTRACTORS_PER_TOOL * 2,
    )

    return (
        chosen[:DISTRACTORS_PER_TOOL],
        chosen[DISTRACTORS_PER_TOOL:],
    )


def make_tool_outputs(
    category,
    needle_position,
    rng,
):
    """
    Tool A and Tool B both mention the TRUE bridge.

    Distractors are different across tools.

    Crucially, the true bridge is NOT placed next to
    ERROR/FAILED/root-cause wording.

    Meanwhile each tool contains a different highly
    salient distractor.

    Therefore:
      V1 may chase local salience.
      V2 can use cross-tool recurrence.
    """

    needle = make_token(
        category,
        needle_position,
    )

    positions_a, positions_b = (
        choose_distractors(
            needle_position,
            rng,
        )
    )

    distractors_a = [
        make_token(category, p)
        for p in positions_a
    ]

    distractors_b = [
        make_token(category, p)
        for p in positions_b
    ]

    tool_a = []

    for token in distractors_a[:5]:
        tool_a.append(
            f"Observed routine candidate {token}"
        )

    # Strong but FALSE local signal.
    tool_a.append(
        f"FAILED while inspecting "
        f"{distractors_a[5]}"
    )

    for token in distractors_a[6:]:
        tool_a.append(
            f"Routine observation {token}"
        )

    # True bridge, deliberately neutral.
    tool_a.append(
        f"Current inspection referenced {needle}"
    )

    tool_b = []

    for token in distractors_b[:5]:
        tool_b.append(
            f"Observed routine candidate {token}"
        )

    # Another strong but FALSE local signal.
    tool_b.append(
        f"Root cause candidate "
        f"{distractors_b[5]}"
    )

    for token in distractors_b[6:]:
        tool_b.append(
            f"Routine follow-up {token}"
        )

    # Same true bridge appears in another tool.
    tool_b.append(
        f"Follow-up inspection referenced {needle}"
    )

    return (
        "\n".join(tool_a),
        "\n".join(tool_b),
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

    compressor = SearchCompressor(
        config
    )

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
        # Old extractor was designed for a single
        # text blob, so concatenate trajectory.
        combined = "\n".join(
            tool_outputs
        )

        additions = (
            extract_bridge_candidates(
                combined,
                top_k=TOP_K,
            )
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
        context += (
            " "
            + " ".join(additions)
        )

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
        search_output = (
            make_search_output(category)
        )

        print()
        print("#" * 78)
        print(
            "CATEGORY:",
            category,
        )
        print("#" * 78)

        for budget in BUDGETS:
            stats = {
                condition: {
                    "hits": 0,
                    "selected": 0,
                    "candidate_count": 0,
                    "added_chars": 0,
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

                    added_chars = (
                        len(context)
                        - len(USER_QUERY)
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

                    stats[
                        condition
                    ][
                        "added_chars"
                    ] += added_chars

                    rows.append(
                        {
                            "category": category,
                            "budget": budget,
                            "position": position,
                            "needle": needle,
                            "condition": condition,
                            "preserved": (
                                preserved
                            ),
                            "needle_selected": (
                                selected
                            ),
                            "candidate_count": (
                                len(additions)
                            ),
                            "added_chars": (
                                added_chars
                            ),
                            "selected_candidates": (
                                " ".join(additions)
                            ),
                        }
                    )

            print()
            print(
                f"BUDGET "
                f"{budget}/{N_LINES}"
            )

            for condition in conditions:
                s = stats[condition]

                recall = (
                    s["hits"]
                    / N_CASES
                )

                selection_rate = (
                    s["selected"]
                    / N_CASES
                )

                avg_candidates = (
                    s["candidate_count"]
                    / N_CASES
                )

                avg_chars = (
                    s["added_chars"]
                    / N_CASES
                )

                print(
                    f"{condition:10s} "
                    f"RECALL={recall:6.1%}  "
                    f"SELECTED="
                    f"{selection_rate:6.1%}  "
                    f"AVG_CAND="
                    f"{avg_candidates:5.1f}  "
                    f"AVG_CHARS="
                    f"{avg_chars:7.1f}"
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
                "added_chars",
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
