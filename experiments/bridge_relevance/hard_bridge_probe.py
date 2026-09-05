import csv
import random
from pathlib import Path

from automatic_bridge_probe import (
    extract_all_identifiers,
    extract_bridge_candidates,
)

from headroom.transforms import (
    SearchCompressor,
    SearchCompressorConfig,
)


N_LINES = 500
N_CASES = 20
BUDGETS = [10, 30, 100]
N_DISTRACTORS = 15
SEED = 123

USER_QUERY = (
    "Investigate why the application sometimes behaves incorrectly."
)

OUTPUT_PATH = Path(
    "experiments/bridge_relevance/hard_bridge_results.csv"
)


# ---------------------------------------------------------------------
# Bridge token types
# ---------------------------------------------------------------------

def make_token(category, i):
    if category in ("request_explicit", "request_neutral"):
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


CATEGORIES = [
    "request_explicit",
    "request_neutral",
    "test_name",
    "exception",
    "file_path",
    "config_key",
    "function_name",
    "class_name",
]


# ---------------------------------------------------------------------
# Search output
# ---------------------------------------------------------------------

def make_search_output(category):
    lines = []

    for i in range(1, N_LINES + 1):
        token = make_token(category, i)

        lines.append(
            f"src/runtime.py:{100 + i}:evidence = '{token}'"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Previous tool output
# ---------------------------------------------------------------------

def make_previous_tool_output(category, position, rng):
    needle = make_token(category, position)

    available = [
        i
        for i in range(50, 451)
        if i != position
    ]

    distractor_positions = rng.sample(
        available,
        N_DISTRACTORS,
    )

    distractors = [
        make_token(category, p)
        for p in distractor_positions
    ]

    lines = []

    for token in distractors[:8]:
        lines.append(
            f"Observed routine value {token}"
        )

    # Different evidence styles.
    if category == "request_explicit":
        lines.append(
            f"Affected request: {needle}"
        )

    elif category == "request_neutral":
        # No ERROR / FAILED / affected cue.
        lines.append(
            f"Current trace includes {needle}"
        )

    elif category == "test_name":
        lines.append(
            f"FAILED {needle}"
        )

    elif category == "exception":
        lines.append(
            f"Exception raised: {needle}"
        )

    elif category == "file_path":
        lines.append(
            f"Root cause located in {needle}"
        )

    elif category == "config_key":
        lines.append(
            f"Root cause setting: {needle}"
        )

    elif category == "function_name":
        lines.append(
            f"Root cause function: {needle}"
        )

    elif category == "class_name":
        lines.append(
            f"Affected component: {needle}"
        )

    for token in distractors[8:]:
        lines.append(
            f"Routine cleanup observed {token}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------

def compress_search(search_output, context, budget):
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


def make_context(condition, previous_tool, needle):
    if condition == "user_only":
        additions = []

    elif condition == "naive":
        additions = extract_all_identifiers(
            previous_tool
        )

    elif condition == "bridge":
        additions = extract_bridge_candidates(
            previous_tool
        )

    elif condition == "oracle":
        additions = [needle]

    else:
        raise ValueError(condition)

    context = USER_QUERY

    if additions:
        context += " " + " ".join(additions)

    return context, additions


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def main():
    rng = random.Random(SEED)

    positions = rng.sample(
        list(range(50, 451)),
        N_CASES,
    )

    conditions = [
        "user_only",
        "naive",
        "bridge",
        "oracle",
    ]

    rows = []

    for category in CATEGORIES:
        search_output = make_search_output(category)

        print()
        print("#" * 76)
        print("CATEGORY:", category)
        print("#" * 76)

        for budget in BUDGETS:
            stats = {
                condition: {
                    "hits": 0,
                    "needle_selected": 0,
                    "candidate_count": 0,
                }
                for condition in conditions
            }

            for position in positions:
                needle = make_token(
                    category,
                    position,
                )

                previous_tool = (
                    make_previous_tool_output(
                        category,
                        position,
                        rng,
                    )
                )

                for condition in conditions:
                    context, additions = make_context(
                        condition,
                        previous_tool,
                        needle,
                    )

                    compressed = compress_search(
                        search_output,
                        context,
                        budget,
                    )

                    preserved = (
                        needle in compressed
                    )

                    needle_selected = (
                        needle in additions
                    )

                    stats[condition]["hits"] += int(
                        preserved
                    )

                    stats[condition][
                        "needle_selected"
                    ] += int(needle_selected)

                    stats[condition][
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
                            "needle_selected": needle_selected,
                            "candidate_count": len(additions),
                            "selected_candidates": " ".join(
                                additions
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

                selection_rate = (
                    s["needle_selected"]
                    / N_CASES
                )

                avg_candidates = (
                    s["candidate_count"]
                    / N_CASES
                )

                print(
                    f"{condition:10s} "
                    f"RECALL={recall:6.1%}  "
                    f"NEEDLE_SELECTED="
                    f"{selection_rate:6.1%}  "
                    f"AVG_CANDIDATES="
                    f"{avg_candidates:4.1f}"
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
        f"Raw results written to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
