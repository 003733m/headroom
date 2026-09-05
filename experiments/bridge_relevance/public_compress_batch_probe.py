import argparse
import json
import random

from headroom.compress import compress


N_LINES = 1200
N_CASES = 100
SEED = 5150


def make_needle(position):
    return f"req-{position:04d}"


def build_messages(position):
    needle = make_needle(position)

    search_lines = []

    for i in range(1, N_LINES + 1):
        search_lines.append(
            f"src/runtime.py:{i}: "
            f"worker state trace_ref='req-{i:04d}'"
        )

    return needle, [
        {
            "role": "user",
            "content": (
                "Investigate the intermittent checkout failure "
                "and determine the root cause."
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "evidence_a",
            "content": (
                f"Observed affected request {needle}"
            ),
        },
        {
            "role": "assistant",
            "content": "Inspect another signal.",
        },
        {
            "role": "tool",
            "tool_call_id": "evidence_b",
            "content": (
                f"Follow-up trace confirms {needle}"
            ),
        },
        {
            "role": "assistant",
            "content": "Search the runtime index.",
        },
        {
            "role": "tool",
            "tool_call_id": "search_target",
            "content": "\n".join(search_lines),
        },
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory-relevance",
        action="store_true",
    )

    args = parser.parse_args()

    rng = random.Random(SEED)

    positions = rng.sample(
        range(50, N_LINES - 49),
        N_CASES,
    )

    retained = 0
    total_before = 0
    total_after = 0

    misses = []

    for position in positions:
        needle, messages = build_messages(
            position
        )

        result = compress(
            messages,
            model="gpt-4o",
            protect_recent=0,
            min_tokens_to_compress=10,
            kompress_model="disabled",
            trajectory_relevance=(
                args.trajectory_relevance
            ),
        )

        target_after = result.messages[-1][
            "content"
        ]

        kept = needle in target_after

        retained += int(kept)

        total_before += result.tokens_before
        total_after += result.tokens_after

        if not kept:
            misses.append(position)

    payload = {
        "trajectory_relevance": (
            args.trajectory_relevance
        ),
        "n_cases": N_CASES,
        "retained": retained,
        "retention_rate": (
            retained / N_CASES
        ),
        "avg_tokens_before": (
            total_before / N_CASES
        ),
        "avg_tokens_after": (
            total_after / N_CASES
        ),
        "avg_compression_ratio": (
            1 - total_after / total_before
        ),
        "miss_positions": misses,
    }

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
