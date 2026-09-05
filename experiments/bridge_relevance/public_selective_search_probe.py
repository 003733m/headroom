import argparse
import json
import random
import re

from headroom.compress import compress


N_MATCHES = 500
N_CASES = 100
SEED = 6187


def needle_for(position: int) -> str:
    return f"req-{position:04d}"


def make_search_output() -> str:
    rows = []

    for i in range(1, N_MATCHES + 1):
        rows.append(
            f"src/module_{i:04d}.py:{100 + i}: "
            f"worker completed ordinary operation "
            f"trace_ref='{needle_for(i)}'"
        )

    return "\n".join(rows)


SEARCH_OUTPUT = make_search_output()


def build_messages(position: int):
    needle = needle_for(position)

    return needle, [
        {
            "role": "user",
            "content": (
                "Investigate the intermittent checkout failure "
                "and locate the relevant implementation evidence."
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
            "content": "Check another diagnostic source.",
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
            "content": "Search the source index.",
        },
        {
            "role": "tool",
            "tool_call_id": "search_target",
            "content": SEARCH_OUTPUT,
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
        range(30, N_MATCHES - 29),
        N_CASES,
    )

    retained = 0
    total_before = 0
    total_after = 0

    misses = []
    retained_id_counts = []
    transform_sets = []

    for position in positions:
        needle, messages = build_messages(position)

        result = compress(
            messages,
            model="gpt-4o",
            protect_recent=0,
            min_tokens_to_compress=10,
            kompress_model="disabled",
            trajectory_relevance=args.trajectory_relevance,
        )

        output = result.messages[-1]["content"]

        kept = needle in output
        retained += int(kept)

        if not kept:
            misses.append(position)

        retained_ids = set(
            re.findall(r"req-\d{4}", output)
        )

        retained_id_counts.append(
            len(retained_ids)
        )

        total_before += result.tokens_before
        total_after += result.tokens_after

        transform_sets.append(
            result.transforms_applied
        )

    payload = {
        "trajectory_relevance": (
            args.trajectory_relevance
        ),
        "n_cases": N_CASES,
        "retained": retained,
        "retention_rate": retained / N_CASES,
        "avg_retained_ids": (
            sum(retained_id_counts)
            / len(retained_id_counts)
        ),
        "min_retained_ids": min(retained_id_counts),
        "max_retained_ids": max(retained_id_counts),
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
        "sample_transforms": transform_sets[0],
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
