import argparse
import json

from headroom.compress import compress


NEEDLE = "req-0421"
N_LINES = 1200


def build_messages():
    search_lines = []

    for i in range(1, N_LINES + 1):
        search_lines.append(
            f"src/runtime.py:{i}: "
            f"worker state trace_ref='req-{i:04d}'"
        )

    search_target = "\n".join(search_lines)

    return [
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
                "Test failure correlates with request "
                f"{NEEDLE}"
            ),
        },
        {
            "role": "assistant",
            "content": "I will inspect another signal.",
        },
        {
            "role": "tool",
            "tool_call_id": "evidence_b",
            "content": (
                "Affected request confirmed in follow-up trace: "
                f"{NEEDLE}"
            ),
        },
        {
            "role": "assistant",
            "content": "Search the runtime index.",
        },
        {
            "role": "tool",
            "tool_call_id": "search_target",
            "content": search_target,
        },
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory-relevance",
        action="store_true",
    )

    args = parser.parse_args()

    messages = build_messages()

    result = compress(
        messages,
        model="gpt-4o",
        protect_recent=0,
        min_tokens_to_compress=10,
        kompress_model="disabled",
        trajectory_relevance=args.trajectory_relevance,
    )

    target_after = result.messages[-1]["content"]

    payload = {
        "trajectory_relevance": (
            args.trajectory_relevance
        ),
        "needle": NEEDLE,
        "needle_retained": NEEDLE in target_after,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "compression_ratio": result.compression_ratio,
        "target_chars_before": len(
            messages[-1]["content"]
        ),
        "target_chars_after": len(target_after),
        "transforms_applied": result.transforms_applied,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
