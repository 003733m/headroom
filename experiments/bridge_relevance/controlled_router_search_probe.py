import argparse
import csv
import json
import random
import re
from pathlib import Path

from headroom.transforms.content_router import ContentRouter


N_MATCHES = 500
N_CASES = 100
SEED = 7291

USER_CONTEXT = (
    "Investigate the intermittent checkout failure "
    "and locate the relevant implementation evidence."
)


class Tokenizer:
    def count_text(self, text: str) -> int:
        # Sufficient for the router's keep/drop accounting in this
        # controlled mechanism experiment. The SearchCompressor itself
        # remains the real Rust-backed production implementation.
        return max(1, len(str(text).split()))


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
            "content": USER_CONTEXT,
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
            "content": "Inspect another diagnostic source.",
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


def make_router() -> ContentRouter:
    router = ContentRouter()

    # CONTROLLED MECHANISM ISOLATION:
    #
    # 1. Disable the unconditional lossless search-prefix fold.
    #    Otherwise all semantic search records survive and there is no
    #    selective-retention problem for trajectory relevance to solve.
    router._lossless_first = (  # type: ignore[method-assign]
        lambda content, strategy: (content, None)
    )

    # 2. Disable the earlier relevance-split stage so the experiment
    #    reaches the real SearchCompressor selection branch.
    router.config.relevance_split = False

    return router


def run_case(
    position: int,
    trajectory_relevance: bool,
):
    needle, messages = build_messages(position)

    # Fresh router per case prevents its content-addressed compression
    # cache from reusing a result selected under another needle/context.
    router = make_router()

    result = router.apply(
        messages,
        Tokenizer(),
        context=USER_CONTEXT,
        trajectory_relevance=trajectory_relevance,
        protect_recent=0,
        min_tokens_to_compress=10,
    )

    output = result.messages[-1]["content"]

    retained_ids = set(
        re.findall(r"req-\d{4}", output)
    )

    return {
        "position": position,
        "needle": needle,
        "needle_retained": needle in retained_ids,
        "retained_ids": len(retained_ids),
        "output_chars": len(output),
        "transforms": "|".join(
            result.transforms_applied
        ),
    }


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

    rows = [
        run_case(
            position,
            args.trajectory_relevance,
        )
        for position in positions
    ]

    retained = sum(
        int(row["needle_retained"])
        for row in rows
    )

    avg_retained_ids = (
        sum(
            int(row["retained_ids"])
            for row in rows
        )
        / len(rows)
    )

    mode = (
        "on"
        if args.trajectory_relevance
        else "off"
    )

    out_dir = Path(
        "experiments/bridge_relevance/results"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        out_dir
        / f"controlled_router_{mode}.csv"
    )

    with raw_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "trajectory_relevance": (
            args.trajectory_relevance
        ),
        "n_cases": N_CASES,
        "needle_retained": retained,
        "retention_rate": (
            retained / N_CASES
        ),
        "avg_retained_ids": avg_retained_ids,
        "min_retained_ids": min(
            int(row["retained_ids"])
            for row in rows
        ),
        "max_retained_ids": max(
            int(row["retained_ids"])
            for row in rows
        ),
        "avg_output_chars": (
            sum(
                int(row["output_chars"])
                for row in rows
            )
            / len(rows)
        ),
        "raw_results": str(raw_path),
        "sample_transforms": (
            rows[0]["transforms"]
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
