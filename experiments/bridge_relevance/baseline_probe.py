import csv
import random
from pathlib import Path

from headroom.transforms import SearchCompressor, SearchCompressorConfig


N_LINES = 500
N_CASES = 100
BUDGETS = [10, 20, 30, 50, 100]
SEED = 42

OUTPUT_PATH = Path("experiments/bridge_relevance/baseline_results.csv")


def make_search_output():
    return "\n".join(
        f"src/runtime.py:{100 + i}:trace_ref = 'req-{i:04d}'"
        for i in range(1, N_LINES + 1)
    )


def compress_with_context(search_output, context, budget):
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


def main():
    random.seed(SEED)

    search_output = make_search_output()

    # Avoid beginning/end positions because SearchCompressor
    # may intentionally preserve boundary matches.
    candidate_positions = list(range(50, 451))

    needles = random.sample(candidate_positions, N_CASES)

    rows = []

    for budget in BUDGETS:
        stale_hits = 0
        enriched_hits = 0

        for position in needles:
            needle = f"req-{position:04d}"

            stale_context = (
                "Investigate why checkout sometimes behaves incorrectly."
            )

            enriched_context = (
                "Investigate why checkout sometimes behaves incorrectly. "
                f"{needle}"
            )

            stale_output = compress_with_context(
                search_output,
                stale_context,
                budget,
            )

            enriched_output = compress_with_context(
                search_output,
                enriched_context,
                budget,
            )

            stale_preserved = needle in stale_output
            enriched_preserved = needle in enriched_output

            stale_hits += int(stale_preserved)
            enriched_hits += int(enriched_preserved)

            rows.append(
                {
                    "budget": budget,
                    "needle": needle,
                    "position": position,
                    "stale_preserved": stale_preserved,
                    "enriched_preserved": enriched_preserved,
                }
            )

        print("=" * 65)
        print(f"BUDGET: {budget} / {N_LINES}")
        print(
            f"STALE RECALL    : "
            f"{stale_hits}/{N_CASES} = {stale_hits / N_CASES:.1%}"
        )
        print(
            f"ENRICHED RECALL : "
            f"{enriched_hits}/{N_CASES} = {enriched_hits / N_CASES:.1%}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "budget",
                "needle",
                "position",
                "stale_preserved",
                "enriched_preserved",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Raw results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
