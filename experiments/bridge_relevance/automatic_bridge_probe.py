import csv
import random
import re
from collections import Counter
from pathlib import Path

from headroom.transforms import SearchCompressor, SearchCompressorConfig


N_LINES = 500
N_CASES = 100
BUDGETS = [10, 20, 30, 50, 100]
N_DISTRACTORS = 15
TOP_K = 6
SEED = 42

USER_QUERY = "Investigate why checkout sometimes behaves incorrectly."

OUTPUT_PATH = Path(
    "experiments/bridge_relevance/automatic_bridge_results.csv"
)


# ---------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------

PATTERNS = [
    # Request / trace-like identifiers
    re.compile(r"\b(?:req|trace|job|task)-[A-Za-z0-9_-]+\b"),

    # Test names
    re.compile(r"\btest_[A-Za-z0-9_]+\b"),

    # Exception / error class names
    re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception)\b"),

    # File paths
    re.compile(
        r"\b(?:[A-Za-z0-9_.-]+/)+"
        r"[A-Za-z0-9_.-]+\.(?:py|js|ts|java|go|rs|cpp|c|h)\b"
    ),

    # Upper-case config / symbolic identifiers
    re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b"),
]


HIGH_SIGNAL_WORDS = {
    "affected": 6,
    "root cause": 6,
    "failed": 5,
    "failure": 5,
    "error": 5,
    "exception": 5,
    "traceback": 5,
    "assertion": 4,
    "expected": 3,
    "actual": 3,
}


def find_candidates(text):
    candidates = []

    for line_no, line in enumerate(text.splitlines()):
        lower = line.lower()

        line_bonus = 0
        for phrase, score in HIGH_SIGNAL_WORDS.items():
            if phrase in lower:
                line_bonus = max(line_bonus, score)

        for pattern in PATTERNS:
            for match in pattern.finditer(line):
                candidates.append(
                    {
                        "token": match.group(0),
                        "line_no": line_no,
                        "line_bonus": line_bonus,
                    }
                )

    return candidates


def extract_all_identifiers(text):
    """Naive baseline: keep every unique candidate."""
    seen = set()
    result = []

    for candidate in find_candidates(text):
        token = candidate["token"]

        if token not in seen:
            seen.add(token)
            result.append(token)

    return result


def extract_bridge_candidates(text, top_k=TOP_K):
    """
    Lightweight bridge extractor.

    Scores identifiers using:
    - proximity to failure/error-like language
    - repeated occurrence
    - slight recency preference
    """

    candidates = find_candidates(text)

    counts = Counter(c["token"] for c in candidates)

    scores = {}

    for candidate in candidates:
        token = candidate["token"]

        score = 0.0

        # Strongest signal: identifier appears on a salient line.
        score += candidate["line_bonus"]

        # Repeated identifiers may represent an active entity.
        score += min(counts[token], 3)

        # Very small recency preference.
        score += candidate["line_no"] * 0.001

        if token not in scores or score > scores[token]:
            scores[token] = score

    ranked = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return [token for token, _ in ranked[:top_k]]


# ---------------------------------------------------------------------
# Synthetic coding-agent trajectory
# ---------------------------------------------------------------------

def make_search_output():
    """
    The second tool returns 500 almost-identical search results.
    The only meaningful discriminator is req-XXXX.
    """
    return "\n".join(
        f"src/runtime.py:{100 + i}:trace_ref = 'req-{i:04d}'"
        for i in range(1, N_LINES + 1)
    )


def make_previous_tool_output(needle_position, rng):
    """
    Simulates a previous test/log tool output.

    It contains:
    - many irrelevant request IDs
    - one failed test
    - one explicitly affected request (the true bridge)
    """

    needle = f"req-{needle_position:04d}"

    available = [
        i
        for i in range(50, 451)
        if i != needle_position
    ]

    distractor_positions = rng.sample(
        available,
        N_DISTRACTORS,
    )

    lines = []

    for pos in distractor_positions[:8]:
        lines.append(
            f"worker completed normal request req-{pos:04d}"
        )

    lines.append(
        "FAILED test_checkout_consistency"
    )

    lines.append(
        f"Affected request: {needle}"
    )

    for pos in distractor_positions[8:]:
        lines.append(
            f"cleanup completed for req-{pos:04d}"
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
        additions = extract_all_identifiers(previous_tool)

    elif condition == "bridge":
        additions = extract_bridge_candidates(previous_tool)

    elif condition == "oracle":
        additions = [needle]

    else:
        raise ValueError(condition)

    if additions:
        context = USER_QUERY + " " + " ".join(additions)
    else:
        context = USER_QUERY

    return context, additions


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def main():
    rng = random.Random(SEED)

    search_output = make_search_output()

    candidate_positions = list(range(50, 451))

    needles = rng.sample(
        candidate_positions,
        N_CASES,
    )

    conditions = [
        "user_only",
        "naive",
        "bridge",
        "oracle",
    ]

    rows = []

    for budget in BUDGETS:
        stats = {
            condition: {
                "hits": 0,
                "added_chars": 0,
                "candidate_count": 0,
            }
            for condition in conditions
        }

        for position in needles:
            needle = f"req-{position:04d}"

            previous_tool = make_previous_tool_output(
                position,
                rng,
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

                preserved = needle in compressed

                added_chars = (
                    len(context) - len(USER_QUERY)
                )

                stats[condition]["hits"] += int(preserved)
                stats[condition]["added_chars"] += added_chars
                stats[condition]["candidate_count"] += len(additions)

                rows.append(
                    {
                        "budget": budget,
                        "position": position,
                        "needle": needle,
                        "condition": condition,
                        "preserved": preserved,
                        "added_chars": added_chars,
                        "candidate_count": len(additions),
                        "selected_candidates": " ".join(additions),
                    }
                )

        print("=" * 72)
        print(f"BUDGET: {budget} / {N_LINES}")
        print("=" * 72)

        for condition in conditions:
            s = stats[condition]

            recall = s["hits"] / N_CASES

            avg_chars = (
                s["added_chars"] / N_CASES
            )

            avg_candidates = (
                s["candidate_count"] / N_CASES
            )

            print(
                f"{condition:10s} "
                f"RECALL={recall:6.1%}   "
                f"AVG_CANDIDATES={avg_candidates:5.1f}   "
                f"AVG_ADDED_CHARS={avg_chars:7.1f}"
            )

        print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "budget",
                "position",
                "needle",
                "condition",
                "preserved",
                "added_chars",
                "candidate_count",
                "selected_candidates",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Raw results written to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
