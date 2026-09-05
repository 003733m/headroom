from collections import Counter, defaultdict

from bridge_extractor_v2 import find_candidates


TOP_K = 6


POSITIVE_CUES = {
    "root cause": 10,
    "affected": 9,
    "failed": 7,
    "exception raised": 7,
    "traceback": 6,
    "assertion": 6,
    "expected": 4,
    "actual": 4,
}


NEGATIVE_CUES = {
    "suspicion": 5,
    "suspect": 5,
    "hypothesis": 5,
    "possible": 4,
    "maybe": 4,
    "candidate": 3,
}


def _context_score(line, token):
    """
    Score the surrounding text, not the identifier itself.

    Example:
        Service0184Error

    must not receive an 'error' bonus simply because
    the identifier itself contains the word Error.
    """

    masked = line.replace(
        token,
        " [CANDIDATE] ",
        1,
    ).lower()

    positive = 0

    for phrase, score in POSITIVE_CUES.items():
        if phrase in masked:
            positive = max(
                positive,
                score,
            )

    negative = 0

    for phrase, score in NEGATIVE_CUES.items():
        if phrase in masked:
            negative = max(
                negative,
                score,
            )

    return positive, negative


def extract_bridge_candidates_v3(
    tool_outputs,
    top_k=TOP_K,
):
    """
    Hybrid trajectory-aware bridge ranking.

    Signals:

    1. Explicit causal / confirmed local evidence.
    2. Occurrence across DISTINCT tool outputs.
    3. Repeated occurrence.
    4. Recency.
    5. Negative penalty for speculative language.

    The weights represent semantic priorities rather
    than being tuned against individual benchmark cases.
    """

    occurrence_count = Counter()

    tool_presence = defaultdict(set)

    best_positive = defaultdict(int)
    strongest_negative = defaultdict(int)

    last_seen_tool = {}
    token_kind = {}

    n_tools = len(tool_outputs)

    for tool_index, text in enumerate(
        tool_outputs
    ):
        lines = text.splitlines()

        candidates = find_candidates(text)

        for candidate in candidates:
            token = candidate["token"]
            line_no = candidate["line_no"]

            occurrence_count[token] += 1

            tool_presence[token].add(
                tool_index
            )

            token_kind[token] = (
                candidate["kind"]
            )

            last_seen_tool[token] = (
                tool_index
            )

            line = lines[line_no]

            positive, negative = (
                _context_score(
                    line,
                    token,
                )
            )

            best_positive[token] = max(
                best_positive[token],
                positive,
            )

            strongest_negative[token] = max(
                strongest_negative[token],
                negative,
            )

    scored = []

    for token in occurrence_count:
        distinct_tools = len(
            tool_presence[token]
        )

        occurrences = occurrence_count[
            token
        ]

        score = 0.0

        # -------------------------------------------------
        # 1. Confirmed / causal local evidence
        # -------------------------------------------------

        score += best_positive[token]

        # -------------------------------------------------
        # 2. Cross-tool corroboration
        #
        # Strong, but no longer dominant over explicit
        # causal evidence.
        # -------------------------------------------------

        score += 6.0 * max(
            0,
            distinct_tools - 1,
        )

        # -------------------------------------------------
        # 3. Repetition
        # -------------------------------------------------

        score += min(
            occurrences,
            3,
        )

        # -------------------------------------------------
        # 4. Recency
        #
        # Normalize latest tool to roughly [0, 2].
        # -------------------------------------------------

        if n_tools > 1:
            recency = (
                last_seen_tool[token]
                / (n_tools - 1)
            )
        else:
            recency = 0.0

        score += 2.0 * recency

        # -------------------------------------------------
        # 5. Speculation penalty
        # -------------------------------------------------

        score -= strongest_negative[
            token
        ]

        scored.append(
            {
                "token": token,
                "kind": token_kind[token],
                "score": score,
                "distinct_tools": (
                    distinct_tools
                ),
                "occurrences": occurrences,
                "positive": (
                    best_positive[token]
                ),
                "negative": (
                    strongest_negative[token]
                ),
                "recency": recency,
            }
        )

    scored.sort(
        key=lambda item: (
            -item["score"],
            item["token"],
        )
    )

    return [
        item["token"]
        for item in scored[:top_k]
    ]

