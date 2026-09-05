import re
from collections import Counter, defaultdict


TOP_K = 6


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


STOP_TOKENS = {
    "FAILED",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "TRACE",
    "PASSED",
}


# Order matters.
# More specific structures come first so that, for example,
# a function-looking substring inside a file path is not counted twice.
PATTERNS = [
    (
        "file_path",
        re.compile(
            r"\b(?:[A-Za-z0-9_.-]+/)+"
            r"[A-Za-z0-9_.-]+\."
            r"(?:py|js|ts|java|go|rs|cpp|c|h)\b"
        ),
    ),
    (
        "request_id",
        re.compile(
            r"\b(?:req|trace|job|task)-"
            r"[A-Za-z0-9_-]+\b"
        ),
    ),
    (
        "test_name",
        re.compile(
            r"\btest_[A-Za-z0-9_]+\b"
        ),
    ),
    (
        "exception",
        re.compile(
            r"\b[A-Z][A-Za-z0-9]*"
            r"(?:Error|Exception)\b"
        ),
    ),
    (
        "config_key",
        re.compile(
            r"\b[A-Z][A-Z0-9_]{3,}\b"
        ),
    ),
    (
        "function_name",
        re.compile(
            r"\b[a-z][a-z0-9]*"
            r"(?:_[a-z0-9]+)+\b"
        ),
    ),
    (
        "class_name",
        re.compile(
            r"\b[A-Z][a-z]+"
            r"(?:[A-Z][A-Za-z0-9]*)+\b"
        ),
    ),
]


def _overlaps(start, end, occupied):
    for old_start, old_end in occupied:
        if start < old_end and end > old_start:
            return True
    return False


def find_candidates(text):
    """
    Extract code-like bridge candidates.

    Important difference from v1:
    the candidate itself is masked before contextual
    salience words are scored.

    Therefore:
        Service0184Error
    does not receive an 'error' bonus merely because
    its own class name ends with Error.
    """

    results = []

    for line_no, line in enumerate(text.splitlines()):
        occupied = []

        for kind, pattern in PATTERNS:
            for match in pattern.finditer(line):
                start, end = match.span()

                if _overlaps(
                    start,
                    end,
                    occupied,
                ):
                    continue

                token = match.group(0)

                if token in STOP_TOKENS:
                    continue

                occupied.append((start, end))

                # Remove the identifier itself before looking
                # for contextual words such as "error".
                masked_line = (
                    line[:start]
                    + " [CANDIDATE] "
                    + line[end:]
                )

                lower = masked_line.lower()

                contextual_bonus = 0

                for phrase, score in (
                    HIGH_SIGNAL_WORDS.items()
                ):
                    if phrase in lower:
                        contextual_bonus = max(
                            contextual_bonus,
                            score,
                        )

                results.append(
                    {
                        "token": token,
                        "kind": kind,
                        "line_no": line_no,
                        "contextual_bonus": (
                            contextual_bonus
                        ),
                    }
                )

    return results


def extract_all_identifiers_from_tools(
    tool_outputs,
):
    """
    Stronger naive baseline:
    take every unique identifier from every tool.
    """

    seen = set()
    result = []

    for text in tool_outputs:
        for candidate in find_candidates(text):
            token = candidate["token"]

            if token not in seen:
                seen.add(token)
                result.append(token)

    return result


def extract_bridge_candidates_v2(
    tool_outputs,
    top_k=TOP_K,
):
    """
    Cross-tool bridge extractor.

    Main signal:
        an identifier appearing in multiple DISTINCT
        tool outputs receives a strong recurrence bonus.

    Additional signals:
        contextual salience
        repeated occurrence
        small recency preference
    """

    occurrence_count = Counter()
    tool_presence = defaultdict(set)
    best_contextual_bonus = defaultdict(float)
    last_seen = {}
    token_kind = {}

    for tool_index, text in enumerate(tool_outputs):
        candidates = find_candidates(text)

        for candidate in candidates:
            token = candidate["token"]

            occurrence_count[token] += 1

            tool_presence[token].add(
                tool_index
            )

            best_contextual_bonus[token] = max(
                best_contextual_bonus[token],
                candidate["contextual_bonus"],
            )

            last_seen[token] = (
                tool_index,
                candidate["line_no"],
            )

            token_kind[token] = candidate["kind"]

    scored = []

    for token in occurrence_count:
        distinct_tools = len(
            tool_presence[token]
        )

        score = 0.0

        # Core V2 idea:
        # appearing across different tools is a
        # strong bridge signal.
        score += 12.0 * max(
            0,
            distinct_tools - 1,
        )

        # Repeated appearances still help slightly,
        # even within the same tool.
        score += 2.0 * min(
            occurrence_count[token],
            3,
        )

        # Contextual error/failure/root-cause cues.
        score += best_contextual_bonus[token]

        # Tiny deterministic recency preference.
        tool_index, line_no = last_seen[token]

        score += tool_index * 0.01
        score += line_no * 0.0001

        scored.append(
            {
                "token": token,
                "kind": token_kind[token],
                "score": score,
                "distinct_tools": distinct_tools,
                "occurrences": (
                    occurrence_count[token]
                ),
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
