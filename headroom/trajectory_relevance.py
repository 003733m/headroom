"""Trajectory-derived relevance hints for search compression.

This module extracts small, high-specificity identifiers from PRIOR tool
results so later search-result compression can preserve evidence connected
to the agent's active investigation.

Important invariants:
- The target message must never contribute to its own relevance context.
- The output is bounded.
- Weak trajectories may abstain and return no bridge identifiers.
- These scores are POC defaults, not final calibrated weights.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

DEFAULT_TOP_K = 6
DEFAULT_MAX_TOOL_OUTPUTS = 8


@dataclass(frozen=True)
class BridgeScoringConfig:
    """Configurable weights for trajectory bridge ranking.

    The default values reproduce the original POC scoring exactly, but
    expose normalized [0, 1] signals so weights can later be calibrated
    on disjoint validation data rather than hard-coded by hand.

    The weights represent the maximum contribution of each signal:
    - causal evidence
    - cross-tool corroboration
    - raw occurrence
    - recency
    - speculative-language penalty

    ``min_score`` is optional and can later be calibrated as an
    abstention threshold. ``None`` preserves the existing behavior.
    """

    # Frozen after calibration on a disjoint synthetic calibration set.
    # Do not retune these values on held-out or coding-agent evaluation.
    causal_weight: float = 4.0
    cross_tool_weight: float = 2.0
    occurrence_weight: float = 2.0
    recency_weight: float = 1.0
    speculation_weight: float = 2.0

    max_cross_tool_steps: int = 2
    max_occurrences: int = 3

    min_score: float | None = 2.0


@dataclass(frozen=True)
class BridgeCandidate:
    """Ranked trajectory identifier."""

    token: str
    kind: str
    score: float
    distinct_tools: int
    occurrences: int
    positive_evidence: int
    negative_evidence: int
    recency: float


@dataclass(frozen=True)
class _Hit:
    token: str
    kind: str
    line_no: int


# Ordered from more structurally specific to more general.
_PATTERN_SPECS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    (
        "file_path",
        re.compile(
            r"(?<![\w.-])"
            r"((?:[A-Za-z0-9_.-]+/)+"
            r"[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)"
        ),
        1,
    ),
    (
        "request_id",
        re.compile(
            r"\b("
            r"(?:req|request|trace|session|job|task)"
            r"[-_][A-Za-z0-9][A-Za-z0-9_-]*"
            r")\b",
            re.IGNORECASE,
        ),
        1,
    ),
    (
        "test_name",
        re.compile(r"\b(test_[A-Za-z0-9_]+)\b"),
        1,
    ),
    (
        "exception",
        re.compile(
            r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b"
        ),
        1,
    ),
    (
        "config_key",
        re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b"),
        1,
    ),
    (
        "function_name",
        re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*\("),
        1,
    ),
    (
        "class_name",
        re.compile(
            r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b"
        ),
        1,
    ),
)


_POSITIVE_CUES: dict[str, int] = {
    "root cause": 10,
    "affected": 9,
    "confirmed": 8,
    "identified": 8,
    "failed": 7,
    "failure": 7,
    "exception raised": 7,
    "traceback": 6,
    "assertion": 6,
    "expected": 4,
    "actual": 4,
}


_NEGATIVE_CUES: dict[str, int] = {
    "suspicion": 5,
    "suspect": 5,
    "hypothesis": 5,
    "possible": 4,
    "maybe": 4,
    "candidate": 3,
}


_MAX_POSITIVE_CUE = max(_POSITIVE_CUES.values())
_MAX_NEGATIVE_CUE = max(_NEGATIVE_CUES.values())


def _text_from_blocks(blocks: list[Any]) -> str:
    texts: list[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)

    return "\n".join(texts)


def _tool_result_text(message: dict[str, Any]) -> str | None:
    """Extract textual tool-result content from OpenAI/Anthropic shapes."""

    # OpenAI format.
    if message.get("role") == "tool":
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text = _text_from_blocks(content)
            return text or None

        return None

    # Anthropic format:
    # role="user", content=[{"type": "tool_result", ...}]
    content = message.get("content")

    if not isinstance(content, list):
        return None

    results: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue

        if block.get("type") != "tool_result":
            continue

        inner = block.get("content")

        if isinstance(inner, str):
            if inner:
                results.append(inner)
        elif isinstance(inner, list):
            text = _text_from_blocks(inner)
            if text:
                results.append(text)

    return "\n".join(results) or None


def extract_prior_tool_outputs(
    messages: list[dict[str, Any]],
    *,
    before_index: int,
    max_tool_outputs: int = DEFAULT_MAX_TOOL_OUTPUTS,
) -> list[str]:
    """Return recent tool outputs strictly BEFORE ``before_index``.

    The target message at ``before_index`` is intentionally excluded.
    This prevents self-leakage: a search result cannot provide the
    identifier used to decide which of its own lines survive compression.
    """

    if before_index <= 0:
        return []

    outputs: list[str] = []

    for message in messages[:before_index]:
        text = _tool_result_text(message)

        if text and text.strip():
            outputs.append(text)

    if max_tool_outputs <= 0:
        return []

    return outputs[-max_tool_outputs:]


def _find_candidates(text: str) -> list[_Hit]:
    hits: list[_Hit] = []

    for line_no, line in enumerate(text.splitlines()):
        occupied: list[tuple[int, int]] = []

        for kind, pattern, group in _PATTERN_SPECS:
            for match in pattern.finditer(line):
                start, end = match.span(group)

                # Do not emit the same substring under multiple kinds.
                if any(
                    not (end <= old_start or start >= old_end)
                    for old_start, old_end in occupied
                ):
                    continue

                token = match.group(group)

                if not token:
                    continue

                occupied.append((start, end))

                hits.append(
                    _Hit(
                        token=token,
                        kind=kind,
                        line_no=line_no,
                    )
                )

    return hits


def _context_evidence(
    line: str,
    token: str,
) -> tuple[int, int]:
    """Score surrounding words without letting the token score itself."""

    masked = line.replace(
        token,
        " [BRIDGE] ",
        1,
    ).lower()

    positive = max(
        (
            score
            for cue, score in _POSITIVE_CUES.items()
            if cue in masked
        ),
        default=0,
    )

    negative = max(
        (
            score
            for cue, score in _NEGATIVE_CUES.items()
            if cue in masked
        ),
        default=0,
    )

    return positive, negative


def rank_bridge_candidates(
    tool_outputs: list[str],
    *,
    top_k: int | None = DEFAULT_TOP_K,
    scoring: BridgeScoringConfig | None = None,
) -> list[BridgeCandidate]:
    """Rank identifiers using bounded hybrid trajectory evidence.

    POC scoring principles:
    - explicit causal evidence is strongest;
    - distinct-tool recurrence is strong corroboration;
    - raw repetition is weak and saturated;
    - recent evidence receives a small bonus;
    - speculative language is penalized.

    A candidate is eligible only when it has either:
    - cross-tool evidence, or
    - explicit positive local evidence.

    This eligibility rule provides simple abstention for weak/noisy
    trajectories instead of always injecting arbitrary identifiers.
    """

    scoring = scoring or BridgeScoringConfig()

    if not tool_outputs:
        return []

    if top_k is not None and top_k <= 0:
        return []

    occurrence_count: Counter[str] = Counter()
    tool_presence: dict[str, set[int]] = defaultdict(set)
    best_positive: dict[str, int] = defaultdict(int)
    strongest_negative: dict[str, int] = defaultdict(int)
    last_seen_tool: dict[str, int] = {}
    token_kind: dict[str, str] = {}

    n_tools = len(tool_outputs)

    for tool_index, text in enumerate(tool_outputs):
        lines = text.splitlines()

        for hit in _find_candidates(text):
            token = hit.token

            occurrence_count[token] += 1
            tool_presence[token].add(tool_index)
            last_seen_tool[token] = tool_index
            token_kind[token] = hit.kind

            line = lines[hit.line_no]

            positive, negative = _context_evidence(
                line,
                token,
            )

            best_positive[token] = max(
                best_positive[token],
                positive,
            )

            strongest_negative[token] = max(
                strongest_negative[token],
                negative,
            )

    ranked: list[BridgeCandidate] = []

    for token, occurrences in occurrence_count.items():
        distinct_tools = len(tool_presence[token])
        positive = best_positive[token]
        negative = strongest_negative[token]

        # Abstain from weak identifiers with neither corroboration
        # nor explicit evidence.
        if distinct_tools < 2 and positive <= 0:
            continue

        # Saturate cross-tool recurrence so a stale hypothesis cannot
        # accumulate unlimited score simply by being investigated often.
        cross_tool_steps = min(
            max(distinct_tools - 1, 0),
            scoring.max_cross_tool_steps,
        )

        bounded_occurrences = min(
            occurrences,
            scoring.max_occurrences,
        )

        if n_tools > 1:
            recency = (
                last_seen_tool[token]
                / (n_tools - 1)
            )
        else:
            recency = 0.0

        # Normalize every signal to [0, 1]. This makes weight calibration
        # meaningful: a weight represents the maximum contribution of that
        # signal rather than compensating for incompatible raw scales.
        causal_signal = (
            float(positive)
            / float(_MAX_POSITIVE_CUE)
        )

        if scoring.max_cross_tool_steps > 0:
            cross_tool_signal = (
                float(cross_tool_steps)
                / float(scoring.max_cross_tool_steps)
            )
        else:
            cross_tool_signal = 0.0

        if scoring.max_occurrences > 0:
            occurrence_signal = (
                float(bounded_occurrences)
                / float(scoring.max_occurrences)
            )
        else:
            occurrence_signal = 0.0

        speculation_signal = (
            float(negative)
            / float(_MAX_NEGATIVE_CUE)
        )

        score = (
            scoring.causal_weight * causal_signal
            + scoring.cross_tool_weight * cross_tool_signal
            + scoring.occurrence_weight * occurrence_signal
            + scoring.recency_weight * recency
            - scoring.speculation_weight * speculation_signal
        )

        if (
            scoring.min_score is not None
            and score < scoring.min_score
        ):
            continue

        ranked.append(
            BridgeCandidate(
                token=token,
                kind=token_kind[token],
                score=score,
                distinct_tools=distinct_tools,
                occurrences=occurrences,
                positive_evidence=positive,
                negative_evidence=negative,
                recency=recency,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.score,
            -item.distinct_tools,
            -item.recency,
            item.token,
        )
    )

    if top_k is None:
        return ranked

    return ranked[:top_k]



@dataclass(frozen=True)
class TargetBridgeCandidate:
    """A frozen V1 candidate conditioned on the current SEARCH target."""

    candidate: BridgeCandidate
    score: float
    target_specificity: float

    @property
    def token(self) -> str:
        return self.candidate.token

    @property
    def kind(self) -> str:
        return self.candidate.kind


# Deliberately narrow. This is not a general software-engineering stopword
# list; it only suppresses obvious builtin/type extraction artifacts.
_GENERIC_BRIDGE_IDENTIFIERS = frozenset(
    {
        "bool",
        "bytes",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "object",
        "set",
        "str",
        "tuple",
        "type",
    }
)


def _normalized_candidate_text(text: str, kind: str) -> str:
    """Normalize representation details needed for exact lexical matching."""
    if kind == "file_path":
        return text.replace("\\", "/")
    return text


def _candidate_match_pattern(
    token: str,
    *,
    kind: str,
) -> re.Pattern[str]:
    """Build a conservative exact lexical matcher."""
    normalized = _normalized_candidate_text(token, kind)
    escaped = re.escape(normalized)

    # Paths may contain dots and dashes as part of components.
    if kind == "file_path":
        boundary = r"A-Za-z0-9_.-"
    else:
        # Keep identifiers exact while still allowing request IDs such as
        # req-0184 to be matched as a whole lexical unit.
        boundary = r"A-Za-z0-9_"

    return re.compile(
        rf"(?<![{boundary}]){escaped}(?![{boundary}])",
        re.IGNORECASE,
    )


def _contains_candidate(
    text: str,
    token: str,
    *,
    kind: str,
) -> bool:
    """Check whether an existing trajectory identifier occurs lexically."""
    normalized = _normalized_candidate_text(text, kind)

    return (
        _candidate_match_pattern(
            token,
            kind=kind,
        ).search(normalized)
        is not None
    )


def _is_generic_bridge_identifier(
    candidate: BridgeCandidate,
) -> bool:
    """Reject only obvious builtin/type extraction artifacts."""
    if candidate.kind not in {"function_name", "class_name"}:
        return False

    return candidate.token.casefold() in _GENERIC_BRIDGE_IDENTIFIERS


def _is_weak_unstructured_bridge_identifier(
    candidate: BridgeCandidate,
) -> bool:
    """Reject low-specificity symbol shapes unless trajectory evidence is causal.

    V1 intentionally extracts broadly. V2 is more conservative about lexical
    forms that commonly arise from ordinary source/prose rather than stable
    join keys.

    Explicit positive evidence always rescues the candidate.
    """

    if candidate.positive_evidence > 0:
        return False

    if candidate.kind == "function_name":
        # A plain single-word call such as get(...) or startswith(...)
        # is highly reusable across unrelated code. Compound snake_case
        # names carry substantially more lexical identity.
        return "_" not in candidate.token

    if candidate.kind == "config_key":
        # The V1 config-key regex also captures short prose acronyms such
        # as API. Preserve longer constants such as PORT/HOST/DEBUG and
        # structured keys such as OPENAI_TARGET_API_URL.
        return (
            "_" not in candidate.token
            and len(candidate.token) <= 3
        )

    return False


def _target_document_frequencies(
    candidates: list[BridgeCandidate],
    target_content: str,
) -> tuple[int, dict[BridgeCandidate, int]]:
    """Count candidate-bearing target lines with batched exact matching.

    This preserves the lexical boundary semantics used by
    ``target_specificity`` while avoiding a full target scan per candidate.

    Candidate strings that can overlap are conservatively evaluated with
    the historical scalar matcher so regex alternation cannot hide a
    legitimate nested match.
    """

    frequencies = dict.fromkeys(candidates, 0)

    raw_lines = target_content.splitlines(keepends=True)

    n_nonempty = sum(
        1
        for line in raw_lines
        if line.strip()
    )

    if not candidates or not raw_lines:
        return n_nonempty, frequencies

    line_starts: list[int] = []
    offset = 0

    for line in raw_lines:
        line_starts.append(offset)
        offset += len(line)

    normalized_tokens = [
        _normalized_candidate_text(
            candidate.token,
            candidate.kind,
        )
        for candidate in candidates
    ]

    folded_tokens = [
        token.casefold()
        for token in normalized_tokens
    ]

    # Alternation consumes a match. If two candidate strings can overlap,
    # evaluating them together could hide the shorter candidate. Preserve
    # exact historical behavior for those candidates with scalar matching.
    ambiguous: set[int] = set()

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            a = folded_tokens[left]
            b = folded_tokens[right]

            if a in b or b in a:
                ambiguous.add(left)
                ambiguous.add(right)

    normalized_line_cache: dict[str, list[str]] = {}

    for index in ambiguous:
        candidate = candidates[index]

        cache_key = (
            "file"
            if candidate.kind == "file_path"
            else "identifier"
        )

        lines = normalized_line_cache.get(cache_key)

        if lines is None:
            lines = [
                _normalized_candidate_text(
                    line,
                    candidate.kind,
                )
                for line in target_content.splitlines()
                if line.strip()
            ]
            normalized_line_cache[cache_key] = lines

        pattern = _candidate_match_pattern(
            candidate.token,
            kind=candidate.kind,
        )

        frequencies[candidate] = sum(
            1
            for line in lines
            if pattern.search(line) is not None
        )

    groups: dict[str, list[int]] = {
        "file": [],
        "identifier": [],
    }

    for index, candidate in enumerate(candidates):
        if index in ambiguous:
            continue

        group = (
            "file"
            if candidate.kind == "file_path"
            else "identifier"
        )

        groups[group].append(index)

    for group, indexes in groups.items():
        if not indexes:
            continue

        if group == "file":
            boundary = r"A-Za-z0-9_.-"
            haystack = target_content.replace("\\", "/")
        else:
            boundary = r"A-Za-z0-9_"
            haystack = target_content

        token_to_index = {
            normalized_tokens[index].casefold(): index
            for index in indexes
        }

        alternatives = sorted(
            (
                re.escape(normalized_tokens[index])
                for index in indexes
            ),
            key=len,
            reverse=True,
        )

        pattern = re.compile(
            rf"(?<![{boundary}])"
            rf"(?:{'|'.join(alternatives)})"
            rf"(?![{boundary}])",
            re.IGNORECASE,
        )

        seen_lines: dict[int, set[int]] = {
            index: set()
            for index in indexes
        }

        for match in pattern.finditer(haystack):
            index = token_to_index.get(
                match.group(0).casefold()
            )

            if index is None:
                continue

            line_index = (
                bisect.bisect_right(
                    line_starts,
                    match.start(),
                )
                - 1
            )

            seen_lines[index].add(line_index)

        for index, lines in seen_lines.items():
            frequencies[candidates[index]] = len(lines)

    return n_nonempty, frequencies


def _target_specificity_from_frequency(
    *,
    n_lines: int,
    document_frequency: int,
) -> float:
    """Compute the frozen normalized local-IDF from a precomputed DF."""

    if n_lines <= 0 or document_frequency <= 0:
        return 0.0

    denominator = math.log(n_lines + 1)

    if denominator <= 0.0:
        return 0.0

    value = (
        math.log(
            (n_lines + 1)
            / (document_frequency + 1)
        )
        / denominator
    )

    return min(1.0, max(0.0, value))


def target_specificity(
    token: str,
    target_content: str,
    *,
    kind: str,
) -> float:
    """Return normalized local-IDF specificity within the current target.

    The identifier itself must have been discovered from PRIOR tool outputs.
    The target is inspected only for applicability and discriminativeness.

        S(c) = log((N + 1) / (df(c) + 1)) / log(N + 1)

    N:
        number of non-empty target lines

    df(c):
        number of target lines containing candidate c

    Values are bounded to [0, 1].
    """
    lines = [
        _normalized_candidate_text(line, kind)
        for line in target_content.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    pattern = _candidate_match_pattern(
        token,
        kind=kind,
    )

    document_frequency = sum(
        1
        for line in lines
        if pattern.search(line) is not None
    )

    # Candidate absent from this target.
    if document_frequency == 0:
        return 0.0

    return _target_specificity_from_frequency(
        n_lines=len(lines),
        document_frequency=document_frequency,
    )


def select_target_bridge_candidates(
    candidates: list[BridgeCandidate],
    *,
    target_content: str,
    user_context: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[TargetBridgeCandidate]:
    """Condition frozen V1 candidates on the current SEARCH result.

    No V1 scoring weights are changed.

    Hard admissibility checks:
    1. candidate is not an obvious builtin/type extraction artifact;
    2. candidate is not a weak unstructured symbol without causal evidence;
    3. candidate occurs in the current target;
    4. candidate is not an exact lexical duplicate of user context.

    Remaining candidates are ranked by:

        V2 score = frozen V1 score * target specificity
    """
    if not candidates or not target_content or top_k <= 0:
        return []

    admissible: list[BridgeCandidate] = []

    # Apply target-independent gates first so the target matcher processes
    # only candidates that could actually survive V2 selection.
    for candidate in candidates:
        if _is_generic_bridge_identifier(candidate):
            continue

        if _is_weak_unstructured_bridge_identifier(candidate):
            continue

        if user_context and _contains_candidate(
            user_context,
            candidate.token,
            kind=candidate.kind,
        ):
            continue

        admissible.append(candidate)

    if not admissible:
        return []

    n_lines, document_frequencies = _target_document_frequencies(
        admissible,
        target_content,
    )

    if n_lines <= 0:
        return []

    selected: list[TargetBridgeCandidate] = []

    for candidate in admissible:
        specificity = _target_specificity_from_frequency(
            n_lines=n_lines,
            document_frequency=document_frequencies[candidate],
        )

        adjusted_score = candidate.score * specificity

        # Absent candidates and identifiers present in effectively every
        # target line both have no selective value.
        if adjusted_score <= 0.0:
            continue

        selected.append(
            TargetBridgeCandidate(
                candidate=candidate,
                score=adjusted_score,
                target_specificity=specificity,
            )
        )

    selected.sort(
        key=lambda item: (
            -item.score,
            -item.candidate.score,
            -item.candidate.distinct_tools,
            -item.candidate.recency,
            item.token,
        )
    )

    return selected[:top_k]


def build_search_relevance_context(
    messages: list[dict[str, Any]],
    *,
    before_index: int,
    user_context: str,
    target_content: str | None = None,
    novelty_context: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_tool_outputs: int = DEFAULT_MAX_TOOL_OUTPUTS,
    scoring: BridgeScoringConfig | None = None,
) -> str:
    """Build bounded search-only relevance context from prior trajectory.

    When ``target_content`` is omitted, preserve the frozen V1 behavior.
    When supplied, condition V1 candidates on the current SEARCH result
    using the V2 target-aware selection layer.
    """

    outputs = extract_prior_tool_outputs(
        messages,
        before_index=before_index,
        max_tool_outputs=max_tool_outputs,
    )

    ranked = rank_bridge_candidates(
        outputs,
        top_k=None if target_content is not None else top_k,
        scoring=scoring,
    )

    if target_content is not None:
        target_ranked = select_target_bridge_candidates(
            ranked,
            target_content=target_content,
            user_context=(
                user_context
                if novelty_context is None
                else novelty_context
            ),
            top_k=top_k,
        )

        if not target_ranked:
            return user_context

        tokens = " ".join(
            candidate.token
            for candidate in target_ranked
        )
    else:
        # Backward compatibility: callers that do not supply the current
        # target retain the frozen V1 semantics exactly.
        if not ranked:
            return user_context

        tokens = " ".join(
            candidate.token
            for candidate in ranked
        )

    if user_context:
        return (
            f"{user_context}\n"
            f"Trajectory bridge identifiers: {tokens}"
        )

    return f"Trajectory bridge identifiers: {tokens}"


def _responses_text(value: Any) -> str:
    """Flatten text-bearing OpenAI Responses content without interpreting it."""
    if isinstance(value, str):
        return value

    if not isinstance(value, list):
        return ""

    parts: list[str] = []
    for part in value:
        if not isinstance(part, dict):
            continue

        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)

    return "\n".join(parts)


def responses_items_to_bridge_messages(
    items: list[Any],
    *,
    before_index: int,
) -> list[dict[str, Any]]:
    """Adapt prior OpenAI Responses items to the canonical bridge message shape.

    Only items before ``before_index`` are considered, preventing the current
    compression target from contributing to its own relevance context.
    """
    messages: list[dict[str, Any]] = []

    for item in items[:before_index]:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "message":
            role = item.get("role")
            if role not in {"user", "assistant", "system", "developer"}:
                continue

            text = _responses_text(item.get("content"))
            if text:
                messages.append(
                    {
                        "role": role,
                        "content": text,
                    }
                )
            continue

        if item_type in {
            "function_call_output",
            "custom_tool_call_output",
            "local_shell_call_output",
        }:
            text = _responses_text(item.get("output"))
            if text:
                messages.append(
                    {
                        "role": "tool",
                        "content": text,
                    }
                )

    return messages



def _latest_bridge_user_context(
    messages: list[dict[str, Any]],
) -> str:
    """Return latest prior user text for lexical deduplication only."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content")
        if isinstance(content, str) and content:
            return content

    return ""

def build_responses_search_relevance_context(
    items: list[Any],
    *,
    before_index: int,
    user_context: str = "",
    target_content: str | None = None,
    scoring: BridgeScoringConfig | None = None,
) -> str:
    """Build SEARCH relevance context from prior OpenAI Responses trajectory."""
    messages = responses_items_to_bridge_messages(
        items,
        before_index=before_index,
    )

    if not messages:
        return user_context

    # Preserve the Responses adapter's existing context semantics. In
    # particular, an empty baseline context remains empty when no reliable
    # bridge exists; enabling this feature must not independently introduce
    # the user query and confound OFF/ON comparisons.
    return build_search_relevance_context(
        messages,
        before_index=len(messages),
        user_context=user_context,
        target_content=target_content,
        novelty_context=_latest_bridge_user_context(messages),
        scoring=scoring,
    )
