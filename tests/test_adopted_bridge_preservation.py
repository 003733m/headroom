from __future__ import annotations

from dataclasses import dataclass

from headroom.trajectory_relevance import (
    _responses_adopted_bridge_identifiers,
    _responses_query_identifiers,
    _responses_search_pattern,
)
from headroom.transforms.relevance_split import plan_relevance_split


@dataclass
class _Score:
    score: float


class _FakeScorer:
    def __init__(self, scores: list[float]):
        self._scores = scores

    def score_batch(self, texts: list[str], query: str) -> list[_Score]:
        assert len(texts) == len(self._scores)
        return [_Score(score=value) for value in self._scores]


def test_search_pattern_excludes_scope_and_glob() -> None:
    command = (
        'rg -nH -C 8 "_background_dedup|DEDUP_AUTO_THRESHOLD" '
        "headroom tests --glob '*.py'"
    )

    assert (
        _responses_search_pattern(command)
        == "_background_dedup|DEDUP_AUTO_THRESHOLD"
    )


def test_query_identifier_extraction_is_conservative() -> None:
    identifiers = set(
        _responses_query_identifiers(
            "def publish|skip_compression_recommended|strategy_hint|failed"
        )
    )

    assert identifiers == {
        "publish",
        "skip_compression_recommended",
        "strategy_hint",
    }


def test_responses_adopted_bridge_requires_prior_bridge_and_current_reuse() -> None:
    items = [
        {
            "type": "function_call_output",
            "call_id": "older",
            "output": "DEDUP_AUTO_THRESHOLD DEDUP_HINT_THRESHOLD",
        },
        {
            "type": "local_shell_call",
            "call_id": "search-1",
            "action": {
                "command": (
                    'rg -nH -C 8 "_background_dedup|DEDUP_AUTO_THRESHOLD" '
                    "headroom tests --glob '*.py'"
                )
            },
        },
        {
            "type": "local_shell_call_output",
            "call_id": "search-1",
            "output": "large search result",
        },
    ]

    adopted = _responses_adopted_bridge_identifiers(
        items,
        before_index=2,
        trajectory_context=(
            "Trajectory bridge identifiers: "
            "LocalBackend DEDUP_HINT_THRESHOLD DEDUP_AUTO_THRESHOLD"
        ),
    )

    assert adopted == ("DEDUP_AUTO_THRESHOLD",)


def test_responses_adopted_bridge_abstains_without_intersection() -> None:
    items = [
        {
            "type": "local_shell_call",
            "call_id": "search-1",
            "action": {"command": 'rg -nH "TrafficLearner" .'},
        },
        {
            "type": "local_shell_call_output",
            "call_id": "search-1",
            "output": "large search result",
        },
    ]

    assert (
        _responses_adopted_bridge_identifiers(
            items,
            before_index=1,
            trajectory_context=(
                "Trajectory bridge identifiers: "
                "tests/test_memory/test_traffic_learner.py warning"
            ),
        )
        == ()
    )


def test_relevance_split_force_keeps_exact_adopted_identifier() -> None:
    content = (
        "alpha unrelated record\n"
        "alpha filler\n"
        "\n"
        "DEDUP_AUTO_THRESHOLD = 0.92\n"
        "dedup threshold evidence\n"
        "\n"
        "omega unrelated record\n"
        "omega filler\n"
    )

    # Three blank-line-delimited records. The middle record is deliberately
    # below threshold.
    scorer = _FakeScorer([0.9, 0.1, 0.8])

    runs = plan_relevance_split(
        content,
        "trajectory context",
        scorer,
        threshold=0.5,
        adaptive=False,
        force_keep_identifiers=("DEDUP_AUTO_THRESHOLD",),
    )

    joined_keep = "".join(text for keep, text in runs if keep)

    assert "DEDUP_AUTO_THRESHOLD = 0.92" in joined_keep


def test_relevance_split_identifier_boundary_prevents_substring_match() -> None:
    content = (
        "first relevant\n"
        "\n"
        "MY_DEDUP_AUTO_THRESHOLD_EXTRA = 1\n"
        "\n"
        "last relevant\n"
    )

    scorer = _FakeScorer([0.9, 0.1, 0.8])

    runs = plan_relevance_split(
        content,
        "trajectory context",
        scorer,
        threshold=0.5,
        adaptive=False,
        force_keep_identifiers=("DEDUP_AUTO_THRESHOLD",),
    )

    dispositions = [
        (keep, text)
        for keep, text in runs
        if "MY_DEDUP_AUTO_THRESHOLD_EXTRA" in text
    ]

    assert dispositions
    assert dispositions[0][0] is False


def test_relevance_split_floor_is_monotonic() -> None:
    content = (
        "KEEP_ME = 1\n"
        "\n"
        "drop me\n"
        "\n"
        "also relevant\n"
    )

    scorer = _FakeScorer([0.9, 0.1, 0.8])

    baseline = plan_relevance_split(
        content,
        "query",
        scorer,
        threshold=0.5,
        adaptive=False,
    )

    floor = plan_relevance_split(
        content,
        "query",
        scorer,
        threshold=0.5,
        adaptive=False,
        force_keep_identifiers=("KEEP_ME",),
    )

    baseline_kept = "".join(text for keep, text in baseline if keep)
    floor_kept = "".join(text for keep, text in floor if keep)

    assert baseline_kept in floor_kept or all(
        piece in floor_kept
        for piece in baseline_kept.splitlines()
        if piece
    )


def test_adopted_bridge_provenance_reaches_relevance_without_structural_search(
    monkeypatch,
):
    """Adopted search provenance must not depend on output-shape detection."""
    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
    )

    router = ContentRouter()
    router.config.relevance_split = True

    calls = []

    def fake_relevance_split(content, kind, context):
        calls.append((content, kind, context))
        return "selected adopted evidence"

    monkeypatch.setattr(
        router,
        "_relevance_split_compress",
        fake_relevance_split,
    )

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError(
            "adopted search provenance incorrectly fell through "
            "to the structural routing seam"
        )

    monkeypatch.setattr(
        router,
        "_apply_strategy_to_content",
        forbidden_fallback,
    )

    context = (
        "Trajectory bridge identifiers: COPILOT_PROVIDER_TYPE\n"
        "Adopted bridge identifiers: COPILOT_PROVIDER_TYPE"
    )

    result = router._compress_pure(
        "output whose formatting is not structurally recognized as search",
        CompressionStrategy.CODE_AWARE,
        context,
        trajectory_search_relevance=True,
    )

    assert result.compressed == "selected adopted evidence"
    assert result.strategy_chain == ["search", "relevance_split"]

    assert len(calls) == 1
    assert calls[0][1] == "search"
    assert calls[0][2] == context
