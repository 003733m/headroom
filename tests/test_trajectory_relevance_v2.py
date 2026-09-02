from __future__ import annotations

import math

import pytest

from headroom.trajectory_relevance import (
    BridgeCandidate,
    select_target_bridge_candidates,
    target_specificity,
)


def candidate(
    token: str,
    *,
    kind: str = "request_id",
    score: float = 4.0,
    distinct_tools: int = 2,
    occurrences: int = 2,
    positive_evidence: int = 8,
    negative_evidence: int = 0,
    recency: float = 0.5,
) -> BridgeCandidate:
    return BridgeCandidate(
        token=token,
        kind=kind,
        score=score,
        distinct_tools=distinct_tools,
        occurrences=occurrences,
        positive_evidence=positive_evidence,
        negative_evidence=negative_evidence,
        recency=recency,
    )


def test_target_specificity_is_normalized_local_idf() -> None:
    target = "\n".join(
        [
            "src/a.py:1:req-0184 failure",
            "src/a.py:2:other",
            "src/a.py:3:other",
            "src/a.py:4:other",
        ]
    )

    actual = target_specificity(
        "req-0184",
        target,
        kind="request_id",
    )

    expected = math.log((4 + 1) / (1 + 1)) / math.log(4 + 1)

    assert actual == pytest.approx(expected)
    assert 0.0 < actual < 1.0


def test_target_specificity_decreases_as_candidate_becomes_common() -> None:
    rare_target = "\n".join(
        [
            "a.py:1:req-0184",
            "a.py:2:other",
            "a.py:3:other",
            "a.py:4:other",
        ]
    )

    common_target = "\n".join(
        [
            "a.py:1:req-0184",
            "a.py:2:req-0184",
            "a.py:3:req-0184",
            "a.py:4:other",
        ]
    )

    rare = target_specificity(
        "req-0184",
        rare_target,
        kind="request_id",
    )
    common = target_specificity(
        "req-0184",
        common_target,
        kind="request_id",
    )

    assert rare > common


def test_candidate_absent_from_target_is_rejected() -> None:
    candidates = [
        candidate("req-0184"),
        candidate("req-missing"),
    ]

    selected = select_target_bridge_candidates(
        candidates,
        target_content=(
            "src/a.py:10:req-0184 failed\n"
            "src/b.py:20:other result"
        ),
        user_context="diagnose the request failure",
    )

    tokens = [item.token for item in selected]

    assert "req-0184" in tokens
    assert "req-missing" not in tokens


def test_candidate_already_in_user_context_is_not_reinjected() -> None:
    candidates = [
        candidate(
            "OpenAI",
            kind="class_name",
            score=5.0,
        ),
        candidate(
            "req-0184",
            score=4.0,
        ),
    ]

    selected = select_target_bridge_candidates(
        candidates,
        target_content=(
            "a.py:1:OpenAI routing\n"
            "a.py:2:req-0184 failed"
        ),
        user_context="Fix the OpenAI routing problem",
    )

    tokens = [item.token for item in selected]

    assert "OpenAI" not in tokens
    assert "req-0184" in tokens


def test_generic_primitive_is_suppressed() -> None:
    candidates = [
        candidate(
            "str",
            kind="function_name",
            score=6.0,
        ),
        candidate(
            "UnicodeEncodeError",
            kind="exception",
            score=4.0,
        ),
    ]

    selected = select_target_bridge_candidates(
        candidates,
        target_content=(
            "a.py:1:value = str(raw)\n"
            "a.py:2:raise UnicodeEncodeError(...)\n"
        ),
        user_context="diagnose the encoding failure",
    )

    tokens = [item.token for item in selected]

    assert "str" not in tokens
    assert "UnicodeEncodeError" in tokens


def test_target_specificity_can_overcome_higher_prior_score() -> None:
    candidates = [
        candidate(
            "provider",
            kind="config_key",
            score=6.0,
        ),
        candidate(
            "req-0184",
            score=4.0,
        ),
    ]

    lines = [
        f"src/x.py:{i}:provider routing"
        for i in range(1, 20)
    ]
    lines.append("src/x.py:20:req-0184 failed")

    selected = select_target_bridge_candidates(
        candidates,
        target_content="\n".join(lines),
        user_context="diagnose the failure",
    )

    tokens = [item.token for item in selected]

    assert tokens[0] == "req-0184"


def test_all_candidates_filtered_means_abstention() -> None:
    candidates = [
        candidate(
            "OpenAI",
            kind="class_name",
            score=5.0,
        ),
        candidate(
            "str",
            kind="function_name",
            score=5.0,
        ),
        candidate(
            "RuntimeError",
            kind="exception",
            score=5.0,
        ),
    ]

    selected = select_target_bridge_candidates(
        candidates,
        target_content=(
            "a.py:1:OpenAI routing\n"
            "a.py:2:value = str(raw)"
        ),
        user_context="Fix the OpenAI routing problem",
    )

    # OpenAI is not novel.
    # str is generic.
    # RuntimeError is absent from the target.
    assert selected == []


def test_sparse_cross_trajectory_join_key_survives_v2() -> None:
    candidates = [
        candidate(
            "req-0184",
            score=5.0,
            distinct_tools=2,
            occurrences=2,
            positive_evidence=8,
        ),
    ]

    target_lines = [
        f"src/module_{i:03d}.py:{i}:ordinary diagnostic result"
        for i in range(1, 101)
    ]

    target_lines[72] = (
        "src/module_073.py:73:"
        "request req-0184 entered fallback"
    )

    selected = select_target_bridge_candidates(
        candidates,
        target_content="\n".join(target_lines),
        user_context="diagnose the failing request",
    )

    assert [item.token for item in selected] == ["req-0184"]
    assert selected[0].target_specificity > 0.7


def test_t5_style_noise_abstains() -> None:
    candidates = [
        candidate(
            "tests/e2e_real_compression.py",
            kind="file_path",
            score=4.8,
        ),
        candidate(
            "tests/test_memory_handler_concurrent_init.py",
            kind="file_path",
            score=4.8,
        ),
        candidate(
            "tests/test_verbosity_learn.py",
            kind="file_path",
            score=4.8,
        ),
        candidate(
            "OpenAI",
            kind="class_name",
            score=4.57,
        ),
        candidate(
            "str",
            kind="function_name",
            score=4.36,
        ),
        candidate(
            "RuntimeError",
            kind="exception",
            score=4.13,
        ),
    ]

    target = "\n".join(
        [
            "headroom/proxy/server.py:10:OpenAI routing",
            "headroom/proxy/server.py:20:value = str(config)",
            "tests/test_provider_registry.py:59:"
            "def test_copilot_openai_target_routes_anthropic_to_copilot():",
        ]
    )

    selected = select_target_bridge_candidates(
        candidates,
        target_content=target,
        user_context=(
            "When the configured OpenAI target is GitHub Copilot, "
            "Claude requests use the wrong endpoint."
        ),
    )

    assert selected == []


def test_semantically_related_but_nonidentical_candidate_is_not_deduplicated() -> None:
    candidates = [
        candidate(
            "anthropic_api_url",
            kind="config_key",
            score=5.0,
        ),
    ]

    selected = select_target_bridge_candidates(
        candidates,
        target_content=(
            "headroom/providers/registry.py:120:provider routing\n"
            "headroom/providers/registry.py:127:"
            "anthropic=anthropic_api_url\n"
            "headroom/providers/registry.py:130:openai=openai_api_url"
        ),
        user_context="Fix the Anthropic endpoint routing problem",
    )

    # "Anthropic endpoint" is semantically related, but the exact lexical
    # identifier anthropic_api_url is new trajectory information.
    assert [item.token for item in selected] == ["anthropic_api_url"]



def test_candidate_present_in_every_target_line_has_zero_specificity() -> None:
    target = "\n".join(
        [
            "a.py:1:provider routing",
            "a.py:2:provider config",
            "a.py:3:provider request",
        ]
    )

    assert target_specificity(
        "provider",
        target,
        kind="config_key",
    ) == pytest.approx(0.0)


def test_builder_v2_uses_current_target_to_keep_applicable_bridge() -> None:
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    messages = [
        {
            "role": "user",
            "content": "diagnose the failing service request",
        },
        {
            "role": "tool",
            "content": "Observed failed request req-0184",
        },
        {
            "role": "tool",
            "content": "Confirmed affected request req-0184",
        },
    ]

    target = "\n".join(
        [
            *[
                f"src/module_{i:03d}.py:{i}:ordinary diagnostic"
                for i in range(1, 20)
            ],
            "src/module_020.py:20:req-0184 entered fallback",
        ]
    )

    context = build_search_relevance_context(
        messages,
        before_index=len(messages),
        user_context="diagnose the failing service request",
        target_content=target,
    )

    assert "req-0184" in context


def test_builder_v2_abstains_when_prior_bridge_is_absent_from_target() -> None:
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    messages = [
        {
            "role": "user",
            "content": "diagnose the failing service request",
        },
        {
            "role": "tool",
            "content": "Observed failed request req-0184",
        },
        {
            "role": "tool",
            "content": "Confirmed affected request req-0184",
        },
    ]

    base = "diagnose the failing service request"

    target = "\n".join(
        f"src/module_{i:03d}.py:{i}:ordinary diagnostic"
        for i in range(1, 21)
    )

    context = build_search_relevance_context(
        messages,
        before_index=len(messages),
        user_context=base,
        target_content=target,
    )

    assert context == base


def test_builder_without_target_preserves_v1_semantics() -> None:
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    messages = [
        {
            "role": "user",
            "content": "diagnose the failing service request",
        },
        {
            "role": "tool",
            "content": "Observed failed request req-0184",
        },
        {
            "role": "tool",
            "content": "Confirmed affected request req-0184",
        },
    ]

    # No target_content means frozen historical V1 behavior.
    context = build_search_relevance_context(
        messages,
        before_index=len(messages),
        user_context="diagnose the failing service request",
    )

    assert "req-0184" in context


def test_responses_builder_v2_conditions_prior_bridge_on_current_target() -> None:
    from headroom.trajectory_relevance import (
        build_responses_search_relevance_context,
    )

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "diagnose the failing service request",
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Observed failed request req-0184",
        },
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": "Confirmed affected request req-0184",
        },
    ]

    target = "\n".join(
        [
            *[
                f"src/module_{i:03d}.py:{i}:ordinary diagnostic"
                for i in range(1, 20)
            ],
            "src/module_020.py:20:req-0184 entered fallback",
        ]
    )

    context = build_responses_search_relevance_context(
        items,
        before_index=len(items),
        target_content=target,
    )

    assert "req-0184" in context


def test_responses_builder_v2_does_not_discover_from_current_target() -> None:
    from headroom.trajectory_relevance import (
        build_responses_search_relevance_context,
    )

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "diagnose the failure",
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Routine diagnostic output",
        },
    ]

    target = "\n".join(
        [
            "src/a.py:1:req-9999 appears here",
            "src/b.py:2:ordinary result",
        ]
    )

    context = build_responses_search_relevance_context(
        items,
        before_index=len(items),
        target_content=target,
    )

    # req-9999 exists only in the current target, never in prior trajectory.
    assert "req-9999" not in context
    assert context == ""


def test_responses_v2_uses_user_query_only_for_exact_deduplication() -> None:
    from headroom.trajectory_relevance import (
        build_responses_search_relevance_context,
    )

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Investigate request req-0184",
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Observed failed request req-0184",
        },
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": "Confirmed affected request req-0184",
        },
    ]

    target = "\n".join(
        [
            *[
                f"src/module_{i:03d}.py:{i}:ordinary result"
                for i in range(1, 20)
            ],
            "src/module_020.py:20:req-0184 entered fallback",
        ]
    )

    context = build_responses_search_relevance_context(
        items,
        before_index=len(items),
        target_content=target,
    )

    # req-0184 is already an exact lexical signal in the user's query.
    # The query is used for deduplication only and must not itself become
    # Responses relevance context.
    assert context == ""


def test_builder_v2_filters_for_target_before_final_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import headroom.trajectory_relevance as tr

    # Six higher-ranked V1 candidates are irrelevant to the current target.
    # The seventh, lower-ranked candidate is the only applicable bridge.
    candidates = [
        tr.BridgeCandidate(
            token=f"req-100{i}",
            kind="request_id",
            score=10.0 - i,
            distinct_tools=2,
            occurrences=2,
            positive_evidence=0,
            negative_evidence=0,
            recency=1.0,
        )
        for i in range(1, 7)
    ]

    candidates.append(
        tr.BridgeCandidate(
            token="req-9999",
            kind="request_id",
            score=1.0,
            distinct_tools=2,
            occurrences=2,
            positive_evidence=0,
            negative_evidence=0,
            recency=0.5,
        )
    )

    def fake_rank_bridge_candidates(
        tool_outputs: list[str],
        *,
        top_k: int | None = tr.DEFAULT_TOP_K,
        scoring: tr.BridgeScoringConfig | None = None,
    ) -> list[tr.BridgeCandidate]:
        del tool_outputs, scoring

        if top_k is None:
            return candidates

        return candidates[:top_k]

    monkeypatch.setattr(
        tr,
        "rank_bridge_candidates",
        fake_rank_bridge_candidates,
    )

    messages = [
        {
            "role": "user",
            "content": "diagnose the failing request",
        },
        {
            "role": "tool",
            "content": "prior diagnostic output",
        },
    ]

    target = "\n".join(
        [
            *[
                f"src/module_{i:03d}.py:{i}:ordinary diagnostic"
                for i in range(1, 20)
            ],
            "src/module_020.py:20:req-9999 entered fallback",
        ]
    )

    context = tr.build_search_relevance_context(
        messages,
        before_index=len(messages),
        user_context="diagnose the failing request",
        target_content=target,
        top_k=6,
    )

    assert "req-9999" in context


def test_v2_rejects_plain_function_name_without_causal_evidence() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        select_target_bridge_candidates,
    )

    candidate = BridgeCandidate(
        token="get",
        kind="function_name",
        score=4.0,
        distinct_tools=2,
        occurrences=3,
        positive_evidence=0,
        negative_evidence=0,
        recency=0.5,
    )

    target = "\n".join(
        [
            "a.py:1:ordinary routing setup",
            'a.py:2:headers.get("authorization")',
            "a.py:3:ordinary routing cleanup",
        ]
    )

    selected = select_target_bridge_candidates(
        [candidate],
        target_content=target,
        user_context="diagnose routing",
    )

    assert selected == []


def test_v2_keeps_structured_function_name_from_cross_tool_evidence() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        select_target_bridge_candidates,
    )

    candidate = BridgeCandidate(
        token="resolve_target",
        kind="function_name",
        score=4.0,
        distinct_tools=2,
        occurrences=2,
        positive_evidence=0,
        negative_evidence=0,
        recency=0.5,
    )

    target = "\n".join(
        [
            "a.py:1:ordinary routing setup",
            "a.py:2:target = resolve_target(config)",
            "a.py:3:ordinary routing cleanup",
        ]
    )

    selected = select_target_bridge_candidates(
        [candidate],
        target_content=target,
        user_context="diagnose routing",
    )

    assert [item.token for item in selected] == ["resolve_target"]


def test_v2_rejects_plain_uppercase_acronym_without_causal_evidence() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        select_target_bridge_candidates,
    )

    candidate = BridgeCandidate(
        token="API",
        kind="config_key",
        score=4.0,
        distinct_tools=2,
        occurrences=4,
        positive_evidence=0,
        negative_evidence=0,
        recency=0.5,
    )

    target = "\n".join(
        [
            "pyproject.toml:1:ordinary dependency",
            "pyproject.toml:2:OpenAI API format support",
            "pyproject.toml:3:ordinary dependency",
        ]
    )

    selected = select_target_bridge_candidates(
        [candidate],
        target_content=target,
        user_context="diagnose routing",
    )

    assert selected == []


def test_v2_keeps_structured_config_key_from_cross_tool_evidence() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        select_target_bridge_candidates,
    )

    candidate = BridgeCandidate(
        token="OPENAI_TARGET_API_URL",
        kind="config_key",
        score=4.0,
        distinct_tools=2,
        occurrences=2,
        positive_evidence=0,
        negative_evidence=0,
        recency=0.5,
    )

    target = "\n".join(
        [
            "server.py:1:ordinary config",
            "server.py:2:url = OPENAI_TARGET_API_URL",
            "server.py:3:ordinary config",
        ]
    )

    selected = select_target_bridge_candidates(
        [candidate],
        target_content=target,
        user_context="diagnose routing",
    )

    assert [item.token for item in selected] == ["OPENAI_TARGET_API_URL"]


def test_v2_causal_evidence_can_rescue_plain_function_name() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        select_target_bridge_candidates,
    )

    candidate = BridgeCandidate(
        token="startswith",
        kind="function_name",
        score=5.0,
        distinct_tools=1,
        occurrences=1,
        positive_evidence=7,
        negative_evidence=0,
        recency=0.5,
    )

    target = "\n".join(
        [
            "a.py:1:ordinary routing setup",
            'a.py:2:value.startswith("copilot")',
            "a.py:3:ordinary routing cleanup",
        ]
    )

    selected = select_target_bridge_candidates(
        [candidate],
        target_content=target,
        user_context="diagnose routing",
    )

    assert [item.token for item in selected] == ["startswith"]


def test_v2_keeps_long_plain_uppercase_config_key() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        select_target_bridge_candidates,
    )

    candidate = BridgeCandidate(
        token="PORT",
        kind="config_key",
        score=4.0,
        distinct_tools=2,
        occurrences=2,
        positive_evidence=0,
        negative_evidence=0,
        recency=0.5,
    )

    target = "\n".join(
        [
            "config.py:1:ordinary config",
            "config.py:2:PORT = 8787",
            "config.py:3:ordinary config",
        ]
    )

    selected = select_target_bridge_candidates(
        [candidate],
        target_content=target,
        user_context="diagnose configuration",
    )

    assert [item.token for item in selected] == ["PORT"]


def test_v2_causal_evidence_can_rescue_short_config_acronym() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        select_target_bridge_candidates,
    )

    candidate = BridgeCandidate(
        token="API",
        kind="config_key",
        score=5.0,
        distinct_tools=1,
        occurrences=1,
        positive_evidence=8,
        negative_evidence=0,
        recency=0.5,
    )

    target = "\n".join(
        [
            "config.py:1:ordinary routing",
            "config.py:2:API selects the affected upstream",
            "config.py:3:ordinary routing",
        ]
    )

    selected = select_target_bridge_candidates(
        [candidate],
        target_content=target,
        user_context="diagnose routing",
    )

    assert [item.token for item in selected] == ["API"]


def test_batched_target_document_frequencies_preserve_scalar_overlap_semantics() -> None:
    from headroom.trajectory_relevance import (
        BridgeCandidate,
        _candidate_match_pattern,
        _normalized_candidate_text,
        _target_document_frequencies,
    )

    candidates = [
        BridgeCandidate(
            token="src/a.py",
            kind="file_path",
            score=4.0,
            distinct_tools=2,
            occurrences=2,
            positive_evidence=0,
            negative_evidence=0,
            recency=0.5,
        ),
        BridgeCandidate(
            token="pkg/src/a.py",
            kind="file_path",
            score=4.0,
            distinct_tools=2,
            occurrences=2,
            positive_evidence=0,
            negative_evidence=0,
            recency=0.5,
        ),
        BridgeCandidate(
            token="req-0184",
            kind="request_id",
            score=4.0,
            distinct_tools=2,
            occurrences=2,
            positive_evidence=0,
            negative_evidence=0,
            recency=0.5,
        ),
    ]

    target = "\n".join(
        [
            "pkg/src/a.py:10:req-0184 failed here",
            "src/a.py:20:ordinary diagnostic",
            "",
            r"pkg\src\a.py:30:Windows-style path",
            "other/module.py:40:req-0184 ordinary evidence",
        ]
    )

    n_lines, batched = _target_document_frequencies(
        candidates,
        target,
    )

    assert n_lines == 4

    for candidate in candidates:
        lines = [
            _normalized_candidate_text(
                line,
                candidate.kind,
            )
            for line in target.splitlines()
            if line.strip()
        ]

        pattern = _candidate_match_pattern(
            candidate.token,
            kind=candidate.kind,
        )

        scalar_df = sum(
            1
            for line in lines
            if pattern.search(line) is not None
        )

        assert batched[candidate] == scalar_df
