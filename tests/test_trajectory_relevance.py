from headroom.trajectory_relevance import (
    build_search_relevance_context,
    extract_prior_tool_outputs,
    rank_bridge_candidates,
)


def test_extracts_openai_tool_results_before_target_only():
    messages = [
        {"role": "user", "content": "fix checkout"},
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": "Observed req-0184",
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "content": "Follow-up req-0184",
        },
        {
            "role": "tool",
            "tool_call_id": "target",
            "content": "TARGET contains req-9999",
        },
    ]

    outputs = extract_prior_tool_outputs(
        messages,
        before_index=3,
    )

    joined = "\n".join(outputs)

    assert "req-0184" in joined
    assert "req-9999" not in joined


def test_extracts_anthropic_tool_result():
    messages = [
        {"role": "user", "content": "fix checkout"},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "Root cause: req-0184",
                }
            ],
        },
        {"role": "assistant", "content": "continue"},
    ]

    outputs = extract_prior_tool_outputs(
        messages,
        before_index=2,
    )

    assert outputs == ["Root cause: req-0184"]


def test_distinct_tool_bridge_beats_same_tool_repetition():
    outputs = [
        "\n".join(
            [
                "Observed req-0184",
                "Observed req-0291",
                "Observed req-0291",
            ]
        ),
        "Observed req-0184",
    ]

    ranked = rank_bridge_candidates(
        outputs,
        top_k=6,
    )

    assert ranked
    assert ranked[0].token == "req-0184"


def test_root_cause_beats_false_recurring_suspicion():
    outputs = [
        "\n".join(
            [
                "Initial suspicion req-0291",
                "Root cause identified: req-0184",
            ]
        ),
        "Follow-up suspicion req-0291",
    ]

    ranked = rank_bridge_candidates(
        outputs,
        top_k=6,
    )

    tokens = [
        candidate.token
        for candidate in ranked
    ]

    assert tokens[0] == "req-0184"
    # The calibrated confidence threshold should now reject the stale
    # speculative candidate rather than merely rank it below the root cause.
    assert "req-0291" not in tokens


def test_hypothesis_shift_prefers_new_root_cause():
    outputs = [
        "Initial suspicion req-0291",
        "Follow-up hypothesis req-0291",
        "Root cause identified: req-0184",
    ]

    ranked = rank_bridge_candidates(
        outputs,
        top_k=6,
    )

    assert ranked[0].token == "req-0184"


def test_weak_single_tool_identifier_abstains():
    outputs = [
        "Routine observation req-0184",
    ]

    ranked = rank_bridge_candidates(
        outputs,
        top_k=6,
    )

    assert ranked == []


def test_build_context_does_not_read_target_message():
    messages = [
        {"role": "user", "content": "fix checkout"},
        {
            "role": "tool",
            "content": "Observed req-0184",
        },
        {
            "role": "tool",
            "content": "Follow-up req-0184",
        },
        {
            "role": "tool",
            "content": (
                "large search target containing req-9999"
            ),
        },
    ]

    context = build_search_relevance_context(
        messages,
        before_index=3,
        user_context="fix checkout",
    )

    assert "req-0184" in context
    assert "req-9999" not in context


def test_no_bridge_keeps_user_context_byte_identical():
    messages = [
        {"role": "user", "content": "fix checkout"},
        {
            "role": "tool",
            "content": "Routine observation req-0184",
        },
    ]

    context = build_search_relevance_context(
        messages,
        before_index=2,
        user_context="fix checkout",
    )

    assert context == "fix checkout"


def test_search_context_is_enriched_only_when_enabled():
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    messages = [
        {
            "role": "user",
            "content": "investigate checkout failure",
        },
        {
            "role": "tool",
            "content": "Observed request req-0184",
        },
        {
            "role": "assistant",
            "content": "continue investigation",
        },
        {
            "role": "tool",
            "content": "Confirmed affected request req-0184",
        },
        {
            "role": "tool",
            "content": "TARGET req-9999",
        },
    ]

    base = "investigate checkout failure"

    enriched = build_search_relevance_context(
        messages,
        before_index=4,
        user_context=base,
    )

    assert enriched != base
    assert "req-0184" in enriched

    # The target message must not leak into its own context.
    assert "req-9999" not in enriched


def test_nonsearch_baseline_context_stays_unchanged():
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    base = "fix checkout"

    messages = [
        {"role": "user", "content": base},
        {
            "role": "tool",
            "content": "Routine observation req-0100",
        },
    ]

    # No qualifying bridge means exact fallback.
    assert (
        build_search_relevance_context(
            messages,
            before_index=2,
            user_context=base,
        )
        == base
    )


def test_router_enriches_context_for_search_target_only():
    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
        RouterCompressionResult,
    )

    class Tokenizer:
        def count_text(self, text: str) -> int:
            return max(1, len(str(text).split()))

    # Use a grep/ripgrep-like multi-file result set. Keeping every match in
    # one file can legitimately classify as log/code as detector heuristics
    # evolve; this test is specifically about the SEARCH route.
    target = "\n".join(
        [
            *[
                (
                    f"src/module_{i:04d}.py:{i}: "
                    f"worker trace_ref='req-{i:04d}'"
                )
                for i in range(1, 501)
            ],
            (
                "src/target_module.py:9999: "
                "target-only trace_ref='req-9999'"
            ),
        ]
    )

    messages = [
        {
            "role": "user",
            "content": "investigate checkout failure",
        },
        {
            "role": "tool",
            "content": "Observed affected request req-0184",
        },
        {
            "role": "assistant",
            "content": "continue",
        },
        {
            "role": "tool",
            "content": "Follow-up confirms req-0184",
        },
        {
            "role": "assistant",
            "content": "search runtime",
        },
        {
            "role": "tool",
            "content": target,
        },
    ]

    def captured_context(enabled: bool) -> str:
        router = ContentRouter()

        # This test exercises trajectory-context wiring, not content-detection
        # heuristics. Force the already-detected target through the SEARCH
        # strategy so detector changes cannot make this wiring test flaky.
        router._strategy_from_detection_type = (  # type: ignore[method-assign]
            lambda _content_type: CompressionStrategy.SEARCH
        )

        seen: dict[str, str] = {}

        def fake_compress(
            content: str,
            context: str = "",
            question=None,
            bias: float = 1.0,
            precomputed_detection=None,
        ):
            if content == target:
                seen["target_context"] = context

            return RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
            )

        router.compress = fake_compress  # type: ignore[method-assign]

        router.apply(
            messages,
            Tokenizer(),
            context="BASE USER QUERY",
            trajectory_relevance=enabled,
            protect_recent=0,
            min_tokens_to_compress=10,
        )

        return seen["target_context"]

    off_context = captured_context(False)
    on_context = captured_context(True)

    assert off_context == "BASE USER QUERY"

    assert on_context != off_context
    assert "BASE USER QUERY" in on_context
    assert "req-0184" in on_context

    # Target message cannot contribute to its own relevance context.
    assert "req-9999" not in on_context


def test_custom_scoring_threshold_can_abstain():
    from headroom.trajectory_relevance import (
        BridgeScoringConfig,
        rank_bridge_candidates,
    )

    outputs = [
        "Observed affected request req-0184",
        "Follow-up confirms req-0184",
    ]

    normal = rank_bridge_candidates(outputs)

    assert normal
    assert normal[0].token == "req-0184"

    strict = rank_bridge_candidates(
        outputs,
        scoring=BridgeScoringConfig(
            min_score=1000.0,
        ),
    )

    assert strict == []


def test_responses_bridge_uses_prior_outputs_only():
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
                    "text": "investigate checkout failure",
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Observed affected request req-0184",
        },
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": "Follow-up confirms req-0184",
        },
        {
            "type": "function_call_output",
            "call_id": "call-target",
            "output": "TARGET req-9999",
        },
    ]

    context = build_responses_search_relevance_context(
        items,
        before_index=3,
    )

    # Responses had no relevance context before this feature; the opt-in
    # should add only reliable bridge evidence, not silently introduce a new
    # user-query signal.
    assert "investigate checkout failure" not in context
    assert "req-0184" in context
    assert "req-9999" not in context


def test_responses_bridge_abstains_without_strong_bridge():
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
                    "text": "investigate checkout failure",
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Maybe request req-0184 is relevant",
        },
    ]

    context = build_responses_search_relevance_context(
        items,
        before_index=2,
        user_context="BASE USER QUERY",
    )

    assert context == "BASE USER QUERY"


def test_responses_bridge_empty_baseline_stays_empty_on_abstention():
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
                    "text": "investigate checkout failure",
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Maybe request req-0184 is relevant",
        },
    ]

    context = build_responses_search_relevance_context(
        items,
        before_index=2,
        user_context="",
    )

    assert context == ""


def test_responses_search_eligibility_accepts_grep_over_source_code():
    from headroom.proxy.handlers.openai import (
        _responses_trajectory_search_eligible,
    )
    from headroom.transforms.content_router import CompressionStrategy

    class CodeAwareRouter:
        @staticmethod
        def _strategy_from_detection_type(_content_type):
            # Reproduce the real OpenCode case: the primary classifier routes
            # grep output containing Python paths as source code.
            return CompressionStrategy.CODE_AWARE

    search_text = "\n".join(
        f"src/module_{i:05d}.py:{i}: worker diagnostic trace_ref='req-{10000 + i}'"
        for i in range(1, 51)
    )

    assert _responses_trajectory_search_eligible(
        CodeAwareRouter(),
        search_text,
    )

    plain_code = "def calculate_total(x: int) -> int:\n    return x + 1\n"

    assert not _responses_trajectory_search_eligible(
        CodeAwareRouter(),
        plain_code,
    )
