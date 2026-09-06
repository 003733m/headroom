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



def test_v3_single_prior_test_name_can_be_target_corroborated():
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    token = "test_count_messages_tolerates_null_tool_calls"

    prior = (
        "tests/test_providers/test_anthropic.py:205: "
        f"def {token}(self, anthropic_provider):"
    )

    target = "\n".join(
        [
            *[
                f"tests/test_misc.py:{i}: unrelated result {i}"
                for i in range(1, 60)
            ],
            (
                "tests/test_providers/test_anthropic.py:205: "
                f"def {token}(self, anthropic_provider):"
            ),
        ]
    )

    messages = [
        {
            "role": "user",
            "content": "fix Anthropic token estimation crash",
        },
        {
            "role": "tool",
            "content": prior,
        },
        {
            "role": "tool",
            "content": target,
        },
    ]

    context = build_search_relevance_context(
        messages,
        before_index=2,
        user_context="fix Anthropic token estimation crash",
        target_content=target,
    )

    assert "Trajectory bridge identifiers:" in context
    assert token in context


def test_v3_single_prior_file_path_does_not_get_singleton_corroboration():
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    token = "headroom/providers/anthropic.py"

    messages = [
        {
            "role": "user",
            "content": "investigate provider crash",
        },
        {
            "role": "tool",
            "content": (
                "headroom/providers/anthropic.py:486: "
                'for tool_call in message.get("tool_calls", []):'
            ),
        },
        {
            "role": "tool",
            "content": "\n".join(
                [
                    *[
                        f"headroom/other_{i}.py:{i}: unrelated"
                        for i in range(1, 50)
                    ],
                    (
                        "headroom/providers/anthropic.py:486: "
                        'for tool_call in message.get("tool_calls", []):'
                    ),
                ]
            ),
        },
    ]

    context = build_search_relevance_context(
        messages,
        before_index=2,
        user_context="investigate provider crash",
        target_content=messages[2]["content"],
    )

    assert token not in context
    assert context == "investigate provider crash"


def test_v3_target_only_identifier_cannot_self_seed():
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    target_only = "test_target_only_regression"

    target = "\n".join(
        [
            *[
                f"tests/test_misc.py:{i}: unrelated result {i}"
                for i in range(1, 50)
            ],
            f"tests/test_target.py:99: def {target_only}():",
        ]
    )

    messages = [
        {
            "role": "user",
            "content": "investigate regression",
        },
        {
            "role": "tool",
            "content": (
                "tests/test_prior.py:10: "
                "def test_unrelated_prior_identifier():"
            ),
        },
        {
            "role": "tool",
            "content": target,
        },
    ]

    context = build_search_relevance_context(
        messages,
        before_index=2,
        user_context="investigate regression",
        target_content=target,
    )

    assert context == "investigate regression"
    assert target_only not in context


def test_v3_target_corroboration_does_not_relax_function_name():
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    messages = [
        {
            "role": "user",
            "content": "investigate parser failure",
        },
        {
            "role": "tool",
            "content": "src/parser.py:20: parse_payload(raw)",
        },
        {
            "role": "tool",
            "content": "\n".join(
                [
                    *[
                        f"src/module_{i}.py:{i}: unrelated_call(value)"
                        for i in range(1, 50)
                    ],
                    "src/parser.py:100: parse_payload(value)",
                ]
            ),
        },
    ]

    context = build_search_relevance_context(
        messages,
        before_index=2,
        user_context="investigate parser failure",
        target_content=messages[2]["content"],
    )

    # parse_payload is a function_name, not one of the structured
    # singleton kinds allowed to use V3 current-target corroboration.
    assert "parse_payload" not in context
    assert "src/parser.py" not in context
    assert context == "investigate parser failure"


def test_v3_default_v1_ranking_still_rejects_structured_singleton():
    from headroom.trajectory_relevance import (
        rank_bridge_candidates,
    )

    token = "test_count_messages_tolerates_null_tool_calls"

    ranked = rank_bridge_candidates(
        [
            (
                "tests/test_providers/test_anthropic.py:205: "
                f"def {token}(self, anthropic_provider):"
            )
        ]
    )

    assert all(candidate.token != token for candidate in ranked)


def test_v3_bash_search_intent_reaches_trajectory_on_lossy_fallthrough(
    monkeypatch,
):
    import json

    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
        RouterCompressionResult,
    )

    class Tokenizer:
        @staticmethod
        def count_text(text: str) -> int:
            return max(1, len(str(text).split()))

    target = "\n".join(
        [
            *[
                (
                    f"src/module_{i:04d}.py:{i}: "
                    f"unrelated diagnostic req-{10000 + i}"
                )
                for i in range(1, 100)
            ],
            "src/target_module.py:999: diagnostic req-0184",
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
            "role": "tool",
            "content": "Follow-up confirms req-0184",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "search-call",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {
                                "command": (
                                    "rg -nH req-0184 src/"
                                )
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "search-call",
            "content": target,
        },
    ]

    router = ContentRouter()

    # Force the exact V3 case: Bash search intent is known, but the
    # byte-lossless fold cannot shrink this result and falls through.
    monkeypatch.setattr(
        router,
        "_bash_search_fold",
        lambda *_args, **_kwargs: None,
    )

    # Reproduce the prospective failure mode in which repository-search
    # output is structurally classified as code rather than SEARCH.
    monkeypatch.setattr(
        router,
        "_strategy_from_detection_type",
        lambda _content_type: CompressionStrategy.CODE_AWARE,
    )

    seen = {}

    def fake_compress(
        content: str,
        context: str = "",
        question=None,
        bias: float = 1.0,
        precomputed_detection=None,
    ):
        if content == target:
            seen["context"] = context

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
        trajectory_relevance=True,
        protect_recent=0,
        min_tokens_to_compress=10,
    )

    assert "target" not in seen or "context" in seen
    assert seen["context"] != "BASE USER QUERY"
    assert "req-0184" in seen["context"]


def test_v3_nonsearch_bash_does_not_gain_search_intent(
    monkeypatch,
):
    import json

    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
        RouterCompressionResult,
    )

    class Tokenizer:
        @staticmethod
        def count_text(text: str) -> int:
            return max(1, len(str(text).split()))

    target = "\n".join(
        [
            *[
                (
                    f"src/module_{i:04d}.py:{i}: "
                    f"unrelated diagnostic req-{10000 + i}"
                )
                for i in range(1, 100)
            ],
            "src/target_module.py:999: diagnostic req-0184",
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
            "role": "tool",
            "content": "Follow-up confirms req-0184",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "pytest-call",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "pytest tests/ -q"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "pytest-call",
            "content": target,
        },
    ]

    router = ContentRouter()

    monkeypatch.setattr(
        router,
        "_bash_search_fold",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        router,
        "_strategy_from_detection_type",
        lambda _content_type: CompressionStrategy.CODE_AWARE,
    )

    seen = {}

    def fake_compress(
        content: str,
        context: str = "",
        question=None,
        bias: float = 1.0,
        precomputed_detection=None,
    ):
        if content == target:
            seen["context"] = context

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
        trajectory_relevance=True,
        protect_recent=0,
        min_tokens_to_compress=10,
    )

    assert seen["context"] == "BASE USER QUERY"



def test_v3_repeated_single_output_paths_do_not_crowd_out_corroborated_test_name():
    from headroom.trajectory_relevance import (
        build_search_relevance_context,
    )

    critical = "test_count_messages_tolerates_null_tool_calls"

    noisy_prior_lines = [
        (
            "tests/test_integrations/langchain/test_chat_model.py:"
            f"{i}: repeated unrelated path"
        )
        for i in range(1, 13)
    ]

    messages = [
        {
            "role": "user",
            "content": "fix Anthropic token estimation crash",
        },
        {
            "role": "tool",
            "content": "\n".join(
                [
                    *noisy_prior_lines,
                    (
                        "tests/test_providers/test_anthropic.py:205: "
                        f"def {critical}(self, provider):"
                    ),
                ]
            ),
        },
        {
            "role": "tool",
            "content": "\n".join(
                [
                    *[
                        f"tests/test_misc_{i}.py:{i}: unrelated result"
                        for i in range(1, 80)
                    ],
                    (
                        "tests/test_integrations/langchain/"
                        "test_chat_model.py:200: unrelated result"
                    ),
                    (
                        "tests/test_providers/test_anthropic.py:205: "
                        f"def {critical}(self, provider):"
                    ),
                ]
            ),
        },
    ]

    context = build_search_relevance_context(
        messages,
        before_index=2,
        user_context="fix Anthropic token estimation crash",
        target_content=messages[2]["content"],
        top_k=1,
    )

    assert critical in context
    assert (
        "tests/test_integrations/langchain/test_chat_model.py"
        not in context
    )


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


def test_structural_search_uses_trajectory_relevance_without_changing_primary_strategy(
    monkeypatch,
):
    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
    )

    router = ContentRouter()
    router.config.relevance_split = True
    router._lossless_then_lossy = False

    monkeypatch.setattr(
        router,
        "_lossless_first",
        lambda content, strategy: ("lossless folded", "lossless_search"),
    )

    calls: list[tuple[str, str]] = []

    def fake_relevance_split(content: str, kind: str, context: str):
        calls.append((kind, context))
        return "relevance selected"

    monkeypatch.setattr(
        router,
        "_relevance_split_compress",
        fake_relevance_split,
    )

    off = router._compress_pure(
        "original search-like output",
        CompressionStrategy.CODE_AWARE,
        "req-0184",
        trajectory_search_relevance=False,
    )

    assert off.strategy_used is CompressionStrategy.CODE_AWARE
    assert off.compressed == "lossless folded"
    assert calls == []

    on = router._compress_pure(
        "original search-like output",
        CompressionStrategy.CODE_AWARE,
        "req-0184",
        trajectory_search_relevance=True,
    )

    assert on.strategy_used is CompressionStrategy.CODE_AWARE
    assert on.compressed == "relevance selected"
    assert on.strategy_chain == ["search", "relevance_split"]
    assert calls == [("search", "req-0184")]


def test_compression_unit_forwards_trajectory_search_relevance_metadata():
    from headroom.transforms.compression_units import (
        CompressionUnit,
        compress_unit_with_router,
    )
    from headroom.transforms.content_router import (
        CompressionStrategy,
        RouterCompressionResult,
    )

    class Tokenizer:
        @staticmethod
        def count_text(text: str) -> int:
            return len(text.split())

    class CapturingRouter:
        def __init__(self):
            self.kwargs = {}

        def compress(self, content: str, **kwargs):
            self.kwargs = kwargs
            return RouterCompressionResult(
                compressed="short result",
                original=content,
                strategy_used=CompressionStrategy.SEARCH,
            )

    router = CapturingRouter()

    unit = CompressionUnit(
        text=("diagnostic output " * 100),
        provider="openai",
        endpoint="responses",
        role="tool",
        item_type="function_call_output",
        min_bytes=1,
        context="req-0184",
        metadata={"trajectory_search_relevance": "true"},
    )

    result = compress_unit_with_router(
        unit,
        router=router,  # type: ignore[arg-type]
        tokenizer=Tokenizer(),
    )

    assert result.modified
    assert router.kwargs["context"] == "req-0184"
    assert router.kwargs["trajectory_search_relevance"] is True


def test_v3_responses_bash_rg_intent_is_call_id_scoped():
    import json
    from types import SimpleNamespace

    from headroom.proxy.handlers.openai import (
        _responses_bash_search_call_ids,
    )

    router = SimpleNamespace(
        config=SimpleNamespace(
            bash_tool_names=frozenset(
                {"bash", "shell", "local_shell"}
            ),
            bash_search_commands=frozenset(
                {
                    "grep",
                    "egrep",
                    "fgrep",
                    "rg",
                    "ripgrep",
                    "ag",
                    "ack",
                }
            ),
        )
    )

    items = [
        {
            "type": "function_call",
            "call_id": "call-search",
            "name": "bash",
            "arguments": json.dumps(
                {
                    "command": (
                        "rg -nH "
                        "'anthropic_api_url|openai_api_url' "
                        "headroom tests"
                    )
                }
            ),
        },
        {
            "type": "function_call_output",
            "call_id": "call-search",
            "output": (
                "def resolve_target(config):\n"
                "    return config.openai_api_url\n"
            ),
        },
        {
            "type": "function_call_output",
            "call_id": "call-other",
            "output": (
                "def unrelated():\n"
                "    return None\n"
            ),
        },
    ]

    search_call_ids = _responses_bash_search_call_ids(
        router,
        items,
        {"call-search": "bash"},
    )

    assert search_call_ids == {"call-search"}
    assert items[1]["call_id"] in search_call_ids
    assert items[2]["call_id"] not in search_call_ids


def test_v3_responses_nonsearch_bash_does_not_gain_search_intent():
    import json
    from types import SimpleNamespace

    from headroom.proxy.handlers.openai import (
        _responses_bash_search_call_ids,
    )

    router = SimpleNamespace(
        config=SimpleNamespace(
            bash_tool_names=frozenset(
                {"bash", "shell", "local_shell"}
            ),
            bash_search_commands=frozenset(
                {
                    "grep",
                    "egrep",
                    "fgrep",
                    "rg",
                    "ripgrep",
                    "ag",
                    "ack",
                }
            ),
        )
    )

    items = [
        {
            "type": "function_call",
            "call_id": "call-pytest",
            "name": "bash",
            "arguments": json.dumps(
                {
                    "command": (
                        "pytest "
                        "tests/test_provider_registry.py"
                    )
                }
            ),
        },
        {
            "type": "function_call_output",
            "call_id": "call-pytest",
            "output": (
                "def test_example():\n"
                "    assert True\n"
            ),
        },
    ]

    search_call_ids = _responses_bash_search_call_ids(
        router,
        items,
        {"call-pytest": "bash"},
    )

    assert search_call_ids == set()


def test_v3_responses_native_local_shell_rg_has_search_intent():
    from types import SimpleNamespace

    from headroom.proxy.handlers.openai import (
        _responses_bash_search_call_ids,
    )

    router = SimpleNamespace(
        config=SimpleNamespace(
            bash_tool_names=frozenset(
                {"bash", "shell", "local_shell"}
            ),
            bash_search_commands=frozenset(
                {
                    "grep",
                    "egrep",
                    "fgrep",
                    "rg",
                    "ripgrep",
                    "ag",
                    "ack",
                }
            ),
        )
    )

    items = [
        {
            "type": "local_shell_call",
            "call_id": "call-shell-search",
            "action": {
                "type": "exec",
                "command": "rg -nH req-0184 headroom tests",
            },
        },
        {
            "type": "local_shell_call_output",
            "call_id": "call-shell-search",
            "output": (
                "headroom/example.py:10:"
                "def handle_req_0184():\n"
            ),
        },
    ]

    search_call_ids = _responses_bash_search_call_ids(
        router,
        items,
        {},
    )

    assert search_call_ids == {"call-shell-search"}


def test_mixed_content_forwards_trajectory_search_relevance_to_sections(
    monkeypatch,
):
    import headroom.transforms.content_router as content_router_module
    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
    )

    router = ContentRouter()
    router.config.relevance_split = True
    router._lossless_then_lossy = False

    # Force this regression test through the exact top-level branch that
    # failed on the naturally observed U01 Responses search output.
    monkeypatch.setattr(
        content_router_module,
        "is_mixed_content",
        lambda _content: True,
    )
    monkeypatch.setattr(
        router,
        "_determine_strategy",
        lambda _content, *, mixed, detection: CompressionStrategy.MIXED,
    )

    # Model a section whose structure is independently recognized as a
    # search result. The existing trajectory override is intentionally
    # conditional on this lossless structural signal.
    monkeypatch.setattr(
        router,
        "_lossless_first",
        lambda content, strategy: (
            content,
            "lossless_search",
        ),
    )

    calls: list[tuple[str, str]] = []

    def fake_relevance_split(
        content: str,
        kind: str,
        context: str,
    ):
        calls.append((kind, context))
        return "relevance selected"

    monkeypatch.setattr(
        router,
        "_relevance_split_compress",
        fake_relevance_split,
    )

    context = (
        "Trajectory bridge identifiers: COPILOT_PROVIDER_TYPE\n"
        "Adopted bridge identifiers: COPILOT_PROVIDER_TYPE"
    )

    off = router.compress(
        "search section one\n\nsearch section two",
        context=context,
        trajectory_search_relevance=False,
    )

    assert off.strategy_used is CompressionStrategy.MIXED
    assert calls == []

    on = router.compress(
        "search section one\n\nsearch section two",
        context=context,
        trajectory_search_relevance=True,
    )

    assert on.strategy_used is CompressionStrategy.MIXED
    assert calls
    assert all(kind == "search" for kind, _ in calls)
    assert all(context in ctx for _, ctx in calls)
