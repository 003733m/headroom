#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

CONTROL = Path(__file__).resolve().parents[3]

RUNTIME = Path(
    "/tmp/headroom-second-unseen-adopted-runtime-89da0898"
)

RAW = (
    Path.home()
    / "headroom-second-unseen-adopted-20260905"
    / "results"
    / "U01-ON"
)

OUT = (
    CONTROL
    / "experiments/bridge_relevance/bridge_hard/"
      "second_unseen_adopted_bridge_replay_results.json"
)

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

# Observed prospectively in the single consumed U01-ON run.
# This is the first live gate event with an adopted bridge.
TARGET_CALL_ID = "call_NM0wmbNuSl4zUdOh8rhqsUR0"


def die(msg: str) -> None:
    raise SystemExit(msg)


def freeze_check() -> None:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=RUNTIME,
        text=True,
    ).strip()

    if sha != FREEZE:
        die(f"runtime mismatch: {sha} != {FREEZE}")

    print("Frozen runtime: OK", sha)


# Force all imports to 89da.
sys.path.insert(0, str(RUNTIME))

from headroom.agent_savings import seed_proxy_env_defaults  # noqa: E402
from headroom.proxy.handlers.openai import OpenAIHandlerMixin  # noqa: E402
from headroom.transforms.content_router import ContentRouter  # noqa: E402
import headroom.trajectory_relevance as tr  # noqa: E402
import headroom  # noqa: E402


if not str(Path(headroom.__file__).resolve()).startswith(str(RUNTIME)):
    die(f"wrong Headroom tree: {headroom.__file__}")


class ProductionLikeTokenCounter:
    def __init__(self, model: str):
        self.model = model
        self.encoding = None
        self.mode = "fallback:utf8_div4"

        try:
            import tiktoken

            try:
                self.encoding = tiktoken.encoding_for_model(model)
                self.mode = f"tiktoken:model:{model}"
            except Exception:
                self.encoding = tiktoken.get_encoding("o200k_base")
                self.mode = "tiktoken:o200k_base"
        except Exception:
            pass

    def count_text(self, text: str) -> int:
        if self.encoding is not None:
            return max(1, len(self.encoding.encode(str(text))))
        return max(
            1,
            len(str(text).encode("utf-8", errors="replace")) // 4,
        )


class Provider:
    def __init__(self, model: str):
        self.counter = ProductionLikeTokenCounter(model)

    def get_token_counter(self, _model: str):
        return self.counter


def make_handler(model: str) -> OpenAIHandlerMixin:
    # Both A and D keep trajectory relevance ON.
    os.environ["HEADROOM_SAVINGS_PROFILE"] = "coding"
    os.environ["HEADROOM_TRAJECTORY_RELEVANCE"] = "1"

    seed_proxy_env_defaults(os.environ)

    router = ContentRouter()

    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[router])
    handler.openai_provider = Provider(model)
    handler.config = SimpleNamespace(savings_profile="coding")

    try:
        lock, cache = handler._openai_responses_unit_cache()
        with lock:
            cache.clear()
    except Exception:
        pass

    return handler


def load_body(path: Path) -> dict:
    d = json.loads(path.read_text(errors="replace"))
    body = d.get("body")

    if isinstance(body, str):
        body = json.loads(body)

    if not isinstance(body, dict):
        die(f"bad body: {path}")

    return body


def output_for_call(payload: dict, call_id: str) -> str | None:
    items = payload.get("input")

    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            continue

        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
        ):
            value = item.get("output")

            if isinstance(value, str):
                return value

            return tr._responses_text(value)

    return None


def find_target_request() -> tuple[Path, dict]:
    matches = []

    for p in sorted(
        (RAW / "codex-wire").glob(
            "*_http_inbound_request.json"
        )
    ):
        body = load_body(p)

        if output_for_call(body, TARGET_CALL_ID) is not None:
            matches.append((p, body))

    if not matches:
        die("target call_id not found in inbound wire")

    # First exposure only.
    return matches[0]


def sha(text: str | None) -> str | None:
    if text is None:
        return None

    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


ORIGINAL_ADOPTED = tr._responses_adopted_bridge_identifiers


def replay(
    payload: dict,
    *,
    model: str,
    adopted_enabled: bool,
):
    if adopted_enabled:
        tr._responses_adopted_bridge_identifiers = ORIGINAL_ADOPTED
    else:
        # Surgical counterfactual:
        # retain the complete V3 trajectory context but disable only
        # the adopted preservation intersection.
        tr._responses_adopted_bridge_identifiers = (
            lambda *args, **kwargs: ()
        )

    try:
        handler = make_handler(model)

        return handler._compress_openai_responses_payload(
            copy.deepcopy(payload),
            model=model,
            request_id=(
                "second_unseen_adopted_D"
                if adopted_enabled
                else "second_unseen_adopted_A"
            ),
            timing={},
            client="opencode",
            savings_tags={},
        )
    finally:
        tr._responses_adopted_bridge_identifiers = ORIGINAL_ADOPTED


def summarize_result(result, target_call_id: str) -> dict:
    payload = result[0]
    target = output_for_call(payload, target_call_id)

    return {
        "modified": bool(result[1]),
        "tokens_saved": int(result[2]),
        "transforms": list(result[3]),
        "reason": result[4],
        "input_bytes": int(result[5]),
        "output_bytes": int(result[6]),
        "attempted_input_tokens": int(result[7]),
        "target_present": target is not None,
        "target_bytes": (
            len(target.encode("utf-8", errors="replace"))
            if target is not None
            else None
        ),
        "target_sha256": sha(target),
        "contains_COPILOT_PROVIDER_TYPE": (
            "COPILOT_PROVIDER_TYPE" in target
            if target is not None
            else False
        ),
        "contains_HEADROOM_BACKEND": (
            "HEADROOM_BACKEND" in target
            if target is not None
            else False
        ),
    }


def main():
    freeze_check()

    path, payload = find_target_request()

    model = payload.get("model")
    if not isinstance(model, str):
        die("model missing")

    original_target = output_for_call(
        payload,
        TARGET_CALL_ID,
    )

    if original_target is None:
        die("original target missing")

    print("wire:", path.name)
    print(
        "original target bytes:",
        len(original_target.encode("utf-8", errors="replace")),
    )
    print("model:", model)

    A = replay(
        payload,
        model=model,
        adopted_enabled=False,
    )

    D = replay(
        payload,
        model=model,
        adopted_enabled=True,
    )

    a = summarize_result(A, TARGET_CALL_ID)
    d = summarize_result(D, TARGET_CALL_ID)

    result = {
        "experiment": "second_unseen_adopted_bridge_exact_replay",
        "classification": (
            "deterministic post-run mechanism replay on "
            "prospectively selected unseen trajectory"
        ),
        "production_freeze": FREEZE,
        "task": "U01",
        "source_condition": "U01-ON",
        "comparison": {
            "A": (
                "frozen 89da trajectory relevance with "
                "adopted preservation disabled only"
            ),
            "D": (
                "exact frozen 89da trajectory relevance "
                "with adopted preservation enabled"
            ),
        },
        "original_target_bytes": len(
            original_target.encode(
                "utf-8",
                errors="replace",
            )
        ),
        "original_target_sha256": sha(original_target),
        "A": a,
        "D": d,
        "A_D_target_equal": (
            a["target_sha256"] == d["target_sha256"]
        ),
        "A_D_output_bytes_delta": (
            d["output_bytes"] - a["output_bytes"]
        ),
        "A_D_tokens_saved_delta": (
            d["tokens_saved"] - a["tokens_saved"]
        ),
        "raw_wire_committed": False,
    }

    OUT.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print()
    print("=" * 72)
    print("EXACT A/D REPLAY")
    print("=" * 72)

    for label, x in (("A", a), ("D", d)):
        print()
        print(label)
        for key in (
            "modified",
            "tokens_saved",
            "transforms",
            "reason",
            "input_bytes",
            "output_bytes",
            "attempted_input_tokens",
            "target_bytes",
            "target_sha256",
            "contains_COPILOT_PROVIDER_TYPE",
            "contains_HEADROOM_BACKEND",
        ):
            print(f"  {key}: {x[key]}")

    print()
    print("A/D target equal:", result["A_D_target_equal"])
    print(
        "output byte delta D-A:",
        result["A_D_output_bytes_delta"],
    )
    print(
        "tokens-saved delta D-A:",
        result["A_D_tokens_saved_delta"],
    )
    print()
    print("result:", OUT)


if __name__ == "__main__":
    main()
