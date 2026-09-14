#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CONTROL = Path(__file__).resolve().parents[3]
RUNTIME = Path("/tmp/headroom-trajrel-heldout-pilot-runtime-89da0898")
RAW = Path.home() / "headroom-trajrel-heldout-pilot-v1-20260914"
EXHAUSTIVE = (
    CONTROL
    / "experiments/bridge_relevance/bridge_hard/exhaustive_discovery_results.json"
)
OUT = (
    CONTROL
    / "experiments/bridge_relevance/bridge_hard/exact_wire_v3_replay_results.json"
)

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

TASK_COMMITS = {
    "F1": "1f5fefffd3e82c73bddd928cfd53334031e807bc",
    "F2": "85e869945138f06471501046c5725eac119dea58",
    "F3": "4e2bbfee3f65e3287ab7693da70f0d7b20dada28",
    "F4": "0cddac632d8ab4e0bcd752b9c7519d5099aa085e",
    "F5": "cc072f0821b76618659a2dd6c81a1daee49dbaab",
    "F6": "fa330f3e2bed3f51b7f4d49bcd64ec4aaa17f44f",
    "F7": "c7b5a24b4ffca78bacf1919ace00087b6ab6f7d0",
    "F8": "d05802b6200b94f198e99319e6c778e78b53db8b",
}

PREKNOWN = {
    ("F1", "headroom/memory/traffic_learner.py", 1258),
    ("F3", "headroom/providers/opencode/config.py", 75),
    ("F7", "headroom/learn/_shared.py", 59),
}


def die(msg: str) -> None:
    raise SystemExit(msg)


def freeze_check() -> None:
    if not RUNTIME.is_dir():
        die(f"Frozen runtime missing: {RUNTIME}")

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=RUNTIME,
        text=True,
    ).strip()

    if sha != FREEZE:
        die(f"Frozen runtime mismatch: {sha} != {FREEZE}")

    print(f"Frozen runtime: OK ({sha})")


# Force imports to come from the frozen V3 worktree, not current control HEAD.
sys.path.insert(0, str(RUNTIME))

from headroom.agent_savings import seed_proxy_env_defaults  # noqa: E402
from headroom.proxy.handlers.openai import OpenAIHandlerMixin  # noqa: E402
from headroom.transforms.content_router import ContentRouter  # noqa: E402
import headroom  # noqa: E402

if not str(Path(headroom.__file__).resolve()).startswith(str(RUNTIME)):
    die(f"headroom imported from wrong tree: {headroom.__file__}")


class ProductionLikeTokenCounter:
    """
    Offline tokenizer adapter.

    Prefer tiktoken with the model encoding when known and o200k_base
    otherwise. The replay validity gate is the captured OFF decision:
    records for which reconstructed OFF does not reproduce captured
    KEEP/DROP are excluded from counterfactual claims.
    """

    def __init__(self, model: str):
        self.model = model
        self.mode = "unknown"
        self.encoding = None

        try:
            import tiktoken

            try:
                self.encoding = tiktoken.encoding_for_model(model)
                self.mode = f"tiktoken:model:{model}"
            except Exception:
                self.encoding = tiktoken.get_encoding("o200k_base")
                self.mode = "tiktoken:o200k_base"
        except Exception:
            self.mode = "fallback:utf8_div4"

    def count_text(self, text: str) -> int:
        if self.encoding is not None:
            return max(1, len(self.encoding.encode(str(text))))
        return max(1, len(str(text).encode("utf-8")) // 4)


class Provider:
    def __init__(self, model: str):
        self.counter = ProductionLikeTokenCounter(model)

    def get_token_counter(self, _model: str):
        return self.counter


def clear_unit_cache(handler: OpenAIHandlerMixin) -> None:
    try:
        lock, cache = handler._openai_responses_unit_cache()
        with lock:
            cache.clear()
    except Exception:
        pass


def make_handler(flag: bool, model: str) -> OpenAIHandlerMixin:
    # Match the benchmark proxy posture:
    # same coding profile; treatment differs only in trajectory relevance.
    os.environ["HEADROOM_SAVINGS_PROFILE"] = "coding"
    os.environ["HEADROOM_TRAJECTORY_RELEVANCE"] = "1" if flag else "0"

    seed_proxy_env_defaults(os.environ)

    router = ContentRouter()

    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[router])
    handler.openai_provider = Provider(model)
    handler.config = SimpleNamespace(
        savings_profile="coding",
    )

    clear_unit_cache(handler)
    return handler


def find_candidate(x: Any, sha: str) -> dict[str, Any] | None:
    if isinstance(x, dict):
        if (
            x.get("source_commit") == sha
            and x.get("status") == "evaluated"
            and "critical_decisions" in x
        ):
            return x
        for v in x.values():
            r = find_candidate(v, sha)
            if r is not None:
                return r

    elif isinstance(x, list):
        for v in x:
            r = find_candidate(v, sha)
            if r is not None:
                return r

    return None


def variants(path: str, line: int, text: str) -> set[str]:
    return {
        f"{path}:{line}:{text}",
        f"./{path}:{line}:{text}",
        f"{path}-{line}-{text}",
        f"./{path}-{line}-{text}",
    }


def has(value: Any, vs: set[str]) -> bool:
    if isinstance(value, str):
        s = value
    else:
        s = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return any(v in s for v in vs)


def sha_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def output_by_call_id(payload: dict[str, Any], call_id: str | None) -> str | None:
    if not call_id:
        return None

    items = payload.get("input")
    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            continue
        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
            and isinstance(item.get("output"), str)
        ):
            return item["output"]

    return None


def call_id_containing(
    payload: dict[str, Any],
    vs: set[str],
) -> str | None:
    items = payload.get("input")
    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call_output":
            continue
        out = item.get("output")
        if isinstance(out, str) and has(out, vs):
            cid = item.get("call_id")
            return cid if isinstance(cid, str) else None

    return None


def load_wire(task: str) -> list[dict[str, Any]]:
    wire = RAW / "results" / f"{task}-OFF" / "wire"
    by_req: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for p in wire.glob("*.json"):
        d = json.loads(p.read_text(errors="replace"))
        ev = d.get("event")
        if ev in {
            "http_inbound_request",
            "http_stream_upstream_request",
        }:
            by_req[d["request_id"]][ev] = d

    rows = []

    for req, events in by_req.items():
        pre = events.get("http_inbound_request")
        post = events.get("http_stream_upstream_request")
        if not pre or not post:
            continue

        rows.append(
            {
                "timestamp_ns": int(pre["timestamp_ns"]),
                "request_id": req,
                "pre": pre["body"],
                "post": post["body"],
            }
        )

    rows.sort(key=lambda x: x["timestamp_ns"])
    return rows


def natural_rg_text(task: str) -> str:
    capdir = RAW / "results" / f"{task}-OFF" / "rg-captures"
    return "\n".join(
        p.read_text(errors="replace")
        for p in sorted(capdir.glob("*.stdout"))
    )


def first_exposure_records(
    task: str,
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rg = natural_rg_text(task)
    result = []

    for rec in candidate["critical_decisions"]:
        path = rec["path"]
        line = int(rec["line"])
        text = rec["text"]
        vs = variants(path, line, text)

        if not has(rg, vs):
            continue

        first = None
        for row in rows:
            if has(row["pre"], vs):
                first = row
                break

        if first is None:
            continue

        cid = call_id_containing(first["pre"], vs)

        result.append(
            {
                "task": task,
                "path": path,
                "line": line,
                "text": text,
                "variants": vs,
                "request_id": first["request_id"],
                "call_id": cid,
                "captured_keep": has(first["post"], vs),
                "controlled_baseline_keep": rec.get("baseline_keep"),
                "controlled_v3_keep": rec.get("v3_keep"),
                "preknown_controlled_dk": (
                    task,
                    path,
                    line,
                )
                in PREKNOWN,
            }
        )

    return result


def replay_task(
    task: str,
    rows: list[dict[str, Any]],
    flag: bool,
    model: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    handler = make_handler(flag, model)

    outputs: dict[str, dict[str, Any]] = {}

    for row in rows:
        payload = copy.deepcopy(row["pre"])

        result = handler._compress_openai_responses_payload(
            payload,
            model=model,
            request_id=f"offline_{task}_{'on' if flag else 'off'}_{row['request_id']}",
            timing={},
            client="opencode",
            savings_tags={},
        )

        replayed = result[0]

        outputs[row["request_id"]] = {
            "payload": replayed,
            "modified": bool(result[1]),
            "tokens_saved": int(result[2]),
            "transforms": list(result[3]),
            "reason": result[4],
            "input_bytes": int(result[5]),
            "output_bytes": int(result[6]),
            "attempted_input_tokens": int(result[7]),
        }

    return outputs, handler.openai_provider.counter.mode


def main() -> None:
    freeze_check()

    if not RAW.is_dir():
        die(f"Raw exact-wire archive missing: {RAW}")

    data = json.loads(EXHAUSTIVE.read_text())

    all_records: list[dict[str, Any]] = []
    task_data: dict[str, Any] = {}

    for task, sha in TASK_COMMITS.items():
        candidate = find_candidate(data, sha)
        if candidate is None:
            die(f"Historical candidate missing for {task}")

        rows = load_wire(task)
        if not rows:
            die(f"No wire rows for {task}")

        model = str(rows[0]["pre"].get("model") or "gpt-5.6-terra")

        records = first_exposure_records(
            task,
            candidate,
            rows,
        )

        print()
        print("=" * 78)
        print(task, f"requests={len(rows)} first_exposure_records={len(records)}")
        print("=" * 78)

        off, off_tokenizer = replay_task(
            task,
            rows,
            False,
            model,
        )
        on, on_tokenizer = replay_task(
            task,
            rows,
            True,
            model,
        )

        for rec in records:
            req = rec["request_id"]
            vs = rec.pop("variants")

            off_payload = off[req]["payload"]
            on_payload = on[req]["payload"]

            off_keep = has(off_payload, vs)
            on_keep = has(on_payload, vs)

            captured_keep = bool(rec["captured_keep"])
            replay_valid = off_keep == captured_keep

            cid = rec["call_id"]
            captured_row = next(
                x for x in rows if x["request_id"] == req
            )

            captured_unit = output_by_call_id(
                captured_row["post"],
                cid,
            )
            off_unit = output_by_call_id(
                off_payload,
                cid,
            )
            on_unit = output_by_call_id(
                on_payload,
                cid,
            )

            unit_hash_match = (
                captured_unit is not None
                and off_unit is not None
                and sha_text(captured_unit) == sha_text(off_unit)
            )

            rec.update(
                {
                    "off_replay_keep": off_keep,
                    "v3_replay_keep": on_keep,
                    "replay_valid": replay_valid,
                    "off_unit_hash_matches_captured": unit_hash_match,
                    "captured_unit_sha256": sha_text(captured_unit),
                    "off_unit_sha256": sha_text(off_unit),
                    "v3_unit_sha256": sha_text(on_unit),
                    "off_tokens_saved": off[req]["tokens_saved"],
                    "v3_tokens_saved": on[req]["tokens_saved"],
                    "off_output_bytes": off[req]["output_bytes"],
                    "v3_output_bytes": on[req]["output_bytes"],
                }
            )

            all_records.append(rec)

            print(
                f"{task} {rec['path']}:{rec['line']} "
                f"| captured={'KEEP' if captured_keep else 'DROP'} "
                f"| OFF={'KEEP' if off_keep else 'DROP'} "
                f"| V3={'KEEP' if on_keep else 'DROP'} "
                f"| valid={replay_valid} "
                f"| unit_hash={unit_hash_match}"
                + (" | PREKNOWN-DK" if rec["preknown_controlled_dk"] else "")
            )

        task_data[task] = {
            "requests": len(rows),
            "first_exposure_records": len(records),
            "off_tokenizer": off_tokenizer,
            "on_tokenizer": on_tokenizer,
        }

    valid = [x for x in all_records if x["replay_valid"]]
    invalid = [x for x in all_records if not x["replay_valid"]]

    captured_keeps = sum(x["captured_keep"] for x in all_records)
    captured_drops = len(all_records) - captured_keeps

    valid_baseline_keeps = sum(x["captured_keep"] for x in valid)
    valid_v3_keeps = sum(x["v3_replay_keep"] for x in valid)

    d2k = [
        x for x in valid
        if not x["captured_keep"] and x["v3_replay_keep"]
    ]

    k2d = [
        x for x in valid
        if x["captured_keep"] and not x["v3_replay_keep"]
    ]

    preknown = [
        x for x in valid if x["preknown_controlled_dk"]
    ]
    preknown_rescued = [
        x for x in preknown
        if (not x["captured_keep"]) and x["v3_replay_keep"]
    ]

    unit_hash_matches = sum(
        bool(x["off_unit_hash_matches_captured"])
        for x in valid
    )

    summary = {
        "population_records": len(all_records),
        "captured_baseline_keep": captured_keeps,
        "captured_baseline_drop": captured_drops,
        "captured_baseline_exact_recall": (
            captured_keeps / len(all_records)
            if all_records
            else None
        ),
        "off_replay_valid_records": len(valid),
        "off_replay_invalid_records": len(invalid),
        "valid_baseline_keep": valid_baseline_keeps,
        "valid_v3_keep": valid_v3_keeps,
        "valid_baseline_exact_recall": (
            valid_baseline_keeps / len(valid)
            if valid
            else None
        ),
        "valid_v3_exact_recall": (
            valid_v3_keeps / len(valid)
            if valid
            else None
        ),
        "drop_to_keep_rescues": len(d2k),
        "keep_to_drop_regressions": len(k2d),
        "off_unit_hash_matches_captured": unit_hash_matches,
        "preknown_controlled_dk_valid": len(preknown),
        "preknown_controlled_dk_rescued": len(preknown_rescued),
    }

    result = {
        "status": "completed",
        "v3_freeze_sha": FREEZE,
        "runtime_import": str(Path(headroom.__file__).resolve()),
        "raw_capture_root": str(RAW),
        "summary": summary,
        "tasks": task_data,
        "records": all_records,
    }

    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    print()
    print("=" * 78)
    print("EXACT-WIRE FROZEN V3 REPLAY SUMMARY")
    print("=" * 78)

    for k, v in summary.items():
        print(f"{k:38}: {v}")

    print()
    print("PREKNOWN CONTROLLED D→K SUBSET:")
    for x in preknown:
        print(
            f"{x['task']} {x['path']}:{x['line']} "
            f"captured={'KEEP' if x['captured_keep'] else 'DROP'} "
            f"OFF={'KEEP' if x['off_replay_keep'] else 'DROP'} "
            f"V3={'KEEP' if x['v3_replay_keep'] else 'DROP'} "
            f"unit_hash={x['off_unit_hash_matches_captured']}"
        )

    print()
    print("Results:", OUT)


if __name__ == "__main__":
    main()
