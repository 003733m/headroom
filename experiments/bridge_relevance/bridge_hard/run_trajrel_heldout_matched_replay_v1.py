#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONTROL = Path(__file__).resolve().parents[3]

ROOT = Path.home() / "headroom-trajrel-heldout-pilot-v1-20260914"
MANIFEST = HERE / "trajrel_heldout_replay_manifest_v1.json"
PROTOCOL = HERE / "trajrel_heldout_matched_replay_protocol_v1.json"
EXACT_PATH = HERE / "trajrel_heldout_exact_helper_v1.py"
OUT = ROOT / "trajrel_heldout_matched_replay_v1.json"

TASKS = ["U04", "U05", "U06", "U07", "U08", "U09", "U10", "U11", "U12", "P10"]
FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "local_shell_call",
}
OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "local_shell_call_output",
}

SEARCH_RE = re.compile(
    r"(^|(?:&&|;|\|)\s*|\s)(?:rg|grep)\s",
    re.I,
)
RECORD_RE = re.compile(
    r"^(?P<path>.+?)(?P<sep>[:-])(?P<line>\d+)(?P=sep)(?P<text>.*)$"
)


def die(msg: str) -> None:
    raise SystemExit(msg)


def load_exact_module():
    spec = importlib.util.spec_from_file_location(
        "hard_exact_replay_helpers",
        EXACT_PATH,
    )
    if spec is None or spec.loader is None:
        die("Could not load exact-wire replay helpers")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.freeze_check()
    return module


exact = load_exact_module()

# exact helper forces frozen runtime imports.
import headroom.trajectory_relevance as tr  # noqa: E402
from headroom.transforms.relevance_split import build_relevance_query  # noqa: E402

ORIGINAL_V3_BUILDER = tr.build_responses_search_relevance_context


def sha(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def normalize_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def text_of(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []
        for item in value:
            t = text_of(item)
            if t:
                parts.append(t)
        return "\n".join(parts)

    if isinstance(value, dict):
        for key in ("text", "output", "content", "stdout", "stderr"):
            if key in value:
                t = text_of(value[key])
                if t:
                    return t

    return ""


def collect_strings(value: Any) -> list[str]:
    out: list[str] = []

    if isinstance(value, str):
        out.append(value)
        try:
            decoded = json.loads(value)
        except Exception:
            decoded = None

        if decoded is not None and decoded != value:
            out.extend(collect_strings(decoded))

    elif isinstance(value, dict):
        for item in value.values():
            out.extend(collect_strings(item))

    elif isinstance(value, list):
        for item in value:
            out.extend(collect_strings(item))

    return out


def command_of(item: dict[str, Any]) -> str:
    strings: list[str] = []

    for key in (
        "arguments",
        "input",
        "action",
        "command",
        "cmd",
    ):
        if key in item:
            strings.extend(collect_strings(item[key]))

    for value in strings:
        if SEARCH_RE.search(value):
            return value

    return "\n".join(strings)


def producing_call(
    items: list[Any],
    before_index: int,
) -> tuple[str, str] | None:
    if not (0 <= before_index < len(items)):
        return None

    output = items[before_index]
    if not isinstance(output, dict):
        return None

    call_id = output.get("call_id")
    if not isinstance(call_id, str):
        return None

    for index in range(before_index - 1, -1, -1):
        item = items[index]

        if not isinstance(item, dict):
            continue

        if item.get("call_id") != call_id:
            continue

        if item.get("type") not in CALL_TYPES:
            continue

        command = command_of(item)

        if not command:
            return None

        tool_name = (
            "local_shell"
            if item.get("type") == "local_shell_call"
            else str(item.get("name") or "bash")
        )

        return tool_name, command

    return None


def current_query_context(
    items: list[Any],
    before_index: int,
    user_context: str = "",
) -> str:
    info = producing_call(items, before_index)

    if info is None:
        return ""

    tool_name, command = info

    if not SEARCH_RE.search(command):
        return ""

    try:
        return build_relevance_query(
            user_context,
            tool_name,
            command,
        ).strip()
    except Exception:
        return (
            f"{user_context}\n{command}".strip()
            if user_context
            else command.strip()
        )


def builder_query_only(
    items: list[Any],
    *,
    before_index: int,
    user_context: str = "",
    target_content: str | None = None,
    scoring=None,
) -> str:
    del target_content, scoring
    return current_query_context(
        items,
        before_index,
        user_context,
    )


def builder_query_plus_v3(
    items: list[Any],
    *,
    before_index: int,
    user_context: str = "",
    target_content: str | None = None,
    scoring=None,
) -> str:
    query = current_query_context(
        items,
        before_index,
        user_context,
    )

    trajectory = ORIGINAL_V3_BUILDER(
        items,
        before_index=before_index,
        user_context=user_context,
        target_content=target_content,
        scoring=scoring,
    )

    if query and trajectory:
        if trajectory == user_context:
            return query
        return f"{query}\n{trajectory}"

    return query or trajectory


def load_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    d = json.loads(MANIFEST.read_text())

    if d.get("algorithm_freeze") != FREEZE:
        die("Manifest freeze mismatch")

    tasks = {
        item["task_id"]: item
        for item in d["tasks"]
        if item["task_id"] in TASKS
    }

    if set(tasks) != set(TASKS):
        die("Held-out pilot manifest task mismatch")

    return d, tasks


def load_wire(task: str) -> list[dict[str, Any]]:
    wire = (
        ROOT
        / "results"
        / f"{task}-OFF"
        / "codex-wire"
    )

    if not wire.is_dir():
        die(f"Missing hard OFF wire: {wire}")

    by_req: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for path in wire.rglob("*.json"):
        try:
            d = json.loads(path.read_text(errors="replace"))
        except Exception:
            continue

        event = d.get("event")

        if event in {
            "http_inbound_request",
            "http_stream_upstream_request",
        }:
            req = d.get("request_id")
            if isinstance(req, str):
                by_req[req][event] = d

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

    if not rows:
        die(f"No paired OFF wire requests for {task}")

    return rows


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=CONTROL,
        text=True,
        stderr=subprocess.DEVNULL,
    )


def historical_oracle(
    task: dict[str, Any],
) -> dict[str, dict[int, str]]:
    commit = task["source_commit"]
    result: dict[str, dict[int, str]] = {}

    for path in task["production_files"]:
        diff = git_text(
            "diff",
            "--unified=0",
            f"{commit}^",
            commit,
            "--",
            path,
        )

        parent = git_text(
            "show",
            f"{commit}^:{path}",
        ).splitlines()

        line_numbers: set[int] = set()

        for match in re.finditer(
            r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@",
            diff,
            flags=re.M,
        ):
            old_start = int(match.group(1))
            old_count = (
                int(match.group(2))
                if match.group(2) is not None
                else 1
            )

            width = max(old_count, 1)

            start = max(1, old_start - 5)
            end = min(
                len(parent),
                old_start + width - 1 + 5,
            )

            line_numbers.update(
                range(start, end + 1)
            )

        result[path] = {
            line: parent[line - 1]
            for line in sorted(line_numbers)
            if 1 <= line <= len(parent)
            and parent[line - 1].strip()
        }

    return result


def parse_records(target: str) -> list[dict[str, Any]]:
    rows = []

    for raw in target.splitlines():
        match = RECORD_RE.match(raw)

        if not match:
            continue

        rows.append(
            {
                "raw": raw,
                "path": normalize_path(match.group("path")),
                "line": int(match.group("line")),
                "text": match.group("text"),
            }
        )

    return rows


def is_fix_adjacent(
    rec: dict[str, Any],
    oracle: dict[str, dict[int, str]],
) -> bool:
    rec_path = normalize_path(rec["path"])
    rec_text = rec["text"].strip()
    rec_line = int(rec["line"])

    if not rec_text:
        return False

    for path, lines in oracle.items():
        norm_path = normalize_path(path)

        if not (
            rec_path == norm_path
            or rec_path.endswith("/" + norm_path)
        ):
            continue

        expected = lines.get(rec_line)

        if (
            expected is not None
            and expected.strip() == rec_text
        ):
            return True

        # Conservative small line-shift fallback.
        for line, text in lines.items():
            if (
                abs(line - rec_line) <= 3
                and text.strip() == rec_text
            ):
                return True

    return False


def first_search_units(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    units = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        items = row["pre"].get("input", [])

        if not isinstance(items, list):
            continue

        calls: dict[str, str] = {}

        for item in items:
            if not isinstance(item, dict):
                continue

            if item.get("type") not in CALL_TYPES:
                continue

            call_id = item.get("call_id")

            if not isinstance(call_id, str):
                continue

            command = command_of(item)

            if SEARCH_RE.search(command):
                calls[call_id] = command

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            if item.get("type") not in OUTPUT_TYPES:
                continue

            call_id = item.get("call_id")

            if call_id not in calls:
                continue

            target = text_of(item.get("output"))

            if (
                len(
                    target.encode(
                        "utf-8",
                        errors="replace",
                    )
                )
                < 512
            ):
                continue

            key = (call_id, sha(target))

            if key in seen:
                continue

            seen.add(key)

            units.append(
                {
                    "request_id": row["request_id"],
                    "call_id": call_id,
                    "before_index": index,
                    "command": calls[call_id],
                    "target": target,
                    "target_sha256": sha(target),
                    "query_context": current_query_context(
                        items,
                        index,
                    ),
                    "v3_context": ORIGINAL_V3_BUILDER(
                        items,
                        before_index=index,
                        target_content=target,
                    ),
                }
            )

    return units


def replay(
    task: str,
    rows: list[dict[str, Any]],
    mode: str,
    model: str,
) -> dict[str, dict[str, Any]]:
    if mode not in {"A", "B", "C", "D"}:
        raise ValueError(mode)

    flag = mode != "A"

    old_builder = tr.build_responses_search_relevance_context

    if mode == "B":
        tr.build_responses_search_relevance_context = builder_query_only
    elif mode == "C":
        tr.build_responses_search_relevance_context = builder_query_plus_v3
    else:
        tr.build_responses_search_relevance_context = ORIGINAL_V3_BUILDER

    try:
        handler = exact.make_handler(flag, model)

        outputs = {}

        for row in rows:
            payload = copy.deepcopy(row["pre"])

            result = handler._compress_openai_responses_payload(
                payload,
                model=model,
                request_id=(
                    f"hard_matched_{task}_{mode}_"
                    f"{row['request_id']}"
                ),
                timing={},
                client="opencode",
                savings_tags={},
            )

            outputs[row["request_id"]] = {
                "payload": result[0],
                "modified": bool(result[1]),
                "tokens_saved": int(result[2]),
                "transforms": list(result[3]),
                "reason": result[4],
                "input_bytes": int(result[5]),
                "output_bytes": int(result[6]),
                "attempted_input_tokens": int(result[7]),
            }

        return outputs

    finally:
        tr.build_responses_search_relevance_context = old_builder


def retained(
    output: str | None,
    rec: dict[str, Any],
) -> bool:
    if output is None:
        return False

    variants = exact.variants(
        rec["path"],
        rec["line"],
        rec["text"],
    )

    return exact.has(output, variants)


def summarize(
    unit_rows: list[dict[str, Any]],
    tokenizer,
) -> dict[str, Any]:
    valid = [
        unit
        for unit in unit_rows
        if unit["replay_valid"]
    ]

    summary: dict[str, Any] = {
        "units_total": len(unit_rows),
        "units_valid": len(valid),
        "units_hash_exact": sum(
            unit["off_unit_hash_matches_captured"]
            for unit in valid
        ),
        "historical_fix_adjacent_records": 0,
        "all_parsed_records": 0,
    }

    for mode in ("A", "B", "C", "D"):
        summary[mode] = {
            "fix_adjacent_kept": 0,
            "non_fix_adjacent_kept": 0,
            "fix_adjacent_kept_record_tokens": 0,
            "non_fix_adjacent_kept_record_tokens": 0,
            "full_output_tokens": 0,
        }

    b_to_c_rescues = 0
    b_to_c_regressions = 0

    for unit in valid:
        for mode in ("A", "B", "C", "D"):
            output = unit["outputs"].get(mode)

            if output:
                summary[mode]["full_output_tokens"] += (
                    tokenizer.count_text(output)
                )

        for rec in unit["records"]:
            summary["all_parsed_records"] += 1

            if rec["fix_adjacent"]:
                summary[
                    "historical_fix_adjacent_records"
                ] += 1

            record_tokens = tokenizer.count_text(
                rec["raw"]
            )

            for mode in ("A", "B", "C", "D"):
                keep = rec["kept"][mode]

                if not keep:
                    continue

                if rec["fix_adjacent"]:
                    summary[mode]["fix_adjacent_kept"] += 1
                    summary[mode][
                        "fix_adjacent_kept_record_tokens"
                    ] += record_tokens
                else:
                    summary[mode]["non_fix_adjacent_kept"] += 1
                    summary[mode][
                        "non_fix_adjacent_kept_record_tokens"
                    ] += record_tokens

            if rec["fix_adjacent"]:
                if (
                    not rec["kept"]["B"]
                    and rec["kept"]["C"]
                ):
                    b_to_c_rescues += 1

                if (
                    rec["kept"]["B"]
                    and not rec["kept"]["C"]
                ):
                    b_to_c_regressions += 1

    denominator = summary[
        "historical_fix_adjacent_records"
    ]

    for mode in ("A", "B", "C", "D"):
        kept = summary[mode]["fix_adjacent_kept"]

        summary[mode]["fix_adjacent_recall"] = (
            kept / denominator
            if denominator
            else None
        )

        nonfix_tokens = summary[mode][
            "non_fix_adjacent_kept_record_tokens"
        ]

        summary[mode]["evidence_efficiency"] = (
            kept / (nonfix_tokens / 1000.0)
            if nonfix_tokens > 0
            else None
        )

    b_nonfix = summary["B"][
        "non_fix_adjacent_kept_record_tokens"
    ]
    c_nonfix = summary["C"][
        "non_fix_adjacent_kept_record_tokens"
    ]

    summary["B_to_C"] = {
        "fix_adjacent_DROP_to_KEEP": b_to_c_rescues,
        "fix_adjacent_KEEP_to_DROP": b_to_c_regressions,
        "non_fix_adjacent_token_delta": c_nonfix - b_nonfix,
        "non_fix_adjacent_suppression_gain": (
            (b_nonfix - c_nonfix) / b_nonfix
            if b_nonfix > 0
            else None
        ),
        "full_output_token_delta": (
            summary["C"]["full_output_tokens"]
            - summary["B"]["full_output_tokens"]
        ),
    }

    return summary


def run_task(
    task_id: str,
    task_meta: dict[str, Any],
    model: str,
) -> tuple[dict[str, Any], Any]:
    rows = load_wire(task_id)
    units = first_search_units(rows)
    oracle = historical_oracle(task_meta)

    replays = {
        mode: replay(
            task_id,
            rows,
            mode,
            model,
        )
        for mode in ("A", "B", "C", "D")
    }

    by_req = {
        row["request_id"]: row
        for row in rows
    }

    tokenizer = exact.ProductionLikeTokenCounter(model)

    analyzed = []

    for unit in units:
        req = unit["request_id"]
        cid = unit["call_id"]

        captured = exact.output_by_call_id(
            by_req[req]["post"],
            cid,
        )

        condition_outputs = {
            mode: exact.output_by_call_id(
                replays[mode][req]["payload"],
                cid,
            )
            for mode in ("A", "B", "C", "D")
        }

        records = parse_records(unit["target"])

        record_rows = []

        captured_vector = []
        off_vector = []

        for rec in records:
            fix_adjacent = is_fix_adjacent(
                rec,
                oracle,
            )

            captured_keep = retained(
                captured,
                rec,
            )

            kept = {
                mode: retained(
                    condition_outputs[mode],
                    rec,
                )
                for mode in ("A", "B", "C", "D")
            }

            captured_vector.append(captured_keep)
            off_vector.append(kept["A"])

            record_rows.append(
                {
                    **rec,
                    "fix_adjacent": fix_adjacent,
                    "captured_keep": captured_keep,
                    "kept": kept,
                }
            )

        replay_valid = (
            bool(records)
            and captured is not None
            and condition_outputs["A"] is not None
            and captured_vector == off_vector
        )

        analyzed.append(
            {
                **unit,
                "records": record_rows,
                "replay_valid": replay_valid,
                "off_unit_hash_matches_captured": (
                    captured is not None
                    and condition_outputs["A"] is not None
                    and exact.sha_text(captured)
                    == exact.sha_text(condition_outputs["A"])
                ),
                "outputs": condition_outputs,
                "output_sha256": {
                    mode: exact.sha_text(
                        condition_outputs[mode]
                    )
                    for mode in ("A", "B", "C", "D")
                },
            }
        )

    summary = summarize(
        analyzed,
        tokenizer,
    )

    return {
        "task_id": task_id,
        "source_commit": task_meta["source_commit"],
        "subject": task_meta["subject"],
        "requests": len(rows),
        "first_exposure_search_units": len(units),
        "units_with_query_context": sum(
            bool(unit["query_context"])
            for unit in units
        ),
        "units_with_frozen_v3_context": sum(
            bool(unit["v3_context"])
            for unit in units
        ),
        "summary": summary,
        "units": analyzed,
    }, tokenizer.mode


def preflight() -> None:
    manifest, tasks = load_manifest()

    if not PROTOCOL.exists():
        die(f"Protocol missing: {PROTOCOL}")

    print("Frozen runtime: OK")
    print(
        "Protocol:",
        json.loads(PROTOCOL.read_text())["status"],
    )

    total_units = 0

    for task in TASKS:
        rows = load_wire(task)
        units = first_search_units(rows)
        oracle = historical_oracle(tasks[task])

        oracle_lines = sum(
            len(lines)
            for lines in oracle.values()
        )

        total_units += len(units)

        print(
            f"{task}: "
            f"requests={len(rows)} "
            f"first_search={len(units)} "
            f"query_ctx={sum(bool(x['query_context']) for x in units)} "
            f"v3_ctx={sum(bool(x['v3_context']) for x in units)} "
            f"oracle_lines={oracle_lines}"
        )

    print(
        f"TOTAL first-exposure search units: {total_units}"
    )
    print("No replay result was computed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
    )
    parser.add_argument(
        "--run",
        action="store_true",
    )
    args = parser.parse_args()

    if args.preflight == args.run:
        die("Use exactly one of --preflight or --run")

    exact.freeze_check()

    manifest, tasks = load_manifest()
    model = manifest["model"]

    if args.preflight:
        preflight()
        return

    results = {}
    tokenizer_modes = set()

    for task in TASKS:
        print()
        print("=" * 72)
        print(f"REPLAY {task}")
        print("=" * 72)

        result, tokenizer_mode = run_task(
            task,
            tasks[task],
            model,
        )

        results[task] = result
        tokenizer_modes.add(tokenizer_mode)

        s = result["summary"]

        print(
            f"units={s['units_total']} "
            f"valid={s['units_valid']} "
            f"critical={s['historical_fix_adjacent_records']}"
        )

        print(
            "  B(query): "
            f"critical_keep={s['B']['fix_adjacent_kept']} "
            f"recall={s['B']['fix_adjacent_recall']} "
            f"nonfix_tokens={s['B']['non_fix_adjacent_kept_record_tokens']} "
            f"output_tokens={s['B']['full_output_tokens']}"
        )

        print(
            "  C(query+trajectory): "
            f"critical_keep={s['C']['fix_adjacent_kept']} "
            f"recall={s['C']['fix_adjacent_recall']} "
            f"nonfix_tokens={s['C']['non_fix_adjacent_kept_record_tokens']} "
            f"output_tokens={s['C']['full_output_tokens']}"
        )

        print(
            "  B→C: "
            f"rescue={s['B_to_C']['fix_adjacent_DROP_to_KEEP']} "
            f"regress={s['B_to_C']['fix_adjacent_KEEP_to_DROP']} "
            f"nonfix_delta={s['B_to_C']['non_fix_adjacent_token_delta']} "
            f"output_delta={s['B_to_C']['full_output_token_delta']}"
        )

    all_units = [
        unit
        for task in TASKS
        for unit in results[task]["units"]
    ]

    # Aggregate using the same summarizer.
    tokenizer = exact.ProductionLikeTokenCounter(model)
    aggregate = summarize(
        all_units,
        tokenizer,
    )

    payload = {
        "evaluation_type": "heldout_pilot_offline_matched_replay_v1",
        "algorithm_freeze": FREEZE,
        "model": model,
        "tokenizer_modes": sorted(tokenizer_modes),
        "protocol": json.loads(PROTOCOL.read_text()),
        "tasks": results,
        "aggregate": aggregate,
    }

    OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print("=" * 72)
    print("AGGREGATE")
    print("=" * 72)

    print(
        f"units={aggregate['units_total']} "
        f"valid={aggregate['units_valid']} "
        f"hash_exact={aggregate['units_hash_exact']} "
        f"critical={aggregate['historical_fix_adjacent_records']}"
    )

    for mode, label in [
        ("A", "OFF"),
        ("D", "frozen V3"),
        ("B", "query only"),
        ("C", "query + trajectory"),
    ]:
        x = aggregate[mode]

        print(
            f"{label}: "
            f"critical={x['fix_adjacent_kept']} "
            f"recall={x['fix_adjacent_recall']} "
            f"nonfix_tokens={x['non_fix_adjacent_kept_record_tokens']} "
            f"output_tokens={x['full_output_tokens']} "
            f"efficiency={x['evidence_efficiency']}"
        )

    delta = aggregate["B_to_C"]

    print(
        "B→C: "
        f"rescue={delta['fix_adjacent_DROP_to_KEEP']} "
        f"regress={delta['fix_adjacent_KEEP_to_DROP']} "
        f"nonfix_delta={delta['non_fix_adjacent_token_delta']} "
        f"suppression_gain={delta['non_fix_adjacent_suppression_gain']} "
        f"output_delta={delta['full_output_token_delta']}"
    )

    print("JSON:", OUT)


if __name__ == "__main__":
    main()
