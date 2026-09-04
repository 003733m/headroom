#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

CONTROL = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

HERE = CONTROL / "experiments/bridge_relevance/bridge_hard"

SCREEN_RUNNER = HERE / "run_historical_difficulty_screen.py"

MANIFEST_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "hard_confirmatory_manifest.json"
)

METRIC_PROTOCOL_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "hard_confirmatory_metrics_protocol.json"
)

ROOT = (
    Path.home()
    / "headroom-hard-confirmatory-20260904"
)

SNAPSHOTS = (
    Path.home()
    / "headroom-hard-screen-snapshots-20260904"
)

spec = importlib.util.spec_from_file_location(
    "difficulty_screen_base",
    SCREEN_RUNNER,
)

if spec is None or spec.loader is None:
    raise SystemExit("cannot import difficulty screen runner")

screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

base = screen.base

base.ROOT = ROOT
base.SNAPSHOTS = SNAPSHOTS

os.environ.pop("VIRTUAL_ENV", None)


def die(msg: str):
    print("ERROR:", msg)
    raise SystemExit(1)


def capture_git(*args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=CONTROL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if cp.returncode:
        die(cp.stdout)

    return cp.stdout


def file_lock_commit(rel: str) -> str:
    return capture_git(
        "log", "-1", "--format=%H", "--", rel
    ).strip()


def verify_committed_unchanged(rel: str):
    cp = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=CONTROL,
    )

    if cp.returncode:
        die(f"{rel} not committed in HEAD")

    status = capture_git(
        "status", "--porcelain=v1", "--", rel
    ).strip()

    if status:
        die(f"{rel} differs from committed HEAD")


def locked_manifest():
    verify_committed_unchanged(MANIFEST_REL)
    verify_committed_unchanged(METRIC_PROTOCOL_REL)

    d = json.loads(
        (CONTROL / MANIFEST_REL).read_text()
    )

    if d.get("status") != (
        "locked_before_any_hard_confirmatory_agent_run"
    ):
        die("confirmatory manifest status mismatch")

    if d.get("algorithm_freeze") != base.FREEZE:
        die("algorithm freeze mismatch")

    if d.get("task_count") != 6:
        die("expected exactly 6 hard tasks")

    if d.get("condition_count") != 12:
        die("expected exactly 12 hard conditions")

    expected_ids = {
        "G10", "G09", "G07",
        "G19", "G04", "G21",
    }

    got_ids = {
        t["task_id"]
        for t in d["tasks"]
    }

    if got_ids != expected_ids:
        die(f"hard-task set mismatch: {got_ids}")

    return d


base.locked_manifest = locked_manifest


# ------------------------------------------------------------------
# Exact wire capture.
#
# Existing Headroom proxy support recognizes these environment
# variables. Each condition receives its own capture directory.
# ------------------------------------------------------------------

_original_start_proxy = base.start_proxy


def wire_start_proxy(port, out, flag):
    wire_dir = out / "codex-wire"
    wire_dir.mkdir(parents=True, exist_ok=True)

    keys = {
        "HEADROOM_CODEX_WIRE_DEBUG": "1",
        "HEADROOM_CODEX_WIRE_DEBUG_DIR": str(wire_dir),
    }

    old = {
        k: os.environ.get(k)
        for k in keys
    }

    os.environ.update(keys)

    try:
        return _original_start_proxy(
            port,
            out,
            flag,
        )
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


base.start_proxy = wire_start_proxy


# Use the natural runner's original metric writer rather than the
# OFF-only screening wrapper.
_original_write_metrics = screen._original_write_metrics


def confirmatory_write_metrics(
    out,
    task,
    cond,
    flag,
    agent_rc,
    oracle_rc,
    wall,
):
    _original_write_metrics(
        out,
        task,
        cond,
        flag,
        agent_rc,
        oracle_rc,
        wall,
    )

    p = out / "metrics.json"
    d = json.loads(p.read_text())

    wire_dir = out / "codex-wire"

    wire_files = (
        [
            x
            for x in wire_dir.rglob("*")
            if x.is_file()
        ]
        if wire_dir.exists()
        else []
    )

    d["evaluation_type"] = (
        "historical_hard_holdout_confirmatory"
    )
    d["screening_only"] = False
    d["confirmatory_estimate"] = True
    d["fresh_condition_session"] = True
    d["hard_manifest_commit"] = (
        file_lock_commit(MANIFEST_REL)
    )
    d["metrics_protocol_commit"] = (
        file_lock_commit(METRIC_PROTOCOL_REL)
    )
    d["wire_debug_required"] = True
    d["wire_debug_file_count"] = len(wire_files)
    d["wire_debug_bytes"] = sum(
        x.stat().st_size
        for x in wire_files
    )

    p.write_text(
        json.dumps(
            d,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print()
    print(
        f"CONFIRMATORY RESULT {task['task_id']}-{cond}: "
        f"{'PASS' if d['task_pass'] else 'FAIL'} "
        f"| act={d['treatment_activated']} "
        f"| req={d['requests']} "
        f"| rg={d['rg_capture_count']} "
        f"| input={d['input_tokens_original_sum']} "
        f"| saved={d['derived_tokens_saved_sum']} "
        f"| wire={d['wire_debug_file_count']}"
    )


base.write_metrics = confirmatory_write_metrics


def verify_wire_after_condition(item: str):
    out = ROOT / "results" / item

    p = out / "metrics.json"

    if not p.exists():
        return

    d = json.loads(p.read_text())

    if d.get("wire_debug_file_count", 0) <= 0:
        die(
            f"{item} completed but exact-wire capture is empty. "
            "Do not continue automatically."
        )


def run_next() -> bool:
    manifest = locked_manifest()

    base.prepare_runtime()
    base.snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    for item in manifest["run_order"]:
        out = ROOT / "results" / item

        if (out / "metrics.json").exists():
            verify_wire_after_condition(item)
            continue

        if (out / "AGENT_STARTED").exists():
            die(
                f"{item} started but is incomplete. "
                "DO NOT rerun automatically."
            )

        tid, cond = item.split("-", 1)

        task = base.task_by_id(
            manifest,
            tid,
        )

        print()
        print("=" * 72)
        print(f"RUNNING HARD CONFIRMATORY: {item}")
        print("=" * 72)

        base.run_condition(
            task,
            cond,
        )

        verify_wire_after_condition(item)

        return True

    print("All 12 hard confirmatory conditions complete.")
    return False


def run_all():
    while run_next():
        show_status()


def show_status():
    manifest = locked_manifest()

    print("===== HARD CONFIRMATORY STATUS =====")

    done = 0

    for item in manifest["run_order"]:
        out = ROOT / "results" / item
        p = out / "metrics.json"

        if p.exists():
            d = json.loads(p.read_text())
            done += 1

            print(
                f"{item:8} COMPLETE "
                f"{'PASS' if d['task_pass'] else 'FAIL':4} "
                f"act={d['treatment_activated']} "
                f"req={d['requests']} "
                f"rg={d['rg_capture_count']} "
                f"input={d['input_tokens_original_sum']} "
                f"wire={d.get('wire_debug_file_count', 0)}"
            )

        elif (out / "AGENT_STARTED").exists():
            print(
                f"{item:8} STARTED-INCOMPLETE"
            )

        else:
            print(
                f"{item:8} PENDING"
            )

    print()
    print(f"complete={done}/12")


def preflight():
    manifest = locked_manifest()

    print("hard tasks:", [
        x["task_id"]
        for x in manifest["tasks"]
    ])

    base.prepare_runtime()
    base.snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    print()
    print("=" * 72)
    print("HARD CONFIRMATORY PREFLIGHT OK")
    print("No benchmark agent was run.")
    print("Exact-wire capture will be enabled per condition.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()

    g = parser.add_mutually_exclusive_group(
        required=True
    )

    g.add_argument("--preflight", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--run-next", action="store_true")
    g.add_argument("--run-all", action="store_true")

    args = parser.parse_args()

    if args.preflight:
        preflight()
    elif args.status:
        show_status()
    elif args.run_next:
        run_next()
    else:
        run_all()


if __name__ == "__main__":
    main()
