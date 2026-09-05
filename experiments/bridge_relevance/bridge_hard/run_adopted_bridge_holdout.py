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

HARD_RUNNER = HERE / "run_hard_confirmatory.py"

EXEC_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "adopted_bridge_holdout_execution.json"
)

METRIC_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "adopted_bridge_holdout_metrics_protocol.json"
)

SELECTION_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "adopted_bridge_holdout_selection.json"
)

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

ROOT = (
    Path.home()
    / "headroom-adopted-bridge-holdout-20260905"
)

RUNTIME = Path(
    "/tmp/headroom-adopted-bridge-holdout-runtime-89da0898"
)

spec = importlib.util.spec_from_file_location(
    "hard_confirmatory_base",
    HARD_RUNNER,
)

if spec is None or spec.loader is None:
    raise SystemExit("cannot import hard confirmatory runner")

hard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hard)

base = hard.base

# ---------------------------------------------------------------------------
# Experiment-specific state.
# ---------------------------------------------------------------------------

hard.ROOT = ROOT
base.ROOT = ROOT

hard.MANIFEST_REL = EXEC_REL
hard.METRIC_PROTOCOL_REL = METRIC_REL

base.FREEZE = FREEZE
base.RUNTIME = RUNTIME
base.MANIFEST_REL = EXEC_REL

# Preserve the already-locked historical buggy snapshots used by the
# difficulty screen / hard-confirmatory experiments.
base.SNAPSHOTS = hard.SNAPSHOTS

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
        "log",
        "-1",
        "--format=%H",
        "--",
        rel,
    ).strip()


def verify_committed_unchanged(rel: str):
    cp = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=CONTROL,
    )

    if cp.returncode:
        die(f"{rel} not committed in HEAD")

    status = capture_git(
        "status",
        "--porcelain=v1",
        "--",
        rel,
    ).strip()

    if status:
        die(f"{rel} differs from committed HEAD")


def locked_manifest():
    verify_committed_unchanged(EXEC_REL)
    verify_committed_unchanged(METRIC_REL)
    verify_committed_unchanged(SELECTION_REL)

    execution = json.loads(
        (CONTROL / EXEC_REL).read_text()
    )

    protocol = json.loads(
        (CONTROL / METRIC_REL).read_text()
    )

    selection = json.loads(
        (CONTROL / SELECTION_REL).read_text()
    )

    if execution.get("status") != (
        "locked_before_any_adopted_bridge_holdout_agent_run"
    ):
        die("execution status mismatch")

    if protocol.get("status") != (
        "locked_before_any_adopted_bridge_holdout_agent_run"
    ):
        die("metrics protocol status mismatch")

    if execution.get("algorithm_freeze") != FREEZE:
        die("execution freeze mismatch")

    if protocol.get("algorithm_freeze") != FREEZE:
        die("metrics freeze mismatch")

    if base.FREEZE != FREEZE:
        die("runtime freeze mismatch")

    if selection.get("selected_task_ids") != ["G11"]:
        die(
            "selection is no longer exactly treatment-blind G11"
        )

    expected_selection_commit = file_lock_commit(
        SELECTION_REL
    )

    if execution.get("selection_lock_commit") != (
        expected_selection_commit
    ):
        die("selection lock commit mismatch")

    if execution.get("task_count") != 1:
        die("expected exactly one holdout task")

    if execution.get("condition_count") != 2:
        die("expected exactly two holdout conditions")

    if execution.get("run_order") != [
        "G11-OFF",
        "G11-ON",
    ]:
        die("holdout run order mismatch")

    if [
        task["task_id"]
        for task in execution["tasks"]
    ] != ["G11"]:
        die("holdout task mismatch")

    task = execution["tasks"][0]

    if task.get("source_commit") != (
        "6137967083936467c570e8c7f20e94f43ccc13aa"
    ):
        die("G11 source commit mismatch")

    return execution


# Every reused natural-runner helper must resolve this manifest.
hard.locked_manifest = locked_manifest
base.locked_manifest = locked_manifest


# ---------------------------------------------------------------------------
# Hard-confirmatory import already enabled exact-wire capture and installed
# its richer metric writer. Add holdout-specific metadata after that writer.
# ---------------------------------------------------------------------------

_hard_write_metrics = base.write_metrics


def holdout_write_metrics(
    out,
    task,
    cond,
    flag,
    agent_rc,
    oracle_rc,
    wall,
):
    _hard_write_metrics(
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

    d["evaluation_type"] = (
        "adopted_bridge_unseen_holdout"
    )
    d["confirmatory_estimate"] = True
    d["applicability_conditioned"] = True
    d["fresh_condition_session"] = True
    d["algorithm_freeze"] = FREEZE
    d["selection_lock_commit"] = file_lock_commit(
        SELECTION_REL
    )
    d["execution_lock_commit"] = file_lock_commit(
        EXEC_REL
    )
    d["metrics_protocol_commit"] = file_lock_commit(
        METRIC_REL
    )
    d["historical_screen_off_reused_as_outcome"] = False

    p.write_text(
        json.dumps(
            d,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print()
    print(
        f"HOLDOUT RESULT {task['task_id']}-{cond}: "
        f"{'PASS' if d['task_pass'] else 'FAIL'} "
        f"| act={d['treatment_activated']} "
        f"| req={d['requests']} "
        f"| rg={d['rg_capture_count']} "
        f"| input={d['input_tokens_original_sum']} "
        f"| saved={d['derived_tokens_saved_sum']} "
        f"| wire={d.get('wire_debug_file_count', 0)}"
    )


base.write_metrics = holdout_write_metrics


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


def prepare_common():
    locked_manifest()

    base.prepare_runtime()
    base.snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    runtime_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=base.RUNTIME,
        text=True,
    ).strip()

    if runtime_head != FREEZE:
        die(
            "prepared runtime HEAD mismatch: "
            f"{runtime_head}"
        )


def preflight():
    manifest = locked_manifest()

    print("holdout tasks:", [
        task["task_id"]
        for task in manifest["tasks"]
    ])
    print("run order:", manifest["run_order"])
    print("algorithm freeze:", FREEZE)
    print("result root:", ROOT)
    print("runtime:", RUNTIME)

    prepare_common()

    print()
    print("=" * 72)
    print("ADOPTED-BRIDGE HOLDOUT PREFLIGHT OK")
    print("NO BENCHMARK AGENT WAS RUN.")
    print("Exact-wire capture is enabled.")
    print("Fresh OFF and ON sessions are pending.")
    print("=" * 72)


def show_status():
    manifest = locked_manifest()

    print("===== ADOPTED-BRIDGE UNSEEN HOLDOUT =====")

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
                f"saved={d['derived_tokens_saved_sum']} "
                f"wire={d.get('wire_debug_file_count', 0)}"
            )

        elif (out / "AGENT_STARTED").exists():
            print(f"{item:8} STARTED-INCOMPLETE")

        else:
            print(f"{item:8} PENDING")

    print()
    print(f"complete={done}/2")


def run_next() -> bool:
    manifest = locked_manifest()

    prepare_common()

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
        print(
            f"RUNNING ADOPTED-BRIDGE HOLDOUT: {item}"
        )
        print("=" * 72)

        base.run_condition(
            task,
            cond,
        )

        verify_wire_after_condition(item)

        return True

    print("Both adopted-bridge holdout conditions complete.")
    return False


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--preflight",
        action="store_true",
    )
    group.add_argument(
        "--status",
        action="store_true",
    )
    group.add_argument(
        "--run-next",
        action="store_true",
    )

    args = parser.parse_args()

    if args.preflight:
        preflight()
    elif args.status:
        show_status()
    else:
        run_next()


if __name__ == "__main__":
    main()
