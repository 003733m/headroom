#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

SELECTED_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "second_unseen_adopted_bridge_selected_task.json"
)
ON_EXEC_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "second_unseen_adopted_bridge_on_execution.json"
)
SNAP_LOCK_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "second_unseen_adopted_bridge_snapshot_lock.json"
)

ROOT = Path.home() / "headroom-second-unseen-adopted-20260905"
RUNTIME = Path("/tmp/headroom-second-unseen-adopted-runtime-89da0898")
SNAPSHOTS = Path("/tmp/headroom-second-unseen-adopted-snapshots-20260905")

spec = importlib.util.spec_from_file_location(
    "second_unseen_base",
    HERE / "run_natural_transfer.py",
)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import natural-transfer runner")

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.FREEZE = FREEZE
base.ROOT = ROOT
base.RUNTIME = RUNTIME
base.SNAPSHOTS = SNAPSHOTS
base.SNAP_LOCK_REL = SNAP_LOCK_REL


def die(msg: str):
    print("ERROR:", msg)
    raise SystemExit(1)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=CONTROL,
        text=True,
    ).strip()


def verify_locked(rel: str):
    subprocess.check_call(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=CONTROL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if git("status", "--porcelain=v1", "--", rel):
        die(f"{rel} differs from committed HEAD")


def locked_manifest():
    verify_locked(SELECTED_REL)
    verify_locked(ON_EXEC_REL)
    verify_locked(SNAP_LOCK_REL)

    selected = json.loads(
        (CONTROL / SELECTED_REL).read_text()
    )
    on_exec = json.loads(
        (CONTROL / ON_EXEC_REL).read_text()
    )

    if selected["status"] != (
        "first_eligible_task_locked_before_on_run"
    ):
        die("selected-task lock mismatch")

    if selected["selected_task"] != "U01":
        die("unexpected selected task")

    if on_exec["status"] != "locked_before_selected_on_run":
        die("ON execution lock mismatch")

    if on_exec["condition"] != "ON":
        die("condition mismatch")

    # Load original locked task definition.
    screen = json.loads(
        (
            CONTROL
            / "experiments/bridge_relevance/bridge_hard/"
              "second_unseen_adopted_bridge_execution.json"
        ).read_text()
    )

    task = next(
        x for x in screen["tasks"]
        if x["task_id"] == "U01"
    )

    return {
        "tasks": [task],
        "run_order": ["U01-ON"],
    }


base.locked_manifest = locked_manifest


def snapshot_lock():
    verify_locked(SNAP_LOCK_REL)

    d = json.loads(
        (CONTROL / SNAP_LOCK_REL).read_text()
    )

    if d["production_freeze"] != FREEZE:
        die("snapshot freeze mismatch")

    return d


base.snapshot_lock = snapshot_lock


_original_start_proxy = base.start_proxy


def wire_start_proxy(port, out, flag):
    wire_dir = out / "codex-wire"
    wire_dir.mkdir(parents=True, exist_ok=True)

    keys = {
        "HEADROOM_CODEX_WIRE_DEBUG": "1",
        "HEADROOM_CODEX_WIRE_DEBUG_DIR": str(wire_dir),
    }

    old = {k: os.environ.get(k) for k in keys}

    try:
        os.environ.update(keys)
        return _original_start_proxy(port, out, flag)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


base.start_proxy = wire_start_proxy


def main():
    manifest = locked_manifest()

    base.prepare_runtime()
    snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    out = ROOT / "results/U01-ON"

    if (out / "metrics.json").exists():
        die("U01-ON already completed; DO NOT rerun")

    if (out / "AGENT_STARTED").exists():
        die("U01-ON already started; DO NOT rerun")

    task = manifest["tasks"][0]

    print("=" * 72)
    print("RUNNING LOCKED SECOND UNSEEN CONDITION: U01-ON")
    print("=" * 72)

    base.run_condition(task, "ON")

    wire = out / "codex-wire"
    inbound = list(
        wire.glob("*_http_inbound_request.json")
    )

    if not inbound:
        die("U01-ON has no exact inbound capture")

    print()
    print("U01-ON consumed exactly once.")
    print("wire inbound requests:", len(inbound))
    print("DO NOT RERUN.")


if __name__ == "__main__":
    main()
