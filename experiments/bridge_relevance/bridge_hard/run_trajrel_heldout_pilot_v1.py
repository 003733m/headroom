#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

EXEC_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "trajrel_heldout_pilot_execution_v1.json"
)
PROTOCOL_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "trajrel_heldout_pilot_protocol_lock_v1.json"
)
SELECTION_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "trajrel_heldout_pilot_selection_lock_v1.json"
)
SNAP_LOCK_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "trajrel_heldout_pilot_snapshot_lock_v1.json"
)

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

ROOT = Path.home() / "headroom-trajrel-heldout-pilot-v1-20260914"
RUNTIME = Path("/tmp/headroom-trajrel-heldout-pilot-runtime-89da0898")
SNAPSHOTS = Path("/tmp/headroom-trajrel-heldout-pilot-snapshots-v1")

spec = importlib.util.spec_from_file_location(
    "second_unseen_base",
    HERE / "run_natural_transfer.py",
)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import natural-transfer runner")

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Reuse the already-tested natural-agent harness, but point every
# production/runtime path at the frozen final adopted-bridge commit.
base.FREEZE = FREEZE
base.ROOT = ROOT
base.RUNTIME = RUNTIME
base.SNAPSHOTS = SNAPSHOTS
base.SNAP_LOCK_REL = SNAP_LOCK_REL


def die(msg: str):
    print(f"ERROR: {msg}")
    raise SystemExit(1)


def git_capture(*args: str) -> str:
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


def verify_committed_unchanged(rel: str):
    cp = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=CONTROL,
    )
    if cp.returncode:
        die(f"{rel} is not committed in HEAD")

    status = git_capture(
        "status", "--porcelain=v1", "--", rel
    ).strip()

    if status:
        die(f"{rel} differs from committed HEAD")


def lock_commit(rel: str) -> str:
    return git_capture(
        "log", "-1", "--format=%H", "--", rel
    ).strip()


def locked_manifest():
    for rel in (EXEC_REL, PROTOCOL_REL, SELECTION_REL):
        verify_committed_unchanged(rel)

    d = json.loads((CONTROL / EXEC_REL).read_text())

    if d.get("status") != "locked_before_any_trajrel_heldout_pilot_run":
        die("execution status mismatch")

    if d.get("production_freeze") != FREEZE:
        die("production freeze mismatch")

    if d.get("task_count") != 10:
        die("expected exactly 10 tasks")

    expected = [
        "U04-OFF",
        "U05-OFF",
        "U06-OFF",
        "U07-OFF",
        "U08-OFF",
        "U09-OFF",
        "U10-OFF",
        "U11-OFF",
        "U12-OFF",
        "P10-OFF",
    ]
    if d.get("run_order") != expected:
        die("run order mismatch")

    if any(not x.endswith("-OFF") for x in d["run_order"]):
        die("screen contains a non-OFF condition")

    return d


base.locked_manifest = locked_manifest


# Enable exact inbound wire capture locally. These files are NEVER intended
# for Git; they can contain request/session metadata.
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


# Relabel metrics only; no scoring/outcome logic changes.
_original_write_metrics = base.write_metrics


def screen_write_metrics(out, task, cond, flag, agent_rc, oracle_rc, wall):
    _original_write_metrics(
        out, task, cond, flag, agent_rc, oracle_rc, wall
    )

    p = out / "metrics.json"
    d = json.loads(p.read_text())

    d["evaluation_type"] = (
        "trajrel_heldout_pilot_v1_off_only"
    )
    d["production_freeze"] = FREEZE
    d["protocol_lock_commit"] = lock_commit(PROTOCOL_REL)
    d["selection_lock_commit"] = lock_commit(SELECTION_REL)
    d["execution_lock_commit"] = lock_commit(EXEC_REL)

    p.write_text(json.dumps(d, indent=2) + "\n")


base.write_metrics = screen_write_metrics


def prepare_snapshots():
    manifest = locked_manifest()

    base.prepare_runtime()
    base.prepare_dirs()

    lock_path = CONTROL / SNAP_LOCK_REL
    if lock_path.exists():
        die(
            "snapshot lock already exists; "
            "do not silently regenerate"
        )

    rows = []

    for task in manifest["tasks"]:
        tid = task["task_id"]

        print()
        print("=" * 72)
        print("Preparing", tid)
        print("=" * 72)

        row = base.create_snapshot(task)
        rows.append(row)

        print(
            f"{tid}: buggy oracle OK, "
            f"snapshot={row['snapshot_head'][:12]}"
        )

    lock = {
        "status": "prepared_before_any_trajrel_heldout_pilot_run",
        "experiment": "trajrel_heldout_pilot_v1",
        "production_freeze": FREEZE,
        "protocol_lock_commit": lock_commit(PROTOCOL_REL),
        "selection_lock_commit": lock_commit(SELECTION_REL),
        "execution_lock_commit": lock_commit(EXEC_REL),
        "model": base.MODEL,
        "opencode_version": base.OPENCODE_VERSION,
        "task_count": len(rows),
        "tasks": rows,
    }

    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    print()
    print("Snapshot lock written:", lock_path)
    print("NO benchmark agent was run.")


def snapshot_lock():
    p = CONTROL / SNAP_LOCK_REL

    if not p.exists():
        die("snapshot lock missing; run --prepare")

    verify_committed_unchanged(SNAP_LOCK_REL)

    d = json.loads(p.read_text())

    if d.get("status") != "prepared_before_any_trajrel_heldout_pilot_run":
        die("snapshot lock status mismatch")

    if d.get("production_freeze") != FREEZE:
        die("snapshot freeze mismatch")

    return d


base.snapshot_lock = snapshot_lock


def preflight():
    locked_manifest()
    base.prepare_runtime()
    snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    print()
    print("=" * 72)
    print("TRAJREL HELDOUT PILOT V1 PREFLIGHT PASSED")
    print("=" * 72)
    print("freeze :", FREEZE)
    print("tasks  : 10")
    print("next   : U04-OFF")
    print("No benchmark agent was run.")


def verify_wire(item: str):
    out = ROOT / "results" / item
    wire = out / "codex-wire"

    inbound = list(wire.glob("*_http_inbound_request.json"))

    if not inbound:
        die(
            f"{item} completed but has no exact inbound wire capture"
        )

    metrics = json.loads((out / "metrics.json").read_text())
    metrics["wire_inbound_request_count"] = len(inbound)
    metrics["wire_capture_local_only"] = True
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    print(
        f"Wire capture: OK "
        f"(inbound requests={len(inbound)})"
    )


def run_next():
    manifest = locked_manifest()

    base.prepare_runtime()
    snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    for item in manifest["run_order"]:
        out = ROOT / "results" / item

        if (out / "metrics.json").exists():
            continue

        if (out / "AGENT_STARTED").exists():
            die(
                f"{item} started but did not finish. "
                "DO NOT rerun automatically."
            )

        tid, cond = item.split("-", 1)
        task = base.task_by_id(manifest, tid)

        print()
        print("=" * 72)
        print("RUNNING TRAJREL HELDOUT PILOT:", item)
        print("=" * 72)

        base.run_condition(task, cond)
        verify_wire(item)

        print()
        print(
            "Condition consumed. Do not rerun it. "
            "Do not inspect held-out outcomes until all ten "
            "pilot conditions are consumed."
        )
        return

    print("All 10 OFF pilot conditions already consumed.")


def status():
    manifest = locked_manifest()

    print("===== TRAJREL HELDOUT PILOT V1 STATUS =====")

    for item in manifest["run_order"]:
        out = ROOT / "results" / item
        metrics = out / "metrics.json"

        if metrics.exists():
            print(f"{item:8} COMPLETE")
        elif (out / "AGENT_STARTED").exists():
            print(f"{item:8} STARTED-INCOMPLETE")
        else:
            print(f"{item:8} PENDING")

    print()
    print(
        "Outcome metrics intentionally hidden "
        "until all pilot conditions are consumed."
    )


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)

    g.add_argument("--prepare", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--run-next", action="store_true")
    g.add_argument("--status", action="store_true")

    args = ap.parse_args()

    if args.prepare:
        prepare_snapshots()
    elif args.preflight:
        preflight()
    elif args.run_next:
        run_next()
    else:
        status()


if __name__ == "__main__":
    main()
