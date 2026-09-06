#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

CONTROL = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

HERE = CONTROL / "experiments/bridge_relevance/bridge_hard"

SCREEN = HERE / "run_routing_refined_unseen_screen.py"
LOCK_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "routing_refined_unseen_u03_lock.json"
)
FREEZE = "8fa4d92529f47174101e4d6fb288c0fde32e8f7c"


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def git(*args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=CONTROL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if cp.returncode:
        die(cp.stdout)
    return cp.stdout.strip()


# The U03 lock must already exist in committed history and be unchanged.
if subprocess.run(
    ["git", "cat-file", "-e", f"HEAD:{LOCK_REL}"],
    cwd=CONTROL,
).returncode:
    die("U03 lock is not committed")

if git("status", "--porcelain=v1", "--", LOCK_REL):
    die("U03 lock differs from committed HEAD")

lock = json.loads((CONTROL / LOCK_REL).read_text())

if lock.get("status") != "locked_before_u03_on_or_retention_analysis":
    die("unexpected U03 lock status")

if lock.get("production_freeze") != FREEZE:
    die("U03 lock freeze mismatch")

selected = lock.get("selected_task")
if not isinstance(selected, dict) or selected.get("task_id") != "U03":
    die("U03 is not the locked selected task")


# Import the already-frozen routing-refined harness.
spec = importlib.util.spec_from_file_location(
    "routing_refined_screen",
    SCREEN,
)
if spec is None or spec.loader is None:
    die("cannot import routing-refined screen harness")

screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

# Harness itself must point at the same frozen production implementation.
if getattr(screen, "FREEZE", None) != FREEZE:
    die(
        f"screen freeze mismatch: "
        f"{getattr(screen, 'FREEZE', None)!r}"
    )

manifest = screen.locked_manifest()

screen.base.prepare_runtime()
screen.base.snapshot_lock()
screen.base.verify_snapshots()
screen.base.opencode_version()

task = screen.base.task_by_id(manifest, "U03")

if task.get("source_commit") != selected.get("source_commit"):
    die("selected source commit mismatch")

if task.get("prompt_sha256") != selected.get("prompt_sha256"):
    die("selected prompt mismatch")

out = screen.ROOT / "results" / "U03-ON"

if (out / "metrics.json").exists():
    die("U03-ON already completed; DO NOT RERUN")

if (out / "AGENT_STARTED").exists():
    die("U03-ON already started; DO NOT RERUN")

print("=" * 76)
print("RUNNING LOCKED ROUTING-REFINED UNSEEN CONDITION: U03-ON")
print("=" * 76)
print("production freeze:", FREEZE)
print("source commit:", task["source_commit"])
print("lock commit:", git("log", "-1", "--format=%H", "--", LOCK_REL))
print()

# Exactly one natural ON agent run.
screen.base.run_condition(task, "ON")

if not (out / "metrics.json").exists():
    die("run returned without metrics")

metrics = json.loads((out / "metrics.json").read_text())

# Metadata correction only. No scoring or outcome is altered.
metrics["evaluation_type"] = (
    "prospective_routing_refined_unseen_selected_u03_on"
)
metrics["production_freeze"] = FREEZE
metrics["u03_lock_commit"] = git(
    "log", "-1", "--format=%H", "--", LOCK_REL
)

wire_dir = out / "codex-wire"
inbound = (
    list(wire_dir.glob("*_http_inbound_request.json"))
    if wire_dir.exists()
    else []
)

metrics["wire_inbound_request_count"] = len(inbound)
metrics["wire_capture_local_only"] = True

(out / "metrics.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
)

print()
print("U03-ON consumed exactly once.")
print("task_pass:", metrics.get("task_pass"))
print("requests:", metrics.get("requests"))
print("rg captures:", metrics.get("rg_capture_count"))
print("relevance_split_units:", metrics.get("relevance_split_units"))
print("search_relevance_chains:", metrics.get("search_relevance_chains"))
print("treatment_activated:", metrics.get("treatment_activated"))
print("wire inbound requests:", len(inbound))
print("DO NOT RERUN.")
