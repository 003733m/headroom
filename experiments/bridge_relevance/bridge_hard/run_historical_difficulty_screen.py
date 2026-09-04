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

BASE_PATH = HERE / "run_natural_transfer.py"
EXEC_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "historical_difficulty_screen_execution.json"
)
SNAP_LOCK_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "historical_difficulty_screen_snapshot_lock.json"
)

ROOT = Path("/tmp/headroom-v3-difficulty-screen")
SNAPSHOTS = Path("/tmp/headroom-v3-difficulty-screen-locked")

spec = importlib.util.spec_from_file_location(
    "natural_transfer_base",
    BASE_PATH,
)

if spec is None or spec.loader is None:
    raise SystemExit("cannot import natural-transfer runner")

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# ------------------------------------------------------------
# Reuse the already-debugged benchmark infrastructure.
# Only experiment-specific state is changed.
# ------------------------------------------------------------

base.CONTROL = CONTROL
base.ROOT = ROOT
base.SNAPSHOTS = SNAPSHOTS
base.SNAP_LOCK_REL = SNAP_LOCK_REL


# Do not leak the caller's control-repo virtualenv into the wrapped
# OpenCode process. The benchmark runtime has its own frozen venv.
os.environ.pop("VIRTUAL_ENV", None)


def screening_validate_agent_isolation(out: Path, run_dir: Path):
    """
    Strict snapshot-isolation check with one narrow exception.

    The original validator rejected any textual occurrence of CONTROL,
    including uv's launcher warning about an inherited VIRTUAL_ENV.
    That warning is runner metadata, not an agent tool/read/test target.

    Every other CONTROL occurrence remains invalid, and every captured
    rg cwd must stay inside the locked condition snapshot.
    """
    expected = run_dir.resolve()
    control = CONTROL.resolve()

    transcript_path = out / "opencode.txt"
    transcript = (
        transcript_path.read_text(errors="replace")
        if transcript_path.exists()
        else ""
    )

    problems = []
    ignored_benign = []

    for line in transcript.splitlines():
        if str(control) not in line:
            continue

        if (
            line.startswith("warning: `VIRTUAL_ENV=")
            and "does not match the project environment path" in line
        ):
            ignored_benign.append(line)
            continue

        problems.append(
            "non-whitelisted CONTROL_REPO transcript reference: "
            + line[:500]
        )

    capdir = out / "rg-captures"

    for meta in sorted(capdir.glob("*.meta")):
        cwd_line = None

        for line in meta.read_text(
            errors="replace"
        ).splitlines():
            if line.startswith("cwd="):
                cwd_line = line[4:]
                break

        if cwd_line is None:
            problems.append(
                f"{meta.name}: missing cwd"
            )
            continue

        got = Path(cwd_line).resolve()

        if not (
            got == expected
            or expected in got.parents
        ):
            problems.append(
                f"{meta.name}: rg cwd escaped snapshot: {got}"
            )

    result = {
        "expected_run_dir": str(expected),
        "control_repo_forbidden": True,
        "benign_virtual_env_warning_ignored": bool(
            ignored_benign
        ),
        "ignored_benign_lines": ignored_benign,
        "rg_capture_count": len(
            list(capdir.glob("*.meta"))
        ),
        "valid": not problems,
        "problems": problems,
    }

    (out / "isolation.json").write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    if problems:
        (out / "INVALID_RUN").write_text(
            "\n".join(problems) + "\n"
        )
        die(
            "agent isolation invariant failed: "
            + "; ".join(problems)
        )

    print("Agent isolation: OK")


base.validate_agent_isolation = (
    screening_validate_agent_isolation
)

# Kept only for legacy metadata fields inside reused helpers.
base.MANIFEST_COMMIT = "abb3e81c"
base.MANIFEST_REL = EXEC_REL


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


def execution_lock_commit() -> str:
    return git_capture(
        "log", "-1", "--format=%H", "--", EXEC_REL
    ).strip()


def locked_manifest():
    path = CONTROL / EXEC_REL

    if not path.exists():
        die("difficulty execution manifest missing")

    # Must already exist in committed HEAD before any agent run.
    cp = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{EXEC_REL}"],
        cwd=CONTROL,
    )
    if cp.returncode:
        die("execution manifest is not committed in HEAD")

    status = git_capture(
        "status",
        "--porcelain=v1",
        "--",
        EXEC_REL,
    ).strip()

    if status:
        die("execution manifest differs from committed HEAD")

    d = json.loads(path.read_text())

    if d.get("status") != (
        "locked_before_any_difficulty_screen_agent_run"
    ):
        die("execution manifest status mismatch")

    if d.get("algorithm_freeze") != base.FREEZE:
        die("algorithm freeze mismatch")

    if d.get("task_count") != 24:
        die("expected exactly 24 screening tasks")

    expected = [
        f"G{i:02d}-OFF"
        for i in range(1, 25)
    ]

    if d.get("run_order") != expected:
        die("screening run order mismatch")

    if any(
        not x.endswith("-OFF")
        for x in d["run_order"]
    ):
        die("screen contains a non-OFF condition")

    return d


# Make every reused base helper resolve our locked manifest.
base.locked_manifest = locked_manifest


def prepare_snapshots():
    manifest = locked_manifest()
    base.prepare_runtime()
    base.prepare_dirs()

    lock_path = CONTROL / SNAP_LOCK_REL

    if lock_path.exists():
        die(
            "snapshot lock already exists; "
            "do not silently regenerate it"
        )

    rows = []

    for task in manifest["tasks"]:
        tid = task["task_id"]

        print()
        print("=" * 72)
        print(f"Preparing {tid}")
        print("=" * 72)

        row = base.create_snapshot(task)
        rows.append(row)

        print(
            f"{tid}: buggy oracle OK, "
            f"snapshot={row['snapshot_head'][:12]}"
        )

    lock = {
        "status": (
            "prepared_before_any_difficulty_screen_agent_run"
        ),
        "algorithm_freeze": base.FREEZE,
        "selection_lock_commit": manifest[
            "selection_lock_commit"
        ],
        "execution_lock_commit": execution_lock_commit(),
        "model": base.MODEL,
        "opencode_version": base.OPENCODE_VERSION,
        "task_count": len(rows),
        "tasks": rows,
    }

    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print()
    print("=" * 72)
    print("DIFFICULTY SCREEN SNAPSHOTS PREPARED")
    print("24 buggy snapshots validated.")
    print("NO BENCHMARK AGENT WAS RUN.")
    print(f"snapshot lock: {lock_path}")
    print("=" * 72)


# Base snapshot_lock() / verify_snapshots() can now use our path.
base.SNAP_LOCK_REL = SNAP_LOCK_REL


_original_write_metrics = base.write_metrics


def screening_write_metrics(
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

    manifest = locked_manifest()

    d["evaluation_type"] = (
        "historical_agent_difficulty_screen"
    )
    d["screening_only"] = True
    d["confirmatory_estimate"] = False
    d["selection_lock_commit"] = manifest[
        "selection_lock_commit"
    ]
    d["execution_lock_commit"] = (
        execution_lock_commit()
    )

    # Screening must never accidentally become treatment.
    if d.get("condition_flag") != 0:
        die("screening condition unexpectedly enabled treatment")

    p.write_text(
        json.dumps(
            d,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print()
    print(
        f"SCREEN RESULT {task['task_id']}: "
        f"{'PASS' if d['task_pass'] else 'FAIL'} "
        f"| req={d['requests']} "
        f"| rg={d['rg_capture_count']} "
        f"| input={d['input_tokens_original_sum']}"
    )


base.write_metrics = screening_write_metrics


def show_status():
    d = locked_manifest()

    print("===== HISTORICAL DIFFICULTY SCREEN =====")

    complete = 0
    passed = 0
    failed = 0

    for item in d["run_order"]:
        out = ROOT / "results" / item

        if (out / "metrics.json").exists():
            x = json.loads(
                (out / "metrics.json").read_text()
            )

            complete += 1

            if x["task_pass"]:
                passed += 1
                outcome = "PASS"
            else:
                failed += 1
                outcome = "FAIL"

            print(
                f"{item:8} COMPLETE {outcome:4} "
                f"req={x['requests']} "
                f"rg={x['rg_capture_count']} "
                f"input={x['input_tokens_original_sum']}"
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
    print(
        f"complete={complete}/24 "
        f"pass={passed} fail={failed}"
    )


def run_next() -> bool:
    manifest = locked_manifest()

    base.prepare_runtime()
    base.snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    for item in manifest["run_order"]:
        tid, cond = item.split("-", 1)
        out = ROOT / "results" / item

        if (out / "metrics.json").exists():
            continue

        if (out / "AGENT_STARTED").exists():
            die(
                f"{item} started but did not complete. "
                "DO NOT rerun automatically."
            )

        task = base.task_by_id(
            manifest,
            tid,
        )

        print()
        print("=" * 72)
        print(
            f"RUNNING DIFFICULTY SCREEN: {item}"
        )
        print("=" * 72)

        base.run_condition(
            task,
            cond,
        )

        return True

    print("All 24 OFF screening conditions complete.")
    return False


def run_all():
    """
    Sequential screening only.

    A normal oracle FAIL is a valid difficulty result and does NOT stop
    the screen. Any harness/preflight/isolation error raises and stops
    execution immediately.
    """
    while run_next():
        show_status()


def preflight():
    # Reuse exact proxy/OpenCode/snapshot checks from the successful
    # natural-transfer harness after replacing its experiment globals.
    base.preflight()

    print()
    print("=" * 72)
    print("HISTORICAL DIFFICULTY SCREEN PREFLIGHT OK")
    print("No benchmark agent was run.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()

    g = parser.add_mutually_exclusive_group(
        required=True
    )

    g.add_argument(
        "--prepare",
        action="store_true",
    )
    g.add_argument(
        "--preflight",
        action="store_true",
    )
    g.add_argument(
        "--status",
        action="store_true",
    )
    g.add_argument(
        "--run-next",
        action="store_true",
    )
    g.add_argument(
        "--run-all",
        action="store_true",
    )

    args = parser.parse_args()

    if args.prepare:
        prepare_snapshots()
    elif args.preflight:
        preflight()
    elif args.status:
        show_status()
    elif args.run_next:
        run_next()
    else:
        run_all()


if __name__ == "__main__":
    main()
