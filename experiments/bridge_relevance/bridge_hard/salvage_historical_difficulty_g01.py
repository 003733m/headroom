#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

CONTROL = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

HERE = (
    CONTROL
    / "experiments/bridge_relevance/bridge_hard"
)

RUNNER = HERE / "run_historical_difficulty_screen.py"

spec = importlib.util.spec_from_file_location(
    "difficulty_screen",
    RUNNER,
)

if spec is None or spec.loader is None:
    raise SystemExit("cannot import difficulty screen")

screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

OUT = (
    Path("/tmp/headroom-v3-difficulty-screen")
    / "results/G01-OFF"
)

RUN_DIR = (
    Path("/tmp/headroom-v3-difficulty-screen")
    / "runs/G01-OFF"
)


def die(msg: str):
    raise SystemExit("ERROR: " + msg)


def ts(value: str) -> float:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).timestamp()


def isum(rows, key):
    vals = [
        x.get(key)
        for x in rows
        if isinstance(
            x.get(key),
            (int, float),
        )
    ]
    return sum(vals) if vals else None


def main():
    if (OUT / "metrics.json").exists():
        die("G01 metrics already exist; refusing salvage twice")

    start_file = OUT / "AGENT_STARTED"

    if not start_file.exists():
        die("AGENT_STARTED missing")

    marker = json.loads(start_file.read_text())

    if marker.get("task") != "G01":
        die("wrong task marker")
    if marker.get("condition") != "OFF":
        die("wrong condition marker")
    if marker.get("flag") != 0:
        die("G01 screening was not OFF")

    # Preserve the original false-positive evidence verbatim.
    for name in ("isolation.json", "INVALID_RUN"):
        src = OUT / name
        if src.exists():
            shutil.copy2(
                src,
                OUT / (
                    name
                    + ".original_false_positive"
                ),
            )

    # Re-evaluate isolation under the narrow amended rule.
    screen.screening_validate_agent_isolation(
        OUT,
        RUN_DIR,
    )

    isolation = json.loads(
        (OUT / "isolation.json").read_text()
    )

    if not isolation.get("valid"):
        die("amended isolation still fails")

    # INVALID_RUN now refers only to the original rejected validator
    # result. It has already been preserved above.
    (OUT / "INVALID_RUN").unlink(
        missing_ok=True
    )

    manifest = screen.locked_manifest()
    task = screen.base.task_by_id(
        manifest,
        "G01",
    )

    request_file = OUT / "requests.jsonl"

    if not request_file.exists():
        die("requests.jsonl missing")

    all_rows = []

    for line in request_file.read_text(
        errors="replace"
    ).splitlines():
        if not line.strip():
            continue
        all_rows.append(json.loads(line))

    start_time = float(marker["time"])

    before = []
    measured = []

    for row in all_rows:
        row_ts = ts(row["timestamp"])

        if row_ts >= start_time:
            measured.append(row)
        else:
            before.append(row)

    if not measured:
        die("no post-AGENT_STARTED requests")

    if before and max(
        ts(x["timestamp"])
        for x in before
    ) >= start_time:
        die("pre-agent partition invariant failed")

    if min(
        ts(x["timestamp"])
        for x in measured
    ) < start_time:
        die("measured request partition invariant failed")

    (OUT / "measured_requests.jsonl").write_text(
        "".join(
            json.dumps(
                x,
                ensure_ascii=False,
            ) + "\n"
            for x in measured
        )
    )

    # Final oracle: no LLM/agent invocation.
    with (OUT / "oracle.txt").open("w") as f:
        cp = subprocess.run(
            [
                "uv", "run",
                "--frozen",
                "pytest", "-q",
                *task["test_files"],
            ],
            cwd=RUN_DIR,
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    oracle_rc = cp.returncode

    status = subprocess.check_output(
        [
            "git", "status",
            "--porcelain=v1",
        ],
        cwd=RUN_DIR,
        text=True,
    )

    (OUT / "git-status.txt").write_text(
        status
    )

    patch = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=RUN_DIR,
        stdout=subprocess.PIPE,
    ).stdout

    (OUT / "agent.patch").write_bytes(
        patch
    )

    original = isum(
        measured,
        "input_tokens_original",
    )
    optimized = isum(
        measured,
        "input_tokens_optimized",
    )

    rg_files = sorted(
        (OUT / "rg-captures").glob(
            "*.stdout"
        )
    )

    changed = [
        line[3:].strip()
        if len(line) >= 4
        else line.strip()
        for line in status.splitlines()
        if line.strip()
    ]

    metrics = {
        "task": "G01",
        "condition": "off",
        "condition_flag": 0,
        "evaluation_type": (
            "historical_agent_difficulty_screen"
        ),
        "screening_only": True,
        "confirmatory_estimate": False,
        "v3_freeze_sha": screen.base.FREEZE,
        "model": screen.base.MODEL,
        "opencode_version": (
            screen.base.OPENCODE_VERSION
        ),
        "task_pass": oracle_rc == 0,
        "oracle_exit": oracle_rc,

        # Process exit was lost because the original validator raised
        # before metrics serialization. Do not invent it.
        "agent_exit": None,
        "agent_exit_unavailable_reason": (
            "original false-positive isolation validator "
            "raised before metrics serialization"
        ),

        # Likewise do not estimate wall time after the fact.
        "wall_seconds": None,
        "wall_seconds_unavailable_reason": (
            "original false-positive isolation validator "
            "raised before wall_seconds artifact was written"
        ),

        "requests": len(measured),
        "pre_agent_request_count": len(before),
        "all_request_log_count": len(all_rows),

        "input_tokens_original_sum": original,
        "input_tokens_optimized_sum": optimized,
        "derived_tokens_saved_sum": (
            original - optimized
            if original is not None
            and optimized is not None
            else None
        ),
        "reported_tokens_saved_sum": isum(
            measured,
            "tokens_saved",
        ),
        "output_tokens_sum": isum(
            measured,
            "output_tokens",
        ),

        # OFF screening: treatment cannot activate.
        "treatment_activated": False,
        "relevance_split_units": None,
        "search_relevance_chains": None,

        "rg_capture_count": len(rg_files),
        "rg_capture_bytes": sum(
            p.stat().st_size
            for p in rg_files
        ),
        "changed_files": changed,

        "salvaged_after_validator_false_positive": True,
        "agent_rerun": False,
        "request_partition_basis": (
            "request timestamp >= pre-existing "
            "AGENT_STARTED.time"
        ),
        "isolation_valid_under_amended_rule": True
    }

    (OUT / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    salvage = {
        "status": "salvaged_without_agent_rerun",
        "task": "G01",
        "condition": "OFF",
        "agent_started_epoch": start_time,
        "request_count_before_agent": len(before),
        "request_count_measured_agent": len(measured),
        "oracle_exit": oracle_rc,
        "task_pass": oracle_rc == 0,
        "isolation": isolation,
        "metrics_sha256": hashlib.sha256(
            (OUT / "metrics.json").read_bytes()
        ).hexdigest()
    }

    (OUT / "salvage.json").write_text(
        json.dumps(
            salvage,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print("=" * 72)
    print("G01 ORIGINAL RUN SALVAGED")
    print("=" * 72)
    print("agent rerun        : NO")
    print("pre-agent requests :", len(before))
    print("agent requests     :", len(measured))
    print("rg captures        :", len(rg_files))
    print("oracle exit        :", oracle_rc)
    print(
        "screen outcome     :",
        "PASS" if oracle_rc == 0 else "FAIL",
    )
    print("changed files      :", changed)
    print("=" * 72)


if __name__ == "__main__":
    main()
