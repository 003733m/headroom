#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

EXHAUSTIVE = HERE / "exhaustive_discovery_results.json"
OUT = HERE / "second_unseen_adopted_bridge_selection.json"

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"
VERSION = "second-unseen-adopted-v1"
MAX_TASKS = 12

USED_MANIFESTS = [
    ROOT / "results/v3/prospective_task_manifest.json",
    HERE / "natural_transfer_manifest.json",
    HERE / "historical_difficulty_screen_manifest.json",
    HERE / "adopted_bridge_holdout_selection.json",
]


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def collect_source_commits(obj: Any) -> set[str]:
    found: set[str] = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {
                "source_commit",
                "source_sha",
                "snapshot_commit",
                "snapshot_sha",
            } and isinstance(v, str) and len(v) >= 7:
                found.add(v)
            found.update(collect_source_commits(v))

    elif isinstance(obj, list):
        for v in obj:
            found.update(collect_source_commits(v))

    return found


def rank_key(source_commit: str) -> str:
    return hashlib.sha256(
        f"{VERSION}:{source_commit}".encode()
    ).hexdigest()


protocol = HERE / "second_unseen_adopted_bridge_validation_protocol.json"

if not protocol.exists():
    raise SystemExit("locked protocol missing")

# Protocol must already be committed and clean.
if subprocess.run(
    ["git", "ls-files", "--error-unmatch", str(protocol)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode:
    raise SystemExit("protocol is not tracked")

if git("status", "--porcelain=v1", "--", str(protocol)):
    raise SystemExit("protocol differs from committed version")

protocol_data = json.loads(protocol.read_text())

if protocol_data["status"] != "locked_before_any_second_unseen_agent_run":
    raise SystemExit("protocol status mismatch")

if protocol_data["production_freeze"] != FREEZE:
    raise SystemExit("production freeze mismatch")


used: set[str] = set()
used_by_file: dict[str, list[str]] = {}

for p in USED_MANIFESTS:
    if not p.exists():
        raise SystemExit(f"required prior manifest missing: {p}")

    commits = collect_source_commits(json.loads(p.read_text()))
    used.update(commits)
    used_by_file[str(p)] = sorted(commits)


data = json.loads(EXHAUSTIVE.read_text())

if not isinstance(data, dict):
    raise SystemExit("unexpected exhaustive result shape")

tasks = data.get("evaluable_tasks")

if not isinstance(tasks, list):
    raise SystemExit("missing evaluable_tasks")

eligible_pool = []

for row in tasks:
    # IMPORTANT:
    # Selection deliberately does NOT inspect:
    #   baseline_opportunity
    #   rescue_task
    #   regression_task
    #   critical_summary
    #   critical_decisions
    #   off_plan / v3_plan
    #
    # Only historical validation status + previous agent usage matter.
    if row.get("status") != "evaluated":
        continue

    sha = row.get("source_commit")

    if not isinstance(sha, str) or len(sha) < 7:
        continue

    if sha in used:
        continue

    required = (
        "subject",
        "production_files",
        "test_files",
        "failing_nodeids",
        "prompt",
    )

    if any(k not in row for k in required):
        continue

    eligible_pool.append(row)


eligible_pool.sort(
    key=lambda r: (
        rank_key(r["source_commit"]),
        r["source_commit"],
    )
)

selected = eligible_pool[:MAX_TASKS]

if len(selected) < MAX_TASKS:
    raise SystemExit(
        f"only {len(selected)} unused validated candidates; "
        f"expected at least {MAX_TASKS}"
    )


out_tasks = []

for i, row in enumerate(selected, 1):
    prompt = row["prompt"]

    out_tasks.append({
        "task_id": f"U{i:02d}",
        "source_commit": row["source_commit"],
        "rank_sha256": rank_key(row["source_commit"]),
        "subject": row["subject"],
        "production_files": row["production_files"],
        "test_files": row["test_files"],
        "failing_nodeids": row["failing_nodeids"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(
            prompt.encode()
        ).hexdigest(),
    })


output = {
    "status": "selected_before_any_second_unseen_agent_run",
    "experiment": "second_unseen_adopted_bridge_validation",
    "selection_version": VERSION,
    "production_freeze": FREEZE,
    "protocol_commit": git(
        "rev-list",
        "-1",
        "HEAD",
        "--",
        str(protocol),
    ),
    "candidate_source": str(EXHAUSTIVE),
    "candidate_count_before_previous_agent_exclusion": len(tasks),
    "previously_used_source_commit_count": len(used),
    "unused_validated_candidate_count": len(eligible_pool),
    "maximum_screen_tasks": MAX_TASKS,
    "ordering": (
        "ascending sha256("
        "'second-unseen-adopted-v1:' + source_commit)"
    ),
    "selection_fields_used": [
        "status",
        "source_commit",
        "subject",
        "production_files",
        "test_files",
        "failing_nodeids",
        "prompt",
    ],
    "selection_fields_explicitly_not_used": [
        "baseline_opportunity",
        "rescue_task",
        "regression_task",
        "critical_record_count",
        "critical_summary",
        "critical_decisions",
        "off_plan",
        "v3_plan",
    ],
    "prior_agent_manifest_sources": used_by_file,
    "tasks": out_tasks,
}

OUT.write_text(
    json.dumps(output, indent=2) + "\n"
)

print("candidate universe:", len(tasks))
print("previously used commits:", len(used))
print("unused validated pool:", len(eligible_pool))
print()
print("LOCKED FIRST 12")
print("-" * 100)

for task in out_tasks:
    print(
        task["task_id"],
        task["source_commit"],
        task["rank_sha256"][:12],
        task["subject"][:70],
    )

print()
print("selection:", OUT)
