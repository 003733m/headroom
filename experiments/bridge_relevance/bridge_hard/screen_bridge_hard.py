#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

V3_FREEZE = "7c4fe80c19b3a9537f395f515a64432e41ee1d43"
MAX_SELECTED = 4

OUT_DIR = ROOT / "experiments/bridge_relevance/bridge_hard"
OUT_JSON = OUT_DIR / "screening.json"

TREATMENT_FILES = {
    "headroom/trajectory_relevance.py",
    "headroom/transforms/content_router.py",
    "headroom/proxy/handlers/openai.py",
    "tests/test_trajectory_relevance.py",
    "tests/test_trajectory_relevance_v2.py",
}

RUNTIME_PY = Path("/tmp/headroom-v3-runtime/.venv/bin/python")

HEX40 = re.compile(r"\b[0-9a-f]{40}\b")
FAILED_NODE = re.compile(
    r"^FAILED\s+(.+?::(?P<test>[A-Za-z_][A-Za-z0-9_\[\]\-]*))"
)
EXCEPTION = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b"
)


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"command failed rc={cp.returncode}: {args}\n{cp.stdout}"
        )
    return cp


def git(*args: str, check: bool = True) -> str:
    cp = run(["git", *args], cwd=ROOT)
    if check and cp.returncode != 0:
        raise RuntimeError(cp.stdout)
    return cp.stdout


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def subject(commit: str) -> str:
    return git("show", "-s", "--format=%s", commit).strip()


def parent(commit: str) -> str:
    return git("rev-parse", f"{commit}^").strip()


def changed_files(commit: str) -> list[str]:
    return [
        x.strip()
        for x in git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        ).splitlines()
        if x.strip()
    ]


def production_files(files: list[str]) -> list[str]:
    return sorted(
        f
        for f in files
        if f.startswith("headroom/")
        and f.endswith(".py")
    )


def test_files(files: list[str]) -> list[str]:
    return sorted(
        f
        for f in files
        if f.startswith("tests/")
        and f.endswith(".py")
    )


def is_fix_subject(s: str) -> bool:
    low = s.lower().strip()
    return (
        low.startswith("fix")
        or low.startswith("bugfix")
        or " fix " in f" {low} "
        or low.startswith("hotfix")
    )


def collect_previously_recorded_shas() -> set[str]:
    """
    Over-exclusion is intentional.

    Any full SHA already written anywhere under experiments/bridge_relevance
    is considered previously used/screened and is excluded.
    """
    used: set[str] = set()

    base = ROOT / "experiments/bridge_relevance"
    for path in base.rglob("*"):
        if not path.is_file():
            continue

        # Exclude the current protocol/script itself from irrelevant scans.
        if path == OUT_JSON:
            continue

        try:
            if path.stat().st_size > 20_000_000:
                continue
            text = path.read_text(errors="replace")
        except Exception:
            continue

        used.update(HEX40.findall(text))

    # Never treat the freeze itself as a candidate.
    used.add(V3_FREEZE)
    return used


def make_prompt(tests: list[str]) -> str:
    joined = ", ".join(tests)

    return (
        f"Fix the regression covered by the tests in: {joined}. "
        "Run the relevant tests, diagnose the root cause, implement the "
        "smallest production fix, and verify it. "
        "Use shell `rg -nH` instead of the dedicated Grep tool for repository "
        "text searches. Otherwise debug normally. "
        "Do not inspect git history or historical commits."
    )


def patch_for_production(commit: str, prod: list[str]) -> str:
    par = parent(commit)
    return git(
        "diff",
        "--binary",
        par,
        commit,
        "--",
        *prod,
    )


def prepare_freeze_base(base: Path) -> None:
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    archive = subprocess.Popen(
        ["git", "archive", V3_FREEZE],
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None

    tar = subprocess.run(
        ["tar", "-x", "-C", str(base)],
        stdin=archive.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    archive.stdout.close()
    rc = archive.wait()

    if rc != 0 or tar.returncode != 0:
        raise RuntimeError(
            f"git archive/tar failed: archive={rc}, tar={tar.returncode}\n"
            f"{tar.stdout}"
        )

    # uv.lock is ignored in this repository and therefore absent from git archive.
    lock = ROOT / "uv.lock"
    if lock.exists():
        shutil.copy2(lock, base / "uv.lock")


def pytest_run(work: Path, tests: list[str]) -> tuple[int, str]:
    if not RUNTIME_PY.exists():
        raise RuntimeError(
            f"Frozen runtime Python not found: {RUNTIME_PY}"
        )

    env = os.environ.copy()

    # Ensure imports come from the candidate snapshot, not installed source.
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(work)
        if not old_pp
        else f"{work}:{old_pp}"
    )

    cp = run(
        [
            str(RUNTIME_PY),
            "-m",
            "pytest",
            *tests,
            "-q",
            "--tb=short",
        ],
        cwd=work,
        env=env,
    )
    return cp.returncode, cp.stdout


def failure_evidence(text: str) -> tuple[list[str], list[str], list[str]]:
    nodeids: list[str] = []
    tests: list[str] = []

    for line in text.splitlines():
        m = FAILED_NODE.match(line.strip())
        if not m:
            continue
        nodeids.append(m.group(1))
        tests.append(m.group("test"))

    exceptions = sorted(
        {
            x
            for x in EXCEPTION.findall(text)
            if x not in {
                "AssertionError",
                # AssertionError by itself is generally too generic to be useful
                # as a bridge identifier.
            }
        }
    )

    return sorted(set(nodeids)), sorted(set(tests)), exceptions


def candidate_commits() -> list[str]:
    out = git(
        "rev-list",
        "--no-merges",
        "--date-order",
        f"{V3_FREEZE}^",
    )
    return [x.strip() for x in out.splitlines() if x.strip()]


def main() -> int:
    print("=" * 100)
    print("V3 BRIDGE-HARD CHALLENGE SCREENING")
    print("=" * 100)
    print("V3 relevance is NOT evaluated by this script.")
    print("No coding agent is run.")
    print("Stop rule: first four eligible tasks.")
    print()

    used = collect_previously_recorded_shas()

    print(f"Previously recorded full SHAs excluded: {len(used)}")

    base = Path("/tmp/headroom-v3-bridge-hard-base")
    prepare_freeze_base(base)

    reject = Counter()
    selected: list[dict] = []
    selected_prod: set[str] = set()
    screened: list[dict] = []

    for commit in candidate_commits():
        if len(selected) >= MAX_SELECTED:
            break

        s = subject(commit)
        rec: dict = {
            "source_commit": commit,
            "subject": s,
        }

        if commit in used:
            reject["previously_used_or_screened"] += 1
            continue

        if not is_fix_subject(s):
            reject["not_fix_subject"] += 1
            continue

        files = changed_files(commit)
        prod = production_files(files)
        tests = test_files(files)

        rec["production_files"] = prod
        rec["test_files"] = tests

        if not prod or not tests:
            reject["missing_prod_or_test"] += 1
            continue

        if len(prod) > 3 or len(tests) > 2:
            reject["too_broad"] += 1
            continue

        if any(f in TREATMENT_FILES for f in files):
            reject["touches_v3_treatment"] += 1
            continue

        if set(prod) & selected_prod:
            reject["production_file_overlap"] += 1
            continue

        patch = patch_for_production(commit, prod)
        if not patch.strip():
            reject["empty_production_patch"] += 1
            continue

        with tempfile.TemporaryDirectory(
            prefix="headroom-bridge-hard-"
        ) as td:
            work = Path(td) / "repo"
            shutil.copytree(base, work)

            patch_path = Path(td) / "production.patch"
            patch_path.write_text(patch)

            # Does the historical production fix cleanly reverse on V3 freeze?
            check = run(
                [
                    "git",
                    "apply",
                    "--reverse",
                    "--check",
                    str(patch_path),
                ],
                cwd=work,
            )

            if check.returncode != 0:
                reject["reverse_patch_not_clean"] += 1
                continue

            apply = run(
                [
                    "git",
                    "apply",
                    "--reverse",
                    str(patch_path),
                ],
                cwd=work,
            )

            if apply.returncode != 0:
                reject["reverse_patch_apply_failed"] += 1
                continue

            buggy_rc, buggy_out = pytest_run(work, tests)

            # We only want deterministic normal pytest failure, not collection /
            # environment / interrupted errors.
            if buggy_rc != 1:
                reject[f"buggy_oracle_rc_{buggy_rc}"] += 1
                continue

            nodeids, test_names, exceptions = failure_evidence(
                buggy_out
            )

            if not (1 <= len(nodeids) <= 2):
                reject[
                    "buggy_failure_count_not_1_or_2"
                ] += 1
                continue

            prompt = make_prompt(tests)

            safe_bridges = [
                {
                    "kind": "test_name",
                    "token": t,
                }
                for t in test_names
                if t and t not in prompt
            ]

            safe_bridges.extend(
                {
                    "kind": "exception",
                    "token": e,
                }
                for e in exceptions
                if e and e not in prompt
            )

            # Deduplicate while preserving deterministic order.
            seen = set()
            dedup_bridges = []
            for b in safe_bridges:
                key = (b["kind"], b["token"])
                if key in seen:
                    continue
                seen.add(key)
                dedup_bridges.append(b)
            safe_bridges = dedup_bridges

            if not safe_bridges:
                reject["no_safe_bridge_outside_prompt"] += 1
                continue

            # Verify the changed tests still PASS on pristine V3 freeze.
            pristine_rc, pristine_out = pytest_run(base, tests)
            if pristine_rc != 0:
                reject["freeze_baseline_tests_not_pass"] += 1
                continue

            task_id = f"H{len(selected) + 1}"

            entry = {
                "task_id": task_id,
                "source_commit": commit,
                "subject": s,
                "production_files": prod,
                "test_files": tests,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "buggy_oracle": {
                    "exit_code": buggy_rc,
                    "failing_nodeids": nodeids,
                    "failure_count": len(nodeids),
                },
                "safe_bridge_candidates_predeclared": safe_bridges,
                "selection_rationale": (
                    "Historical regression with deterministic 1-2 test failures; "
                    "safe bridge identifier becomes available only after running "
                    "the relevant tests and is absent from the user prompt."
                ),
            }

            selected.append(entry)
            selected_prod.update(prod)

            print()
            print(
                f"SELECTED {task_id}: {commit[:12]}  {s}"
            )
            print(f"  production: {', '.join(prod)}")
            print(f"  tests     : {', '.join(tests)}")
            print(
                "  failures  : "
                + ", ".join(nodeids)
            )
            print(
                "  bridges   : "
                + ", ".join(
                    f"{x['kind']}={x['token']}"
                    for x in safe_bridges
                )
            )
            print(f"  prompt sha: {entry['prompt_sha256']}")

            # Save bounded debugging output only for selected tasks.
            rec["selected"] = True
            rec["task_id"] = task_id
            rec["buggy_oracle_tail"] = "\n".join(
                buggy_out.splitlines()[-100:]
            )
            screened.append(rec)

    result = {
        "status": (
            "selection_complete"
            if len(selected) == MAX_SELECTED
            else "selection_exhausted_before_four"
        ),
        "v3_algorithm_freeze": V3_FREEZE,
        "selection_protocol": str(
            OUT_DIR / "selection_protocol.json"
        ),
        "selection_did_not_evaluate_v3": True,
        "selection_did_not_run_agent": True,
        "stop_rule": f"first {MAX_SELECTED} eligible tasks",
        "selected_count": len(selected),
        "selected": selected,
        "rejection_counts": dict(
            sorted(reject.items())
        ),
        "selected_debug": screened,
    }

    OUT_JSON.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print("SCREENING COMPLETE")
    print("=" * 100)
    print(f"Selected: {len(selected)}/{MAX_SELECTED}")
    print(f"Output  : {OUT_JSON}")
    print()

    print("Rejections:")
    for k, v in sorted(
        reject.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(f"  {v:4d}  {k}")

    print()
    print("IMPORTANT:")
    print("  No V3 relevance score was inspected.")
    print("  No V3 KEEP/DROP result was inspected.")
    print("  No coding agent was run.")
    print("  Do not replace selected tasks based on later V3 outcomes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
