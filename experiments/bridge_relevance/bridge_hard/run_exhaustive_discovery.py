#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path


ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

FREEZE = "7c4fe80c19b3a9537f395f515a64432e41ee1d43"
PRIOR_TREE = "c0df5b70"

HELPER_PATH = (
    ROOT
    / "experiments/bridge_relevance/bridge_hard/"
      "probe_retrieval_controlled.py"
)

OUT = (
    ROOT
    / "experiments/bridge_relevance/bridge_hard/"
      "exhaustive_discovery_results.json"
)

CHECKPOINT = Path(
    "/tmp/headroom-v3-exhaustive-discovery-checkpoint.jsonl"
)

TREATMENT_FILES = {
    "headroom/trajectory_relevance.py",
    "headroom/transforms/content_router.py",
    "headroom/proxy/handlers/openai.py",
    "tests/test_trajectory_relevance.py",
    "tests/test_trajectory_relevance_v2.py",
}

MAX_PROD_FILES = 3
MAX_TEST_FILES = 2

FIX_RE = re.compile(
    r"(^|[\s(:\[])"
    r"(fix(?:es|ed|ing)?|bugfix|hotfix)\b",
    re.I,
)

FAILED_RE = re.compile(
    r"(?m)^FAILED\s+"
    r"([^\s]+::[^\s]+)"
)

SHA_RE = re.compile(
    r"\b[0-9a-f]{40}\b"
)


# -------------------------------------------------------------------
# Load the exact retrieval-controlled mechanism implementation
# that was frozen and already executed for H1-H4.
# -------------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "retrieval_controlled_helper",
    HELPER_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not load helper: {HELPER_PATH}"
    )

rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def run(args, cwd=None):
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def git(*args):
    cp = run(
        ["git", *args],
        ROOT,
    )

    if cp.returncode:
        raise RuntimeError(cp.stdout)

    return cp.stdout


def subject(commit: str) -> str:
    return git(
        "show",
        "-s",
        "--format=%s",
        commit,
    ).strip()


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


def parent_of(commit: str) -> str | None:
    cp = run(
        [
            "git",
            "rev-parse",
            f"{commit}^",
        ],
        ROOT,
    )

    if cp.returncode:
        return None

    return cp.stdout.strip()


def candidate_file_sets(files):
    prod = [
        x
        for x in files
        if (
            x.startswith("headroom/")
            and x.endswith(".py")
        )
    ]

    tests = [
        x
        for x in files
        if (
            x.startswith("tests/")
            and x.endswith(".py")
        )
    ]

    return sorted(prod), sorted(tests)


def challenge_prompt(test_files):
    joined = ", ".join(test_files)

    return (
        "Fix the regression covered by the tests in: "
        f"{joined}. "
        "Run the relevant tests, diagnose the root cause, "
        "implement the smallest production fix, and verify it. "
        "Use shell `rg -nH` instead of the dedicated Grep tool "
        "for repository text searches. Otherwise debug normally. "
        "Do not inspect git history or historical commits."
    )


def failing_nodeids(text):
    out = []

    for x in FAILED_RE.findall(text):
        if x not in out:
            out.append(x)

    return out


def failing_test_names(nodeids):
    names = []

    for node in nodeids:
        last = node.split("::")[-1]
        last = last.split("[", 1)[0]

        if not last.startswith("test_"):
            continue

        if last not in names:
            names.append(last)

    return names


def bridge_dicts(test_names):
    return [
        {
            "kind": "test_name",
            "token": name,
        }
        for name in test_names
    ]


def collect_prior_recorded_shas():
    """
    This is ONLY a flagging mechanism.

    Candidates found in earlier experiment artifacts are still evaluated.
    They are never excluded from the exhaustive denominator.
    """
    cp = run(
        [
            "git",
            "grep",
            "-h",
            "-E",
            r"[0-9a-f]{40}",
            PRIOR_TREE,
            "--",
            "experiments/bridge_relevance",
        ],
        ROOT,
    )

    if cp.returncode not in (0, 1):
        raise RuntimeError(cp.stdout)

    return set(
        SHA_RE.findall(cp.stdout)
    )


def read_checkpoint():
    records = {}

    if not CHECKPOINT.exists():
        return records

    for raw in CHECKPOINT.read_text().splitlines():
        if not raw.strip():
            continue

        obj = json.loads(raw)

        if obj.get("kind") == "candidate":
            records[obj["source_commit"]] = obj

    return records


def append_checkpoint(obj):
    with CHECKPOINT.open("a") as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def verify_frozen_production():
    cp = run(
        [
            "git",
            "diff",
            "--quiet",
            FREEZE,
            "--",
            "headroom",
        ],
        ROOT,
    )

    if cp.returncode != 0:
        raise RuntimeError(
            "Current headroom/ production tree differs "
            "from the frozen V3 algorithm commit."
        )


def record_decisions(
    critical,
    off_runs,
    on_runs,
):
    out = []

    for r in critical:
        off = rc.record_keep(
            off_runs,
            r,
        )

        on = rc.record_keep(
            on_runs,
            r,
        )

        out.append(
            {
                "path": r["path"],
                "line": r["line"],
                "text": r["text"],
                "baseline_keep": off,
                "v3_keep": on,
            }
        )

    return out


def summarize_decisions(decisions):
    summary = Counter()

    for r in decisions:
        off = r["baseline_keep"]
        on = r["v3_keep"]

        if off is False:
            summary["baseline_drop"] += 1

        if off is True:
            summary["baseline_keep"] += 1

        if off is False and on is True:
            summary["drop_to_keep"] += 1

        elif off is True and on is False:
            summary["keep_to_drop"] += 1

        elif off is True and on is True:
            summary["keep_to_keep"] += 1

        elif off is False and on is False:
            summary["drop_to_drop"] += 1

        else:
            summary["unmapped"] += 1

    return dict(summary)


def evaluate_candidate(
    commit,
    subj,
    prod,
    tests,
    base,
    scorer,
    router,
    pristine_cache,
    prior_recorded,
):
    # ---------------------------------------------------------------
    # All eligibility criteria up to here are V3-outcome blind.
    # ---------------------------------------------------------------

    key = tuple(tests)

    if key not in pristine_cache:
        pristine_rc, pristine_out = (
            rc.pytest_output(
                base,
                tests,
            )
        )

        pristine_cache[key] = (
            pristine_rc,
            pristine_out,
        )

    pristine_rc, _ = pristine_cache[key]

    if pristine_rc != 0:
        return {
            "kind": "candidate",
            "source_commit": commit,
            "subject": subj,
            "status": "freeze_baseline_tests_not_pass",
            "production_files": prod,
            "test_files": tests,
            "previously_recorded": (
                commit in prior_recorded
            ),
        }

    with tempfile.TemporaryDirectory(
        prefix="v3-exhaustive-"
    ) as td:
        work = Path(td) / "repo"
        shutil.copytree(base, work)

        try:
            patch = rc.production_patch(
                commit,
                prod,
            )

            rc.reverse_patch(
                work,
                patch,
            )

        except Exception as exc:
            return {
                "kind": "candidate",
                "source_commit": commit,
                "subject": subj,
                "status": "reverse_patch_not_clean",
                "production_files": prod,
                "test_files": tests,
                "previously_recorded": (
                    commit in prior_recorded
                ),
                "error": repr(exc),
            }

        buggy_rc, failure = rc.pytest_output(
            work,
            tests,
        )

        if buggy_rc != 1:
            return {
                "kind": "candidate",
                "source_commit": commit,
                "subject": subj,
                "status": (
                    f"buggy_oracle_rc_{buggy_rc}"
                ),
                "production_files": prod,
                "test_files": tests,
                "previously_recorded": (
                    commit in prior_recorded
                ),
            }

        nodeids = failing_nodeids(
            failure
        )

        if not (1 <= len(nodeids) <= 2):
            return {
                "kind": "candidate",
                "source_commit": commit,
                "subject": subj,
                "status": "buggy_failure_count_not_1_or_2",
                "production_files": prod,
                "test_files": tests,
                "failing_nodeids": nodeids,
                "previously_recorded": (
                    commit in prior_recorded
                ),
            }

        names = failing_test_names(
            nodeids
        )

        prompt = challenge_prompt(
            tests
        )

        names = [
            name
            for name in names
            if name not in prompt
        ]

        if not names:
            return {
                "kind": "candidate",
                "source_commit": commit,
                "subject": subj,
                "status": "no_safe_test_name_bridge",
                "production_files": prod,
                "test_files": tests,
                "failing_nodeids": nodeids,
                "previously_recorded": (
                    commit in prior_recorded
                ),
            }

        bridges = bridge_dicts(
            names
        )

        critical = rc.critical_records(
            base,
            work,
            prod,
        )

        if not critical:
            return {
                "kind": "candidate",
                "source_commit": commit,
                "subject": subj,
                "status": "no_critical_records",
                "production_files": prod,
                "test_files": tests,
                "failing_nodeids": nodeids,
                "safe_bridges": bridges,
                "previously_recorded": (
                    commit in prior_recorded
                ),
            }

        anchors = rc.bridge_anchor_records(
            work,
            tests,
            bridges,
        )

        if not anchors:
            return {
                "kind": "candidate",
                "source_commit": commit,
                "subject": subj,
                "status": "no_bridge_anchor",
                "production_files": prod,
                "test_files": tests,
                "failing_nodeids": nodeids,
                "safe_bridges": bridges,
                "critical_record_count": (
                    len(critical)
                ),
                "previously_recorded": (
                    commit in prior_recorded
                ),
            }

        excluded = {
            (r["path"], r["line"])
            for r in critical + anchors
        }

        # IMPORTANT:
        # helper's second parameter is simply the deterministic
        # seed key. For the exhaustive protocol we key it by the
        # immutable source commit, exactly as predeclared.
        distractors = (
            rc.distractor_records(
                work,
                commit,
                excluded,
            )
        )

        target, records = rc.render(
            critical
            + anchors
            + distractors
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "tool",
                "content": failure,
            },
        ]

        v3_context = (
            rc.tr.build_search_relevance_context(
                messages,
                before_index=len(messages),
                user_context=prompt,
                target_content=target,
            )
        )

        enriched = (
            v3_context != prompt
        )

        try:
            off_runs = rc.plan(
                router,
                scorer,
                target,
                prompt,
            )

            on_runs = rc.plan(
                router,
                scorer,
                target,
                v3_context,
            )

        except Exception as exc:
            return {
                "kind": "candidate",
                "source_commit": commit,
                "subject": subj,
                "status": "plan_error",
                "production_files": prod,
                "test_files": tests,
                "failing_nodeids": nodeids,
                "safe_bridges": bridges,
                "critical_record_count": (
                    len(critical)
                ),
                "previously_recorded": (
                    commit in prior_recorded
                ),
                "error": repr(exc),
            }

        decisions = record_decisions(
            critical,
            off_runs,
            on_runs,
        )

        ds = summarize_decisions(
            decisions
        )

        opportunity = (
            ds.get("baseline_drop", 0) > 0
        )

        rescue = (
            ds.get("drop_to_keep", 0) > 0
        )

        regression = (
            ds.get("keep_to_drop", 0) > 0
        )

        return {
            "kind": "candidate",
            "source_commit": commit,
            "subject": subj,
            "status": "evaluated",
            "production_files": prod,
            "test_files": tests,
            "failing_nodeids": nodeids,
            "safe_bridges": bridges,
            "previously_recorded": (
                commit in prior_recorded
            ),
            "fresh": (
                commit not in prior_recorded
            ),
            "prompt": prompt,
            "critical_record_count": (
                len(critical)
            ),
            "bridge_anchor_count": (
                len(anchors)
            ),
            "distractor_count": (
                len(distractors)
            ),
            "target_record_count": (
                len(records)
            ),
            "target_tokens_estimate": (
                rc.cr._estimate_tokens(
                    target
                )
            ),
            "v3_context_enriched": (
                enriched
            ),
            "baseline_opportunity": (
                opportunity
            ),
            "rescue_task": rescue,
            "regression_task": (
                regression
            ),
            "critical_summary": ds,
            "critical_decisions": (
                decisions
            ),
            "off_plan": {
                "runs": len(off_runs),
                "keep": sum(
                    1
                    for k, _ in off_runs
                    if k
                ),
                "drop": sum(
                    1
                    for k, _ in off_runs
                    if not k
                ),
            },
            "v3_plan": {
                "runs": len(on_runs),
                "keep": sum(
                    1
                    for k, _ in on_runs
                    if k
                ),
                "drop": sum(
                    1
                    for k, _ in on_runs
                    if not k
                ),
            },
        }


def main():
    verify_frozen_production()

    print("=" * 112)
    print(
        "FROZEN-V3 EXHAUSTIVE HISTORICAL "
        "CRITICAL-RESCUE SCAN"
    )
    print("=" * 112)
    print(
        "Exploratory exhaustive scan; "
        "separate from N5-N8 and H1-H4."
    )
    print(
        "No coding agent. No external LLM. "
        "No task replacement."
    )
    print(
        "Stopping rule: historical candidate "
        "universe exhaustion."
    )
    print()

    prior_recorded = (
        collect_prior_recorded_shas()
    )

    commits = [
        x.strip()
        for x in git(
            "rev-list",
            "--no-merges",
            "--date-order",
            f"{FREEZE}^",
        ).splitlines()
        if x.strip()
    ]

    print(
        "Historical non-merge commits:",
        len(commits),
    )
    print(
        "Previously recorded SHA flags:",
        len(prior_recorded),
    )

    checkpoint = read_checkpoint()

    if checkpoint:
        print(
            "Resume checkpoint candidates:",
            len(checkpoint),
        )

    base = Path(
        "/tmp/headroom-v3-exhaustive-base"
    )

    rc.make_base(
        base
    )

    router = rc.make_router()
    scorer = rc.get_scorer(
        router
    )

    pristine_cache = {}

    # Cheap metadata outcomes are regenerated every run.
    cheap_status = {}
    expensive_commits = []

    for commit in commits:
        subj = subject(commit)

        if not FIX_RE.search(subj):
            cheap_status[commit] = (
                "not_fix_subject"
            )
            continue

        files = changed_files(
            commit
        )

        prod, tests = (
            candidate_file_sets(
                files
            )
        )

        if not prod or not tests:
            cheap_status[commit] = (
                "missing_prod_or_test"
            )
            continue

        if (
            len(prod) > MAX_PROD_FILES
            or len(tests) > MAX_TEST_FILES
        ):
            cheap_status[commit] = (
                "too_many_prod_or_test_files"
            )
            continue

        if set(files) & TREATMENT_FILES:
            cheap_status[commit] = (
                "touches_v3_treatment"
            )
            continue

        if parent_of(commit) is None:
            cheap_status[commit] = (
                "no_parent"
            )
            continue

        expensive_commits.append(
            (
                commit,
                subj,
                prod,
                tests,
            )
        )

    print(
        "Fix-like candidates requiring "
        "historical validation:",
        len(expensive_commits),
    )
    print()

    start = time.time()

    for idx, (
        commit,
        subj,
        prod,
        tests,
    ) in enumerate(
        expensive_commits,
        1,
    ):
        if commit in checkpoint:
            continue

        rec = evaluate_candidate(
            commit,
            subj,
            prod,
            tests,
            base,
            scorer,
            router,
            pristine_cache,
            prior_recorded,
        )

        append_checkpoint(
            rec
        )

        checkpoint[commit] = rec

        if (
            idx % 10 == 0
            or rec["status"] == "evaluated"
        ):
            elapsed = (
                time.time() - start
            )

            n_eval = sum(
                1
                for x in checkpoint.values()
                if x.get("status")
                == "evaluated"
            )

            print(
                f"[{idx:4}/{len(expensive_commits)}] "
                f"validated={len(checkpoint):4} "
                f"evaluable={n_eval:3} "
                f"elapsed={elapsed/60:.1f}m"
            )

    # ---------------------------------------------------------------
    # Entire candidate universe is now exhausted.
    # Only NOW reveal rescue outcomes.
    # ---------------------------------------------------------------

    final_records = []

    for commit in commits:
        if commit in cheap_status:
            final_records.append(
                {
                    "source_commit": commit,
                    "status": (
                        cheap_status[commit]
                    ),
                }
            )

        elif commit in checkpoint:
            final_records.append(
                checkpoint[commit]
            )

        else:
            raise RuntimeError(
                "Historical universe not exhausted: "
                f"{commit}"
            )

    status_counts = Counter(
        x["status"]
        for x in final_records
    )

    evaluated = [
        x
        for x in final_records
        if x["status"] == "evaluated"
    ]

    opportunities = [
        x
        for x in evaluated
        if x["baseline_opportunity"]
    ]

    rescue_tasks = [
        x
        for x in evaluated
        if x["rescue_task"]
    ]

    regression_tasks = [
        x
        for x in evaluated
        if x["regression_task"]
    ]

    fresh_rescues = [
        x
        for x in rescue_tasks
        if x["fresh"]
    ]

    total_critical = sum(
        x["critical_record_count"]
        for x in evaluated
    )

    baseline_drops = sum(
        x["critical_summary"].get(
            "baseline_drop",
            0,
        )
        for x in evaluated
    )

    drop_to_keep = sum(
        x["critical_summary"].get(
            "drop_to_keep",
            0,
        )
        for x in evaluated
    )

    keep_to_drop = sum(
        x["critical_summary"].get(
            "keep_to_drop",
            0,
        )
        for x in evaluated
    )

    output = {
        "evaluation_class": (
            "post_hoc_exploratory_exhaustive"
        ),
        "algorithm_freeze": FREEZE,
        "candidate_universe": {
            "non_merge_commits": (
                len(commits)
            ),
            "fix_like_validated_candidates": (
                len(expensive_commits)
            ),
            "status_counts": dict(
                sorted(
                    status_counts.items()
                )
            ),
        },
        "summary": {
            "evaluable_tasks": (
                len(evaluated)
            ),
            "baseline_opportunity_tasks": (
                len(opportunities)
            ),
            "rescue_tasks": (
                len(rescue_tasks)
            ),
            "regression_tasks": (
                len(regression_tasks)
            ),
            "fresh_rescue_tasks": (
                len(fresh_rescues)
            ),
            "critical_records": (
                total_critical
            ),
            "baseline_critical_drops": (
                baseline_drops
            ),
            "critical_drop_to_keep": (
                drop_to_keep
            ),
            "critical_keep_to_drop": (
                keep_to_drop
            ),
        },
        "evaluable_tasks": evaluated,
        "rescue_tasks": [
            x["source_commit"]
            for x in rescue_tasks
        ],
        "fresh_rescue_tasks": [
            x["source_commit"]
            for x in fresh_rescues
        ],
        "regression_tasks": [
            x["source_commit"]
            for x in regression_tasks
        ],
    }

    OUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 112)
    print(
        "FINAL EXHAUSTIVE HISTORICAL SUMMARY"
    )
    print("=" * 112)

    print(
        "Historical non-merge commits:",
        len(commits),
    )

    print(
        "Fix-like validated candidates:",
        len(expensive_commits),
    )

    print(
        "Evaluable tasks:",
        len(evaluated),
    )

    print(
        "Baseline opportunity tasks:",
        len(opportunities),
    )

    print(
        "Tasks with CRITICAL DROP->KEEP:",
        len(rescue_tasks),
    )

    print(
        "Fresh tasks with CRITICAL DROP->KEEP:",
        len(fresh_rescues),
    )

    print(
        "Tasks with CRITICAL KEEP->DROP:",
        len(regression_tasks),
    )

    print()
    print(
        "Critical records total:",
        total_critical,
    )

    print(
        "Baseline critical DROP:",
        baseline_drops,
    )

    print(
        "V3 critical DROP->KEEP:",
        drop_to_keep,
    )

    print(
        "V3 critical KEEP->DROP:",
        keep_to_drop,
    )

    if rescue_tasks:
        print()
        print(
            "===== ALL RESCUE TASKS ====="
        )

        for x in rescue_tasks:
            label = (
                "FRESH"
                if x["fresh"]
                else "PREVIOUSLY_RECORDED"
            )

            print()
            print(
                x["source_commit"],
                label,
            )
            print(
                x["subject"]
            )
            print(
                "  production:",
                ", ".join(
                    x["production_files"]
                ),
            )
            print(
                "  tests:",
                ", ".join(
                    x["test_files"]
                ),
            )
            print(
                "  bridges:",
                ", ".join(
                    b["token"]
                    for b in x[
                        "safe_bridges"
                    ]
                ),
            )

            for r in x[
                "critical_decisions"
            ]:
                if (
                    r["baseline_keep"]
                    is False
                    and r["v3_keep"]
                    is True
                ):
                    print(
                        "  RESCUE "
                        f"{r['path']}:"
                        f"{r['line']}:"
                        f"{r['text']}"
                    )

    if regression_tasks:
        print()
        print(
            "===== ALL REGRESSION TASKS ====="
        )

        for x in regression_tasks:
            print()
            print(
                x["source_commit"],
                x["subject"],
            )

            for r in x[
                "critical_decisions"
            ]:
                if (
                    r["baseline_keep"]
                    is True
                    and r["v3_keep"]
                    is False
                ):
                    print(
                        "  REGRESSION "
                        f"{r['path']}:"
                        f"{r['line']}:"
                        f"{r['text']}"
                    )

    print()
    print(
        "Full results:",
        OUT,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )


if __name__ == "__main__":
    main()
