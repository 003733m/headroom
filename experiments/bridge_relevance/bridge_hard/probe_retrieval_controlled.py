#!/usr/bin/env python3

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import headroom.trajectory_relevance as tr
import headroom.transforms.content_router as cr


ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

FREEZE = "7c4fe80c19b3a9537f395f515a64432e41ee1d43"
MANIFEST = ROOT / "experiments/bridge_relevance/bridge_hard/screening.json"
OUT = ROOT / "experiments/bridge_relevance/bridge_hard/retrieval_controlled_results.json"

RUNTIME_PY = Path("/tmp/headroom-v3-runtime/.venv/bin/python")

SEED = "headroom-v3-retrieval-controlled-v1"
N_DISTRACTORS = 600
RADIUS = 5


def run(args, cwd=None, env=None):
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def git(*args):
    cp = run(["git", *args], ROOT)
    if cp.returncode:
        raise RuntimeError(cp.stdout)
    return cp.stdout


def make_base(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True)

    p1 = subprocess.Popen(
        ["git", "archive", FREEZE],
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )

    assert p1.stdout is not None

    p2 = subprocess.run(
        ["tar", "-x", "-C", str(dst)],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    p1.stdout.close()
    rc = p1.wait()

    if rc or p2.returncode:
        raise RuntimeError(
            f"archive failed: {rc}/{p2.returncode}: {p2.stdout}"
        )


def production_patch(commit, prod):
    parent = git("rev-parse", f"{commit}^").strip()

    return git(
        "diff",
        "--binary",
        parent,
        commit,
        "--",
        *prod,
    )


def reverse_patch(work: Path, patch: str):
    patch_file = work.parent / "production.patch"
    patch_file.write_text(patch)

    cp = run(
        ["git", "apply", "--reverse", str(patch_file)],
        work,
    )

    if cp.returncode:
        raise RuntimeError(cp.stdout)


def pytest_output(work: Path, tests):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work)

    cp = run(
        [
            str(RUNTIME_PY),
            "-m",
            "pytest",
            *tests,
            "-q",
            "--tb=short",
        ],
        work,
        env,
    )

    return cp.returncode, cp.stdout


def diff_windows(fixed_file: Path, buggy_file: Path):
    fixed = fixed_file.read_text(
        errors="replace"
    ).splitlines()

    buggy = buggy_file.read_text(
        errors="replace"
    ).splitlines()

    sm = difflib.SequenceMatcher(
        a=buggy,
        b=fixed,
        autojunk=False,
    )

    windows = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if i1 == i2:
            center = i1 + 1
            lo = max(1, center - RADIUS)
            hi = min(len(buggy), center + RADIUS)
        else:
            lo = max(1, i1 + 1 - RADIUS)
            hi = min(len(buggy), i2 + RADIUS)

        windows.append((lo, hi))

    return windows


def critical_records(base: Path, work: Path, prod_files):
    records = []
    seen = set()

    for rel in prod_files:
        fixed = base / rel
        buggy = work / rel

        if not fixed.exists() or not buggy.exists():
            continue

        lines = buggy.read_text(
            errors="replace"
        ).splitlines()

        for lo, hi in diff_windows(fixed, buggy):
            for lineno in range(lo, hi + 1):
                text = lines[lineno - 1]

                if not text.strip():
                    continue

                key = (rel, lineno)

                if key in seen:
                    continue

                seen.add(key)

                records.append(
                    {
                        "path": rel,
                        "line": lineno,
                        "text": text,
                        "kind": "critical",
                    }
                )

    return records


def bridge_anchor_records(work: Path, test_files, bridges):
    test_names = [
        b["token"]
        for b in bridges
        if b["kind"] == "test_name"
    ]

    records = []
    seen = set()

    for rel in test_files:
        p = work / rel

        if not p.exists():
            continue

        lines = p.read_text(
            errors="replace"
        ).splitlines()

        for lineno, text in enumerate(lines, 1):
            if not any(
                name in text
                for name in test_names
            ):
                continue

            key = (rel, lineno)

            if key in seen:
                continue

            seen.add(key)

            records.append(
                {
                    "path": rel,
                    "line": lineno,
                    "text": text,
                    "kind": "bridge_anchor",
                }
            )

    return records


def distractor_records(
    work: Path,
    task_id: str,
    excluded: set[tuple[str, int]],
):
    ranked = []

    for p in sorted(
        (work / "headroom").rglob("*.py")
    ):
        rel = str(p.relative_to(work))

        try:
            lines = p.read_text(
                errors="replace"
            ).splitlines()
        except Exception:
            continue

        for lineno, text in enumerate(lines, 1):
            if not text.strip():
                continue

            if (rel, lineno) in excluded:
                continue

            payload = (
                f"{SEED}|{task_id}|"
                f"{rel}|{lineno}|{text}"
            )

            h = hashlib.sha256(
                payload.encode()
            ).hexdigest()

            ranked.append(
                (
                    h,
                    {
                        "path": rel,
                        "line": lineno,
                        "text": text,
                        "kind": "distractor",
                    },
                )
            )

    ranked.sort(key=lambda x: x[0])

    return [
        r
        for _, r in ranked[:N_DISTRACTORS]
    ]


def render(records):
    ordered = sorted(
        records,
        key=lambda r: (
            r["path"],
            r["line"],
            r["kind"],
        ),
    )

    for r in ordered:
        r["record"] = (
            f"{r['path']}:{r['line']}:"
            f"{r['text']}\n"
        )

    return "".join(
        r["record"]
        for r in ordered
    ), ordered


def make_router():
    cfg = cr.ContentRouterConfig()
    cfg.relevance_split = True
    return cr.ContentRouter(cfg)


def get_scorer(router):
    scorer = router._get_relevance_scorer()

    if scorer is None:
        raise RuntimeError(
            "relevance scorer unavailable"
        )

    ensure = getattr(
        scorer,
        "ensure_background_load",
        None,
    )

    ready = getattr(
        scorer,
        "is_ready",
        None,
    )

    if callable(ensure):
        try:
            ensure()
        except Exception:
            pass

    if callable(ready):
        deadline = time.time() + 90

        while time.time() < deadline:
            try:
                if ready():
                    break
            except Exception:
                break

            time.sleep(0.5)

    return scorer


def plan(router, scorer, target, query):
    return cr.plan_relevance_split(
        target,
        query,
        scorer,
        threshold=(
            router.config.relevance
            .relevance_threshold
        ),
        adaptive=(
            router.config
            .relevance_adaptive_threshold
        ),
        max_records=(
            router.config
            .relevance_max_records
        ),
    )


def record_keep(runs, record):
    needle = record["record"].rstrip("\n")

    for keep, text in runs:
        if needle in text:
            return bool(keep)

    prefix = (
        f"{record['path']}:"
        f"{record['line']}:"
    )

    for keep, text in runs:
        if prefix in text:
            return bool(keep)

    return None


def main():
    manifest = json.loads(
        MANIFEST.read_text()
    )

    base = Path(
        "/tmp/headroom-v3-retrieval-controlled-base"
    )

    make_base(base)

    result = {
        "algorithm_freeze": FREEZE,
        "task_manifest_commit": "c0df5b70",
        "seed": SEED,
        "distractor_count": N_DISTRACTORS,
        "interpretation": "conditional_on_retrieval",
        "tasks": [],
    }

    print("=" * 112)
    print("V3 RETRIEVAL-CONTROLLED BRIDGE-HARD CHALLENGE")
    print("=" * 112)
    print("No coding agent. No external LLM. No task replacement.")
    print()

    for task in manifest["selected"]:
        tid = task["task_id"]

        print("#" * 112)
        print(tid, task["subject"])
        print("#" * 112)

        with tempfile.TemporaryDirectory(
            prefix=f"rc-{tid}-"
        ) as td:
            work = Path(td) / "repo"
            shutil.copytree(base, work)

            patch = production_patch(
                task["source_commit"],
                task["production_files"],
            )

            reverse_patch(work, patch)

            rc, failure = pytest_output(
                work,
                task["test_files"],
            )

            print("pytest rc:", rc)

            if rc != 1:
                print("NOT EVALUABLE: oracle rc != 1")
                result["tasks"].append(
                    {
                        "task_id": tid,
                        "status": "oracle_not_reproduced",
                        "pytest_rc": rc,
                    }
                )
                continue

            critical = critical_records(
                base,
                work,
                task["production_files"],
            )

            anchors = bridge_anchor_records(
                work,
                task["test_files"],
                task[
                    "safe_bridge_candidates_predeclared"
                ],
            )

            excluded = {
                (r["path"], r["line"])
                for r in critical + anchors
            }

            distractors = distractor_records(
                work,
                tid,
                excluded,
            )

            target, records = render(
                critical + anchors + distractors
            )

            print(
                "critical records:",
                len(critical),
            )
            print(
                "bridge anchors:",
                len(anchors),
            )
            print(
                "distractors:",
                len(distractors),
            )
            print(
                "target records:",
                len(records),
            )
            print(
                "target estimated tokens:",
                cr._estimate_tokens(target),
            )

            messages = [
                {
                    "role": "user",
                    "content": task["prompt"],
                },
                {
                    "role": "tool",
                    "content": failure,
                },
            ]

            v3_context = (
                tr.build_search_relevance_context(
                    messages,
                    before_index=len(messages),
                    user_context=task["prompt"],
                    target_content=target,
                )
            )

            enriched = (
                v3_context != task["prompt"]
            )

            print(
                "V3 context enriched:",
                enriched,
            )

            print(
                "V3 context tail:",
                repr(v3_context[-450:]),
            )

            router = make_router()
            scorer = get_scorer(router)

            off_runs = plan(
                router,
                scorer,
                target,
                task["prompt"],
            )

            on_runs = plan(
                router,
                scorer,
                target,
                v3_context,
            )

            critical_out = []

            for r in critical:
                off = record_keep(
                    off_runs,
                    r,
                )

                on = record_keep(
                    on_runs,
                    r,
                )

                critical_out.append(
                    {
                        **r,
                        "baseline_keep": off,
                        "v3_keep": on,
                    }
                )

            rescues = [
                r
                for r in critical_out
                if (
                    r["baseline_keep"] is False
                    and r["v3_keep"] is True
                )
            ]

            lost = [
                r
                for r in critical_out
                if (
                    r["baseline_keep"] is True
                    and r["v3_keep"] is False
                )
            ]

            already = [
                r
                for r in critical_out
                if (
                    r["baseline_keep"] is True
                    and r["v3_keep"] is True
                )
            ]

            both_drop = [
                r
                for r in critical_out
                if (
                    r["baseline_keep"] is False
                    and r["v3_keep"] is False
                )
            ]

            print(
                "OFF plan:",
                len(off_runs),
                "keep=",
                sum(1 for k, _ in off_runs if k),
                "drop=",
                sum(1 for k, _ in off_runs if not k),
            )

            print(
                "V3 plan :",
                len(on_runs),
                "keep=",
                sum(1 for k, _ in on_runs if k),
                "drop=",
                sum(1 for k, _ in on_runs if not k),
            )

            print()
            print("CRITICAL DECISIONS")
            print(
                "  DROP->KEEP:",
                len(rescues),
            )
            print(
                "  KEEP->KEEP:",
                len(already),
            )
            print(
                "  DROP->DROP:",
                len(both_drop),
            )
            print(
                "  KEEP->DROP:",
                len(lost),
            )

            for r in critical_out:
                label = (
                    "RESCUE"
                    if (
                        r["baseline_keep"] is False
                        and r["v3_keep"] is True
                    )
                    else "SAME"
                    if r["baseline_keep"] == r["v3_keep"]
                    else "REGRESSION"
                )

                print(
                    f"  {label:10} "
                    f"OFF={str(r['baseline_keep']):5} "
                    f"V3={str(r['v3_keep']):5} "
                    f"{r['path']}:{r['line']}:"
                    f"{r['text']}"
                )

            result["tasks"].append(
                {
                    "task_id": tid,
                    "status": "evaluated",
                    "subject": task["subject"],
                    "critical_record_count": len(
                        critical
                    ),
                    "bridge_anchor_count": len(
                        anchors
                    ),
                    "distractor_count": len(
                        distractors
                    ),
                    "target_record_count": len(
                        records
                    ),
                    "target_tokens_estimate": (
                        cr._estimate_tokens(target)
                    ),
                    "v3_context_enriched": enriched,
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
                    "critical_drop_to_keep": len(
                        rescues
                    ),
                    "critical_keep_to_keep": len(
                        already
                    ),
                    "critical_drop_to_drop": len(
                        both_drop
                    ),
                    "critical_keep_to_drop": len(
                        lost
                    ),
                    "critical_records": critical_out,
                }
            )

        print()

    OUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    print("=" * 112)
    print("FINAL RETRIEVAL-CONTROLLED SUMMARY")
    print("=" * 112)

    total_rescues = 0
    total_regressions = 0

    for task in result["tasks"]:
        if task["status"] != "evaluated":
            print(
                task["task_id"],
                task["status"],
            )
            continue

        total_rescues += (
            task["critical_drop_to_keep"]
        )

        total_regressions += (
            task["critical_keep_to_drop"]
        )

        print(
            f"{task['task_id']}: "
            f"critical={task['critical_record_count']} "
            f"DROP->KEEP="
            f"{task['critical_drop_to_keep']} "
            f"KEEP->KEEP="
            f"{task['critical_keep_to_keep']} "
            f"DROP->DROP="
            f"{task['critical_drop_to_drop']} "
            f"KEEP->DROP="
            f"{task['critical_keep_to_drop']}"
        )

    print()
    print(
        "TOTAL CRITICAL DROP->KEEP:",
        total_rescues,
    )
    print(
        "TOTAL CRITICAL KEEP->DROP:",
        total_regressions,
    )
    print("Results:", OUT)


if __name__ == "__main__":
    main()
