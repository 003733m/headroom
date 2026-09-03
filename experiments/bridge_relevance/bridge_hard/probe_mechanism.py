#!/usr/bin/env python3

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
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

MANIFEST = (
    ROOT
    / "experiments/bridge_relevance/bridge_hard/screening.json"
)

OUT = (
    ROOT
    / "experiments/bridge_relevance/bridge_hard/mechanism_results.json"
)

RUNTIME_PY = Path(
    "/tmp/headroom-v3-runtime/.venv/bin/python"
)

STOPWORDS = {
    "test",
    "tests",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "from",
    "for",
    "in",
    "of",
    "with",
    "without",
    "not",
    "uses",
    "use",
    "using",
    "build",
    "sets",
    "set",
    "creates",
    "create",
    "expected",
    "defaults",
    "default",
    "content",
    "file",
    "command",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


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
    rc1 = p1.wait()

    if rc1 or p2.returncode:
        raise RuntimeError(
            f"archive failure {rc1}/{p2.returncode}: {p2.stdout}"
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
    p = work.parent / "production.patch"
    p.write_text(patch)

    cp = run(
        ["git", "apply", "--reverse", str(p)],
        work,
    )

    if cp.returncode:
        raise RuntimeError(cp.stdout)


def pytest_output(work: Path, test_files):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work)

    cp = run(
        [
            str(RUNTIME_PY),
            "-m",
            "pytest",
            *test_files,
            "-q",
            "--tb=short",
        ],
        work,
        env,
    )

    return cp.returncode, cp.stdout


def normalized_words(text: str):
    return [
        x.lower()
        for x in TOKEN_RE.findall(text)
        if len(x) >= 3
    ]


def candidate_terms(bridges):
    """
    Construct terms only from the PREDECLARED safe bridge identifiers.

    No production patch or V3 score is consulted.
    """
    raw = []

    for b in bridges:
        token = b["token"]

        if b["kind"] == "exception":
            raw.append(token.lower())
            continue

        words = normalized_words(token)

        if words and words[0] == "test":
            words = words[1:]

        words = [
            w
            for w in words
            if w not in STOPWORDS
        ]

        raw.extend(words)

        # Adjacent bigrams carry much more task-specific information
        # (cwd_shadow, keep_id, persistent_docker, etc.).
        raw.extend(
            f"{a}_{b}"
            for a, b in zip(words, words[1:])
        )

    # deterministic unique
    return sorted(set(raw))


def flexible_pattern(term: str):
    """
    foo_bar also matches foo-bar / foo_bar.
    """
    parts = [
        re.escape(x)
        for x in re.split(r"[_-]+", term)
        if x
    ]

    if len(parts) == 1:
        return re.compile(parts[0], re.I)

    return re.compile(
        r"[-_]".join(parts),
        re.I,
    )


def corpus_python_files(work: Path, tests):
    paths = []

    h = work / "headroom"
    if h.exists():
        paths.extend(sorted(h.rglob("*.py")))

    for t in tests:
        p = work / t
        if p.exists():
            paths.append(p)

    # deterministic de-dup
    out = []
    seen = set()

    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)

    return out


def term_frequency(work, files, term):
    pat = flexible_pattern(term)
    count = 0

    for p in files:
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue

        count += len(pat.findall(text))

    return count


def select_terms(work, files, bridges):
    candidates = candidate_terms(bridges)

    ranked = []

    for term in candidates:
        freq = term_frequency(
            work,
            files,
            term,
        )

        if freq <= 0:
            continue

        ranked.append(
            (
                freq,
                -len(term),
                term,
            )
        )

    ranked.sort()

    # up to three rare task-specific terms
    return [
        x[2]
        for x in ranked[:3]
    ], ranked


def build_search_target(work, files, bridges, chosen_terms):
    patterns = [
        flexible_pattern(t)
        for t in chosen_terms
    ]

    # Always include exact predeclared test-name identifiers so the
    # target-aware selector can validate trajectory candidates.
    exact_test_names = [
        b["token"]
        for b in bridges
        if b["kind"] == "test_name"
    ]

    exact_patterns = [
        re.compile(re.escape(x), re.I)
        for x in exact_test_names
    ]

    records = []

    for p in sorted(
        files,
        key=lambda x: str(x.relative_to(work)),
    ):
        rel = str(p.relative_to(work))

        try:
            lines = p.read_text(
                errors="replace"
            ).splitlines()
        except Exception:
            continue

        for lineno, line in enumerate(lines, 1):
            if (
                any(x.search(line) for x in patterns)
                or any(
                    x.search(line)
                    for x in exact_patterns
                )
            ):
                records.append(
                    {
                        "path": rel,
                        "line": lineno,
                        "text": line,
                        "record": (
                            f"{rel}:{lineno}:{line}\n"
                        ),
                    }
                )

    target = "".join(
        x["record"]
        for x in records
    )

    return target, records


def changed_buggy_line_windows(
    fixed_file: Path,
    buggy_file: Path,
    radius=5,
):
    """
    Map historical-fix differences onto line windows in the BUGGY file.

    Insertions in the fixed version correspond to zero-width positions
    in the buggy version; those positions receive +/- radius context.
    """
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
            lo = max(1, center - radius)
            hi = min(
                len(buggy),
                center + radius,
            )
        else:
            lo = max(
                1,
                i1 + 1 - radius,
            )
            hi = min(
                len(buggy),
                i2 + radius,
            )

        windows.append(
            {
                "tag": tag,
                "buggy_start": i1 + 1,
                "buggy_end": i2,
                "fixed_start": j1 + 1,
                "fixed_end": j2,
                "window_start": lo,
                "window_end": hi,
            }
        )

    return windows


def is_critical_record(
    record,
    prod_windows,
):
    path = record["path"]

    for w in prod_windows.get(path, []):
        if (
            w["window_start"]
            <= record["line"]
            <= w["window_end"]
        ):
            return True

    return False


def make_router():
    cfg = cr.ContentRouterConfig()
    cfg.relevance_split = True
    return cr.ContentRouter(cfg)


def get_scorer(router):
    scorer = router._get_relevance_scorer()

    if scorer is None:
        raise RuntimeError(
            "No relevance scorer available"
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


def keep_for_record(runs, record):
    needle = record["record"].rstrip("\n")

    hits = []

    for idx, (keep, text) in enumerate(runs):
        if needle in text:
            hits.append(
                {
                    "run": idx,
                    "keep": bool(keep),
                }
            )

    if not hits:
        # fallback using path:line prefix
        prefix = (
            f"{record['path']}:"
            f"{record['line']}:"
        )

        for idx, (keep, text) in enumerate(runs):
            if prefix in text:
                hits.append(
                    {
                        "run": idx,
                        "keep": bool(keep),
                    }
                )

    if not hits:
        return None

    return any(x["keep"] for x in hits)


def main():
    manifest = json.loads(
        MANIFEST.read_text()
    )

    base = Path(
        "/tmp/headroom-v3-bridge-hard-mechanism-base"
    )

    make_base(base)

    results = {
        "algorithm_freeze": FREEZE,
        "manifest_commit": "c0df5b70",
        "tasks": [],
    }

    print("=" * 110)
    print("V3 BRIDGE-HARD MECHANISM PROBE")
    print("=" * 110)
    print(
        "No coding agent is run. "
        "No task is replaced."
    )

    for task in manifest["selected"]:
        task_id = task["task_id"]

        print()
        print("#" * 110)
        print(task_id, task["subject"])
        print("#" * 110)

        with tempfile.TemporaryDirectory(
            prefix=f"bridge-hard-{task_id}-"
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
                print(
                    "NOT EVALUABLE: reversed oracle "
                    "did not reproduce rc=1"
                )

                results["tasks"].append(
                    {
                        "task_id": task_id,
                        "status": (
                            "oracle_not_reproduced"
                        ),
                        "pytest_rc": rc,
                    }
                )
                continue

            files = corpus_python_files(
                work,
                task["test_files"],
            )

            chosen, term_ranking = select_terms(
                work,
                files,
                task[
                    "safe_bridge_candidates_predeclared"
                ],
            )

            print(
                "search terms:",
                chosen,
            )

            if not chosen:
                print(
                    "NOT EVALUABLE: no bridge-derived "
                    "term occurs in corpus"
                )

                results["tasks"].append(
                    {
                        "task_id": task_id,
                        "status": (
                            "no_search_terms"
                        ),
                    }
                )
                continue

            target, records = build_search_target(
                work,
                files,
                task[
                    "safe_bridge_candidates_predeclared"
                ],
                chosen,
            )

            print(
                "target records:",
                len(records),
            )
            print(
                "target estimated tokens:",
                cr._estimate_tokens(target),
            )

            if not target.strip():
                results["tasks"].append(
                    {
                        "task_id": task_id,
                        "status": "empty_target",
                    }
                )
                continue

            prod_windows = {}

            for prod in task["production_files"]:
                fixed_file = base / prod
                buggy_file = work / prod

                if (
                    fixed_file.exists()
                    and buggy_file.exists()
                ):
                    prod_windows[prod] = (
                        changed_buggy_line_windows(
                            fixed_file,
                            buggy_file,
                            radius=5,
                        )
                    )

            for r in records:
                r["critical"] = (
                    is_critical_record(
                        r,
                        prod_windows,
                    )
                )

            critical = [
                r
                for r in records
                if r["critical"]
            ]

            production_records = [
                r
                for r in records
                if r["path"]
                in set(task["production_files"])
            ]

            print(
                "production records:",
                len(production_records),
            )
            print(
                "patch-adjacent critical records:",
                len(critical),
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

            added_context = (
                v3_context != task["prompt"]
            )

            print(
                "V3 context enriched:",
                added_context,
            )

            if added_context:
                print(
                    "context tail:",
                    repr(v3_context[-500:]),
                )

            router = make_router()
            scorer = get_scorer(router)

            try:
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

            except Exception as exc:
                print(
                    "NOT EVALUABLE:",
                    repr(exc),
                )

                results["tasks"].append(
                    {
                        "task_id": task_id,
                        "status": "plan_error",
                        "error": repr(exc),
                    }
                )
                continue

            changed = []
            critical_rescues = []
            production_rescues = []

            for r in records:
                off = keep_for_record(
                    off_runs,
                    r,
                )
                on = keep_for_record(
                    on_runs,
                    r,
                )

                r["baseline_keep"] = off
                r["v3_keep"] = on

                if (
                    off is not None
                    and on is not None
                    and off != on
                ):
                    changed.append(r)

                if (
                    off is False
                    and on is True
                    and r["path"]
                    in set(
                        task["production_files"]
                    )
                ):
                    production_rescues.append(r)

                    if r["critical"]:
                        critical_rescues.append(r)

            print(
                "OFF runs:",
                len(off_runs),
                "keep=",
                sum(
                    1 for k, _ in off_runs
                    if k
                ),
                "drop=",
                sum(
                    1 for k, _ in off_runs
                    if not k
                ),
            )

            print(
                "V3 runs :",
                len(on_runs),
                "keep=",
                sum(
                    1 for k, _ in on_runs
                    if k
                ),
                "drop=",
                sum(
                    1 for k, _ in on_runs
                    if not k
                ),
            )

            print(
                "records changed:",
                len(changed),
            )
            print(
                "production DROP->KEEP:",
                len(production_rescues),
            )
            print(
                "CRITICAL DROP->KEEP:",
                len(critical_rescues),
            )

            if critical_rescues:
                print()
                print(
                    ">>> CRITICAL TRAJECTORY RESCUES"
                )

                for r in critical_rescues[:20]:
                    print(
                        f"  {r['path']}:{r['line']}:"
                        f"{r['text']}"
                    )

            elif production_rescues:
                print()
                print(
                    ">>> NON-PATCH-ADJACENT "
                    "PRODUCTION RESCUES"
                )

                for r in production_rescues[:20]:
                    print(
                        f"  {r['path']}:{r['line']}:"
                        f"{r['text']}"
                    )

            print()
            print(
                "Critical records and decisions:"
            )

            for r in critical[:30]:
                print(
                    f"  OFF={str(r['baseline_keep']):5} "
                    f"V3={str(r['v3_keep']):5} "
                    f"{r['path']}:{r['line']}:"
                    f"{r['text']}"
                )

            results["tasks"].append(
                {
                    "task_id": task_id,
                    "status": "evaluated",
                    "subject": task["subject"],
                    "search_terms": chosen,
                    "term_ranking": [
                        {
                            "frequency": x[0],
                            "term": x[2],
                        }
                        for x in term_ranking
                    ],
                    "target_record_count": len(
                        records
                    ),
                    "target_tokens_estimate": (
                        cr._estimate_tokens(
                            target
                        )
                    ),
                    "v3_context_enriched": (
                        added_context
                    ),
                    "v3_context": v3_context,
                    "off_plan": {
                        "runs": len(off_runs),
                        "keep": sum(
                            1 for k, _ in off_runs
                            if k
                        ),
                        "drop": sum(
                            1 for k, _ in off_runs
                            if not k
                        ),
                    },
                    "v3_plan": {
                        "runs": len(on_runs),
                        "keep": sum(
                            1 for k, _ in on_runs
                            if k
                        ),
                        "drop": sum(
                            1 for k, _ in on_runs
                            if not k
                        ),
                    },
                    "changed_record_count": len(
                        changed
                    ),
                    "production_rescue_count": len(
                        production_rescues
                    ),
                    "critical_rescue_count": len(
                        critical_rescues
                    ),
                    "production_rescues": (
                        production_rescues
                    ),
                    "critical_rescues": (
                        critical_rescues
                    ),
                    "critical_records": critical,
                    "diff_windows": prod_windows,
                }
            )

    OUT.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 110)
    print("FINAL MECHANISM SUMMARY")
    print("=" * 110)

    total_critical = 0
    total_prod = 0

    for x in results["tasks"]:
        if x["status"] != "evaluated":
            print(
                x["task_id"],
                x["status"],
            )
            continue

        total_critical += (
            x["critical_rescue_count"]
        )
        total_prod += (
            x["production_rescue_count"]
        )

        print(
            f"{x['task_id']}: "
            f"context_enriched="
            f"{x['v3_context_enriched']}  "
            f"changed="
            f"{x['changed_record_count']}  "
            f"prod_DROP->KEEP="
            f"{x['production_rescue_count']}  "
            f"CRITICAL_DROP->KEEP="
            f"{x['critical_rescue_count']}"
        )

    print()
    print(
        "TOTAL production DROP->KEEP:",
        total_prod,
    )
    print(
        "TOTAL critical DROP->KEEP:",
        total_critical,
    )
    print(
        "Results:",
        OUT,
    )


if __name__ == "__main__":
    main()
