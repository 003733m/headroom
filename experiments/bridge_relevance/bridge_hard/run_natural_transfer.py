#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

CONTROL = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

FREEZE = "7c4fe80c19b3a9537f395f515a64432e41ee1d43"
MANIFEST_COMMIT = "1c5500dc"
MANIFEST_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "natural_transfer_manifest.json"
)
SNAP_LOCK_REL = (
    "experiments/bridge_relevance/bridge_hard/"
    "natural_transfer_snapshot_lock.json"
)

MODEL = "openai/gpt-5.6-terra"
OPENCODE_VERSION = "1.18.25"

ROOT = Path("/tmp/headroom-v3-natural-transfer-final")
RUNTIME = Path("/tmp/headroom-v3-runtime")
SNAPSHOTS = Path("/tmp/headroom-v3-natural-transfer-locked")

GLOBAL_PROXY_LOG = (
    Path.home() / ".headroom/logs/proxy.log"
)

FAILED_RE = re.compile(
    r"(?m)^FAILED\s+([^\s]+)"
)

PROXY: subprocess.Popen | None = None


def die(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def run(
    cmd,
    *,
    cwd=None,
    env=None,
    stdout=None,
    stderr=None,
    check=False,
):
    cp = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=stdout,
        stderr=stderr,
    )
    if check and cp.returncode:
        die(
            f"command failed rc={cp.returncode}: "
            + " ".join(map(str, cmd))
        )
    return cp


def capture(cmd, cwd=None):
    cp = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if cp.returncode:
        die(
            f"command failed rc={cp.returncode}:\n"
            f"{' '.join(cmd)}\n{cp.stdout}"
        )
    return cp.stdout


def git(*args):
    return capture(
        ["git", *args],
        CONTROL,
    )


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def prepare_dirs():
    for p in (
        ROOT,
        ROOT / "results",
        ROOT / "runs",
        ROOT / "prompts",
        ROOT / "prime",
        SNAPSHOTS,
    ):
        p.mkdir(parents=True, exist_ok=True)


def locked_manifest():
    prepare_dirs()

    text = git(
        "show",
        f"{MANIFEST_COMMIT}:{MANIFEST_REL}",
    )

    dst = ROOT / "locked_manifest.json"

    if dst.exists():
        if dst.read_text() != text:
            die(
                "existing locked_manifest.json "
                "differs from lock commit"
            )
    else:
        dst.write_text(text)

    d = json.loads(text)

    if d["status"] != (
        "locked_before_any_transfer_agent_run"
    ):
        die("manifest status mismatch")

    if d["algorithm_freeze"] != FREEZE:
        die("manifest freeze mismatch")

    if d["task_count"] != 8:
        die("expected exactly 8 tasks")

    expected = [
        "F1-OFF", "F1-ON",
        "F2-ON", "F2-OFF",
        "F3-OFF", "F3-ON",
        "F4-ON", "F4-OFF",
        "F5-OFF", "F5-ON",
        "F6-ON", "F6-OFF",
        "F7-OFF", "F7-ON",
        "F8-ON", "F8-OFF",
    ]

    if d["run_order"] != expected:
        die("locked condition order mismatch")

    return d


def task_by_id(manifest, task_id):
    for t in manifest["tasks"]:
        if t["task_id"] == task_id:
            return t
    die(f"unknown task {task_id}")


def ensure_uv_lock(dst: Path):
    target = dst / "uv.lock"
    if target.exists():
        return

    src = CONTROL / "uv.lock"
    if not src.exists():
        die("uv.lock missing from both snapshot and control repo")

    shutil.copy2(src, target)


def prepare_runtime():
    prepare_dirs()

    try:
        git("cat-file", "-e", f"{FREEZE}^{{commit}}")
    except SystemExit:
        die("V3 freeze commit unavailable")

    if (RUNTIME / ".git").exists():
        got = capture(
            ["git", "rev-parse", "HEAD"],
            RUNTIME,
        ).strip()
        if got != FREEZE:
            die(
                "existing runtime is not exact "
                f"V3 freeze: {got}"
            )
    else:
        if RUNTIME.exists():
            die(
                f"{RUNTIME} exists but is not "
                "the frozen Git worktree"
            )

        run(
            ["git", "worktree", "prune"],
            cwd=CONTROL,
        )

        run(
            [
                "git", "worktree", "add",
                "--detach",
                str(RUNTIME),
                FREEZE,
            ],
            cwd=CONTROL,
            check=True,
        )

    ensure_uv_lock(RUNTIME)

    out = ROOT / "runtime-uv-sync.txt"
    with out.open("w") as f:
        cp = run(
            [
                "uv", "sync",
                "--python", "3.12",
                "--extra", "dev",
                "--extra", "proxy",
                "--frozen",
            ],
            cwd=RUNTIME,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if cp.returncode:
        die(
            "uv sync failed for frozen V3 runtime; "
            f"see {out}"
        )

    py = RUNTIME / ".venv/bin/python"
    if not py.exists():
        die("frozen runtime Python missing after uv sync")

    print(f"Frozen runtime: OK ({FREEZE})")


def archive_freeze(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "freeze.tar"

        run(
            [
                "git", "archive",
                "--format=tar",
                "-o", str(tar_path),
                FREEZE,
            ],
            cwd=CONTROL,
            check=True,
        )

        with tarfile.open(tar_path) as tf:
            tf.extractall(dst)

    ensure_uv_lock(dst)


def historical_prod_patch(task):
    commit = task["source_commit"]

    parent = git(
        "rev-parse",
        f"{commit}^",
    ).strip()

    cmd = [
        "git", "diff",
        parent,
        commit,
        "--binary",
        "--",
        *task["production_files"],
    ]

    cp = subprocess.run(
        cmd,
        cwd=CONTROL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if cp.returncode:
        die(
            f"{task['task_id']} historical "
            "production diff failed"
        )

    return cp.stdout


def init_single_commit_snapshot(
    dst: Path,
    task_id: str,
):
    run(
        ["git", "init", "-q"],
        cwd=dst,
        check=True,
    )

    run(
        ["git", "config", "user.name", "Headroom Benchmark"],
        cwd=dst,
        check=True,
    )
    run(
        [
            "git", "config",
            "user.email",
            "benchmark@invalid.local",
        ],
        cwd=dst,
        check=True,
    )

    run(
        ["git", "add", "-A"],
        cwd=dst,
        check=True,
    )

    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = "2000-01-01T00:00:00+00:00"
    env["GIT_COMMITTER_DATE"] = (
        "2000-01-01T00:00:00+00:00"
    )

    cp = run(
        [
            "git", "commit", "-q",
            "-m", f"Locked buggy snapshot {task_id}",
        ],
        cwd=dst,
        env=env,
    )

    if cp.returncode:
        die(f"{task_id} snapshot commit failed")


def pytest_with_runtime(repo: Path, tests):
    py = RUNTIME / ".venv/bin/python"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)

    cp = subprocess.run(
        [
            str(py),
            "-m", "pytest",
            "-q",
            "--tb=short",
            *tests,
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return cp.returncode, cp.stdout


def observed_failures(text: str):
    out = []
    for x in FAILED_RE.findall(text):
        if x not in out:
            out.append(x)
    return out


def create_snapshot(task):
    tid = task["task_id"]
    dst = SNAPSHOTS / tid

    archive_freeze(dst)

    patch = historical_prod_patch(task)

    with tempfile.NamedTemporaryFile(
        prefix=f"{tid}-",
        suffix=".patch",
        delete=False,
    ) as f:
        patch_path = Path(f.name)
        f.write(patch)

    try:
        check = run(
            [
                "git", "apply",
                "--reverse",
                "--check",
                str(patch_path),
            ],
            cwd=dst,
        )
        if check.returncode:
            die(f"{tid} reverse patch check failed")

        apply = run(
            [
                "git", "apply",
                "--reverse",
                str(patch_path),
            ],
            cwd=dst,
        )
        if apply.returncode:
            die(f"{tid} reverse patch apply failed")
    finally:
        patch_path.unlink(missing_ok=True)

    init_single_commit_snapshot(
        dst,
        tid,
    )

    expected = sorted(
        task["failing_nodeids"]
    )

    rc, text = pytest_with_runtime(
        dst,
        task["test_files"],
    )

    got = sorted(
        observed_failures(text)
    )

    if rc != 1 or got != expected:
        print(text[-8000:])
        die(
            f"{tid} buggy oracle mismatch: "
            f"rc={rc}, expected failures={expected}, "
            f"observed={got}"
        )

    status = capture(
        ["git", "status", "--porcelain"],
        dst,
    ).strip()

    if status:
        die(
            f"{tid} snapshot became dirty "
            f"during validation:\n{status}"
        )

    head = capture(
        ["git", "rev-parse", "HEAD"],
        dst,
    ).strip()

    tree = capture(
        ["git", "rev-parse", "HEAD^{tree}"],
        dst,
    ).strip()

    count = int(
        capture(
            ["git", "rev-list", "--all", "--count"],
            dst,
        ).strip()
    )

    if count != 1:
        die(f"{tid} exposes {count} commits")

    return {
        "task_id": tid,
        "source_commit": task["source_commit"],
        "snapshot_head": head,
        "snapshot_tree": tree,
        "commit_count": count,
        "uv_lock_sha256": sha256_file(
            dst / "uv.lock"
        ),
        "prompt_sha256": task["prompt_sha256"],
        "expected_buggy_failures": expected,
    }


def prepare_snapshots():
    manifest = locked_manifest()
    prepare_runtime()

    rows = []

    for task in manifest["tasks"]:
        tid = task["task_id"]
        print(f"Preparing {tid} ...")
        row = create_snapshot(task)
        rows.append(row)
        print(
            f"{tid}: oracle OK, "
            f"snapshot={row['snapshot_head'][:12]}"
        )

    lock = {
        "status": (
            "prepared_before_any_transfer_agent_run"
        ),
        "algorithm_freeze": FREEZE,
        "manifest_commit": MANIFEST_COMMIT,
        "model": MODEL,
        "opencode_version": OPENCODE_VERSION,
        "tasks": rows,
    }

    out = CONTROL / SNAP_LOCK_REL
    out.write_text(
        json.dumps(
            lock,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print()
    print("==========================================")
    print("NATURAL TRANSFER SNAPSHOTS PREPARED")
    print("No benchmark agent was run.")
    print(f"snapshot lock: {out}")
    print("==========================================")


def snapshot_lock():
    p = CONTROL / SNAP_LOCK_REL

    if not p.exists():
        die(
            "snapshot lock missing; run --prepare first"
        )

    # Before benchmark execution the lock itself
    # must already be committed and unchanged.
    cp = subprocess.run(
        [
            "git", "cat-file", "-e",
            f"HEAD:{SNAP_LOCK_REL}",
        ],
        cwd=CONTROL,
    )
    if cp.returncode:
        die(
            "snapshot lock is not committed in HEAD"
        )

    status = capture(
        [
            "git", "status",
            "--porcelain=v1",
            "--", SNAP_LOCK_REL,
        ],
        CONTROL,
    ).strip()

    if status:
        die(
            "snapshot lock differs from committed HEAD"
        )

    return json.loads(p.read_text())


def verify_snapshots():
    manifest = locked_manifest()
    lock = snapshot_lock()

    by_id = {
        x["task_id"]: x
        for x in lock["tasks"]
    }

    for task in manifest["tasks"]:
        tid = task["task_id"]
        snap = SNAPSHOTS / tid

        if not (snap / ".git").exists():
            die(f"{tid} snapshot missing")

        row = by_id[tid]

        head = capture(
            ["git", "rev-parse", "HEAD"],
            snap,
        ).strip()

        tree = capture(
            ["git", "rev-parse", "HEAD^{tree}"],
            snap,
        ).strip()

        count = int(
            capture(
                ["git", "rev-list", "--all", "--count"],
                snap,
            ).strip()
        )

        status = capture(
            ["git", "status", "--porcelain"],
            snap,
        ).strip()

        if head != row["snapshot_head"]:
            die(f"{tid} snapshot HEAD mismatch")
        if tree != row["snapshot_tree"]:
            die(f"{tid} snapshot tree mismatch")
        if count != 1:
            die(f"{tid} snapshot history exposed")
        if status:
            die(f"{tid} snapshot dirty")
        if sha256_file(snap / "uv.lock") != (
            row["uv_lock_sha256"]
        ):
            die(f"{tid} uv.lock hash mismatch")

        print(
            f"{tid} snapshot: OK "
            f"({head[:12]})"
        )


def find_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_livez(port, proc):
    url = f"http://127.0.0.1:{port}/livez"

    for _ in range(60):
        if proc.poll() is not None:
            return False

        try:
            with urllib.request.urlopen(
                url,
                timeout=1,
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass

        time.sleep(1)

    return False


def start_proxy(port, out, flag):
    env = os.environ.copy()
    env.update({
        "HEADROOM_TRAJECTORY_RELEVANCE": str(flag),
        "HEADROOM_SAVINGS_PROFILE": "coding",
        "HEADROOM_AGENT_TYPE": "opencode",
        "HEADROOM_STACK": "wrap_opencode",
        "HEADROOM_TELEMETRY": "off",
    })

    log = (out / "proxy_process.log").open("w")

    proc = subprocess.Popen(
        [
            "uv", "run", "--frozen",
            "headroom", "proxy",
            "--port", str(port),
            "--log-file", str(
                out / "requests.jsonl"
            ),
        ],
        cwd=RUNTIME,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if not wait_livez(port, proc):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        die("proxy failed to become ready")

    return proc, log


def stop_proxy(proc, log_handle):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    try:
        log_handle.close()
    except Exception:
        pass


def warm_proxy(port, out, flag):
    prime = ROOT / "prime"
    prime.mkdir(parents=True, exist_ok=True)

    script = prime / "prime.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - <<'PY'\n"
        "for i in range(1600):\n"
        "    print("
        "f\"Warmup record {i:04d}: deterministic "
        "diagnostic payload segment {i:04d}; "
        "subsystem state stable.\""
        ")\n"
        "PY\n"
    )
    script.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "HEADROOM_TRAJECTORY_RELEVANCE": str(flag),
        "HEADROOM_SAVINGS_PROFILE": "coding",
    })

    with (out / "prime.txt").open("w") as f:
        cp = run(
            [
                "uv", "run",
                "--project", str(RUNTIME),
                "--frozen",
                "headroom", "wrap", "opencode",
                "--no-proxy",
                "--port", str(port),
                "--no-mcp",
                "--no-serena",
                "--",
                "run",
                "-m", MODEL,
                (
                    "Run "
                    + str(script.resolve())
                    + " exactly once and then reply only "
                      "PRIME_DONE."
                ),
            ],
            cwd=prime,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if cp.returncode:
        return False

    url = f"http://127.0.0.1:{port}/debug/warmup"

    for _ in range(120):
        try:
            with urllib.request.urlopen(
                url,
                timeout=2,
            ) as r:
                d = json.load(r)

            def status(x):
                if isinstance(x, dict):
                    k = x.get("kompress")
                    if isinstance(k, dict):
                        if k.get("status"):
                            return k["status"]
                    for v in x.values():
                        s = status(v)
                        if s:
                            return s
                elif isinstance(x, list):
                    for v in x:
                        s = status(v)
                        if s:
                            return s
                return None

            if status(d) == "loaded":
                (out / "warmup.json").write_text(
                    json.dumps(d, indent=2)
                )
                return True

        except Exception:
            pass

        time.sleep(1)

    return False


def smoke_proxy():
    out = ROOT / "preflight"
    out.mkdir(parents=True, exist_ok=True)

    port = find_port()
    proc, log = start_proxy(
        port,
        out,
        0,
    )

    stop_proxy(proc, log)
    print("Proxy smoke test: OK")


def opencode_version():
    cp = subprocess.run(
        ["opencode", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    version = (
        cp.stdout.strip().splitlines()[-1]
        if cp.stdout.strip()
        else ""
    )

    if version != OPENCODE_VERSION:
        die(
            "OpenCode version mismatch: "
            f"expected {OPENCODE_VERSION}, "
            f"got {version!r}"
        )

    print(f"OpenCode: OK ({version})")


def preflight():
    locked_manifest()
    prepare_runtime()
    snapshot_lock()
    verify_snapshots()
    opencode_version()
    smoke_proxy()

    print()
    print("==========================================")
    print("NATURAL TRANSFER PREFLIGHT PASSED")
    print("No benchmark agent was run.")
    print(f"model : {MODEL}")
    print(f"freeze: {FREEZE}")
    print("next  : F1 OFF")
    print("==========================================")


def create_rg_wrapper(out: Path):
    real_rg = shutil.which("rg")

    if not real_rg:
        die("rg executable not found")

    capbin = out / "capture-bin"
    capdir = out / "rg-captures"
    capbin.mkdir(parents=True, exist_ok=True)
    capdir.mkdir(parents=True, exist_ok=True)

    wrapper = capbin / "rg"

    wrapper.write_text(
        """#!/usr/bin/env bash
dir="${RG_CAPTURE_DIR:?}"
real="${REAL_RG:?}"
mkdir -p "$dir"

id="$(date +%s%N)-$$-${RANDOM:-0}"
meta="$dir/$id.meta"
stdout_file="$dir/$id.stdout"
stderr_file="$dir/$id.stderr"

{
    printf 'cwd=%q\\n' "$PWD"
    printf 'argv='
    printf '%q ' "$@"
    printf '\\n'
    printf 'started_ns=%s\\n' "$(date +%s%N)"
} > "$meta"

"$real" "$@" > "$stdout_file" 2> "$stderr_file"
rc=$?

cat "$stdout_file"
cat "$stderr_file" >&2

{
    printf 'rc=%s\\n' "$rc"
    printf 'finished_ns=%s\\n' "$(date +%s%N)"
} >> "$meta"

exit "$rc"
"""
    )

    wrapper.chmod(0o755)

    return capbin, capdir, real_rg


def tail_jsonl(path: Path, first_line: int):
    if not path.exists():
        return ""

    lines = path.read_text(
        errors="replace"
    ).splitlines()

    return "\n".join(
        lines[first_line:]
    ) + ("\n" if len(lines) > first_line else "")


def bytes_since(path: Path, offset: int):
    if not path.exists():
        return b""

    with path.open("rb") as f:
        f.seek(offset)
        return f.read()


def verify_precondition(
    run_dir,
    task,
    out,
):
    with (out / "precondition.txt").open("w") as f:
        cp = run(
            [
                "uv", "run", "--frozen",
                "pytest", "-q",
                "--tb=short",
                *task["test_files"],
            ],
            cwd=run_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    text = (
        out / "precondition.txt"
    ).read_text(errors="replace")

    got = sorted(
        observed_failures(text)
    )
    expected = sorted(
        task["failing_nodeids"]
    )

    if cp.returncode != 1 or got != expected:
        die(
            f"{task['task_id']} precondition "
            f"mismatch rc={cp.returncode} "
            f"expected={expected} observed={got}"
        )

    print("Precondition failure set: OK")


def condition_flag(cond):
    return 1 if cond == "ON" else 0


def write_metrics(
    out,
    task,
    cond,
    flag,
    agent_rc,
    oracle_rc,
    wall,
):
    rows = []

    p = out / "measured_requests.jsonl"
    if p.exists():
        for line in p.read_text(
            errors="replace"
        ).splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    def isum(key):
        vals = [
            r.get(key)
            for r in rows
            if isinstance(
                r.get(key),
                (int, float),
            )
        ]
        return sum(vals) if vals else None

    original = isum(
        "input_tokens_original"
    )
    optimized = isum(
        "input_tokens_optimized"
    )

    proxy_text = (
        out / "measured_proxy.log"
    ).read_text(
        errors="replace"
    ) if (
        out / "measured_proxy.log"
    ).exists() else ""

    status = (
        out / "git-status.txt"
    ).read_text(
        errors="replace"
    ) if (
        out / "git-status.txt"
    ).exists() else ""

    rg_files = list(
        (out / "rg-captures").glob(
            "*.stdout"
        )
    )

    metrics = {
        "task": task["task_id"],
        "condition": cond.lower(),
        "condition_flag": flag,
        "evaluation_type": (
            "post_hoc_naturalistic_transfer"
        ),
        "v3_freeze_sha": FREEZE,
        "manifest_commit": MANIFEST_COMMIT,
        "model": MODEL,
        "opencode_version": OPENCODE_VERSION,
        "agent_exit": agent_rc,
        "oracle_exit": oracle_rc,
        "task_pass": oracle_rc == 0,
        "wall_seconds": wall,
        "requests": len(rows),
        "input_tokens_original_sum": original,
        "input_tokens_optimized_sum": optimized,
        "derived_tokens_saved_sum": (
            original - optimized
            if original is not None
            and optimized is not None
            else None
        ),
        "reported_tokens_saved_sum": isum(
            "tokens_saved"
        ),
        "output_tokens_sum": isum(
            "output_tokens"
        ),
        "relevance_split_units": len(
            re.findall(
                r"strategy_chain=.*relevance_split",
                proxy_text,
                flags=re.I,
            )
        ),
        "search_relevance_chains": len(
            re.findall(
                r"strategy_chain=.*search.*"
                r"relevance_split",
                proxy_text,
                flags=re.I,
            )
        ),
        "treatment_activated": (
            flag == 1
            and bool(
                re.search(
                    r"strategy_chain=.*"
                    r"relevance_split",
                    proxy_text,
                    flags=re.I,
                )
            )
        ),
        "rg_capture_count": len(
            rg_files
        ),
        "rg_capture_bytes": sum(
            x.stat().st_size
            for x in rg_files
        ),
        "changed_files": [
            line[3:].strip()
            if len(line) >= 4
            else line.strip()
            for line in status.splitlines()
            if line.strip()
        ],
    }

    (out / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )


def run_condition(task, cond):
    flag = condition_flag(cond)
    tid = task["task_id"]

    out = ROOT / "results" / f"{tid}-{cond}"
    run_dir = ROOT / "runs" / f"{tid}-{cond}"

    if (out / "metrics.json").exists():
        die(f"{tid}-{cond} already completed")

    if (out / "AGENT_STARTED").exists():
        die(
            f"{tid}-{cond} already began but has no "
            "completed metrics. DO NOT rerun automatically."
        )

    if out.exists():
        shutil.rmtree(out)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    out.mkdir(parents=True)
    shutil.copytree(
        SNAPSHOTS / tid,
        run_dir,
    )

    print(
        f"Preparing {tid}-{cond} "
        f"(trajectory={flag})"
    )

    with (out / "uv-sync.txt").open("w") as f:
        cp = run(
            [
                "uv", "sync",
                "--python", "3.12",
                "--extra", "dev",
                "--extra", "proxy",
                "--frozen",
            ],
            cwd=run_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if cp.returncode:
        die(
            f"uv sync failed for {tid}-{cond}"
        )

    verify_precondition(
        run_dir,
        task,
        out,
    )

    port = find_port()
    proc, proxy_log = start_proxy(
        port,
        out,
        flag,
    )

    try:
        if not warm_proxy(
            port,
            out,
            flag,
        ):
            die(
                "Kompress warmup failed before "
                "benchmark agent"
            )

        print("Kompress warm.")

        requests_path = out / "requests.jsonl"

        request_start = 0
        if requests_path.exists():
            request_start = len(
                requests_path.read_text(
                    errors="replace"
                ).splitlines()
            )

        proxy_start = (
            GLOBAL_PROXY_LOG.stat().st_size
            if GLOBAL_PROXY_LOG.exists()
            else 0
        )

        capbin, capdir, real_rg = (
            create_rg_wrapper(out)
        )

        env = os.environ.copy()
        env.update({
            "HEADROOM_TRAJECTORY_RELEVANCE": str(flag),
            "HEADROOM_SAVINGS_PROFILE": "coding",
            "RG_CAPTURE_DIR": str(capdir),
            "REAL_RG": real_rg,
            "PATH": (
                str(capbin)
                + os.pathsep
                + env.get("PATH", "")
            ),
        })

        prompt = task["prompt"]

        got_hash = hashlib.sha256(
            prompt.encode()
        ).hexdigest()

        if got_hash != task["prompt_sha256"]:
            die(
                f"{tid} prompt hash mismatch"
            )

        (out / "AGENT_STARTED").write_text(
            json.dumps(
                {
                    "task": tid,
                    "condition": cond,
                    "flag": flag,
                    "time": time.time(),
                    "prompt_sha256": got_hash,
                },
                indent=2,
            ) + "\n"
        )

        start = time.monotonic()

        with (out / "opencode.txt").open("w") as f:
            cp = run(
                [
                    "uv", "run",
                    "--project", str(RUNTIME),
                    "--frozen",
                    "headroom", "wrap", "opencode",
                    "--no-proxy",
                    "--port", str(port),
                    "--no-mcp",
                    "--no-serena",
                    "--",
                    "run",
                    "-m", MODEL,
                    prompt,
                ],
                cwd=run_dir,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        wall = time.monotonic() - start
        agent_rc = cp.returncode

        measured = tail_jsonl(
            requests_path,
            request_start,
        )
        (out / "measured_requests.jsonl").write_text(
            measured
        )

        proxy_bytes = bytes_since(
            GLOBAL_PROXY_LOG,
            proxy_start,
        )
        (out / "measured_proxy.log").write_bytes(
            proxy_bytes
        )

    finally:
        stop_proxy(
            proc,
            proxy_log,
        )

    with (out / "oracle.txt").open("w") as f:
        oracle = run(
            [
                "uv", "run", "--frozen",
                "pytest", "-q",
                *task["test_files"],
            ],
            cwd=run_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    oracle_rc = oracle.returncode

    (out / "git-status.txt").write_text(
        capture(
            [
                "git", "status",
                "--porcelain=v1",
            ],
            run_dir,
        )
    )

    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=run_dir,
        stdout=subprocess.PIPE,
    ).stdout

    (out / "agent.patch").write_bytes(
        diff
    )

    (out / "wall_seconds.txt").write_text(
        f"{wall:.6f}\n"
    )

    write_metrics(
        out,
        task,
        cond,
        flag,
        agent_rc,
        oracle_rc,
        wall,
    )


def run_next():
    manifest = locked_manifest()
    prepare_runtime()
    snapshot_lock()
    verify_snapshots()
    opencode_version()

    for item in manifest["run_order"]:
        tid, cond = item.split("-", 1)
        out = ROOT / "results" / item

        if (out / "metrics.json").exists():
            continue

        if (out / "AGENT_STARTED").exists():
            die(
                f"{item} began but did not finish. "
                "Do NOT rerun automatically."
            )

        task = task_by_id(
            manifest,
            tid,
        )

        print()
        print("=" * 70)
        print(f"RUNNING NEXT LOCKED CONDITION: {item}")
        print("=" * 70)

        run_condition(
            task,
            cond,
        )
        return

    print(
        "All 16 locked natural-transfer "
        "conditions already have metrics."
    )


def show_status():
    manifest = locked_manifest()

    print(
        "===== NATURAL TRANSFER STATUS ====="
    )

    for item in manifest["run_order"]:
        out = ROOT / "results" / item

        if (out / "metrics.json").exists():
            d = json.loads(
                (out / "metrics.json").read_text()
            )
            print(
                f"{item:7} COMPLETE "
                f"pass={d['task_pass']} "
                f"act={d['treatment_activated']} "
                f"rg={d['rg_capture_count']} "
                f"req={d['requests']}"
            )
        elif (out / "AGENT_STARTED").exists():
            print(
                f"{item:7} STARTED-INCOMPLETE"
            )
        else:
            print(
                f"{item:7} PENDING"
            )


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
        "--run-next",
        action="store_true",
    )
    g.add_argument(
        "--status",
        action="store_true",
    )

    args = parser.parse_args()

    if args.prepare:
        prepare_snapshots()
    elif args.preflight:
        preflight()
    elif args.run_next:
        run_next()
    else:
        show_status()


if __name__ == "__main__":
    main()
