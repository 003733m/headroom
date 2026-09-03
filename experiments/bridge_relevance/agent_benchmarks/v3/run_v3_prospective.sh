#!/usr/bin/env bash
set -uo pipefail

CONTROL_REPO="$HOME/projects/headroom"

V3_FREEZE="7c4fe80c19b3a9537f395f515a64432e41ee1d43"
LOCK_COMMIT="c775a240ceefb62a6b59737acc92d2f1431fed26"
MANIFEST_REL="experiments/bridge_relevance/results/v3/prospective_task_manifest.json"

RUNTIME="/tmp/headroom-v3-runtime"
SNAPSHOT_ROOT="/tmp/headroom-v3-prospective-locked"
ROOT="/tmp/headroom-v3-prospective-final"

MODEL="openai/gpt-5.6-terra"
EXPECTED_OPENCODE_VERSION="1.18.25"

ORDER_TASKS=(N5 N5 N6 N6 N7 N7 N8 N8)
ORDER_CONDS=(OFF ON ON OFF OFF ON ON OFF)

PROXY_PID=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

stop_proxy() {
    local pid="${1:-}"

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    stop_proxy "${PROXY_PID:-}"
}

trap cleanup EXIT INT TERM


# ---------------------------------------------------------------------
# Locked manifest
# ---------------------------------------------------------------------

prepare_locked_manifest() {
    mkdir -p "$ROOT" "$ROOT/results" "$ROOT/runs" "$ROOT/prompts" "$ROOT/prime"

    local tmp="$ROOT/locked_manifest.json.tmp"
    local dst="$ROOT/locked_manifest.json"

    git -C "$CONTROL_REPO" show \
        "$LOCK_COMMIT:$MANIFEST_REL" > "$tmp" \
        || die "Cannot read manifest from lock commit."

    if [ -f "$dst" ]; then
        cmp -s "$tmp" "$dst" \
            || die "Existing locked_manifest.json differs from lock commit."
        rm -f "$tmp"
    else
        mv "$tmp" "$dst"
    fi

    python3 - "$dst" "$V3_FREEZE" <<'PY'
import json
import sys

path, freeze = sys.argv[1:]
d = json.load(open(path, encoding="utf-8"))

assert d["status"] == "locked_before_any_v3_prospective_agent_run"
assert d["freeze_sha"] == freeze
assert [t["id"] for t in d["tasks"]] == ["N5", "N6", "N7", "N8"]

expected = {
    "N5": ["OFF", "ON"],
    "N6": ["ON", "OFF"],
    "N7": ["OFF", "ON"],
    "N8": ["ON", "OFF"],
}

for task in d["tasks"]:
    assert task["order"] == expected[task["id"]]

print("Locked manifest: OK")
PY
}


task_scalar() {
    local task="$1"
    local field="$2"

    python3 - "$ROOT/locked_manifest.json" "$task" "$field" <<'PY'
import json
import sys

path, task_id, field = sys.argv[1:]
d = json.load(open(path, encoding="utf-8"))

task = next(t for t in d["tasks"] if t["id"] == task_id)
value = task[field]

if value is None:
    print("")
elif isinstance(value, (str, int, float)):
    print(value)
else:
    raise SystemExit(f"{field} is not scalar")
PY
}


task_snapshot_scalar() {
    local task="$1"
    local field="$2"

    python3 - "$ROOT/locked_manifest.json" "$task" "$field" <<'PY'
import json
import sys

path, task_id, field = sys.argv[1:]
d = json.load(open(path, encoding="utf-8"))

task = next(t for t in d["tasks"] if t["id"] == task_id)
value = task["snapshot"][field]

print("" if value is None else value)
PY
}


task_list() {
    local task="$1"
    local field="$2"

    python3 - "$ROOT/locked_manifest.json" "$task" "$field" <<'PY'
import json
import sys

path, task_id, field = sys.argv[1:]
d = json.load(open(path, encoding="utf-8"))

task = next(t for t in d["tasks"] if t["id"] == task_id)

for value in task[field]:
    print(value)
PY
}


write_prompt() {
    local task="$1"
    local out="$2"

    python3 - "$ROOT/locked_manifest.json" "$task" "$out" <<'PY'
import json
import pathlib
import sys

manifest, task_id, out = sys.argv[1:]
d = json.load(open(manifest, encoding="utf-8"))
task = next(t for t in d["tasks"] if t["id"] == task_id)

# Locked prompt hash was computed over the prompt without the trailing
# filesystem newline. Preserve exactly that text for the agent.
pathlib.Path(out).write_text(task["task_prompt"], encoding="utf-8")
PY
}


# ---------------------------------------------------------------------
# Exact frozen V3 runtime
# ---------------------------------------------------------------------

prepare_runtime() {
    git -C "$CONTROL_REPO" cat-file -e "$V3_FREEZE^{commit}" \
        || die "V3 freeze commit not available."

    git -C "$CONTROL_REPO" cat-file -e "$LOCK_COMMIT^{commit}" \
        || die "Prospective lock commit not available."

    if [ -e "$RUNTIME/.git" ]; then
        local got
        got="$(git -C "$RUNTIME" rev-parse HEAD)" \
            || die "Cannot read runtime HEAD."

        [ "$got" = "$V3_FREEZE" ] \
            || die "Runtime is not exact V3 freeze: $got"

        git -C "$RUNTIME" diff --quiet \
            || die "Frozen runtime has tracked modifications."

        git -C "$RUNTIME" diff --cached --quiet \
            || die "Frozen runtime has staged modifications."
    else
        if [ -e "$RUNTIME" ]; then
            die "$RUNTIME exists but is not a Git worktree."
        fi

        git -C "$CONTROL_REPO" worktree prune

        git -C "$CONTROL_REPO" worktree add \
            --detach "$RUNTIME" "$V3_FREEZE" \
            || die "Could not create frozen runtime worktree."
    fi

    local got
    got="$(git -C "$RUNTIME" rev-parse HEAD)"

    [ "$got" = "$V3_FREEZE" ] \
        || die "Runtime freeze verification failed."

    (
        cd "$RUNTIME" || exit 99
        uv sync --python 3.12 --extra dev --extra proxy --frozen
    ) > "$ROOT/runtime-uv-sync.txt" 2>&1 \
        || die "uv sync failed for frozen V3 runtime."

    echo "Frozen runtime: OK ($V3_FREEZE)"
}


# ---------------------------------------------------------------------
# Snapshot verification
# ---------------------------------------------------------------------

verify_snapshot() {
    local task="$1"
    local snapshot="$SNAPSHOT_ROOT/$task"

    [ -d "$snapshot/.git" ] \
        || die "$task locked snapshot is missing."

    local expected_head got_head
    expected_head="$(task_snapshot_scalar "$task" head)"
    got_head="$(git -C "$snapshot" rev-parse HEAD)"

    [ "$got_head" = "$expected_head" ] \
        || die "$task snapshot HEAD mismatch."

    local count
    count="$(git -C "$snapshot" rev-list --all --count)"

    [ "$count" = "1" ] \
        || die "$task snapshot exposes more than one Git commit."

    [ -z "$(git -C "$snapshot" status --porcelain)" ] \
        || die "$task locked snapshot is dirty."

    local expected_lock
    expected_lock="$(task_snapshot_scalar "$task" uv_lock_sha256)"

    if [ -n "$expected_lock" ]; then
        local got_lock
        got_lock="$(sha256sum "$snapshot/uv.lock" | awk '{print $1}')"

        [ "$got_lock" = "$expected_lock" ] \
            || die "$task uv.lock hash mismatch."
    fi

    echo "$task snapshot: OK ($got_head)"
}


# ---------------------------------------------------------------------
# No-model proxy smoke test
# ---------------------------------------------------------------------

smoke_proxy() {
    local port=9380
    local out="$ROOT/preflight-proxy.log"

    while ss -ltn 2>/dev/null | grep -q ":$port "; do
        port=$((port + 1))
    done

    (
        cd "$RUNTIME" || exit 99

        env \
          HEADROOM_TRAJECTORY_RELEVANCE=0 \
          HEADROOM_SAVINGS_PROFILE=coding \
          HEADROOM_AGENT_TYPE=opencode \
          HEADROOM_STACK=wrap_opencode \
          HEADROOM_TELEMETRY=off \
          uv run --frozen headroom proxy \
            --port "$port" \
            --log-file "$ROOT/preflight-requests.jsonl"
    ) > "$out" 2>&1 &

    PROXY_PID=$!

    local ready=0

    for _ in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:$port/livez" \
            > "$ROOT/preflight-livez.json" 2>/dev/null; then
            ready=1
            break
        fi

        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            tail -100 "$out"
            die "Preflight proxy died."
        fi

        sleep 1
    done

    stop_proxy "$PROXY_PID"
    PROXY_PID=""

    [ "$ready" -eq 1 ] \
        || die "Preflight proxy did not become ready."

    echo "Proxy smoke test: OK (no benchmark agent call)"
}


# ---------------------------------------------------------------------
# Old-harness-compatible Kompress warmup
# ---------------------------------------------------------------------

warm_proxy() {
    local port="$1"
    local out="$2"
    local condition="$3"

    cat > "$ROOT/prime/prime.sh" <<'EOF'
#!/usr/bin/env bash
python3 - <<'PY'
for i in range(1600):
    print(
        f"Warmup record {i:04d}: deterministic diagnostic payload "
        f"segment {i:04d}; subsystem state stable."
    )
PY
EOF

    chmod +x "$ROOT/prime/prime.sh"

    (
        cd "$ROOT/prime" || exit 99

        env \
          HEADROOM_TRAJECTORY_RELEVANCE="$condition" \
          HEADROOM_SAVINGS_PROFILE=coding \
          uv run --project "$RUNTIME" --frozen \
            headroom wrap opencode \
              --no-proxy \
              --port "$port" \
              --no-mcp \
              --no-serena \
              -- run \
              -m "$MODEL" \
              "Run ./prime.sh exactly once and then reply only PRIME_DONE."
    ) > "$out/prime.txt" 2>&1

    local prime_rc=$?

    if [ "$prime_rc" -ne 0 ]; then
        echo "Prime agent failed rc=$prime_rc"
        return 1
    fi

    for _ in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:$port/debug/warmup" \
            > "$out/warmup.json" 2>/dev/null; then

            python3 - "$out/warmup.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)

def find_status(x):
    if isinstance(x, dict):
        k = x.get("kompress")
        if isinstance(k, dict):
            return k.get("status")

        for v in x.values():
            s = find_status(v)
            if s is not None:
                return s

    elif isinstance(x, list):
        for v in x:
            s = find_status(v)
            if s is not None:
                return s

    return None

raise SystemExit(0 if find_status(d) == "loaded" else 1)
PY
            if [ $? -eq 0 ]; then
                return 0
            fi
        fi

        sleep 1
    done

    return 1
}


# ---------------------------------------------------------------------
# Precondition validation
# ---------------------------------------------------------------------

verify_failure_set() {
    local task="$1"
    local pytest_file="$2"
    local rc="$3"

    python3 - \
        "$ROOT/locked_manifest.json" \
        "$task" \
        "$pytest_file" \
        "$rc" <<'PY'
import json
import pathlib
import re
import sys

manifest, task_id, output_path, rc = sys.argv[1:]
rc = int(rc)

d = json.load(open(manifest, encoding="utf-8"))
task = next(t for t in d["tasks"] if t["id"] == task_id)

text = pathlib.Path(output_path).read_text(
    encoding="utf-8",
    errors="replace",
)

observed = re.findall(
    r"^FAILED\s+([^\s]+)",
    text,
    flags=re.MULTILINE,
)

expected = task["expected_buggy_failures"]

if rc != 1:
    raise SystemExit(
        f"precondition rc={rc}, expected 1"
    )

if sorted(observed) != sorted(expected):
    raise SystemExit(
        "precondition failure mismatch\n"
        f"expected={expected}\n"
        f"observed={observed}"
    )

print("Precondition failure set: OK")
PY
}


# ---------------------------------------------------------------------
# Result metrics
# ---------------------------------------------------------------------

write_metrics() {
    local out="$1"
    local task="$2"
    local cond="$3"
    local flag="$4"
    local agent_rc="$5"
    local oracle_rc="$6"
    local start_tree="$7"
    local prompt_hash="$8"
    local snapshot_head="$9"

    python3 - \
        "$out" \
        "$ROOT/locked_manifest.json" \
        "$task" \
        "$cond" \
        "$flag" \
        "$agent_rc" \
        "$oracle_rc" \
        "$start_tree" \
        "$prompt_hash" \
        "$snapshot_head" \
        "$V3_FREEZE" \
        "$LOCK_COMMIT" \
        "$MODEL" \
        "$EXPECTED_OPENCODE_VERSION" <<'PY'
import json
import pathlib
import re
import sys

(
    out_s,
    manifest_s,
    task_id,
    cond,
    flag,
    agent_rc,
    oracle_rc,
    start_tree,
    prompt_hash,
    snapshot_head,
    freeze_sha,
    lock_commit,
    model,
    opencode_version,
) = sys.argv[1:]

out = pathlib.Path(out_s)
manifest = json.load(open(manifest_s, encoding="utf-8"))
task = next(t for t in manifest["tasks"] if t["id"] == task_id)

agent_rc = int(agent_rc)
oracle_rc = int(oracle_rc)

rows = []

requests_path = out / "measured_requests.jsonl"
if requests_path.exists():
    for line in requests_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if not line.strip():
            continue

        try:
            rows.append(json.loads(line))
        except Exception:
            pass


def isum(key):
    total = 0
    seen = 0

    for row in rows:
        value = row.get(key)

        if isinstance(value, (int, float)):
            total += value
            seen += 1

    return total if seen else None


orig = isum("input_tokens_original")
opt = isum("input_tokens_optimized")
reported_saved = isum("tokens_saved")
output_tokens = isum("output_tokens")

derived_saved = None
if orig is not None and opt is not None:
    derived_saved = orig - opt

proxy_text = ""
p = out / "measured_proxy.log"
if p.exists():
    proxy_text = p.read_text(
        encoding="utf-8",
        errors="replace",
    )

opencode_text = ""
p = out / "opencode.txt"
if p.exists():
    opencode_text = p.read_text(
        encoding="utf-8",
        errors="replace",
    )

status_text = ""
p = out / "git-status.txt"
if p.exists():
    status_text = p.read_text(
        encoding="utf-8",
        errors="replace",
    )

wall = float(
    (out / "wall_seconds.txt").read_text().strip()
)

relevance_split_units = len(
    re.findall(
        r"strategy_chain=.*relevance_split",
        proxy_text,
        flags=re.I,
    )
)

search_relevance_chains = len(
    re.findall(
        r"strategy_chain=.*search.*relevance_split",
        proxy_text,
        flags=re.I,
    )
)

critical = [
    *task["critical_evidence"]["test_names"],
    *task["critical_evidence"]["production_paths"],
]

critical_observation = {}

for identifier in critical:
    critical_observation[identifier] = {
        "agent_transcript_occurrences": opencode_text.count(identifier),
        "measured_proxy_log_occurrences": proxy_text.count(identifier),
    }

changed_files = []

for line in status_text.splitlines():
    if not line.strip():
        continue

    # porcelain format: XY<space>path
    changed_files.append(
        line[3:].strip() if len(line) >= 4 else line.strip()
    )

metrics = {
    "task": task_id,
    "condition": cond.lower(),
    "condition_flag": int(flag),
    "evaluation_type": "prospective_frozen_v3_condition",
    "v3_freeze_sha": freeze_sha,
    "prospective_lock_commit": lock_commit,
    "snapshot_sha": snapshot_head,
    "start_tree": start_tree,
    "prompt_sha256": prompt_hash,
    "model": model,
    "opencode_version": opencode_version,
    "agent_exit": agent_rc,
    "oracle_exit": oracle_rc,
    "task_pass": oracle_rc == 0,
    "wall_seconds": wall,
    "requests": len(rows),
    "input_tokens_original_sum": orig,
    "input_tokens_optimized_sum": opt,
    "derived_tokens_saved_sum": derived_saved,
    "reported_tokens_saved_sum": reported_saved,
    "output_tokens_sum": output_tokens,
    "relevance_split_units": relevance_split_units,
    "search_relevance_chains": search_relevance_chains,
    "treatment_activated": (
        int(flag) == 1 and relevance_split_units > 0
    ),
    "lossless_search_mentions": len(
        re.findall(r"lossless_search", proxy_text, flags=re.I)
    ),
    "search_result_like_units": len(
        re.findall(
            r"text_shape=search_result_like",
            proxy_text,
            flags=re.I,
        )
    ),
    "source_code_units": len(
        re.findall(
            r"content_type=source_code",
            proxy_text,
            flags=re.I,
        )
    ),
    "dedicated_grep_mentions": len(
        re.findall(r"(?:✱\s*)?Grep", opencode_text)
    ),
    "shell_rg_mentions": len(
        re.findall(r"\brg\s+", opencode_text)
    ),
    "changed_files": changed_files,
    "critical_evidence_observation": critical_observation,
}

(out / "metrics.json").write_text(
    json.dumps(
        metrics,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print(
    json.dumps(
        metrics,
        indent=2,
        ensure_ascii=False,
    )
)
PY
}


# ---------------------------------------------------------------------
# Preflight: deliberately NO benchmark-agent/model call
# ---------------------------------------------------------------------

preflight() {
    prepare_locked_manifest
    prepare_runtime

    local version
    version="$(opencode --version 2>/dev/null | tail -1 | tr -d '\r')"

    [ "$version" = "$EXPECTED_OPENCODE_VERSION" ] \
        || die "OpenCode version mismatch: expected $EXPECTED_OPENCODE_VERSION, got $version"

    echo "OpenCode: OK ($version)"
    echo "Model pinned: $MODEL"

    for task in N5 N6 N7 N8; do
        verify_snapshot "$task"

        local prompt_file="$ROOT/prompts/$task.txt"
        write_prompt "$task" "$prompt_file"

        local got_prompt expected_prompt
        got_prompt="$(sha256sum "$prompt_file" | awk '{print $1}')"
        expected_prompt="$(task_scalar "$task" task_prompt_sha256)"

        [ "$got_prompt" = "$expected_prompt" ] \
            || die "$task prompt hash mismatch."

        echo "$task prompt: OK ($got_prompt)"
    done

    smoke_proxy

    echo
    echo "=================================================="
    echo "V3 PROSPECTIVE PREFLIGHT PASSED"
    echo "No prospective benchmark agent was run."
    echo "freeze: $V3_FREEZE"
    echo "lock  : $LOCK_COMMIT"
    echo "next  : N5 OFF"
    echo "=================================================="
}


# ---------------------------------------------------------------------
# Execute exactly one next prospective condition
# ---------------------------------------------------------------------

run_next() {
    prepare_locked_manifest
    prepare_runtime

    local version
    version="$(opencode --version 2>/dev/null | tail -1 | tr -d '\r')"

    [ "$version" = "$EXPECTED_OPENCODE_VERSION" ] \
        || die "OpenCode version mismatch."

    local task=""
    local cond=""
    local index=-1

    for i in "${!ORDER_TASKS[@]}"; do
        local t="${ORDER_TASKS[$i]}"
        local c="${ORDER_CONDS[$i]}"
        local o="$ROOT/results/${t}-${c}"

        if [ -f "$o/metrics.json" ]; then
            continue
        fi

        if [ -f "$o/AGENT_STARTED" ]; then
            die "$t $c already began but has no completed metrics. Do NOT rerun automatically."
        fi

        task="$t"
        cond="$c"
        index="$i"
        break
    done

    if [ -z "$task" ]; then
        echo "All eight locked prospective conditions already have metrics."
        return 0
    fi

    local flag
    if [ "$cond" = "ON" ]; then
        flag=1
    else
        flag=0
    fi

    echo
    echo "=================================================="
    echo "NEXT LOCKED PROSPECTIVE CONDITION"
    echo "task      : $task"
    echo "condition : $cond"
    echo "flag      : $flag"
    echo "sequence  : $((index + 1))/8"
    echo "freeze    : $V3_FREEZE"
    echo "=================================================="

    verify_snapshot "$task"

    local snapshot="$SNAPSHOT_ROOT/$task"
    local expected_snapshot_head
    expected_snapshot_head="$(task_snapshot_scalar "$task" head)"

    local run="$ROOT/runs/${task}-${cond}"
    local out="$ROOT/results/${task}-${cond}"
    local prompt_file="$ROOT/prompts/$task.txt"

    rm -rf "$run" "$out"
    mkdir -p "$out"

    git clone -q --no-hardlinks "$snapshot" "$run" \
        || die "Could not clone locked $task snapshot."

    # uv.lock exists in the sanitized snapshot and is hash-locked in the
    # prospective manifest, but it is Git-ignored and therefore is not copied
    # by `git clone`. Restore that exact locked file explicitly.
    if [ -f "$snapshot/uv.lock" ]; then
        cp "$snapshot/uv.lock" "$run/uv.lock" \
            || die "Could not restore locked uv.lock for $task $cond."

        local expected_run_lock got_run_lock
        expected_run_lock="$(task_snapshot_scalar "$task" uv_lock_sha256)"
        got_run_lock="$(sha256sum "$run/uv.lock" | awk '{print $1}')"

        [ "$got_run_lock" = "$expected_run_lock" ] \
            || die "$task $cond restored uv.lock hash mismatch."
    else
        die "$task locked snapshot unexpectedly has no uv.lock."
    fi

    local start_head
    start_head="$(git -C "$run" rev-parse HEAD)"

    [ "$start_head" = "$expected_snapshot_head" ] \
        || die "$task $cond cloned snapshot HEAD mismatch."

    [ "$(git -C "$run" rev-list --all --count)" = "1" ] \
        || die "$task $cond run clone exposes historical Git commits."

    write_prompt "$task" "$prompt_file"

    local prompt_hash expected_prompt_hash
    prompt_hash="$(sha256sum "$prompt_file" | awk '{print $1}')"
    expected_prompt_hash="$(task_scalar "$task" task_prompt_sha256)"

    [ "$prompt_hash" = "$expected_prompt_hash" ] \
        || die "$task prompt hash changed."

    (
        cd "$run" || exit 99
        uv sync --python 3.12 --extra dev --extra proxy --frozen
    ) > "$out/uv-sync.txt" 2>&1 \
        || die "uv sync failed for $task $cond."

    mapfile -t TESTS < <(task_list "$task" oracle_tests)

    (
        cd "$run" || exit 99

        env HEADROOM_TRAJECTORY_RELEVANCE=0 \
            uv run --frozen pytest \
            -q \
            --tb=no \
            "${TESTS[@]}"
    ) > "$out/precondition.txt" 2>&1

    local pre_rc=$?

    verify_failure_set \
        "$task" \
        "$out/precondition.txt" \
        "$pre_rc" \
        || {
            tail -80 "$out/precondition.txt"
            die "$task $cond precondition mismatch."
        }

    local start_tree
    start_tree="$(git -C "$run" rev-parse HEAD^{tree})"

    local port=$((9400 + index))

    while ss -ltn 2>/dev/null | grep -q ":$port "; do
        port=$((port + 100))
    done

    (
        cd "$RUNTIME" || exit 99

        env \
          HEADROOM_TRAJECTORY_RELEVANCE="$flag" \
          HEADROOM_SAVINGS_PROFILE=coding \
          HEADROOM_AGENT_TYPE=opencode \
          HEADROOM_STACK=wrap_opencode \
          HEADROOM_TELEMETRY=off \
          uv run --frozen headroom proxy \
            --port "$port" \
            --log-file "$out/requests.jsonl"
    ) > "$out/proxy_process.log" 2>&1 &

    PROXY_PID=$!

    local ready=0

    for _ in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:$port/livez" \
            > "$out/livez.json" 2>/dev/null; then
            ready=1
            break
        fi

        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            tail -100 "$out/proxy_process.log"
            die "Proxy died before $task $cond."
        fi

        sleep 1
    done

    [ "$ready" -eq 1 ] || die "Proxy never became ready."

    if ! warm_proxy "$port" "$out" "$flag"; then
        stop_proxy "$PROXY_PID"
        PROXY_PID=""
        die "Kompress warmup failed before benchmark agent."
    fi

    echo "Kompress warm."

    # Measurement starts only after warmup.
    local request_start=0
    if [ -f "$out/requests.jsonl" ]; then
        request_start="$(wc -l < "$out/requests.jsonl")"
    fi

    local global_log="$HOME/.headroom/logs/proxy.log"
    local global_start=0

    if [ -f "$global_log" ]; then
        global_start="$(wc -l < "$global_log")"
    fi

    local start_ns
    start_ns="$(date +%s%N)"

    # From this marker onward the condition is prospectively consumed.
    # If interrupted after here, DO NOT automatically rerun it.
    {
        echo "task=$task"
        echo "condition=$cond"
        echo "started_ns=$start_ns"
        echo "freeze=$V3_FREEZE"
        echo "snapshot=$start_head"
        echo "prompt_sha256=$prompt_hash"
    } > "$out/AGENT_STARTED"

    (
        cd "$run" || exit 99

        env \
          HEADROOM_TRAJECTORY_RELEVANCE="$flag" \
          HEADROOM_SAVINGS_PROFILE=coding \
          uv run --project "$RUNTIME" --frozen \
            headroom wrap opencode \
              --no-proxy \
              --port "$port" \
              --no-mcp \
              --no-serena \
              -- run \
              -m "$MODEL" \
              "$(cat "$prompt_file")"
    ) > "$out/opencode.txt" 2>&1

    local agent_rc=$?
    local end_ns
    end_ns="$(date +%s%N)"

    {
        echo "agent_rc=$agent_rc"
        echo "finished_ns=$end_ns"
    } > "$out/AGENT_FINISHED"

    python3 - "$start_ns" "$end_ns" > "$out/wall_seconds.txt" <<'PY'
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])

print(f"{(end - start) / 1_000_000_000:.6f}")
PY

    if [ -f "$out/requests.jsonl" ]; then
        tail -n "+$((request_start + 1))" \
            "$out/requests.jsonl" \
            > "$out/measured_requests.jsonl"
    else
        : > "$out/measured_requests.jsonl"
    fi

    if [ -f "$global_log" ]; then
        tail -n "+$((global_start + 1))" \
            "$global_log" \
            > "$out/measured_proxy.log"
    else
        : > "$out/measured_proxy.log"
    fi

    stop_proxy "$PROXY_PID"
    PROXY_PID=""

    (
        cd "$run" || exit 99
        uv run --frozen pytest -q "${TESTS[@]}"
    ) > "$out/oracle.txt" 2>&1

    local oracle_rc=$?

    git -C "$run" status --porcelain > "$out/git-status.txt"
    git -C "$run" diff > "$out/agent.patch"
    git -C "$run" diff --stat > "$out/diff-stat.txt"

    write_metrics \
        "$out" \
        "$task" \
        "$cond" \
        "$flag" \
        "$agent_rc" \
        "$oracle_rc" \
        "$start_tree" \
        "$prompt_hash" \
        "$start_head"

    cp "$out/metrics.json" \
        "$ROOT/results/${task}-${cond}-metrics.json"

    echo
    echo "=================================================="
    echo "$task $cond COMPLETE"
    echo "=================================================="
    cat "$out/metrics.json"

    echo
    echo "Oracle tail:"
    tail -15 "$out/oracle.txt"

    echo
    echo "IMPORTANT: this condition is now consumed."
    echo "Do not rerun or replace it based on outcome."
    echo

    # Keep source/results, remove bulky environment.
    rm -rf "$run/.venv"
}


usage() {
    cat <<EOF
Usage:
  $0 --preflight
  $0 --run-next

--preflight
    Verifies frozen runtime, locked manifest, snapshots, prompt hashes,
    OpenCode version, and proxy startup. It does NOT run a prospective
    benchmark agent.

--run-next
    Runs exactly ONE next locked condition in this sequence:
      N5 OFF
      N5 ON
      N6 ON
      N6 OFF
      N7 OFF
      N7 ON
      N8 ON
      N8 OFF

    A condition is never automatically rerun after AGENT_STARTED exists.
EOF
}


case "${1:-}" in
    --preflight)
        preflight
        ;;
    --run-next)
        run_next
        ;;
    *)
        usage
        exit 2
        ;;
esac
