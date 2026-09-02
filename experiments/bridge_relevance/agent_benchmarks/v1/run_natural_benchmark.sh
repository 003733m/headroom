#!/usr/bin/env bash
set -uo pipefail

MAIN="$HOME/projects/headroom"
ROOT="/tmp/headroom-natural-eval-final"
MODEL="openai/gpt-5.6-terra"

cd "$MAIN" || exit 1

FREEZE="$(git rev-parse HEAD)"
FREEZE_SHORT="$(git rev-parse --short HEAD)"

echo "=================================================="
echo "FINAL NATURAL CODING-AGENT BENCHMARK"
echo "freeze: $FREEZE_SHORT"
echo "=================================================="

if [ "$FREEZE_SHORT" != "2fffb401" ]; then
    echo "WARNING: expected freeze 2fffb401, got $FREEZE_SHORT"
    echo "Aborting rather than silently benchmarking another implementation."
    exit 1
fi

mkdir -p "$ROOT/templates" "$ROOT/runs" "$ROOT/results" "$ROOT/prompts" "$ROOT/prime"

# --------------------------------------------------
# Frozen task definitions
# --------------------------------------------------

declare -A FIX
declare -A PROD
declare -A TESTS
declare -A PROMPT

FIX[T1]="7784bb18"
PROD[T1]="headroom/transforms/content_detector.py"
TESTS[T1]="tests/test_transforms_content_detection.py tests/test_transforms_content_router.py"
PROMPT[T1]="A user message prefixed by an XML-like current_datetime line is being misclassified as grep/search results, which can cause the remaining prompt content to be dropped during routing/compression.

Diagnose the cause and implement a minimal robust fix. Preserve legitimate grep/search-result detection. Run the relevant tests to verify the fix.
Do not make unrelated changes."

FIX[T2]="455f4f26"
PROD[T2]="headroom/cache/semantic.py"
TESTS[T2]="tests/test_cache/test_semantic.py"
PROMPT[T2]="Semantic cache lookups can incorrectly match entries across unrelated contexts when the query string is empty.

Diagnose the cause and implement a minimal robust fix. Preserve normal semantic-cache matching behavior for meaningful non-empty queries. Run the relevant tests to verify the fix.
Do not make unrelated changes."

FIX[T3]="6262c28a"
PROD[T3]="headroom/memory/adapters/sqlite_graph.py"
TESTS[T3]="tests/test_sqlite_graph_store.py"
PROMPT[T3]="A single corrupt row in the SQLite graph store can abort an entire graph scan, preventing otherwise valid rows from being returned.

Diagnose the cause and implement a minimal robust fix so malformed rows are handled safely while valid rows continue to be processed. Run the relevant tests to verify the fix.
Do not make unrelated changes."

FIX[T4]="7c0b8860"
PROD[T4]="headroom/providers/vertex/runtime.py"
TESTS[T4]="tests/test_provider_vertex_runtime.py"
PROMPT[T4]="Vertex location input is interpolated into the regional API hostname without sufficiently strict validation, allowing malformed or unsafe location values to produce unintended targets.

Diagnose the cause and implement a minimal robust fix. Preserve valid Vertex regional routing while safely falling back for invalid location values. Run the relevant tests to verify the fix.
Do not make unrelated changes."

FIX[T5]="1e448b55"
PROD[T5]="headroom/providers/registry.py"
TESTS[T5]="tests/test_provider_registry.py"
PROMPT[T5]="When the configured OpenAI target is GitHub Copilot, Claude/Anthropic requests are still routed to the normal Anthropic endpoint instead of following the managed Copilot target.

Diagnose the routing decision and implement a minimal robust fix. Preserve normal provider routing for non-Copilot targets. Run the relevant tests to verify the fix.
Do not make unrelated changes."

FIX[T6]="284ff319"
PROD[T6]="headroom/proxy/body_forwarding.py"
TESTS[T6]="tests/test_proxy_byte_faithful_forwarding.py"
PROMPT[T6]="A lone Unicode surrogate in thinking/reasoning content can cause the body-forwarding path to raise an encoding error and turn the request into a server 500.

Diagnose the cause and implement a minimal robust fix that avoids the crash while preserving byte-faithful forwarding behavior. Run the relevant tests to verify the fix.
Do not make unrelated changes."

TASKS=(T1 T2 T3 T4 T5 T6)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

is_prod_path() {
    local task="$1"
    local candidate="$2"

    for p in ${PROD[$task]}; do
        if [ "$p" = "$candidate" ]; then
            return 0
        fi
    done

    return 1
}

run_tests() {
    local dir="$1"
    local task="$2"
    local outfile="$3"

    (
        cd "$dir" || exit 99

        # Build/install this snapshot in its own environment so the native
        # headroom._core extension matches the Python sources under test.
        uv sync --extra dev --extra proxy --frozen

        uv run --frozen pytest ${TESTS[$task]} -q
    ) >"$outfile" 2>&1

    return $?
}

stop_proxy() {
    local pid="${1:-}"

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

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
          uv run --project "$MAIN" --frozen \
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

    for i in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:$port/debug/warmup" \
            > "$out/warmup.json" 2>/dev/null; then

            python3 - "$out/warmup.json" <<'PY'
import json, sys

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

# --------------------------------------------------
# Freeze prompts now, before results
# --------------------------------------------------

for task in "${TASKS[@]}"; do
    printf '%s\n' "${PROMPT[$task]}" > "$ROOT/prompts/$task.txt"
done

echo
echo "===== FROZEN PROMPT HASHES ====="
sha256sum "$ROOT"/prompts/*.txt

# --------------------------------------------------
# PHASE 1: build all buggy snapshots first
# --------------------------------------------------

echo
echo "=================================================="
echo "PHASE 1 — PREPARE BUGGY SNAPSHOTS"
echo "=================================================="

for task in "${TASKS[@]}"; do
    template="$ROOT/templates/$task"

    echo
    echo "----- $task / ${FIX[$task]} -----"

    rm -rf "$template"

    if ! git clone -q --shared "$MAIN" "$template"; then
        echo "ERROR: clone failed for $task"
        exit 1
    fi

    (
        cd "$template" || exit 99

        git checkout -q "$FREEZE"

        # Reverse the historical bug-fix commit using the real git history.
        # Then restore every non-production file to the frozen implementation,
        # keeping current regression tests rather than historical tests.
        if ! git revert -n "${FIX[$task]}"; then
            echo "ERROR: git revert conflict for $task"
            git status
            exit 20
        fi

        mapfile -t changed < <(git diff HEAD --name-only)

        for f in "${changed[@]}"; do
            if ! is_prod_path "$task" "$f"; then
                git restore --source=HEAD --staged --worktree -- "$f" 2>/dev/null || true
            fi
        done

        echo "Production diff retained:"
        git diff HEAD --name-status

        remaining="$(git diff HEAD --name-only)"

        if [ -z "$remaining" ]; then
            echo "ERROR: no buggy production diff remains for $task"
            exit 21
        fi

        while IFS= read -r f; do
            [ -z "$f" ] && continue
            if ! is_prod_path "$task" "$f"; then
                echo "ERROR: unexpected changed path after revert: $f"
                exit 22
            fi
        done <<< "$remaining"

        # Remove all historical history so the coding agent cannot inspect
        # the original fix commit.
        rm -rf .git

        git init -q
        git config user.email "benchmark@example.invalid"
        git config user.name "Natural Benchmark"

        git add -A
        [ -f uv.lock ] && git add -f uv.lock >/dev/null 2>&1 || true
        git commit -qm "Buggy benchmark snapshot"

        echo "snapshot tree: $(git rev-parse HEAD^{tree})"
    )

    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: preparation failed for $task rc=$rc"
        exit 1
    fi

    # Check the intended current regression suite actually fails BEFORE
    # spending model calls.
    run_tests "$template" "$task" "$ROOT/results/${task}-template-precondition.txt"
    pre_rc=$?

    if [ "$pre_rc" -eq 0 ]; then
        echo "ERROR: $task regression suite unexpectedly passes."
        echo "This is not a valid bug reversion."
        exit 1
    fi

    if [ "$pre_rc" -ne 1 ]; then
        echo "ERROR: $task precondition returned pytest rc=$pre_rc instead of 1."
        tail -80 "$ROOT/results/${task}-template-precondition.txt"
        exit 1
    fi

    echo "$task precondition: EXPECTED FAIL"
done

echo
echo "All six snapshots reproduce a failing regression suite."
echo

# --------------------------------------------------
# PHASE 2: natural OpenCode OFF/ON runs
# --------------------------------------------------

echo "=================================================="
echo "PHASE 2 — NATURAL AGENT A/B"
echo "=================================================="

RUN_NO=0

for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    task_num=$((idx + 1))

    # Frozen counterbalancing:
    # T1 OFF→ON, T2 ON→OFF, ...
    if (( task_num % 2 == 1 )); then
        CONDITIONS=(off on)
    else
        CONDITIONS=(on off)
    fi

    for cond in "${CONDITIONS[@]}"; do
        RUN_NO=$((RUN_NO + 1))

        if [ "$cond" = "on" ]; then
            FLAG=1
        else
            FLAG=0
        fi

        run="$ROOT/runs/${task}-${cond}"
        out="$ROOT/results/${task}-${cond}"

        rm -rf "$run" "$out"
        mkdir -p "$out"

        echo
        echo "=================================================="
        echo "RUN $RUN_NO/12 — $task $cond"
        echo "=================================================="

        git clone -q "$ROOT/templates/$task" "$run"

        START_TREE="$(git -C "$run" rev-parse HEAD^{tree})"
        PROMPT_HASH="$(sha256sum "$ROOT/prompts/$task.txt" | awk '{print $1}')"

        echo "$START_TREE" > "$out/start_tree.txt"
        echo "$PROMPT_HASH" > "$out/prompt_sha256.txt"
        cp "$ROOT/prompts/$task.txt" "$out/prompt.txt"

        # Prepare dependencies OUTSIDE measured agent wall time.
        (
            cd "$run" || exit 99
            uv sync --extra dev --extra proxy --frozen
        ) > "$out/uv-sync.txt" 2>&1

        if [ $? -ne 0 ]; then
            echo "ERROR: uv sync failed for $task $cond"
            tail -100 "$out/uv-sync.txt"
            exit 1
        fi

        # Independent condition precondition.
        (
            cd "$run" || exit 99
            uv run --frozen pytest ${TESTS[$task]} -q
        ) > "$out/precondition.txt" 2>&1

        pre_rc=$?

        if [ "$pre_rc" -ne 1 ]; then
            echo "ERROR: invalid precondition $task $cond rc=$pre_rc"
            tail -80 "$out/precondition.txt"
            exit 1
        fi

        PORT=$((8900 + RUN_NO))
        while ss -ltn 2>/dev/null | grep -q ":$PORT "; do
            PORT=$((PORT + 100))
        done

        cd "$MAIN"

        env \
          HEADROOM_TRAJECTORY_RELEVANCE="$FLAG" \
          HEADROOM_SAVINGS_PROFILE=coding \
          HEADROOM_AGENT_TYPE=opencode \
          HEADROOM_STACK=wrap_opencode \
          HEADROOM_TELEMETRY=off \
          uv run --frozen headroom proxy \
            --port "$PORT" \
            --log-file "$out/requests.jsonl" \
            > "$out/proxy_process.log" 2>&1 &

        PROXY_PID=$!

        READY=0

        for i in $(seq 1 120); do
            if curl -fsS "http://127.0.0.1:$PORT/livez" \
                > "$out/livez.json" 2>/dev/null; then
                READY=1
                break
            fi

            if ! kill -0 "$PROXY_PID" 2>/dev/null; then
                echo "ERROR: proxy died for $task $cond"
                tail -100 "$out/proxy_process.log"
                exit 1
            fi

            sleep 1
        done

        if [ "$READY" -ne 1 ]; then
            echo "ERROR: proxy never became ready for $task $cond"
            stop_proxy "$PROXY_PID"
            exit 1
        fi

        if ! warm_proxy "$PORT" "$out" "$FLAG"; then
            echo "ERROR: Kompress warmup failed for $task $cond"
            stop_proxy "$PROXY_PID"
            exit 1
        fi

        echo "Kompress warm."

        # Measurement boundaries AFTER warmup.
        if [ -f "$out/requests.jsonl" ]; then
            REQUEST_START="$(wc -l < "$out/requests.jsonl")"
        else
            REQUEST_START=0
        fi

        GLOBAL_LOG="$HOME/.headroom/logs/proxy.log"

        if [ -f "$GLOBAL_LOG" ]; then
            GLOBAL_START="$(wc -l < "$GLOBAL_LOG")"
        else
            GLOBAL_START=0
        fi

        START_NS="$(date +%s%N)"

        (
            cd "$run" || exit 99

            env \
              HEADROOM_TRAJECTORY_RELEVANCE="$FLAG" \
              HEADROOM_SAVINGS_PROFILE=coding \
              uv run --project "$MAIN" --frozen \
                headroom wrap opencode \
                  --no-proxy \
                  --port "$PORT" \
                  --no-mcp \
                  --no-serena \
                  -- run \
                  -m "$MODEL" \
                  "$(cat "$ROOT/prompts/$task.txt")"
        ) > "$out/opencode.txt" 2>&1

        AGENT_RC=$?

        END_NS="$(date +%s%N)"

        python3 - "$START_NS" "$END_NS" > "$out/wall_seconds.txt" <<'PY'
import sys
start = int(sys.argv[1])
end = int(sys.argv[2])
print(f"{(end-start)/1_000_000_000:.6f}")
PY

        # Cut out ONLY measured requests.
        if [ -f "$out/requests.jsonl" ]; then
            tail -n "+$((REQUEST_START + 1))" \
                "$out/requests.jsonl" \
                > "$out/measured_requests.jsonl"
        else
            : > "$out/measured_requests.jsonl"
        fi

        if [ -f "$GLOBAL_LOG" ]; then
            tail -n "+$((GLOBAL_START + 1))" \
                "$GLOBAL_LOG" \
                > "$out/measured_proxy.log"
        else
            : > "$out/measured_proxy.log"
        fi

        stop_proxy "$PROXY_PID"

        # Frozen external oracle.
        (
            cd "$run" || exit 99
            uv run --frozen pytest ${TESTS[$task]} -q
        ) > "$out/oracle.txt" 2>&1

        ORACLE_RC=$?

        git -C "$run" status --porcelain > "$out/git-status.txt"
        git -C "$run" diff > "$out/agent.patch"
        git -C "$run" diff --stat > "$out/diff-stat.txt"

        # Machine-readable per-run metrics.
        python3 - \
            "$out" \
            "$task" \
            "$cond" \
            "$AGENT_RC" \
            "$ORACLE_RC" \
            "$START_TREE" \
            "$PROMPT_HASH" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
task = sys.argv[2]
cond = sys.argv[3]
agent_rc = int(sys.argv[4])
oracle_rc = int(sys.argv[5])
start_tree = sys.argv[6]
prompt_hash = sys.argv[7]

rows = []

p = out / "measured_requests.jsonl"
if p.exists():
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

def isum(key):
    total = 0
    seen = 0
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            total += v
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
pp = out / "measured_proxy.log"
if pp.exists():
    proxy_text = pp.read_text(encoding="utf-8", errors="replace")

opencode_text = ""
op = out / "opencode.txt"
if op.exists():
    opencode_text = op.read_text(encoding="utf-8", errors="replace")

status_text = ""
sp = out / "git-status.txt"
if sp.exists():
    status_text = sp.read_text(encoding="utf-8", errors="replace")

wall = float((out / "wall_seconds.txt").read_text().strip())

metrics = {
    "task": task,
    "condition": cond,
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
    "relevance_split_units": len(
        re.findall(r"strategy_chain=.*relevance_split", proxy_text, flags=re.I)
    ),
    "search_relevance_chains": len(
        re.findall(r"strategy_chain=.*search.*relevance_split", proxy_text, flags=re.I)
    ),
    "lossless_search_mentions": len(
        re.findall(r"lossless_search", proxy_text, flags=re.I)
    ),
    "search_result_like_units": len(
        re.findall(r"text_shape=search_result_like", proxy_text, flags=re.I)
    ),
    "source_code_units": len(
        re.findall(r"content_type=source_code", proxy_text, flags=re.I)
    ),
    "dedicated_grep_mentions": len(
        re.findall(r"(?:✱\s*)?Grep", opencode_text)
    ),
    "shell_rg_mentions": len(
        re.findall(r"\brg\s+", opencode_text)
    ),
    "changed_file_entries": len(
        [x for x in status_text.splitlines() if x.strip()]
    ),
    "start_tree": start_tree,
    "prompt_sha256": prompt_hash,
}

(out / "metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print(json.dumps(metrics, indent=2))
PY

        cp "$out/metrics.json" "$ROOT/results/${task}-${cond}-metrics.json"

        echo
        echo "$task $cond result:"
        cat "$out/metrics.json"

        echo
        echo "Oracle tail:"
        tail -12 "$out/oracle.txt"

        # Dependencies consume most disk space; results/source remain.
        rm -rf "$run/.venv"
    done
done

# --------------------------------------------------
# PHASE 3: aggregate
# --------------------------------------------------

python3 - "$ROOT" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []

for p in root.glob("results/*-metrics.json"):
    try:
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass

order = {"off": 0, "on": 1}

rows.sort(
    key=lambda r: (
        int(r["task"][1:]),
        order.get(r["condition"], 9),
    )
)

fields = [
    "task",
    "condition",
    "task_pass",
    "agent_exit",
    "oracle_exit",
    "wall_seconds",
    "requests",
    "input_tokens_original_sum",
    "input_tokens_optimized_sum",
    "derived_tokens_saved_sum",
    "reported_tokens_saved_sum",
    "relevance_split_units",
    "search_relevance_chains",
    "lossless_search_mentions",
    "search_result_like_units",
    "dedicated_grep_mentions",
    "shell_rg_mentions",
    "changed_file_entries",
    "start_tree",
    "prompt_sha256",
]

with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

md = []
md.append("# Natural coding-agent benchmark")
md.append("")
md.append("| Task | Cond | Pass | Wall s | Requests | Orig input | Optimized | Derived saved | relevance_split | search+bridge | search-shaped | Grep mentions | shell rg |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    md.append(
        "| {task} | {condition} | {task_pass} | {wall:.1f} | {requests} | "
        "{orig} | {opt} | {saved} | {rel} | {chain} | {shape} | {grep} | {rg} |".format(
            task=r["task"],
            condition=r["condition"],
            task_pass="PASS" if r["task_pass"] else "FAIL",
            wall=r["wall_seconds"],
            requests=r["requests"],
            orig=r.get("input_tokens_original_sum"),
            opt=r.get("input_tokens_optimized_sum"),
            saved=r.get("derived_tokens_saved_sum"),
            rel=r.get("relevance_split_units"),
            chain=r.get("search_relevance_chains"),
            shape=r.get("search_result_like_units"),
            grep=r.get("dedicated_grep_mentions"),
            rg=r.get("shell_rg_mentions"),
        )
    )

md.append("")
md.append("## Pairwise activation")
md.append("")

for task in [f"T{i}" for i in range(1, 7)]:
    rr = {r["condition"]: r for r in rows if r["task"] == task}
    if not rr:
        continue

    off = rr.get("off")
    on = rr.get("on")

    if off and on:
        md.append(
            f"- {task}: OFF pass={off['task_pass']}, ON pass={on['task_pass']}; "
            f"ON relevance_split={on['relevance_split_units']}, "
            f"OFF relevance_split={off['relevance_split_units']}."
        )

(root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print()
print("=" * 70)
print((root / "summary.md").read_text(encoding="utf-8"))
print("=" * 70)
print("CSV:", root / "summary.csv")
print("Markdown:", root / "summary.md")
PY

echo
echo "DONE."
echo "Results: $ROOT/results"
echo "Summary: $ROOT/summary.md"
