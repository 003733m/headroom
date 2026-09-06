#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

FREEZE = "89da0898a80c313b6a320f47763391dea7c376e2"

CONTROL = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

RUNTIME = Path(
    "/tmp/headroom-second-unseen-adopted-runtime-89da0898"
).resolve()

RESULT = (
    Path.home()
    / "headroom-second-unseen-adopted-20260905"
    / "results/U01-ON"
)

EXECUTION = (
    CONTROL
    / "experiments/bridge_relevance/bridge_hard/"
      "second_unseen_adopted_bridge_execution.json"
)

OUT = (
    Path.home()
    / "headroom-second-unseen-adopted-20260905"
    / "u01_adopted_floor_counterfactual.json"
)


def die(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr)
    raise SystemExit(1)


import headroom.trajectory_relevance as tr  # noqa: E402
from headroom.transforms.content_router import ContentRouter  # noqa: E402


if RUNTIME not in Path(tr.__file__).resolve().parents:
    die(f"wrong Headroom import: {tr.__file__}")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=CONTROL,
        text=True,
    )


# ------------------------------------------------------------
# 1. Recover frozen U01 task and already-selected ON gate event
# ------------------------------------------------------------

manifest = json.loads(EXECUTION.read_text())

task = next(
    x for x in manifest["tasks"]
    if x["task_id"] == "U01"
)

app_path = RESULT / "applicability.json"
if not app_path.exists():
    die("U01-ON applicability.json missing")

app = json.loads(app_path.read_text())

event = app.get("first_eligible_event")
if not event:
    die("U01-ON has no eligible adopted event")

call_id = event["call_id"]
expected_adopted = tuple(event["adopted_bridges"])

if not expected_adopted:
    die("selected event has no adopted identifiers")


# ------------------------------------------------------------
# 2. Recover exact first-exposure inbound Responses target
# ------------------------------------------------------------

wire_files = sorted(
    (RESULT / "codex-wire").glob(
        "*_http_inbound_request.json"
    )
)

selected_items = None
selected_index = None
target = None
selected_wire = None

for wire_path in wire_files:
    d = json.loads(
        wire_path.read_text(errors="replace")
    )

    body = d.get("body")

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            continue

    if not isinstance(body, dict):
        continue

    items = body.get("input")

    if not isinstance(items, list):
        continue

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
        ):
            value = item.get("output")

            if isinstance(value, str):
                text = value
            else:
                text = tr._responses_text(value)

            selected_items = items
            selected_index = idx
            target = text
            selected_wire = wire_path
            break

    if target is not None:
        break

if (
    target is None
    or selected_items is None
    or selected_index is None
):
    die("could not recover exact selected target")


# ------------------------------------------------------------
# 3. Recompute frozen context and adopted IDs
# ------------------------------------------------------------

context_d = tr.build_responses_search_relevance_context(
    selected_items,
    before_index=selected_index,
    target_content=target,
)

adopted = tr._responses_adopted_bridge_identifiers(
    selected_items,
    before_index=selected_index,
    trajectory_context=context_d,
)

if tuple(adopted) != expected_adopted:
    die(
        "adopted identifier mismatch: "
        f"{adopted} vs {expected_adopted}"
    )

ADOPTED_MARKER = "Adopted bridge identifiers:"

marker_lines = [
    line
    for line in context_d.splitlines()
    if line.strip().startswith(ADOPTED_MARKER)
]

if not marker_lines:
    die(
        "frozen production context does not contain "
        "the adopted preservation marker"
    )

# A keeps identical trajectory relevance but disables only
# the monotonic adopted preservation floor.
context_a = "\n".join(
    line
    for line in context_d.splitlines()
    if not line.strip().startswith(ADOPTED_MARKER)
).strip()


# ------------------------------------------------------------
# 4. Frozen deterministic relevance-split counterfactual
# ------------------------------------------------------------

router_a = ContentRouter()
router_d = ContentRouter()

output_a = router_a._relevance_split_compress(
    target,
    "search",
    context_a,
)

output_d = router_d._relevance_split_compress(
    target,
    "search",
    context_d,
)

# None means the relevance splitter itself abstained.
a_abstained = output_a is None
d_abstained = output_d is None

if output_a is None:
    output_a = target

if output_d is None:
    output_d = target


# ------------------------------------------------------------
# 5. Historical-fix-adjacent oracle
#
# Same interpretation used elsewhere in this study:
# parent-version production lines ±5 around historical fix hunks.
# This is a historical adjacency oracle, NOT causal ground truth.
# ------------------------------------------------------------

source = task["source_commit"]
parent = git("rev-parse", f"{source}^").strip()

critical: set[tuple[str, int, str]] = set()

HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? "
    r"\+(\d+)(?:,(\d+))? @@"
)

for path in task["production_files"]:
    diff = git(
        "diff",
        "--unified=0",
        parent,
        source,
        "--",
        path,
    )

    ranges: list[tuple[int, int]] = []

    for line in diff.splitlines():
        m = HUNK.match(line)

        if not m:
            continue

        old_start = int(m.group(1))
        old_len = int(m.group(2) or "1")

        # Pure insertion still gets a parent-side anchor.
        width = max(old_len, 1)

        lo = max(1, old_start - 5)
        hi = old_start + width - 1 + 5

        ranges.append((lo, hi))

    if not ranges:
        continue

    try:
        parent_text = git(
            "show",
            f"{parent}:{path}",
        )
    except subprocess.CalledProcessError:
        continue

    lines = parent_text.splitlines()

    for lo, hi in ranges:
        for lineno in range(
            lo,
            min(hi, len(lines)) + 1,
        ):
            text = lines[lineno - 1]

            if text.strip():
                critical.add(
                    (
                        path,
                        lineno,
                        text.strip(),
                    )
                )


# ------------------------------------------------------------
# 6. Parse exact rg records and compare retention
# ------------------------------------------------------------

RG_RECORD = re.compile(
    r"^(.+?)([:\-])(\d+)([:\-])(.*)$"
)


def records(text: str):
    result = []

    for raw in text.splitlines():
        m = RG_RECORD.match(raw)

        if not m:
            continue

        if m.group(2) != m.group(4):
            continue

        result.append({
            "raw": raw,
            "path": m.group(1).removeprefix("./"),
            "line": int(m.group(3)),
            "text": m.group(5),
        })

    return result


target_records = records(target)
a_lines = set(output_a.splitlines())
d_lines = set(output_d.splitlines())


def is_critical(rec) -> bool:
    return (
        rec["path"],
        rec["line"],
        rec["text"].strip(),
    ) in critical


scorable = [
    rec
    for rec in target_records
    if is_critical(rec)
]

rows = []

for rec in scorable:
    rows.append({
        "path": rec["path"],
        "line": rec["line"],
        "a_keep": rec["raw"] in a_lines,
        "d_keep": rec["raw"] in d_lines,
    })


rescues = [
    row for row in rows
    if not row["a_keep"] and row["d_keep"]
]

regressions = [
    row for row in rows
    if row["a_keep"] and not row["d_keep"]
]

critical_raw = {
    rec["raw"]
    for rec in scorable
}

new_records = [
    rec
    for rec in target_records
    if (
        rec["raw"] not in a_lines
        and rec["raw"] in d_lines
    )
]

new_noncritical = [
    rec
    for rec in new_records
    if rec["raw"] not in critical_raw
]

new_noncritical_bytes = sum(
    len(rec["raw"].encode("utf-8")) + 1
    for rec in new_noncritical
)


# ------------------------------------------------------------
# 7. Safe summary only — no raw wire/output content
# ------------------------------------------------------------

summary = {
    "experiment": (
        "u01_exact_capture_adopted_floor_counterfactual"
    ),
    "classification": (
        "deterministic isolated preservation-floor counterfactual"
    ),
    "production_freeze": FREEZE,
    "source_commit": source,
    "selected_wire_sha256": hashlib.sha256(
        selected_wire.read_bytes()
    ).hexdigest(),
    "target_sha256": hashlib.sha256(
        target.encode("utf-8", errors="replace")
    ).hexdigest(),
    "command": event["command"],
    "adopted_identifiers": list(adopted),
    "target_bytes": len(
        target.encode("utf-8", errors="replace")
    ),
    "live_run_observation": {
        "handler_search_gate": True,
        "context_nonempty": True,
        "adopted_nonempty": True,
        "runtime_relevance_split_units": 0,
        "runtime_search_relevance_chains": 0,
        "runtime_treatment_activated": False
    },
    "counterfactual": {
        "A": (
            "same trajectory relevance context; "
            "adopted preservation floor disabled"
        ),
        "D": (
            "same trajectory relevance context; "
            "frozen adopted preservation floor enabled"
        ),
        "a_relevance_split_abstained": a_abstained,
        "d_relevance_split_abstained": d_abstained,
        "a_output_bytes": len(
            output_a.encode("utf-8", errors="replace")
        ),
        "d_output_bytes": len(
            output_d.encode("utf-8", errors="replace")
        ),
        "scorable_historical_critical_records": len(
            rows
        ),
        "a_critical_kept": sum(
            row["a_keep"] for row in rows
        ),
        "d_critical_kept": sum(
            row["d_keep"] for row in rows
        ),
        "critical_rescues": len(rescues),
        "critical_regressions": len(regressions),
        "newly_retained_records": len(new_records),
        "newly_retained_noncritical_records": len(
            new_noncritical
        ),
        "newly_retained_noncritical_bytes": (
            new_noncritical_bytes
        ),
        "critical_decisions": rows,
    },
    "claim_boundary": (
        "This replay isolates the preservation floor on an exact "
        "captured unseen search output. The live ON run did not "
        "execute relevance_split, so this is not an observed "
        "end-to-end treatment effect."
    ),
}

OUT.write_text(
    json.dumps(summary, indent=2) + "\n"
)

c = summary["counterfactual"]

print("=" * 78)
print("FINAL U01 DETERMINISTIC COUNTERFACTUAL")
print("=" * 78)
print("freeze:", FREEZE)
print("command:", summary["command"])
print("adopted:", summary["adopted_identifiers"])
print("target bytes:", summary["target_bytes"])
print()
print(
    "live run: relevance_split_units=0, "
    "treatment_activated=False"
)
print()
print("A abstained:", c["a_relevance_split_abstained"])
print("D abstained:", c["d_relevance_split_abstained"])
print("A output bytes:", c["a_output_bytes"])
print("D output bytes:", c["d_output_bytes"])
print()
print(
    "scorable historical-critical:",
    c["scorable_historical_critical_records"],
)
print("A critical kept:", c["a_critical_kept"])
print("D critical kept:", c["d_critical_kept"])
print("critical rescues:", c["critical_rescues"])
print("critical regressions:", c["critical_regressions"])
print()
print(
    "new noncritical records:",
    c["newly_retained_noncritical_records"],
)
print(
    "new noncritical bytes:",
    c["newly_retained_noncritical_bytes"],
)
print()
print("result:", OUT)
