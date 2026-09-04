#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTROL = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

BASE_PATH = HERE / "run_natural_transfer.py"
PROTOCOL_PATH = HERE / "exact_wire_validation_protocol.json"

ROOT = Path("/tmp/headroom-v3-exact-wire-validation")

OFF_ORDER = [
    "F1-OFF",
    "F2-OFF",
    "F3-OFF",
    "F4-OFF",
    "F5-OFF",
    "F6-OFF",
    "F7-OFF",
    "F8-OFF",
]

PRE_EVENTS = {
    "ws_inbound_first_frame",
    "ws_inbound_client_frame",
    "http_inbound_request",
}

POST_EVENTS = {
    "ws_upstream_first_frame",
    "ws_upstream_client_frame",
    "http_upstream_request",
    "http_stream_upstream_request",
}


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location(
        "natural_transfer_base",
        BASE_PATH,
    )
    if spec is None or spec.loader is None:
        die("cannot import natural-transfer base harness")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base()

# Reuse the already-frozen runtime and locked snapshots,
# but NEVER reuse the previous experiment result root.
base.ROOT = ROOT
base.PROXY = None


def load_protocol() -> dict[str, Any]:
    d = json.loads(PROTOCOL_PATH.read_text())

    if d["status"] != (
        "locked_before_any_exact_wire_validation_agent_run"
    ):
        die("protocol status mismatch")

    if d["algorithm_freeze"] != base.FREEZE:
        die("protocol freeze mismatch")

    if d["model"] != base.MODEL:
        die("protocol model mismatch")

    if d["opencode_version"] != base.OPENCODE_VERSION:
        die("protocol OpenCode version mismatch")

    if d["run_order"] != OFF_ORDER:
        die("protocol run order mismatch")

    return d


def assert_committed_clean() -> None:
    for p in (PROTOCOL_PATH, Path(__file__).resolve()):
        rel = p.relative_to(CONTROL)

        cp = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{rel}"],
            cwd=CONTROL,
        )
        if cp.returncode:
            die(f"{rel} is not committed")

        cp = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(rel)],
            cwd=CONTROL,
        )
        if cp.returncode:
            die(f"{rel} differs from committed HEAD")


def verify_frozen_runtime() -> None:
    if not base.RUNTIME.exists():
        die(f"frozen runtime missing: {base.RUNTIME}")

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=base.RUNTIME,
        text=True,
    ).strip()

    if sha != base.FREEZE:
        die(
            "frozen runtime SHA mismatch: "
            f"{sha} != {base.FREEZE}"
        )

    print(f"Frozen runtime: OK ({sha})")


def prepare_root() -> None:
    for p in (
        ROOT,
        ROOT / "results",
        ROOT / "runs",
        ROOT / "prime",
    ):
        p.mkdir(parents=True, exist_ok=True)


def restore_env(name: str, old: str | None) -> None:
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


# ------------------------------------------------------------------
# Wire-debug hooks
# ------------------------------------------------------------------

_original_start_proxy = base.start_proxy
_original_warm_proxy = base.warm_proxy


def start_proxy_with_wire(port, out, flag):
    wire = out / "wire"
    wire.mkdir(parents=True, exist_ok=True)

    old_enabled = os.environ.get(
        "HEADROOM_CODEX_WIRE_DEBUG"
    )
    old_dir = os.environ.get(
        "HEADROOM_CODEX_WIRE_DEBUG_DIR"
    )

    os.environ["HEADROOM_CODEX_WIRE_DEBUG"] = "1"
    os.environ["HEADROOM_CODEX_WIRE_DEBUG_DIR"] = str(
        wire.resolve()
    )

    try:
        # start_proxy copies os.environ before Popen, so the child
        # receives the condition-specific wire-debug configuration.
        return _original_start_proxy(
            port,
            out,
            flag,
        )
    finally:
        restore_env(
            "HEADROOM_CODEX_WIRE_DEBUG",
            old_enabled,
        )
        restore_env(
            "HEADROOM_CODEX_WIRE_DEBUG_DIR",
            old_dir,
        )


def warm_proxy_then_clear_wire(port, out, flag):
    result = _original_warm_proxy(
        port,
        out,
        flag,
    )

    wire = out / "wire"
    wire.mkdir(parents=True, exist_ok=True)

    warm_files = sorted(wire.glob("*.json"))

    manifest = {
        "purpose": (
            "Wire artifacts generated by Kompress/OpenCode warmup "
            "before the benchmark agent started. These files were "
            "hashed and then removed so measured wire captures contain "
            "only the benchmark-agent phase."
        ),
        "count": len(warm_files),
        "files": [
            {
                "name": p.name,
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            for p in warm_files
        ],
    }

    (
        out / "wire_warmup_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    for p in warm_files:
        p.unlink()

    return result


base.start_proxy = start_proxy_with_wire
base.warm_proxy = warm_proxy_then_clear_wire


def read_wire(out: Path) -> list[dict[str, Any]]:
    wire = out / "wire"

    if not wire.exists():
        return []

    records: list[dict[str, Any]] = []

    for p in sorted(wire.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            die(
                f"invalid wire JSON {p}: {exc}"
            )

        d["_file"] = p.name
        d["_sha256"] = sha256_file(p)
        records.append(d)

    return records


def validate_wire_capture(
    out: Path,
) -> dict[str, Any]:
    records = read_wire(out)

    pre = [
        r
        for r in records
        if r.get("event") in PRE_EVENTS
        and (
            r.get("body") is not None
            or r.get("raw_text") is not None
        )
    ]

    post = [
        r
        for r in records
        if r.get("event") in POST_EVENTS
        and (
            r.get("body") is not None
            or r.get("raw_text") is not None
        )
    ]

    event_counts: dict[str, int] = {}
    for r in records:
        event = str(r.get("event"))
        event_counts[event] = (
            event_counts.get(event, 0) + 1
        )

    transports = sorted(
        {
            str(r.get("transport"))
            for r in records
            if r.get("transport")
        }
    )

    request_ids = sorted(
        {
            str(r.get("request_id"))
            for r in records
            if r.get("request_id")
        }
    )

    session_ids = sorted(
        {
            str(r.get("session_id"))
            for r in records
            if r.get("session_id")
        }
    )

    valid = (
        len(records) > 0
        and len(pre) > 0
        and len(post) > 0
    )

    summary = {
        "valid": valid,
        "wire_file_count": len(records),
        "pre_body_capture_count": len(pre),
        "post_body_capture_count": len(post),
        "event_counts": event_counts,
        "transports": transports,
        "request_id_count": len(request_ids),
        "session_id_count": len(session_ids),
        "files": [
            {
                "name": r["_file"],
                "sha256": r["_sha256"],
                "event": r.get("event"),
                "request_id": r.get("request_id"),
                "session_id": r.get("session_id"),
                "transport": r.get("transport"),
                "direction": r.get("direction"),
            }
            for r in records
        ],
    }

    (
        out / "wire_capture_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    if not valid:
        (
            out / "WIRE_CAPTURE_INVALID"
        ).write_text(
            "Expected at least one content-bearing "
            "pre-Headroom and post-Headroom wire capture.\n"
        )
        die(
            "wire capture validity check failed; "
            "do not rerun automatically"
        )

    print(
        "Wire capture: OK "
        f"(files={len(records)} "
        f"pre={len(pre)} post={len(post)})"
    )

    return summary


def task_by_id(
    manifest: dict[str, Any],
    tid: str,
):
    return base.task_by_id(
        manifest,
        tid,
    )


def preflight() -> None:
    load_protocol()
    assert_committed_clean()
    prepare_root()

    # Reuse all mature validity checks from the original harness.
    base.preflight()

    print()
    print("=" * 58)
    print("EXACT-WIRE VALIDATION PREFLIGHT PASSED")
    print("No exact-wire benchmark agent was run.")
    print(f"root  : {ROOT}")
    print(f"model : {base.MODEL}")
    print(f"freeze: {base.FREEZE}")
    print(f"next  : {OFF_ORDER[0]}")
    print("=" * 58)


def run_next() -> None:
    protocol = load_protocol()
    assert_committed_clean()
    prepare_root()

    manifest = base.locked_manifest()

    verify_frozen_runtime()
    base.snapshot_lock()
    base.verify_snapshots()
    base.opencode_version()

    for item in protocol["run_order"]:
        tid, cond = item.split("-", 1)

        if cond != "OFF":
            die(
                f"non-OFF condition in wire protocol: {item}"
            )

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
        print(
            "RUNNING NEXT LOCKED EXACT-WIRE CONDITION: "
            f"{item}"
        )
        print("=" * 70)

        base.run_condition(
            task,
            cond,
        )

        validate_wire_capture(out)
        return

    print(
        "All 8 locked exact-wire OFF conditions "
        "already have metrics."
    )


def show_status() -> None:
    load_protocol()
    prepare_root()

    print(
        "===== EXACT-WIRE VALIDATION STATUS ====="
    )

    for item in OFF_ORDER:
        out = ROOT / "results" / item

        if (out / "metrics.json").exists():
            d = json.loads(
                (out / "metrics.json").read_text()
            )

            wire_summary = (
                json.loads(
                    (
                        out
                        / "wire_capture_summary.json"
                    ).read_text()
                )
                if (
                    out
                    / "wire_capture_summary.json"
                ).exists()
                else None
            )

            if wire_summary:
                wire_txt = (
                    f"wire={wire_summary['wire_file_count']} "
                    f"pre={wire_summary['pre_body_capture_count']} "
                    f"post={wire_summary['post_body_capture_count']} "
                    f"wvalid={wire_summary['valid']}"
                )
            else:
                wire_txt = "wire=UNVALIDATED"

            print(
                f"{item:7} COMPLETE "
                f"pass={d['task_pass']} "
                f"rg={d['rg_capture_count']} "
                f"req={d['requests']} "
                f"{wire_txt}"
            )

        elif (out / "AGENT_STARTED").exists():
            print(
                f"{item:7} STARTED-INCOMPLETE"
            )
        else:
            print(
                f"{item:7} PENDING"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--preflight",
        action="store_true",
    )
    group.add_argument(
        "--run-next",
        action="store_true",
    )
    group.add_argument(
        "--status",
        action="store_true",
    )

    args = parser.parse_args()

    if args.preflight:
        preflight()
    elif args.run_next:
        run_next()
    else:
        show_status()


if __name__ == "__main__":
    main()
