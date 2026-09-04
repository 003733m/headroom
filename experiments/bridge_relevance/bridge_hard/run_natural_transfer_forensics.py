#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TASK_COMMITS = {
    "F1": "1f5fefffd3e82c73bddd928cfd53334031e807bc",
    "F2": "85e869945138f06471501046c5725eac119dea58",
    "F3": "4e2bbfee3f65e3287ab7693da70f0d7b20dada28",
    "F4": "0cddac632d8ab4e0bcd752b9c7519d5099aa085e",
    "F5": "cc072f0821b76618659a2dd6c81a1daee49dbaab",
    "F6": "fa330f3e2bed3f51b7f4d49bcd64ec4aaa17f44f",
    "F7": "c7b5a24b4ffca78bacf1919ace00087b6ab6f7d0",
    "F8": "d05802b6200b94f198e99319e6c778e78b53db8b",
}


HERE = Path(__file__).resolve().parent

EXHAUSTIVE = (
    HERE / "exhaustive_discovery_results.json"
)


def find_source_commit(
    obj: Any,
    sha: str,
) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if (
            obj.get("source_commit") == sha
            and obj.get("status") == "evaluated"
            and "critical_decisions" in obj
        ):
            return obj

        for value in obj.values():
            found = find_source_commit(value, sha)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = find_source_commit(value, sha)
            if found is not None:
                return found

    return None


def read_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    for line in path.read_text(
        errors="replace"
    ).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value

    return out


def has_shell_pipe_or_redirect(command: str) -> bool:
    """
    Detect |, >, < outside shell quotes.

    A regex alternation such as "foo|bar" inside quotes does not
    count as a shell pipeline.
    """
    quote: str | None = None
    escaped = False

    for ch in command:
        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if quote:
            if ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            continue

        if ch in ("|", ">", "<"):
            return True

    return False


def rg_segments(
    transcript: str,
) -> list[dict[str, Any]]:
    lines = transcript.splitlines()

    starts = [
        i
        for i, line in enumerate(lines)
        if re.match(r"^\$\s+.*\brg\s+", line)
    ]

    segments: list[dict[str, Any]] = []

    for pos, start in enumerate(starts):
        end = len(lines)

        # Stop at the next explicit tool invocation.
        for j in range(start + 1, len(lines)):
            if re.match(r"^(?:\$\s+|→\s+)", lines[j]):
                end = j
                break

        command = lines[start]

        segments.append(
            {
                "index": pos + 1,
                "command": command,
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(
                    lines[start + 1:end]
                ),
                "piped_or_redirected":
                    has_shell_pipe_or_redirect(command),
            }
        )

    return segments


def record_variants(
    path: str,
    line: int,
    text: str,
) -> tuple[str, str]:
    bare = f"{path}:{line}:{text}"
    dotted = f"./{path}:{line}:{text}"
    return bare, dotted


def contains_variant(
    haystack: str,
    variants: tuple[str, str],
) -> bool:
    return any(v in haystack for v in variants)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home()
        / "headroom-natural-transfer-raw-20260904",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=HERE
        / "natural_transfer_forensics_results.json",
    )

    args = parser.parse_args()

    root = args.root.resolve()

    if not root.exists():
        raise SystemExit(
            f"raw archive missing: {root}"
        )

    exhaustive = json.loads(
        EXHAUSTIVE.read_text()
    )

    output: dict[str, Any] = {
        "analysis_type":
            "post_hoc_exploratory_natural_transfer_forensics",
        "exact_natural_v3_replay": False,
        "exact_natural_v3_replay_reason":
            "full request message payloads were not persisted",
        "root": str(root),
        "tasks": {},
    }

    all_candidates: list[dict[str, Any]] = []

    for task, sha in TASK_COMMITS.items():
        result_dir = (
            root / "results" / f"{task}-OFF"
        )

        if not result_dir.exists():
            raise SystemExit(
                f"missing OFF result: {result_dir}"
            )

        historical = find_source_commit(
            exhaustive,
            sha,
        )

        if historical is None:
            raise SystemExit(
                f"missing exhaustive result for {task} {sha}"
            )

        rescue_records = [
            r
            for r in historical[
                "critical_decisions"
            ]
            if (
                r.get("baseline_keep") is False
                and r.get("v3_keep") is True
            )
        ]

        transcript_path = (
            result_dir / "opencode.txt"
        )

        transcript = transcript_path.read_text(
            errors="replace"
        )

        segments = rg_segments(transcript)

        capdir = result_dir / "rg-captures"
        metas = sorted(capdir.glob("*.meta"))

        paired = len(metas) == len(segments)

        captures: list[dict[str, Any]] = []

        for idx, meta_path in enumerate(
            metas,
            start=1,
        ):
            meta = read_meta(meta_path)
            stem = meta_path.stem

            stdout_path = (
                capdir / f"{stem}.stdout"
            )

            raw = (
                stdout_path.read_text(
                    errors="replace"
                )
                if stdout_path.exists()
                else ""
            )

            segment = (
                segments[idx - 1]
                if idx <= len(segments)
                else None
            )

            captures.append(
                {
                    "index": idx,
                    "meta": meta,
                    "raw": raw,
                    "segment": segment,
                }
            )

        task_candidates: list[
            dict[str, Any]
        ] = []

        retrieved_record_keys = set()
        visible_record_keys = set()
        hidden_record_keys = set()

        for record in rescue_records:
            path = str(record["path"])
            lineno = int(record["line"])
            text = str(record["text"])

            variants = record_variants(
                path,
                lineno,
                text,
            )

            key = (
                path,
                lineno,
                text,
            )

            for cap in captures:
                if not contains_variant(
                    cap["raw"],
                    variants,
                ):
                    continue

                retrieved_record_keys.add(key)

                segment = cap["segment"]

                if segment is None:
                    task_candidates.append(
                        {
                            "task": task,
                            "capture_index":
                                cap["index"],
                            "path": path,
                            "line": lineno,
                            "text": text,
                            "raw_retrieved": True,
                            "agent_visible":
                                None,
                            "paired": False,
                            "piped_or_redirected":
                                None,
                            "clean_candidate":
                                False,
                            "reason":
                                "capture has no paired transcript rg segment",
                        }
                    )
                    continue

                visible = contains_variant(
                    segment["text"],
                    variants,
                )

                if visible:
                    visible_record_keys.add(key)
                else:
                    hidden_record_keys.add(key)

                clean = (
                    paired
                    and not visible
                    and not segment[
                        "piped_or_redirected"
                    ]
                )

                candidate = {
                    "task": task,
                    "source_commit": sha,
                    "capture_index":
                        cap["index"],
                    "path": path,
                    "line": lineno,
                    "text": text,
                    "raw_retrieved": True,
                    "agent_visible":
                        visible,
                    "paired": paired,
                    "piped_or_redirected":
                        segment[
                            "piped_or_redirected"
                        ],
                    "command":
                        segment["command"],
                    "transcript_start_line":
                        segment["start_line"],
                    "transcript_end_line":
                        segment["end_line"],
                    "controlled_exhaustive":
                        "baseline_DROP_to_V3_KEEP",
                    "clean_candidate":
                        clean,
                }

                task_candidates.append(
                    candidate
                )

                if clean:
                    all_candidates.append(
                        candidate
                    )

        metrics_path = (
            result_dir / "metrics.json"
        )

        metrics = (
            json.loads(metrics_path.read_text())
            if metrics_path.exists()
            else {}
        )

        output["tasks"][task] = {
            "source_commit": sha,
            "off_task_pass":
                metrics.get("task_pass"),
            "rg_capture_count":
                len(metas),
            "transcript_rg_count":
                len(segments),
            "capture_transcript_pairing_exact":
                paired,
            "controlled_rescue_record_count":
                len(rescue_records),
            "natural_retrieved_controlled_rescue_records":
                len(retrieved_record_keys),
            "natural_visible_controlled_rescue_records":
                len(visible_record_keys),
            "natural_hidden_controlled_rescue_records":
                len(hidden_record_keys),
            "clean_candidate_occurrences":
                sum(
                    1
                    for x in task_candidates
                    if x["clean_candidate"]
                ),
            "occurrences":
                task_candidates,
        }

    output["summary"] = {
        "tasks": len(TASK_COMMITS),
        "tasks_with_any_clean_candidate":
            sum(
                1
                for x in output["tasks"].values()
                if x[
                    "clean_candidate_occurrences"
                ] > 0
            ),
        "clean_candidate_occurrences":
            len(all_candidates),
        "clean_candidates":
            all_candidates,
        "claim_boundary":
            (
                "A clean candidate proves natural retrieval and "
                "baseline transcript non-visibility for an exact "
                "record independently observed as DROP->KEEP in "
                "the controlled exhaustive experiment. It does "
                "not prove exact natural-trajectory V3 retention."
            ),
    }

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print(
        "=========================================="
    )
    print("NATURAL TRANSFER FORENSICS")
    print(
        "=========================================="
    )

    for task, result in output[
        "tasks"
    ].items():
        print(
            f"{task}: "
            f"controlled_DK="
            f"{result['controlled_rescue_record_count']} "
            f"natural_retrieved="
            f"{result['natural_retrieved_controlled_rescue_records']} "
            f"visible="
            f"{result['natural_visible_controlled_rescue_records']} "
            f"hidden="
            f"{result['natural_hidden_controlled_rescue_records']} "
            f"clean_occ="
            f"{result['clean_candidate_occurrences']} "
            f"pairing="
            f"{result['capture_transcript_pairing_exact']}"
        )

    print()
    print(
        "tasks with clean candidate:",
        output["summary"][
            "tasks_with_any_clean_candidate"
        ],
    )
    print(
        "clean candidate occurrences:",
        output["summary"][
            "clean_candidate_occurrences"
        ],
    )
    print()
    print("output:", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
