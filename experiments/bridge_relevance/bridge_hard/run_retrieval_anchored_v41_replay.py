#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / "headroom-hard-confirmatory-20260904"
OUT = ROOT / "retrieval_anchored_v41_matched_replay.json"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = load_module(
    "matched_base",
    HERE / "run_hard_matched_context_replay.py",
)

v4 = load_module(
    "v41_probe",
    HERE / "probe_retrieval_anchored_v4.py",
)

tr = base.tr
ORIGINAL = tr.build_responses_search_relevance_context

TASKS = ["G10", "G09", "G07", "G19", "G04", "G21"]


def builder_v41(
    items,
    *,
    before_index,
    user_context="",
    target_content=None,
    scoring=None,
):
    del scoring

    query = base.current_query_context(
        items,
        before_index,
        user_context,
    )

    if not target_content:
        return query

    local_ctx, _anchors, _pattern = v4.anchored_context(
        items,
        before_index=before_index,
        target_content=target_content,
    )

    if query and local_ctx:
        return f"{query}\n{local_ctx}"

    return query or local_ctx


def replay_v41(task, rows, model):
    old = tr.build_responses_search_relevance_context
    tr.build_responses_search_relevance_context = builder_v41

    try:
        handler = base.exact.make_handler(True, model)
        outputs = {}

        for row in rows:
            import copy

            result = handler._compress_openai_responses_payload(
                copy.deepcopy(row["pre"]),
                model=model,
                request_id=f"v41_{task}_{row['request_id']}",
                timing={},
                client="opencode",
                savings_tags={},
            )

            outputs[row["request_id"]] = {
                "payload": result[0],
                "tokens_saved": int(result[2]),
                "output_bytes": int(result[6]),
            }

        return outputs

    finally:
        tr.build_responses_search_relevance_context = old


def main():
    base.exact.freeze_check()

    manifest, tasks = base.load_manifest()
    model = manifest["model"]

    all_units = []
    task_results = {}

    for task in TASKS:
        print()
        print("=" * 72)
        print(task)
        print("=" * 72)

        rows = base.load_wire(task)
        units = base.first_search_units(rows)
        oracle = base.historical_oracle(tasks[task])

        # B=query only, C=query+global V3
        b = base.replay(task, rows, "B", model)
        c = base.replay(task, rows, "C", model)
        e = replay_v41(task, rows, model)

        by_req = {r["request_id"]: r for r in rows}

        analyzed = []

        for unit in units:
            req = unit["request_id"]
            cid = unit["call_id"]

            captured = base.exact.output_by_call_id(
                by_req[req]["post"],
                cid,
            )

            # Reconstruct A solely for replay validity.
            a = base.replay(task, [by_req[req]], "A", model)
            a_unit = base.exact.output_by_call_id(
                a[req]["payload"],
                cid,
            )

            b_unit = base.exact.output_by_call_id(
                b[req]["payload"],
                cid,
            )
            c_unit = base.exact.output_by_call_id(
                c[req]["payload"],
                cid,
            )
            e_unit = base.exact.output_by_call_id(
                e[req]["payload"],
                cid,
            )

            records = base.parse_records(unit["target"])

            captured_vec = []
            a_vec = []
            record_rows = []

            for rec in records:
                fix = base.is_fix_adjacent(rec, oracle)

                captured_keep = base.retained(captured, rec)
                a_keep = base.retained(a_unit, rec)

                kept = {
                    "B": base.retained(b_unit, rec),
                    "C": base.retained(c_unit, rec),
                    "E": base.retained(e_unit, rec),
                }

                captured_vec.append(captured_keep)
                a_vec.append(a_keep)

                record_rows.append({
                    **rec,
                    "fix_adjacent": fix,
                    "captured_keep": captured_keep,
                    "A_keep": a_keep,
                    "kept": kept,
                })

            valid = (
                bool(records)
                and captured is not None
                and a_unit is not None
                and captured_vec == a_vec
            )

            analyzed.append({
                **unit,
                "replay_valid": valid,
                "records": record_rows,
                "outputs": {
                    "B": b_unit,
                    "C": c_unit,
                    "E": e_unit,
                }
            })

        tokenizer = base.exact.ProductionLikeTokenCounter(model)

        valid = [u for u in analyzed if u["replay_valid"]]

        summary = {
            "units": len(analyzed),
            "valid": len(valid),
            "critical_records": 0,
        }

        for mode in ("B", "C", "E"):
            summary[mode] = {
                "critical_kept": 0,
                "nonfix_kept_tokens": 0,
                "full_output_tokens": 0,
            }

        bc = {"rescue": 0, "regress": 0}
        be = {"rescue": 0, "regress": 0}
        ce = {"rescue": 0, "regress": 0}

        for unit in valid:
            for mode in ("B", "C", "E"):
                out = unit["outputs"][mode]
                if out:
                    summary[mode]["full_output_tokens"] += (
                        tokenizer.count_text(out)
                    )

            for rec in unit["records"]:
                if rec["fix_adjacent"]:
                    summary["critical_records"] += 1

                rtok = tokenizer.count_text(rec["raw"])

                for mode in ("B", "C", "E"):
                    if not rec["kept"][mode]:
                        continue

                    if rec["fix_adjacent"]:
                        summary[mode]["critical_kept"] += 1
                    else:
                        summary[mode]["nonfix_kept_tokens"] += rtok

                if rec["fix_adjacent"]:
                    pairs = [
                        ("B", "C", bc),
                        ("B", "E", be),
                        ("C", "E", ce),
                    ]

                    for left, right, counter in pairs:
                        if (
                            not rec["kept"][left]
                            and rec["kept"][right]
                        ):
                            counter["rescue"] += 1

                        if (
                            rec["kept"][left]
                            and not rec["kept"][right]
                        ):
                            counter["regress"] += 1

        denom = summary["critical_records"]

        for mode in ("B", "C", "E"):
            summary[mode]["critical_recall"] = (
                summary[mode]["critical_kept"] / denom
                if denom
                else None
            )

            n = summary[mode]["nonfix_kept_tokens"]

            summary[mode]["evidence_efficiency"] = (
                summary[mode]["critical_kept"] / (n / 1000)
                if n
                else None
            )

        summary["B_to_C"] = bc
        summary["B_to_E"] = be
        summary["C_to_E"] = ce

        task_results[task] = {
            "summary": summary,
            "units": analyzed,
        }

        all_units.extend(analyzed)

        print(
            f"units={summary['units']} valid={summary['valid']} "
            f"critical={summary['critical_records']}"
        )

        for mode in ("B", "C", "E"):
            x = summary[mode]
            print(
                f"  {mode}: critical={x['critical_kept']} "
                f"recall={x['critical_recall']} "
                f"nonfix_tokens={x['nonfix_kept_tokens']} "
                f"output_tokens={x['full_output_tokens']}"
            )

        print(
            f"  B→E rescue={be['rescue']} regress={be['regress']}"
        )
        print(
            f"  C→E rescue={ce['rescue']} regress={ce['regress']}"
        )

    # Aggregate directly over already-analyzed valid units.
    valid = [u for u in all_units if u["replay_valid"]]
    tok = base.exact.ProductionLikeTokenCounter(model)

    agg = {
        "units": len(all_units),
        "valid": len(valid),
        "critical_records": 0,
    }

    for mode in ("B", "C", "E"):
        agg[mode] = {
            "critical_kept": 0,
            "nonfix_kept_tokens": 0,
            "full_output_tokens": 0,
        }

    contrasts = {
        "B_to_C": {"rescue": 0, "regress": 0},
        "B_to_E": {"rescue": 0, "regress": 0},
        "C_to_E": {"rescue": 0, "regress": 0},
    }

    for unit in valid:
        for mode in ("B", "C", "E"):
            out = unit["outputs"][mode]
            if out:
                agg[mode]["full_output_tokens"] += tok.count_text(out)

        for rec in unit["records"]:
            if rec["fix_adjacent"]:
                agg["critical_records"] += 1

            rtok = tok.count_text(rec["raw"])

            for mode in ("B", "C", "E"):
                if rec["kept"][mode]:
                    if rec["fix_adjacent"]:
                        agg[mode]["critical_kept"] += 1
                    else:
                        agg[mode]["nonfix_kept_tokens"] += rtok

            if rec["fix_adjacent"]:
                for name, left, right in (
                    ("B_to_C", "B", "C"),
                    ("B_to_E", "B", "E"),
                    ("C_to_E", "C", "E"),
                ):
                    if (
                        not rec["kept"][left]
                        and rec["kept"][right]
                    ):
                        contrasts[name]["rescue"] += 1

                    if (
                        rec["kept"][left]
                        and not rec["kept"][right]
                    ):
                        contrasts[name]["regress"] += 1

    for mode in ("B", "C", "E"):
        denom = agg["critical_records"]
        agg[mode]["critical_recall"] = (
            agg[mode]["critical_kept"] / denom
            if denom
            else None
        )

        n = agg[mode]["nonfix_kept_tokens"]
        agg[mode]["evidence_efficiency"] = (
            agg[mode]["critical_kept"] / (n / 1000)
            if n
            else None
        )

    agg.update(contrasts)

    result = {
        "evaluation_type": "post_hoc_exploratory_matched_replay",
        "base_algorithm_freeze": base.FREEZE,
        "tasks": task_results,
        "aggregate": agg,
    }

    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    print()
    print("=" * 72)
    print("AGGREGATE")
    print("=" * 72)

    print(
        f"units={agg['units']} valid={agg['valid']} "
        f"critical={agg['critical_records']}"
    )

    for mode, label in (
        ("B", "query-only"),
        ("C", "global-V3"),
        ("E", "anchored-V4.1"),
    ):
        x = agg[mode]

        print(
            f"{label}: "
            f"critical={x['critical_kept']} "
            f"recall={x['critical_recall']} "
            f"nonfix_tokens={x['nonfix_kept_tokens']} "
            f"output_tokens={x['full_output_tokens']} "
            f"eff={x['evidence_efficiency']}"
        )

    print("B→C:", agg["B_to_C"])
    print("B→E:", agg["B_to_E"])
    print("C→E:", agg["C_to_E"])
    print("JSON:", OUT)


if __name__ == "__main__":
    main()
