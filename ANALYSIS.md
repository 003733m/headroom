# Trajectory-Aware Relevance for Coding-Agent Context Compression

## Summary

Headroom already conditions tool-output compression on the user request and the current tool call. I investigated a narrower failure mode: a coding agent's information state evolves during a trajectory. It can learn task-local identifiers from earlier tool outputs and rely on them several steps later even when they are absent from the original request or only weakly represented by the current retrieval query.

I implemented a bounded, target-conditioned cross-tool trajectory signal and integrated it into the OpenAI Responses/search compression path. Controlled evaluation showed that this signal can recover evidence that baseline relevance drops: exact evidence fidelity increased from **6/20 (30%) to 20/20 (100%)**, at a cost of **6.34 percentage points less compression**. Frozen prospective agent runs did **not** show an end-to-end task-success improvement: all OFF and ON conditions passed. Subsequent replays showed why the problem is harder than simply adding more context: trajectory state can preserve useful evidence, but it can also retain noise, and retrieval/routing can prevent a useful preservation rule from executing at all.

This led to a conservative production refinement: an **adopted-bridge preservation floor**. A search record is protected only when an identifier is both supported by prior trajectory state and explicitly reused by the agent in its current `rg`/`grep` query. In a post-hoc strict matched replay, this raised historical-critical retention from **5/9 to 6/9** while adding **504 non-critical tokens**. A naive policy that protected all structured current-query identifiers achieved the same 6/9 retention but added **3,249 non-critical tokens**. The adopted rule therefore obtained the same observed rescue with **84.5% less added non-critical burden** than the naive query-hit floor.

I do not claim a general coding-agent success improvement. The contribution is a characterized state-loss failure mode, a production implementation, a selective monotonic preservation invariant, and an evaluation that separates controlled mechanism effects, naturalistic transfer, post-hoc refinement, and negative unseen validation.

## 1. Gap: relevance is trajectory-dependent

The baseline already has two useful signals:

1. the global user/task request; and
2. the current tool-call arguments, including the current search query.

The missing signal is **task-local state learned on the way to the current tool call**.

An agent may discover a test name, exception, request ID, configuration key, class, or function in one tool output and later use that identifier to navigate the repository. Such identifiers behave like join keys between trajectory steps. Re-sending the complete history would defeat compression, so the goal is to retain only a bounded summary of the strongest learned identifiers.

The main implementation is in [`headroom/trajectory_relevance.py`](headroom/trajectory_relevance.py), with routing integration in [`headroom/transforms/content_router.py`](headroom/transforms/content_router.py) and the final preservation floor in [`headroom/transforms/relevance_split.py`](headroom/transforms/relevance_split.py).

The extractor considers only bounded prior tool outputs. It recognizes structured candidate kinds such as file paths, request IDs, test names, exceptions, configuration keys, function names, and class names. Candidates are ranked using causal/positive evidence, cross-output support, occurrence, recency, and specificity. The current target is never allowed to self-source a bridge candidate.

For OpenAI Responses, producing search calls and outputs are correlated by `call_id`. Search outputs can receive trajectory context only when they pass the production search-reachability path. This matters because a source-code-looking `rg` result may not be classified as SEARCH by the primary classifier but can still be recognized through the structural detector or the linked Bash-search call.

## 2. Final production rule: adopted bridges

Global trajectory relevance was useful but not sufficiently selective on harder natural replays. I therefore added a monotonic preservation rule.

Let:

- `B` = identifiers admitted by the frozen trajectory bridge builder; and
- `Q` = conservative structured identifiers explicitly reused in the current `rg`/`grep` pattern.

The adopted set is:

`A = B ∩ Q`.

If ordinary relevance scoring would DROP a record containing an exact-boundary occurrence of an identifier in `A`, that record is forced to KEEP. Existing KEEP decisions are unchanged.

This is intentionally not a new relevance model. It is a narrow preservation invariant: **prior trajectory support must be followed by explicit current behavioral adoption**.

Production freeze: `89da0898a80c313b6a320f47763391dea7c376e2`.

Focused tests are in [`tests/test_adopted_bridge_preservation.py`](tests/test_adopted_bridge_preservation.py).

## 3. Evidence hierarchy

I separate results by when the hypothesis/policy was fixed.

| Status | Evaluation | Result | What it supports |
|---|---|---:|---|
| Controlled development | Multi-target evidence fidelity | 6/20 → 20/20 | Clean mechanism capability |
| Prospective / frozen | N5–N8 paired agent runs | OFF 4/4, ON 4/4; ON activated 3/4 | Natural reachability, no task-success lift |
| Confirmatory harder tasks | Six paired hard tasks | OFF 6/6, ON 6/6 | Ceiling effect; no end-to-end lift |
| Secondary natural replay | Exact captured requests | 1/18 baseline drops rescued, 0 regression | Small natural retention transfer |
| Exploratory historical replay | Reverse-patch scan | 60/1,326 drops rescued, 3 regressions | Conditional larger-sample mechanism evidence |
| Post-hoc refinement | Adopted-bridge floor | critical 5/9 → 6/9; +504 non-critical tokens | Selective preservation on development replay |
| Post-hoc comparator | Naive query-hit floor | critical 5/9 → 6/9; +3,249 non-critical tokens | Same rescue, much broader retention |
| Unseen validation | G11 + second unseen U01 | final preservation treatment did not activate | No unseen causal retention estimate |

### Controlled fidelity

In a 20-case controlled probe, the bridge signal increased exact evidence retention from 30% to 100%. Average compression reduction changed from 25.46% to 19.13%, a 6.34 percentage-point cost. This is the cleanest positive result: trajectory state can preserve evidence that the baseline relevance signal loses.

The controlled result is recorded in [`evidence_fidelity_multitarget_summary_pre_freeze.json`](experiments/bridge_relevance/results/v3/evidence_fidelity_multitarget_summary_pre_freeze.json) with per-case data in [`evidence_fidelity_multitarget_pre_freeze.csv`](experiments/bridge_relevance/results/v3/evidence_fidelity_multitarget_pre_freeze.csv).

### Frozen prospective agent runs

For N5–N8, the mechanism and task manifest were frozen before execution, completed conditions were never rerun, and tasks were not replaced after observing outcomes. The treatment activated in 3/4 ON tasks. All four tasks passed in both OFF and ON conditions.

Therefore these runs show that trajectory relevance can reach real agent trajectories, but they do **not** show an end-to-end success improvement. Aggregate token savings differed descriptively, but I do not interpret that difference causally because enabling compression can change the agent's subsequent retrieval behavior.

The locked manifest and runner are [`prospective_task_manifest.json`](experiments/bridge_relevance/results/v3/prospective_task_manifest.json) and [`run_v3_prospective.sh`](experiments/bridge_relevance/agent_benchmarks/v3/run_v3_prospective.sh).

## 4. Naturalistic and historical replay

To create more retention opportunities without repeatedly sampling an agent, I mined historical bug-fix commits under a fixed validation protocol. Production patches were reversed, pristine tests were checked, and only snapshots reproducing the expected historical failures were retained. Parent-version lines around the historical source fix were used as a **historical-fix-adjacent oracle**.

This oracle is useful but limited: adjacency to a historical fix is not proof that a record was causally necessary for a new agent.

Across 92 tasks where baseline DROP created an opportunity, trajectory relevance rescued 10 tasks. At record level, it rescued **60/1,326 (4.52%)** baseline-dropped fix-adjacent records, with 3 regressions. This result is exploratory and conditional on retrieval and the historical oracle.

I also replayed exact inbound requests captured from natural agent trajectories. At first exposure, baseline retained 1/19 scorable historical-critical records. A frozen production-wrapper OFF replay reproduced all 19 baseline KEEP/DROP decisions. V3 retained 2/19, rescuing **1/18 baseline drops with no regression**. However, the three predeclared controlled D→K candidates were not rescued (0/3), limiting a stronger transfer claim.

The historical scan is implemented by [`run_exhaustive_discovery.py`](experiments/bridge_relevance/bridge_hard/run_exhaustive_discovery.py) with results in [`exhaustive_discovery_results.json`](experiments/bridge_relevance/bridge_hard/exhaustive_discovery_results.json). Exact-wire replay is implemented by [`run_exact_wire_v3_replay.py`](experiments/bridge_relevance/bridge_hard/run_exact_wire_v3_replay.py) with results in [`exact_wire_v3_replay_results.json`](experiments/bridge_relevance/bridge_hard/exact_wire_v3_replay_results.json). Corrected matched-retention measurement is implemented by [`measure_normalized_retention.py`](experiments/bridge_relevance/bridge_hard/measure_normalized_retention.py).

## 5. Selectivity: why the adopted floor was added

On the strict matched hard replay, query-only baseline and global V3 trajectory context both retained **5/9** scorable historical-critical records. V3 retained more non-critical material without improving critical recall. A retrieval-anchored variant reduced much of that extra material but still did not increase critical recall.

I then tested a naive query-hit floor: preserve any dropped record containing a structured identifier from the current search query. It increased critical retention from 5/9 to 6/9, but added 117 records and 3,249 non-critical tokens.

The adopted-bridge floor used the intersection of prior trajectory support and explicit query reuse. On the same post-hoc replay, it also increased critical retention from 5/9 to 6/9, but added only 21 records and 504 non-critical tokens.

Thus, for the single observed rescue:

- naive query floor: +3,249 non-critical tokens;
- adopted bridge floor: +504 non-critical tokens;
- reduction in added non-critical burden: **84.5%**.

This comparison motivated keeping the adopted rule in production. It is still development evidence because the rule was designed after inspecting the hard replay; it is not an unseen generalization result.

The relevant probes are [`probe_query_hit_preservation.py`](experiments/bridge_relevance/bridge_hard/probe_query_hit_preservation.py) / [`query_hit_preservation_probe.json`](experiments/bridge_relevance/bridge_hard/query_hit_preservation_probe.json) and [`probe_adopted_bridge_floor.py`](experiments/bridge_relevance/bridge_hard/probe_adopted_bridge_floor.py) / [`adopted_bridge_floor_probe.json`](experiments/bridge_relevance/bridge_hard/adopted_bridge_floor_probe.json).

## 6. Unseen validation: applicability remained the bottleneck

I performed two attempts to validate the frozen final rule on unseen natural trajectories.

### G11

A treatment-blind historical applicability screen selected G11. In the fresh run, the agent naturally reused `resolve_litellm_model_name`, but it did so after only one supporting prior output. The frozen bridge admission rule did not admit the singleton `function_name`, so the preservation treatment never activated. Both OFF and ON passed. This produced no causal retention estimate.

A separate treatment-blind screen of 17 historical trajectories found no natural opportunity for the frozen safe-singleton kinds (`exception`, `request_id`, `test_name`).

### Second unseen screen: U01

I then locked a second unseen protocol, deterministically ordered previously unused validated reverse-patch tasks, capped the screen at 12 OFF runs, and committed all 12 snapshots before any new agent execution.

The first task, U01, produced a natural trajectory in which the frozen builder admitted `COPILOT_PROVIDER_TYPE` and `HEADROOM_BACKEND`, and a later `rg` query explicitly reused both. In the single fresh ON run, a live production-gate audit confirmed:

- the target was a call-id-linked Bash search;
- the handler search gate was true;
- trajectory context was non-empty; and
- `COPILOT_PROVIDER_TYPE` was adopted.

However, runtime metrics still reported:

- `relevance_split_units = 0`;
- `search_relevance_chains = 0`;
- `treatment_activated = false`.

Therefore the final preservation floor itself did **not** execute on a relevance-split unit. The upstream bridge/adoption signal reproduced naturally, but the experiment again yielded no unseen causal estimate of retention improvement. The available measurements do not justify attributing the downstream non-activation to a more specific gate.

This negative result is important: in the observed natural trajectories, **reachability through the complete compression pipeline is a larger limitation than the local preservation rule itself**.

The second-unseen protocol is [`second_unseen_adopted_bridge_validation_protocol.json`](experiments/bridge_relevance/bridge_hard/second_unseen_adopted_bridge_validation_protocol.json); the deterministic selection, execution lock, snapshot lock, and final safe summary are in the same directory, including [`second_unseen_adopted_bridge_summary.json`](experiments/bridge_relevance/bridge_hard/second_unseen_adopted_bridge_summary.json).

## 7. Validation and limitations

Focused adopted-bridge tests passed 7/7; the combined trajectory/relevance-split focused suite passed 54/54. A full repository test run produced 11,796 passes, 597 skips, and one failure; that single failure passed when rerun in isolation, so I report it as a full-suite failure that did not reproduce rather than silently treating the suite as fully green.

Main limitations are:

- natural end-to-end tasks had a strong success ceiling;
- naturally applicable final-treatment events were sparse;
- the historical oracle is fix-adjacent, not causal ground truth;
- the final adopted-floor result is post-hoc development evidence;
- agent trajectories are stochastic, so token/runtime comparisons are descriptive unless replayed on identical captured inputs;
- the unseen U01 run reproduced upstream adoption but not final relevance-split treatment activation.

A post-hoc provisional one-search lease was also explored after G11 to separate confidence from lifetime. It detected earlier behavioral adoption but did not change retention in the fresh G11 search result; I therefore leave it as future work rather than production code.

## Conclusion

The study supports a narrower claim than “trajectory relevance makes coding agents solve more tasks.”

**Cross-tool trajectory state is a real information channel beyond the current retrieval query.** It can recover evidence lost by baseline compression, but adding it globally can also retain noise. Requiring both prior trajectory support and explicit current agent reuse yields a selective monotonic preservation invariant that, on the development replay, obtained the same observed critical rescue as a naive current-query policy with substantially lower context cost.

The unseen evaluations did not establish a causal natural retention improvement because the final preservation treatment did not activate end-to-end. That negative result changes the engineering conclusion: further gains likely require improving the full retrieval → search-reachability → routing → relevance-split chain, not simply making bridge extraction more permissive.

I therefore keep the adopted-bridge floor as a conservative production safeguard, while treating its observed 5/9 → 6/9 result as post-hoc mechanism evidence rather than a general benchmark win.
