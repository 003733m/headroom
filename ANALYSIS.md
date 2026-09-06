# Trajectory-Aware Relevance for Coding-Agent Context Compression

## Summary

Headroom already conditions tool-output compression on the user request and the current tool call. I investigated a narrower failure mode: a coding agent's information state changes during a trajectory. It may learn a test name, exception, configuration key, function, class, request ID, or other task-local identifier from an earlier tool output and rely on that identifier several steps later. The current retrieval query is therefore a strong local relevance signal, but it is not always a complete representation of what the agent has learned.

I implemented a bounded, target-conditioned cross-tool trajectory signal and integrated it into Headroom's OpenAI Responses/search compression path. In a controlled 20-case evidence-retention probe, trajectory context increased exact evidence fidelity from **6/20 (30%) to 20/20 (100%)**, while average compression reduction moved from **25.46% to 19.13%** (a 6.34 percentage-point compression cost). Frozen prospective agent runs showed no end-to-end task-success improvement: all OFF and ON tasks passed. Natural and historical replays nevertheless showed that the mechanism can change which evidence survives compression.

The experiments also exposed a selectivity problem. Global trajectory context sometimes preserved extra non-critical material without improving critical recall. I therefore implemented a conservative **adopted-bridge preservation floor**: an identifier can force an otherwise-dropped search segment to remain verbatim only when it is both (1) admitted as a prior trajectory bridge and (2) explicitly reused by the agent in its current `rg`/`grep` query. On a post-hoc strict matched replay, this raised historical-critical retention from **5/9 to 6/9** while adding **504 non-critical tokens**. A naive current-query preservation floor achieved the same 6/9 retention but added **3,249 non-critical tokens**. The adopted rule therefore produced the same observed rescue with **84.5% less added non-critical burden** than the naive query floor.

I do **not** claim a general coding-agent success improvement. The contribution is a characterized state-loss failure mode, a production integration, a selective preservation invariant, and an evaluation that separates controlled mechanism effects, natural activation, post-hoc refinement, and negative unseen validation.

## 1. Gap: relevance changes during a trajectory

The existing Headroom path already uses the current user request and triggering tool-call arguments as relevance context. My extension is not "use the search query"; Headroom already does that. The missing signal is **task-local state learned on the way to the current retrieval**.

A useful analogy is a join key. Suppose an earlier test output exposes `DEDUP_AUTO_THRESHOLD`, `resolve_provider_type`, or a specific failing test name. The agent may later search for that identifier, or for a related subsystem, even though the original user request never contained it. Re-sending the full prior trajectory would undermine compression, so the goal is to preserve only a bounded set of high-signal identifiers.

The implementation is centered in [`headroom/trajectory_relevance.py`](headroom/trajectory_relevance.py), with routing integration in [`headroom/transforms/content_router.py`](headroom/transforms/content_router.py) and the preservation floor in [`headroom/transforms/relevance_split.py`](headroom/transforms/relevance_split.py).

## 2. Implementation

The trajectory extractor considers only bounded prior tool outputs and recognizes structured candidates including file paths, request IDs, test names, exceptions, configuration keys, functions, and classes. Candidates are ranked using causal evidence, cross-tool corroboration, repeated occurrence, recency, specificity, and speculation penalties. The target output is excluded from its own evidence, preventing self-leakage.

For OpenAI Responses, I added adapter logic that reconstructs prior tool-output state and associates shell search calls with their outputs by `call_id`. Search outputs can receive trajectory relevance when they are structurally recognized as search or when their producing call is a recognized Bash `rg`/`grep` search. This was necessary because source-code-looking `path:line:content` can otherwise be classified as code even when it is semantically a search result.

The final production refinement is the adopted-bridge floor. Let `B` be the frozen trajectory bridge set and `Q` be conservative structured identifiers explicitly present in the current search pattern. The preservation set is:

`Adopted = B ∩ Q`

When the ordinary relevance scorer would DROP a segment containing an exact-boundary match to an adopted identifier, the segment is forced to KEEP. Existing KEEP decisions are unchanged. This makes the refinement monotonic: it can only preserve additional evidence, not remove evidence the baseline already retained.

Production algorithm freeze: `89da0898a80c313b6a320f47763391dea7c376e2`.

Focused adopted-floor tests are in [`tests/test_adopted_bridge_preservation.py`](tests/test_adopted_bridge_preservation.py).

## 3. Evaluation design and evidence hierarchy

I used several evaluation layers because "does the compression mechanism retain the intended evidence?" and "does the coding agent solve more tasks?" are different questions.

| Status | Evaluation | Main result | What it supports |
|---|---|---:|---|
| Controlled development | 20-case multi-target fidelity probe | 30% → 100% fidelity | Clean mechanism capability |
| Prospective / frozen | N5–N8 paired agent runs | OFF 4/4, ON 4/4; ON activated 3/4 | Natural activation, no success lift |
| Secondary natural replay | Exact captured inbound requests | 1/18 baseline drops rescued; 0 regression | Direct transfer to real requests |
| Exploratory historical replay | Reverse-patch historical scan | 60/1,326 drops rescued; 3 regressions | Larger conditional opportunity sample |
| Confirmatory hard agent runs | Six hard paired tasks | OFF 6/6, ON 6/6 | Ceiling effect; no end-to-end lift |
| Post-hoc refinement | Adopted-bridge strict replay | 5/9 → 6/9; +504 non-critical tokens | Selective preservation |
| Post-hoc comparator | Naive query-hit floor | 5/9 → 6/9; +3,249 non-critical tokens | Simpler rule is much broader |
| First unseen applicability holdout | G11 | behavior adopted, bridge not admitted | Conservative bridge gate can lag adoption |
| Second unseen validation | U01 | live bridge adoption reached handler gate, but no `relevance_split` treatment activation | Unseen natural applicability, but no retention-effect estimate |

The distinction between prospective, confirmatory, and post-hoc evidence is deliberate. I froze algorithms and manifests before prospective runs, did not replace completed conditions, and recorded harness corrections separately rather than silently regenerating favorable results.

A further design choice was to keep retrieval behavior as natural as possible while constraining the interface enough to make measurements reproducible. Agent prompts required shell `rg -nH` rather than the dedicated Grep tool, but did not prescribe the query content. This let the agent choose what to search while allowing the harness to capture producing commands and outputs consistently. Where exact-wire replay was used, I replayed the captured inbound request rather than reconstructing a hypothetical trajectory.

I also separated algorithm changes from measurement corrections. For example, when a replay matcher failed on folded search-output formatting, I corrected the normalization logic and withdrew the earlier measurement instead of treating the first number as evidence. Similarly, harness fixes were committed independently from production changes. This matters here because several apparent failures were actually reachability or measurement issues rather than failures of the bridge-ranking idea itself.

## 4. Results

### Controlled evidence fidelity

In the controlled 20-case probe, the trajectory signal raised exact evidence retention from 6/20 to 20/20. Average compression reduction changed from 25.46% to 19.13%. This is the cleanest positive result: the additional state can preserve evidence that query-local relevance loses, but preserving that evidence costs compression.

### Prospective natural agent runs

The V3 mechanism was frozen before N5–N8. All eight OFF/ON conditions were completed without replacement or rerun. The ON mechanism activated in 3/4 tasks, but all tasks passed in both conditions. Therefore these runs show **natural reachability and activation**, not a task-success improvement.

### Historical and exact-wire replay

To create more retention opportunities, I mined historical bug-fix commits under a fixed reverse-patch protocol. A production fix was reversed, the pristine snapshot was verified to pass, the reversed snapshot had to produce the expected failing tests, and parent-version lines around the source fix were used as a **historical-fix-adjacent oracle**. This oracle is useful for replay, but it is not claimed to be causal ground truth.

Across 92 baseline-opportunity tasks, trajectory relevance rescued 10 tasks. At record level, it rescued **60/1,326 baseline-dropped fix-adjacent records (4.52%)**, with 3 regressions. Because this analysis is historical and conditional on retrieval and oracle construction, I treat it as exploratory evidence.

I also replayed exact captured inbound requests from natural agent trajectories. On 19 naturally retrieved historical-critical records at first exposure, baseline retained 1/19. A frozen production-wrapper OFF replay reproduced all 19 baseline KEEP/DROP decisions. V3 retained 2/19, rescuing **1/18 baseline drops with no regression**. The three predeclared controlled D→K candidates were not rescued (0/3), which limits any stronger transfer claim.

### Hard paired runs and the selectivity problem

A 24-task OFF-only difficulty screen produced no agent failures, so I selected six harder tasks under a predeclared effort rule and ran paired OFF/ON conditions. All six passed in both conditions. In the final relevance-split stage, the ON treatment did not activate on these runs, so task success could not estimate the preservation rule.

Natural-search analysis nevertheless showed that agents do reuse identifiers learned from prior outputs. Across the hard trajectories, many later queries contained structured identifiers that had appeared earlier, confirming that cross-tool state is a real information channel.

A corrected strict matched replay then showed the main weakness of global trajectory context: baseline/query-only and V3 both retained **5/9** scorable historical-critical records, while V3 retained more non-critical context. This result motivated a more selective rule rather than a stronger global context injection.

## 5. Adopted-bridge preservation

I first tested a naive query-hit floor: preserve any otherwise-dropped record containing a structured identifier from the current search query. It raised critical retention from 5/9 to 6/9, but added 117 records and 3,249 non-critical tokens.

The adopted-bridge floor intersects two independent signals: prior trajectory support and explicit current agent reuse. On the same strict development replay it also raised 5/9 to 6/9, but added only 21 records and **504 non-critical tokens**. Relative to the naive query floor, this is **84.5% less added non-critical token burden for the same observed single critical rescue**.

The concrete rescue occurred for `DEDUP_AUTO_THRESHOLD`: it had been admitted from prior trajectory state and was explicitly reused in the current search. The preservation floor changed the matching historical-critical segment from DROP to KEEP.

This is a real algorithmic refinement, but it is **post-hoc development evidence** because the rule was designed after observing the harder replay behavior. I therefore required unseen validation before making any generalization claim.

## 6. Unseen validation: two negative but informative outcomes

The first treatment-blind holdout screen selected G11. In a fresh run, the agent naturally reused `resolve_litellm_model_name`, but only after one supporting prior output. The frozen bridge gate did not admit the singleton function name, so the preservation treatment never activated. Both OFF and ON passed. This exposed an **early-adoption lag** in the conservative bridge-admission rule.

I then locked a second unseen validation protocol before running new tasks: deterministic candidate ordering, maximum 12 fresh OFF screens, first eligible task only, no replacement, and no retention/fix-oracle inspection during selection.

U01, the first screened task, naturally produced a later search for `COPILOT_PROVIDER_TYPE|HEADROOM_BACKEND`. The frozen builder admitted both identifiers as bridges and the query explicitly reused them. In the fresh U01-ON run, the agent again passed the task. A live gate audit showed that the relevant 20,671-byte output was attached to a recognized Bash search call, so the actual handler search gate was true even though the structural detector alone returned false. The trajectory context was non-empty and `COPILOT_PROVIDER_TYPE` was adopted.

However, runtime metrics still reported **0 relevance-split units, 0 search-relevance chains, and `treatment_activated=False`**. The unseen run therefore demonstrates that bridge admission, behavioral reuse, and the provider-level search/context gate can all occur naturally, but it does **not** provide an unseen retention-effect estimate for the preservation floor.

This is an important limitation rather than a result to hide. The full path is:

**retrieval → search identification → trajectory context → relevance-split routing → preservation**

G11 stopped at bridge admission. U01 advanced further, through natural adoption and handler-level trajectory context, but still did not reach the final preservation stage. The remaining bottleneck is therefore not only candidate quality; end-to-end routing determines whether the preservation invariant can act at all.

## 7. Conclusion

The study supports a narrower claim than "trajectory relevance improves coding-agent success."

First, prior tool outputs contain task-local state that is not always captured by the original request or current query alone. Second, a bounded trajectory signal can strongly improve evidence fidelity in controlled cases and occasionally rescue naturally retrieved historical-critical records. Third, global trajectory context is not sufficiently selective by itself. Intersecting prior bridge support with explicit agent reuse produced a much more selective monotonic preservation rule: in development replay, it achieved the same observed rescue as a naive query floor with 84.5% less added non-critical burden.

However, end-to-end agent benchmarks showed a ceiling effect, and two unseen validation attempts did not produce a causal retention estimate for the final preservation rule. G11 failed at conservative bridge admission; U01 reproduced bridge admission, agent reuse, and the handler-level context gate but did not reach final relevance-split activation.

The strongest current conclusion is therefore:

> **Global task intent, current retrieval intent, and evolving trajectory-local state are distinct relevance signals. Cross-tool state is real and useful, but its value depends on reliable end-to-end routing and selective preservation rather than simply injecting more history.**

## Reproducibility

The production implementation is in [`headroom/trajectory_relevance.py`](headroom/trajectory_relevance.py), [`headroom/transforms/content_router.py`](headroom/transforms/content_router.py), and [`headroom/transforms/relevance_split.py`](headroom/transforms/relevance_split.py).

The main experiment code and protocols are under [`experiments/bridge_relevance/`](experiments/bridge_relevance/) and [`experiments/bridge_relevance/bridge_hard/`](experiments/bridge_relevance/bridge_hard/). Historical replay is implemented in [`run_exhaustive_discovery.py`](experiments/bridge_relevance/bridge_hard/run_exhaustive_discovery.py); exact-wire replay is in [`run_exact_wire_v3_replay.py`](experiments/bridge_relevance/bridge_hard/run_exact_wire_v3_replay.py).

The second unseen validation is documented by [`second_unseen_adopted_bridge_validation_protocol.json`](experiments/bridge_relevance/bridge_hard/second_unseen_adopted_bridge_validation_protocol.json), [`second_unseen_adopted_bridge_selection.json`](experiments/bridge_relevance/bridge_hard/second_unseen_adopted_bridge_selection.json), [`run_second_unseen_adopted_bridge_screen.py`](experiments/bridge_relevance/bridge_hard/run_second_unseen_adopted_bridge_screen.py), and the locked execution/snapshot files in the same directory.

Focused trajectory/relevance tests passed. A prior full-suite run completed with **11,796 passed, 597 skipped, and one failure that did not reproduce when rerun in isolation**.
