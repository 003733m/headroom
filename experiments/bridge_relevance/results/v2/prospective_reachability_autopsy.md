# Frozen V2 prospective reachability autopsy

Freeze: `6e97d220477ca7d4e01a9ab044bb5606d77ff32a`

No V2 implementation changes were made during this analysis.

## Prospective result

Across the four predeclared unseen coding-agent tasks, measured
`relevance_split` activation was 0/4.

This means downstream task-success differences cannot be attributed
to V2.

## Routing reachability

N1, N2, and N4 did not expose a SEARCH-routed current tool result in
the measured trajectory despite the agents performing repository
searches through shell `rg`.

This exposes a distinction between tool intent and output content:
a shell repository search can be classified as mixed, code, or log
content even though its operational role is search.

## N3 temporal analysis

N3 did reach SEARCH-classified outputs.

An offline replay over agent-visible search-result text showed:

- S1: 0 prior outputs, 0 V2 selections
- S2: 1 prior output, 2 frozen-V1 ranked candidates, 0 V2 selections
- S3: 2 prior outputs, 3 frozen-V1 ranked candidates, 0 V2 selections
- S4: 3 prior outputs, 19 frozen-V1 ranked candidates, 2 offline
  V2 selections

The task-critical regression identifier
`test_count_messages_tolerates_null_tool_calls` was visible in S3 and
S4, but not S1 or S2.

Before S4 it therefore had only one prior independent search-output
observation:

- distinct prior outputs: 1
- occurrences: 1
- positive evidence: 0
- frozen score: 1.667

It consequently failed the frozen V1 eligibility rule requiring
either two prior-output observations or positive causal evidence.

The current S4 target contained the same identifier, but frozen V2
uses the current target only after V1 eligibility. It therefore could
not use S4 as corroborating evidence. After S4 the identifier had been
observed twice, but no later repository search remained in the task.

This is a temporal evidence-maturation failure: useful evidence becomes
sufficiently corroborated one search later than it is needed.

## Offline/production discrepancy

The reconstructed offline S4 replay selected `OpenAI` and `MagicMock`,
while the actual prospective run recorded no `relevance_split`.

Therefore the replay is diagnostic rather than an exact reconstruction
of the production Responses request state. Candidate availability in
the visible transcript is not equivalent to candidate availability at
the production integration point.

## Design implication

The results do not justify retuning the frozen scoring weights.

Two higher-level changes are motivated instead:

1. recognize repository-search tool intent independently of output
   content classification;
2. allow a structured identifier observed in one prior tool output to
   use an exact, discriminative occurrence in the current SEARCH target
   as corroboration, while still forbidding current-target-only
   candidate discovery.

These changes belong to a post-prospective iteration and must not be
reported as part of the frozen V2 prospective result.
