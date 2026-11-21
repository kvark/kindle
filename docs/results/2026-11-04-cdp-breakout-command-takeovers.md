# The command-dependent teacher rescues fewer current Breakout failures

Retain the existing teacher. From identical states eight decisions before
recorded life losses, the existing teacher survives the next32 decisions in
**84/120 cases (70.00%)**, versus **79/120 (65.83%)** for the saved command-
dependent predictor. The candidate loses five net rescues (−4.17 percentage
points), improving only two of eight lanes. It fails the fixed requirement of
12 net additional rescues and improvement in at least five lanes.

This makes teacher replacement less compelling for these current student
states. The existing teacher still avoids many losses, so useful corrective
behavior is available that the learned policy does not consistently reproduce.
Neither result identifies a perceptual, memory or optimization cause, nor proves
that different supervision will improve learning. No teacher integration,
coefficient/window retry or GPU learner is selected.

## Fixed paired outcomes

The [protocol](../experiments/2026-11-04-cdp-breakout-command-takeovers.md) selects
all120 observed life losses from the first three natural episodes in each of
eight lanes of the retained143.667-point greedy policy. This is the current
encoder4e-5/behavior8e-5/replay-value1 checkpoint, not the older104.667-point
policy used in the earlier recoverability work. Each origin is eight decisions
before its recorded loss, bounded by the current life start. All120 windows
contain nine recorded actions including the losing action; no origin is replaced.

| Outcome | Existing teacher | Command-dependent candidate |
|---|---:|---:|
| Survives all32 decisions without life loss |84/120|79/120|
| Avoids loss at the original ninth-action endpoint |92/120|89/120|
| Loses a life within32 decisions |36|41|
| Mean reward through32 decisions or first loss |3.692|3.575|
| Mean first-loss action, among branches that lose |12.639|13.049|
| Truncations |0|0|

The first-loss range is9–28 actions in both arms. Reward and loss latency use
stopped branches with different lengths and populations; they are descriptive,
not full-game scores or alternative selection criteria. Both teachers survive
in67 cases, only the existing teacher in17, only the candidate in12, and
neither in24. Failure of both does not establish physical irrecoverability.

| Lane | Origins | Existing32-action survival | Candidate survival | Difference |
|---|---:|---:|---:|---:|
| 0 | 15 | 11 | 11 | +0 |
| 1 | 15 | 8 | 7 | -1 |
| 2 | 15 | 11 | 11 | +0 |
| 3 | 15 | 11 | 10 | -1 |
| 4 | 15 | 12 | 13 | +1 |
| 5 | 15 | 12 | 9 | -3 |
| 6 | 15 | 8 | 10 | +2 |
| 7 | 15 | 11 | 8 | -3 |

The120 cases are outcome-conditioned, correlated states from one reused model
and evaluation cohort. They are not120 independent gameplay trials. The test
measures a fixed short takeover, not complete-game quality, training utility
or reliability across learner seeds. The original864/90%-two-wall task remains
unfinished; the native policy's retained scores and training cost do not change.

## Policy histories and replay validation

Both teachers use native pooled RGB and actually executed full18 actions, under
sticky.25/repeat4/noops0. Their histories are reconstructed on the same saved
student trajectory. The private candidate adapter feeds the executed horizontal
command back into the saved predictor, preserving its older-command and observed-
velocity rules. Its predictor source, coefficients, detector, clipping and planner
remain byte-identical to the [prior command-dependent teacher](2026-11-01-cdp-breakout-command-response.md).
Its earlier fresh8.87% full-game gain still fails that experiment's10% screen;
this new comparison neither revises that threshold nor promotes that result.

Each origin has an exact recorded-action branch through the original loss and
two teacher branches, stopping at the first life loss or terminal, up to32
four-frame decisions. Process copies preserve ALE sticky-action memory/RNG.
Every recorded branch matches its saved pixels, actions, rewards, life counts,
frames and boundaries. Parent ALE state, pooled buffers, task/counters and both
policy histories remain unchanged. The two teacher branches are repeated in
reverse order at the first origin in each lane, with exact results in all eight
checks. These are declared validation branches, not repeated experiment stages.

There are240 teacher branches,120 recorded controls and16 reverse-order teacher
checks. Teacher branches execute24,804 raw frames; all validation branches
execute5,967. The two full CPU replays each reproduce **43,424 actions /173,660
frames**, all25 original completed episodes and every partial tail. They have
identical annotation hashes. The original logs lack live pixel hashes, so this
is equality between CPU replays, not an unavailable native-image comparison.
Life counts are observer-only and never enter either policy. Independent
recount verifies zero life gains and first-loss stopping in every branch.
No native model is imported, GPU context constructed or learning update made.

## Completion and preserved import failure

All12 original CPU fixture cases pass, including real ALE process-copy parity,
feedback history, origin selection and primary decision arithmetic. Original
preflight takes3.188s. The first controller, source84de5632, then fails during
imports: the general cohort module pulls in a newer runner requiring
`kindle._return_boundaries`, absent from the preserved runtime. The.100s import
failure and1.415s controller execution remain unchanged. No emulator or data
stage starts in that attempt, and it has no passing completion receipt.

A separate `cpu-import-repair` extracts the exact unchanged cohort function and
changes only its private import. Four repair checks pass in2.831s, including the
actual annotation entrypoint stopped before game construction. No method,
coefficient, source trajectory, cap or selection rule changes. The original
12 tests are retained without repetition. Source16ddeae1 selects the repair.

Annotation completes in20.830s and branch replay in47.276s. The corrected
controller exits successfully **01:24:46.262 UTC November4**, after70.996s.
A1.406s independent recount checks all saved origins, trajectory outcomes,
selection arithmetic and original/repair inputs. Separate closure completes
**01:25:18.665** in4.008s as `complete_with_cpu_import_repair`. All470 original
and490 repair pins pass; no completed replay/analysis stage or GPU work repeats.
The original controller remains failed and the repaired execution remains
separately identified.

Evidence root: `/x/Temp/kindle-breakout-command-takeovers-20261104`.
Results and separate closure: `cpu-import-repair/`. Retain `annotations.jsonl`,
`origins.json`, `branches.jsonl`, both stage reviews, `independent-recount.json`,
`closure-review.json`, exact policy sources and every original failure receipt.
No Kindle allocation is active at closure. The five-game goal remains unfinished.

| Artifact | SHA256 |
|---|---|
| Original import failure | `74da3e34637a0433b4762ce776ef6d898deed08b9b94d95f26e53f2c92af023c` |
| Repaired completion | `ce1d06cd2162dbb55ed58dabc8376b92b27ae3f9dcb58ec67d0d6fce0645abaa` |
| Independent recount | `9166b57196e72083d6b4a4194243fdd85a6b16924559ab04f5e96f7827c81c47` |
| Fixed origins | `0e1b8c75478abfd47c6406b60c34ef0edcec3701e5f7985bdf892d37dd51330e` |
| Every branch trajectory | `2a23e13c8f016a9a21df96732c90c5783046e24fbe8dea3d10c2efda571708b4` |
| Both replay annotations | `1a148bbaf5d60cbb1c18913e6635502c67cf235bd214f22875c867c506c57c3b` |
