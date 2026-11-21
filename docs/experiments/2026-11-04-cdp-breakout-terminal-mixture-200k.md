# Fresh Breakout200k comparison with terminal-containing replay

Select one fresh200k-action learner after the [bounded mixture check](../results/2026-11-04-cdp-breakout-terminal-mixture.md)
passes execution and its30-minute cost gate. At8,192 actions, the optional.25
mixture raises terminal replay rows322→1918 and updates containing endings225→435,
with46.900s training and conservative200k projection20m59.517s. Short frozen play
completes no natural episodes. This full comparison tests gameplay benefit within
the original five-game goal; no success is inferred from exposure or finite losses.

## Fixed learner and source differences

Keep centeredCDP12M/RGB64/Auto/N8/B16/T16/full-BPTT/micro16/H15/R16,
base/encoder4e-5, dynamics4e-4, behavior8e-5, replay-value1, continuation1,
expert100/unit weights/world gradients, action-effects1, replay100k/context1,
sampled collection and posteriors, actor-unimix.01, full18/sticky.25/repeat4/noops0.
Preserve native pooled RGB and one GPU resize, all-state corrective labels and
20% episode guidance/cap1920. Development seed1009 is reused. Start from a fresh
initialization; no bounded weights, replay or experience are transferred.

Set `terminal_replay_fraction=.25`. Fresh sequences retain priority. For each
remaining sequence, choose with probability.25 uniformly among valid starts
containing a true terminal in training rows; otherwise use uniform replay. Empty
terminal pools fall back to uniform. Deduplicate overlapping ranges; exclude
prefix-only terminals and timeouts. Intermediate life losses remain nonterminal.
There is no importance correction. This changes the data distribution for every
objective and is not a calibrated-probability or isolated continuation-loss test.

Reuse native source`2b5a3eea`, SHA256
`5ce1b443ac270bd130ac1b1ead54d3e6b573913cfebd953fa3604367032a84a2`, from
`/x/Temp/kindle-terminal-replay-20261104/runtime`, and the bounded runner bytes.
No rebuild or optional runtime mixing. Native and runner differ from retained
447fe50. The new package also has later guidance wrappers for optional block
selection/minimal actions; their source bytes differ from the retained package.
Those modes are disabled here. All three pixel-policy classes have identical ASTs
and constants; the scheduler methods other than construction are AST-identical.
A.545s CPU comparison matches schedules/RNG/overrides over4001 full18 episode
steps/eight lanes,41 synthetic pixel batches andfive partial-reset batches. This
supports the active recipe, not equivalence for every trajectory or disabled mode.
Pin both source versions and this qualification; do not claim byte-identical teacher
files. Bounded default-off frozen compatibility matches128 semantic events and
357 invariant tensors; initial357 tensors/47 events match the retained recipe.

## Budget, recovery and frozen assessment

Train exactly200,000 aggregate actions/12,477 updates with a1,800-second cap.
Require first update376 and residual credit.5. Save actual initialization and
100k/6,227 and200k/12,477 checkpoints. The100k observer verifies finite weights
and moments and makes that recovery directory/files read-only; a second CPU
reader verifies it once while preserving the original learner identity. These
checkpoints omit replay, games, live beliefs and full RNG state.

After training and its audit, run four frozen arms in order: final sampled,
actual-initial sampled, final greedy, actual-initial greedy. Each selects the
firstthree natural episodes per each ofeight lanes,24 episodes, seed4000001009,
with guidance, labels and updates off. Each arm has600k-action/300-second caps
and the existing100k-frame episode cap. Require zero truncations and357 invariant
tensors per arm. Preserve every cutoff or failure; do not raise caps, repeat
completed arms, extend the learner or change selection based on observed scores.

Require bounded/retained initial tensor equality, paired pre-update semantics,
actual-initial sampled/greedy semantic comparisons, all finite reports and correct
world/behavior loss totals. Audit route totals and terminal-row counts against
each continuation-target mean; report total and50k-window exposure. Retain full
training pixel/action/label replay, frozen recorded-action/outcome/frame/boundary
replays and videos, coverage, guidance exposure, curves and host guards. Ordinary
frozen logs have no live pixel hashes; do not claim equality to unrecorded images.
Stage audits gate subsequent GPU stages. Training replay cap900s, frozen replay
cap500s, main7200s and terminal observer7500s. CPU co-host costs remain uncorrected.

## Locked comparison and decision

Retained control: sampled70.750/greedy143.666667, actual initials1.416667/2.208333,
all0/24 two-wall,200k actions/12,477 updates in1287.910495s. Reuse its closed
evidence without replaying completed stages. Greedy return is primary; sampled
return is secondary. Compare absolute return and each arm's own-initial-subtracted
return per training minute. Use paired eight-lane bootstrap,10,000 resamples,
seed0. Favor the candidate only when primary return and gain/min both rise and
the gain/min difference interval's lower endpoint exceeds zero. Report both modes,
all absolute scores and the original mean864/90%-two-wall gate separately.

One reused development seed/cohort and a historical control cannot establish
training-seed reliability or a uniquely causal sampler effect. Native, runner,
guidance-wrapper bytes and subsequent experience differ. No mixture sweep,
second learner, unchanged extension or automatic successor follows this test.

Root `/x/Temp/kindle-terminal-replay-20261104/full`; service
`kindle-breakout-terminal-mixture-200k-20261104`. Preserve the earlier source-Python
metadata mismatch and supplementary bounded CPU-reader failure/repair. New helper,
comparison and all five actual queue preflights must pass before launch on an idle
host. The original goal remains unfinished.
