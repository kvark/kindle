# Breakout replay16k: bounded execution and cost check

**Closed with CPU audit repair:**24,576 actions/98,167 frames/1,513 updates
in158.358s;1,024 frozen/.903s,zero updates/no natural episodes.515 saturated
reports,16k semantic prefix and378 original/408 repair pins pass. Conservative
200k projection23m37.590s clears30min. Original main/collector remain failed
after an over-specific wall-time comparison; separate CPU closure succeeds
04:59:11.269 UTC November3. No GPU stage repeats. No full learner is selected.
See the [result and preserved failure](../results/2026-11-03-cdp-breakout-replay16k.md).
The selection and launch sections below describe historical states.

Select one fresh check changing replay capacity100,000→16,384 stored frames
from the retained lambda0.95/replay-value1 control. Its frozen means remain
70.750 sampled/143.667 greedy, with zero two-wall successes. The completed
[lambda1 comparison](../results/2026-11-03-cdp-breakout-lambda1.md) is closed;
do not repeat any of its stages or readers.

Native447fe50's replay retains recurrent states from acting or a later sampled
refresh. Sampling uses one cached context frame before each T16 sequence and
prioritizes fresh queued starts before uniform sampling. Learning refreshes
sampled states using the pre-update posterior. Capacity is a shared arrival
budget across eight streams, not a per-stream limit. The source review does
not measure cache age or prove a stale-state defect. A smaller buffer may
adapt more quickly as the representation and policy change, but also removes
older experience and changes diversity, reuse and visited-state coverage.
A gameplay difference would not isolate any of those effects.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16, lambda.95,
encoder/base4e-5, behavior8e-5, dynamics4e-4, replay-value1, expert100 with unit
weights/world gradients/all-state corrective labels, sampled collection and
posteriors, actorunimix.01, action-effects1 and warmup1000. Keep20% episode
guidance/cap1920, seed1009/evaluation4000001009, full18/sticky.25/repeat4/noops0,
native pooled RGB and one GPU64 resize. Native447fe50/SHA35ad0db, exact retained
runnercf4ea238 and teacher remain unchanged; the existing `--replay-capacity`
option needs no source change or build. There are16,334,353 parameters.

Start fresh with no transferred weights, replay, belief, games, RNG or prior
check experience. This is one reused development seed/cohort, not independent
reliability evidence. Preserve the original Breakout90%-two-wall gate.

## Fixed bounds

- Train24,576 aggregate actions within240s, expected1,513 updates, first update
  at376 and residual credit.5. This exceeds the16,384-frame capacity and tests
  learning after eviction. Save actual-initial and final checkpoints.
- Require357 exact control-matching initial tensors,47 pre-update transitions
  and complete semantic action/reward/pixel/label/override equality with the
  retained control through16,000 actions, before capacity affects sampling.
  Do not require equality after evictions begin or infer hidden-state equality.
- Require the exact capacity-only configuration,1,513 finite supervised reports,
  changed CNN/reward/actor/value weights,full guided replay and host guards.
- Restore final weights for1,024 greedy frozen actions within60s, guidance and
  labels off, zero updates,357 invariant tensors. This is not a competence test.
- Keep replay120s/process150s, total CPU audit300s, main service720s and original
  collector840s with780s observation deadline. No overlapping GPU allocation.
- Cost gate: check training seconds×200,000/24,576×1.10 ≤1,800s. A full learner
  requires separate selection after original closure; no automatic promotion.

CPU preflight must check both actual commands, exact config acceptance and
rejection, unchanged runner/native/teacher identity,357-tensor schema, schedule
through capacity saturation, retained control pins and preceding terminal
services. Pin exact native replay source and the new controller sources.
Preserve any failure and inspect saved evidence before a separate CPU repair;
no retry, cap increase, sweep, extension or repeat of a successful stage.

Root `/x/Temp/kindle-breakout-replay16k-20261103/native-check`, trial
`breakout-12m-64-b16-r16-replay16k-expert100-check`. Main service
`kindle-breakout-replay16k-check-20261103`, original collector
`kindle-breakout-replay16k-check-collection-20261103`. Check live state before
launch; no full learner is selected by this protocol.

## CPU qualification

Preflight passes **04:50:14.518 UTC November 3** in1.033s (1.067s wrapper):
two actual commands,20 exact config cases,357 retained tensor shapes,338
unchanged control pins and schedule376/1,513/6,227/12,477 through capacity
saturation. No native module, model or emulator is constructed. The runner
is byte-identical tocf4ea238; no source patch, build or repeated test is needed.

The saved native447fe50 replay sourceSHA is
`c252d85d787726fb1a5038b76fc84a0ce821da78dc6af69735cea3d0cef80355`.
Its storage/sampling/context-refresh prefix matches current production source;
later additions concern optional open-loop weights and tests. The exact saved
agent source confirms that refresh uses the posterior computed before the
world update. Source-reviewSHA
`b7c951996c141bca7cf0f7ad85949e3d9a6fa9893ba13a4182269e58fc4d8233`.
Neither source inspection nor the cadence calculation measures actual cache age.

The GPU audit also requires final replay length16,384 and at least512 reports
with saturated replay, as well as the fixed16,000-action semantic prefix.
Source preparation takes.036s. No failed preparation or GPU allocation occurs
at qualification; full learning andgameplay benefit remain unmeasured.

## Launch

The check launches04:51:39.595 UTC November3/source1e277a5 with378 pins,
declarationSHA`c70548faa4622c1154babe39a7c8a6fe99bb96a39d2322dcdc957a283cef3178`.
Main976085/invocationaf346c90f72b4c26bbee0fa9333738bf andcollector976088/
invocationc96a1d9b2ced4483b1ed81f4aa0b7615 areactive atstartup. Actual header
matches the selected config/native/runner. Aread-only status scan attempts to
parse the partial first line of a tail buffer andfails; its separate receipt is
preserved. The corrected observation excludes partial lines. No experiment
stage fails orrepeats, andthe original process identities remain unchanged.

## CPU repair and closure

Both GPU stages complete within their original bounds, but the long-prefix
audit also compares episode `elapsed_seconds`. The first mismatch is action1,224;
all40 differences in2,078 events are wall-clock fields. Original main/collector
fail04:54:31.441/04:54:39.720 UTC andremain failed. No original review orcompletion
exists. The separate repair excludes wall-clock time from semantic equality,
retaining all gameplay fields andthe fixed16k boundary; eight fixture cases pass.

Corrected CPU audit completes04:59:11.269 in17.639s, including the previously
unrun12.906s guided replay. All378 original/408 repair pins,357 initial/invariant
tensors,47 pre-update events,1,513 finite reports,515 saturated replay reports,
2,078 matching prefix events andguards pass. Training24,576/98,167frames/158.358s
andfrozen1,024/4,096frames/.903s areunchanged,withzero frozen updates/no natural
episodes. Conservative200k projection1,417.590s passes1,800s; gameplay remains
unmeasured. Closure`complete_with_cpu_audit_repair`,SHAfa2de49f; no GPU stage,
successful replay ororiginal collector repeats. Original success-only terminal
reader staysunrun. Bothroots areclosed; no full learner orallocation isactive.
Do not repeat completed stages; preserve all original failures andcheck live
state before separately selected work. The five-game goal remains unfinished.
