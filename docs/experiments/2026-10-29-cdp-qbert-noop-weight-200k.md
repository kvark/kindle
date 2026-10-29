# Fresh Qbert200k with reduced wait-label weight

The [bounded check](../results/2026-10-29-cdp-qbert-noop-weight.md) passes all45
pins,357 matching initial/invariant frozen tensors,weighted loss accounting,
full guided replay and host guards.8192 actions/489 updates take70.160s; the
linear200k projection28m33s becomes31m24s with a10% margin, below35min. This is
execution/cost evidence, not task competence. The prior CPU import assertion
failure is preserved and corrected before GPU launch; no GPU stage repeated.

Separately select one fresh200k-action/12,477-update learner. Change only the
NOOP expert-label weight1→.25 from the retained strongest12M control. Keep
12M48/Auto/N8/B16/T16/H15/R16/centeredCDP coefficient1,actor-unimix0/full18,
original rates,20%episode continuation guidance/cap1920/expert100/world gradients/
all-state corrective labels,seed4013/evaluation4000004013. Other17 action weights
remain1. Native447fe50, the copied runner and Qbert teacher remain identical;
use `/x/Temp/kindle-breakout-corrective-20261028/runtime`. Frames retain one GPU
resize. No bounded/control weights, replay, belief, games, RNG or experience
are reused. This is a reused development seed/cohort, not independent confirmation.

The loss averages all replay rows, not the sum of weights. Reducing wait weight
changes relative class pressure and overall expert magnitude in actor and world
training together. It is not a pure class-balance change at fixed total strength.
The control's late-training passive post-pyramid wait/hop agreement95.044%/55.968%
motivates this test but does not establish a causal bottleneck or teacher accuracy.

The primary comparison is fixed-final greedy mean against retained11984.375/24
first pyramids in30m27s. Keep sampled final6759.375/24 as a separate secondary
comparison; do not switch primary mode based on outcomes. Evaluate candidate
final and actual-initial checkpoints with both sampled and greedy actions,
sampled posterior throughout. Each arm selects first3 natural episodes per8
lanes=24, with guidance/labels off, zero updates/truncations and retained partial
tails/excess episodes. Original mean15000/90%-first-pyramid gate remains fixed.
Compare both mean and success fraction, training wall time, actions and emulator
frames. Online teacher-guided scores are not frozen competence.

Save and audit100k recovery at6227 updates; final200k has12,477 updates, first
update376/residual credit.5. Require357 matching control/bounded initial tensors,
357 finite/invariant frozen tensors per arm, paired actual-initial semantic
traces, exact config/source/counters, finite reports, changed CNN/reward/actor/
value, aligned raw actor/world expert losses and valid weighted sums for every
report. The implied wait/hop row counts must be nonnegative integers with
`.25 * wait_rows + hop_rows == weight_sum`; they are inferred from reported
sums because replay indices are not independently logged. Require complete
training label/input-pixel/action replay, frozen outcome replays/videos, host
guards, curves, coverage, actual guidance exposure and completed comparison.
Ordinary frozen logs lack live pixel hashes; do not claim direct live-image
identity. Preserve original declarations and resolve historical input paths
through their existing verified archive map.

Caps:training2100s, each frozen arm300s/600k actions, training replay1200s,
CPU audit1800s, main service5400s and postprocessing6000s. Automatic read-only
progress and100k recovery observers use the exact parsed commands and fixed
deadlines. Observer timeout does not stop or restart training. Stop on real
faults/numerical failure; no driver recovery, retry, cap increase, extension,
checkpoint selection or automatic successor. A favorable result would need
separate seed confirmation; this run alone cannot complete the five-game goal.

Root `/x/Temp/kindle-qbert-noop-weight-20261029/full`, trial
`qbert-12m-48-b16-r16-expert100-noop025-200k-seed4013`, service
`kindle-qbert-noop025-200k-20261029`. Check live state before launch. The bounded
service is terminal success and the host is idle at selection.
