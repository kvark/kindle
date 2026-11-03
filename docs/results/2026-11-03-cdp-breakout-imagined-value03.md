# Breakout imagined-value weight0.3: bounded execution and cost pass

The fresh 8,192-action check passes in **47.892s**, with 32,727 frames and
489 updates. Its conservative 200k projection is **21m26.159s**, below the
30-minute cap. The frozen check takes .879s for 1,024 actions, with zero updates
and no natural completed episodes. This establishes execution and early cost;
it does not measure gameplay benefit.

Only `loss_scales.value` changes from1 to.3 relative to the retained replay-value1
control. This weights imagined return fitting and its slow-value regularizer
together. Replay-value remains1 and actor/critic representation gradients remain
off. Native447fe50/SHA35ad0db and the teacher remain unchanged. The private
runner76440d32 adds12 lines for a fresh-CDP coefficient option; omission preserves
defaults. Kindle's replay targets still bootstrap from imagined returns.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-imagined-value03.md)
records the primary NanoDreamer source that motivates the coefficient, the
unchanged recipe and the limits of that analogy. Different architecture, replay
sampling and targets prevent a matched-baseline claim.

All393 pins,357 control-matching initial and invariant frozen tensors,47 paired
pre-update semantic events,16,334,353 parameters, exact cadence/configuration,
finite reports, changed CNN/reward/actor/value, full guided pixel/action/label
replay and host guards pass. Actual overrides are3,733; guided behavior is not
unaided competence. GPU telemetry and derivative correctness are unmeasured.

Original main/collector succeed **08:12:43.805 /08:12:49.826 UTC November3**,
source3079329989454399a6475d92e030945bbcca7dce. Independent terminal review
completes once **08:14:11.721**, in1.443s CPU (1.462s wrapper). Main identity:
PID994517/invocation3381250063eb444883acad5be14cbd02; collector PID994520/
invocation67867552f5fc4eb9b71e25cefda4e55a. Completion SHA
`8f47e6d89bc2b80a15ba6fe09e78021bde507b9b591f42cafbc6f7d7e7124885`;
review SHA `b932e6888b8d27f1583389c965ee5855830f5772cc2e566306ef834c586937fb`.

One earlier CLI fixture accidentally constructs a native agent when lazy module
loading replaces its stub, then fails on a dummy frame before completing episode
initialization. No update or policy evaluation occurs. Preserve that failure,
original source, correction and kernel journal; do not describe the attempt as
CPU-only. The corrected proxy fixture and native-free actual-command preflight
pass. The captured kernel interval has resume messages, with no explicit fault,
reset or hang line. No host recovery or successful GPU-stage repeat occurs.

Root `/x/Temp/kindle-breakout-imagined-value03-20261103/native-check` is closed;
do not repeat its stages or original collector. A separate
[fresh full comparison](../experiments/2026-11-03-cdp-breakout-imagined-value03-200k.md)
is selected after this result, without reusing its experience. Retained control
scores70.750 sampled/143.667 greedy remain the benchmark, all0/24 two-wall.
The five-game goal remains unfinished.
