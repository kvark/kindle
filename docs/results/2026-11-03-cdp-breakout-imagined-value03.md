# Lower imagined-value weighting worsens frozen Breakout

Reducing `loss_scales.value` from 1 to 0.3 lowers sampled frozen score by
**49.53%** and greedy score by **29.00%**, with **1.28% more training time**.
Retain imagined-value1/replay-value1 and actor/critic representation gradients off.
This local result does not support adopting NanoDreamer's critic weighting in
this Kindle recipe. It does not isolate the failure mechanism or establish that
the coefficient is generally harmful across architectures or seeds.

| Measure | Retained value1 | Candidate value0.3 |
| --- | ---: | ---: |
| Sampled final / actual initial | 70.750 / 1.417 | 35.708 / 1.417 |
| Greedy final / actual initial | 143.667 / 2.208 | 102.000 / 2.208 |
| Two-wall successes, each final arm | 0/24 | 0/24 |
| Training time | 21m27.910s | 21m44.356s |
| Initial-subtracted sampled score/min | 3.230038 | 1.577406 |
| Initial-subtracted greedy score/min | 6.590132 | 4.590387 |

The candidate completes **200,000 actions / 799,594 frames / 12,477 updates**.
Initial-subtracted score per training minute falls **51.16% sampled / 30.34%
greedy**. Four frozen arms total 75,720 actions / 302,707 frames / 66.473s,
with zero updates and truncations. Each uses the first three natural episodes
in eight lanes, guidance and labels off. The 90% two-wall gate is unchanged.

Only the imagined-value coefficient changes in the retained native configuration.
It scales imagined return fitting and its slow-value regularizer together;
replay lambda targets still bootstrap from imagined returns. Replay-value stays 1,
actor/critic representation gradients stay off, and native447fe50/SHA35ad0db and
the teacher remain fixed. Qualified runner76440d32 adds 12 lines for this option.
The [full protocol](../experiments/2026-11-03-cdp-breakout-imagined-value03-200k.md)
records the unchanged recipe and reused development seed/cohort. No experience
transfers from the bounded check or retained control.

Replay expert CE over the final 50k actions slightly improves .560514→.558339,
and guided online last50 rises 67.78→78.46, despite worse unaided play. Actual
guidance overrides are 87,974 (43.987%). These training measures do not establish
competence or explain the regression. Fixed score-bin accounting finds native
training actions at raw scores216–431 fall 1,262→404 (.202% of all actions),
with none at432+. Candidate greedy reaches216 in3/24 selected episodes;
sampled reaches it in none. These bins describe exposure, not verified wall
transitions or a causal coverage limitation.

All 448 input pins, 100k/6,227 recovery, 357 matching initial/invariant frozen
tensors, 47 pre-update events, 800/795 paired initial semantic events, finite
reports, full pixel/action/label and frozen replays/videos, host guards, curves,
exposure and comparison pass. Fixed coverage reconciles 275,720 actions /
1,102,301 frames with 48 diagnostic/448 full pins, reusing the control analysis.
The comparison figure is visually checked and its footer fits. No learner,
replay or GPU work occurs in the final readers.

Original main/postprocess/collection succeed at **08:44:22.585 /08:44:33.979 /
08:45:10.209 UTC November3**, source44f2cb481e6ae22035fb75f6181dfcefdc0fd2e3.
All five original services succeed. Original recovery and its confirmation
already passed. Terminal/coverage readers complete once at **08:45:43.886 /
08:45:54.789**, taking 2.036s/2.972s CPU. Keep CPU co-host timings uncorrected.
No full-run failure, successful GPU-stage repeat or collector repeat occurs.
Five prelaunch CPU failures/corrections and the bounded fixture's unintended
native initialization remain preserved in the protocol.

Root `/x/Temp/kindle-breakout-imagined-value03-20261103/full` is closed. Do not
repeat stages, observers or final readers. No allocation remains active at closure;
check live state before other work. No extension, coefficient sweep, confirmation
learner or automatic successor is selected. The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Learning review | `1d938cea8705e7f4186ccbce6b1bdb8423b3b2a66699ab036de2b78d8cf55513` |
| Completion | `90fcf03d936cfc232b62cdf67cd35857df91b1ad6e99998bb49ebbb48b827c2e` |
| Terminal summary | `314d52371c4b04a377921e3b91834189dcad337e036e9d466c95b8d51b5df63e` |
| Fixed coverage review | `788a17f1edbfdd62ee5d5fa5d5a7b4c57441130ceee1f1f34047b10768751500` |

## Bounded execution and early cost

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
was selected after this result and completes as reported above, without reusing
bounded experience. Retained control
scores70.750 sampled/143.667 greedy remain the benchmark, all0/24 two-wall.
The five-game goal remains unfinished.
