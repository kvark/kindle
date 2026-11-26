# Position loss at encoder4e-5 lowers sampled Breakout quality

The fresh200k comparison finishes **200,000 actions /799,590 frames /12,477
updates in21m47s**, within its fixed30-minute cap. Sampled return falls
**56.292→35.542 (−36.86%)**; greedy rises **104.667→112.750 (+7.72%)**. Both
candidate modes still havezero of24 two-wall successes. Training takes1.50%
more observed wall time than the retained control's21m28s.

| Frozen mode | No-centroid control | Centroid100 / encoder4e-5 | Own actual initial, both | Two-wall successes, both |
| --- | ---: | ---: | ---: | ---: |
| Sampled, primary | 56.292 | 35.542 | 1.417 | 0/24 |
| Greedy | 104.667 | 112.750 | 2.208 | 0/24 |

Retain the existing no-centroid recipe. The sampled-primary result declines,
and the separate greedy gain does not establish an overall win or task success.
The original90%-two-wall/864 gate remains unchanged. No extension,coefficient
sweep,cap increase,confirmation or automatic successor is selected. The five-game
goal remains unfinished.

## Recipe and learning interpretation

Use12M64/Auto/N8/B16/T16/H15/R16/expert100,base/actor/encoder4e-5,dynamics4e-4,
centeredCDP/action-effects1,sampled collection and posteriors,actor-unimix.01,
unit expert weights/world gradients,all-state corrective labels,20% episode
guidance/cap1920,full18/sticky.25/repeat4 and one GPU resize from native pooled
RGB. Add the existing scale100 current-pixel centroid objective. Its detached
red-excess targets use the fixed color/crop prior;the target and656,386-parameter
head are training-only. Total trainable parameters are16,990,739.

The qualified native2974160/SHAe7ba0d4 and byte-identical centroid runner are
reused without a build. Native and runner differ from the447fe50 control;
matching initial values/traces does not prove full learning parity or isolate
the loss's causal effect. The teacher and schedule hashes match the control.
Learner1009/evaluation4000001009 are reused development seeds. No weights,
replay,beliefs,RNG,game state or bounded-check experience are reused.

Raw centroid loss falls from.044301 in the first50k window to.0006446 in the
last50k;valid-target fraction rises.8363→.9146. These are changing replay-training
statistics,not held-out position accuracy or a matched-state representation test.
Late expert CE.559106 is nearly the control's.558097. The auxiliary target is
learned without an improvement to the primary gameplay result;neither loss
curve identifies why sampled and greedy results differ.

Actual guidance overrides87,259 actions (**43.6295%**),versus99,343 (49.6715%)
in the control. Aided online last50 return84.68 is not frozen competence.
Initial-subtracted return per training minute is1.567 sampled and5.074 greedy.
Retain CPU co-host time and native/runner differences without wall-time correction.

## Complete verification and closure

All541 declared inputs,372 bounded-matching initial tensors,357 shared control
values and47 pre-update semantic events pass. Final/actual-initial sampled and
greedy assessments use the first three natural episodes per eight lanes. Their
actual-initial semantic sequences match the control exactly:800 sampled and795
greedy events. All372 tensors remain invariant in every frozen arm. Finite
supervised/auxiliary reports and changed CNN,reward,actor,value and both centroid
weight layers pass.

Four frozen arms total80,320 actions/321,115 frames/73.707s,zero updates andzero
truncations across the complete arms. Full training pixel/action/teacher-label
replay,frozen outcome/frame/reset replays/videos,host guards,curves,exposure and
comparison pass. Ordinary frozen logs lack live image hashes,so frozen replay
does not compare unavailable original images. The comparison plot is visually
inspected and its labels/values match the completed review.

The100k/6,227 recovery audit passes11:26:28 UTC. All four checkpoint files are
retained under `recovery-100k`, totaling213,946,193 bytes;the.190s independent
retained-file/configuration confirmation passes11:27:05. No checkpoint is selected
for evaluation from this recovery snapshot.

The original service starts11:15:21 UTC November1/source`cd92f9f`,PID787678 /
invocation`709367befab94110b4afc489bd51d55c`. Main/postprocessing/collection finish
**11:41:19/11:41:26/11:42:02**. All five original services succeed; no original
pipeline or GPU stage fails or repeats. No learner/GPU allocation is active at
closure;check live state before any new work.

A separate terminal CPU verifier fails at11:42:27 because it requires a systemd
Consumed summary for the lightweight progress observer. Its source,stderr and
.214s execution receipt remain preserved. A separate corrected verifier accepts
the exact worker-completion report,checks original invocation identities and
passes in **1.553s at11:43:44**. The progress timestamp11:41:38 is explicitly a
worker-completion report,not an exact process-exit timestamp. The original
completion review is unchanged;this is an ancillary verifier repair,not a repaired
training or evaluation run. Do not rerun either verifier or any completed stage.

The provisional training/frozen log receipts remain unchanged beside the final
reviews. CPU co-host caveats include startup/recovery reviews,the.314s Qbert phase
analysis wrapper,the.514s Qbert linear-readout wrapper,and light source/document
preparation. No time is subtracted from observed training duration.

## Bounded execution and qualification

The preceding fresh check completes8,192 actions/32,722 frames/489 updates in
47.291s. Its1,024 frozen actions take.883s withzero updates andno natural episodes.
All302 pins,372 matching centroid initial/invariant frozen tensors,357 shared
control tensors,47 pre-update events,finite reports,full replay and guards pass.
Main/collector succeed11:03:56/11:04:29 UTC November1/source77866fc,withzero
failure or repeated stages. The conservative200k projection21m10s passes30min;
that early projection does not guarantee the full-run cost or task quality.

The bounded CPU preflight takes2.821s. Full qualification then passes five
commands,18 configuration cases,seven startup cases,four observer argument
cases,five recovery cases,372 schema tensors and fixed schedules in2.610s.
Plot/import/exposure qualification takes.698s. All201 historical build-source
hashes resolve. Stale collector names and explicit recovery retention were
corrected before the first full qualification;no native rebuild occurs.
Earlier centroid numerical/schema/full-evaluation failures remain preserved.

[Bounded protocol](../experiments/2026-11-01-cdp-centroid-encoder4e5.md),
[fixed full protocol](../experiments/2026-11-01-cdp-centroid-encoder4e5-200k.md).
Package `/x/Temp/kindle-centroid-encoder4e5-20261101`; full trial
`full/breakout-12m-64-b16-r16-centroid100-encoder4e5-expert100-200k-seed1009`.
The bounded trial and all full stages are terminal;do not restart them.

| Full artifact | SHA256 |
| --- | --- |
| declaration.json | `b2126cedf8c9794096b3fc660fbc59eb20e11dd48ee7657d44360e6f2761bda3` |
| learning-review.json | `6ddc5f2fdad27ae71eefc5a583b39a139c0ad4297afdd90c7ec60d498b41a248` |
| postprocess-review.json | `3e3437900a6f31cf7be9013868b524ba63798d1209f4ea4a258dc5d50322b4b0` |
| completion-review.json | `72330f4ba4a22ec1f2e39b1651ce1507641ca6f8f208aa9e529a2c0afdefcc83` |
| exposure-comparison/comparison.json | `953e43cf45a1ec9dc403feeaad61483d2151f431e61cdeb70c43a5219f916802` |
| terminal-confirmation.json | `fd9b6429529867a9f4ca3f8b42d12b265df730ef7b1c85808158f4a2daaebe6b` |

Bounded declaration `bf7e648d7febfa2f1ba191c9866137f2d08c18a7afb156e9d410a510e45710ef`,
review `43a5482f9fb9fb23592442181ece5e93d20e656bb720e6b02671b3e55dc57059`,
completion `bb953beb006da1c7efd5a5b7e7dfcaa4becbe559bb27e5523bb383d28337396b`.
