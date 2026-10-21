# Current gap is1.85×; tiny-gradient optimizer qualification fails

**JAX rebaseline passes; native optimizer correctness needs attention before
another performance experiment.** Production and Atari remain stopped. No GPU
wedge, recovery or permission blocker. [Compact evidence](2026-10-21-driver615-baseline.json).

## Fresh comparison on615.71.09

| RGB12M/B32/T16/H15/BF16 | Complete update |
|---|---:|
| Preceding native control |48.281ms|
| JAX completed core |25.606ms|
| JAX caller-inclusive |26.066ms|
| Native / inclusive JAX |1.852×|

The unchanged pinned JAX runner uses upstream`e3f02248`, JAX0.6.2, cuDNN9.24,
the original resident fixture/seed701 and16 warmup+256 measured updates. All272
reports and288 initial/final tensor fingerprints match the historical run exactly.
All940 static fusion bodies,112 library-call configurations/physical shapes,
reachable heavy work and24 command-buffer definitions match. Ten semantic-reader
regressions pass. These checks do not prove identical driver machine code.

Inclusive first/last32 means26.038/26.078ms retain all samples. Seed/metric handling
costs.360/.100ms; final deferred2,631,808-byte drain takes.211ms outside the retained
timing boundary. One guard, zero new warnings,272 synthetic updates, zero actions.
JAX review finishes06:42:16UTC. No package or source refresh.

Native and JAX still use different weights/RNG/fixtures: matched-shape throughput,
not numerical or learning parity. The historical580 ratio1.58× remains historical.
Cross-session timing differences do not isolate the driver's effect. No native
retime, new hardware counters or updated pure-kernel latency claim.

## Why the next native profile was not started

A CPU transfer check compares the retained cached-timestamp observer's ordinary
control with the new native control. All four programs, constants, pipeline keys,
shader-source multisets and actual world groups match. **Numerical equality fails.**
The intervening upstream-refresh control on580 reproduces the old observer's
states/reports exactly, so that refresh alone does not account for the difference.
No assumption of current observer qualification or automatic profiling follows.

The first posterior deterministic/stochastic features and eight of nine exported
target arrays match exactly. Only686/7,680 imagined weights differ (relativeL2
.0005344). The first world report differs only in continuation loss by one F32 ULP.
All101 stored learner-parameter tensors remain exact after update0; that update
has zero learning rate under warmup. However55 first-moment and55 second-moment
tensors differ, plus three slow-critic tensors (maximum1.49e-8). Moments are not
parameter errors. After272 updates,100/101 stored learner-parameter tensors differ;
their maximum absolute difference is.00260055. This does not establish a faulty
new-driver model, nor permit an exact cross-session trajectory claim.

Historical first-step LaProp states contain four momentum elements exceeding
.1001, maximum29.2403, all with zero stored second moment. Initial moments and
optimizer step are zero; first checkpoint step is1. The current model fixture
has no such oversized first-step entries. This observation motivated a separate
component test rather than assuming the driver fixed or caused the behavior.

## Reproduced current-driver failure

Use the preserved native library, no backend edit: loss
`mean(parameter * input)` supplies16 explicit gradients, with mean scaling
cancelled exactly. Native raw gradients match all16 supplied values. With
beta1=.9, beta2=.999 and epsilon1e-20, six signed tiny-gradient cases fail:

| Gradient | F64 expected second moment | Stored second moment | Expected momentum | Actual momentum |
|---|---:|---:|---:|---:|
|3e-18|about9e-39|0|about.099668|30.000008|
|1e-18|about1e-39|0|about.099010|10.000003|
|2.924e-18|about8.55e-39|0|about.099659|29.240005|

Negative counterparts fail symmetrically. These positive second moments are
representable F32 subnormals. With variance zero, normalization divides by
epsilon instead of the intended RMS scale. In this component probe's explicit
lr=.1 case, parameter1 becomes about−29 for gradient3e-18, rather than about.90033.
This is not the production learner's learning rate or an observed game update.

The first-step real-arithmetic momentum bound is1-beta1 when moments start at
zero. The independent F64 recurrence checks variance with relative1e-5 plus two
minimum-subnormal ULPs, and momentum/parameters with1e-5*(1+abs(reference)). Six
cases fail all three checks; normal-scale cases pass. No tolerance is relaxed.
Pinned Dreamer also places epsilon outside sqrt: moving epsilon is not the fix.

The original probe stops at variance index9 before exporting arrays. Its failure
is retained. One separately declared diagnostic execution exports after the
existing readbacks, then fails the **same unchanged assertion**. Both guards
remain failed, with zero allocation warnings/Xids/recovery or unfinished children.
Two standalone optimizer steps total, no Kindle learner/game actions. The intended
second-step/cache-replay qualification is unrun. CPU observation review finishes
06:57:50UTC. No full-model independent oracle or JAX tiny-gradient comparison yet.

## Next

Qualify a clean GPU-side solution for F32 optimizer underflow before resuming
performance work. First compare this explicit tiny-gradient case with the pinned
JAX optimizer and inspect the emitted float-control contract; distinguish valid
default flush behavior from a driver/specification defect. Do not assume a driver
bug or infer native SASS from source. Latest Blade`eae0c6f7` preserves **F16**
subnormals; it does not establish F32 preservation for this optimizer. Latest
Meganeura`a75c38e` qualification/search changes are not this repair.

Keep epsilon/model/precision semantics fixed initially; no CPU workaround, silent
source refresh, wholesale learning-result invalidation or automatic promotion.
The defect could affect stability, but its contribution to historical game scores
or the JAX performance gap is not measured. Both goals remain open.

Artifacts: `runs/driver615-jax-20261021.MBKwxsfk` contains declarations, all source
receipts, raw JAX outputs, failed observer-transfer result, three-way state delta,
moment census, both failed optimizer probes and raw arrays. All workers finish.
