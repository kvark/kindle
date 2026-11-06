# Prior reward raises point scores but misses the fixed retention rule

The [full200k comparison](#full200k-comparison) scores79.167 greedy/47.000 sampled
versus matched75.250/44.708, with nearly unchanged training time. Both point
estimates improve, but both paired gain/min intervals span zero. Keep optional
prior reward off and ordinary posterior reward1; no gameplay benefit is established.
The [bounded cost check](#bounded-learning-closure) passes. Native qualification,
prelaunch correction and publication-reader failures remain preserved with their
separate repairs; no GPU stage repeats.

The new loss applies the existing reward head to sampled one-step prior states
before the arrival image enters the posterior. It uses the same real reward
targets, masks episode starts and averages over all training rows. Gradients
reach prior probabilities, dynamics and earlier states. The coefficient is
independent of the ordinary posterior reward scale. The default is off; scale1
was selected for the bounded check. No parameters or inference inputs are added.

This differs from the closed [posterior reward10](2026-11-06-cdp-breakout-reward10.md)
and [joint H3](2026-11-03-cdp-breakout-open-loop.md) failures. Those failures remain
relevant and neither supports a causal claim about the current reward model.

## Qualified evidence

- 33 distinct CPU Rust tests pass in35 executions: config round trips and rejection,
  graph dependencies/gradient routing, parameter/inference invariance and both
  ordinary/staged graph compilation. Both strict workspace and binding Clippy pass.
- 47 targeted CLI cases pass against both the compatible CPU wrapper and the new
  packaged runtime, with zero native-agent or emulator construction.
- The masked255-bin CE fixture covers1/17/256/513 rows, extreme logits, two-hot
  targets and masked rows. Maximum F64-reference loss error is1.101963e-6;
  maximum logit-gradient error is5.839967e-10, within the fixed tolerances.
- The tiny full graph sends finite nonzero derivatives through the reward head,
  recurrent dynamics and prior. Changing the arrival image/posterior does not
  change its loss. Masking all rows makes both loss and all parameter gradients
  exactly zero. The reward output is deliberately nonzero for this routing
  fixture because ordinary fresh initialization would suppress upstream gradients.
- The default-off seed22003 restore executes1,024 actions/4,096 frames in.901719s
  of runner time, with zero learner updates. All128 transition/reset semantic
  events match the retained prefix and all357 tensors remain exactly unchanged.
  Its fresh config serialization adds only the explicit legacy default
  `replay_value_bootstrap=imagined`. This is scoped frozen compatibility evidence.

The native fixtures complete in3.679641s including guard orchestration; all277
pins pass. The legacy queue succeeds in5.869928s; the corrected CPU reader takes
1.479940s and verifies502 pins. Device identity is RX7900XT/RADV, boot
`a8648671-8577-4f42-9b46-a4345d3ac297`, amdgpu module`9BDBC866F55BF296F1B7FA2`.
All three host guards pass, with no recovery or NVML polling. There are seven
synthetic forward/backward calls, zero learner updates and no natural-episode
score assessment. These checks do not establish useful policy gradients or
broad equivalence with the retained training runtime.

## Preserved preparation failures

Source`69e48ef6` compiles, but its CPU graph fixture fails when it assumes a
detached gradient must have a different shape from its parameter. The scalar
continuation bias and scalar zero gradient both have shape[1]. The separate
test-only repair at`a89b0b8e` requires an explicit zero constant for detached
parameters; production code and native tolerances are unchanged. Original source,
logs, return101 and the repair explanation remain in`cpu-fixture-failure`.

The default-off GPU run and all comparisons succeed, but the original CPU reader
then requests an absent`accounting.emulator_frames` key while writing its report.
The separate reader sums the recorded`run_end.executed_action_frames` instead,
rechecks pins and comparisons, and writes`legacy/repaired-review.json`. The
original reader/error and successful GPU queue remain unchanged. There is no
second restore or repeated GPU stage.

## Runtime

Root `/mnt/data/kindle-breakout-prior-reward-20261106`. Packaged native source
`a89b0b8e47d3e869ce373934280130a4d44e46f6`, native SHA256
`19b6f46c93cdc331bbb1fe7a2775a379aa0335a3681676be503f3acd47358c60`;
validation executable SHA256
`eef8945c46154e689857ceb2b21993b6db15b61a1c99c1d16728a10ce88edd95`.
The wheel, runtime and validation executable remain retained. The disposable
build cache is released with a separate receipt; the original build receipt stays
unchanged. `qualification-review.json` seals the evidence and both failures.

## Bounded learning closure

The fresh seed22003 check completes8,192 actions/32,725 frames/489 updates in
46.736739s, with first update376 and residual credit.5. Frozen restore executes
1,024 actions/4,096 frames in.897090s with zero updates and357 invariant tensors.
All357 initial tensors and47 pre-update events match retained22003. CNN, reward,
continuation, actor and critic parameters change. All finite reports, full
pixel/action/label replay, weighted world/behavior losses and host guards pass.

The auxiliary CE mean is4.378472; valid fraction mean.997835 and minimum.98828125,
with271 masked replay rows. Fractions pass the256-row lattice and bounds derived
from independently reported labeled rows. Sampled row identities are not logged;
these checks do not establish exact per-row masking independently of the native
fixtures. Actual guidance overrides4,412/8,192 actions (53.8574%); the20% setting
selects episodes. Guided returns and these short frozen actions are not competence.

The fixed conservative200k estimate is46.736739*25*1.10=1,285.260324s
(21m25.260s), below30 minutes. All783 allocation pins,338 original/90 matched-seed
pins and69 matched completion artifacts pass. The CPU actual-command preflight
passes in2.435385s without native/emulator construction. The separate runtime
receipt pins the qualified native plus exact retained Python adapters/teacher;
only the prior-reward option is added to the retained runner. The previous
native/compatibility checks are reused, not repeated.

MainPID1561744/invocation2cba4f3fdf6442ba8431da140ef7959a and collectorPID1561747/
invocationb2315f7db1b44fe58ca15286d2d4de9c both finish successfully. Main/collector
complete22:38:55.266/22:39:06.552 UTC November6; independent closure completes
22:39:54.108 in6.152240s, source5453b914. It recounts489 reports, both exact tensor
comparisons, initial events, loss arithmetic and cost;86 artifacts are sealed.
Maximum weighted world-total error is.00011715 within the fixed relative tolerance.
No bounded preparation/run failure, repeated GPU stage, learner or collector.
Earlier qualification CPU failures remain preserved above.

The bounded allocation is closed. The separately declared full comparison is
complete below; no bounded learner or successful qualification stage repeats.

## Full200k comparison

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-prior-reward-200k.md)
adds only prior reward CE scale1, with posterior reward1, against the retained
seed22003 control. Native/runner bytes change as declared; original teacher and
Python adapters remain exact. The full candidate completes200,000 aggregate
actions/799,622 emulator frames/12,477 updates in **21m33.955s**, versus21m30.822s
for the control, a.243% time increase. First update376 and residual credit.5 pass.

| Frozen measure | Prior reward1 | Matched control |
| --- | ---: | ---: |
| Final greedy |79.166667|75.250000|
| Actual-initial greedy |2.291667|2.291667|
| Greedy gain/min |3.564653|3.391251|
| Final sampled |47.000000|44.708333|
| Actual-initial sampled |1.875000|1.875000|
| Sampled gain/min |2.092422|1.990980|
| Final two-wall successes, either mode |0/24|0/24|

The primary greedy gain/min rises5.113%, but its paired difference is.173402 with
95% interval[−1.322212,1.717042]. Four of eight lane gains improve. Raw greedy
return difference3.916667 has interval[−28.291667,37.167708]. Sampled gain/min
rises5.095%, difference.101442 and interval[−.258723,.496962]; raw difference
2.291667 has interval[−5.458333,10.792708]. The fixed10,000-resample eight-lane
bootstrap uses NumPy seed0. Both intervals crossing zero means the candidate
fails the declared retention rule despite higher point scores. Keep the option
off. This reused development seed/cohort does not assess broad reliability, and
no causal reward-model improvement follows. Older1009/cohort143.667/70.750 is
not the comparator;1009's94.625/70.500 on this cohort remains context only.

All357 actual-initial tensors and47 pre-update events match retained22003;
initial sampled/greedy semantic traces also match. Each of four frozen arms
selects24 natural episodes, three per lane, with no aids, labels, updates or
truncation and357 invariant tensors. All pixel/action/outcome/boundary replays
and videos pass. Initial,100k and200k checkpoints remain retained.

Full training replay completes in107.675s and reconstructs252 completed episodes,
every action, reward, frame and teacher label. Actual overrides are76,208 actions
(38.104%); repeated labeled imitation exposure totals2,989,630 rows. Training
reaches maximum cumulative score417, with1,276 native actions in216–431 and
none at432 or above. These guided training statistics are not frozen competence.
Learning takes1,148.966s; Atari stepping62.744s. Mean prior reward CE is.275464,
with5,156 masked replay rows and mean valid fraction.998386. All loss arithmetic,
mask lattice and labeled-row boundary bounds pass; sampler row identities are
not logged, so these bounds are not an independent exact per-row reconstruction.

Execution source`dda5bf6f` launches22:46:47.382 UTC November6, original main
PID1562317/invocation`3fbc09cb5f2947459016bc52a123e6f8`. Independent100k/6,227-update
recovery passes22:57:49.284 in1.700331s with357 finite tensors, matching weights/
optimizer moments and854 unchanged pins. Directory/files are0555/0444 and the
original learner remains active throughout. This does not preserve replay,
live games/beliefs or full RNG state; observer time stays in measured training.

The original main finishes23:12:36.975 UTC and terminal collection records
completion23:13:06.762. Independent closure finishes **23:13:49.902** in5.507148s,
recounting192 candidate/control frozen episodes, raw and gain/min intervals,
all12,477 learner reports and loss totals. All854 allocation/338 original retained
pins,53 completion artifacts and28 checkpoint files pass. Maximum total errors
are.000117150 world/.000032490 behavior within declared tolerances. All five
GPU children and three original services exit successfully; no render holder
remains. No learner, GPU stage, collector or completed diagnostic repeats.

Root `/mnt/data/kindle-breakout-prior-reward-20261106/full` is closed. No extension,
coefficient sweep, fresh confirmation or successor is selected. Original864/90%
two-wall and the full five-game goal remain unfinished.

## Preserved full-preparation and publication corrections

Before launch, static review found a copied recovery-verifier service name still
pointing to the closed reward10 allocation. The original declaration, verifier
and passing preflight remain in`preparation-revision1`; the correction receipt
records a CPU fixture of the actual systemctl command. After correcting the
literal and repinning, all five actual commands pass again in2.848325s, without
native/emulator construction. This preparation correction causes no GPU repeat.

After successful independent closure, a read-only publication extractor fails
on a journal record whose MESSAGE is null. Original source, output, exception
and return1 are preserved in the parent's`publication` directory. A separate
reader skips non-text journal records and succeeds in.013066s. It changes no
experiment artifact, executes no model and does not rerun any completed stage.
The original failure remains failed; the independent closure was already passed.
