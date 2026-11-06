# Prior-reward supervision passes the bounded cost gate

The [optional prior-reward loss](../experiments/2026-11-06-cdp-breakout-prior-reward.md)
passes corrected CPU qualification, packaging, two guarded native fixtures and
a default-off frozen restore. The original CPU graph assertion failure and
subsequent CPU reporting failure remain preserved with separate repairs. No GPU
stage repeats. The fixed8,192-action cost check is complete and passes. A separately declared
[full200k comparison](../experiments/2026-11-06-cdp-breakout-prior-reward-200k.md)
is running. No gameplay benefit is established.

The new loss applies the existing reward head to sampled one-step prior states
before the arrival image enters the posterior. It uses the same real reward
targets, masks episode starts and averages over all training rows. Gradients
reach prior probabilities, dynamics and earlier states. The coefficient is
independent of the ordinary posterior reward scale. The default is off; scale1
is selected for the bounded check. No parameters or inference inputs are added.

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

## Runtime and remaining work

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

The bounded allocation is closed. The [fixed200k comparison](../experiments/2026-11-06-cdp-breakout-prior-reward-200k.md)
is running against matched22003 with actual-initial controls and four24-episode
frozen arms. CPU helper qualification and all five actual commands pass; startup
review passes. Original main PID1562317/invocation
`3fbc09cb5f2947459016bc52a123e6f8`, execution source`dda5bf6f`, collector and
recovery observer are active. Root is`/mnt/data/kindle-breakout-prior-reward-20261106/full`.

Before launch, static review found a copied recovery-verifier service name still
pointing to the closed reward10 allocation. The original declaration, verifier
and passing preflight remain in`preparation-revision1`; the correction receipt
records a CPU fixture of the actual systemctl command. After correcting the
literal and repinning, all five actual commands pass again in2.848325s, without
native/emulator construction. This preparation correction causes no GPU repeat.
Retain the current recipe pending the full comparison's outcome.
Original864/90%-two-wall and the full five-game goal remain unfinished.
