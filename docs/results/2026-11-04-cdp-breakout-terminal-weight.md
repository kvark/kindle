# Terminal-only weighting does not improve the retained Breakout comparison

The fresh200k-action comparison scores **120.542 greedy /66.375 sampled**, below
retained143.667/70.750, with **0/24 two-wall successes** in both modes. Training
takes **21m37.140s**, .72% longer than the control. Greedy initial-subtracted
return/min falls16.94%; its paired interval crosses zero, so this does not establish
reliable harm or equality. The locked improvement rule fails. Retain the existing
sampler and continuation loss with terminal weighting off. No weight sweep,
second learner, extension or automatic successor is selected. The original
864/90%-two-wall and five-game goal remain unfinished.

## Full200k comparison

The [fixed full protocol](../experiments/2026-11-04-cdp-breakout-terminal-weight-200k.md)
keeps12M64/Auto/N8/B16/T16/H15/R16/replay-value1/expert100, base/encoder4e-5,
dynamics4e-4, behavior8e-5, full18/sticky.25/repeat4, sampled collection/posteriors
and20% episode guidance/cap1920. Only terminal continuation rows receive weight10;
continuing rows retain1, ordinary row denominator/global continuation1 and the
existing sampler. Mixture and open-loop supervision remain off. Initialization
is fresh; no bounded weights, replay or experience are reused.

| Measure | Retained | Terminal weight10 |
|---|---:|---:|
|Training actions /updates|200,000 /12,477|200,000 /12,477|
|Executed emulator frames|799,595|799,593|
|Training time|21m27.910s|21m37.140s|
|Frozen greedy final /actual initial|143.667 /2.208|120.542 /2.208|
|Frozen sampled final /actual initial|70.750 /1.417|66.375 /1.417|
|Final two-wall successes, each mode|0/24|0/24|
|Greedy initial-subtracted return/min|6.590|5.474|
|Sampled initial-subtracted return/min|3.230|3.005|

Greedy return difference is−23.125, paired lane95% interval
**[−74.251,30.417]**; greedy gain/min difference is−1.116550,
**[−3.490321,1.371544]**. Sampled return difference is−4.375,
[−10.917,2.167]; sampled gain/min difference is−.225350,[−.528088,.076852].
All intervals cross zero. These10,000-resample/seed0 estimates use eight reused
evaluation lanes; they are not training-seed reliability intervals.

## Exposure and training fit

The sampler stays unchanged. Both learners process3,194,112 world replay rows.
Candidate terminal exposure is5,257 (.164584%), versus5,376 (.168310%) retained;
8,250 candidate updates have no ending versus8,105 retained. Candidate experience
contains256 natural game endings versus263 retained. The loss weighting changes
shared updates and subsequent experience, so these slightly different exposures
are not an isolated sampling effect.

| Training actions | Retained terminal rows | Candidate terminal rows | Retained updates without endings | Candidate updates without endings |
|---|---:|---:|---:|---:|
|1–50,000|1,938|1,971|1,643|1,635|
|50,001–100,000|1,505|1,419|1,891|1,977|
|100,001–150,000|1,106|1,043|2,185|2,250|
|150,001–200,000|827|824|2,386|2,388|

Integer counts are independently recovered from continuation-target means; no
mixture sampling report appears. Mean continuation row weight is1.014813 over
the full run and1.009270 in the final50k actions; each terminal row still receives10.
Counts describe repeated training exposures, not unique examples. The candidate's
weighted continuation loss is not directly comparable with retained unweighted
BCE. This gameplay test does not measure whether ending predictions improved. The subsequent [matched-arrival diagnostic](2026-11-04-cdp-breakout-terminal-weight-arrivals.md) finds terminal posterior MSE.644558→.054219, but intermediate-life posterior MSE.003178→.099410 and more false ending indications.

Late teacher-label cross entropy is.546991 versus.560514 retained on their own
replay distributions, without a frozen gameplay gain. Actual overrides are
86,851/200,000 (**43.4255%**), versus43.0585% retained. The53 completed episodes
with teacher actions average194.642;203 without average15.739. Guided online
returns are not frozen competence. Candidate pre-action scores216–431 contain
4,182 teacher and1,484 native actions; none reaches432, and maximum including
tails is412. These score bands are not verified wall events.

## Full-run verification and closure

Reuse native3672a21a/SHAa8b483a3 and its bounded runner. Packaged teacher/scheduler
bytes match the preceding mixture runtime; its original source/fixture receipt
is reused by exact hashes. Native, runner, guidance-wrapper bytes and the enabled
stable-logit BCE path differ from retained447fe50. Together with reused1009/cohort
and changed experience, this precludes an isolated coefficient or calibration
claim. The scoped teacher qualification is not proof of every possible trajectory.

All370 current/338 retained pins pass, including the exact historical replay-source
resolution. Full helper/comparison checks take3.448s; allfive exact queued commands
pass.729s CPU preflight with zero native/emulator construction. All357 initial
tensors match bounded and retained controls;47 pre-update semantic events agree.
First update376 and residual credit.5 are correct. All12,477 finite reports pass
weighted world/behavior totals; maximum world-total sum error is.000118132.
Inspected CNN, reward, continuation hidden/output, actor and value parameters change.

The100k/6,227 recovery passes **06:47:12 UTC**; independent.552s CPU verification
completes **06:47:26**, preserving357 finite tensors, read-only weights/moments,
all370 pins and the original live identity. A separately prepared independent
closure reader reproduces retained scores/exposure in.305s CPU before candidate
outcomes exist. These co-host CPU costs remain included without correction.

All four frozen arms select24 natural episodes each, with zero updates/truncations
and357 invariant tensors. Actual-initial sampled/greedy traces match799/794
semantic events. Frozen work totals85,800 actions/343,034 frames/55.087s. Full
training pixel/action/label replay takes110.794s including its surrounding audit;
all frozen recorded-action/outcome/frame/boundary replays and videos pass. Coverage,
guidance exposure, curves and comparison pass; the curve is visually checked.
Its final online last50 mean88.56 is distinct from frozen performance.

Construction takes2.395644s. Training learning work is1,151.856s, acting63.591s
and environment stepping62.842s. The.72% total-time difference does not establish
a systematic speed change. All host guards pass, with no recovery/restart.

Source **8ee590c574fadf2a788a9940925f7aa8c3c433ef**, main PID1141191/invocation
`6a601955edc441c98fa39c453d177faa`, launches **06:36:25 UTC November4**.
Main succeeds **07:02:09**, terminal collection **07:02:43**. Independent
**3.346s CPU closure** completes **07:03:46**, verifying53 completion artifacts,
all pins, selected episode returns/outcomes, exposure windows, bootstrap intervals
and the locked decision. Main, recovery and collector are terminal success;
no full-stage failure or repair occurs. Earlier bounded CPU failures/repairs remain
below. No GPU, training or completed replay stage repeats.

Root `/x/Temp/kindle-terminal-weight-20261104/full` is closed. No learner or GPU
allocation is active at closure. Check live state before future work; do not
restart completed stages. Weights, optimizer state, logs, videos and runtime remain.

| Full artifact | SHA256 |
|---|---|
|`declaration.json`|`ef355af4b833e88a74359c3099b58171f9c76ee5d4234547a9e406322d69eb1f`|
|`candidate/learning-review.json`|`c05e5f5fdda429692475a9714193cd697294ce0d8015f1e5376fb38b92aa2585`|
|`candidate/guidance-replay.json`|`2690243b250cb51d896a346d4d21413abac06136178bb3ab63d51cc4cee9cc88`|
|`comparison.json`|`5785ae781123bdfa42957dddb1416a6c888dc923ffc416792e8ae0245edd0617`|
|`completion.json`|`9f16d0ef026473db75b13f2aa9e4a78972ca6eb39b9a1bb99c99d416343c103d`|
|`terminal-summary.json`|`b95bcdad03e27c4741547df3001d16db416b7346627d972afad282c387d57519`|
|`closure.json`|`bdd854a4bd48077a5062ebf0d708ed509132c12aa776f905790f0996da17c67c`|

## Earlier bounded execution check

The optional weight10 loss passes native loss/gradient verification, default-off
compatibility and fresh Breakout training: **8,192 actions /32,722 emulator frames /
489 updates in46.934s**. Its conservative200k projection is **21m00.438s**, below
the30-minute gate. This establishes execution and early cost; the short frozen
capture completes no natural episodes and does not assess gameplay. At that bounded closure, no full
learner, sweep, extension or automatic successor was selected. Retain70.750 sampled/
143.667 greedy and the existing recipe. The original864/90%-two-wall and five-game
goal remain unfinished.

## Change and numerical qualification

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-terminal-weight.md)
uses `terminal_continuation_weight=10` only on true terminal continuation targets.
Continuing targets retain weight1, including intermediate life losses. The ordinary
row denominator and global continuation scale1 remain. The existing sampler is
unchanged; terminal replay mixture and open-loop supervision are off. Other
objectives retain their sample distribution and explicit weights, while shared
parameter updates and subsequent experience can change.

The enabled loss is `mean(w*(softplus(z)−target*z))`, where
`w=1+(gamma−target)*(terminal_weight−1)/gamma`. Omitted weight and explicit1 keep
the old sigmoid/BCE graph. The enabled stable-logit path differs numerically from
older probability BCE, so this is not an isolated coefficient comparison or a
claim about calibrated probabilities. Configuration checks restrict this option
to CDP without open-loop supervision, a horizon greater than1 and finite effective
scales. Restore requires a native runtime advertising the new capability.

Source **3672a21a982fc7d24e4849bbfb4ed032714739d3** passes181 Rust library/5 gym
CPU tests, strict workspace/binding Clippy,18 focused Python adapter tests and
**1,046 packaged Python tests** (10 deselected). The default/explicit1 graph matches
and the weighted graph compiles. A single separately guarded GPU fixture checks
10 cases: row counts1,17,256,513,1024, weights1/10, terminal/continuing targets,
moderate logits and−30/+30. Every logit derivative and the mean loss are compared
with F64, using the production helper composed with global scale.3. Maximum loss
error is **3.939224e−7** and derivative error **1.081654e−8**. Continuing-row
derivatives remain unchanged within the declared tolerance. The fixture takes.78s
inside the test executable; its guard passes. No tolerance is relaxed.

## Bounded learner and compatibility

The fresh recipe stays12M64/Auto/N8/B16/T16/H15/R16, replay-value1/expert100,
base/encoder4e-5, dynamics4e-4, behavior8e-5, full18/sticky.25/repeat4/noops0,
20% episode guidance/cap1920 and seed1009. No prior weights, replay or experience
are transferred. The only selected objective change is terminal continuation
weight10; native/runner versions also differ from retained447fe50.

Before learning, default-off1,024-action frozen compatibility matches128 semantic
events and all357 invariant tensors against the retained final control. CPU
preflight intercepts all three actual commands before native/emulator construction,
checks357 tensor shapes and47 pre-update events, and rejects invalid fixtures.

Fresh training matches all357 initial tensors and47 pre-update semantic events.
The first update is at376 actions, the final count is489 and residual credit.5.
All reports are finite with correctly weighted world totals. Inspected CNN,
reward, continuation hidden/output, actor and value parameters change. Full
recorded pixel/action/label replay, guidance accounting and both host guards pass.
The final1,024-action/4,096-frame frozen capture takes **.882167s**, with zero updates
and all357 tensors invariant; it contains no completed natural episodes.

Construction takes2.018849s. Training includes40.536424s learning,2.512437s acting,
2.565842s environment stepping and.258509s checkpoint work. Retained bounded
training took47.689684s; different native code and subsequent experience mean this
small early difference does not establish a speedup. The projection uses the
fixed formula `46.934137*200000/8192*1.10`, not a full-run timing measurement.

| Early exposure | Retained sampler/control | Terminal weight10 |
|---|---:|---:|
|World replay rows|125,184|125,184|
|Terminal rows|322|345|
|Terminal row fraction|.257221%|.275594%|
|Updates with /without terminals|225 /264|242 /247|
|Executed emulator frames|32,727|32,722|

Counts are independently reconstructed from continuation-target means. No mixture
sampling report appears. Candidate mean continuation row weight is1.024803;
terminal rows each still receive10. These are repeated training exposures, not
unique endings. Actual teacher overrides are3,727/8,192 (45.4956%); the22 online
completed episodes are guided training experience, not unaided competence.

## Failures, process identity and retained artifacts

Three CPU qualification failures remain preserved. The first focused Python
invocation imports an older installed package missing a guidance module; the
qualified runtime passes unchanged assertions. After the successful GPU fixture,
a CPU reader looks for child exit fields in the guard audit summary rather than
`result.json`; the separate reader correction completes without repeating the
fixture. The first bounded CPU rejection helper does not catch the expected
argparse `SystemExit(2)` for zero weight; its original source/failure remain, and
the corrected helper verifies that exact rejection. No GPU stage or completed
learning/replay stage repeats, and no acceptance tolerance changes.

The main service `kindle-breakout-terminal-weight-check-20261104` launches
**06:28:14 UTC November4**, PID1140605/invocation
`21d02c9b5f434a01905f29759fe3b3e4`, with298 pins. It completes **06:29:25**;
independent collection completes **06:29:36**. All298 current and338 retained pins
pass, with the exact historical replay-source resolution preserved. A separate
**1.782s CPU closure** completes **06:30:07**, verifies66 artifacts and independently
recounts exposure and the cost projection. Closure status is
`complete_with_cpu_preflight_repairs`; the main and collector both succeed.
No learner or GPU allocation is active at closure. Check live state before work;
do not restart any completed stage.

Root: `/x/Temp/kindle-terminal-weight-20261104`. Use its `runtime/` with matching
runner; wheel and `validation/kindle-tests` remain. Native SHA256 is
`a8b483a3229e1c57c39d8742c36a3dc12edefcb8bc7c705306c9bf32bc85f8e2`.
The build cache is still retained; the immutable build receipt records that state.

| Artifact under root | SHA256 |
|---|---|
|`build-receipt.json`|`47eef2c57f42ce267d2d7e37a711ada354d2aa87a10aad2b1a8cf1db171bb2c4`|
|`numerics/review.json`|`42ce67214a8cbdd89149a6bcac6413a62797f01a4e0f6b1f84d71fecd1be75d5`|
|`bounded/declaration.json`|`73a27ff94c118307829d9e91d1a9b6ff2ce0dd3e5a3b146b817de3e36d3fd502`|
|`bounded/check/review.json`|`fff0f07b032768d4430dda9e48781f4e5332c5ed747cb8033ecf6dfc9c0c6933`|
|`bounded/check/completion.json`|`607448f8501800b41a80ccc7ad6968c6e6e7af5c2fada8b76d81990eb471ff5c`|
|`bounded/check/terminal-review.json`|`432d570fae4d9be9a415864d8bfbadf7ed65ebfa85e3de9069e67c6ddaf28432`|
|`bounded/closure.json`|`afc2718a3b0879c553d36125403fa4cebd1d4e87eebc4c9409eac7833659a42f`|
