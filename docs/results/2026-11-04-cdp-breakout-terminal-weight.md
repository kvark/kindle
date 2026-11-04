# Terminal-only continuation weighting passes its bounded execution check

The optional weight10 loss passes native loss/gradient verification, default-off
compatibility and fresh Breakout training: **8,192 actions /32,722 emulator frames /
489 updates in46.934s**. Its conservative200k projection is **21m00.438s**, below
the30-minute gate. This establishes execution and early cost; the short frozen
capture completes no natural episodes and does not assess gameplay. No full
learner, sweep, extension or automatic successor is selected. Retain70.750 sampled/
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
