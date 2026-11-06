# Optional stride-two RGB encoder: completed negative learning comparison

The [default-off experiment](../experiments/2026-11-06-cdp-rgb-stride2.md)
implements `rgb_stride2` and fresh CLI `--rgb-stride2`. Four stride-two5×5
convolutions replace stride-one convolution plus max-pool in the encoder.
Parameter names/shapes/count, observation/replay geometry, embedding width and
latent/control dimensions remain unchanged. Decoder graphs remain exact.

The encoder convolutions compute one quarter of their former output positions;
no encoder max-pool follows them. This changes sampling and nonlinear reduction,
so behavior can change. The operation count alone does not establish speedup.
The fresh200k comparison below regresses both frozen modes despite9.722% less
training time. Keep the option off. This is one reused development seed and a
historical timing comparison, not a broad reliability or isolated-cause result. Native max-pooled
RGB and the existing single GPU resize remain; no labels or emulator state enter
the model. The earlier pixel diagnostic already recovers precise ball positions
fromRGB64, so resizing alone is not a sufficient explanation of feature errors.

## Completed CPU checks

| Stage | Result | Wall time |
|---|---|---:|
| Initial Rust compile |Fails in the new standalone test fixture: missing imports|25.307s|
| Repaired Rust suite |216 Kindle +5 adapter tests pass;125 native/reference tests ignored|13.834s|
| Python checks |1,246 pass,10 constructor checks deselected;zero native construction attempts|4.923s|
| Workspace/all-target Clippy |Pass, warnings denied|16.305s|
| Binding/all-target Clippy |Pass, warnings denied|7.380s|

The initial failed fixture and its receipt remain retained. Source review also
corrects an overly broad fixture assertion: posterior graphs include the changed
encoder, so their parameter/output shapes match, while their operations differ.
Imagined dynamics and behavior graphs still match exactly. These repairs change
tests only; production implementation and tolerances remain unchanged. The
original failed stage is not relabeled passed.

CPU evidence covers config defaults/roundtrip and invalid encoders;32/48/64/96
geometry; unchanged parameter shapes and control/decoder graphs; quartered CNN
output elements; absence of encoder max-pooling; CDP/reconstruction backward
graph compilation and all12 encoder gradient owners. An independent scalar
fixture checks nonconstant spatial sampling, channel order and every token.

Fourteen new CLI cases cover fresh settings, default behavior, native capability,
restore preservation and fresh-override/evaluation rejection. Python tests use
current Python sources with retained native7704d25c for defaults; the new capability
is mocked where required and all native constructors are forbidden. This is not
qualification of a newly built package. Existing CPU adapter tests may exercise
emulators; no native learning or GPU work occurs.

## Completed native qualification

Two new ignored fixtures compile: TinyRGB64 encoder values/loss and four selected
parameter derivatives against an independent F64 strided reference; two synthetic
CDP updates with exact parameter/optimizer/config restore. Both pass once under the
[guarded native protocol](../experiments/2026-11-06-cdp-rgb-stride2-native.md):
encoder.77s, synthetic learner/restore.58s; complete guarded controller3.568s.
All266 pins remain unchanged. Both children exit0 and are gone, guards pass,
and no render holder or Kindle service remains. Zero Atari actions.
Existing native resize, CNN and learner fixtures are not selected for repetition.

Release test compilation passes in63.677s and wheel build in6.935s. Installed
package checks pass1,246 cases with10 deselected and zero native construction
attempts in4.572s. Source `22c262ea`, native SHA256
`cde123e1aa0d68f61cd6111fe392d605d2ae6864de39b4349a139c82301f6b3e`,
test executable SHA256
`da5dcc9901bba568e260d0db99f5546f3c373c0c163223e944ab4930a1687374`.
The receipt pins260 source files and all packaged artifacts.

Root `/x/Temp/kindle-rgb-stride2-20261106` retains source pins, exact commands,
logs and the original failed fixture. Native qualification is complete.
The [default-off compatibility protocol](../experiments/2026-11-06-cdp-rgb-stride2-legacy.md)
passes both frozen1,024-action traces: retained.860575s and actual-initial
.880874s,4,096/4,094 emulator frames respectively, zero learner updates. All128 transition
rows per arm match saved controls except wall timestamps; all357 tensors match
exactly. All40 pins and both guards pass, children are gone. Queue11.700s and
CPU review.947s; no stage repeated.

The [bounded cost protocol](../experiments/2026-11-06-cdp-rgb-stride2-bounded.md)
completes8,192 fresh actions/489 updates plus1,024 frozen actions once. CPU preflight
passes in.167s, exercises both actual queue commands with zero native/emulator
construction, and retains exact357-tensor initial/frozen checks. Only the
encoder stride option changes the retained learning config. The declared gates are25min padded200k and2GiB conservative replay headroom.
All packaged artifacts and the exact original native-pinned build receipt remain
retained after disposable build-cache release. These checks do not by themselves
select a full learner; the separate learning declaration follows below.


## Completed bounded cost check

Execution source `edb702087ac3aaa2899db8c75aac3444f9d88559`, same packaged
native `cde123e1`, current matching runner, original teacher and retained config
with only `rgb_stride2=true`. All357 initial tensors exactly equal the retained
actual-initial checkpoint. Changed encoder operations can change initial actions;
no pre-update trajectory identity is required or claimed.

| Measure | Result |
|---|---:|
| Fresh training |8,192 aggregate actions /32,719 emulator frames /489 updates|
| Training wall time |41.940327s versus historical47.689684s;12.056% less|
| Throughput change |13.708% more actions/s; single historical comparison|
| Agent construction |2.573873s, separate from training elapsed|
| Padded200k projection |1,126.327s =18m46.327s; passes25min cap|
| Conservative full-replay headroom |12,527,583,232 bytes =11.667GiB; passes2GiB cap|
| Frozen execution |1,024 actions /4,096 frames /.827412s;zero updates|
| Complete controller |61.002771s, including construction, guards and CPU audits|

All489 learner reports are finite and pass independent loss-sum checks. Encoder,
reward, continuation, actor and value weights change. All357 final/frozen tensors,
moments and optimizer counts match exactly. Independent replay verifies every
training action/reward/frame/terminal, all8,192 teacher recommendations and3,836
actual overrides; frozen action/reward/frame replay also passes. All51 bounded
pins and260 source pins match. Both guards pass; all six native/legacy/bounded
children are terminal and gone, with no render holders or running Kindle services.
No GPU stage repeats. Original CPU fixture failure and exact original bytes remain.

### Remaining measured cost

| Learner stage | Saved control | Stride-two |
|---|---:|---:|
| Posterior inference |6.209s|4.469s|
| World training |13.613s|10.293s|
| Imagined rollouts |16.647s|16.754s|
| Behavior training |2.502s|2.519s|
| Total reported learner time |40.438s|35.455s|

Most saved learner time is in posterior inference and world training, which use
the encoder. Imagination is unchanged by this encoder option and now accounts
for47.256% of measured learner time. These are instrumented stage elapsed times,
including synchronization/host work, not isolated GPU kernel durations. Different
sampling/trajectories and historical execution limit causal timing claims. Computing one quarter as many convolution output positions does not
imply four times faster training, and shorter training does not establish better frozen competence.

Qualification closes at **17:26:13.324 UTC November6**. Root retains
`qualification-closure.json`, native/legacy/bounded declarations and reviews,
all logs/checkpoints/replays, `bounded/timing-comparison.json`, packaged binaries
and original failures. Closure verifies266 native,40 legacy,51 bounded and260
source pins; the native-pinned pre-cache-release build receipt is resolved to its
exact archived copy. The3.9GiB disposable build cache is gone.

The cost/memory gates pass. No full learner is selected by these stages. Any
separately justified fresh fixed-budget comparison must judge frozen gain per
training minute against its own actual-initial control and report both modes;
it cannot reuse check experience or claim an independent seed. Retain the current
RGB64 teacher/recipe and143.667 greedy/70.750 sampled until useful learning is
measured. Original864/90% two-wall and five-game gates remain unfinished.


## Completed fresh200k comparison

The [separate fixed-budget protocol](../experiments/2026-11-06-cdp-breakout-rgb-stride2-200k.md)
selects one fresh200k-action/12,477-update learner. The measured12.056% bounded
cost reduction warrants testing frozen learning efficiency; no gameplay benefit
is inferred from speed alone. Only `rgb_stride2` changes the retained recipe.
Reuse seed1009/cohort4000001009, with no weights or experience transferred. Keep
25min training cap,100k recovery and four24-natural-episode frozen arms, including
own actual-initial sampled/greedy controls. All357 initial tensors must match
bounded and retained; pre-update traces must match the same-stride bounded run,
while retained initial actions may differ. Greedy is primary under the declared
raw-score and own-initial gain/min criterion with paired lane bootstrap.

Existing controllers are copied and adapted to the flag/preprocessing change,
25min cap and declared initial-behavior differences. CPU helper qualification
passes in4.142s: retained/bounded audit inputs, loss sums, configuration rejection,
full comparison entry, actual curve input and recovery/collector conditions.
Native model construction is absent. All five exact queued parser/configuration
checks pass in.386s with zero native/emulator construction; preparation takes
1.228s. Declaration pins372 files, with338 retained pins verified using the
existing exact historical replay-helper resolution.

Launched17:32:44.497 UTC November6 from `8d4ab88daae667158accca8ee05c4b1941486588`.
Root `/x/Temp/kindle-rgb-stride2-20261106/full`, main service
`kindle-breakout-rgb-stride2-200k-20261106`, PID1528001, invocation
`6304e163ba8d4aa5a831bfa602f2eb77`. Recovery PID1528007 and terminal collector
PID1528004 are attached. Startup passes all357 bounded/retained initial tensors,
47 same-stride pre-update events and first update at action376; observer.339s.
Compared to retained pre-update behavior,15/47 events match and32 differ, as
allowed by the changed encoder computation. No causal performance claim follows.

The100k checkpoint is saved at6,227 updates and passes the independent CPU
recovery check at **17:42:45.048 UTC November6**. All357 tensors/51,502,659
values are finite, shapes/config/counts and stored hashes match,372 source/input
pins are unchanged, and directory/files are0555/0444. Verification takes.355s
(.414s observer command), with zero model forwards/learner updates. Original
main PID/invocation is unchanged before and after; recovery service exits0.
This preserves weights and moments, not replay, live games/beliefs or complete
RNG state. At the following live observation, training has passed105,832 actions/
6,591 updates; the terminal collector remains attached.

Training completes at17:52:11.444 UTC, the main pipeline at17:55:06.192 UTC
and original terminal collection at17:55:17.128 UTC November6. The original
main, recovery and terminal services exit successfully; all GPU children are
terminal and gone. No render holder or running Kindle service remains.


### Fixed-budget outcome

| Measure | Retained convolution/max-pool | Stride-two |
|---|---:|---:|
| Training actions /updates |200,000 /12,477|200,000 /12,477|
| Actual emulator frames |799,595|799,452|
| Training wall time |21m27.910s|19m22.697s|
| Final greedy mean |143.667|16.458|
| Actual-initial greedy mean |2.208|1.875|
| Greedy own-initial gain/min |6.590132|.752561|
| Final sampled mean |70.750|13.000|
| Actual-initial sampled mean |1.417|1.667|
| Sampled own-initial gain/min |3.230038|.584847|
| Final two-wall successes |0/24 in both modes|0/24 in both modes|

The candidate uses9.722% less training time but loses88.580% primary greedy
own-initial gain/min and81.893% sampled gain/min. All eight lanes regress in both
raw return and gain/min for both modes. The paired eight-lane95% intervals are:

- Greedy raw return difference:−127.208, interval[−164.500,−93.000].
- Greedy gain/min difference:−5.837571, interval[−7.573528,−4.251078].
- Sampled raw return difference:−57.750, interval[−67.708,−47.833].
- Sampled gain/min difference:−2.645191, interval[−3.106194,−2.193570].

These are10,000-resample lane intervals on reused seed1009/cohort4000001009,
not training-seed reliability intervals. The fixed retention criterion fails.
Candidate final greedy range9–26 and sampled6–31 remain far below the original
mean864/90% two-wall gate. Own actual-initial controls are required because the
changed encoder produces different initial behavior despite identical weights.

| Frozen arm | Aggregate actions | Emulator frames | Wall time | Selected mean |
|---|---:|---:|---:|---:|
| Final sampled |15,528|62,080|8.336s|13.000|
| Actual-initial sampled |6,664|26,589|3.952s|1.667|
| Final greedy |16,304|65,180|9.055s|16.458|
| Actual-initial greedy |5,536|22,109|3.267s|1.875|

Each arm selects the first3 natural episodes per lane,24 total, with no
truncation, guidance, labels or learning. All357 tensors/moments remain exact.
Action/reward/frame/outcome replay and videos pass for all four arms. Frozen
logs lack live image hashes; replay does not prove unrecorded image equality.

### Execution and limits

All12,477 learner reports are finite; independent world/behavior loss sums,
parameter changes, actor sharing, cadence, config and initial identity pass.
Full training replay verifies all200,000 actions, native pixel policy decisions,
corrective labels, overrides, rewards, frames and boundaries in98.206s including
CPU audit. Actual overridden fraction is58.0985%;20% applies to episode selection.
Guided online data are not frozen competence. There are357 completed training
episodes, maximum score419 including tails, and no exposure at pre-action score432
or above. The50k-window reported imitation loss declines1.272→.862→.752→.628;
that decline does not establish independent control competence.

Training stage times: learning1,018.866s, acting62.572s, environment62.211s,
observation12.386s, reset1.536s and checkpoint.524s. Agent construction2.465s
is separate. Reported learner components total1,006.157s, including posterior
133.918s, world training270.686s, imagination459.783s and behavior training66.992s.
These elapsed times include synchronization/host work; CPU observer time remains
in training wall time. The full-run saving is smaller than the12.056% bounded
saving, and neither establishes useful learning efficiency.

Publication closure at **17:57:14.858 UTC November6** independently recounts all
192 candidate/control natural episodes and both paired gain/min intervals,
verifies372 active and338 retained pins,55 completion artifacts and28 checkpoint
files, original service identities and five terminal GPU children. It takes1.742s
on CPU with no native import. Root retains `publication-closure.json`,
`comparison.json`, `completion.json`, `terminal-summary.json`,
`recovery-verification.json`, all checkpoints/logs/replays/videos and curves.
Source `8d4ab88d`, native `cde123e1`; original CPU fixture compile failure and
repair remain preserved. No full-run stage failed or repeated.

Keep `rgb_stride2` off and the retained convolution/max-pool encoder. This recipe
changes sampling and nonlinear reduction together; numerical qualification and
parameter identity do not isolate why learning regresses. It does not reject all
cheaper encoders or prove a general max-pooling advantage. No extension, second
learner, diagnosis or successor is automatically selected. Retain143.667 greedy/
70.750 sampled; original864/90% two-wall and the five-game goal remain unfinished.
