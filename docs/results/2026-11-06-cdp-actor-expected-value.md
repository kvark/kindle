# Enumerated one-step actor: qualification and full learning result

The completed fresh200k comparison **fails retention**: greedy138.417/sample58.083
versus143.667/70.750, with30.341% more training time and26.126% lower primary
own-initial gain/min. All0/24 two-wall. Keep the default off; this one recipe does
not isolate a categorical-gradient cause or reject every related method.
[Full results](#full-200k-comparison). No GPU allocation or successor is selected.

The [development protocol](../experiments/2026-11-06-cdp-actor-expected-value.md)
implements `actor_expected_value`, default false. It replaces the sampled-action
score-function term with the policy-weighted expectation of detached one-step
values over every action at each replay posterior start. Each value includes the
predicted arrival reward, current action-effects bonus and predicted discounted
continuation times current critic value. Existing entropy, imitation, critic,
imagination, return normalization and collection behavior remain.

The action branches reuse the first ordinary imagination step's latent uniforms
for each start, with one sampled prior state per action. World parameters are
private GPU snapshot inputs copied before world learning. The critic and Q
targets are detached from this actor term. The actor has one optimizer owner;
no learned parameter, optimizer slot, replay field or image transform is added.
The new option changes the policy objective's horizon and state distribution
together with its estimator. It is not an isolated test of discrete-gradient
bias or an exact expectation over latent futures.

Legacy serialization omits the false flag. Configuration and CLI guards reject
other actor experiments, device targets, fresh overrides on restore/evaluation
and enabled checkpoints on old native packages. Reports expose the unscaled
expected-value term and snapshot submission time separately. The total policy
loss already contains this term; it must not be added again as an auxiliary loss.

CPU qualification passes **207 Kindle +5 adapter tests**, **192 focused Python
tests**, a separate **20-test guard run** (15 overlap the first Python run), and
strict workspace/all-target and binding/all-target Clippy. Graph checks preserve
actor/critic parameter names and shapes, compile the combined graph and isolated
objective, verify only actor gradients from that objective, and check branch
layout at3/18 actions. Python checks cover the actual CLI configuration path,
old-runtime rejection, restore/device guards, report identity and malformed
snapshot timing. These Python runs use current sources with the previous
qualified binary, mocking the new capability and constructing no native agent.

The first Rust compile fails on an untyped index closure in the prepared native
fixture. Its original source and failed receipt are preserved. Adding the explicit
`usize` annotation repairs that test-only issue; corrected CPU execution and
Clippy pass. No GPU fixture or learner has executed at CPU closure.

Two ignored native fixtures are prepared and compile. One checks the centered
mixed-policy objective and all six actor-logit derivatives against independent
F64 arithmetic, frozen values/weights and zero masks. The other checks every
Tiny branch against ordinary transition/head/value/bonus inference, matches the
original imagination's selected first-action branch, verifies private pre-update
snapshots and shared draws, then checks two synthetic updates and exact optimizer
restore. Nonuniform reward/value weights prevent an all-zero branch check.
Compilation does not establish native values, derivatives or learning benefit.

Root `/x/Temp/kindle-actor-expected-value-20261106` retains commands, source pins,
logs and the original failed fixture. Next package and separately qualify the
native path before selecting a bounded cost check. No GPU allocation, learner,
sweep or automatic successor is selected. Retain143.667 greedy/70.750 sampled,
ordinary inference and32×16; original864/90%-two-wall and five-game gates remain
unfinished.

## Packaging and native declaration

Release test binary and wheel package source `f2b80fe596ae553bbea3cc7a3c638ea55529dc22`,
with253 source pins unchanged during both builds. Native SHA256 is
`961f79583c1e2384f5dd2daba56fc2357c1a8dc280abd6bb8aef4d35fbf8b069`;
test executable SHA256 is
`e582bbce455f6e05b033ddf6a322c14ba513cfd756299e01b7708a96a93d1783`.
Packaged Python passes1194 tests with10 constructor checks deselected and zero
native construction attempts, in4.322s including the wrapper. This verifies the
actual package capability flag and legacy defaults without GPU allocation.

The [separate native protocol](../experiments/2026-11-06-cdp-actor-expected-value-native.md)
selects the two prepared fixtures once, with the existing AMD host guard. Two
synthetic optimizer steps and two Tiny learner updates, zero Atari actions.
A bounded cost check and full learner remain unselected.

## Native qualification complete

Both exact fixtures pass on AMD RX7900XT/RADV, source `bd22618f97fde749008314030161cf893dc39a0d`.
The objective fixture takes.71s and the branch/update/restore fixture.77s; guarded
controller3.467s. All259 pins remain unchanged. Independent F64 objective and
six derivatives meet2e-5 absolute tolerance; all24 branches per update, selected
ordinary-imagination branches and reconstructed objective meet declared1e-3
relative-plus-absolute tolerance. Private pre-update snapshots and shared inputs
are exact; two learner updates and exact parameter/moment restore pass.

Children1489300/1489350 exit0; host guards, terminal checks and idle render/service
checks pass at13:02:05.115 UTC November6. Two synthetic optimizer steps, two Tiny
learner updates, zero Atari actions. The original CPU compile failure remains
preserved; no native fixture fails or repeats. This is correctness evidence,
not a gameplay or efficiency result.

The [separate bounded protocol](../experiments/2026-11-06-cdp-actor-expected-value-bounded.md)
now selects8,192 fresh learning actions and1,024 frozen actions with this package.
The one-step expectation removes the sampled-action term but preserves entropy,
imitation, critic and ordinary imagination; its extra branch cost is unmeasured.
No full learner is selected. Keep the retained recipe pending measured outcomes.

## Bounded learning-cost check complete

The fresh8,192-action learner completes489 updates in **62.640236s**, versus
retained47.689684s: **31.3497% more** observed wall time. Training executes32,725
emulator frames; construction adds2.568818s. First update is at376 actions and
final debt.5. Its20%-episode schedule actually overrides3,788/8,192 actions;
guided online returns are not frozen competence. Native pooled RGB, one GPU
resize,12M64/B16/T16/H15/R16,32×16 and the retained teacher/settings remain.

A separate1,024-action/4,096-frame frozen greedy restore takes.848175s plus
2.320214s construction, with zero updates and no completed natural episode.
All357 tensors remain invariant and the expected-value flag survives restore.
This arm checks execution/restore; it supplies no natural-episode score.
All357 actual-initial tensors/moments and47 pre-update transition events match
retained exactly. Encoder, reward, continuation, actor and critic change during
learning; frozen world actor copies match their source actor. All489 finite
reports and both weighted totals pass. The expected-value metric is already
inside policy_loss and is not added a second time.

The padded200k projection `62.640236*200000/8192*1.1` is **1682.232912s**
(**28m02.233s**), below the declared1800s cap. All3098 memory snapshots are valid.
Minimum observed headroom is16,356,782,080 bytes; subtracting the entire full
5,939,200,000-byte replay leaves10,417,582,080 bytes (**9.702GiB**), above2GiB.
Both gates pass. This is a bounded projection, not measured full-training cost.

Learning occupies56.242s of the62.640s loop. Summed wrapper timings include
imagination16.635s, world training13.593s and behavior training18.187s; the new
snapshot wrapper sums.145s. These timers do not isolate GPU kernels, and neither
this cost change nor derivative correctness establishes a better policy.

Both independent full action replays pass, including training pixels/corrective
labels and frozen outcomes/frames/boundaries. All253 source/51 execution pins,
host guards and queue/audit receipts pass. Source `ad5fa2934cdc9820795031359e567d2b6a19757e`,
controller1489646 completes **13:06:26.569 UTC November6**,81.268s including
checks (outer wrapper81.309s). The read-only publication closure completes
**13:07:03.835 UTC** in.072s, confirming original controller/children gone,
no Kindle service/render holder, initial/trace/frozen identities and cost math.
No GPU stage or audit fails/repeats. Original compile failure remains preserved.

The3,449,779,219-byte disposable build cache is released after verifying all
three retained package artifacts. Root `/x/Temp/kindle-actor-expected-value-20261106/bounded`
is closed. No full learner, sweep, extension or automatic successor is selected.
A separately declared fresh fixed comparison is now feasible; it must judge
frozen own-initial gameplay gain per training minute against143.667/70.750,
not loss magnitude or bounded guided returns. Retain the existing recipe and
keep the new option off. Original864/90%-two-wall and five-game gates remain.

## Fresh200k comparison selected

The [separate full protocol](../experiments/2026-11-06-cdp-breakout-actor-expected-value-200k.md)
selects one fresh200k-action learner, with no checkpoint/bounded-experience
transfer, plus actual-initial/final sampled and greedy24-natural-episode arms.
Greedy raw return and own-initial gain/min must improve, with a positive lower
bound on the paired eight-lane95% gain/min interval. Retained143.667/70.750 and
original864/90%-two-wall gates remain. This is a reused development seed/cohort.

Current CPU helper checks pass in3.477s including wrapper. They exercise exact
configuration comparisons, weighted totals, required expected-value metric and
positive snapshot timing, retained/bounded exposure accounting, complete paired
comparison entry, malformed headers, plotting and null journal messages. No
native module or emulator is constructed. Exact five-command qualification and
launch remain next; no learner is active at this declaration. Root
`/x/Temp/kindle-actor-expected-value-20261106/full` retains its source copies and
historical source resolution. No old stage or failed queue is repeated.

## Fresh200k comparison running

The exact five queued commands pass CPU qualification in.379s including wrapper,
with zero native/emulator construction and365 pinned inputs. The guarded service
launches **13:14:33.501 UTC November6**, source
`2de5bd0f41f2eb00dc6524dfa25d4e2986aa8777`; mainPID1491004, invocation
`9cf81a2d7f52454a9bec3ef5685633b8`. Collector1491007 and recovery observer1491010
run independently against that original identity. Preparation/qualification/
launch pass without failure or repeated GPU stage.

Startup verifies all357 initial tensors/moments against both bounded and retained,
all47 pre-update semantic events and first update at376 actions. The exact header
change is only the expected-value flag plus explicit existing bootstrap default
and declared source hashes. Learning is active with finite reports. No final
score or gain/min result exists yet. The100k recovery check, all four frozen
arms, full action replays, curves, exposure and terminal collection remain pending.
An observation timeout is not a stopped learner; inspect the original service and
process before any action. No successor or extension is selected.

## 100k recovery verified; learner continues

The original learner saves100,000 actions/6,227 updates. Its checkpoint review
passes at **13:28:30.873 UTC November6**; the independent CPU verification passes
at **13:28:34.052**, taking.328s (.414s including its wrapper). All357 tensors
(world280/behavior66/slow11) are finite, with exact metadata/config, tensor/file
hashes and expected shapes. Directory0555/files0444 retain the recovery read-only.
All365 pinned inputs remain unchanged. The observer performs zero model forwards
or learner updates and confirms the same live mainPID1491004/invocation before
and after verification. Its service exits successfully.

The learner and collector remain active under their original identities. This
checkpoint retains weights/optimizer moments; it does not restore replay, live
beliefs, game state or complete RNG. CPU observer time remains in measured training
wall time. Final200k training, four frozen arms, full audits and terminal collection
remain pending; no gameplay or efficiency conclusion is available.

## Full 200k comparison

The [declared fresh comparison](../experiments/2026-11-06-cdp-breakout-actor-expected-value-200k.md)
completes **200,000 actions/799,571 emulator frames/12,477 updates** in
**1678.681539s (27m58.682s)**, plus2.630956s construction. Retained training takes
1287.910495s (21m27.910s), so observed training cost rises **30.3415%**. The run
stays within its1800s cap but fails the primary learning-efficiency criterion.

| Frozen result | Retained | Enumerated actor |
|---|---:|---:|
| Greedy final mean |143.666667|138.416667|
| Sampled final mean |70.750000|58.083333|
| Greedy actual-initial mean |2.208333|2.208333|
| Sampled actual-initial mean |1.416667|1.416667|
| Greedy own-initial gain/min |6.590132|4.868404|
| Sampled own-initial gain/min |3.230038|2.025399|
| Final two-wall successes, either mode |0/24|0/24|

Primary greedy gain/min falls **26.1258%**. Its paired eight-lane difference is
−1.721728,95% interval **[−4.253377,.800655]**. Raw greedy difference−5.250 has
interval[−66.418750,55.879167]. Sampled gain/min falls37.2949%; difference−1.204639
has interval[−1.746170,−.629304], while raw score difference−12.666667 has interval
[−25.584375,.792708]. Bootstrap10,000 resamples/seed0. These describe the reused
cohort, not training-seed reliability. The primary point estimates regress and
its gain/min interval lacks a positive lower bound; keep the option off.

All357 initial tensors/moments match both bounded and retained, as do47 pre-update
semantic events. The actual-initial sampled/greedy799/794 events match retained
exactly apart from permitted time fields. Each frozen arm selects24 natural
episodes with zero updates/truncations and357 invariant tensors. Together the
four arms execute84,840 actions/339,194 frames, with all replay/video checks
passing. Fixed cohort selection is unchanged; no episode or stage is substituted.

All12,477 finite reports, both weighted losses, expected-value metrics and
positive snapshot timings pass. Expected-value loss remains inside policy_loss,
not a second auxiliary addition. Encoder, reward, continuation, actor and critic
parameters change, and world actor copies stay synchronized. The100k recovery
and its independent verification pass with exact365 pins. All full training
pixels/actions/labels/overrides and frozen action/outcome/frame/boundary replays
pass, together with curves, exposure and original host guards. No full
preparation, GPU, audit or publication stage fails/repeats; the original CPU
fixture compile failure stays preserved.

Late own-replay label cross-entropy is.554410 versus retained.560514 on different
experience. This does not measure frozen skill or explain the regression.
Training spends96.4445% of actions below216 points and none at432 or above;
1,528 native actions occur in216–431. The20%-episode guidance schedule overrides
44.511% of all actions. These cumulative-score/experience statistics are not
wall-success events, and guided online returns remain separate from evaluation.
Learning occupies1533.983s of1678.682s; component wrappers do not isolate kernels.

Source `2de5bd0f41f2eb00dc6524dfa25d4e2986aa8777`; training completes
**13:42:36.615 UTC November6**. Original main/collector complete **13:46:32.617 /
13:46:51.144 UTC**, with365 current/338 retained pins verified. Read-only publication
passes **13:47:40.250 UTC** in.024s. All original services exit successfully;
original main/collector/recovery PIDs and all five guarded children are gone,
with no Kindle service or render-device holder. Exact receipts/logs/checkpoints/
replays/videos/curves remain under `/x/Temp/kindle-actor-expected-value-20261106/full`.

The new objective changes horizon and replay-start distribution with its policy
estimator. Its correct native derivative and all-action expectation do not prove
accurate action values or useful gameplay guidance. This is neither an isolated
gradient-bias explanation nor a SuperMimic reproduction. No coefficient sweep,
longer extension or automatic successor follows. Retain32×16, ordinary inference
and143.667 greedy/70.750 sampled; original864/90%-two-wall and five-game gates
remain unfinished.
