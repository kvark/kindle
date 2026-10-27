# A simple pixel follower also fails Breakout

A fixed CPU controller scores **11.333 across24 natural episodes**, with0/24
two-wall successes. It constructs no Kindle agent, performs no GPU work or
learning, and supplies no training guidance. Its complete recorded-action replay
matches every reward, boundary, frame count and task outcome.

The rule uses native max-pooled RGB frames: a red-color mask estimates the ball
centroid in rows90–184 and the paddle centroid in rows185–197, columns8–151.
It chooses RIGHT/LEFT when horizontal error exceeds two pixels, otherwise NOOP,
and FIRE when either object is unavailable. This is hand-written game-specific
control with access to native image detail, not a learned visual representation.
No RAM is read. The fixed rule was checked against existing random-trajectory
images before one evaluation; no controller sweep or selected checkpoint follows.

Protocol matches the compact Breakout evaluation: eight independent lanes,
seed4000001009 and recorded per-lane seeds, full18 actions, sticky.25, repeat4,
zero no-op starts and100000-frame episode limit. The budget is the first three
natural episodes per lane, capped at600k aggregate actions and180 CPU seconds.

Collection completes in **6.062s**, using13,936 actions/55,694 emulator frames.
It collects27 natural episodes, selects the first24 as declared, and retains
three excess episodes and partial tails. There are zero truncations. Independent
full replay takes6.994s and retains a complete stream0 video.

This weak rule is not a useful teacher candidate on present evidence. Its score
does not isolate a control, detector or timing failure, and it does not establish
an encoder limitation. The retained learned policy's current-host score67.583
is stronger, although its training budget and model differ. Keep the reference
separate from native learning results and retain the original two-wall gate.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-pixel-follow-reference`, including
the declaration, complete action ledger, collection/replay review and video.
Controller: `breakout_pixel_reference.py`, SHA256
`60fef11558709921e976eb3d682798cd6d95a01abc81e5cacbe0bb80d4ec04a2`.
Review SHA256`df56e078e85e607b8cc38741d629ea7dd078ec83c01ae6e61bdea4b4ccbac6a0`.
One selected development seed; no training allocation or task-mastery claim.

## Two-frame interception improves the reference but remains weak

A second fixed CPU reference scores **20.292 versus11.333** for position-only
following, still0/24 two-wall successes. It uses the same red mask, object regions,
two-pixel action dead zone, FIRE fallback and environment protocol. Two
consecutive visible ball centroids estimate motion. For a descending ball, it
projects to paddle_y−2, reflects the target between fixed x10 and149 wall bounds,
and clamps it to the estimated paddle reach. Otherwise it follows current ball_x.
Visibility gaps and episode ends reset motion history; implausible per-step
displacements above16 pixels fall back to following. These are declared geometric
assumptions, not learned physics or an isolated test of one control choice.

The single declared run reuses evaluation4000001009 and the same eight lane
seeds, full18/sticky.25/repeat4/noops0/native max-pooled frames. It retains the
first-three-natural-episodes-per-lane selection and600k-action/180s CPU caps.
Collection completes in9.848s at20,616 aggregate actions/82,426 emulator frames.
It collects30 natural episodes, selects24, and retains six excess episodes and
partial tails; zero truncations. Projection is used for10,385 actions; both
objects are visible for19,089. Complete recorded-action replay reproduces every
reward, boundary, frame count, episode outcome and partial tail in8.767s,
with a full stream0 video.

No Kindle agent, GPU work, learner updates, RAM, reward-based action choice or
training guidance is used. The score is still below the retained learned policy's
67.583, with different training/model context. Motion projection helps this
particular weak rule, but does not establish a native representation defect or
a useful teacher. No training-aid allocation, controller sweep or gate change
follows from this result.

Artifacts:`/x/Temp/kindle-cheap-20261025/breakout-pixel-intercept-reference`;
script:`breakout_pixel_intercept_reference.py`, SHA256
`e222deb85221bbc9f02bb9a30b1c7a0433cba2981569e615d4a56bf0344fe720`;
review`585dc64a7b73ec2c6df77b525a9e3b60ec174570155dad40c5dfaf31948d486d`;
declaration`3b55486ca98214a1774611113d9966b5cbb835856cc2fce45e10bdc10aca8744`.
The prior reference, new ledger, full episode records and replay video remain
outside Git. Both use selected development seeds and provide no training-seed
reliability or Kindle task-mastery evidence.

## One motion-aware development episode remains weak

A fixed-action CPU calibration on seed9001 shows continued paddle motion after
NOOP: the latest-frame centroid moves118.5→102.5→101.5 after a leftward command.
The first calibration mask accidentally included the red right wall; its output
is retained, and the corrected mask uses the original interior columns8–151.
This concerns the hand-coded controller's motion assumptions, not the learned
policy or its representation. No RAM or agent is used.

A fixed correction estimates stopping position as paddle_x plus0.75 times its
previous-step displacement, uses a four-pixel dead zone, and corrects partial
side-wall velocity estimates from the previous magnitude. Native max-pooled RGB
remains its only policy input. One development episode on9001 scores31, ending
naturally after1018 actions/4071 frames in0.419s; exact fresh pixel/action/reward/
boundary replay takes0.418s. It fails the two-wall gate. These joint rule changes
do not isolate one control choice, and one episode is not comparable to a
24-episode mean. No full-cohort assessment or training aid is selected.

Artifacts:`/x/Temp/kindle-cheap-20261025/breakout-pixel-guidance-dev`;
policy`breakout_pixel_policy_v1.py`, driver`run_breakout_pixel_dev_v1.py` in its
parent. Review SHA256`5e1da37bae09609ea96a905715a72d3f2777e6556bb1f22217d810d6bcff3c9e`;
trace`7061274507cea3ac771ef75cf271563cf5668ba63f01c8253dc8ec34c0ebc1f5`.

## Calibrated paddle response improves the reference but misses the two-wall gate

A revised CPU reference scores **166.167 across24 natural episodes**, with
**0/24 two-wall successes**. The older position follower and two-frame reference
score11.333 and20.292 on the same evaluation protocol. This is reference control,
not learned Kindle performance or evidence that guidance transfers to a learner.
No training integration or GPU allocation is selected from this result.

An exact replay of the retained9001 development episode reproduces31 points,
1018 actions/4071 frames and all five life losses. The saved controller repeatedly
crosses its predicted landing point before those losses. Retrospective life
counts label the failures only; they never enter the policy. Native snapshots
were inspected. This concerns the reference controller, not a diagnosis of
Kindle's representation or policy.

Using only saved pixel centroids and chosen actions, a four-coefficient linear
model predicts the next paddle displacement from a constant, previous observed
displacement, current direction and previous direction. Fit426 interior samples
before step680; the later204 samples have RMSE3.444 pixels. On29 later NOOP
samples, RMSE3.124 versus10.049 for the previous0.75-displacement estimate.
This chronological split is within one reused episode, not an independent
trajectory or validation at walls. The exploratory fit first had a CPU indentation
error, corrected before calibration and rollouts; no GPU work was repeated.

The new rule uses that fixed response model to score243 five-action plans over
NOOP/LEFT/RIGHT, retaining the latest landing estimate when the ball reaches the
bottom of the detector crop. It keeps the existing native RGB detector and
FIRE fallback. Paddle clipping and constant-velocity ball projection remain
hand-written assumptions. Response control and late-ball tracking change
together; the result does not isolate either change.

On9001, the new rule scores64 in1055 actions/4217 frames, with exact replay.
It then scores144.875 versus23.750 for the original damped rule on eight fixed
first natural episodes; both0/8 two-wall successes. Every saved input/output
pixel hash, action, reward, frame count and boundary replays exactly. These
development checks precede the standard24 selection; that cohort includes
the same eight episodes, whose complete traces match exactly. Do not count
them as independent confirmation.

The standard24 run keeps eight independent environments, full18/sticky.25/
repeat4/noops0/native max-pooled RGB and eval4000001009. First three natural
episodes per lane,600k aggregate-action/180-second collection cap. It completes
43,496 actions/173,949 emulator frames in20.899s, collecting27 natural episodes
and retaining three excess episodes plus eight partial tails. Zero truncations.
Selected returns range16–405. Independent recorded-action replay takes19.950s
and verifies every recorded input/output pixel hash, reward, boundary, frame
count, episode and tail; a complete stream0 video is retained.

All diagnostics/calibration/reference collection and replay total55.935 CPU
seconds on one affined low-priority core alongside the active Pong learner.
This is a co-host timing caveat. All42 Pong input pins remain unchanged, with
the same PID66140/invocation; no agent, native weights or active runtime changed.
The reference is a possible guidance candidate, still far below the unchanged
two-wall gate and not a native learning result.

Artifacts:`/x/Temp/kindle-breakout-control-20261027`.
Eight-episode comparison SHA256:
`42dced746cb295836cb92b6d3480abf175e7f359e96c040df9ca98816135d47e`.
Standard24 review:
`7992b6d31b278a2d3995ee477fbfc625207477b59bb19b0a4ea3671b999caf4b`.
`reference-summary.json` pins the diagnostics, calibration, policies, drivers,
reviews and exact eight-episode overlap check. All artifacts remain outside Git.


## Guidance adapter passes CPU checks and remains staged

The optional Breakout rule is packaged in an external staging tree without
changing the checkout, active runtime or native binary. The adapter shares
Qbert's episode schedule, adds explicit probability/cap flags, native expert
labels and source hashes for both policy and shared schedule. It rejects frozen,
restored, nonnative-pixel, non-full18 and combined-aid use. The rule takes only
RGB frames and its own prior choices; RAM stays outside the policy.

442 CPU tests pass with10 unchanged native-constructor validation cases excluded;
both public and native constructors are replaced before tests import, with zero
construction attempts. Exact replay of11,809 retained Breakout actions reproduces
all choices, input/output hashes, rewards and boundaries. The extracted Qbert
schedule/rule reproduces4096 saved actions and512 guided hashes/choices exactly.
A separate8192-action CPU fixture supplies512 trusted labels and verifies2834
released actions, caps, resets and full guided-input/action replay; it uses an
explicit CPU actor and zero learner updates. These are execution checks, not
Kindle learning evidence. Guided inputs are live-hashed; unguided inputs are not.

Measured CPU test/parity/fixture portions total16.038s, co-hosted on one affined
low-priority core during Pong. All42 active pins and106 original Python sources
remain unchanged. The eight-file patch passes `git apply --check` only; it is not
applied. Evidence:`/x/Temp/kindle-breakout-guidance-20261027`;
validation SHA256`0e0ff2298f8f6ef0ba4befee7f2b6b1de21f4b37cc94784c89afa6101f5d3d63`. Old guided logs retain their matching old
runtime. A later shell-only schedule check initially used system Python without
NumPy and failed before native/environment construction, then used the Atari
venv successfully.

A4096-action native check was prepared but never selected or launched. Its
proposed25%episode/cap128 schedule would exercise labels and release, because
5% selects no episodes within the old4096-action control prefix; that is a
validation setting, not a selected full-run recipe. Pong's passing final result
instead justifies a fresh fixed-recipe confirmation first. Keep this staged
adapter available; do not mutate the confirmation's pinned files.


## Capped reference labels support equal action weights

CPU analysis of the existing24 selected reference episodes informs the proposed
Breakout imitation recipe without collecting new experience or constructing an
agent. Reconcile every saved episode length/reward and the full43,496-action/
173,949-frame log before selecting the same24 episodes. These are prefixes of
the pure reference, not counterfactual guided-learner scores.

| Prefix cap per episode | Retained labels | Mean accumulated reward | NOOP | FIRE | LEFT/RIGHT |
|---|---:|---:|---:|---:|---:|
|128|3,072|2.917|43.46%|3.22%|53.32%|
|640|15,269|24.125|39.94%|3.40%|56.66%|
|Whole natural episode|36,727|166.167|34.94%|9.24%|55.82%|

All24 prefixes at128 and640 include positive reward. One episode ends before640;
the remaining23 would be released at that cap. At640,96.60% of input frames have
a visible ball under the reference detector. FIRE contributes only5.48% of the
weighted label mass even with the prior Qbert NOOP weight.05; FIRE does not
dominate this prefix cohort. NOOP is a substantial part of the predictive
controller's stopping behavior, so the proposed Breakout test keeps **all action
weights1**, including NOOP, rather than inheriting Qbert's.05. Label mass is not
gradient magnitude; no weight optimum or learning benefit has been measured.

The external execution-check draft keeps4096 actions/963 updates,25%episodes/
cap128, expert scale1, full action weights,180-second training cap and1024 frozen
actions/60-second cap. It must follow the active Pong confirmation's terminal
audits and application of the validated eight-file adapter. The separate full
candidate is200k actions/49,939 updates under25 minutes,4M64/Auto/N8/B8/T16/H15/
R32/coefficient1/actor.01/full18/original rates,5%episodes/cap640/expert1/allweights1.
It retains100k recovery, fixed final/own initial24 natural episodes, original
90%-two-wall gate, guidance off in evaluation, full audit/replay/guard evidence
and no extension or automatic successor. The full candidate also requires the
short native check to pass and a separate selection; **neither is selected**.

Five candidate commands parse without agent/environment construction; eight
external scripts compile and comparison readers accept the existing full18
7.083 and minimal4 9.542 controls. All53 active Pong pins and the staged patch
stay unchanged. Successful prefix/parser portions take0.630s on one affined
low-priority core during Pong; no claim of isolated timing. An initial CPU-only
analysis used Pong's `mean_natural_return` field instead of the existing task
score's `mean_completed_return`; that KeyError and its correction are retained.
No trajectory, source/runtime, score criterion or GPU stage changes.

Artifacts remain in`/x/Temp/kindle-breakout-guidance-20261027`:
`reference-prefix-review.json` SHA256`e148de7b1f1afd41a5612f82f94c6e81a9265ad6452305ca790ae756c244d008`;
`draft-preflight.json` SHA256`309637aae43829a8beb1dcaa614edb036758e6eb820960c6f3410fdc4b9447cf`.
The draft scripts identify their exact predecessors, controls, caps and hashes.


## A settling penalty gives only a small selected-cohort gain

An exact CPU replay of the strongest saved first episode, lane1/seed4001001012,
reproduces405 points,1980 actions and7917 frames, including all recorded input/
output hashes, rewards and boundaries. Its five life-loss steps are662,1162,1424,
1533,1980. Last-step snapshots and predicted/observed paddle positions show
fast target crossings and several-pixel forecast errors near impact. These are
observations from one selected reference episode, not a causal diagnosis of
native learning. Life counts label the replay retrospectively and never enter
policy decisions. Diagnostic replay takes0.980s.

One external CPU variant adds0.5 times squared predicted paddle displacement
at the impact segment to the existing near-impact planning cost. It retains
the calibrated response, detector, native frames and action vocabulary. On the
same eight reused first natural episodes it scores **152.750 versus144.875**,
with **0/8 two-wall successes for both**. Variant scores are90,112,106,91,125,
376,106,216;12,259 actions /49,022 frames,5.572s collection and3.899s exact
input/output/action/reward/boundary replay. All eight finish naturally; no cap
increase or further variant is selected.

The variant was chosen after inspecting this cohort's strongest baseline episode;
it is development evidence, not a held-out comparison. The modest7.875-point
gain does not justify replacing the already validated staged teacher. Retain
the original predictive rule and the separately proposed all-action-weight1
imitation recipe. No native learner, staged source, checkpoint or active runtime
changes. All53 active Pong pins remain unchanged; low-priority CPU work during
that run remains a timing caveat.

Miss review SHA256`43dbff1c477e20c67ae27117694a0bf9699f894a35acaee20866c1d608db8f1e`;
variant review`cb41918e71d164fea904f2df491d6e094cef0c3b66f9ec5c80b2f22061643156`.
Both are retained under`/x/Temp/kindle-breakout-guidance-20261027` with exact
source pins, traces and snapshots.


## Validated adapter applied after Pong completion

After Pong4013 exits02:41:08 UTC and its CPU completion audit passes02:41:23,
all53 preceding pins are verified unchanged. The exact validated eight-file
patch is applied; native79c9a7f remains unchanged. The442-test CPU result and
saved trajectory parity apply to identical bytes. Application receipt is
`/x/Temp/kindle-breakout-guidance-20261027/application-review.json`.
Select only the [bounded4096-action check](../experiments/2026-10-27-cdp-breakout-expert.md);
the prepared200k candidate remains unselected until separate review.


## Bounded native check passes

The applied adapter completes **4096 aggregate actions /16,350 emulator frames
/963 updates in24.792s**. First update248, zero residual credit,4,131,729
parameters, all finite reports and changed CNN/reward/actor/value weights.
All346 initial tensors match the full18 control. Guidance executes669 trusted
actions (16.333%):309 NOOP,27 FIRE,149 RIGHT and184 LEFT;660 updates have
nonzero imitation loss. Episode probability25% is separate from actual exposure.
Full guided input-pixel/action/reward/boundary replay and both host guards pass.

Ordinary frozen restore performs1024 actions /4096 frames in.488s, zero updates,
all346 tensors unchanged. It completes no natural episodes and establishes no
task competence. Service exits02:50:25 UTC on October27,34.550s total wall.
Source`d49c0b7`, native79c9a7f unchanged, no repeated GPU stage. Review SHA256:
`f55e5d90dab8e79b379fe45398d54087597500246a3960308bd0b520722bdc8f`.

After reviewing the finite native updates, meaningful expert labels, matched
initialization and complete restore/replay checks, select the separately declared
fresh200k comparison. Keep5% episodes/cap640/expert1/all action weights1 and a
25-minute training cap, with fixed final/own initial24-natural evaluations.
No check experience is reused, and no transfer benefit is assumed. See the
[full declaration](../experiments/2026-10-27-cdp-breakout-expert.md#selected-full-comparison).


## Capped native imitation lowers frozen Breakout quality

The fresh expert learner completes **200k actions /798,964 emulator frames
/49,939 updates in22m38s** (1358.085s), within25 minutes. Fixed final frozen mean
**4.000**, versus its own actual initial1.375, is below the full18 control7.083
and stronger minimal4 control9.542. All have0/24 two-wall successes. The measured
training wall ratio to full18 is1.00698; the recipe provides no useful gain.
Retain the unaided controls; no extension or confirmation follows this learner.

Configuration4M64/Auto/N8/B8/T16/H15/R32/coefficient1/actor.01/full18, original
rates, reused1009/eval4000001009,5% episode guidance capped640, expert scale1,
all18 action weights1. Guidance plus expert loss, qualified native and older
control kernel differ; this does not isolate an imitation mechanism. Training
uses20,463 trusted actions (10.2315%):8157 NOOP,683 FIRE,5622 RIGHT,6001 LEFT.
32 of557 episode starts are selected;28,839 updates contain expert labels.

All finite reports, config/source/counters,46 pins,100k recovery,346 initial
tensors matching both control/check,346 invariant tensors in each frozen arm,
complete guided input/action/outcome replay, frozen recorded-action replay,
videos, guards, curves and comparison pass. Final evaluation selects first3
natural episodes per lane from25 completed episodes:9336 actions /37,305 frames
/4.669s. Own initial selects24 from29:5944 /23,717 /3.051s. Zero learning or
truncations; excess episodes and partial tails retained. Frozen logs lack live
pixel hashes, so their replay does not establish direct live-image equality.
Service exits03:16:09 UTC on October27; postprocessing03:16:14. No GPU repeats.

Exposure analysis reconciles every action/report and completed/partial reward.
31 completed guided episodes average24.871, with707 points during teacher
prefixes and64 after release across6209 released actions. The518 unguided
episodes average2.948; their last50 average5.100. All0 two-wall successes.
After-release rewards inherit a guided state; these online figures are not
causal attribution or frozen competence. Label cross-entropy falls1.168→1.062
across50k update-time windows on changing replay samples, not held-out labels.
The negative frozen result remains decisive.

Independent Qbert reference/adapter CPU work contributes461.083s of measured
affined successful portions during this life, plus.015s exposure-reader validation.
All46 pinned inputs remain unchanged until terminal audits. Retain this co-host
wall-time caveat; no correction or isolated timing claim is made.

Artifacts:`/x/Temp/kindle-breakout-guidance-20261027/breakout-4m-64-r32-expert1-cap640-noop1-200k`.
Learning review SHA256`93a5c42bf90b6d1036880b04f101539cbda6be0b8b05edea3ac1ac1553354c2f`;
completion`ae048f66de4dda8ff92f97c4c37f30d62268eceee3dc2527fd251fe1d5787da7`;
comparison`0d6ec959dd417a4e08f4a750684ed6d78ea95037f2bcd74c28efaa8b409b38cc`;
exposure`356e7aa2073a1125aaef9ebf1662f61c1e27b38c06b7080189802d972c869539`.


## Frozen movement agreement does not track task quality

A CPU-only diagnostic replays the saved final/actual-initial expert trajectories
and the full18/minimal4 controls. Only recorded native actions are executed.
The unchanged pixel teacher provides advice on each reconstructed native frame;
its previous-command state is set to the actual recorded movement command.
This avoids predicting from an unexecuted teacher action. Movement aliases are
grouped as LEFT/neutral/RIGHT, ignoring FIRE for this comparison. All actions,
rewards, episode boundaries, reset counts and emulator-frame counters reproduce;
selected24 natural-episode scores equal the completed reviews. No native module
loads, learner runs or checkpoint changes.

| Frozen arm | Mean return | Visible-ball movement agreement | Near-impact samples | Mean distance to predicted landing point |
| --- | ---: | ---: | ---: | ---: |
| Expert actual initial |1.375|32.87%|316|57.53 pixels|
| Expert final |4.000|40.32%|438|36.60 pixels|
| Full18 control |7.083|33.76%|610|26.88 pixels|
| Minimal4 control |9.542|37.42%|632|15.42 pixels|

Near impact means the teacher projects a descending ball with at most two
remaining action intervals. Distance is measured before the recorded action,
in native pixels, against that teacher's estimate; it is not ground-truth
collision error. The detector excludes balls above row90, and an undetected
ball does not establish waiting to serve. Policies visit different trajectories,
so these conditional summaries do not isolate training effects. The teacher is
also imperfect: direction agreement is a descriptive proxy, not action accuracy.
Visible-ball agreement expected from independent empirical action marginals is
33.11/33.36/31.77/32.57% respectively. Temporal dependence and reused episodes
preclude treating these action counts as independent confidence samples.

The expert policy shows some movement association with the teacher beyond its
initial policy, while its lower frozen return remains decisive. Higher teacher
agreement does not order these policies by task quality. This evidence does
not justify increasing imitation weight from loss/overall agreement alone or
extending the completed expert learner. Retain the unguided controls; no new
Breakout GPU allocation is selected by this analysis.

The four successful replay arms cover9336/5944/16296/52080 aggregate actions
and take50.202s on one affined low-priority core during the active Qbert life.
All56 Qbert pins remain unchanged. The initial attempt repeats the two expert
CPU arms before rejecting an older control replay-helper hash; its timestamp
interval is11.050s. A second historical helper is required for the minimal4
control; already completed receipts are reused on that repair. Both original
failures are retained. Exact helper versions are recovered from Git, and their
own historical receipts pass without editing current source. No GPU stage is
repeated. Co-host CPU work remains a Qbert timing caveat.

Artifacts:`/x/Temp/kindle-breakout-transfer-20261027` contains preparations,
original attempts, exact historical helpers, per-action observations, per-arm
reviews and combined review SHA256
`8d77cc026e295bd98f34c14f72427feb25a29bdf13e96c59b6006fb18eb48609`.
Ordinary frozen logs lack live pixel hashes; reconstructed input/output hashes
are retained but do not establish direct equality with absent live hashes.

## Longer saved prefixes expose most teacher reward

One additional log-only check compares the existing640-action limit with a
predetermined1920-action prefix on the same24 saved predictive-reference
episodes. Full-reference mean remains166.167, with0/24 two-wall successes;
no new policy, play, action decision or learner update occurs.

| Prefix cap | Retained action labels | Mean prefix reward | Fraction of full reward | Episodes ending within cap |
| --- | ---: | ---: | ---: | ---: |
|640|15,269|24.125|14.52%|1/24|
|1920|36,242|159.833|96.19%|20/24|
|Full natural episode|36,727|166.167|100%|24/24|

The longer prefix retains98.68% of full-reference labels, versus41.57% at640.
Its action mix is35.20% NOOP,8.75% FIRE and56.05% lateral movement. All action
weights remain1 for this description; action mass is not a gradient norm.
The640/full counts, visibility fractions and rewards exactly match the earlier
prefix receipt, and the complete logged action/frame/episode reward accounting
reconciles with the verified source review.

This makes later reward exposure a concrete hypothesis for a future cheap
Breakout recipe. It does not diagnose the failed native learner or predict
transfer: these reference trajectories use the reused evaluation cohort, and
real guidance changes the learner's visited states. The completed cap640
learner remains frozen4.000, below its unaided controls; a stronger teacher
prefix cannot relabel that result. Even the full teacher misses the original
two-wall gate. No new Breakout allocation, source edit, cap change or further
prefix sweep is selected here.

The check takes.261s on one affined low-priority CPU core during the active
Qbert R16/20% life, with all47 input pins unchanged. Artifacts:
`/x/Temp/kindle-breakout-exposure-20261027/reference-prefix-review.json`, SHA256
`1bc3d5263cb12cfcc1aba083a51222e087a8bc48cfa2f66fa32244e758bf0825`;
script`d94b0b51c27e3a96b21313a4f148b62bc7ab17ad42b9b07896446f5ad1ad2669`.

## Longer-prefix native check passes

The fixed bounded check completes16,384 aggregate actions/65,473 emulator
frames/4035 updates in114.327s, followed by1024 frozen actions/4096 frames
in.558s with zero updates and no natural episodes. Original native79c9a7f,
Breakout runner/runtime and all32 pins stay unchanged. All346 initial tensors
match the cap640 primary control; all346 frozen tensors remain invariant.
All reports are finite, first update248/residual0 matches, and every element
of the selected CNN/reward/actor/value projections changes. Full guided
input-pixel/action/outcome replay and both guards pass. Service exits05:38:24
UTC on October27,2m10.891s including construction and audits.

The execution-check25% episode schedule selects7/35 starts and10,125 trusted
actions (61.7981%):3449 NOOP,675 FIRE,2907 RIGHT,3094 LEFT.3829 updates contain
labels;219,758 repeated replay rows/weight mass219758 and loss sum2386.045.
Two completed guided episodes average141.5;25 unguided average1.44, all0 two-wall
successes. The check has0 released native actions: natural ends or late starts
preempt reaching cap1920. It does not qualify cap release or frozen competence.

The additional log-only exposure helper initially assumes the old4096/963
smoke budget. Its original script/failure are retained; correcting the reader
to the declared16384/4035 makes complete action/report/reward decomposition
pass. The native service was already successful and no GPU stage repeats.

The early rate extrapolates to23.26 minutes at200k, with uncertainty and a
higher check guidance probability than the proposed5% full run. After reviewing
terminal audits, separately select one fresh200k comparison with a25-minute
training cap; no cap increase occurs on an existing allocation. Only the
640→1920 guidance limit changes against the primary expert control. No check
weights, replay, belief, games or RNG are reused. Fixed final/own initial
assessments and the original two-wall gate remain. This is an exposure test,
not a predicted success or justification for an unchanged extension.

Artifacts:`/x/Temp/kindle-breakout-exposure-20261027/breakout-4m-64-cap1920-check`.
Review SHA256`f93023cad808e5a85143aef76a0b6c98f0747215baceec00e667a34331ad570a`;
completion`d85dd71ccc1a8fc505bd3bb6c4fa7a39b86c40fe36c229b8ac6724548097bef7`;
exposure`90a18f0d95e7fea1ff714590681af1e94f4548c5f6daf91f43662202c6a36b3e`.
