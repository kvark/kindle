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
