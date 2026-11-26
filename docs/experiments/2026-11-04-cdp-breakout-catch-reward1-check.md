# Bounded Breakout check with one-point pixel catch feedback

Select development of an optional one-point catch reward after the completed
[raw-motion validation](../results/2026-11-04-cdp-breakout-catch-validation.md).
All1,476 teacher and1,132 retained-policy detections match raw near-paddle
reversals;all arrive on zero-game-reward transitions. Spatial exclusions are
crop-edge cases and remain exclusions. This supports testing a nearer positive
outcome signal without claiming exact contact ground truth or gameplay benefit.
The previous one-point life-loss penalty regressed; do not combine or retry it.

Use the fixed causal detector and pixel extraction from the two CPU diagnostics.
No threshold tuning or new actor inputs. Award exactly+1 training point on each
detected catch,added to the raw extrinsic arrival reward before the unchanged
native learner consumes it. Initialize detector state from each reset frame;
advance once per returned pooled observation and clear at episode reset. The
arrival after actionj credits actionj,not the following action. Do not infer
catches from lives,teacher commands,future frames,raw-motion witnesses or RAM.

Expose an optional fresh-training-only Breakout flag,default off. Reject use
with evaluation,restore,other games or unsupported observation geometry. Keep
raw game rewards,episode returns,task outcomes,termination and policy frames
unchanged. Log the declared aid and each emitted event,raw reward and training
reward so independent replay can verify every native reward input. All frozen
assessments have the aid,guidance and labels off and report original game scores.
Preserve the default-off semantic path. Python handles this explicit game aid;
learning and inference stay native on Meganeura + Blade.

Use native447fe50/SHA35ad0db and the corrective runtime. Start from exact retained
runnercf4ea238 with only the declared catch integration; do not inherit unrelated
current-runner defaults or optional losses. Keep CDP12M64/Auto/N8/B16/T16/full
BPTT/micro16/H15/R16/context1,base/encoder4e-5,dynamics4e-4,behavior8e-5,
policy1/value1/replay-value1,representation.1,dynamicsKL1/free-nats1,continuation1,
CDPfuture500,disagreement1,expert100/unit weights/world gradients/all-state
corrective labels,sampled collection/posteriors,actor-unimix.01,replay100k,
20% episode guidance/cap1920,full18/sticky.25/repeat4/noops0 and one GPU64 resize
from native pooled frames. Teacher and command feedback remain unchanged.

Before GPU work,qualify meaningful CPU fixtures for arrival timing,episode and
lane isolation,duplicate suppression,raw-versus-training rewards,default-off
compatibility and CLI rejection. Retain prior detector qualifications. Verify
the new adapter against saved observations from the completed CPU capture
without another emulator replay. Exercise queued CLI and audit entry points
before native construction. Pin exact source/runtime,helper,ROM and inputs;
preserve historical sources rather than rewriting old declarations.

Then run one fresh8,192-action/489-update check on seed1009,within120s training.
Retain actual initial and final weights. Require357 matching initial tensors
against the retained control,47 matching shared pre-update action/raw-outcome
events,with differences restricted to the declared training-reward fields,finite
losses/cadence,changed CNN/reward/actor/value weights and independently verified
pixel/action/teacher-label/catch/reward replay. Use raw scores for curves and
exposure; shaped training returns are not task competence. Verify no changes
to native binary or teacher bytes. Keep the existing host guard.

Run one1,024-action frozen execution check with aid/guidance/labels off,zero
updates and357 invariant tensors. Natural episodes are not required in this
short check and no gameplay benefit can be claimed from it. Require a conservative
200k-action cost projection within30 minutes before considering a full comparison.
Use the retained projection convention,including startup/warmup and padding.

Training replay cap300s,frozen replay120s,main900s,terminal collection960s.
Check actual idle state before launch. Preserve all failures; do not repeat
successful stages,raise caps or change coefficients after seeing outcomes.
No full learner,experience transfer,extension,second seed,sweep or automatic
successor. A separately selected full comparison would need fresh experience,
100k recovery and actual-initial/final sampled/greedy24-episode frozen arms,
with primary greedy raw score and initial-subtracted gain per training minute.
The original864/90%-two-wall gate remains unchanged.

Root `/x/Temp/kindle-breakout-catch-reward1-20261104`;service
`kindle-breakout-catch-reward1-check-20261104`. Implementation and qualification
are pending at selection; no GPU allocation or full learner is active.
