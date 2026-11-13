# The Qbert teacher avoids most selected losses from learner states

The unchanged feedback-aware pixel teacher avoids 25 of 32 recorded life losses
when it takes control up to 32 actions earlier on the retained encoder4e-5
policy's own states. It survives the separate 64-action follow-up at 23 origins.
This establishes that useful teacher continuations exist on many of these
later-play failure states; its better standalone score was not the only evidence.
It does not identify why the learned policy fails to follow a successful sequence.

| Outcome | Fixed diagnostic result |
|---|---:|
| Source episodes / selected losses | 8 / 32 |
| Teacher avoids loss through original loss-action endpoint | 25 / 32 (78.125%) |
| Teacher survives 64 actions after takeover | 23 / 32 (71.875%) |
| Teacher minus recorded reward to endpoint or earlier teacher loss, summed | +525 |
| Avoided losses per lane, each out of four | 3, 4, 3, 3, 3, 3, 4, 2 |

The [fixed protocol](../experiments/2026-11-01-cdp-qbert-recoverability.md)
selects every loss in the first natural episode of each of eight retained lanes.
These are a subset of the already inspected development episodes, selected by
episode index rather than score. Each origin starts immediately before
`max(previous_loss_action + 1, loss_action - 31)`, with action one as the initial
floor. The recorded and teacher branches start from the same emulator, wrapper,
pixel and teacher state. Teacher history includes proposals and actual-command
feedback from the beginning of the recorded episode.

The primary endpoint is the original loss action. Avoiding that loss can mean
delaying it; two of the 25 successful branches lose a life later within the
64-action follow-up. Reward accumulation stops at an earlier teacher loss. Neither the summed
reward difference nor either survival
fraction is a new game score or an estimate over arbitrary states. A stronger
controller could rescue teacher failures. Changing the complete action sequence
cannot distinguish representation, forecasting, actor fitting, timing or training
coverage as the cause, nor identify a particular erroneous command.

This supports examining how the learned policy represents and chooses actions
on these later-play states. Simply assuming that the teacher has no useful
continuation there is not supported by these interventions. No new learner,
loss, teacher change, extension or automatic successor is selected.

## Completed checks and evidence

The parent replay reproduces all 16,613 source actions / 66,440 raw frames in
the eight episodes, including reconstructed input/output pixels, rewards, frame
counts, boundaries, scores, life events and existing first-pyramid milestones.
Every recorded branch reproduces its saved trajectory through the loss. Process
copies preserve sticky-action memory; serialized parent emulator and teacher
states remain unchanged after every branch. Reversing branch order at one
origin in every lane reproduces both branches exactly.

Life gains and losses are counted separately at each raw frame. The continuation
test explicitly covers a bonus followed by a loss within one action, whose net
life count is unchanged. All eight CPU tests pass. Teacher branches execute
7,260 frames; recorded validation and reverse-order checks execute another 6,388.

All 309 diagnostic inputs and 265 protected Breakout inputs pass. The original
Breakout learner has already finished successfully when this diagnostic starts;
its original invocation and terminal state remain unchanged. The 63.910s
analysis (65.230s controller wall) completes 05:16:04 UTC November 1, source
`1a4fffe`, under both caps. It uses one low-priority CPU core at the highest
allowed affinity index. Brief overlap with Breakout's CPU postprocessing and
terminal confirmation remains recorded; no training-time correction applies.
No native model, GPU work, learner update, failure or retry occurs.

Root: `/x/Temp/kindle-qbert-recoverability-20261101`. Full origins, child
trajectories, per-lane results, test/preflight records and completion receipt
remain there. The source life replay and learner checkpoint are unchanged.

- Declaration: `8f6b54abe906337d1ece1efd6ba050a5eec20f513982f61635e69001efdb9c03`.
- Review: `8b7d327bb05dd5eecd6506e8e55198544c5035128a7e61928e8a04375c9392fd`.
- Origins: `eeed6e6a1fd06d418958b1a34b78472194c912a24065eecc84290e23cf3afae6`.
- Completion: `4e2e8a752aec75585f9bd541fa10afa17d95f084b69c921d508a7e940510881f`.
