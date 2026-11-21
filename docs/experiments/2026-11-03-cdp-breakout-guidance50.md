# Bounded Breakout check with more guided episodes

Separately select one fresh execution and cost check after the completed
[current-critic sequence comparison](../results/2026-11-03-cdp-breakout-value1-context32.md).
That comparison misses its primary improvement rule; its original services
and collector are closed. Do not repeat them or carry their experience forward.

Raise episode guidance probability from 0.20 to 0.50 in the retained B16/T16
recipe. This tests a different visited-state mix. The retained teacher-guided
returns exceed unaided returns, and later-score native experience is scarce;
neither observation proves that more guidance will improve the learner.
Earlier longer-cap and greedy-collection trials remain negative or mixed.
Keep the existing pixel teacher and 1920-action cap. All actions already receive
corrective labels; this changes actual interventions, not label availability.
Episode selection probability is not the fraction of overridden actions.

Use the unchanged native447fe50/SHA35ad0db, runnercf4ea238 and qualified runtime
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Keep centered CDP,
12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1, encoder/base4e-5,
dynamics4e-4, behavior8e-5, policy1/critic1/replay-value1, disagreement1,
expert100/unit weights/world imitation gradients, sampled collection/posteriors,
actor-unimix.01, replay100k, full18/sticky.25/repeat4/noops0 and native pooled RGB
with one GPU64 resize. No build, production-code change or prior experience reuse.

Train fresh learner1009 for8192 actions /489 updates, first update376 and final
residual credit.5, within180 seconds. Save actual initial and final checkpoints.
Require all357 initial tensors to match the retained control. Guidance changes
before the first update, so equal training trajectories are not expected.
Independently verify the first-episode selection mask, complete episode/reset/
cap accounting and full training pixel/action/label replay. Require finite
reports, correct cadence and expert-row bounds, and changed CNN/reward/actor/value.

Restore the final checkpoint for1024 frozen greedy actions with evaluation seed
4000001009, guidance and labels off, within60 seconds. Require zero updates and
all357 tensors unchanged, exact native/source/ROM/host guards and full accounting.
This short arm is an execution check, not a24-episode competence assessment.
Replay cap120 seconds, CPU audit180, main600 and terminal observer650.

Qualify both actual queued commands on CPU before launch. Resolve native metadata
before mocking all native agent exports. Test the guidance probability and cap,
episode selection/reset/cap behavior, config/cadence/label contracts, checkpoint
hashes, actual replay argv/manifest paths and terminal identity handling. No
real agent, emulator or GPU descriptor may be created by these command fixtures.
Preserve exact historical sources needed by retained pins; do not rewrite prior
declarations or failure receipts.

Compare measured training time with the retained8192-action control47.689684s;
record construction separately. The padded200k projection is
`training_seconds * 200000 / 8192 * 1.1` and must fit30 minutes before a full
comparison can be separately selected. Passing execution/cost does not promote
the recipe. A full comparison would use fresh experience, four fixed24-natural-
episode final/actual-initial arms, greedy primary against143.667 and sampled
secondary against70.750, with the original864/90%-two-wall gate unchanged.

Root `/x/Temp/kindle-breakout-guidance50-20261103`; service
`kindle-breakout-guidance50-check-20261103`. Check the host is idle before launch.
Stop on actual faults or nonfinite reports. No overlapping allocation, repeated
successful stage, cap increase, probability sweep, full learner, extension or
automatic successor is selected. The five-game goal remains unfinished.
