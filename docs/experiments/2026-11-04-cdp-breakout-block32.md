# Bounded Breakout check with intermittent teacher control

Select one fresh execution and cost check after the completed
[teacher takeovers](../results/2026-11-04-cdp-breakout-command-takeovers.md).
The existing teacher survives32 decisions in84/120 sampled current-policy loss
states. The candidate teacher fails its fixed improvement screen and stays parked.
Increasing whole-episode guidance previously lowered frozen gameplay. These
findings motivate testing when the existing teacher takes control; they do not
establish that this schedule will improve learning.

Replace20% whole-episode selection with independent50% selection for consecutive
32-action blocks within each episode. Keep the1920-action episode-age cap: no
override at or after that age. Each lane retains its seeded RNG; reset only
completed lanes. Draw a new block when its first action is requested, so terminal
actions do not consume phantom next-block draws. Teacher history continues
through both native and guided blocks, with all-state corrective labels and
actual-executed-action feedback. Episode `guided` means at least one selected
block; separately report started/guided blocks and actual overridden actions.
Selection probability is not the realized intervention fraction. The probability
and temporal distribution change together; no isolated causal claim is intended.

Keep native447fe50/SHA35ad0db and the retained teacher's exact production classes
and coefficients. Package the old qualified runtime with only the new Breakout
schedule and audit recognition, plus the block option in runnercf4ea238. The
canonical implementation also preserves minimal-action support, unused here.
No native rebuild or optional loss/runtime overlay. Preserve the original
episode scheduler. Require CPU default-schedule equivalence and packaged block,
feedback, reset, cap, CLI and accounting checks before GPU work.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
encoder/base4e-5,dynamics4e-4,behavior8e-5,policy1/critic1/replay-value1,
disagreement1,expert100/unit weights/world imitation gradients, sampled
collection/posteriors,actor-unimix.01,replay100k,full18/sticky.25/repeat4/noops0
and native pooled RGB with one GPU64 resize. Reuse development seed1009 with
fresh experience. No RAM, life counts or future outcomes enter the policy.

Train8192 aggregate actions/489 updates, first update376 and residual credit.5,
within180 seconds; save actual initial/final checkpoints. All357 initial arrays
must match the retained current-critic control. The first32 lane decisions match
the previous50%-episode bounded run before any block resampling; compare those
semantic events, but do not require all47 pre-update events to match. Require
finite reports, correct cadence/label bounds, changed CNN/reward/actor/value,
complete training pixel/action/label replay and independent block accounting.

Restore the final checkpoint for1024 frozen greedy actions, seed4000001009,
guidance/labels off, within60 seconds. Require zero updates and all357 arrays
unchanged, exact source/native/ROM identities and host guards. This is a bounded
execution check, not a natural-episode competence result. Replay cap120 seconds,
CPU audit180, main600 and terminal observer650. Qualify both actual queued
commands, replay argv and terminal identity handling on CPU before launch.

Measure training time against the retained47.689684s bounded control. A padded
200k projection `seconds * 200000 / 8192 * 1.1` must fit30 minutes before selecting
a full learner separately. A full comparison would use fresh experience and four
fixed24-natural-episode final/actual-initial arms, greedy primary against143.667
and sampled secondary against70.750. Original864/90%-two-wall gate stays fixed.

Root `/x/Temp/kindle-breakout-block32-20261104`, service
`kindle-breakout-block32-check-20261104`. Verify no other GPU allocation before
launch; stop on real faults or nonfinite reports. No repeated successful stage,
cap increase, sweep, full learner, extension or automatic successor is selected.
The five-game goal remains unfinished.
