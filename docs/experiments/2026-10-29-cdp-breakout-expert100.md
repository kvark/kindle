# Stronger Breakout supervision at the retained cheap budget

The corrective 4M64 learner scores4.292 sampled/5.000 greedy in17m25s, versus
own initials1.375/2.583, with zero two-wall successes. Raising the encoder rate
lowers sampled quality to1.458. Retained-label analysis finds only41.753% exact
teacher agreement on passive actions in the original learner's final50k window.
That changing training cohort does not establish teacher correctness, a causal
bottleneck or frozen action accuracy. It motivates one supervision test.

Select one fresh bounded check changing only **expert imitation scale10→100**.
Keep centered CDP,4M64/Auto/N8/B8/T16/H15/R16/coefficient1/actor-unimix.01/full18,
base/actor rate4e-5,encoder6e-6,dynamics4e-4,warmup1000,replay100k. Keep20%
episode guidance/cap1920/all18 action weights1/expert-world gradients/all-state
corrective labels and the unchanged edge-reset feedback teacher. Native447fe50,
the matching Breakout runtime and the retained control's runner remain fixed.
Use native max-pooled frames with one GPU resize, sticky.25/repeat4, seed1009
and evaluation4000001009. These are reused development inputs.

The coefficient affects actor and world supervision together. The completed
Qbert scale1000 regression supplies no reason to extend Qbert or assume a
universal coefficient. This is a separately selected Breakout comparison.
All preceding services are terminal success and the GPU is idle at selection.

## Bounded execution and cost prerequisite

Train8192 actions/994 updates, first update248 and zero residual credit. Restore
for1024 greedy frozen actions with zero updates. Require357 control-matching
initial and invariant frozen tensors, exact source/config/counters, finite
reports, changed CNN/reward/actor/value, equal labeled-row/weight sums, aligned
raw world/actor expert losses, passive corrective labels, full training
label/input-pixel/action replay and both AMD host guards. No natural-episode
competence is inferred from this check.

Caps:120s training,60s frozen,180s CPU audit,420s whole service. Before observing
cost, fix the conservative full-budget rule to
`check_training_seconds * (200000/8192) * 1.10 <= 1500`.
All execution checks and this25-minute cost gate must pass before separately
selecting a full learner. Failure or an inconclusive prerequisite ends this
candidate; no retry, coefficient replacement, cap increase or backend campaign.

Root `/x/Temp/kindle-breakout-expert100-20261029/native-check`;
service `kindle-breakout-expert100-check-20261029`. No full learner is selected
at this stage. Check live state before GPU work.

## Fixed assessment if a full learner is separately selected

One fresh200k-action/24,970-update life,25-minute training cap,100k/12,470
recovery. No check weights, experience, replay, belief, games or RNG are reused.
Assess only fixed final and own actual initial in sampled and greedy modes,
sampled primary: first3 natural episodes per8 lanes=24 per arm,600k-action/300s
caps,guidance and labels off,zero updates/truncations and357 invariant tensors.
The original90%-two-wall gate remains unchanged; report scores separately.
Keep excess episodes and partial tails. Ordinary frozen logs lack live pixel
hashes; outcome replay cannot prove equality to unavailable live images.

Compare with the retained scale10 learner at the same experience/update budget.
Require matching initial tensors and paired initial semantic traces, source/
config/counters,100k recovery, finite reports, full training and frozen replays/
videos, guards, curves, exposure and comparison. Guided online returns are not
unaided competence. No checkpoint selection, extension, retry, cap increase,
coefficient sweep or automatic successor follows. Any useful gain remains
development evidence until separately confirmed on another fresh learner.

## CPU preparation

Preflight passes in.585s. Both actual CLI paths validate and stop before output,
environment or native construction. Eight checks exercise the real audit config
assertion, including rejection of the old scale, changed rates, ratio, capacity,
batch, world-gradient flag and action weights. Replay-readiness accounting checks
first248 and994/12,470/24,970 updates at8192/100k/200k actions with zero credit.
All64 retained Breakout and229 preceding Qbert pins verify; the357-tensor initial
schema verifies. Six controllers/helpers compile. No preparation failure or
native/runtime change occurs. The exact control runner is copied unchanged.
