# More Qbert experience with similar replay work

The compatible RGB comparison favors the retained CDP recipe on the selected
development cohort. It does not pass Qbert's full gate or automatically select
another learner. This separate decision targets the remaining mean-score gap:
the strongest12M48/B16/R16/expert100 learner scores11,984.375 with24/24 first
pyramids after200k actions/12,477 updates in30m27s.

Select one bounded check for a fresh12M48/Auto/N8/B16/T16/H15/**R8** learner.
A later full allocation may be selected only after that check passes. Its fixed
candidate budget is400k actions/12,489 updates, with a **2700s/45-minute training
cap**. Doubling experience and halving replay ratio preserve approximately the
same total replay work, while changing optimizer cadence, warmup duration in
actions and visited experience. This is not an isolated ratio or experience
effect, matched wall time, or an extension of the completed life.

The retained200k life spends1185.820s learning and641.165s elsewhere. Holding
learning time fixed while doubling the latter gives an early2468.150s (41m08s)
projection. This is a feasibility estimate, not a measured speed or quality
prediction. The earlier4M100k/R32→200k/R16 comparison improves greedy score
8365.625→9654.167 at approximately equal updates, but does not guarantee the
result under12M/R8. No new code, backend work or pretrained encoder is needed.

## Locked settings and bounded prerequisite

Keep centered CDP/coefficient1/actor-unimix0/full18, original base/actor4e-5,
encoder6e-6, dynamics4e-4, warmup1000, replay100k, native max-pooled frames with
one GPU resize, sticky.25/repeat4. Keep20% episode continuation guidance,
cap1920/expert100/all18 weights1/world gradients/all-state corrective labels.
Seed4013/evaluation4000004013 are reused development seeds. Native447fe50,
matching Breakout-era runtime, runner and teacher are unchanged. The only saved
configuration change relative to the retained12M control is `train_ratio:16→8`.
No weights, replay, belief, game state, RNG or check experience are reused.

Check8192 actions/245 updates, first update376, residual credit.25; then1024
greedy frozen actions with zero updates. Require357 matching initial and
invariant frozen tensors, exact source/config/counters, finite reports, changed
CNN/reward/actor/value, equal labeled-row/weight sums, aligned raw world/actor
expert losses, full training label/input-pixel/action replay and both guards.
No natural-episode competence is inferred from this bounded check.

Caps:180s training,60s frozen,180s CPU audit,480s whole service. Before seeing
the R8 cost, fix the conservative estimate to
`check_training_seconds * (400000/8192) * 1.10 <= 2700`.
Require every execution check and this cost gate before separately selecting
one full learner. Failure or an inconclusive prerequisite ends this candidate;
no retry, cap increase, replacement ratio or backend qualification campaign.

## Fixed full-run assessment, if separately selected

Train fresh to400k, save/audit100k/200k/300k recovery checkpoints at3114/6239/9364
updates, and evaluate only the fixed final and own actual initial. All milestones
and final have residual credit.25. No checkpoint selection or warm-start resume.
Four arms retain sampled and greedy actions, greedy primary: first3 natural
episodes per8 lanes=24 per arm, seed4000004013,600k-action/300s caps, guidance/
labels off, zero updates/truncations and357 invariant tensors. Full outcome
replays/videos and training label/pixel/action replay remain required.

Primary success remains the original mean15000/90%-first-pyramid gate, with
fixed final quality and training cost reported against retained12M200k/R16
(greedy11984.375/24, sampled6759.375/24,30m27s). Report own initials and full
exposure. A weaker or incomplete result does not justify extending the budget,
selecting an intermediate checkpoint or trying another nearby ratio automatically.
Even a gate pass requires a separately selected fresh learner before a claim of
seed reliability. No full learner or successor is selected at this bounded stage.

The completed RGB comparison's negative/inconclusive stop was not triggered.
Its positive result is narrower than task success and does not change any gate.
This decision keeps the five-game goal intact: Qbert and Breakout remain open.

Root `/x/Temp/kindle-qbert-expert100-r8-20261029/native-check`.
Check live state before GPU work; all preceding services are terminal at selection.

## CPU preparation

Preflight passes in.565s:two actual CLI validation paths stop before output/
environment construction,five config checks exercise the real audit assertion,
and independent readiness/credit accounting verifies first376 and all declared
milestones. The retained357-tensor schema and all190 control/195 preceding RGB
pins verify. Capability metadata is stubbed only for the CPU boundary cases;
no native module,agent or environment is constructed. All copied controllers
compile. No native edit,build,preparation failure or GPU stage occurs here.
