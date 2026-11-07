# Fixed-plan Breakout world-model challenge

The [closed motion-weighted learner](../results/2026-11-07-breakout-world-motion-attention.md)
passes observed-ball reconstruction but drifts in forecasts. Select a fresh,
bounded **development diagnosis**, with no training or actor/critic. Ask whether
its frozen forecasts predict the consequences of commands selected before the
future unfolds. Recorded future teacher commands were retrospective controls.

## Fixed allocation

Root `/mnt/data/kindle-breakout-world-branches-20261107`. Use the unchanged
motion native/source3fa37b6d, SHA256
`eafe8359ae5019ea0a5a7517b7acfe75aacaf64b4e84a3bcd9ec4548cf939dc5`, and its
matching runtime. Python collector/evaluator is
`python/examples/breakout_world_branches.py`; the execution receipt pins copied
sources, helpers, runtime, checkpoints, package versions and commands before work.

Keep sticky probability.25, repeat4, full18 commands, native210x160 max-pooled
RGB, one native GPU resize to64, and eight independent seed streams per group.
Model inputs are submitted commands and images only. RAM selects/audits origins;
snapshots, RAM, categories and future outcomes never enter the learner.

Seven groups: serve, upward free flight, descending paddle approach, upward brick
approach, side-wall approach, descending multi-life state, descending last-life
state. These are origin categories, not guaranteed future events. Exact predicates
are fixed in source. Seeds are `251001+10000*group+1009*lane`; source collection
caps at1,024 ticks per lane. Use the fixed pixel controller, except holding
LEFTFIRE for the life-loss/ending source groups. Retain32 observed arrivals and
the complete command/reset transcript. Stop on missing origins; save partial
states and transcripts without automatic recollection.

Serve has18 plans, each holding one command. Other groups each have NOOP, LEFT,
RIGHT, and RIGHT for4 then LEFT for28. All plans last32 decisions and are fixed
before seeing any branch outcome. Thus56 origins and42 plan batches. For each
origin/plan, collect8 oracle repetitions; retain native images at1/4/8/16/32,
the full first trace, all rewards/endings/lives/positions and validity masks.
Stop at natural game-over; do not reset inside a branch. Count source, oracle
and source-replay actions/emulator frames separately, and report wall time.

ALE v0.12.1's [clone/restore implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/environment/stella_environment.cpp)
does not serialize its private previous sticky action. Therefore capture the
desired physical/RNG snapshot, reset/replay the original seeded source transcript
to reconstruct that private state, verify exact original full ALE state, then
restore the desired snapshot. Repeat0 uses the original RNG; subsequent repeats
continue the preceding oracle RNG stream. This retains the same physical origin
and action history while varying future randomness. Repetitions share an origin;
they are not eight additional independent examples. Save transcript and wrapper
state as well as full/RNG-excluding ALE snapshots.

## Frozen evaluation and interpretation

Evaluate the actual-initial checkpoint with history16 and the final8,192-update
motion checkpoint with histories16 and32. History16 is primary; history32 is a
separate warmup/memory diagnostic. Use8 fixed model draws, seeds983001+1009*i.
No best-draw/checkpoint selection. Preserve all204 parameters/moments and the
update counter exactly before/after each evaluation. Use the existing AMD host
guard, one job at a time, initial/final timeouts1,200/1,800 seconds. No new native
qualification or rebuild is needed. Stop on a real fault or numerical failure.

Score each horizon/origin/plan separately before class aggregation:

- Finite-mixture energy scores for observable ball/paddle features, including a
  distinct absorbing game-over atom weighted by predicted survival. Compare
  always-alive image persistence and simple constant velocity without reflections.
- Report oracle pairwise dispersion as a noisy entropy reference; eight draws
  cannot establish distribution calibration. The ball detector measures a
  supported below-brick red sprite, not universal ball presence.
- Brier scores for game-over and any positive reward by horizon; cumulative
  reward error. Remove gamma from continuation before multiplying survival.
  Reward at the terminal decision is retained. This uses the model's conditional
  head factorization; it does not establish exact reward/termination dependence.
- Alive-only brick-occupancy Brier against native cell labels, excluding ambiguous
  half cells. This is a conditional visual diagnostic, separate from the joint
  survival score. Export RGB forecasts and posterior reconstruction for review.

Compare action-plan contrasts and identify free-flight drift, contact timing,
response direction, brick memory, serve/restart and life-loss/game-over errors.
Longer context improvements suggest initialization/history sensitivity; they do
not by themselves isolate a memory-capacity cause. A fresh seed challenge selected
after inspecting development is not an untouched test. Event categories have
deliberately altered frequencies, so do not claim natural-frequency calibration.
Eight source seeds per class and small oracle samples do not establish reliability.
No blanket pass threshold is inferred from one mean; use these failures to choose
one bounded training/data change. No successor learner is preselected.

## Preparation evidence and preserved failure

Eight CPU tests pass: score spread/location, terminal reward/absorption, native
brick axes/ambiguity, eight-frame resize contract, fixed command plans, origin
boundaries, snapshot file roundtrip, and absorbing endpoint scoring. The original
CPU clone fixture reproduces two complete traces exactly and exposes5 unique
traces in16 RNG-excluding repetitions; it did not check the private sticky action.

The separate sticky-state fixture **fails its exposure assertion**: none of16
four-frame-pooled raw trace pairs differ. Preserve that failure and all arrays.
The separate CPU-only `sticky-review.json` verifies all16 replay-repaired native
trace pairs exactly; source replay also verified full original ALE state. This
does not empirically demonstrate a raw-clone defect or relabel the fixture as
passed. No GPU/emulator construction occurs in the repair review; no fixture is
rerun. Replay reconstruction is retained as the source-supported correction.

Later-game/tunnel/second-wall states remain outside this challenge. Passing early
reconstruction or outperforming controls here does not prove the full world,
CDP superiority, planning competence or gameplay. The overall goal stays active.
