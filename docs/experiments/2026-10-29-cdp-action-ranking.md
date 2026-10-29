# Compare predicted action-block rankings with actual emulator branches

World forecasts and live critic calibration both show useful but imperfect
learning. Directly test whether the retained Qbert12M model distinguishes the
outcomes of alternative controls. This is a frozen diagnostic in the existing
five-game goal; no new learner or training objective is selected.

## Fixed data and interpretation

Use the retained expert100/seed4013 greedy evaluation, source trial
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/qbert-12m-48-b16-r16-expert100-200k-seed4013`.
Keep its eight environment seeds, sticky.25, repeat4, full18 and natural episodes.
Select four evenly spaced valid origins per phase per test episode (the second
and third natural episode per lane), on the existing stride16 forecast grid.
The fixed selection has128 origins:64 before/64 after the first pyramid. Origin
selection does not inspect rewards, model predictions or branch outcomes.

For each of18 actions, repeat that action for four policy steps (16 emulator
frames), then use the recorded continuation through15 total policy steps. A19th
plan uses the exact recorded15 actions for replay/prediction parity. Holding a
short block gives a control a chance to affect Qbert between jumps. These are
short open-loop control plans, not the full return under a policy that adapts
to changed observations. Actor probabilities for the first action are therefore
descriptive context, not a direct test of optimal full-policy action selection.

The actual outcomes share the origin's complete emulator/RNG state. They are
one paired stochastic realization per candidate, not repeated estimates of its
expected return. Initial and continued emulator pixels are unavailable to the
model; it receives only the current replayed observation history and candidate
action plans. Oracle generation and native capture are separate processes.

## Completed CPU oracle check and preserved first failure

The first cloneSystemState/restoreSystemState prototype fails source replay on
stream0 at action index3879 after15 branched origins, despite matching serialized
state. A source replay without branches matches all6032 actions. Inspection of
ALE0.12.1 shows sticky previous-action members in `StellaEnvironment::act` outside
the fields saved by `ALEState`. Do not use the failed branch outcomes or modify
the pinned runtime. Original source and log remain in the external evidence root.

The replacement uses one sequential CPU child per plan via process copying,
which preserves the previous-action memory as well as serialized state and RNG.
It refuses a parent with an open GPU device. No GPU context is created in this
CPU process. Every candidate starts from the same unmodified parent; the source
emulator advances only through its recorded actions.

The CPU oracle completes48256 source actions and154607 additional branch frames
in26.878s, including reverse-order checks at eight origins. All128 recorded-plan
branches match actual rewards, boundaries and frame counts. The entire source
replay matches after branching, all parent snapshots remain unchanged, and
reversing candidate order produces identical outcomes. The original failure is
retained; no GPU stage has run or repeated at this point. Source references:
[sticky controls](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/environment/stella_environment.cpp),
[state serialization](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/environment/ale_state.cpp).

## Selected native captures

Use unchanged native447fe50 and the matching Breakout-era runtime. Run one
128-action execution check with all19 plans at origin0, capped at120s. Require
finite predictions, state/probability/counter invariance across forecasts, exact
repeated-plan predictions,357 invariant tensors, zero updates and host guard.
After it passes, capture the128 selected origins while re-observing all48256
recorded actions, capped at600s. No training experience is reused.

Pin the native, runtime, source, source logs/replay, fixed selection, completed
CPU oracle, original forecast arrays and analysis before GPU work. Require
matching input-pixel hashes across all eight CPU/native streams, exact source
transitions, all357 unchanged tensors, original counters and guards. Every
recorded-plan reward/continuation prediction must exactly match the previously
completed forecast diagnostic at the same origin. No retry or host recovery.

## Ranking measurements

Primary modeled score is the15-step external reward sum weighted by cumulative
predicted continuation (which includes gamma). Actual score uses gamma-discounted
rewards and stops at a true terminal/cutoff. Compare the18 alternative blocks,
excluding the recorded-plan parity control from the candidate set. Report true
outcome spread, informative-origin count, pairwise agreement on unequal actual
outcomes, selected actual return, regret versus the best candidate and gain over
uniform action selection. Model ties receive their average actual outcome and
half pairwise credit. Do not call tied actual outcomes successful discrimination.

Report all/before/after strata, with lane-bootstrap intervals for mean gain over
uniform (5000 resamples,seed20261029). These conditional intervals do not establish
learner reliability or average over independent emulator futures. Model action
blocks and scalar actor preference are diagnostics, not a planner deployed into
the live game. Weak ranking would motivate investigating action-conditioned
world losses; useful ranking would shift attention toward behavior learning and
longer-horizon objectives. Neither observation alone establishes a causal cure.
