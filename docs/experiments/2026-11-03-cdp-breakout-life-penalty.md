# Optional sparse life-loss feedback: bounded Breakout check

**Complete:** the native check passes8,192 actions/489 updates in47.317s; all408
pins,357 matchedinitial/invariant tensors,118 independently replayed penalties,
raw scores andguards pass. Frozen1,024/.891s haszero updates/episodes. Original
main/collector succeed; CPU fixture/path/terminal-reader failures arepreserved.
No originalstage repeats. The root isclosed; do not repeat stages orreaders.
Separately select the [fresh200k comparison](2026-11-03-cdp-breakout-life-penalty-200k.md);
full preparation andlaunch remainunfinished. See the
[implementation result](../results/2026-11-03-cdp-breakout-life-penalty.md).
The original bounded requirements and selection state below are retained.

Select development and qualification of one default-off training aid after the
[saved reward-visibility result](../results/2026-11-03-cdp-breakout-life-feedback.md).
Its120 selected lost lives all have zero immediate reward;96 do not end the
game. This motivates an experiment, not a missing-loss bug or causal diagnosis.
The replay16k comparison isclosed; retain shared replay100k andallother settings.

For each decision in fresh Breakout training, use
`training_reward = raw_reward - max(0, lives_before - lives_after)`.
The coefficient isfixed atone point per observed lost life for this check.
Read ALE life counts only for training-reward accounting; the actor receives
the same pixels. Use net changes across the existing repeat4 decision, including
game-over losses. Bonus-life gains have no penalty. Reset the life baseline
after a natural episode reset, without treating the reset as a gain or loss.
Keep actual termination/truncation flags and belief-reset behavior unchanged.
No artificial life terminals, serving aid or reward for remaining idle.

This changes the training reward andmust be disclosed as an explicit aid.
Preserve raw rewards in transition logs, online returns, task scores and replay
comparisons. Record a separate declared shaping contract, before/after lives,
loss counts and actual training rewards. Require native stored extrinsic reward
to equal the declared training reward with explicit F32 handling. Reject
undeclared fields, inconsistent life sequences, incorrect arithmetic and
misaligned resets. The existing raw-reward audit rule cannot silently accept
changed inputs. A dedicated qualified Python overlay may extend that contract;
do not mutate any retained runtime or historical source.

The aid must default off and be refused for frozen evaluation, incompatible
games or restored training. With it off, preserve existing execution and log
semantics. All frozen assessments use raw rewards and no shaping or guidance.
Native weights/configuration, learning and inference remain onMeganeura/Blade;
this is game-adapter feedback andneeds no native implementation or build.

Keep centeredCDP12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16,lambda.95,
encoder/base4e-5,behavior8e-5,dynamics4e-4,replay-value1,expert100/unit weights/
world gradients/all-state corrective labels,sampled collection/posteriors,
actor-unimix.01,action-effects1,replay100k,warmup1000,20% episode guidance/cap1920,
learnerseed1009/evaluation4000001009,full18/sticky.25/repeat4/noops0,native pooled
RGB andone GPU64 resize. Native447fe50/SHA35ad0db andteacher stayfixed. Use the
exact retained runnercf4ea238 as the overlay base, with only declared reward
handling changes. No prior weights,replay,belief,games,RNG or experience transfer.

Before GPU work, qualify actual command parsing and refusals, raw score versus
training reward separation, life loss/gain/terminal/reset/F32 cases, default-off
compatibility, log auditing and the new replay checks. Pin sources and runtime
packaging. Original control scores70.750/143.667 and its338 pinned inputs remain
unchanged; inspect live state before allocating one bounded check.

Then train one fresh8,192-action check, expected489 updates/firstupdate376,
within120s, and retain actual-initial/final weights. Require357 initial tensors
matching control, exact intended config/header, finite negative training rewards
at observed losses, finite learning reports, changed CNN/reward/actor/value,
full raw pixel/action/label replay plus independently verified shaping fields,
and host guards. Verify raw online and task accounting separately from shaping.
Restore the final weights for1,024 greedy frozen actions within60s, shaping and
guidance off, zero updates and357 invariant tensors. It is not a competence test.

Retain existing bounded CPU audit/replay/service envelopes and stop on genuine
GPU faults or numerical failures. Require training seconds×200,000/8,192×1.10
≤1,800s before separately selecting a full learner. No coefficient sweep, retry,
cap increase, extension, automatic promotion or overlapping GPU work. Any
failure remains preserved; inspect evidence before a separate CPU-only repair.

Prospective root `/x/Temp/kindle-breakout-life-penalty-20261103`. At selection,
implementation and GPU qualification are unfinished; no learner is running.
The five-game goal and original90%-two-wall acceptance gate remain unchanged.
