# Breakout command semantics

The repaired CPU/emulator check finds **608/608 nonidentity comparisons exact**:
384 constant aliases,192 within-class cycling plans and32 mixed plans. All832
rollouts complete16 actions;224 identity comparisons are bookkeeping only.
The18 full Atari commands support six physical classes over these fixed states.
Every pair of canonical controls differs on at least one parent; preserve
movement+FIRE rather than collapsing to the four-command minimal action set.

Use three proposed Boolean submitted-command features LEFT,RIGHT,FIRE, padded
to18 channels. This encodes controller knowledge,not observed objects or emulator
state. No feature implementation,world-model query or learning benefit follows
from the semantics check itself. The complete world-model goal remains unproven.

## Evidence and limits

The [protocol](../experiments/2026-11-09-breakout-world-action-features.md) fixes
32 parent selections from eight retained fitting/waiting lanes: eight each at
contact, before first brick reward, before first life decrement and clean wait.
There are28 distinct retained16-image history hashes among the32 selections.
These are reused development states,not untouched families. Each of26 plans per
parent constructs a fresh environment with the exact original reset seed and
raw-action prefix. Last16 available retained native hashes,source rewards,
terminals,lives and cumulative-frame differences match before every branch.

The wrapper uses full18,sticky.25,repeat4,no random reset NOOPs,native max-pooled
210×160RGB,no resize and a108,000-frame natural-episode limit. Exact equality
covers every branch arrival's RGB/RAM SHA256,reward,terminal,truncation,lives
and executed-frame count. New traces include208 positive-reward arrivals and326
life decrements,with no game-over or truncation; alias repeats are correlated,
not independent event tests. No game-over equivalence is established here.

| Canonical pair | Different parents /32 |
|---|---:|
|0 vs 1|19|
|0 vs 3|30|
|0 vs 4|27|
|0 vs 11|32|
|0 vs 12|30|
|1 vs 3|32|
|1 vs 4|30|
|1 vs 11|30|
|1 vs 12|27|
|3 vs 4|32|
|3 vs 11|20|
|3 vs 12|32|
|4 vs 11|32|
|4 vs 12|23|
|11 vs 12|32|

Canonical indices0/1/3/4/11/12 mean NOOP/FIRE/RIGHT/LEFT/RIGHTFIRE/LEFTFIRE.
Within each class,UP and DOWN variants remain exactly matched here. Clamping
or a ball already in play can make distinct controls locally indistinguishable;
finite matched states do not prove universal equivalence.

The [official ALE paddle-input source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/4f0ee906488d1b6521f21cc61943022c5db0d496/src/ale/environment/ale_state.cpp)
handles horizontal direction and FIRE separately. That inspected commit is not
asserted to match the installed wheel. Installed ALE reports0.12.1+8a8fafb;
Gymnasium1.3.0,NumPy2.5.2. ROM SHA256
`376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`;
ALE binary SHA256
`c6e68f06bf2d23f4c6df8758ef28b108978e332cc27dbd91d05c4d12271230dd`.

## Execution and retained failure

Source declarationdd3ee3a8;separate repair3da9670f. The v1 pin verifier fails
on a string/Path mismatch before imports,emulator construction or rollout-file
creation. Its script,declaration and traceback remain unchanged. The separately
declared v2 corrects Path conversion and traps the actual Kindle constructor
export names directly,with new logs and outputs. Never relabel v1 successful.

The repaired run finishes in111.163s,exit0,under its600s cap. All162 execution
pins match before and after;separate CPU analysis independently recounts832
traces,608 comparisons,480 canonical pairs,prefix hashes and all frame/action
counts. Terminal inputs are pinned and rechecked around analysis. No GPU/model
construction,forecast,optimizer step,checkpoint write or old evidence change.

| Work | Aggregate actions | Emulator frames |
|---|---:|---:|
| Exact source-prefix replay |76,934|307,736|
| New branches |13,312|53,248|

There are832 emulator resets. Runtime,source,data,plans,transcripts,original
failure,audit and closure remain outside Git at
`/x/Temp/kindle-world-action-features-20261109/semantics`.

The six classes now justify a separately declared default-off native feature-table
qualification. They do not diagnose the existing model's failure or establish
physical forecasting,event calibration,brick memory,late-game coverage or fresh
confirmation. Existing learners remain closed;no learning extension is selected.
