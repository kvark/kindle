# Guided Qbert capacity with B16 and R16

Select one bounded12M/RGB48 execution and cost check after the completed4M/R16
expert-world comparison. Its frozen3670.833/19of24 first pyramids is a small gain
over3333.333/18, below stronger5%/R32's4730.208/23 and the mean15000/90% gate.
No unchanged extension or confirmation follows that small result.

## Evidence and hypothesis

A CPU-only analysis of the fixed24 frozen episodes in each retained4013 arm
joins recorded actions to the already replayed first-pyramid boundary. Among
successful episodes, NOOP does not abruptly increase after that milestone:

| Recipe | First pyramids | NOOP before | NOOP after | After excluding first240 frames |
|---|---:|---:|---:|---:|
|4M/R16/world gradients|19/24|76.41%|74.80%|74.67%|
|4M/R16/control|18/24|74.11%|75.02%|73.26%|
|4M/R32/5% control|23/24|68.82%|64.91%|65.12%|

All0/120/240-frame delays are retained. The milestone-crossing action is excluded;
celebration and bonus timing may remain. This describes actions, not position,
teacher agreement, visual sufficiency or a cause of weak later play. No native
module, environment or new policy decisions are used. Source hashes and every
selected episode's counts are retained under `/x/Temp/kindle-qbert-phase-20261028`.
The initial read-only inspection accidentally printed many transition rows;
the bounded analysis output replaces that noisy inspection, without changing
any source run or experiment input.

Source review also confirms that expert labels currently accompany executed
teacher overrides; unguided policy trajectories receive no expert targets.
Neither fact diagnoses the gap. Greater model capacity may help learn a state
representation and policy that use the existing guidance. This is unproven:
the earlier unaided12M/B16/R32 Qbert result959.375/0 was weak and took32m19s.
Pong's useful12M/B16/R16 result motivates measuring this combination's cost,
not assuming that its task benefit transfers to Qbert.

## Selected bounded check

Use12M/RGB48/Auto/N8/B16/T16/H15/R16/coefficient1/actor0/full18/original rates,
world microbatch16, fresh4013/evaluation4000004013. Keep20% episode continuation
guidance/cap1920/expert scale1/NOOPweight.05 and expert world gradients enabled.
Capacity4M→12M and batch/microbatch8→16 change together; replay positions per
action stay16 while optimizer cadence halves. Do not claim an isolated effect.
Native max-pooled observations still receive one GPU resize.

Train8192 aggregate actions/489 updates, first376/residual.5, within180s. Restore
for1024 frozen actions/60s, guidance off and zero updates. Require finite reports,
positive world/actor expert losses when labeled, zero masked losses and agreement
within.002×(1+absolute actor loss). Verify trusted directional labels and changed
CNN/reward/actor/value parameters without assigning joint changes to one term.
The model has14,123,473 trainable parameters. Check357 finite checkpoint tensors,
11 world actor copies equal to behavior, and357 unchanged frozen tensors.

Use the retained12M48 seed4013 Pong initialization only as a346-shared-tensor
initialization reference: shape/model/seed match, and reward coefficient changes
do not require different initialized tensors. No reference weights are restored,
and this is not a cross-game performance comparison. Exact guided input/action/
outcome replay, source/config/counter accounting and host guards are required.
An8192/1024 check does not measure natural-episode competence.

Reuse `/x/Temp/kindle-expert-world-20261027/runtime-audited`, native`0e157df`,
SHA256`6a9c41c44aef4747cef6504a2525195d8f2f2a62dbece7c6ecc6d1128fff1c1e`.
No native or CLI change/rebuild. The separate checkpoint helper has the audited
lazy import used by the completed CPU repair. CPU preparation parses both exact
commands, compiles4 scripts, checks config against the retained12M initialization
and preserves all52 preceding-run pins, without native/environment construction.
The new declaration pins41 inputs, including schema, source runs and diagnosis.

RX7900XT host/boota8648671/AMD module9BDBC866F55BF296F1B7FA2 remain unchanged.
Use the host guard, one GPU job at a time, and stop on real faults or numerical
failure. No driver reset/reload/reboot. Audit cap180s; whole service480s. No retry,
cap increase, reuse of check experience or automatic full promotion. Review cost
and correctness before separately selecting any full learner.

Trial:`/x/Temp/kindle-qbert-expert-capacity-20261028/reviewed/qbert-12m-48-b16-r16-world-check`.
Owner:`kindle-qbert-expert-capacity-reviewed-check-20261028`.

## Prelaunch accounting correction

A final check against the retained B16/R16 source catches a mistaken first384/
zero-credit assumption: the observed scheduler starts376 and retains.5 credit;
8192 still schedules489 updates. The shell continues past that failed read-only
assertion to the original service, whose unselected-declaration guard refuses
before creating guards, a training log or any GPU stage. Preserve its15:45:31
UTC failed service/journal, original prepared declaration and all40 unchanged
input pins. A separate `reviewed` preparation corrects both assertions, verifies
the reference counters/config and parses both commands again. It pins41 inputs,
including the original failure receipt, with no model/budget/cap change. The
reviewed owner above is a distinct first GPU check, not a repeated GPU stage.
