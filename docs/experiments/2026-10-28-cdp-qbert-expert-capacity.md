# Guided Qbert capacity with B16 and R16

**Full comparison complete:**200k/12,477 updates in24m51s; frozen1792.708/9of24
first pyramids versus own initial177.083/0. Same-native4M/B8 scores3670.833/19
in21m28s. All audits pass. Retain4M; no extension or confirmation. See the
[completed result](../results/2026-10-25-cdp-cheap-qbert.md#guided12m-b16-r16-lowers-quality-and-costs-more).

**Bounded check complete:**8192 actions/489 updates in60.333s;1024 frozen
actions/.810s, zero updates and no natural episodes. All audits pass. The
early200k projection is24.55 minutes, not a measured full-run cost or task result.

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

## Completed bounded result

The reviewed service starts15:48:24 UTC/source`1083a65`, PID169963/invocation
`221aad6399eb4708878555325397b49a`, and succeeds at15:49:49 on October28.
Whole service1m25.463s. Training8192 aggregate actions /32,756 emulator frames /
489 updates takes60.332659s, first376/residual.5. Frozen1024 /4096 frames takes
.809770s with zero updates and no natural completed episodes.

All489 finite reports contain expert labels;58,825 repeated label rows have
total weight15,333.036.4492 trusted actions (54.834%) include3466 NOOP and1026
directional hops. World/actor expert-loss sums161.943426/161.943407 agree, with
maximum per-report difference1.4e−7. All41 input pins,357 finite/invariant
tensors, actor-copy equality,346 shared initialization tensors, changed CNN/
reward/actor/value parameters, full guided input/action/outcome replay and both
host guards pass. The original40 unselected pins and CPU-only failure remain
unchanged; no GPU stage repeats or cap changes.

The early rate extrapolates to24.55min/200k, with later exposure and cost unknown.
It is affordable enough to consider a separately declared30-minute200k screen,
but this bounded check establishes execution/cost only. No full learner,
confirmation or automatic promotion is selected by this receipt. Use the
`reviewed` controller/schema and first376/residual.5 accounting for future B16
preparation; a fresh full life must not reuse check experience.

Selected declaration SHA256
`0098cf7083cf54f419a65fbb16238f41c03a28a4b61b592e65e81005fbd2c27d`;
bounded review`b81334532ca32cbd1d3e1e7e3329af8bee69fc861c315941b9e4a7773952004e`;
completion`0514244aa3718ea64b7ecf549e4b7e99a0c2c21f9d0a167f72eb181e9277b8c5`.
The preceding CPU action diagnostic is SHA256
`90c745723feb77d0999a101fe64a04f194183d34b08291cef8b39923f69a683c`.


## Separately selected200k comparison

The completed60.333s check projects24.55 minutes at200k, within a fixed30-minute
training cap. Select one fresh12M/RGB48/Auto/N8/B16/T16/H15/R16 learner with the
same20% continuation/cap1920/expert1/NOOPweight.05/world-gradient recipe and
original rates. Seed4013/evaluation4000004013 are reused development seeds.
No weights, replay, belief, games or RNG are restored from the bounded check.

Train200,000 aggregate actions/12,477 updates, first376 and residual.5; save and
audit100,000/6227 recovery, then the fixed final checkpoint. Compare against the
completed4M/B8/R16/world-gradient control3670.833/19of24 in21m28s. Native, runner,
teacher, host, geometry, guidance probability, reward coefficient and rates
match that control. Capacity and batch/microbatch change together, halving
optimizer cadence at the same replay positions per action. This is a recipe
comparison, not an isolated capacity effect. Stronger5%/R32/world-off4730.208/23
in25m23s remains secondary context.

Evaluate final and own actual initial with sampled actions, no aid or updates,
first3 natural episodes in each of8 lanes (24 each), no truncations, at most
600k actions/300s per arm. Retain the original mean15000/90%-first-pyramid gate.
Require357 finite/invariant tensors,357 initial tensors matching the bounded12M
initialization and346 shared tensors matching the retained12M reference only.
Different-size initial policies are evaluated separately; neither weights nor
initial action traces must equal the4M control. Preserve guided input/action/
outcome replay, frozen outcome replay/videos, curves, coverage, expert exposure,
source/config/counters and host guards. Guided online returns are not competence.

CPU preflight parses all3 exact commands, validates the observed B16 scheduler,
compiles10 scripts and exercises the completed-control reader in both the normal
and plotting environments without native/environment construction. The plotting
path uses the audited lazy checkpoint import. A copied exposure assertion is
corrected to reconcile report counts against audited accounting before selection;
the earlier preflight receipt is retained. All41 bounded input pins remain fixed.
The new declaration pins63 inputs. Training cap1800s, each frozen arm300s, CPU
audit1200s, whole service3660s. Stop on real faults/numerical failure. No retry,
cap increase, extension, checkpoint selection, confirmation or automatic successor.

Trial:`/x/Temp/kindle-qbert-expert-capacity-20261028/full/qbert-12m-48-b16-r16-p20-world-200k-seed4013`.
Owner:`kindle-qbert-expert-capacity-200k-20261028`. CPU-only recovery and completion
observers cover this declared life; they never start another GPU job.


## Full-run launch and startup

The selected service starts16:09:28 UTC on October28/source`eebf838`, PID171410,
invocation`0a687e9f00f747189c3546f29534f036`. Its first startup review passes
all63 pins,357 finite initial tensors matching the bounded check,346 shared
reference tensors, exact config/native/runner/guidance identities, and the first
six updates at376/392/408/424/440/456. The complete prefix has4520 actions and259
finite supervised reports;4096-action progress takes27.935s with.5 residual.
This is startup evidence only. Recovery and completion observers are active;
the fixed final outcome remains pending.

Selected declaration SHA256
`616019129a983dfe59af77ebfecd85bd6df695d38f296bf7ae8e1f5c5a2b5e97`.
The CPU exposure-count correction precedes selection; no GPU stage is repeated.


## Full-run100k recovery

At16:22:27 UTC the read-only observer passes the100,000-action/6227-update
checkpoint with357 finite tensors. Its complete log prefix contains102,408
actions/6377 finite reports;102,400-action progress takes774.017s. Recovery
review SHA256`b8c4833e37e9423c8c0b5dddc761f9d6207cd4edbf174b0b5318cac1d69bdb1c`.
All63 input pins remain unchanged, and the original PID171410/invocation remains
active. The watcher exits successfully after this sole declared milestone;
training and completion observation continue. This does not evaluate competence
or select a checkpoint. The final200k/own-initial assessment remains pending.

Separately, the unchanged CPU exposure reader reproduces all supervision,
episode-group and released-action fields of the completed4M control in1.044s,
using scratch output and no native/environment construction. Its receipt lives
at`full/exposure-reader-control-check/review.json`; all63 active pins stay fixed.


## Full comparison complete

The fixed200k life completes799,693 emulator frames/12,477 updates in1490.781s.
Frozen final1792.708/9of24 versus own initial177.083/0 is51.16% below the4M/B8
control3670.833/19, at15.74% more training cost. Both original gates fail.
Coverage19.833/21 versus20.625 control and4.917 initial. All63 pins,357 finite/
invariant tensors,100k recovery,initialization references, finite losses and
parameter changes, full guided/frozen replays/videos, guards, curves, coverage,
exposure and comparison pass. Different-size initial traces are not equated.
Training overrides92,822 actions (46.411%); guided scores are not competence.

Main service succeeds16:40:13 UTC on October28,30m44.506s total; CPU postprocessing
succeeds16:40:29. No GPU stage repeats or cap changes. Retain4M/B8 and the stronger
5%/R32 world-off budget point; no extension, confirmation or automatic successor.
All services are terminal and no GPU job is active. Exact tables, limitations,
artifact paths and hashes are in the linked completed result. The five-game goal
remains unfinished.
