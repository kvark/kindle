# Bounded Breakout pixel-guided expert imitation

After both Pong12M48 learners finish and pass their execution audits, select
one short native training/restore check. The five-game goal remains unfinished;
Pong passes its mean15 gate on one of two learners. Neither is extended.

The unchanged calibrated CPU reference scores166.167 across24 reused natural
episodes,0/24 two-wall successes. It motivates testing transfer to a native
learner, not a claim of learned competence. The settling variant's small gain
on eight selected episodes is insufficient to replace this teacher.

The exact eight-file adapter patch is applied after Pong4013's terminal audit.
442 CPU tests pass with native construction blocked;11,809 saved Breakout and
4096 Qbert actions reproduce exactly. An8192-action ALE fixture verifies512
trusted labels and cap release. Validation/application receipts retain hashes.
Qbert's policy and episode schedule retain their behavior; new guided logs pin
both policy and shared schedule source. Old guided logs use their old runtime.

## Selected execution check

- Centered CDP4M/RGB64/Auto/N8/B8/T16/H15/R32, coefficient1, actor mixture.01,
  original rates, full18/sticky.25/repeat4/noops0, one GPU resize from native RGB.
- Fresh1009/eval4000001009;4096 aggregate actions /963 updates, first248,
  residual0. Training cap180s; frozen1024 actions/60s; audit180s; service480s.
- Select25% of episodes for pixel-only guidance, release after128 actions.
  Native expert loss scale1, all18 action weights1, including NOOP braking.
  The higher check probability exercises labels/release; it is not the proposed
  full learning recipe. No RAM enters the teacher or native policy.
- Require finite reports/all346 tensors,4,131,729 parameters, initial equality
  to the full18 control, changed CNN/reward/actor/value weights, movement and
  NOOP labels, nonzero expert updates, unchanged frozen tensors, full guided
  input-pixel/action/outcome replay and passing host guards.

Native source79c9a7f and SHA256
`7c07d6702e37ca11aa20c1b35b48951f0ed01e0e1dfe72e187eac52253f8455e`
are unchanged. Package the applied Python files at
`/x/Temp/kindle-breakout-guidance-20261027/runtime`; retain the old runtime.
Device RX7900XT/RADV0x744c, boota8648671, kernel7.0.0-38, Mesa26.0.8.
The queue uses the existing host guard. No reset, retry, cap increase or reuse
of check experience. A passing check establishes execution/restore only.

Artifacts and controllers: `/x/Temp/kindle-breakout-guidance-20261027`.
Trial`breakout-4m-64-expert1-check`, owner`kindle-breakout-expert-check-20261027`.
`prepare_check.py` and `run_check.py` pin the source, native, queue, runtime,
control and terminal Pong review; declaration records the launch source.
Patch SHA256`fe109513fa4f8476321a0ba5b9cef6486a85854acff18317029e1263a844b940`;
validation`0e0ff2298f8f6ef0ba4befee7f2b6b1de21f4b37cc94784c89afa6101f5d3d63`.

## Selected full comparison

The bounded check passes4096 actions /16,350 frames /963 updates in24.792s,
669 trusted actions and660 nonzero expert updates; all initial/frozen346 tensors,
parameter changes, replay and guards pass. Frozen1024 actions take.488s with
zero updates. Service exits02:50:25 UTC. Review SHA256
`f55e5d90dab8e79b379fe45398d54087597500246a3960308bd0b520722bdc8f`.
This establishes execution only. After separate review, select one fresh
200k-action/49,939-update comparison,5% episodes/cap640/expert1/all weights1,
with the same model/settings and a25-minute training cap. Fixed final and own
actual initial each select24 natural episodes; original864-point/90%-two-wall
gate, no guidance or learning in evaluation, full audits and100k recovery.
Compare full18 control7.083 and stronger minimal4 control9.542, retaining the
action-space/native/kernel differences. No automatic promotion or successor.


The full trial is`breakout-4m-64-r32-expert1-cap640-noop1-200k`, service
`kindle-breakout-expert-cap640-200k-20261027`. Preparation`prepare_full.py`,
controller`run_full.py`, completion`finish_full.py`, and
`compare_breakout_expert.py` are retained in the artifact root. Declaration
pins their exact bytes, runtime/source, controls, completed check and Pong review.
Recover100k/24,939 updates; fixed final200k/49,939, first248, residual0.
Each frozen arm has600k-action/300-second caps; audit1200s, service3360s.
Report actual guidance exposure and guided/unguided outcomes. This changes
guidance plus expert loss versus the full18 control, with native/control-kernel
differences retained. No retry, checkpoint selection, extension, cap increase,
automatic next allocation or promotion of the eight-episode settling variant.


## Recorded full-run startup

Service starts02:51:51 UTC on October27 from
`da8dee8bacefb9f474d2411aa2012d3f95372698`, PID76965,
invocation`07c8d434e643404d90d48e26b383c286`. Preflight verifies46 pinned inputs.
Read-only startup passes at6840 complete actions/1647 finite updates, including
34 supervised reports. Exact recipe/device/counters,346 finite initial tensors
matching both the full18 control and bounded check, and first six updates
248:1/256:2/256:3/264:4/264:5/272:6 pass. All46 pins remain unchanged.
Startup SHA256`2df0189db11426d60c0020bc7a4e9b3fa26034651d59e8538a7716d05f8e9938`.
No full-run frozen score is known yet. Progress,100k recovery and terminal CPU
postprocessing observers are attached; they never start/extend GPU work.


## Recovery and independent CPU work

The100k/24,939-update recovery passes03:03:42 UTC:346 finite checkpoint tensors,
complete prefix104,880 actions/26,159 finite reports, last progress102,400/
25,539 in691.392s with zero debt. Receipt SHA256
`821c0e3176180bc1ba7c12b86efaf0d78ceab6d35659d0be385e471d0a574462`.
The same learner/PID76965/invocation07c8d434 continues; fixed frozen result pending.

A separate CPU exposure reader reconciles the completed4096-action check's669
trusted actions,963 reports and all completed/partial rewards. It distinguishes
guided prefixes, released actions and wholly unguided episodes, and is attached
after terminal postprocessing. It does not alter any active input or scoring.
Reader`analyze_exposure.py`, observer`finish_exposure.py`, validation
`exposure-validation.json` live in the artifact root. Reader SHA256
`a2ff2d08288ef0634686380a34b06b07eae03318561662349b3fd7157b164737`.
After-release rewards still inherit a guided starting state and are not a causal
measure of native transfer; fixed unguided frozen evaluation remains decisive.

Independent CPU Qbert reference developments improve6743.750→14703.125→21825.000
on eight reused first episodes, all with exact replay. No native module loads;
all46 active pins remain unchanged. Their measured successful portions123.747s,
plus the.015s exposure-reader check and the pending fixed24 reference, are
co-host timing caveats. See the [reference evidence](../results/2026-10-25-cdp-cheap-qbert.md#repeated-movement-and-pixel-avoidance-strengthen-the-cpu-reference).
No new GPU allocation, cap increase or extension is selected by this CPU work.


## Completed result

200k actions /798,964 frames /49,939 updates complete in22m38s. Frozen final4.000
versus own initial1.375 is below the full18 control7.083 and minimal4 9.542;
all0/24 two-wall successes. All audits,46 pins,100k recovery,346 matched initial/
invariant frozen tensors, complete guided replay, frozen outcome replay,
guards, curves, comparison and exposure analysis pass. Service exits03:16:09 UTC,
postprocessing03:16:14. Retain controls; no extension or confirmation.
See [complete evidence](../results/2026-10-26-cdp-breakout-pixel-reference.md#capped-native-imitation-lowers-frozen-breakout-quality).
The following Qbert adapter application occurs only after all terminal reviews
and pin verification; it does not alter this completed run's runtime or evidence.

## Selected bounded check of a longer guidance prefix

After the completed paired Pong greedy assessment, select one16,384-action
Breakout check of cap1920. The saved reference prefix analysis retains96.19%
of full-teacher reward at1920 versus14.52% at640. This motivates an exposure
hypothesis, not a diagnosis of frozen4.000 or a forecast of learned success;
even the full teacher misses the two-wall gate.

Keep the exact original Breakout runner SHA256
`684666a882378575ace1b8dcee67f99ea90827dba4d097812905473a00abfa18`,
original packaged runtime/native79c9a7f, host,4M/RGB64/Auto/N8/B8/T16/H15/R32,
coefficient1/actor.01/full18/original rates, seed1009/eval4000001009 and expert
scale1/all18 weights1. Use25% episode guidance solely for this short execution
check, as in the prior bounded check. A future full candidate would keep5% and
change only the cap640→1920 against the completed expert control.

- Train16,384 aggregate actions/4035 updates, first248/residual0;180-second cap.
  First updates248:1/256:2/256:3/264:4/264:5/272:6 match the retained control.
- Restore the check's final weights for1024 frozen actions, zero updates and
  guidance off,60-second cap. This is not a natural-episode competence test.
- Require finite reports/all346 tensors,346 initial tensors matching cap640,
 346 invariant frozen tensors, changed CNN/reward/actor/value parameters,
  trusted movement labels/nonzero auxiliary reports, exact guided input/action/
  outcome replay and host guards. Audit180s; service480s.
- 2048 actions per lane allow a1920 prefix, but natural episode ends may preempt
  the cap; measure actual override/release counts rather than assume them.
  No retry, cap increase, extension, automatic full promotion or check experience
  reuse. Select a full run only after reviewing this completed check and cost.

CPU preflight parses both commands, compiles four scripts, verifies the exact
original runtime and4035 retained-prefix updates in.059s. No native module,
agent or environment loads; no repository runtime changes. All32 declared pins
match. Preceding Pong main/postprocessing are terminal with complete receipt
`57128cfdbbb46719cd1b87c27063b7f36834473318800880e2043d46ef1e5b35`.
Artifacts:`/x/Temp/kindle-breakout-exposure-20261027/breakout-4m-64-cap1920-check`;
owner`kindle-breakout-cap1920-check-20261027`. Prepared declaration SHA256
`e8f50e75a90466cced5320dc5a50a0d4b86449cb397c7b2f0cc162ac3767b782`.

## Longer-prefix check completes and fresh comparison is selected

The bounded check completes16,384 actions/65,473 frames/4035 updates in114.327s,
1024 frozen/.558s, all32 pins/346 initial and frozen tensors/finite reports/
parameter changes/guided replay/guards pass. Main exits05:38:24 UTC.10,125
trusted actions and3829 supervised reports; no frozen natural episodes and0
released native actions, so neither competence nor cap-release coverage follows.
The extra CPU exposure reader's old4096/963 assumption is repaired without
repeating GPU work; its original failure is retained. Full result and receipt
are in [the reference report](../results/2026-10-26-cdp-breakout-pixel-reference.md#longer-prefix-native-check-passes).

The early rate projects23.26 minutes for200k at the higher25% execution-check
probability. Select one fresh comparison with **25-minute training cap**,
5% episode guidance and cap1920. Change only cap640→1920 against the completed
primary expert control4.000/0of24 in22m38s. Keep exact original Breakout runner,
packaged runtime/native79c9a7f, host, seed1009/eval4000001009 and all model/rate/
loss settings:4M64/Auto/N8/B8/T16/H15/R32/coefficient1/actor.01/full18/expert1/
all action weights1. No weights or experience are restored.

- 200k aggregate actions/49,939 updates, first248/residual0;100k recovery at
 24,939 updates and final200k. Fixed25-minute training cap.
- Fixed final and own actual initial, first3 natural episodes per8 lanes=24;
  guidance off, zero updates/truncations, original90%-two-wall gate.600k actions/
  300s per arm. Retain excess episodes and partial tails.
- Require all finite reports/346 tensors,346 initial tensors matching cap640
  and bounded check,346 invariant tensors per frozen arm, paired initial
  semantic traces, complete guided input/action/outcome replay, frozen outcome
  replays/videos, guards, recovery, curves, comparison and exposure. Frozen
  logs do not have live pixel hashes. Audit1200s; total service3360s.
- No retry, cap increase, extension, checkpoint selection or automatic successor.
  The existing5% schedule can have a much larger actual guided-action fraction;
  measure that and guided/unguided outcomes without claiming frozen competence.

Three commands parse, six scripts compile, the retained primary control reader
and24,939/49,939 recovery/final arithmetic pass in.071s with no native module,
agent or environment. All32 bounded pins remain unchanged. The full declaration
pins47 inputs; prepared SHA256
`fb9b2652145a4a1654bc6e2673186dfc9665f77b6b1848f7c89580cc3857dd20`.
Owner`kindle-breakout-cap1920-200k-20261027`; trial
`/x/Temp/kindle-breakout-exposure-20261027/breakout-4m-64-r32-expert1-cap1920-noop1-200k`.
The reference/evaluation cohorts and1009 seed are reused development evidence;
stronger teacher prefixes do not predict transfer or satisfy the two-wall goal.

## Longer-prefix full comparison startup passes

Service starts05:41:26 UTC/source`8c888aa`, PID94810,
invocation`8114cb5a27d74a8787a4b1297ab179d2`. Selected declaration SHA256
`9faaed7a9298dddece9180ac2d8195b70c11a192f40fc50d2268d29a955b8880`.
All47 pins match. All346 finite initial tensors exactly match primary cap640
and the bounded check; native/runner/runtime/host/config/seed identity passes.
First updates248:1/256:2/256:3/264:4/264:5/272:6 match. The10,496-action
complete prefix contains2561 finite reports,604 trusted actions and323 reports
with expert labels. Read-only startup receipt takes.172s and constructs no
native module or environment. Receipt SHA256
`90cca477d67f72dc89df37cab33ddbb6609707f9105fc8669411c1c4a42c461f`.
Progress,100k recovery and terminal CPU postprocessing observers are attached;
all four service handles are live. Keep the declared200k/25-minute cap and full
fixed evaluations/audits. No frozen result or automatic successor exists.

While the original Breakout service remains active, a.162s log-only Qbert
action-timing analysis and.607s of four-arm greedy preparation/checks run on one
affined low-priority core. They construct no native module, agent or environment
and leave all47 Breakout inputs unchanged. This.769s measured CPU work is a
co-host timing caveat, not a wall-time correction. The separate Qbert assessment
remains prepared/unselected until Breakout and its postprocessing are terminal.

## Longer-prefix100k recovery passes

At05:53:24 UTC, the100k recovery contains24,939 updates and346 finite tensors.
The watcher verifies25,309 finite reports in the101,488-action complete prefix;
last progress98,304/24,515 at691.636s, zero debt. Recovery checkpoint SHA256
`35b5576802e15b09ee698829e6c9d1deef2f3375a687b18b4e399d58403110a8`;
review`81f9b8f8aa51005944c13bb2e7a3271752dda823f377f42e514822d58daa518b`.
The recovery observer completes successfully; the original main service remains
active under PID94810/invocation`8114cb5a27d74a8787a4b1297ab179d2`.
Keep200k/25-minute cap and fixed evaluations/audits. No frozen result exists.
