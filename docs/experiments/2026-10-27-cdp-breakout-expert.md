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


## Active full-run startup

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
