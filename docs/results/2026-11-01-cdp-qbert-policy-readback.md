# Qbert failure windows expose weak movement agreement behind strong wait agreement

The exact batched actor readout reproduces all20,584 original greedy actions
and saved pixel trajectories. In the1,024 decisions preceding32 recorded
losses,overall teacher agreement is86.33%,but teacher movement agreement is
only52.34%,versus97.66% for NOOP. On the25 windows rescued by teacher takeover,
movement agreement is38.07%. This locates a concrete policy-preference gap on
selected later-play states; it does not identify its architectural cause.

| Fixed failure-window decisions | Rows | Teacher agreement | Mean teacher probability | Teacher CE |
|---|---:|---:|---:|---:|
| All |1,024|884 /1,024 (86.33%)|.8582|.4093|
| Teacher NOOP |768|750 /768 (97.66%)|.9670|.0822|
| Teacher movement |256|134 /256 (52.34%)|.5319|1.3906|
| Movement in teacher-rescued windows |197|75 /197 (38.07%)|.3946|1.8040|
| All decisions in seven unrescued windows |224|224 /224 (100%)|.9956|.0046|

The NOOP/movement rows are a post-hoc arithmetic decomposition of the same fixed
windows,not a new selection criterion. Of122 movement disagreements,34 choose
NOOP and88 choose another action. Across all140 disagreements,the recorded
choice has mean probability.7021 and the teacher choice.1358;70 teacher choices
receive probability below.1. Thus the captured greedy policy often prefers a
different command, rather than merely sampling an unlikely action.

Teacher and policy already agree at31/32 intervention origins and25/32 last
actions. At the seven unsuccessful origins,all224 teacher recommendations match
the recorded commands through the loss. Those branches supply no different
sequence within the chosen window. The successful interventions change a later
feedback-driven sequence; their outcome cannot establish the causal value of
each disagreement or the exact first harmful decision.

All windows come from the first natural episode of each retained lane,using
the [existing teacher-intervention selection](2026-11-01-cdp-qbert-recoverability.md).
This is conditional evidence on eight reused development episodes,not average
policy accuracy,independent reliability or a new game score. The failure-window
CE and the learner's late replay CE.0893 concern different state distributions.
They cannot be substituted for each other or treated as a matched generalization
measurement. No replacement readout is fitted here.

This supports investigating movement choices within successful alternative
sequences. It does not by itself select stronger imitation or lower wait weight:
the [earlier NOOP-weight.25 learner](2026-10-29-cdp-qbert-noop-weight.md) reduced
frozen quality,with a small hop-agreement gain and a much larger wait-agreement
loss. Representation precision,action sequencing,training coverage and actor
optimization remain distinguishable explanations. No new loss,learner,rate
queue or automatic successor is selected.

## Readout qualification and exact replay

The [fixed protocol](../experiments/2026-11-01-cdp-qbert-policy-readback.md) adds
`VectorAgent.policy_inputs_and_logits()`. After `act` and before `observe`,it
copies the existing actor inputs and raw logits in stream order. It performs
no model forward pass,normalization,sampling or update. Invalid call timing is
rejected. Logits precede temperature and categorical unimix;this Qbert checkpoint
uses unimix0 and ordinary sampled categorical posteriors.

The isolated native build is447fe50 plus only identical readout,binding and GPU
test additions. It passes130 Rust CPU tests,585 packaged Python CPU tests and
strict native/binding Clippy. The guarded GPU test covers greedy,sampled and
overridden actions,call timing,repeated readbacks,RNG,counters and recurrent
feature invariance. The original context-only backport failure is retained;
its correction changes patch placement,not the added code. No native build or
GPU stage was attempted before that correction.

A256-action/1,024-frame prefix takes4.020s; the full20,584-action/82,324-frame
capture takes17.631s. It preserves all eight lanes and original resets,covering
16,613 selected first-episode decisions plus validated later-episode tails.
Every original greedy action is reproduced without forcing it. Input/output
pixels,rewards,boundaries and cumulative frames match the CPU replay. All357
checkpoint tensors remain invariant in each capture,withzero learner updates.

All native raw-logit argmax choices match their recorded actions. At128 fixed
opening decisions,F64 actor probabilities from the copied features/weights
match probabilities from the native logits within4.762e-10 absolute error,
below1e-4;all128 greedy choices agree. This is an opening-prefix numerical
check,not an all-state error bound. Smoke/full prefix logits and all qualification
features match exactly. The full capture performs29 repeated readback checks:
the initial tick and28 distinct vector ticks containing the32 origins.
Saved failure features have width2,560. Original hidden arrays were not logged,
so no independent equality check against those unavailable arrays is claimed.

Service`kindle-qbert-policy-readback-20261101` succeeds05:31:49 UTC November1,
source`fd1c586`,after29.478s overall. OriginalPID752193/invocation
`06a9f0fded8c4fe99f4e37ec0145c101`;all three guarded stages and525 inputs pass.
The first separate terminal collector rejects a null journal-message field.
Its source/failure remain preserved; a.661s CPU-only repair classifies string
messages and completes05:34:08. The original service and all GPU stages already
succeeded;no capture,analysis or GPU stage repeats. Four pure analysis tests and
the2.243s CPU preflight also pass. No learner/GPU allocation remains active at
completion;check live state before new work. The five-game goal is unfinished.

The857,072,433-byte build-file tree is released after qualification;source,
wheels,native runtime and the GPU test executable remain. The original packaging
receipt stays immutable;`cache-release.json` records the later cleanup.

Roots:
`/x/Temp/kindle-vector-policy-readback-20261101` and
`/x/Temp/kindle-qbert-policy-readback-20261101`. The retained learner runtime,
checkpoint,teacher and source trajectories remain unchanged.

| Artifact | SHA256 |
|---|---|
| Native readout runtime |`f864fe4b7af23a8a2c7db15edcec5a13febebf58ee3a8bbf3383429454ae1669`|
| Declaration |`31d18dce75cbfbd78761b94263518eeeab33ffbdc1a3fbc6b3569d100524425c`|
| Review |`b40747b9b47c6e05b35b165f32ab5283112a00b655aa36e1908d84f7bc7b3801`|
| Analysis |`d6c5920c7d4439e9a5f67bb32c009e1ce0fce507c95757d78d81cbcf40c60dda`|
| Label split |`d73e6c8cfc2ac9e167b21ef6c9d531997aa3b258075a84a9fde7b1750a7902ec`|
| Completion |`d4f5011db5e443551ba4132e79792d601013b708987fb7e88703db81326966d8`|
