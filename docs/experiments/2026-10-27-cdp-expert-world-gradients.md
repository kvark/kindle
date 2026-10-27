# Optional expert supervision of world state

The completed continuation Qbert learners improve with expert imitation but
miss mean15000. More guidance with less replay and greedy frozen actions both
lower quality. Breakout's longer guidance also lowers frozen quality. Source
review shows that expert labels currently train the behavior actor from detached
replay features, without direct expert gradients into the CNN or recurrent
state. This motivates one optional native learning change, not a diagnosed
cause of the score gaps or a default architecture replacement.

## Selected implementation

Add `expert_imitation.world_gradient`, default false and omitted when false
in serialized configuration. The world graph evaluates the same weighted,
masked cross entropy through the current frozen actor. The behavior optimizer
continues owning actor parameters; the new path trains only world/encoder
parameters. Both updates use the same pre-update targets and actor parameters.
Keep expert scale and actor-start gating. Emit an optional world expert-loss
metric. Existing configuration/checkpoint behavior remains unchanged when off.

Reuse the existing trusted outgoing-action targets: state t receives the expert
action stored with frame t+1. Last/reset boundaries and untrusted actions stay
masked; the final sampled state has no outgoing target. Correct the world T
average to the behavior T−1 average, including chunked/microbatched training.
Expose `--guidance-world-gradient` only with fresh guided imitation, reject
restore overrides and older native extensions without explicit support.
Inference receives pixels and its own history; the new flag adds no frozen aid.

## Bounded validation before Atari allocation

CPU checks cover legacy configuration round trips, default graph equality,
target rows/masks/normalization, graph connectivity into CNN/posterior with
frozen actor/critic parameters, staged and ordinary graphs, combined optional
actor/critic gradients, CLI configuration and rejection before construction.
Run relevant Python tests, Rust CPU tests and strict Clippy. Preserve failed
development checks externally; a failing fixture is not a GPU result.

Select two guarded native GPU checks only after CPU validation:

- Isolate expert loss in a tiny RGB32/CDP world. Require finite nonzero CNN and
  posterior gradients, an actual CNN update, unchanged frozen actor parameters,
  and zero loss/gradients with all expert targets masked. Cap180 seconds.
- Run the native vector trusted-action integration with world gradients on.
  Require aligned labels, finite positive world/behavior expert losses matching
  within2e−4 from pre-update parameters, and an actor update. Cap240 seconds.

These checks establish their stated gradient/alignment scope, not task success,
general numerical parity or a causal explanation of prior Atari failures. Use
the existing RX7900XT host guard, Auto precision, one GPU job at a time, and
stop on real faults or numerical failures. No reset/reload/reboot or automatic
retry. A failed assertion needs review before any distinct corrected check.

Build/package separately under `/x/Temp/kindle-expert-world-20261027`, retain
the exact source/native/runtime hashes and original79c9a7f runtime. Release
the disposable build cache only after packaging and retaining test executables.
No Atari learner, full training allocation or automatic promotion is selected
by this implementation protocol. A bounded Atari check must follow a separate
review of these completed checks and their cost.

## Native checks pass; bounded Qbert execution selected

Source`0e157df` passes129 Rust CPU tests,466 relevant Python tests both before
and after packaging, workspace/binding strict Clippy and formatting. Initial
compile/fixture failures (missing struct field, unsupported graph debug format,
CLI fixture action meanings/override signature and one absent test path) are
retained externally; no native GPU stage fails or repeats.

Both selected GPU checks pass on the guarded RX7900XT in5.986s total service
wall time,06:25:52–06:25:58 UTC. Isolated expert loss1.099951 has CNN squared
gradient norm.0003636701 and posterior norm.0001376968; a CNN update occurs,
frozen actor weights stay unchanged, and zero targets produce exactly zero
loss/gradients. The vector check has3 labels/weight3/loss.5461493 and passes
world/behavior pre-update agreement within2e−4. These are actual isolated
native derivatives and updates, not an explanation of past Atari failures.
Review SHA256`f0c24b6ec2f5aeb9974c91de159ab4649971f8f0915f4a17aafe607887164b9b`.
Native SHA256`6a9c41c44aef4747cef6504a2525195d8f2f2a62dbece7c6ecc6d1128fff1c1e`;
the original runtime is retained separately.

After terminal review, select one fresh8192-action/1987-update Qbert check:
4M/RGB48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original rates,
seed4013/eval4000004013, continuation guidance25%/cap1920/expert1/NOOPweight.05,
with world gradients enabled. The25% episode probability is execution-check
exposure, not a selected full-run recipe. First248/residual0;180s training cap.
Follow with1024 frozen actions/60s, no guidance/updates. It cannot establish
natural-episode competence. CPU audit cap180s; whole service480s.

Require all1987 finite reports, positive weighted actor/world expert losses
when labeled (agreement within.002×(1+actor loss)), zero masked losses, trusted
hop labels, full input/action/outcome guidance replay, host guards and all357
finite checkpoint tensors. The11 additional world tensors are frozen actor
copies and must equal behavior actor weights. All346 shared initial tensors
must match the retained4013 control; all357 final/frozen tensors must match.
Check changed CNN/reward/actor/value parameters without attributing changes
in a joint update exclusively to the new term. Preserve full config, source,
seeds, frame/action/time counts and every failed attempt.

Trial:`/x/Temp/kindle-expert-world-20261027/qbert-4m-48-expert-world-check`.
Owner:`kindle-qbert-expert-world-check-20261027`. No retry, cap increase,
extension, reuse of check experience, full learner or automatic promotion.

## Bounded Qbert check completes after a CPU audit repair

8192 aggregate actions /32,749 emulator frames /1987 updates take **80.263s**.
Frozen1024 actions /4096 frames take.489s with zero updates and no completed
natural episodes. All357 finite and unchanged frozen tensors,346 shared initial
tensors, additional frozen actor-copy equality, changed CNN/reward/actor/value
parameters,35 original input pins and both host guards pass.

4520 trusted actions (55.1758%) include3489 NOOP and1031 directional hops.
1978 reports contain expert targets, with120,250 repeated replay-label rows
and31,054.498 total weight. Actor/world loss sums452.522193/452.522246 agree;
the maximum per-update difference is1.5e−7. Full saved input/action/outcome
replay passes. These demonstrate execution and alignment, not learned task
success or that joint parameter changes came exclusively from expert loss.

The service starts06:30:28 UTC/source`ef72b66`, PID109171/invocation
`64a09b02033749ab94cd21295b6c52df`. Both GPU stages succeed, but the final CPU
audit rejects the new config field because its schema allows only `scale` and
`action_weights`; the original service exits failed at06:31:56. Preserve that
failure, original controller, declaration, runtime and all35 pins. A separate
CPU-only overlay accepts the optional strictly Boolean `world_gradient` field
and still rejects malformed/unknown fields.474 relevant Python checks pass,
including new schema cases. An initial added-test decorator error is retained
and corrected before the passing run. No GPU stage repeats and no cap changes.
The repaired audit completes all checks; the original failed service is not
relabeled successful. No native module loads during terminal verification.

Bounded review SHA256`a47ab971be958dd6298e862b9e99b17b0e42f3316123a722fe1633beefa53503`.
The separate `completion-review.json` links original failure/journal, repaired
validator/controller, complete bounded review and unchanged original pins.
Use the separately packaged audited runtime for future work; do not mutate the
runtime pinned by this check. The native binary remains source`0e157df`.

The early R32 rate extrapolates to32.66 minutes at200k, before uncertainty in
later cost/exposure. Do not automatically allocate that full recipe. The
retained20%-guidance/R16 control offers a cheaper primary comparison for a
future world-gradient test:200k/24,970 updates in21m59s, frozen3333.333/18 first
pyramids, while the stronger5%/R32 control scores4730.208/23 in25m23s. A new
R16 comparison would need its own fixed budget/cap,357-tensor recovery/evaluation
audits and a separate selection. No full learner is selected by this result.

## Packaging and current state

Completed receipt SHA256`b8ff705544e56b1c64e9615d3671c95f434bec3ec491ab2a4ccb9abd9e05aa95`.
The corrected Python validator is source`44085dd`; Rust/native code stays exactly
source`0e157df`. The separately exported
`/x/Temp/kindle-expert-world-20261027/runtime-audited` contains the same native
SHA256`6a9c41c44aef4747cef6504a2525195d8f2f2a62dbece7c6ecc6d1128fff1c1e`.
Audited wheel SHA256`15c4beae2ccd97da3cea6d1089adadd38e96c08ef65153628861f55bbb3b7a60`;
validator`700b8c19f534b60cd8fec09b14eea9251a6cfc4332f9658f5ea9e20f32f76925`.
All474 relevant tests also pass against that packaged runtime. The original
wheel/runtime and failed audit remain unchanged. `packaging-receipt.json`
records both source revisions, exact files and the retained test executable.
The disposable2.6GiB build cache is released after packaging and verification.
No GPU allocation is active and no full learner is selected.
