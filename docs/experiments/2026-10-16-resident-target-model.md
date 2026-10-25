# Resident targets in the actual learner

The [compensated-return component repair](2026-10-16-resident-return-precision.md)
passes the original19,652,262 F64 comparisons at22:28:18UTC. All144 fixture arrays
match the failed cohort,759 output arrays and301 restored files pass; one guard,
no new warnings,12 target generations and zero learner updates/actions. Keep the
earlier failure. GPU scalar results are not bit-identical to the CPU compensated
model (20,415/30,978 exact); independent original gates pass, not a lowering proof.

## Full-model smoke and checkpoint scope

Use a new isolated observer/source/cache under
`runs/resident-target-model-20261016.tEVYDi5C`. Runtime source remains the qualified
b78e9d1d component implementation. Only the test observer changes: expose initial
normalizer state and, outside the update timer, save/drop/restore the actual
learner checkpoint. Check normalizer bits, learner/environment/optimizer counters
and every restored model/optimizer/slow tensor. This is not a replay/RNG/live-
belief lifetime resume. Restore after dropping the original core; no concurrent
learners or doubled model residency. Reuse the unchanged backend libraries.

After CPU build/offline source review, run four serial guarded processes:
RGB CPU-target control, RGB resident-target candidate, CDP control, CDP candidate.
Each uses the retained B32/T16/H15/12M BF16 recipe/checkpoint,16+8 fixed synthetic
updates:96 total, no game actions. Select copy composition/image views/word
staging/device means in all arms; only candidate selects device targets. Export
all complete programs/pipelines/world groups, initial/first/final tensors and
first targets. No host trace, Nsight injection or performance claim from24 steps.

Require control reports/first targets/initial-first tensors to match qualified
58aff452 exactly. Both arms have exact initial tensors/config/compiled programs.
First candidate targets and metrics must meet the original component envelopes
(1e-5+2e-5 relative for dense targets/weights;1e-4+2e-5 relative for scalar metrics/
weighted actions); posterior, slow targets and replay masks remain exact.
Report all first/final tensor differences and all later finite reports; do not
require or claim bitwise stochastic trajectory parity. Every checkpoint restore
must preserve the arm's own complete tensors/counters/normalizer exactly.

Each GPU guard is bounded to360seconds,12GiB host memory,no swap,no CPU quota.
Expected device0x2c02,driver580.178.04,boot3e89d55c-a9e5-472f-a18a-06508c5bafa7,
>=2GiB sampled Vulkan budget headroom. Record allocation warnings; API/numerical/
hard-fault/deadline failures stop for review. CPU build/analysis uses one CPU,
2GiB,no swap,720seconds. No retry, production change, learner restart or timing
follows a failure. Complete matched timing is separately declared after review.
