# Refresh the qualified learner onto current upstream

The user requests the merged Meganeura fixes before further gap diagnosis.
This takes priority over the unimplemented recurrence/subgroup proposal.
Atari and production promotion remain stopped.

## Scope

Upstream on October 19: Meganeura `c19cb1ca`, Blade `4845d652`, its Naga
`13f582dd`. Compare with the qualified isolated `e45b9c43` learner: RGB12M,
B32/T16/H15, explicit BF16, resident targets, cached recording, composed image
views/copies and convolution word staging. Its complete update was 43.197 ms;
the retained caller-inclusive JAX reference is 27.222569 ms. CDP is a separate
workload (40.997 ms), not another numerator for the RGB/JAX ratio.

The isolated backend includes extensions not yet present upstream: typed
convolution derivatives/grouped products/reductions, staged world execution,
image views, private copy composition and cached command recording. Retain
those mechanisms while adopting current upstream dense kernels, compiler
interfaces, validation and search code. Record this as **upstream plus retained
extensions**, not stock main. Preserve the original trees and binaries. The
automatic three-way merge is preparation only, not proof of correct integration.
No user checkout is modified.

## Qualification and one complete comparison

1. Resolve and build the isolated integration, reviewing API/layout/precision
   changes. Format and run scoped CPU/shader tests and Clippy. Retain failed
   preparations/builds; review before a corrected build.
2. Run the existing fixed-batch checkpoint harness, old and new RGB/CDP, 24
   synthetic updates each. Both arms enable all qualified learner features.
   Require matching initial tensors/configuration, finite reports/state, actual
   parameter changes, exact per-arm checkpoint restoration and device/budget
   checks. Upstream changes dense accumulation schedules; measure first targets,
   losses and later tensor differences without asserting trajectory equivalence.
   This follows the earlier cost/finite-execution recheck contract. Before any
   GPU allocation, this replaces the draft's inappropriate exact-state/pointwise
   target gate: posterior states are computed by changed products, not merely
   copied arrays. It does not revise any completed numerical result or tolerance.
   Inspect complete plans and shader bindings; an upstream change is not required
   to produce an identical plan. No relaxation after a failed gate.
3. Only after review, run one four-process comparison: RGB old/new, CDP old/new;
   16 warmups + 256 measured complete updates each. Retain all samples. Each
   arm must reproduce its own smoke prefix exactly. The old control must
   reproduce retained reports/tensors and calibrate within 5% of its retained
   mean. Report complete time and stage times; do not add overlapping stages.
   An improvement is material at >=3% RGB with <=2% CDP regression. A failure
   is retained, not followed by a favorable subset or repeat.

No runtime kernel tuning is enabled in this first comparison. The newer
measured-search improvements are conditional, not automatically exercised by
ordinary construction. A separate bounded whole-program search would need its
own declaration after this comparison; no closed manual geometry sweep resumes.
This initial experiment is a cost/finite-execution diagnostic, not independent
full-model numerical qualification or learning parity. A material win warrants
separate numerical qualification before adoption. No game actions or learner queue.

## Bounds and evidence

CPU preparation/build/review: one CPU, 2 GiB, no swap, 720-second services.
GPU processes: serialized host guard, 360 seconds each, 12 GiB, no swap and no
CPU quota; controller <=2400 seconds. Record allocation warnings and the known
reviewed validation VUID. Stop and review API/numerical/hard-fault/deadline
failures. Require RTX 5080/device `0x2c02`, driver `580.178.04`, current boot
`3e89d55c-a9e5-472f-a18a-06508c5bafa7`, >=2 GiB sampled Vulkan budget headroom.
No build during timing, NVML polling, driver recovery or automatic promotion.

Sources, merge inventory, failures and results stay in
`runs/upstream-refresh-20261019.KWEh8dEH`. PR31 is the status dashboard. The
objective is a fresh complete comparison and concrete remaining bottlenecks,
not another large collection of standalone kernel wins.
