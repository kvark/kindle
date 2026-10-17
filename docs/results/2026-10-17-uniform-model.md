# Uniform-buffer policy does not close the learner gap

October17,2026. [Evidence](2026-10-17-uniform-model.json) ·
[Instruction diagnosis](2026-10-17-uniform-interface.md) ·
[Backend plan](../backend_training_performance_plan.md).

**Reject the global UBO policy:** complete RGB updates improve only1.45%, below
the declared3% gate, while CDP gets1.89% slower. Numerical and lifetime checks
pass. The preceding single-kernel14.05% instruction reduction is real, but is
not a corresponding learner-speed improvement or explanation of the full gap.

| Complete B32/T16/H15/12M update | Inline control | Existing UBO fallback | Change |
| --- | ---: | ---: | ---: |
| RGB |43.164ms|42.540ms|1.45% less time|
| Centered CDP |41.044ms|41.820ms|1.89% more time|

Keep isolated native`e45b9c43` and its existing flags: qualified RGB remains
**43.197ms versus inclusive JAX27.223ms,1.59×**. Production and Atari remain
unchanged. There is no retime, favorable kernel/model selector or integration
of the rejected policy. The wider root-cause/parity goal stays open.

## Existing storage, not a new backend

CPU review covers all ten actual RGB/CDP program views,430 emitted modules,
six capacity-negative checks and19 source-backed lifetime checks. World and
posterior are aliases, not two separate training workloads. Every graph fits
the existing1MiB command-buffer scratch at conservative256-byte alignment;
the largest CDP whole-graph bound is1,046,288bytes. Actual device alignment is64.
These extents are not measured traffic or a guarantee for arbitrary programs.

Blade already allocates this scratch even when using inline uniforms. Immutable
cached graph commands own their scratch; dynamic optimizer constants remain
outside those commands. Primary/helper reuse retains its existing waits and
host-coherent allocation contract. Command caching stays enabled: this does
not qualify a combined uncached CDP graph-plus-optimizer allocation.

One isolated binary,`b01fe8f3`, changes only Blade's effective inline-size policy:
unset retains the control, `BLADE_UNIFORM_BUFFERS=1` selects the existing UBO
fallback for all Plain bindings. Device feature enabling, WGSL bytes, kernels,
precision, scratch limits, model and submissions are unchanged. Three local
manifest paths select the same isolated dependencies; Naga remains byte-exact.
Build, Clippy and independent source checks pass. This switch is not promoted
as a production API.

## Complete checks and ordinary timing

Four24-update numerical guards each pass review before four ordinary timing
guards run. Every timing arm retains16 warmup and all256 measured updates;
there is no profiling, compilation, sample removal or alternate timing cohort.
Fresh controls differ from retained means by−.076% RGB/+.116% CDP, within5%.
All upload/scratch/synchronization work in the original caller boundary stays.

All24/272 reports, first5,618,720 target/output values, initial/first/final292
RGB or346 CDP tensors match exactly between arms. Controls also reproduce the
retained baseline. Each timing arm matches its own numerical prefix. All full
plans/constants, pipeline labels, groups, cached-command counts and205/225
RGB/CDP shader files match. Descriptor types intentionally differ; identical
labels do not claim identical descriptors. Eight checkpoint roundtrips restore
tensors, normalizer and counters exactly, not replay/live-belief lifetimes.

RGB's caller world-training/sync interval falls24.907→23.925ms, partly offset by
increases in the other intervals. CDP's corresponding interval rises12.574→
12.983ms. These intervals include dependencies and waits, not pure convolution
or kernel costs. No whole-family instruction, source-line or SM-utilization
claim follows from this timing. All first/last32 means and tails remain in the
evidence.

Eight guards pass with four recorded allocation warnings,1,184 synthetic
updates and zero game actions. GPU work finishes02:06:04UTC; the controller
finishes02:06:18UTC. Independent publication checks32 tensor-state pairs,
all reports and3,196 source/result seals, finishing**02:07:51UTC**. No GPU
worker remains; no recovery or separate NVML polling occurs.

## Retained failures and next question

The first CPU layout reader wrongly requires every dumped shader to have a
uniform variable; six construction-specialized modules instead contain a named
constant. Its original failure remains, and the corrected complete review
passes. The first build invokes Rust2021 formatting on Rust2024 Blade code;
the corrected invocation then exposes a duplicate local Blade dependency.
Kindle's direct Blade path is aligned with Meganeura's before compilation.
Original scripts/logs remain; no lockfile unlock, raised limits or GPU retry.

The old normalization counter allocation remains stopped and failed. Neither
that queue nor this global uniform-policy search resumes. The next CPU question
is the **current convolution-to-normalization/activation and adjoint boundary**:
match actual JAX fusion bodies to native versioned producers, consumers, required
tapes and rounding points. Establish whether dependent image operations can be
composed without losing required values, rather than assuming dispatch counts
are removable time. Use existing captures; no new GPU allocation, compiler
project, geometry sweep or learning restart follows this result. Library feeding
and non-image work remain separate unresolved costs.

Artifacts: `runs/ubo-contract-20261017.BG5JhGQn` and
`runs/ubo-model-20261017.9I4iuDbs`, including declarations, rejected build logs,
source snapshots, full GPU outputs, guards and independent reviews.
