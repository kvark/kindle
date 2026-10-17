# Staged world-model execution

Investigate the remaining complete-update gap by retaining posterior activations
for world differentiation. Control: qualified shared-kernel candidate RGB39.37ms;
JAX12.417ms is a retained reference, not a new matched-trajectory control.
No kernel-layout sweep, precision change, Atari allocation or production repin.

Add a runtime preparation boundary defined by requested graph outputs. Compute
their dependency closure before remaining work; keep activation lifetimes across
the boundary, prevent early-input/parameter mutation during preparation, and run
clipping/optimizer only after completion. Reject unsupported multi-writer plans;
retain ordinary whole-step execution. Preserve temporal accumulation behavior,
and keep the existing whole-step fallback for microbatch/truncated paths until
their activation lifetime is explicitly supported. This is not a generic BPTT
or public video-path qualification.

The single-pass Kindle training graph samples its posterior from the existing
CPU-owned uniform stream and exports h/z, then consumes late replay/behavior
targets. Reuse those activations instead of recomputing an independent posterior.
Keep old-weight consumers before optimizer writes, reset semantics, detached
sampling/straight-through gradients and existing losses. Device RNG/targets/EMA
and inter-session feature readbacks are a separate following change, not claimed
by this staged step. Keep the useful chunk schedule proportionate to stage size.

CPU checks cover dependency closure, distinct/unknown outputs, unsupported writes,
and stage boundaries under scheduling/fusion. GPU checks compare a graph run
whole/staged for forward values, raw gradients, parameters/moments, repeated steps,
poison and production aliasing; late-input perturbation must affect the tail,
not retained outputs. Reject premature optimizer/early-input writes. Then one
ordinary whole-update pair each for RGB12M/B8/BF16 and CDP12M/B32/mixed,
16 warmup+256 measured updates, exact initial checkpoints and configurations.
Audit operation counts, removed recomputation, finite metrics/states, RNG/sample
agreement and complete cost. Changed grouping/categorical trajectories must be
reported, not equated with learning quality. No profiling/timing instrumentation
or compilation overlap. Preserve numerical gates and every failure.

Artifacts: `runs/staged-world-20261013.djLV1cJT`. Upstream2214412/fe5f5b5 checked;
base056562b with qualified shared-kernel prototypes, Bladefe5f5b5/Nagaa4b4183.
CPU preparation/analysis uses one CPU,2GiB, zero swap. Serialize guarded GPU work
in bounded user services, Restart=no/KillMode=control-group, no CPU quota or NVML
polling. Require5080 and≥2GiB sampled estimated headroom, retain allocation
warnings; other failures stop for review. No driver recovery or blind retries.

Pre-GPU code review: Mixed uses BF16 posterior inference but F16-input/F32-storage
world training. Those activations are not interchangeable without a precision
change. Keep Mixed on the existing path; its pair checks the unchanged fallback,
not a staged-CDP speed claim. This prototype enables staging only for a single
BF16 pass, where both sessions explicitly use the same forward precision.
Other precision policies and optional video encoding remain outside this
qualification. CPU integration found a missing default field and a Clippy
pattern-style error; retain both source snapshots and service logs.

Qualification review20:53 UTC: all four guards pass (one allocation warning).
The eight F32/BF16 × alias/no-alias × Adam/LaProp cases match whole and staged
values, post-clipping gradients, parameters and moments exactly through144 updates;
the two late-target controls differ only after completion. Hidden activations
cross the boundary; early writes/updates and invalid phase transitions reject.
Both24-update production smokes pass. CDP's24 reports, all346 tensors at
initial/first/final and first outputs match the retained same-length control.
RGB initials292/292, first categorical samples and all215 first world tensors
match exactly. Deterministic state differs: t0 relative L2 4.44e-8, overall
5.15e-4;26 behavior moment tensors then differ, while all first model parameters
and slow values remain exact. This is consistent with forward compilation
rounding magnified through BF16 recurrence, not established bitwise trajectory
parity. The complete-world numerical check and preserved world update support
the declared timing comparison; no tolerance changes. The smoke audit's RGB
final state is a self-finiteness check, not a24-vs272-update equality claim.
Retain `smoke-difference-review.json` and the CPU audit's unintentional BLAS
oversubscription (76s); subsequent analysis explicitly uses one BLAS thread.
Proceed with the one declared full-update pair per recipe, without compilation
or analysis overlap. No Atari allocation or new precision setting.

Post-timing publication review corrects the gradient scope: AGC writes gradients
in place, so the optimizer-bearing checks are post-clipping, not raw gradients.
Add one separately guarded backward-only diagnostic with optimizer cleared;
unchanged parameters/moments/counter, F32/BF16 and alias/no-alias, two late
targets each. Eight cases/24 backward passes, zero optimizer updates. The runtime
and measured Kindle binary do not change; retain the original and supplemental
test source/binary separately. No timing rerun or gate relaxation.
