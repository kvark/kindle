# Qualify connected image views, then measure complete updates

Declared October15,2026 after the CPU selection/indexing/codegen checks pass,
before any GPU dispatch of the [image-view candidate](2026-10-15-image-views.md).
Five CPU tests reproduce the independent full-plan rewrite and observer/multiple-
writer fallbacks, preserve the default pipeline key, validate144 existing image
variants and validate/emit SPIR-V for432 new typed/layout variants. Meganeura
and Kindle Clippy pass. Finish/review the isolated binary build before launch.
The original reference-style Clippy failure/source remain. Upstream main
99e3de27/e7c3f7b rechecked unchanged before implementation.

One RGB12M/B32/T16 and one trained centered-CDP12M/B32/T16 control/candidate pair,
24 synthetic updates each, with the existing seed701 changing-batch fixture.
Both arms use the new isolated binary; only `MEGANEURA_IMAGE_VIEWS=0/1` differs.
Keep device means/command reuse, source checkpoints/configs and ordinary update
staging fixed. The rejected2D packing repair is absent. No game actions or
learner allocation. This is exact implementation-equivalence qualification,
not a new independent F64 oracle or evidence of learning competence.

Require exact paired24 non-timing reports, first targets and initial/first/final
292 RGB/346 CDP tensors, including optimizer state; finite values and unchanged
initial configurations. Compare control's reports/initial/first with the retained
076f3135 prefix. Full plan/pipeline exports must exist, without per-dispatch
profiling/NVTX capture. RGB must match the CPU proof's13 changed image kernels/
31 removed dispatches; CDP must have5/11. Convolution arithmetic/selection and
all other program bodies stay unchanged except the declared input-view aliases
and dependency regrouping. Audit graph-visible metadata and exact non-world
programs between arms. Retained full RGB control plans also match; do not invent
missing historical full CDP exports. Review any failure before follow-up.

Only after successful CPU numerical/program review: one ordinary full-update
comparison per workload,16 warm+256 measured/arm, all samples retained. Require
the same exact paired report/target/tensor checks, each arm's exact24-update
qualification prefix, and unchanged full program contracts. Adoption consideration
requires>=3% complete RGB improvement and no>2% CDP regression. No subset,
retiming, tolerance relaxation or geometry/precision sweep if it misses. Any
retention remains isolated; no production promotion or Atari restart.

Serialize GPU jobs in guarded persistent systemd services, Restart=no and
KillMode=control-group; each worker<=240s, each four-worker service<=1200s.
Expected5080/580.178.04/boot3e89d55c-a9e5-472f-a18a-06508c5bafa7 and>=2GiB sampled
Vulkan estimated budget headroom. Record allocation warnings; API/numerical/
hard/deadline failures stop for review, no blind retry/recovery/separate NVML
polling. No CPU quota on GPU workers or concurrent compile/analysis during
timing. CPU preparation/review1CPU/2GiB/no swap. Preserve all source receipts,
failures, exports and extra96 qualification/1088 timing updates separately.
