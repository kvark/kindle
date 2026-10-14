# Locate the cost that scales worse than JAX

October14,2026. Follow the [packing-grid result](../results/2026-10-14-packing-grid.md)
after its completed CPU review. No GPU work is started by this declaration.
The grid correction changes only four intended convolutions but saves1.49%
of complete B32 time, below its3% gate. Do not promote it or expand the grid/
batch/geometry search. The retained B32 control remains the comparison runtime.

Native's complete B8/B32 throughput scales1.33× versus JAX1.87× despite exact
fourfold nominal heavy work. The prior cached family attribution is B8; it
does not establish which family causes the larger-batch discrepancy. Determine
that before choosing another kernel/compiler intervention. Do not assume
that the fast-path admission finding explained the timing difference.

Reuse the already-qualified cached fixed-session family-mask diagnostic in
`runs/cached-residual-20261014.l68izkI8`, changing only the isolated Kindle
logical convolution-gradient scratch cap64→128MiB required by the existing
B32 control. Keep that diagnostic's sources/locks and immutable original
binary. Audit the new source delta and compiler plans against the retained
B32 control; no new shader, precision, optimizer, schedule or memory-bound
change beyond that previously reviewed scratch allowance. Upstream recheck,
formatting/release Clippy and the exact source audit precede execution.

One bounded600-second GPU job: same B32/T16/H15 initial checkpoint and fixed
batch, one synthetic preparatory update, then optimizers disabled. Replay
the four fixed sessions with the existing eight convolution/matrix/other
subsets, three rotated orders, two warmups+16 measured calls per subset and
64 bracketing full calls before/after: at most2,380 optimizer-free replays.
Preserve every binding and real dependency, record/cache each mask, and verify
reuse for all measured calls. Restore full work before checking outputs.
Require exact292 initial/first-update tensors and reports against the retained
B32 control, exact full outputs after every sweep and all292 tensors afterward.
Compare complete plans/pipelines and first-update inputs, not just family counts.
Masked outputs are invalid model results; no gameplay or learning evidence.

This is a fixed-graph counterfactual attribution, not new whole-update latency,
SM utilization, an additive savings estimate or a matched JAX hardware profile.
Use its measured dominant family to select actual large-batch operations and
their real reachable JAX counterparts. Any subsequent primitive/CUDA Nsight
allocation needs its own concrete unanswered question and review; no automatic
tile, precision, packing or kernel sweep. No CUDA learner or unsafe Vulkan
profiler capture. The broader performance goal remains open.

Serial persistent host-guarded service, expected5080 and>=2GiB sampled Vulkan
estimated budget headroom. Record allocation warnings; API/numerical/hard
failures stop for review, no blind retry/recovery or NVML polling. Preparation/
analysis one CPU,2GiB,zero swap; GPU work has no CPU quota. Production pins,
learning defaults and Atari remain unchanged. Declare exact artifact directory,
commands and binary hashes after the CPU build review and before GPU launch.

Artifacts: `runs/batch-residual-20261014.b7KtbHQM`. Reuse the immutable cached
Meganeura/Blade trees; only a new Kindle snapshot is needed. Add test-only
full-plan/pipeline and first-output exports, plus recorded start/end headroom,
outside the measured calls. The fixture and seed701 remain identical to the
retained control; audit that construction as well as actual first outputs.
No shader or measured replay loop changes. The first CPU build fails before
compilation because the rejected multi-file edit left the old relative dependency
path in place. Preserve that manifest/log; correct the path to the retained
backend and apply the already-declared scratch setting before building again.
No GPU job or numerical gate was affected. Upstream7574ad1/e7c3f7b is unchanged.
The first CPU preflight also retains a Python delimiter error; corrected before
any queue is declared. No GPU retry or acceptance change.

The first allocation completes and its CPU review passes: full fixed graph
54.828ms, allocated4.102 command stream/21.858 convolution/7.543 matrix/21.325
other. Relative to retained B8, convolution and other work account for89.0%
of the36.095ms increase; the command stream does not grow. All five complete
plans/pipelines, first outputs/report and initial/before/after292 tensors match
the ordinary B32 control. All1,536 measured calls verify cache reuse and full
outputs restore. One passing guard/one retained warning,1 update/2,380 replays,
zero actions/workers. These are interacting allocations, not removable savings.

## Bounded follow-up: the world's non-heavy work

World training supplies15.922 of21.325ms of the other-operation allocation.
Before proposing a change, separate its layout/cast materialization, row/
reduction and pointwise/image work. Reuse the explicit classifier and two CPU
tests from `runs/materialization-cost-20261014.yLfoQTsc`, but keep the current
cached backend and all actual B32 plans/weights. Heavy operations stay active
in every subset. Unknown shader families fail CPU preflight; no permissive
fallback category. This is attribution, not a revived row-fusion trial.

One new isolated test-only Kindle snapshot/binary; no production/backend edit.
One separately declared600-second GPU job, one fixed preparatory update and
only the world session: three rotated eight-subset orders, two warmups+16
measured samples/subset,64 full calls before/after and restoration calls,
595 optimizer-free replays. Require exact complete plans/pipelines, first
outputs/report and292 saved tensors against the same ordinary control;
full world fingerprints must also match the first allocation after every
restoration. Check every measured cache reuse and original dependency group.
Keep the first allocation and sources unchanged. Independent CPU classification
and source-delta audit precede launch. Same GPU/CPU/headroom/failure bounds;
no optimizer changes, game actions, new JAX capture or automatic optimization.
The follow-up's first CPU source normalizer catches rustfmt's wrapping of the
longer diagnostic log label. Preserve the failed check; normalize only that
exact reviewed log statement. The source/binary, replay loop and gates do not
change, and no follow-up GPU job had started.

Both allocations and their independent CPU reviews are now complete; see the
[result](../results/2026-10-14-batch-residual.md). The world-only allocation
assigns4.442ms layout/casts,5.379ms row/reduction and5.893ms pointwise/image
work, with26.902ms heavy+command stream. Both exact full-plan/state restorations
pass. Total two guards/one retained warning, two synthetic updates/2,975
optimizer-free replays/1,920 measured cached calls, zero actions/workers.
CPU reachable-dataflow proof identifies a concrete CNN storage/fusion
difference; it is not a measured block speedup or automatic implementation
allocation. Keep the broader goal active and all three CPU failures retained.
