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
