# A type-aware narrow-product alternative

October14,2026. The [weighted diagnosis](../results/2026-10-14-heavy-issued-work.md)
locates3.785ms in five checked RGB-boundary convolutions. Two three-output
kernels each occupy10.67x their useful matrix products. Test one coherent
algorithm alternative, not a tile/threshold sweep or a new precision policy.

Reuse the checked family's existing dense/grouped/convolution operand indexing,
types, K partitions and output layout. For1–8 logical rows, each scalar lane
owns one column, loads B once and keeps all real row accumulators in registers.
Use F32 arithmetic and256-element compensated blocks, explicit typed loads and
the existing BF16 RNE store. No padded cooperative products or shared staging.
One128-column workgroup geometry; no search. The generation is type-aware,
not a BF16/Dreamer-specific shader. Existing general kernels remain the control.

Explicit narrow selection is independently tested for F32/F16/BF16 and matrix
layouts; automatic diagnostic selection initially changes only eligible checked
convolutions. No unrelated matrix intervention or installed runtime change.

CPU first: WGSL/SPIR-V validation across types/layouts and narrow-selection
guards, existing shader tests and Clippy, one CPU/2GiB/zero swap. Retain failures.
Then one guarded numerical allocation<=240s: ragged/transposed/partitioned
matrix controls and forward/dX/dW convolution cases, including3-channel
boundaries, long4096-position partials, aliases/poison and explicit output
rounding. Independent F64 gates stay unchanged; compare reduced storage to
rounded inputs, not an unrounded oracle. F32 candidate need not require F32
cooperative hardware. No alternative optimizer updates before numerical review.

After that review, one24-update RGB12M/B8/T16/H15/BF16 smoke per arm. Exact
initial292 tensors and unrelated plans/pipelines/descriptors required; retain
changed first outputs/derivatives/trajectories, not assumed bitwise equality.
Only the reviewed narrow shader/grid selection may change. Then one ordinary
16-warmup+256-measured full-update pair, four world chunks, no profiler or
concurrent compilation. No timing repeats. A>=3% complete-update improvement
may justify further reviewed integration; otherwise close this candidate,
without relaxing gates or adding a geometry sweep. No automatic CDP extension.

Artifacts: `runs/narrow-products-20261014.YS5lxYKB`; retained optimized056562b
backend andBlade96c1dc7/Naga a97559b. Upstream7574ad1/e7c3f7b rechecked unchanged.
Persistent guarded systemd services, Restart=no/KillMode=control-group,
expected RTX5080/>=2GiB sampled Vulkan budget headroom; record allocation
warnings, no GPU CPU quota. API/numerical/hard faults stop for review; no blind
retry or driver recovery. Individual learner fixtures<=600s/service<=1800s.
No Atari, production adoption, learning-equivalence or whole-gap closure claim.
