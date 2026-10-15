# Compose repeated-row views with typed scalar fusion

Declared October15,2026 before implementation/GPU work. The current CDP
imagination attribution identifies four F32 broadcast→transpose→typed RMSNorm
chains, writing1,132,462,080 logical intermediate bytes per pass. The existing
row-addressing lowering runs before typed fusion and rejects its precision
marker. This is a compiler composition experiment, not an exploration change.
Upstream Meganeura99e3de27/Bladee7c3f7b were rechecked unchanged.

Extract the existing repeated-row fold into one shared pass. Keep its original
position and run it after typed scalar fusion too. Preserve arithmetic, BF16
rounding, all materialization flags and visible buffers. Only plain F32 copies
may be bypassed; reject broadcast-indexed/gathered/previously repeated inputs,
invalid shapes, and reads crossing a logical source overwrite. Include fused
metadata operands in use counts. No shader geometry, precision or model change.

Use isolated copies of the ordinary99f8aab0 Kindle source and its qualified
Meganeura source. Keep old sources/binaries sealed; production remains unchanged.
CPU tests cover typed outputs, observed/multi-use copies, explicit barriers,
fused metadata reads, source overwrites and broadcast-indexed rejection. Reuse
existing repeated-row/typed-row CPU tests and validate generated SPIR-V. One
bounded preparation/build under1CPU/2GiB/no swap; no GPU tests outside the guard.

Numerical qualification uses original and candidate compilers with identical
new tests: repeated states plus per-action/shared bias, RMSNorm, SiLU, optional
BF16 stored output and a linear probe loss. Shapes(rows,width,actions) are
(3,19,5),(7,65,3),(8,256,18),(129,256,18), two deterministic dyadic fixtures each,
both F32/BF16 outputs. Compare forward and raw state/action-bias/shared-bias/norm
gradients with independent F64 formulas. F32 absolute+relative envelope
2e-5+2e-5*abs(reference), relativeL2<=2e-5; BF16 forward uses explicit RNE and
one BF16 ULP, with straight-through raw gradients checked against the unrounded
linear-probe formula. Require control/candidate arrays bit-exact. Preserve every
case/failure; no tolerance relaxation or geometry sweep.

Each of the16 fixtures executes forward-only and raw-backward graphs: this
checks typed fusion without backward's saved intermediates protecting it, as
well as derivatives. Forward plan must contain one composed view in the
candidate and none in the original. A pre-GPU build-cache provenance check
caught identical original/candidate test binaries; keep that first build and
use a distinct control target directory with CPU plan assertions before any GPU
work. This is a preparation correction, not a failed GPU comparison or retry.

After numerical review, run one24-update candidate smoke each for the unchanged
BF16 CDP12M/B32/T16 and RGB12M/B32/T16 fixtures. Require exact initial/first
tensors, first outputs and metric reports versus their retained controls.
Then one fresh original/candidate pair per workload,16warm+256measured ordinary
complete updates. Require exact272 metric reports and final tensors; no GPU
timestamps/tracing/build/CPU review during measurement. Compare all stage times,
sampled headroom and graph inventory. Accept this candidate with>=3% complete
CDP improvement and no>3% RGB regression; report RGB/JAX separately even if
unchanged. This is not a learning or full-independent-model parity claim.

At most eight serialized GPU processes (two numerical, two smokes, four ordinary
timings),120s each, persistent bounded services,Restart=no,KillMode=control-group.
Expected5080/580.178.04/current boot,>=2GiB sampled Vulkan estimated headroom;
allocation warnings retained. API/numerical/hard/deadline failures stop affected
queue for review, no blind retry. No NVML polling/recovery, Atari, promotion or
old queue restart. All sources/results stay in the experiment directory; update
PR31 at completion. The broader RGB/JAX gap remains an active goal.
