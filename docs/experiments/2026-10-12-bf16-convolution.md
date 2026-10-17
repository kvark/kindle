# BF16 convolution storage boundary

Declared October12,2026 before GPU execution. Dense BF16 improves full updates
but regresses world training. Extend explicit reduced storage to convolution
forward operands/results; keep F32 masters, accumulation, backward reductions
and optimizer. Reuse the qualified BF16 cooperative product loop with implicit
NCHW gathers, bounded partial-tile stores and no im2col-sized global temporary.
Keep dense-only selection, strict F32 and live acting unchanged. No PR248,
layoutPR249 or hardware-rowPR252 is stacked in this isolated check.

First qualify independently indexed F64 forward and F32-master/input gradients
on BF16-rounded operands, including RGB channels, rectangular kernels, padding,
stride, multiple images and partial tiles. Check fused BF16 output against
format-aware bounds and retain the existing strict forward/gradient gates for
unrounded F32 output. Require actual native BF16 convolution selection and
at least2GiB sampled estimated-budget headroom. Recheck the shared dense kernel
because its loop is reused. Two120-second numerical processes; stop/review any
failure before follow-up, retaining all failed source/binaries/guards.

Only after qualification, compare unchanged dense-only versus dense+convolution
on the retained CDP12M/B32/T16 and RGB12M/B8/T16 fixtures,16 warmup+256 measured
complete synthetic updates each,600 seconds/process. Exact initial346/292 tensors,
finite targets/reports/checkpoints, all selected sessions and ordinary full/stage
latencies must be reviewed. No exact reduced-precision trajectory claim. One
matched timing per recipe, no shape/policy sweep. Keep only a useful whole-update
improvement; report world-stage acceptance separately.

Foundation Meganeurab0ba23c/Bladeb52965b/Naga forkc9fd6a85a. Isolated worktree
`meganeura-kindle-bf16-conv`; artifacts `runs/bf16-convolution-20261012.Y61SZP2u`.
Archived source and immutable binaries precede serialized guarded systemd jobs;
no compilation during matched timing. Record allocation warnings, no NVML polling,
GPU recovery, game actions, default change or resumed learning campaign.
