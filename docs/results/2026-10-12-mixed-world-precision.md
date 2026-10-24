# Fixed mixed-stage precision improves CDP

October12,2026. [Declaration](../experiments/2026-10-12-mixed-world-precision.md)
· [Data](2026-10-12-mixed-world-precision.json).

CDP12M/B32 complete synthetic updates fall **146.132→140.726ms (3.70%)**.
World training/synchronization falls61.779→56.533ms; posterior16.827→16.833ms,
imagination57.687→57.538ms and behavior9.839→9.822ms are essentially unchanged.
`mixed` retains native BF16 posterior/imagination/behavior but uses the existing
F16-input/F32-storage world-gradient path. One fixed policy, no per-title branch.

Both guards pass, one allocation warning retained. Identical initial346 saved
tensors and first imagined targets; all272 reports and first/final tensors are
finite and masters update. The BF16 control exactly reproduces the preceding
convolution study's reports/targets/tensors. Selection confirms BF16 in all four
retained sessions and none in mixed world gradients, where native F16 cooperative
convolutions execute. Twelve config tests, the new world-policy isolation test
and strict workspace Clippy pass. No tolerance relaxation, game actions, GPU
recovery, SM-utilization or learning-quality claim.

One binary,16 warmup+256 measured updates/arm, unchanged17c9287/b52965b/c9fd6a85a
foundation, exact checkpoint/RNG/workload. No layout/alignment patch yet.
Logical world buffers5.038→4.130GB and sampled headroom9.905→10.290GB; these are
not physical peak VRAM or measured memory traffic. Preserve source, binary,
guards, all reports/checkpoints and CPU audit in
`runs/mixed-world-20261012.63YIxyjc`. Native SHA256
`5afb0c172a3da99997ca969e91eec844d060eb592ece79fb40e36c05d8dc1777`.

Keep `mixed` as the CDP throughput candidate for final integration. RGB retains
its separately useful `bf16` candidate; this is not a universal precision-policy
promotion. Auto remains default. Full CDP world reduced storage stays unfinished;
the practical result is useful mixed precision, not BF16 everywhere. Final pinned
integration/upstream refresh must qualify before considering fresh learning.
