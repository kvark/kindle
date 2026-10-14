# Does wide image storage explain the residual gap?

The retained complete RGB12M/B32 update is65.74ms native versus27.19ms JAX.
The earlier wider-JAX control did not widen the image islands. Before adding a
connected image-training kernel, measure that omitted policy directly. Artifacts:
`runs/image-region-20261014.2wQILUdF`. This is a diagnostic, not a native change.

One JAX intervention keeps convolution inputs/weights/results and every matrix
product BF16, but converts convolution results to F32 before channel bias.
Image pooling, channel normalization and activation then remain F32; the next
convolution narrows its input. The decoder's rank-four spatial normalization
also widens. Encoder tokens narrow at the RSSM boundary; other networks stay
unchanged. Parameters, optimizer, batch, work, seeds and benchmark boundaries
stay fixed. Preserve upstream scopes and parameter names. This approximates
the native image policy, not its layouts, pooling tie derivative, convolution
weight-gradient outputs or full rounding contract. No numerical-parity or
storage-only claim: precision can change fusion and library algorithm selection.

First qualify the changed nonconvolution region against independent F64 NumPy
values/derivatives: channels3/32, pooling/no pooling, random values and exact
pooling ties. Pointwise gate3e-5 +3e-5*abs(reference), relative L2<=3e-5;
zeros use the pointwise gate. This does not retry the failed miniature convolution
comparison. Require a CPU audit of the exact runtime method edits. A guarded
120-second process runs these checks with no learner updates or game actions.

After review, one fresh original/wide-image complete B32 pair,16 warmup+256
measured updates/arm,900 seconds/arm. Require identical initial288 state tensors,
finite272 metric reports/final tensors, unchanged heavy MACs/shapes and BF16
operands/results in the compiled executable. Audit actual image reduction and
activation dtypes, not only Python intentions. Preserve all timings and compiler
algorithm changes. Final trajectories are expected to differ. No parameter
sweep, new capture, production repin or Atari allocation. If widening does not
explain most of the residual, do not present BF16 image storage as its main cause.

GPU work uses persistent guarded services, Restart=no, KillMode=control-group,
no CPU quota, expected5080 and at least2GiB sampled Vulkan estimated headroom.
Record standalone allocation warnings; review every numerical/API/hard fault or
deadline before follow-up. CPU preparation/review uses one CPU,2GiB,zero swap.
No NVML polling or driver recovery. The connected native forward/backward
implementation remains unstarted pending this causal result.
