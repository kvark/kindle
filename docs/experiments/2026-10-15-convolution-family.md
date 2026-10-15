# Check the shared spatial contraction across the actual image path

October15,2026. Follow the [qualified CNN1 derivative](../results/2026-10-15-convolution-qualification.md).
That one operation saves0.856ms versus native; it occurs once in the retained
60.041ms update. Do not start compiler integration from that isolated gain.

One shared typed forward/input-gradient contraction, using the fixed valid
GPU-halo input contract and K32 reduction. Cover all thirteen distinct actual
RGB12M/B32 forward/dX contracts: seven forward and six dX shapes, representing
eight forward/seven dX occurrences in the staged world plan. The repeated
64→64/HW8 contract is counted twice, not executed as two independent cases.
The first encoder RGB input has no dX. All are K5/stride1/pad2, batch512 for
timing and batch1 for full F64 checks. No weight-gradient implementation yet.

Use128 threads,32 output channels per block and min(256,HW²) spatial positions;
this is one fixed shape rule, not a tile search. Pad input channels to32 and
spatial bounds with a two-pixel GPU halo. Forward/dX differ in filter transpose
and spatial direction, not in matrix machinery. BF16 operands/F32 accumulation;
forward output rounds to BF16 in an included GPU conversion. Keep NCHW external
interfaces, all packing/dependencies/scratch inside each measured operation.
Compile the same source for F16 as coverage, without claiming F16 numerics or
performance. No Naga/Blade change or production dependency.

First compile and independently check all indexing/coverage on CPU. Thirteen
validated BF16 processes each run three numerical operations (original, input
times−.5, restore), signed wide-exponent operands from the preceding fixture0.
Save every input/output and check all outputs against F64 spatial contraction.
dX gates: relativeL2<=2e−5, pointwise<=1e−4+1e−5|reference|. For forward, compare
against independently BF16-rounded F64 with the declared BF16 rounding envelope:
absolute error<=1e−4+0.008|reference| and relativeL2<=.004. Also retain exact
bit comparison with the native dyadic control; no changed tolerance after data.

Only after numerical review passes: one native/candidate ordinary pair,16
warmup+128 timed calls/contract. Native uses the retained current backend,
with per-contract heavy shader/parameters/grid/pipeline keys checked against
the actual world plan. Every output must match the dyadic control; include
changed-input/restore tests and independent F64 probes. Aggregate each distinct
operation with its actual occurrence count; retain all per-shape results.
The whole spatial family must save>=20% in both GPU and queued-wall operation
time to justify further integration. No cherry-picked selector, repeated timing,
geometry sweep or additive whole-update speed claim. Failure closes this fixed
family check for review; it does not authorize tuning.

Persistent serialized host-only guards, current5080/580/boot,>=2GiB sampled
Vulkan headroom.120s numerical processes/600s native timing/300s candidate
timing,60min service bound,Restart=no,KillMode=control-group. CPU preparation/
review1CPU/2GiB/no swap; no compilation/review during matched timing. Allocation
warnings retained; numerical/API/hard faults stop the affected job for review.
No NVML polling, recovery, game actions, learner updates or automatic promotion.
Preserve all failures. Full JAX gap and production/Atari goals remain open.
