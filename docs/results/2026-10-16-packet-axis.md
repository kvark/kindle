# Column-major packet staging is rejected

October16,2026. [Declaration](../experiments/2026-10-16-convolution-packet-axis.md)
· [Compact evidence](2026-10-16-packet-axis.json).

The new packet orientation is numerically correct but **11.15% slower** across
the complete convolution family. Keep the qualified word-staging control;
there is no full-update trial, favorable selector or production promotion.
The complete RGB/JAX gap remains **48.068/27.223ms,1.77×**. Atari stays stopped.

| Complete23-product family | Word-staging control | Packet-axis candidate |
| --- | ---: | ---: |
| GPU time |19.089ms|21.217ms|
| Queued wall time |19.136ms|21.260ms|

All20 distinct contracts,23 actual occurrences, original transforms and dW
reductions are included. Each arm retains16 warm-ups and all128 measured samples.
The fresh control is within.13% of its retained19.112ms result. Seven of the
nine changed contracts regress; the14 unchanged packed occurrences remain in
the total. The5% improvement gate fails, so standalone results do not authorize
a whole-update test or revival through a subset.

## The hypothesis and exact change

[Word staging](2026-10-16-word-staging.md) removed most measured shared-bank
conflicts, but dW improved only9.1%. Its uniform-pipe instruction count fell
from315.42M to.46M while ALU instructions rose507.57→765.88M. A possible cause
was losing shared address work when reduction bases changed from`lane/32` to
`lane/8`. That was a source/counter correlation, not source-instruction attribution.

This candidate stores both operands column-major in the same typed padded
shared arrays. A packet holds four M values at one K; B holds four K values at
one N. Each local invocation owns two packets. Pitch40 for BF16/F16 and36 for
F32,128 invocations,32x32x32 tiles, K order, compensation, splits, barriers,
rounding and output interfaces stay unchanged. No global-vector-load or async
copy mechanism is added. All nine scalar-staged occurrences change together.

Ownership uses local invocation indices, not a fabricated dense subgroup index.
The CPU proof explicitly permits permuted subgroup membership and rejects an
unsafe partial-subgroup counterexample. Shared address calculations being
lowered to uniform instructions remains a hardware question: Vulkan does not
guarantee a relationship between local and subgroup invocation indices.
[Vulkan specification](https://docs.vulkan.org/spec/latest/chapters/interfaces.html).

## Counters reject uniformity as a sufficient explanation

| CNN1 kernel | Word-staging control | Packet-axis candidate |
| --- | ---: | ---: |
| dW warp instructions |1,091.53M|1,277.77M|
| dW uniform-pipe instructions |.46M|372.22M|
| dW ALU instructions |765.88M|569.86M|
| dW shared-bank conflicts |26.87M|26.85M|
| dW complete-operation time |1.838ms|1.857ms|
| dX warp instructions |841.09M|946.08M|
| dX ALU instructions |509.15M|619.38M|
| dX shared-bank conflicts |21.53M|21.54M|
| dX complete-operation time |1.120ms|1.424ms|

dW's uniform instructions do return, and ordinary ALU work falls25.59%, but
total warp instructions rise17.06%. Latency is1.01% worse. Occupancy rises
32.76→48.70% without an ordinary speed benefit. The candidate tracks eight
reduction cursors instead of two; source sharing is not a guarantee of fewer
issued instructions. Instruction categories are not interchangeable costs, nor
do these counters identify individual address expressions or machine opcodes.

dX instructions rise12.48% and latency27.08%. Its L2 traffic falls1.512→.867GB
while DRAM traffic stays about114MB and shared conflicts stay essentially flat.
This is a measured counterexample to explaining these timings by traffic or
shared conflicts alone. Useful tensor-pipe activity falls35.36→25.95% of
active-cycle peak. Neither occupancy nor traffic by itself is the target.

Tensor instruction counts stay exact for every contract. All40 fresh control
warp/tensor counts reproduce the retained word-staging capture. Current control
dX/dW still execute8.63×/8.17× the retained full-JAX warp instructions; the
rejected candidate increases those ratios to9.70×/9.57×. The comparison retains
its different layouts/output precision/context and is not a latency prediction.
Ordinary timings and profiling are separate; dW counters cover its main kernel,
whereas the quoted operation timing includes its reduction.

**Keep the successful word-staging layout.** This experiment rejects restoring
uniform-pipe work as a sufficient fix for the remaining gap; it does not prove
all address work is unavoidable, or that memory feeding no longer matters.
No new packet/layout/tile search follows. Next reconcile the remaining native
non-convolution programs and host boundaries with the retained complete JAX
capture before choosing another intervention. Keep common workloads, actual
dependencies and incomparable timing boundaries explicit; do not subtract
standalone kernel sums from the whole-update gap. Convolution feeding remains
open alongside those costs. This next CPU audit implies no new GPU allocation.

## Qualification

The CPU proof passes5,559,244 B addresses,131,072 A addresses,10,491,616 cursor
transitions,40,960 ownership/fragment checks and164,767 masked packet values.
It covers the nine actual contracts and12 small exhaustive rectangular/strided
cases. Four local-ID permutations preserve all packet values. Wrong orientation,
pitch, duplicate ownership and unsafe dense subgroup IDs are distinguished.

All304 typed/control modules validate offline. Fresh controls equal all152
retained WGSL/SPIR-V modules;123 candidate modules change and29 remain exact.
The changed modules retain four cooperative loads, two products, four vector
shared stores, three workgroup barriers and9,216/13,312B explicit shared memory.
Build and Clippy pass. These static counts are not hardware instruction counts.

The numerical phase passes2,690,432 independent F64 values with unchanged
pointwise/relative limits and exact control/candidate outputs. Dense timing
fixtures also match exactly, including changed-input/restoration checks.
Actual plans, pipeline keys, parameters, workgroups and generated shaders are
checked; this is not merely a binary-hash comparison. GPU numerical coverage
is the current BF16 family; F16/F32 variants here receive CPU shader validation.

All120 guards pass, with22 retained allocation warnings and no failed GPU jobs.
There are774,861,888 exact full-output pairs across the three phases and1,274
source/input seals. All116 counter ranges/4,756 metrics re-evaluate exactly from
saved counter images; no ranges or trace bytes are dropped.9,320 standalone
calls include1,360 counter replay passes, with zero learner updates/game actions.
Minimum sampled Vulkan estimated headroom is15,828,123,648B, not physical peak
free memory. GPU work ends06:38:36UTC; independent CPU review06:40:19UTC
(9.212s,2GiB cgroup peak). No GPU worker remains. A malformed exploratory`jq`
read was corrected without changing artifacts or rerunning GPU work; build,
shader/address tests, Clippy and the independent publication audit pass.

Artifacts: `runs/convolution-packet-axis-20261016.JfluRtMn`. The qualified
`834241b6` full-update binary and production sources are untouched. Upstream
Meganeura remains`f648a453`; Blade`01be8446` adds per-entry-point/override
resolution. All152 tested modules prove its unchanged single-entry/override-free
fast path, so that fix does not require repinning this comparison. The user's
dirty Blade tree is untouched.
