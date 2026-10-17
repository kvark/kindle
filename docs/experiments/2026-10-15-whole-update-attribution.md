# Refresh current whole-update attribution

Declared October15,2026 at09:02UTC before source edits/build/GPU work. The
halo-cache family is rejected; no geometry/precision/feature sweep follows.
The root-cause goal remains open at54.245ms native versus26.632ms retained
JAX. Previous complete native GPU traces predate whole-program improvements.

Use the isolated076f3135 implementation/checkpoint/configuration unchanged:
RGB12M/B32/T16/BF16, device categorical means, cached commands and the current
four world submission chunks. Meganeura99e3de27/Bladee7c3f7b upstream heads
rechecked unchanged at08:58UTC; no backend dependency/source change.

One test-only instrumentation repair in an isolated Kindle copy: mirror
UniformGpu's existing timestamp harvesting in CategoricalGpu.wait, and drain
its final pending timing before saving the trace. Record the existing single
three-head mean pass; do not add a submission, change arithmetic, synchronization
of acting/training, shaders, scheduling, fixtures or precision. Ordinary mode
does not resolve timestamps. Use a distinct Cargo cache, exclude source-crate
fingerprints from its dependency seed, and verify actual compiled provenance.

Before GPU, Clippy and filtered CPU tests. Then one serial ordinary/coarse
pair,16 warm+256 measured updates each,544 synthetic updates/zero actions.
Each worker<=300s, service<=720s, Restart=no,KillMode=control-group. No concurrent
compile/analysis, performance counter collection, NVML polling or quarantined
profiler retries. Expected5080/580.178.04/boot
`3e89d55c-a9e5-472f-a18a-06508c5bafa7`,>=2GiB sampled Vulkan estimated headroom;
record standalone allocation warnings. API/numerical/hard/deadline failures
stop for review; no blind retries or host recovery. CPU work1CPU/2GiB/no swap.

Require exact272 metric reports/first targets/initial-first-final292 tensors
against retained076f3135; all five plans, pipelines, groups and precision
metadata must match between ordinary/coarse arms. Every measured command-cache
reuse passes: world5 cached chunks/update (prefix plus four backward chunks),
and imagination/slow-value/behavior one each. Keep all samples and tails.
Instrumentation overhead must be within5% of the fresh ordinary control.
One passing numerical trajectory is not an independent model or learning oracle.

Reconcile every coarse GPU pass against source, including272 mean-decoder
passes and asynchronous tails. Separate complete timed intervals, GPU-pass
union and uncovered intervals; pass ranges include issue/dependency costs,
not pure shader time or SM utilization. Uncovered time is not automatically
CPU/idle. Preserve checkpoint/initialization/export work separately. Compare
the retained actual JAX capture at its original kernel/copy/whole-interval
scope, with exact provenance; do not subtract incompatible kernel sums from
native pass intervals or add bypass effects as independent savings.

First classify current full-update costs and check which earlier attributions
still apply to the unchanged current graphs. Select further causal work only
from that evidence; this declaration authorizes no optimization or learning
campaign, no JAX rerun and no automatic follow-up allocation.
