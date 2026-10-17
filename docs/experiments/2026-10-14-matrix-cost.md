# Current matrix-path attribution — October14

Current isolated complete RGB24.340ms remains1.90× retained JAX12.836ms.
The retained weighted profile allocates4.475ms native matrix work versus2.366ms
JAX matrix kernels, with different timing boundaries. The checked/tiled native
split has not been measured:521/195 dispatches and nominal products alone are
not a reliable priority. This is a bounded attribution, not an optimization sweep.

Use the exact incremental-address backend/command-cache Blade and same RGB12M/
B8/T16/H15/BF16 fixture. Isolate only a new Kindle test under
`runs/matrix-cost-20261014.5lrQ8LIi`. One original update initializes state, then
disable optimizer/accumulation/clipping. Four fixed sessions, four masks:
checked matrices off/on crossed with aligned tiled matrices off/on. Keep every
other kernel, binding, command and dependency. Masked outputs are invalid and
never update any actor. Restore full work before each mask and after each round.

Three rotated rounds, two warmups+16 measurements/mask/session/round,64 baseline
samples before/after with eight initial warmups. All first reports/initial-first
states, exact dispatch/pipeline descriptors, restored output fingerprints and
all292 tensors must match the retained fixture. Classify explicit typed dense/
grouped matrix entries and actual kernel selection; reject unclassified matrix
entries. Preserve grouped/dense/shape inventory separately. GPU intervals are
counterfactual interaction allocations, not pure kernel time, SM utilization,
whole-update speedup or additive possible savings.

One GPU process<=600s through the existing host guard, persistent bounded
service, no CPU quota on GPU work. Expected5080/580.178.04/recorded boot,
>=2GiB sampled Vulkan headroom; record allocation warnings, stop other faults
for review. No NVML polling/recovery/driver changes. CPU preparation/analysis
one CPU/2GiB/zero swap. Zero game actions and exactly one synthetic update;
no automatic kernel/geometry/learning follow-up. Root-cause goal remains open.
