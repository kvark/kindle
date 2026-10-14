# Do independent convolution derivatives interfere in one group?

The [intermediate comparison](2026-10-14-middle-convolution.md) measures exact
actual native encoder dX/dW descriptors separately at2.148/2.015ms. Their sum
is4.163ms, while the mixed intact trace reports8.085ms. That trace fails its
3% whole-update overhead gate and is not ordinary component latency. Neither
number establishes the cause. Test one possible execution-context difference
before changing arithmetic. Artifacts: `runs/convolution-group-context-20261014.riFTxPRm`.

One test-only process, same retained backend/compiler. Build both derivatives
in one graph with shared BF16 dy and the exact B512/32→48/32×32/K5 shape.
Compare independent heavy dispatches in one group versus an additional barrier
between them; the weight-partial reduction retains its dependency in both.
Both arms explicitly use the learner's dX/dW/reduction order; the miniature's
initial CPU test incorrectly assumed its compiler emitted that order. Retain
that failure and verify both heavy operations read only external graph inputs.
The intervention removes no real dependency and changes no operation order or shader. Include
the same output buffers, layouts, data,128 weight partitions and no-alias policy.
This is not the full learner's allocator/cache context or a production change.

Require all outputs to match the retained separate native kernels exactly,
changed-input linearity and exact heavy params/grid/pipeline identity. Each
arm gets16 warmup/128 measured cached-input calls. Native pass timestamps and
queued wall time are reported separately; no cache/shape/tile sweep. One
300-second guarded GPU process in a330-second persistent service. If a group
effect exceeds10%, review before a separately declared full-context check;
otherwise close this explanation, not the overall root-cause investigation.
No whole-update speed or SM-utilization claim from the miniature fixture.

The auxiliary JAX F32 dW control remains numerically failed (127/128 probes);
canonical BF16 JAX/native checks pass with all16,815,616 rounded outputs exact.
No relaxed bound, JAX retry or training. Existing GPU guard/device/headroom/
allocation-warning rules apply, no CPU quota on GPU; CPU preparation and review
use1CPU/2GiB/zero swap. Preserve all prior timing and CPU/numerical failures.
