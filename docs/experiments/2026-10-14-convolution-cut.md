# Separate the captured convolution group's operations

The captured-input study reproduces the original five-operation interval at
8.088ms with0.315% fixed-graph overhead. Replaying saved inputs in their original
bindings takes7.435ms; synthetic values take7.530ms. Values do not recover the
isolated4.165ms derivative pair. Requested memory placement and three omitted
operations remain unseparated. Artifacts:
`runs/convolution-cut-20261014.4BJY277g`.

Restore the captured study's exact292-tensor `before` state, graph, allocation
recipe and seven saved inputs. No preparatory learner update, game actions,
new arithmetic, shaders, geometry, dtype, optimizer or production change.
The original five operations are independent within one compiler group. Add
diagnostic recording of a nonempty contiguous subset of that original group,
using its original buffers/pipelines; cache keys remain distinct dispatch ranges.
Verify descriptors, pipeline keys, group partition and buffer mapping against
the captured study. These are logical mappings/requested memory classes, not
queried physical Vulkan memory types or guaranteed identical GPU addresses.

Five modes, each16 warmup+64 measured calls: intact five-operation group;
all five as consecutive separate sections; auxiliary Add/two SumRows only;
dX only; dW partials only. Restore all seven raw inputs before every call and
time that transfer separately. Compare exact selected outputs to the captured
study after every mode. The separated complete group must stay within3% of
the intact group's target time for an ordinary group-cost attribution; preserve
a failure without rerunning for a pass. Individual replay costs remain distinct
cache contexts, not additive whole-learner savings. Final intact replay and all
292 checkpoint tensors must match;401 optimizer-free cut replays, zero updates.

One300-second guarded GPU process/330-second persistent service, no CPU quota,
Restart=no/KillMode=control-group. Require the5080 and >=2GiB sampled Vulkan
budget headroom; retain allocation warnings. CPU preparation/review1CPU/2GiB/
zero swap. No blind retries, no JAX repeat, masked ranking, placement/tile sweep
or Atari restart. If omitted work explains the interval, investigate that work;
otherwise distinguish binding/allocation context before changing kernels.
