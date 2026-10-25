# Test one barrier before the intermediate convolutions

The captured cut takes7.384ms intact versus4.230ms in five sections, with
86,990,848 exact output checks. Auxiliary work alone is0.0186ms; dX/dW alone
are2.172/2.016ms in the original learner allocation. Separation changes
execution by42.7%, so it is not a passive cost attribution. The first CPU
reviewer's one-sided3% gate was wrong for a declared *within3%* test; its
source/output remain, and the corrected absolute gate fails. No GPU retry.

Artifacts: `runs/convolution-boundary-20261014.axfmoYnu`. Keep the qualified
image candidate's graph, shaders, dtype, allocation, bindings and original
command cache. Split only existing group3974..3979 before3977: the addition
and two reductions precede one new barrier, then both convolutions retain
their shared group. This is one in-pass barrier inside the same cached section,
not five separately timed sections, new passes, altered kernels or a tile sweep.
Apply the split after allocation; require it to occur after the staged prefix,
invalidate the old cached recording, and preserve all original boundaries.

First restore the same292-tensor snapshot and seven captured inputs. Measure
intact/single-boundary target replay,16 warmup+64 measured calls each, with
restoration separately timed. Require exact five outputs per arm, unchanged
plans/pipelines/allocation/292 tensors and >=10% lower target latency before
the separately bounded full-update comparison. Zero learner updates/actions.

If that check passes CPU review, run one ordinary RGB12M/B32 control/candidate
pair,16 warmup+256 measured complete updates each. Same binary/config and only
the boundary option differ; GPU timing stays off. Require exact initial/first/
final292 tensors, first targets and all272 reports, five plans/pipeline keys,
unchanged submission/cache counts, and independently checked staged-prefix and
chunk boundaries. At least3% whole-update gain retains the isolated scheduling
candidate. A failure closes it without timing retries or a boundary sweep.
Neither result proves a particular driver/cache/resource mechanism or permits
adding primitive savings into a full-update prediction. No production promotion,
game allocation, arithmetic change, new JAX capture or goal closure.

Each GPU job has300 seconds under the host guard, persistent service bounds
330 seconds for replay and630 for the conditional pair; Restart=no and
KillMode=control-group, no GPU CPU quota. Require5080 and >=2GiB sampled Vulkan
headroom; record allocation warnings. CPU preparation/review1CPU/2GiB/zero swap.
Review every failure; no blind retries. Upstream99e3de27/e7c3f7b remains the last
checked upstream state; neither supplies a known scheduling fix for this case.
