# Intermediate encoder derivatives against JAX

The intact cached trace preserves all results but fails its whole-update
overhead gate (3.346% versus 3%). Its mixed group3974..3979 contains encoder
input/weight derivatives and two small reductions; its8.085ms is a lead, not
qualified ordinary component latency. Separate the two derivatives before
changing kernels. Artifacts: `runs/middle-convolution-20261014.mSEtvvHC`.

Use the retained image-training backend, no sparse instrumentation or kernel
changes. Both derivatives have B512,CI32,CO48,H/W32,K5,stride1,padding2; BF16
inputs/multiplication, F32 native output. Verify their actual shader/grid/params
and pipeline keys against the full learner, including128 weight partitions.
The original packing cliff and previous rejected packing-grid trial remain.

One native process and one JAX/CUDA Nsight Systems process,300 seconds each,
660-second guarded service. Each case checks finite outputs and changed-input
linearity, then16 warmup/128 measured calls. JAX gets16 additional captured
calls/case. Native pass timestamps and queued wall time have different scopes;
report both. JAX's normal BF16 output is compared after explicit rounding;
separate F32-output controls may widen operands and are not storage-only tests.
Save executable HLO and check algorithm identity against the full JAX learner.
Inputs are deterministic bounded synthetic values, not captured activations.

Independently check128 coordinates/derivative in F64 (native1e-4+1e-5|reference|;
JAX BF16 one rounding step plus that bound) and compare all outputs with explicit
dtype accounting. No timing sweep, optimization, production repin or Atari work.
Do not infer whole-update savings or native hardware counters from miniature
kernels. Only follow a demonstrated expensive derivative with a causal test.

Use persistent serialized GPU guards, Restart=no,KillMode=control-group, expected
5080, >=2GiB sampled Vulkan budget headroom; retain allocation warnings. CPU
preparation/review1CPU/2GiB/zero swap; no CPU quota or compilation during GPU work.
Keep failures and review them before follow-up, no blind retries. Upstream
Meganeura99e3de27/Bladee7c3f7b was rechecked this round; no identified new fix.
