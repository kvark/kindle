# Trace the actual B32 JAX image-training pipeline

October14,2026. Follow the [native image-block attribution](2026-10-14-image-block.md).
The independently qualified native block allocates78.4% of its complete cost
to non-convolution work. Its miniature JAX counterpart failed the actual-size
gradient gate, so it establishes no speed ratio and receives no further retry.

Question: which image-training operations and library calls actually consume
time in the complete, previously qualified upstream B32 learner, and which
native intermediates are absent or combined in that executable? Attribute
forward and backward through reachable compiled HLO and actual CUDA launches;
do not infer speed from kernel names/counts or call posterior values forecasts.

Artifacts: `runs/jax-image-profile-20261014.qpFghMhM`. One bounded900-second
GPU job in a960-second persistent systemd service, serial host guard,
Restart=no, KillMode=control-group, no GPU-worker CPU quota. Same benchmark,
configuration, initial288 tensors, Python environment and source as the
completed B32 control. Only add its existing NVTX markers and the previously
working Nsight Systems2025.5.1 CUDA/NVTX/library trace, including graph nodes.
No Vulkan GPU timestamps/Graphics capture, metrics collection or NVML polling.

Exactly16 warm+256 measured synthetic updates, with updates16–23 captured;
zero game actions. Complete all updates and drain deferred outputs after the
capture stops. Retain full compiled HLO, contracts, fingerprints,272 metric
reports, timing/accounting, trace and guard. Check initial/final state and
reports against the retained control, disclosing any compiler nondeterminism
instead of calling changed outputs exact. Keep ordinary complete27.187ms as
the timing reference; instrumented duration is not a new throughput result.

CPU preparation/review uses one CPU,2GiB,zero swap. Before launch verify
benchmark/dependency/profiler hashes and the declared device/driver/boot;
require5080 and>=2GiB sampled Vulkan estimated budget headroom. Record
standalone allocation warnings. Numerical/API failures, hard faults and
deadlines stop for review; no blind retry, recovery or tolerance change.
No production repin, learning allocation, automatic dtype/layout/tile sweep
or claim of full-goal completion. Select an intervention only from measured
connected costs and verify it on a whole update.

The CUDA capture completes: one passing guard/zero warnings, all272 reports
and initial/final288 tensors exact. All instruction lines match after sorting
backend-config JSON; the module-header fingerprint differs. Native leave-one-out
checks separately locate large bias/pool/cast/backward materializations.

Complete the whole-update comparison with one ordinary/coarse native B32
pair, reusing the already built/qualified B32 fixed-mask binary and the
previously qualified cached primary-boundary timestamps (not Nsight Vulkan).
Both use one world submission chunk, unchanged graphs/configuration/precision
and initial checkpoint;16+256 updates each,544 total/zero actions. Two<=300s
jobs, serial service<=660s, same headroom/guard rules. No build or new kernel.
Retain ordinary four-chunk68.344ms separately. Require exact272 reports,
all five plans/pipeline keys/first outputs and initial/first/final292 tensors
against that control; all measured cache reuses must pass. Require<5% paired
instrumentation overhead before interpreting timings. Attribute complete GPU
pass coverage and host decode/target intervals, not SM utilization. One pair,
no automatic follow-up allocation or optimization sweep.

Both native GPU arms pass. The CPU audit initially stops during NumPy import:
the local trace helper `inspect.py` shadows the standard library, just as the
earlier block fixture's `queue.py` did. Rename the unchanged source to
`inspect_trace.py`, retain the failed service log, and rerun only the CPU
audit. No GPU work, output/tolerance change or recovery is required.
