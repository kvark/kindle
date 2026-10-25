# Complete-update selection acceptance

Conditional on independent actual-model qualification review, compare one fresh
untuned/tuned complete RGB pair, then one CDP pair, B32/T16/H15/BF16. Existing
common selector/catalog/options and512MiB bound; no geometry sweep or retiming.
Each process reruns the actual-output/raw-gradient/staging/state qualification
for its own choices. Untuned uses the same call with zero search time. Selection
may differ from the earlier qualification; save/review the actual timed plans.

After one initializer and diagnostic replays, continue the same core and fixture
RNG for16 warmup+256 held-out measured complete updates. Include posterior,
imagination/device targets, world/behavior updates, synchronization and slow EMA.
No profiling/readback diagnostics or fixture construction inside the timer.
Check restored optimizer configuration, counters, finite reports/state, target
normalizer and save/drop/restore roundtrip. Initializer states and untuned metric
prefix must reproduce the retained control; every raw qualification gate stays.
No bit-exact stochastic trajectory requirement between differently selected
kernels, and no learning parity claim.

Require at least3% less complete RGB time, no more than2% CDP regression and
untuned timing within5% of42.983/40.741ms, with every sample retained. A failed
gate stays failed. Search overhead is reported separately, not hidden as training
time or added microbenchmark savings. No production/Atari promotion follows.

Distinct timing source/cache under the model-selector study. CPU build/review
one CPU/2GiB/no swap/720s each. Four serialized guarded GPU jobs600s each,
controller2700s/12GiB/no swap/no CPU quota, same expected5080/driver/boot and
>=2GiB sampled Vulkan headroom. Warnings recorded; numerical/API/hard faults
stop for review. No NVML polling, recovery, automatic retries or concurrent build.

CPU observer scheduling: the first independent array audit completes in397s
under its one-CPU quota with unrestricted BLAS threads. It finishes before an
attempted stop; no interruption/repeat occurs. Set OPENBLAS_NUM_THREADS=1 and
OMP_NUM_THREADS=1 for the upcoming timing audit, with unchanged arithmetic,
720s/2GiB limits and gates. This changes no GPU worker settings.
