# Actual-model typed selection

Continue the qualified typed selector, not its small-fixture winners. Keep the
qualified d3ff4948 RGB/CDP control and latest c19cb1ca/4845d652 dependencies.
Use a distinct source/cache in `runs/model-selector-20261019.0xWDbwn1`.
No production or game learning. CPU preparation/build/review: one CPU,2GiB,
no swap,720s per service. First inspect the complete candidate scratch bound,
including split reductions, staging and per-candidate encoder allowances.

One fixed qualification per RGB/CDP12M B32/T16/H15 workload restores the original
checkpoint and runs one ordinary initializer update to populate actual program
inputs/late targets. Save initial/first states. Disable optimizers for diagnostic
replays, preserving moments/counters. Each of the four distinct programs
(world, imagination, behavior, slow value; posterior aliases the world prefix)
gets one common dense-selector race: existing catalog,128 classes,512MiB scratch,
60s soft deadline/program, default calibration/pairs/improvement rule. No
manual geometry/subset, changed precision, convolution or grouped-product search.
Search uses its private fixtures; qualification uses populated learner inputs.

Save all graph outputs/raw parameter gradients and plans before/after selection;
require exact parameter/moment state across search and diagnostic passes. Compare
outputs/gradients using the existing world qualification gates: pointwise
2e-5+.02*abs(reference), relativeL2<=.01 per array, all finite. Retain exact counts
and all residuals. Stochastic outputs are not exempt. World whole/staged replays
must match exactly within each arm, with poisoned late reward targets overwritten
after preparation, two reward-target variants and original restoration. Other
programs also replay their original inputs after search. Require command-cache
invalidation and complete reports for every visited class. Incomplete races keep
the incumbent and are reported; errors/numerical failures stop for review.

Guarded serialized GPU processes:600s each,12GiB/no swap/no CPU quota; controller
1500s. Expected5080/580.178.04/current approved boot,>=2GiB sampled Vulkan
headroom, recorded allocation warnings and reviewed historical application faults.
No NVML polling, driver changes or recovery. No blind retry. Numerical review
precedes any separately declared complete-update timing; no speed or learning
claim from these private search timings.
