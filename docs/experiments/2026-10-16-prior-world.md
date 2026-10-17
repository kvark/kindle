# Prior batching: actual-world staging qualification

October16,2026. Continue the reviewed [finite-precision diagnosis](../results/2026-10-16-prior-gradient.md)
and the [original declaration](2026-10-16-prior-batching.md). The original14 F64
array failures remain failed; this is not a replacement cohort or relaxed retry.
Local forward/backward arithmetic, exact rounding, both retained arms, all eight
pairs and the clamped controls support testing the actual world-model boundary.
No full-model independent oracle or bit-exact Vulkan simulation is claimed.

Use the already built isolatedaca80566, unchanged834241b6 backend and existing
RGB/centered-CDP12M B32/T16 BF16 checkpoints/configurations. Four processes:
RGB serial, RGB batched, CDP serial, CDP batched. Change only
`KINDLE_BATCH_TRAINING_PRIOR`; keep word staging/image views/device means/cache.
Do not rebuild. Each process runs the existing actual-world test: six raw
backwards, two whole/staged pairs, original/changed late targets, NaN-poisoned
prefix isolation and original restoration. No optimizer updates or game actions.

Require exact whole/staged outputs/gradients within each arm, immutable model
and optimizer tensors, unchanged plan/groups/pipelines, and exact initial
tensors/posterior/first targets between arms. Check every original and changed
raw gradient with the original pointwise2e-5+.02|reference| and relativeL2<=.01
gates. Compare all output arrays under the same gates, retaining exact counts.
Confirm the disabled graph matches the retained compiled control; verify actual
prior product reduction and unchanged convolution contracts. Retain all failures.
If any gate fails, stop and review; no timing follows automatically.

After this reviewed qualification only, the original one-pair complete timing
allocation may proceed unchanged:16 warm-ups/256 samples per arm/workload,
>=3% RGB improvement and <=2% CDP regression, all samples and trajectories kept.
No timing repeat, subset, batching-length sweep or production/Atari promotion.

Use the host guard and persistent bounded services. GPU processes240s each,
queue1080s, no CPU quota; expected5080/580.178.04, >=2GiB sampled estimated
headroom. Record allocation warnings; API/numerical faults and deadlines stop
for review. CPU preparation/review1CPU/2GiB/no swap. No recovery/driver changes,
separate NVML polling or additional compiler project. Upstream remains
f648a453/4845d652. Artifacts: `runs/prior-world-20261016.YVA1AFTT`.
