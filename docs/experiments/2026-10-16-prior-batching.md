# Batch the independent training prior

October16,2026. Follow the [execution-contract audit](../results/2026-10-16-nonconvolution-contract.md).
Keep the qualified834241b6 backend, RGB/CDP12M B32/T16 recipes, checkpoints,
precision, losses, replay ratio and exploration. Production/Atari stay stopped.

One isolated Kindle graph change batches training-prior and KL work over time.
Observation core/posterior and imagined prior remain sequential. Retain reset
masks, per-stream sampling, free-nat clamping per state, asymmetric stop-gradients,
temporal weighting, BF16 conversion boundaries and late-target staging. An
explicit experimental switch retains the old graph in the same candidate build;
its disabled path must match the retained control before timing is interpreted.
No backend kernel, model or compiler change is mixed into this trial.

CPU checks establish parameter/input/output compatibility, time-row ordering,
posterior independence, reduced prior product count and unchanged non-world
graphs. Use a separate source copy and Cargo cache; build/Clippy and selected
CPU tests precede GPU qualification. Check actual compiled programs, not binary
hashes alone. Native BF16 rounded matrix inputs/results remain explicit.

First qualify standalone production Prior+KL with independent F64 forward and
analytic derivatives, including BF16 operand/result/derivative rounding. Include
Tiny B3/T3 free-nats0, Tiny B3/T4 free-nats1, 12M B32/T16 free-nats1 and12M
B32/T16 fully clamped free-nats1e6, each with two deterministic input variants
and both serial/batched graphs. Retain every output and raw gradient. Forward
values use a one-BF16-ULP envelope plus1e-5; losses use3e-4 times max(1,abs(ref)).
Raw gradients require pointwise error <=2e-5+.02*abs(ref) and relativeL2<=.01;
both-zero arrays pass exactly, nonzero norms use only1e-12 denominator floor.
The same derivative gates apply to serial/batched pairs; no relaxed rerun.
The oracle is independent arithmetic, not a full-RSSM reference.

Then check actual RGB and centered-CDP12M whole/staged world gradients, changed
late targets, NaN-poisoned prefix isolation and restored inputs without optimizer
updates. Compare every raw gradient between old/new graphs with the above gates;
within each arm whole/staged outputs and gradients must match exactly. Initial
tensors and first posterior/targets remain exact; immutable parameters and plans
must stay unchanged. Preserve complete sources, failures and warnings.

Only after reviewed qualification, run one control/candidate complete-update
comparison per workload:16 warm-ups and256 measured updates each, same source,
checkpoint, random fixture and timing boundary as the retained comparison.
Keep all samples, first outputs and initial/first/final tensors. Changed floating
trajectories are not exact parity or learning evidence. Require >=3% RGB gain
and <=2% CDP regression for a qualified performance candidate; no retime, subset
or alternate batching-length sweep if this fails. A passing result is isolated,
not automatic production promotion or a solution to the full JAX gap.

Persistent bounded services: CPU preparation/audit1CPU/2GiB/no swap; GPU workers
have no CPU quota, are serialized by the existing host guard and have bounded
per-process/queue deadlines. Expected5080/580.178.04, >=2GiB sampled Vulkan
estimated headroom. Record allocation warnings; API/numerical failures, hard
faults or deadlines stop for review. No driver/clock changes or separate NVML
polling. No new game actions or learning campaign.

Upstream checked: Meganeura f648a453 unchanged; Blade now4845d652 includes
workgroup cooperative matrices and BF16 capability alignment. Those features
do not implement training-prior batching and are reserved for a separate
feeding-contract review; do not silently repin the matched control.
Artifacts: `runs/prior-batching-20261016.w4yGvAmq`.
