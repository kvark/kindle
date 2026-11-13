# One-step prior reward supervision: qualification and bounded cost

Select optional native reward supervision on sampled one-step prior states.
The retained reward loss fits observation-conditioned posterior states; imagined
rewards use prior states. Direct prior supervision may improve the latter, but
neither a perceptual cause nor a gameplay benefit is established. The completed
[posterior reward10 comparison](../results/2026-11-06-cdp-breakout-reward10.md)
is negative. The [H3 comparison](../results/2026-11-03-cdp-breakout-open-loop.md)
jointly supervised feature/reward/continuation from one start per sequence and
also regressed. This experiment supervises only one-step reward, at every valid
training row. It does not repeat either allocation. No full learner is selected.

## Objective and invariants

Use the existing deterministic transition and prior categorical distribution,
before the arriving image enters the posterior. Draw one prior sample using the
existing straight-through estimator and separate open-loop RNG stream. Pass its
feature through the shared reward head and use the same aligned two-hot reward
target as the posterior. Add `prior_reward_scale=1` times its cross-entropy to the
ordinary world loss. This scale is independent of the posterior reward scale,
which remains1. No new parameters, inference inputs or checkpoint tensors.

Mask episode starts with the existing keep-action mask. Preserve the ordinary
B*T denominator, including masked rows, and microbatch/chunk averaging. Do not
mask genuine terminal arrival rewards. Prior gradients reach dynamics and earlier
posterior states; the arrival image/posterior cannot affect its own prior reward
loss. Report unscaled loss and valid fraction. Omitted option preserves legacy
serialization, training graph and RNG consumption. Reject nonpositive/nonfinite
F32 scales, non-CDP, simultaneous open-loop/prior-continuation supervision and
fresh-training overrides on restore/evaluation. Restore requires native support.

## Qualification before learning

CPU checks cover config round trips/rejections, CLI construction boundaries,
native capability guards, causal graph dependencies and derivative routing,
parameter/inference invariance, ordinary/staged compilation and strict Clippy.
Package the wheel and validation executable outside Git. Pin source, dependency,
native, runner and teacher bytes before each GPU allocation.

Run two guarded native fixtures: masked CE values/logit gradients against an
independent F64 reference at1/17/256/513 rows and255 bins, then a tiny full graph
with a deliberately nonzero reward output to check upstream derivatives, arrival
invariance and zero loss/gradients on masked rows. Include extreme logits,
two-hot targets and all-masked input. Loss tolerance5e-5; logit derivative
tolerance1e-7+3e-6*abs(reference). The straight-through sampling rule is reused;
hard-argmax finite differences are not a derivative oracle. Each fixture has a
180-second cap. Preserve original failures and separately declare any repair.

Check one default-off frozen1,024-action restore against the retained seed22003
checkpoint and its exact runner, with invariant tensors and matching semantic
traces. This is scoped compatibility evidence, not broad runtime equivalence.

## Bounded learning declaration

After qualification, run one fresh8,192-action Breakout check, seed22003,
200-second training cap, followed by one1,024-action frozen restore. Use
12M64/Auto/N8/B16/T16/fullBPTT/micro16/context1/H15/R16, base/encoder4e-5,
dynamics4e-4,behavior8e-5,policy1/critic1/replay-value1,reward1,continuation1,
CDP500/disagreement1,expert100/unit weights/world gradients,actor/latent mixing.01.
Keep full18 actions,sticky.25,repeat4,noops0,native pooled RGB/one GPU64 resize,
sampled posterior/collection,replay100k,20% episode guidance/cap1920 and the
retained teacher. Other optional interventions remain off.

Require357 initial tensors and47 pre-update events matching retained22003,
first update376,489 finite updates, changed CNN/reward/actor/critic, exact loss
arithmetic, correct mask fractions, full recorded pixel/action/label replay,
zero frozen updates and357 invariant tensors. Record aggregate actions and raw
frames separately. Require2GiB measured GPU headroom and host guards. Conservative
200k projection is bounded time*25*1.10; it must be below30 minutes before a
separately declared200k comparison can be considered. No automatic extension,
coefficient sweep, second learner or confirmation follows.

Root `/mnt/data/kindle-breakout-prior-reward-20261106`. Build caches stay on that
filesystem because the system partition is nearly full. Single GPU job at a time;
check live service/process identity, stop on actual faults/nonfinite values and
never reset/reload the driver. Existing negative allocations remain closed.
Retained22003 scores75.250/44.708 and1009 scores94.625/70.500 on the common cohort;
all0/24 two-wall. Original864/90% gate and the full five-game goal remain unfinished.
