# Dynamics-term interaction with the retained actor update

The test-only diagnostic is implemented. All192 Kindle and5 adapter CPU tests
and strict Clippy pass. Package the release executable, then select the single
new guarded diagnostic below. No learner or optimizer update is selected.

The [fixed-plan probe](../results/2026-11-06-cdp-breakout-action-path-derivatives.md)
does not show generally reversed return directions, while the full dynamics.5
learner regresses. Test whether adding that term materially changes the retained
actor's raw gradient and conditional optimizer update. This does not measure
early training or explain the candidate's trajectory causally.

Reuse both completed November3 actor-gradient batches, their full saved behavior
inputs, 22 parameter/gradient arrays per term, original retained200k checkpoint
and optimizer moments. Do not repeat the old RGB capture, posterior collector,
baseline component backward stages or optimizer readers. All3,840 action targets
per batch have exactly one nonzero entry, retaining the sampled action. The
first256 archived imagined features retain every initial posterior state.

Build a test-only native diagnostic using the unchanged production imagination
and behavior graph builders. Regenerate the original independent imagination
RNG stream from checkpoint seed/counters, continuing it between the two batches.
Use the saved initial states and exact original world/actor/critic weights.
Require every sampled action to match and imagined features to match with
maximum absolute error1e-6. Use the original GPU categorical-mean decoder and
normalizer update to reconstruct return scale, weights and weighted advantages;
require normalized maximum error1e-5 against archived weights/action targets.
No fallback to different draws or newly chosen batches is allowed.

At the retained weights enable only actor_dynamics_scale=.5 in memory, without
changing checkpoint files. Load the world snapshot as frozen behavior inputs,
copy the shared action/latent draws and detached disagreement bonuses, and use
the reconstructed normalizer. Critic parameters retain their ordinary value
losses but receive exactly zero auxiliary derivative. World weights have no
optimizer slots in the connected behavior graph. No learner or optimizer step
is called; no emulator, CNN resize or new posterior sampling is required.

Perform two new backward passes per batch: the combined production objective
and the weighted dynamics term alone. A test-only scalar gate preserves the
production graph prefix. Require total gradients and scalar loss to equal the
archived baseline plus the new term, with relative L2<=1e-3 and normalized
maximum error<=1e-3, separately for actor and critic. Store full new derivatives,
draws, snapshot inputs, target comparisons and unchanged parameter checks.
CPU graph tests verify gate construction, input/gradient ownership and exact
production prefix. Existing dynamics F64 qualification remains reused; do not
rerun its closed fixtures. A mismatch stops this new diagnostic without retry.

On CPU, extend the previously qualified AGC/LaProp arithmetic to include the
new term. Calculate both the baseline and combined hypothetical update from the
same saved moments; neither is applied or chained. Baseline vectors were not
archived, so recomputing those vectors is necessary for the new paired
direction/shift comparison. Require their aggregate norms to reproduce the
saved November3 optimizer result. Hold combined clipping/variance fixed only
when attributing its components, and separately recompute each full counterfactual
update's clipping/variance. Report raw and conditional component norms/cosines,
saved momentum, actor update shift/cosine, clipping changes and critic invariance.
These are F64 estimates, not measured native parameter movement or a tuned
coefficient. Reuse the original native-optimizer comparison failure and limits.

Root `/x/Temp/kindle-dynamics-actor-update-20261106`. Build/CPU checks allocate no
GPU. After those pass, declare one serial native job capped at180s/controller240s
under the existing host guard, on an idle healthy device with2GiB minimum native
budget headroom. Pin binary, source, checkpoint, old evidence and inputs. Zero
Atari actions, learner updates and optimizer steps. Preserve failures; no GPU
recovery, automatic retry, coefficient sweep or new learner is selected.
