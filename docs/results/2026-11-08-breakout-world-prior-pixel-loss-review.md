# Breakout: sampled-prior pixel reconstruction encourages averaged futures

The [new fitting forecasts](2026-11-08-breakout-world-launch-fitting.md) show
multiple faint ball images in a single sampled frame. The exact retained native
source identifies a relevant objective mismatch: the extra H16 loss compares
each independently sampled prior future with one realized RGB future.
This standalone term rewards pixel marginals even when valid futures differ.
It does not establish one cause for every failure of the coupled learner.

Root `/mnt/data/kindle-breakout-world-local-h0-20261108` retains the exact native
7f797a2b source, official-source snapshots and `loss-review.json`.

## The extra term differs from posterior reconstruction

In [the retained rollout implementation](https://github.com/kvark/kindle/blob/7f797a2babebec97edf4a801e4c92e590283ad2e/kindle/src/dreamer/world/pixel_rollout.rs),
each future state samples the prior without future images, decodes RGB, feeds
those generated pixels into history, then scores elementwise pixel cross-entropy
against the recorded next image. Motion weights depend on stopped actual frames.
The rollout lacks an image-conditioned future posterior to associate a sampled
branch with its matching outcome. Straight-through derivatives are surrogates;
they do not change the stated objective into a joint marginal likelihood.

The [pixel loss](https://github.com/kvark/kindle/blob/7f797a2babebec97edf4a801e4c92e590283ad2e/kindle/src/dreamer/networks.rs)
is soft binary cross-entropy per RGB channel. For a latent sampled independently
of the realized target, expected cross-entropy is minimized at each pixel's
expected target, separately for each latent sample. Target-dependent motion
weights change this to a weighted pixel expectation. They do not supply the
missing outcome assignment or a joint image likelihood. Observed reconstruction,
KL, shared parameters and biased gradients complicate the whole learner; the
argument concerns this added term's incentive.

An analytic two-pixel example makes the distinction concrete. The only real
images are[1,0] and[0,1], each with probability1/2. Compare a sharp mixture with
outputs[.99,.01]/[.01,.99] to an averaged output[.5,.5]:

|Objective; lower is better|Sharp mixture|Averaged image|
|---|---:|---:|
|Mean per-sampled-image pixel BCE|4.615221|1.386294|
|Negative log likelihood of the joint mixture|.713146|1.386294|

The first criterion penalizes a valid sampled outcome when it differs from the
particular recorded outcome. The second marginalizes alternatives. This tiny
CPU calculation checks the illustration, not native gradients, learned causal
fractions or the population distribution of Breakout. No model query or update.

The official Dreamer implementation decodes image-informed `repfeat`, with prior/
posterior KL from observed sequences. Its open-loop image reconstruction is in
the reporting path, rather than an additional sampled-prior RGB training term.
See [agent loss/report](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py)
and [RSSM observe/loss](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py).
This is a source-level distinction, not a claim of full Dreamer parity or a JAX
reproduction. No official code or pretrained weights are executed.

## A minimal next recipe

Keep the native local-action CNN/categorical RSSM and inspectable pixel targets.
Enable its existing observed-posterior RGB/KL/reward/continuation training with
H0, removing the extra sampled-prior RGB and imagined-event objectives. At
prediction time, use past images/actions and sampled prior states only. Future
images inform posterior targets during training, never inference.

This local-action/H0 combination is currently rejected by explicit native guards;
it needs a new documented checkpoint format and bounded qualification. The old
H0 control has a global decoder. Previous local/spatial/fusion experiments use
H16. The new combination is therefore not a repeat of those completed learners.
Removing both added losses changes a bundle; success could not isolate one term.

Earlier joint-energy, score-function, latent-overshooting and recurrent-state
experiments remain failed or limited as published. Their negative results do not
justify repeating them, nor prove that a new objective will fix this model.
DIAMOND-style denoising remains another architectural option, not selected here.

Use a separate protocol for implementation, packaging and qualification before
declaring any fixed learner. Retain all failures and existing physical/event
gates. Sharp samples or this loss argument alone cannot establish full-world
accuracy, calibration, brick memory, late-game coverage or fresh confirmation.
