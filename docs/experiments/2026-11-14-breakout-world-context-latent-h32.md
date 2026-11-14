# 4M context-frame world with 32-step latent overshooting

The [131k run](../results/2026-11-13-breakout-world-context-h32-131k.md) and
the [sticky-action oracle](../results/2026-11-14-breakout-sticky-oracle.md)
show that the best world is a near-deterministic hedger. Its four prior
draws agree on held-out play (eight steps: 71 to 74 single-draw hits, 75 for
any of four, against the oracle's .903 / .955), and after a paddle bounce
15 of 23 eight-step forecasts show a smeared ball (30 to 90 strong red pixels
in the detector window against 2 to 7 for a real ball). The cause is the
[objective](../results/2026-11-08-breakout-world-prior-pixel-loss-review.md):
the 32-step image suffix scores decoded prior samples against one recorded
outcome, which rewards per-pixel averages wherever sticky actions make the
outcome uncertain.

## Intervention

A fresh life of `context-config-96-4m-h32.json` (4M, 96x96, batch 8, 48-step
windows, context lag 8, spatial decoder, fusion, local action, action
features, motion RGB cross-entropy on observed frames) with
`--pixel-loss motion_latent_overshooting --prediction-length 32` in place of
`motion_event_rollout`: from the detached posterior at index 15, 32 priors
roll under the recorded actions and each matches its detached image-informed
posterior by categorical KL at dynamics scale 1, without a free-nat floor.
No decoder runs on prior states in training, so the decoder never learns to
hedge; the prior learns a distribution over outcomes instead. The image
suffix's imagined reward and continuation terms go with it, so this changes
a bundle, not one term. Native change: three guards admit this objective for
the context-frame world only (spatial decoder, local action, action
features); a graph test checks equal parameters and no prior image sampling.

`--steps 65536 --eval-every 8192 --dense 128` on `corpus/fit96` and
`corpus/dev96`, deterministic, output `train-96-context-4m-latent-h32`,
unit `kindle-world-context-latent-h32-20261114.service`, direct launch under
a 24G cap without the host guard, `MEGANEURA_DEVICE_ID=0x744c`. A 256-update smoke
ran at 392 ms per update (the image suffix: 894), so about 7 hours.

## Readouts and reading

Dense development tracking (one draw) per point against the image-suffix
world's points. At the end, the held-out probe (seed 777001) plus four prior
draws: per-draw, any-of-four and all-four tracking, the event split (paddle,
brick or ceiling, none) and undetected balls after paddle bounces, against
the oracle's one-draw (.903 / .700 / .406 at 8 / 16 / 32) and any-of-four
(.955 / .829 / .612) ceilings, and the image-suffix world's 65,536 point
(held-out 110/90/72/51/37, any-of-four at 131k 75 / 58 / 44).

- **Sampler:** any-of-four exceeds the per-draw mean by more than the
  image-suffix world's spread (about 4 counts at eight steps, 6 at
  thirty-two) and most paddle-bounce forecasts contain a detected ball.
- **Better world:** held-out one-draw tracking at eight to thirty-two steps
  at or above the image-suffix world's 65,536 point beyond checkpoint noise
  (about 8 counts at thirty-two steps).
- A sampler with worse one-draw tracking is a mixed result (outcomes
  represented, dynamics less accurate); a hedger with worse tracking is
  negative. One run, one held-out corpus, development screen.
