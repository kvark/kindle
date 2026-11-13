# Pixel world models: architecture lessons from DIAMOND

Research only; no successor learner is selected. The fixed
[small RGB64 trial](../experiments/2026-11-08-breakout-world-small64.md) continues.
This review does not change its budget, losses or acceptance gates.

Kindle's [state/history diagnostic](2026-11-08-breakout-world-state-history.md)
shows strong sensitivity to generated image history. Its event heads do not
read that history. The subsequent action/transport composition fails long
physical and event gates. These findings motivate investigating coherent sampled
visual transitions and event readouts; they do not isolate a root cause.

DIAMOND models the next image from four past frames/actions using an EDM
denoiser. Its authors use three denoising evaluations per generated frame,
following a study of ambiguous transitions and rollout stability. The project
reports a 4.4M-parameter Atari diffusion component. This is a different image
generator from Kindle's RGB decoder within each categorical latent draw.
[Project](https://diamond-wm.github.io/),
[paper, sections3/5 and appendixD](https://arxiv.org/html/2405.12399v2).

The official implementation also has a separately trained CNN/LSTM reward and
ending model. It reads the previous image, incoming action and next image.
The simulated environment generates that next image first, then predicts events
from the generated transition. Training clips reward signs into three classes
and classifies termination separately.
[Event model](https://github.com/eloialonso/diamond/blob/main/src/models/rew_end_model.py),
[environment ordering](https://github.com/eloialonso/diamond/blob/main/src/envs/world_model_env.py).

Three implications for Kindle are hypotheses, not established repairs:

- A sampled latent followed by pixel means can still leave unresolved visual
  uncertainty inside each draw. A joint image generator is worth distinguishing
  from another change to the RGB output head.
- Reading events from the generated transition could make image/event predictions
  more coherent. The earlier generated-image-to-posterior feedback experiment
  also changed subsequent RSSM dynamics; it did not isolate a separate event model.
- Independent event fitting could avoid some shared-loss interference. It cannot
  repair wrong visual physics, missing experience or poorly calibrated outcomes.

DIAMOND is not evidence of a cheap replacement. Its paper reports approximately
2.9 days per complete Atari run on an RTX4090, with different training workloads
and life-loss termination. Those costs and agent returns are not matched to
Kindle's world-only tests. A finite frame stack also does not establish persistent
world memory. [Paper, appendicesE/H/I](https://arxiv.org/html/2405.12399v2),
[environment configuration](https://github.com/eloialonso/diamond/blob/main/config/env/atari.yaml).

Any separately selected native experiment must retain physical/event scoring,
ordinary unassisted rollouts, fixed final selection and measured cost. Attractive
sampled images alone cannot establish the full Breakout world-model goal.

Sources inspected November8,2026. Official source-directory blob IDs include
denoiser `8f07bdd99452d2f3409f3eb63b5738a202a5001b`,
sampler `36f6d61cec810a9de618e8ee0c3671e4b6f3ae1c` and
event model `9f661a1692fe0cc26551bff626c35fcb42622a38`.
