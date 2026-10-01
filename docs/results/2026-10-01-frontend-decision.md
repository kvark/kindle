# Phase 2 decision: learned RGB for 2D screening

**Phase 2 is complete.** Use the faithful jointly learned Dreamer RGB frontend
for small 2D Atari experiments. Keep frozen causal Tiny as an explicit video/3D
hypothesis. Do not launch another representation matrix or extend replication.
[Machine-readable decision and costs](2026-10-01-frontend-decision.json).

## Evidence

The [replication](2026-09-30-small-replication-learning.md) finishes three
upstream/native seed pairs in **7/10 attempts**, including the retained
interruption. The separate [JEPA study](2026-10-01-small-jepa-learning.md)
finishes its **six declared runs**, reusing all three native RGB controls.
All completion, counter, checkpoint-finiteness and guard audits pass; workers
are reaped. No failures or unfinished episode tails were discarded.

Seaquest is held out from Tiny's pretraining titles. Each run uses
Size1M/N8/B8/T16/H15/R32, 200,000 actual actions and 49,939 learner updates.
Seeds are 1009/2017/3019. These are online last-50 episode means, not mastery.

| Frontend | Final score [seed-bootstrap 95% CI] | Mean run time | World train ms/update | Trainable + frozen parameters |
| --- | ---: | ---: | ---: | ---: |
| Learned RGB | 368.0 [328.8,440.4] | 26m31s | 12.769 | 688,004 + 0 |
| Pretrained Tiny | 225.3 [206.8,258.0] | 31m17s | 6.208 | 830,513 + 5,486,592 |
| Initial Tiny | 230.7 [218.8,239.2] | 31m19s | 6.208 | 830,513 + 5,486,592 |

Pretrained-minus-RGB paired final difference is −142.7 [−229.2,−76.8].
Pretrained-minus-initial is −5.3 [−32.4,+24.0]: no established pretraining gain.
Three-seed bootstrap intervals are coarse. Construction adds about 5.3–5.5s/run.

The full action/time curves matter: at 98,304 actions, pretrained/initial Tiny
means 257.7/264.5 exceed RGB's 205.7; at 147,456 they are 250.5/289.9 versus 248.4.
At 900s, the interpolated means are 258.0/266.5 versus 191.0; at 1500s they are
244.6/291.3 versus 340.8. These descriptive points lie inside the common measured
time support (36.11–1569.60s), not extrapolated endpoints or new selection gates.
RGB does **not** dominate every part of every curve. Tiny's extra pretraining
nevertheless establishes neither a consistent learning benefit over initial
weights nor enough benefit to justify its whole-agent cost here.

Tiny reduces world-training time by 51.4%, but the whole run takes 18.0% longer.
The [timing diagnosis](2026-10-01-jepa-timing-attribution.md) shows that the first
replay readback also waits for earlier queued observation/perception work;
readback is already batched. Do not call that entire wait copy latency or GPU
utilization. Isolating perception kernels is a future optimization, not a reason
to rerun this fixed comparison. Memory records are sampled Vulkan estimated
budget headroom, not physical free memory or a measured peak.

The [held-out probes](2026-09-27-representation-probes.md) likewise do not
establish Tiny's value: Seaquest ridge position/motion R² is .529/.006 with
pretraining versus .526/.032 at initialization. Large decodes state best, but
changes capacity and video corpus; it does not rescue the small-frontend result.
The corrected short MLP fits are weak and are not evidence that pixels lack
information. The cancelled 24-run 12M partial study remains historical evidence,
not a completed three-seed benchmark.

## Implementation and verification

- `atari_vector.py ENV --output PATH` now selects the existing learned-RGB
  constructor and reconstruction 1 / future-prediction 0 objectives. Native frames
  undergo one GPU RGB64 resize. No new native learner or CPU workaround.
- `--encoder-checkpoint PATH` opts into frozen causal Tiny; `--encoder levjepa`
  explicitly selects Large. JEPA's native-detail preprocessing is unchanged.
  The positional checkpoint/`learned-cnn` sentinel is removed, without a
  compatibility layer. The profiler forwards the same explicit option.
- **976 Python tests pass**, including fresh default RGB, explicit JEPA,
  independent streams, failure cleanup and both frozen restore routes with
  zero learner updates. The chosen native constructor/loss path already has
  the three complete learning runs above; this interface change needs no new
  training campaign or unchanged Rust rebuild. [CI250](https://github.com/kvark/kindle/actions/runs/36835906681)
  passes on Linux/lavapipe, macOS/Metal and Python for implementation `f84d9b6`.
- Weight-taking pixel-agent constructors and native capture integration keep
  causal Tiny for video/3D research. This is a scoped 2D default, not removal of
  the latent world-model hypothesis or a change to existing checkpoints.

RGB versus Tiny compares complete observation/objective packages, not solely
encoder pretraining. Tiny previously saw 250k RGB64 random-play frames from
Boxing/Pong/Freeway/Breakout/Qbert, additional experience. Only pretrained versus
its own initial Tiny isolates that intervention. One short held-out game cannot
settle generalization, asymptotic performance or arbitrary-video learning.

Next is Phase 3: one GPU-compatible exploration mechanism versus extrinsic-only
on the cheap recipe, without Freeway's action aid, with three seeds and an
explicit small allocation. This session does not start that study. No asynchronous
learner, swarm, new pretraining or unchanged mastery queue is added.
