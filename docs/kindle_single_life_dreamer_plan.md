# Kindle: 100-Hour Game Learning Plan

**Decision date:** 2026-08-23
**Status:** baseline implemented; environment-learning validation in progress

## Executive decision

Kindle's first target is deliberately narrow:

> Given one previously unseen game, can a single agent learn to play it from
> pixels, actions, and game-native feedback within 100 hours of real-time
> interaction?

This is not yet the full "single digital life" problem. It is the smallest
credible experiment that exercises the machinery we care about: visual
perception, action-conditioned world modeling, memory, latent imagination,
online actor-critic learning, and long-run stability.

The proposed agent is a synthesis rather than a direct implementation of either
Dreamer or LeWorldModel:

- a **LeVJEPA-style block-causal video encoder** provides compact perceptual
  targets and can be pretrained on general video and adapted to Pixels2Play;
- a **DreamerV3 categorical RSSM** turns those perceptual observations and
  actions into a recurrent belief state;
- a **reconstruction-free latent prediction objective** replaces pixel
  reconstruction;
- Dreamer's **reward, continuation, imagination, actor, and critic loop** remain
  the behavior-learning baseline;
- optional action-conditioned pretraining gives the world model a head start,
  while the target-game actor and critic still start from scratch.

There is no target image, LeWM controller, CEM planner, runtime tree search, or
GRPO in the first experiment. Those are later ablations. The immediate question
is simply whether the agent's gameplay visibly and measurably improves during
one 100-hour exposure.

The current repository is a useful control: it already runs DreamerV3 with a
frozen DINOv3 frontend and reconstructs compressed DINO features rather than
RGB. The next architectural change is therefore not merely "remove RGB
reconstruction"; RGB reconstruction is already absent. The meaningful change
is to replace the current same-step feature decoder with an explicit
JEPA/CDP-style perceptual-latent prediction target, then swap or augment DINO
with a causal video encoder.

---

## 1. The research question

### 1.1 Primary question

Can Kindle enter a held-out game with:

- no target-game policy checkpoint;
- no target-game demonstrations;
- no privileged game state;
- no cloned or accelerated environments;
- and no per-state save/restore;

then improve from its own experience enough that, after 100 hours, it plays
substantially better than it did at the beginning?

### 1.2 What pretraining is allowed

Pretraining is part of the hypothesis, not a hidden exception.

The agent may begin with:

- a visual encoder pretrained on broad internet video;
- a visual encoder further adapted on gameplay video from Pixels2Play;
- an action-conditioned world model pretrained on other games with aligned
  keyboard and mouse actions.

The target game must be excluded from action-conditioned pretraining and from
architecture or hyperparameter selection. Any passive-video leakage of the
same title must be documented.

Before the target-game run:

- actor parameters are reset;
- critic and slow-critic parameters are reset;
- game-specific reward and continuation heads are reset;
- replay is empty;
- recurrent state is empty.

The visual encoder and compatible world-model dynamics may retain their
pretrained weights. This tests whether generic perceptual and causal priors help
online reinforcement learning rather than whether behavior cloning already
knows the game.

### 1.3 What 100 hours means

The primary budget is **100 hours of agent-visible gameplay**.

At 20 observed frames per second this is 7.2 million frames. At 10 policy
decisions per second it is 3.6 million decisions. The exact observation and
action rates are fixed before the run and recorded in the experiment contract.

All of the following count against the 100-hour budget:

- exploratory play;
- evaluation episodes;
- natural deaths and restarts;
- menu navigation performed by the agent;
- time spent temporarily freezing learning for evaluation.

Offline pretraining does not count against the 100 target-game hours, but its
data, compute, checkpoints, and target-game overlap are reported separately.

### 1.4 Real-time and compute contract

The first target is a single actor on a single locally controlled game instance.
There are no parallel rollout workers.

The intended systems budget is one consumer workstation:

One replay observation is 7×7×64 = 3,136 floats. The seed and projection are
part of the architecture contract, not learned parameters. The Dreamer decoder
reconstructs this compressed DINO feature map rather than raw pixels.
Its final per-patch hidden width is 64: the 4–16 channels inherited by smaller
Dreamer pixel presets are appropriate before RGB output but impose an
unnecessary affine rank limit before a 64-channel feature target.

The actor must continue meeting its control deadline while learning is enabled.
The learner may temporarily fall behind, but the backlog must not grow
monotonically throughout the run. Environment steps, replayed sequence steps,
imagined steps, wall-clock time, and GPU utilization are all reported.

The first spatial-information gate supports keeping the compact representation.
An independent-seed Pong probe collected 512 labeled frames per seed, trained
linear coordinate decoders on seeds zero and one, selected regularization on
seed two, and tested on seed three. Projected 14×14×64 patches reach 0.9890 mean
R² across ball and paddle coordinates; the production pooled 7×7×64 features
reach 0.9867 while using one quarter as many values. Ball-x R² changes only
from 0.9650 to 0.9602 (3.58 to 4.03 source-pixel MAE). Raw resized pixels reach
0.9158 mean R² and only 0.6881 on ball x under the same linear probe. Pooling is
therefore not the evident Pong perception bottleneck, although this static
probe does not establish temporal sufficiency or downstream control quality.

A second seed-split probe tests the actual published-protocol reward frames.
It trains on random trajectories from seeds 10 and 11, selects ridge penalties
on seed 12, and tests on seed 13 while retaining every reward event and a fixed
5% sample of zero-reward frames. The held-out split contains 117 negative, four
positive, and 248 zero targets. Pooled 7×7 features reach 0.9929 reward-event
ROC-AUC and 0.9911 negative-vs-rest ROC-AUC, compared with 0.9886 and 0.9830
for raw resized pixels. Positive-vs-negative AUC is 0.9936 for both, although
four positives make that estimate directional. The compact DINO input therefore
preserves the immediate visual evidence of scoring; the 10,000-step checkpoint's
weak 0.646 posterior reward-event ranking arises downstream of frozen vision.

## 4. Baseline contract

## 2. Operational definition of “learn how to play”

| Area | Baseline |
|---|---|
| Replay | Fresh non-overlapping sequences FIFO before uniform fallback, 100k-frame capacity, batch 16, length 64, one context frame, 1,024 valid-sequence warmup |
| World BPTT | 8-step truncated chunks, gradient-averaged over the full length-64 batch |
| Train ratio | 32 replayed samples per real environment step |
| State | Block-recurrent deterministic state plus 32 categorical variables |
| Categorical support | Preset-dependent classes, 1% uniform mixture |
| World losses | Feature reconstruction, reward, continuation, dynamics KL, representation KL |
| KL | Dynamics scale 1.0, representation scale 0.1, 1 free nat |
| Reward/value | 255-bin symmetric exponentially spaced two-hot distributions |
| Imagination | 15 steps from every posterior sequence state |
| Actor | Categorical REINFORCE, 1% uniform mixture, entropy scale 3e-4 |
| Critic | Imagined lambda returns plus replay-value loss and slow-value regularization |
| Discount | Learned continuation including horizon discount, horizon 333 |
| Lambda | 0.95 |
| Slow value | EMA rate 0.02 |
| Return scale | 5th/95th percentile EMA, rate 0.01, minimum scale 1 |
| Optimizer | Per-parameter AGC 0.3, RMS normalization, momentum, 4e-5, 1,000-step warmup |
| Initialization | Truncated fan-in normal; zero reward/value outputs; actor output scale 0.01 |

Meganeura implements DreamerV3's optimizer chain directly: per-parameter
adaptive gradient clipping at 0.3, RMS normalization, then momentum. Meganeura
also compiles a statically unrolled recurrent graph: unrolling all 64 steps made
even small presets impractical to build. A fresh 1M compile probe ran for more
than 11 minutes and reached roughly 1.5 GB resident memory without completing.
Kindle therefore uses 8-step truncated BPTT, accumulates and averages gradients
across all eight chunks, and retains the full 64-step replay sample and all
posterior imagination starts. The chunk length remains configurable for
targeted scaling checks. This is an implementation accommodation, not a claimed
D3-equivalent gradient path.

For the selected game, collect:

World and behavior parameters live in separate optimizer sessions because their
static graphs execute in stages. Targets are formed before either update, the
replay-value representation gradient is retained, and D3's adaptive clipping is
applied independently to each parameter leaf.

The network-size presets `1M` through `200M` mirror upstream. Kindle defaults
to `12M` for practical iteration; the pinned upstream default associated with
the published Atari-100k score artifact is `200M`. A 12M run is a labeled
scaling ablation, not an official-score reproduction. DINO's frozen parameters
are additional. On the current 29 GiB host and 16 GiB GPU, a guarded 200M
construction probe reached its 22 GiB host-memory ceiling before the training
graph reached the GPU, so that preset is not runnable without backend or memory
layout work. The 12M Pong canary constructs in 23.5 seconds, peaks near 6.25 GiB
of VRAM, and sustains 0.129 learner updates/s. A repeated published-protocol
canary with explicit RTX selection and the final D3 gradient path constructs in
22.5 seconds, uses 6.28 GiB, and sustains 0.122 learner updates/s over 104
updates. At Atari-100k's train ratio, one full 12M seed is therefore roughly 56
GPU-hours. The corresponding 1M run sustains about 0.45 updates/s and is a
14--15-hour scaling curve. Neither is an official-size substitute.

Select one primary metric that reflects the game's actual objective, such as:

- score per episode;
- level or checkpoint progress;
- survival-adjusted score;
- win rate;
- objective completion rate.

Do not construct a new dense shaping score merely because it is easier to
optimize. Auxiliary metrics may diagnose behavior, but one game-native outcome
must remain primary.

A useful normalized score is:

1. Replay drains D3's FIFO of fresh non-overlapping sequences, then samples
   contiguous sequences uniformly; every sample has one preceding context frame.
2. An inference pass samples hard posterior categoricals.
3. A training pass inserts those samples with a straight-through estimator and
   updates reconstruction, reward, continuation, and balanced KL losses.
4. Every posterior state in the sequence starts a latent imagined trajectory.
5. The current actor samples imagined actions; the frozen world optimizer
   boundary prevents behavior gradients from rewriting dynamics.
6. Reward, continuation, current value, and slow value produce lambda returns,
   score-function policy targets, and distributional value targets.
7. Actor/value optimizer updates run, the slow value network receives its EMA
   update, and inference sessions receive the new parameters.

Scheduled optimization begins after replay contains 1,024 complete sequence
starts. As in D3's runner, prefill interactions do not accumulate a learner
backlog: the first eligible scheduler call yields one update, and subsequent
calls follow the configured replay-sample ratio.

The initial diagnostics below used uniform-only sampling before D3's small
online-first queue was identified in the upstream replay implementation. They
remain useful matched implementation controls, but post-correction curves must
be labeled separately. With the default 65-frame context-plus-training span,
fresh starts are 1, 66, 131, and so on; queued starts are consumed before the
uniform sampler.

D3's replay-value loss also sends a representation gradient into the RSSM. To
retain that path with separate static learner graphs, Kindle evaluates a fixed
copy of the current critic inside the world graph. Its input gradient updates
the world representation; the critic parameters remain owned and updated once
by the behavior optimizer.

### 2.3 Minimum success gate

A 100-hour run is a success only if all of the following hold:

- the final evaluation window is better than the random baseline;
- the final evaluation window is better than the first evaluation window by a
  predeclared statistically meaningful margin;
- improvement persists at hour 100 rather than appearing only in a cherry-picked
  checkpoint;
- video inspection shows game-relevant behavior rather than exploitation of a
  scoring artifact;
- no privileged state or target-game demonstration entered the learning path.

### 2.4 Strong success gate

A strong result additionally reaches one of:

- the novice-human performance range;
- a nontrivial win or completion rate;
- a game-specific competency threshold agreed before the run.

The engineering milestone is one complete 100-hour lifetime. A scientific claim
requires fixed hyperparameters and multiple independent lifetimes, preferably
three or more.

---

## 3. Target-game contract

The first target game should be difficult enough to require learning but simple
enough that a failed run can be diagnosed.

### 3.1 Required properties

The game must be:

- locally runnable and legally automatable;
- free of public-server anti-cheat and service-policy complications;
- controlled entirely through a stable keyboard, mouse, or gamepad adapter;
- observable through RGB frames from the player's viewpoint;
- naturally episodic or restartable through ordinary game actions;
- equipped with a trustworthy game-native score, progress, or terminal signal;
- compatible with a compact initial action vocabulary;
- absent from action-conditioned world-model pretraining.

A locally controlled Doom/OpenArena-like game is a better first benchmark than
Fortnite. Fortnite is an aspirational follow-up once the learning system works;
it should not be entangled with network conditions, anti-cheat, changing game
versions, or other players in the first architectural test.

### 3.2 Observation boundary

The agent receives:

- RGB frames;
- its own previous actions;
- episode boundary flags;
- game-native scalar reward or event feedback.

It does not receive object coordinates, player velocity, map state, inventory
state, enemy state, or emulator RAM unless those quantities are also visibly
available to a human and extracted through a separately declared perception
path.

The environment adapter may use privileged instrumentation to produce the
reward and terminal flags. That instrumentation must not enter the observation
or world-model input.

### 3.3 Initial action boundary

The current Kindle baseline supports categorical actions. The first game should
therefore admit a vocabulary of roughly 12–32 meaningful controls, for example:

- no-op;
- forward, backward, and strafing;
- coarse turn or aim increments;
- jump, crouch, interact, fire, and weapon-change actions;
- a few common simultaneous key combinations.

Invalid actions should have defined benign semantics, preferably no-op. A fully
factorized keyboard-and-mouse policy is useful later, but it is not required to
answer the first 100-hour question.

---

## 4. Proposed model

```text
RGB frame stream
      │
      ▼
LeVJEPA-style block-causal video encoder E
(pretrained; frozen during the first online experiment)
      │
      ▼
compact perceptual target u_t
(global token + compressed spatial tokens)
      │
      ▼
Dreamer categorical RSSM
  h_t = recurrent deterministic state
  z_t = stochastic categorical state
  q(z_t | h_t, u_t) = posterior
  p(z_t | h_t)      = prior
      │
      ├── perceptual-latent prediction head
      ├── game reward head
      ├── continuation head
      └── posterior states used as imagination starts
                    │
                    ▼
            latent imagination
                    │
                    ▼
       categorical actor + two-hot critic
```

### 4.1 Perceptual latent

The video encoder produces a deterministic perceptual representation:

\[
u_t=E(o_{t-L:t}).
\]

- world-model parameters and optimizer moments;
- actor/value parameters and optimizer moments;
- slow value parameters;
- full configuration and pinned source/model revisions;
- fixed DINO projection seed and observation shape;
- learner/environment counters;
- return-normalizer state.

Do not rely only on a clip-level class token. Games contain small, localized,
fast-changing cues. The first practical representation should retain:

- one global summary token;
- a small grid or learned set of spatial summary tokens;
- optional dedicated HUD-region tokens if the target game makes them necessary.

To minimize migration cost, the first implementation may preserve Kindle's
existing `7×7×64` perceptual interface. The source of those tokens changes from
projected DINO patches to projected LeVJEPA tokens; the RSSM and replay shapes do
not need to change at the same time.

### 4.2 Two temporal scales

The LeVJEPA encoder and RSSM are not redundant.

- The encoder integrates short visual history: local motion, appearance changes,
  and frame-to-frame correspondence.
- The RSSM carries action-conditioned belief over longer and partially observed
  history: hidden enemies, inventory consequences, temporal commitments, and
  state not visible in the current frame.

During imagined rollouts, the video encoder is not executed. Only the RSSM prior
advances, keeping imagination substantially cheaper than visual inference.

### 4.3 Reconstruction-free world-model objective

The current same-step DINO feature decoder is replaced by a representation
prediction head. The target remains in latent space; no RGB decoder is part of
the optimization path.

- native DINOv3 ViT-S/16 with fixed checkpoint and source revisions;
- numerical golden parity against Transformers/PyTorch;
- categorical RSSM, balanced KL, two-hot heads, continuation, and feature
  reconstruction;
- sequence replay and exact terminal/truncation alignment;
- posterior-start imagination, actor/value learning, slow critic, and return
  normalization;
- held-out horizon-wise decoded-DINO prediction error against a persistence
  baseline, affine low-rank floors for the decoder's 64-channel output, and
  posterior/prior reward and representation probes;
- complete synthetic acting/learning/checkpoint GPU canary;
- minimal Rust and Python visual environment APIs.

\[
\begin{aligned}
\mathcal L_{world}={}&
\lambda_u\,d(\hat u_t,\operatorname{sg}(u_t))\\
&+\beta_{dyn}D_{KL}(\operatorname{sg}(q_t)\|p_t)\\
&+\beta_{rep}D_{KL}(q_t\|\operatorname{sg}(p_t))\\
&+\lambda_r\mathcal L_{reward}
+\lambda_c\mathcal L_{continuation}
+\lambda_v\mathcal L_{replay\ value}.
\end{aligned}
\]

Here:

- \(d\) is normalized MSE or cosine distance in perceptual latent space;
- `sg` is stop-gradient;
- the balanced KL and free-nat behavior remain DreamerV3-style;
- reward, continuation, and replay-value losses retain their current roles.

Validation snapshot (updated 2026-08-25):

- formatting, Clippy, Rust/Python tests, serialized GPU learner canaries, and
  full-checkpoint DINO parity pass on the pinned stack;
- the Gymnasium/ALE adapter discovers 104 bundled games; Pong, Private Eye,
  and Montezuma's Revenge reset to RGB observations. A small local wrapper now
  matches the pinned D3 Atari-100k protocol on the consequential points where
  Gymnasium's stock preprocessing differs: 0--30 reset no-ops, terminal-frame
  capture before the two-frame max pool, and Pillow bilinear resize. Repeated
  seeded 100-step Pong controls produce identical actions, pixels, and rewards;
- under the pinned source's `current` profile and over 100,000 decisions
  (400,000 nominal emulator frames), random Pong has
  completed-episode means -20.210, -20.189, and -20.308 across seeds zero
  through two, or -20.236 pooled over 318 episodes. Random Private Eye has
  means 47.730, -2.135, and -6.270, or 13.108 pooled over 111 episodes. The
  benchmark-aligned 350,000--400,000-frame window averages -20.128 on Pong
  and 68.13 on Private Eye across the three random seeds, using respectively
  13 and 5 completed episodes per seed. The pinned D3 score artifact was
  committed with D3's 200M default and an older environment profile (all 18
  actions, zero reset no-ops, and a 100,000-frame episode cap),
  so it is a published target rather than a matched 12M/current-protocol
  control. The runner now exposes those historical wrapper settings as
  `--atari-protocol published` while retaining the pinned source's newer
  defaults as `current`; it records the selected profile, action-space mode,
  no-op range, and episode cap in every run header. Full comparison curves use
  `published`. New matched three-seed random controls under that exact profile
  score -20.615, -20.714, and -20.538 on Pong and 80, -20, and 80 on Private
  Eye in the benchmark window, for seed means -20.623 and 46.67. The windows
  contain 13, 14, and 13 completed Pong episodes and five Private Eye episodes
  per seed, so the sparse-game estimate remains noisy. The published D3 target
  is -4.537 on Pong and 2,895.24 on Private Eye, and zero on Freeway. These
  targets average each seed's episode scores over the same recorded-frame
  window, then average five seeds. Pong and Private Eye remain the first dense
  and sparse Atari curves;
- the first published-protocol 1M Pong curve is running with the calibrated
  reconstruction scale and D3 representation gradients. At the 10,000-step
  checkpoint it had made 2,229 learner updates and averaged -20.636 over 11
  completed online episodes. A separate 5,000-step sampled-policy probe scored
  -20.2 over five episodes. On a separate fixed random-action trajectory reused
  across checkpoints, its posterior reward means for positive, zero, and
  negative targets are -0.02235, -0.02232, and -0.02234; the one-step prior is
  equally flat. Ranking is weak but above chance for the 108 negative events:
  posterior negative-vs-rest and any-reward-vs-zero ROC-AUC are 0.636 and
  0.641, while the corresponding one-step-prior values are 0.543 and 0.546.
  Only three positive events occur, so their ranking is too noisy to interpret.
  Thus the model is finite and reconstructing substantially better (first-100
  to final-100 loss 3,977.4 to 86.84), but has not learned calibrated
  reward-conditioned state at 10% of the interaction budget. The curve
  continues; this is an early marker, not an endpoint comparison;
- by the 15,000-step checkpoint, reward ordering improves materially without
  yet changing control. On the identical fixed trajectory, posterior
  reward-event-vs-zero ROC-AUC rises from 0.641 to 0.790 and the one-step prior
  rises from 0.546 to 0.765. Negative-vs-rest AUC reaches 0.792 posterior and
  0.767 prior. The posterior negative-minus-zero prediction gap grows from
  about -0.000017 to -0.000961. Separately, the sampled policy remains
  random-level at -20.6 over five episodes. This is a causal checkpoint-model
  learning curve, but not calibrated reward prediction or learned control yet;
- a held-out nearest-centroid probe over 500 GridWorld frames classifies raw
  frozen-DINO position at 85/100 and food state at 95/100, observing all 25
  and 3 classes. It also recognizes both held-out reward frames, although that
  rare two-example test is only directional evidence;
- the valid-action random control averages 21.073 rewards per 1,000 steps over
  ten independent 100,000-step seeds;
- tiny-model wiring controls remained at or below random return at both the
  default and a diagnostic 1e-3 learning rate; they stayed finite and reduced
  reconstruction loss, but did not learn reward-conditioned predictions;
- the corrected upstream 1M preset first learned at environment step 1,081
  with exactly 1,088 replay frames, then completed 10,000 interactions and
  2,230 updates. Online reward was 216 versus 217 for the matched random seed;
  its frozen greedy policy scored 0/2,000 while choosing only up/down;
- over the final 100 updates of that 1M run, rewarded and unrewarded
  predictions averaged 0.019785 and 0.019775 and policy entropy averaged
  1.3829 (uniform over four actions is 1.3863). Reconstruction fell from
  4,017.8 on the first update to 2,237.6 on the final update;
- on the same held-out trajectory, the trained 196-value adapter still
  classified position at 37/100, food state at 83/100, and both reward frames.
  The sampled 640-value posterior classified position at 4/100 and missed both
  reward frames; its deterministic and categorical components were similarly
  near chance for position. The current failure is therefore downstream of
  DINO and primarily at the RSSM posterior boundary.

The follow-up uses a dense diagnostic reward for either of the two rightmost
columns. Forced random actions give every learner the same trajectory, while
randomizing position after every action makes the next observation and reward
unpredictable from action history. This distinction matters: on the ordinary
grid, the validity mask lets a policy move globally right without recognizing
the frame. Reconstruction scale `1 / 3,136` turns the summed DINO-feature event
loss into a per-feature mean while leaving the reported raw loss comparable.
Except for the named 12M run, these controls use the 1M preset, learning rate
`1e-3`, zero warmup, train ratio 256, and 16 free nats.

| Dense diagnostic (3,000 frames, ~480 updates) | Position regime | Reconstruction scale | Final-100 reward gap | Held-out randomized dense-right accuracy |
|---|---:|---:|---:|---:|
| Reward objective only | ordinary | 0 | 0.991 | posterior 81%, adapter 75% |
| Full objective | ordinary | 1 | 0.053 | — |
| Full objective | ordinary | 0 | 0.634 | — |
| Full objective | ordinary | 1 / 3,136 | 0.427 | posterior 47%, adapter 65% |
| Full objective | randomized | 1 / 3,136 | 0.00021 | posterior 58%, adapter 63% |
| Full objective | randomized | 0 | 0.000035 | posterior 51%, adapter 71% |
| Full objective, 12M | randomized | 1 / 3,136 | 0.00415 | posterior 52%, adapter 87% |
| No reconstruction or replay-value representation gradient | randomized | 0 | 0.217 | posterior 99%, adapter 86% |
| Replay-value representation gradient stopped | randomized | 1 / 3,136 | 0.0545 | posterior 89%, adapter 80% |
| Replay-value representation gradient stopped, agent-controlled (5,000 frames) | ordinary | 1 / 3,136 | 0.999 | posterior 68%, adapter 80% |

The reward-only model transfers to unseen randomized positions, proving that
the frozen-DINO adapter, posterior straight-through path, and reward head can
carry the visual label. Apparent full-objective success on the ordinary grid
mostly follows dynamics and action history: its deterministic latent reached
83% there but only 48% after randomization. The strict 1M and 12M runs both
remain failures when all D3 representation gradients are preserved. The 12M
adapter classified exact position at 79% and the dense-right label at 87%, but
its posterior was at 52% and its own reward head predicted 0.40484 versus
0.40493 on held-out negatives and positives. Increasing capacity therefore
improves the adapter without carrying visual state across the posterior
boundary. Removing feature reconstruction also leaves the 1M posterior at
chance despite 80% exact-position accuracy at the adapter.

The replay-value auxiliary gradient can instead be stopped at the RSSM
boundary while retaining normal critic training. With reconstruction disabled,
that control succeeds after a late phase transition: its final replay
predictions are 0.049 versus 0.923, and on held-out positions its stochastic
posterior classifies dense-right at 99%. The model's own reward head reaches
97.6% accuracy with predictions of 0.077 and 0.951. Reintroducing mean-scaled
feature reconstruction preserves the result, though it delays learning: the
final 20 replay gaps average 0.264, the held-out posterior reaches 89%, and the
reward head reaches 85.2% accuracy (87.7% balanced) with a 0.540 prediction
gap. The no-reconstruction run remains the sharper wiring control; the next
sparse-food run keeps reconstruction and stops only the replay-value
representation gradient.

That matched sparse-food follow-up completed 10,000 interactions and 2,230
updates at D3's `4e-5` learning rate. It earned 192 reward versus 217 for the
matched random seed and 216 for the original full-gradient agent. Its final-100
replay reward gap was -0.000002, policy entropy was 1.3861, and its frozen
greedy policy earned 0/10,000 while alternating left/right, versus 207 for the
seed-matched random evaluation. A frozen 500-frame probe does find weak reward
ordering: ROC AUC 0.659, average precision 0.0365 at a 1.8% positive rate, and
best balanced accuracy 0.691. The prediction gap is only 0.0000096, however,
so this is an early ranking signal rather than a calibrated reward model or a
usable controller. Stopping the interfering gradient is necessary for the
dense visual control, but is not sufficient at this short D3-rate budget.

The agent-controlled dense follow-up uses the same repaired representation with
the diagnostic `1e-3` learning rate, zero warmup, and 16 free nats. It undergoes
a clear phase transition near update 240: reward separation rises from 0.002 to
0.166 and then to 0.87, while interval return recovers from 24--69/250 to
246/250. It earns 1,518/3,000 at the old comparison point, versus 1,109 for
random actions and 570 for the old full-gradient agent, and finishes at
3,472/5,000. The last 3,000 interactions earn 2,927 reward. Its frozen greedy
policy scores 1,959/2,000 versus 707 for the exact random control, establishing
that world-model, actor, and critic learning can work end to end.

This dense result is not yet robust visual navigation. On held-out positions
randomized after every action, the posterior classifies dense-right at 68% and
the adapter at 80%, but the model's own reward head has only 0.541 ROC AUC and
a 0.059 prediction gap. The ordinary task can be solved largely from action
history and the live validity mask. The next sparse-food diagnostic therefore
uses the successful `1e-3`, zero-warmup, 16-free-nat regime; it tests whether
the remaining weak reward ordering can become grounded, sequential control
rather than extending the low-rate run blindly.

That high-rate sparse run grounds the posterior reward but still fails control.
With the shared `1e-3` world/behavior rate, it earns only 43/10,000 before
falling into a deterministic loop; its frozen policy scores 0/10,000. Yet the
held-out posterior reward head is perfect (ROC AUC and average precision 1.0,
predictions 0.0020 versus 0.9983). Slowing only actor/critic optimization to
`1e-4` preserves substantially more exploration and improves online reward to
126/10,000, but remains below the 217 random control and also freezes at
0/10,000. Its final-100 entropy is 0.780, posterior reward gap is 0.998, and
imagined reward is only 0.00066. Premature actor collapse is therefore harmful
but not the remaining root cause.

An open-loop prior probe locates the next failure. From held-out posterior
states, the split-rate model's action-conditioned reward prediction has ROC AUC
0.883 and a 0.136 gap after one transition. After two transitions, ROC AUC is
0.580 and predictions collapse to 0.00020 on positives versus 0.00044 on
negatives; horizon three is at chance. The diagnostic's 16-free-nat setting
explains this: dynamics and representation KL average 16.00013 across the run
and 16.000004 over the final 100 updates. The clamped KL objective therefore
supplies essentially no prior/posterior alignment gradient. The earlier
16-free-nat runs remain valid posterior/reward wiring probes, but cannot
validate Dreamer imagination. The next matched sparse run restores D3's one
free nat while retaining the repaired representation and split optimizer rates.

That one-free-nat run restores an actual prior-learning signal, but only learns
a partial policy. It earns 130/10,000 online versus 217 for the seed-matched
valid-action random control. Its frozen greedy policy earns 96/10,000 versus
207 random by reliably collecting the first food once per episode and then
failing to continue the sequence. Over the final 100 updates, dynamics KL is
1.072 rather than pinned to the floor, posterior reward separation is 0.972,
policy entropy is 0.067, and predicted reward on actor-selected imagined states
is 0.307 per step despite real reward averaging only 0.013 per step.

Held-out prior measurements identify model exploitation rather than missing
reward recognition. On a random valid-action trajectory, the posterior reward
head reaches 0.940 ROC AUC and a 0.129 gap. The one-step prior has 0.732 ROC AUC
but a reversed calibration gap: rewarded transitions average 0.00022 prediction
versus 0.00771 for negatives. At horizon two, ROC AUC falls to 0.537 and the
gap is -0.0177; no later horizon has useful absolute calibration. The actor is
therefore finding high-reward imagined latents that held-out real trajectories
do not reach.

GridWorld's live validity mask is also a material confound. Across the random
valid-action probe, the unmasked posterior policy assigns 8.2% probability to
invalid border moves and chooses one as its argmax on 8.0% of states. More
decisively, removing the mask during frozen evaluation leaves return at 96 but
changes the 10,000 actions to 9,904 `down` and 96 `right`: after reaching the
bottom edge, the policy repeatedly selects a hidden no-op that the masked
runner had redirected into horizontal motion. An all-action held-out trajectory
also changes prior calibration substantially (one-step ROC AUC 0.814 and gap
0.036), showing that the action distribution affects the learned latent regime.
The next one-variable control therefore trains with all four actions exposed,
including defined border no-ops, against exact random controls of 188/10,000
(seed 0) and 178/10,000 (seed 1). It retains the same free nat, learning rates,
model, and update budget.

The all-action control removes the hidden collapse but does not yet beat
random. It earns 158/10,000 online and its frozen greedy policy earns
176/10,000, versus matched random returns of 188 and 178. Applying the old live
mask to that same frozen model reduces return to 99/10,000, confirming that
masking changes the recurrent action/state distribution rather than harmlessly
removing no-ops. Over the final 100 updates, KL is 1.109, posterior reward gap
is 0.932, entropy is 0.398, and imagined reward is 0.046. This is substantially
better calibrated than the masked run's 0.067 entropy and 0.307 imagined
reward, and it preserves random-level behavior across freezing.

The matched all-action probe localizes the remaining failure. Posterior reward
recognition reaches 1.0 ROC AUC and a 0.524 gap. Prior reward stays useful at
horizons one and two (ROC AUC 0.851/0.824 and gaps 0.117/0.107), but horizon
three falls to 0.679 AUC and a 0.004 gap; later horizons are inconsistent. On
31 states with an immediately rewarding action, the prior-reward argmax chooses
one 51.6% of the time, while the actor argmax does so only 35.5% of the time
and assigns 32.5% probability to rewarding actions versus a 25% uniform
baseline. Actor and prior argmaxes agree on only 41.4% of all probe states.
Both unreliable multi-step prediction and actor/value credit assignment are
therefore measurable bottlenecks. The next one-variable ablation shortens
imagination from 15 transitions to the two-transition horizon that the held-out
prior currently supports, retaining all other settings.

The first short-horizon run exposed an experimental confound: Kindle used one
RNG for live actions, live and replay posterior samples, replay selection, and
imagination. Changing imagination length therefore changed every later random
trajectory, not only behavior targets. These concerns are now independent,
deterministically seeded streams, and read-only probes use a separate seed
derived from learner/environment step. This also prevents learner throughput
from silently perturbing the environment policy.

The repeated two-step run with isolated streams still rejects the ablation. It
earns 94/10,000 online and 0/10,000 frozen; the frozen policy selects `down`
9,841 times. The final-100 posterior reward gap is 0.496 and held-out posterior
reward reaches 0.864 ROC AUC with a 0.439 gap, but one-step prior reward is at
chance (0.491 AUC with a negative gap). Actor/prior argmax agreement is 9.8%,
and the actor argmax chooses an available rewarding action only 22.6% of the
time, below the 25% uniform baseline. Short imagination weakens useful value
context and does not solve prior calibration or behavior learning here. The
next control returns to 15 steps and lowers behavior optimization from `1e-4`
to D3's `4e-5`, leaving the fast world learner unchanged so sparse reward can
ground before the actor specializes.

That lower-rate behavior control also fails. It preserves near-uniform
exploration much longer and eventually recovers to 1.327 final-100 policy
entropy, but earns only 120/10,000 online versus the 188 all-action random
control. Its frozen greedy policy selects `left` on all 10,000 steps and earns
zero. The final-100 replay reward gap is 0.806 and a held-out trajectory gives
the posterior reward head 0.9998 ROC AUC with a 0.915 gap, so sparse reward is
grounded. It does not transfer through the dynamics: one-step prior reward is
only 0.554 AUC with a -0.0080 gap, and the actor and prior argmaxes agree on
just 1.4% of states. On states with an immediately rewarding action, the actor
assigns 24.4% probability to rewarding actions versus the 25% uniform baseline.
Merely slowing actor/critic learning therefore trades premature specialization
for underfit behavior without repairing the prior. The next controlled
diagnostic must improve posterior/prior alignment while holding replay coverage
fixed; further actor-rate or imagination-length tuning is not justified by
these results.

A paired forced-random control then gives both KL settings the exact same
10,000 actions, 188 rewards, observation/action/reward trajectory, replay
sample indices, and update count.
With D3's one free nat, the final-100 replay reward gap is 0.893 and held-out
posterior reward reaches 1.0 ROC AUC with a 0.903 gap, but one-step prior reward
is only 0.581 AUC with a 0.0016 gap. The frozen actor selects `down` on every
step and earns zero. Its deterministic latent retains only 17% held-out
position accuracy versus 72% at the encoder, and the prior argmax chooses an
actually rewarding action on 22.6% of eligible states, below the 25% uniform
baseline. Exact random coverage therefore does not repair the clipped prior.

Removing the free-nat floor from both balanced-KL directions produces the
opposite tradeoff. Raw KL falls to 0.00069 over the final 100 updates and the
held-out posterior reward head degrades to 0.659 AUC with a 0.0053 gap. The
prior becomes materially more useful, however: one-step reward reaches 0.650
AUC with a positive 0.0082 gap, horizons two through five retain positive gaps,
deterministic-state position accuracy rises to 34%, and the prior argmax chooses
an actually rewarding action 51.6% of the time. Actor/prior agreement also
rises from 43.2% to 54.4%, but the actor itself chooses rewarding actions only
29.0% of the time and its frozen down/right loop still earns zero. Global
free-zero aligns by starving the posterior; global free-one preserves sparse
reward information but leaves almost all prior gradients clipped. The next
diagnostic therefore separates the two floors: zero free nats for the dynamics
loss that trains the prior, while retaining one free nat for the representation
loss that protects posterior information.

That asymmetric-floor run is a real but incomplete improvement. It uses the
same forced-random trajectory and earns the expected 188 training rewards. Its
final-100 posterior reward gap is 0.735, raw dynamics KL is 0.335,
representation KL is 1.127, policy entropy is 0.177, and imagined reward is
0.069 versus 0.019 in replay. Frozen greedy evaluation earns 59/10,000 by
selecting `right` 9,356 times and intermittently reaching only the first food.
A larger 5,000-sample held-out probe, shared by all three floor controls,
contains 93 rewarded transitions and makes the tradeoff less noisy. Posterior
and one-step-prior AUC are respectively 1.000/0.544 for global free-one,
0.740/0.741 for global free-zero, and 0.979/0.630 for asymmetric free nats.
The asymmetric prior remains above chance through horizon five but its gaps are
small (0.0055 at one step), while raw KL rises over the final training blocks.
The actor chooses an immediately rewarding action on 34.5% of eligible states
and the prior-reward argmax does so on 40.4%, both improvements over global
free-one but below a reliable controller. Continuous prior gradients therefore
retain most posterior task information and improve alignment, but are not
strong enough to track the rapidly changing posterior at the diagnostic
`1e-3` world rate. The next matched control raises only the dynamics-loss
coefficient, leaving both free-nat floors, data, rates, and update budget fixed.

A four-times dynamics coefficient overshoots that balance. Over the final 100
updates, raw KL falls to 0.0053 and representation KL stays exactly at its
one-nat floor, but posterior reward separation reaches only 0.038, policy
entropy is 0.322, and imagined reward is 0.060. Frozen evaluation earns
96/10,000 by choosing `right` 9,712 times and `down` 288 times: it reliably
collects only the first food. On the 5,000-sample held-out probe, posterior and
one-step-prior AUC converge to 0.707/0.706 with positive 0.0233/0.0227 gaps;
prior AUC remains about 0.70 through horizon five. This is well aligned but
less informative than even global free-zero's 0.740/0.741 pair. Increasing the
dynamics coefficient changes the shared recurrent-core gradient mixture even
though LaProp largely normalizes away a pure scale on prior-only parameters.
That shared pressure can therefore starve task information despite the explicit
representation free-nat floor. Scale one and four now bracket the useful
tradeoff; the next matched control tests a two-times dynamics coefficient.

The two-times coefficient does not recover the missing middle. Its final-100
raw KL is 0.116, posterior reward gap is 0.426, representation KL is 1.084,
policy entropy is 0.099, and imagined reward is 0.061. Frozen evaluation earns
zero by choosing `down` on all 10,000 steps. On the same 5,000-sample held-out
probe, posterior/one-step-prior AUC are 0.852/0.586; prior AUC stays between
0.557 and 0.574 through horizons two to five, with small and sometimes negative
reward gaps. The actor and prior argmax choose an immediately rewarding action
on 29.7% and 34.5% of eligible states. Scale two is therefore dominated by the
asymmetric scale-one run, which reaches 0.979/0.630 posterior/prior AUC and 59
frozen rewards under otherwise identical data and compute. Coefficient tuning
ends here: scale one is the world-model candidate for the next behavior test.

That next test delays only actor parameter updates until learner update 1,300,
the point where the matched scale-one run first develops a grounded replay
reward gap. The behavior critic continues learning from update zero, so the
experiment preserves early value fitting and broad imagined-state coverage
while asking whether policy specialization—not world or critic learning—happened
too soon. Actor optimizer moments continue tracking gradients while the
parameters are frozen. The gate defaults to zero and is diagnostic rather than
a baseline change.

The first execution used an older version of the gate that skipped actor
optimizer calls entirely, leaving LaProp's clock and moments cold until update
1,301. It is therefore a negative implementation control, not the intended
A/B. Entropy fell from 1.386 at the boundary to 0.875 by update 1,350 and 0.213
by update 1,500 before rebounding; its final-100 mean was 0.229. Forced-random
training earned the fixed 188 rewards, and frozen evaluation earned 78/10,000
by selecting `right` 9,360 times, still below both exact random controls. The
isolation itself worked: all 153 non-value tensors in the world checkpoint and
all non-value world report fields are byte-for-byte identical to the ungated
run. On the shared 5,000-sample probe, posterior/one-step-prior reward AUC are
0.979/0.630 with gaps 0.678/0.0055, while the actor chooses an immediately
rewarding action on only 33.6% of eligible states. Delaying specialization with
a cold optimizer does not repair weak prior calibration or actor credit. The
corrected control warms moments while holding actor parameters fixed.

The corrected control confirms that conclusion. Warming the actor optimizer
smooths the initial response to the gate, but the trajectories converge and
then undergo the same late collapse: final-100 policy entropy is 0.209 versus
0.229 for the cold-moment control, while imagined reward is 0.069 for both.
Frozen greedy evaluation earns 65/10,000 by selecting `right` 9,361 times,
below the cold control's 78. The held-out posterior/one-step-prior reward AUC
and gaps are exactly unchanged at 0.979/0.630 and 0.678/0.0055. The actor's
immediately rewarding argmax rate moves only from 33.6% to 33.9%. All 153
non-value tensors in the world checkpoint remain exact; the only 11 changed
tensors are the embedded frozen critic copy used to report replay-value loss.
Optimizer startup state therefore changes the gate transient but does not
repair prior calibration or behavior credit, and this diagnostic ends here.

Direct decoded-DINO scoring exposes a separate architectural limit in the 1M
diagnostic preset. On a 100-frame held-out trajectory, the asymmetric
scale-one checkpoint has posterior reconstruction MSE 0.03603 and one-step
prior MSE 0.03604, while copying the current observation forward scores
0.00930. The prior therefore adds almost no decoded error beyond the posterior
floor, but both are substantially worse than persistence. More importantly,
the decoder maps only four hidden patch channels into the 64-channel DINO
target. An affine PCA bound over the same frozen-DINO data gives minimum MSE
0.01825 at rank four, so this decoder cannot beat persistence even with a
perfect latent and optimizer. Rank 16 lowers the bound to 0.00337 (97.0% of
variance explained), and rank 64 removes it. Fresh configurations now use
depth 64; zero or a missing field retains the legacy preset depth so existing
checkpoints remain loadable. The next visual-model run uses the widened decoder
before decoded prediction error is interpreted as a dynamics result.

That widened, mean-scaled control removes the decoder bottleneck but does not
repair control. It follows the same 10,000 forced-random actions and 2,230
updates as the rank-four run, with effectively unchanged throughput (0.449
versus 0.448 updates/s). Reward grounding is delayed: the replay prediction gap
first exceeds 0.1 at update 1,877 rather than 1,224. Over the final 100 updates,
reconstruction MSE is 0.01351, replay reward separation is 0.397, raw KL is
0.100, and policy entropy is 0.149. Frozen evaluation earns 0/10,000 by choosing
`down` on every step.

The shared 5,000-state probe shows that the wider decoder learned a better
marginal reconstruction without a correspondingly better state model. The
trainable DINO adapter retains 66.1% exact-position accuracy, but the complete
posterior feature retains only 12.4%. Posterior and one-step-prior reward AUC
are 0.858 and 0.574, with gaps 0.477 and 0.0092. The actor and prior argmaxes
choose an immediately rewarding action on only 29.7% and 33.9% of eligible
states. Posterior reconstruction MSE is 0.01351; open-loop MSE changes only from
0.01355 at horizon one to 0.01391 at horizon fifteen. It beats persistence only
from horizon four because persistence degrades while the model remains nearly
constant, not because it tracks the trajectory. Width 64 is still the correct
capacity control, but a per-feature-mean reconstruction coefficient lets this
decoder fit observation statistics without forcing useful recurrent state. The
next exact control retains the width and raises only reconstruction scale to
0.25. The pinned D3 source divides uint8 images by 255, predicts them with a
sigmoid decoder, and sums squared error over all image event dimensions at
coefficient one. On Kindle's first matched batch, 0.25 maps the raw DINO loss
from 4,201.95 to 1,050.49, putting it in the same order as D3's summed pixel
objective instead of dividing the visual gradient by 3,136.

That one-variable control solves the integration task. Reward separation first
exceeds 0.1 at update 681 rather than 1,877 and stays near one. Over its final
100 updates, reconstruction MSE is 0.00241, reward separation is 0.9988, raw KL
is 0.778, imagined reward is 0.220, and policy entropy is 0.246. Throughput is
unchanged at 0.452 updates/s. Frozen greedy evaluation earns exactly 50 rewards
in each of 50 200-step episodes, or 2,500/10,000, while cycling all four actions;
the mean-scaled checkpoint earned zero.

The independent 5,000-state probe supports the behavioral result. The complete
posterior and deterministic state decode position at 90% and 100%, up from
12.4% and 11.5%. Posterior and one-step-prior reward AUC are both 1.0, with
prediction gaps 0.998 and 0.991. On all 354 states with an immediately rewarding
action, both policy and prior argmax select one, and the policy assigns it 96.5%
probability on average. Open-loop DINO MSE changes only from 0.00239 at horizon
one to 0.00248 at horizon fifteen while beating persistence on 78.6% to 97.3%
of samples. This is still a single-seed, forced-random-data diagnostic with a
gated actor and non-default optimizer settings, not a claim of Atari
performance.

A matched fidelity control restores both D3 gradient choices that the
diagnostic had changed: the dynamics and representation KL terms share one
free nat, and replay-value loss again sends its representation gradient into
the RSSM. It follows the same forced-random 10,000-step trajectory. Reward
separation first exceeds 0.1 at update 720, only 39 updates later than the
adapted control. Its final-100 reconstruction loss, reward gap, raw KL,
imagined reward, and policy entropy are 7.01, 0.9991, 0.554, 0.223, and 0.264.
Frozen greedy evaluation again earns exactly 2,500/10,000. The held-out probe
decodes position from the full and deterministic states at 92.7% and 95.1%;
posterior, one-step-prior, and horizon-15 prior reward AUC are all 1.0 with
about 0.999 prediction gaps. The policy selects an immediately rewarding
action on 99.7% of eligible states and assigns rewarding actions 97.1%
probability. Open-loop DINO MSE rises only from 0.00223 at horizon one to
0.00233 at horizon fifteen. The earlier full-gradient failures were therefore
conditional on an underweighted visual objective, not evidence that D3's
gradient routing must be removed. D3's routing remains the baseline and the
isolated path remains an explicit ablation.

The first unforced online-first run used that isolated ablation because it was
already in flight. It nevertheless establishes that the integration result
survives agent-controlled data: reward separation first exceeds 0.1 at update
705, total online reward is 1,179/10,000, and the last four 1,000-step intervals
earn 238, 241, 242, and 246 rewards. Frozen greedy evaluation earns the exact
2,500/10,000 ceiling. A second seed crossed the reward threshold around update
520 before this demoted multi-seed job was stopped at 5,352 interactions to
spend the device on the selected configuration.

The selected D3-gradient configuration then passes the unforced gate directly.
Reward separation first exceeds 0.1 at update 712. It earns 1,234/10,000
online, with its final five 1,000-step intervals scoring 193, 243, 244, 246,
and 244 rewards while using all four actions evenly. Over the final 100 learner
updates, reward separation is 0.9996, reconstruction loss is 6.16, raw KL is
0.638, imagined reward is 0.228, and policy entropy is 0.217. Frozen greedy
evaluation earns the exact 2,500/10,000 ceiling across 50 episodes.

Its independent 5,000-state probe decodes position from the full and
deterministic states at 95.5% and 97.0%. Posterior and one-step-prior reward AUC
are 0.99999 and 0.99939 with prediction gaps 0.954 and 0.920; horizon-15 prior
reward AUC remains 0.99791 with a 0.855 gap. The actor chooses an immediately
rewarding action on 99.2% of eligible states and assigns rewarding actions
95.5% probability. Open-loop DINO MSE rises from 0.00277 at horizon one to
0.00321 at horizon fifteen while still beating persistence on 96.2% of the
last-horizon samples. This completes the native online integration gate for
the D3 gradient path; Atari learning curves, not another GridWorld objective
ablation, are now the baseline-critical experiment.

The native result demonstrates learning in the integration environment but not
benchmark-level generality. A pinned-source follow-up also found that D3 waits
for 1,024 eligible replay items, discards prefill-time train credit, and
evaluates warmup at optimizer step zero; Kindle had started as soon as one
sequence existed and used the next step's warmup rate. Those startup semantics
are now corrected and measured end to end. The tiny network remains a wiring
preset rather than an upstream size. Online return and falling scalar losses
alone are not sufficient.

### Phase 1 — Externally rewarded baseline

1. **posterior representation prediction**, which ensures the belief state
   retains the perceptual information needed for behavior;
2. **prior/future representation prediction**, scored before the next
   observation is incorporated, which measures whether action-conditioned
   dynamics actually predict the next perceptual state.

The posterior variant is the safer initial replacement for the current decoder.
The prior/future term is then added as an ablation rather than changing the
encoder and world loss simultaneously.

### 4.4 Encoder training

For the first 100-hour experiment, the video encoder is frozen.

This gives:

- stable replay targets;
- no online representational collapse;
- no need to run SIGReg inside every Dreamer update;
- clear attribution when the world model succeeds or fails.

SIGReg and LeVJEPA's view-invariance loss belong to offline encoder pretraining.
Slow online encoder adaptation is deferred until a frozen encoder has produced a
credible learning curve.

### 4.5 Actor and critic

The first behavior learner remains Dreamer-style:

- start latent imagination from posterior replay states;
- roll the RSSM prior for a short fixed horizon;
- predict reward and continuation in imagination;
- compute lambda returns;
- train a two-hot critic and slow critic;
- train the actor with REINFORCE and entropy regularization.

The critic is retained because it is the mature control baseline and provides a
terminal value beyond the short imagined horizon. Critic-free group-relative
imagination remains a later experiment after the world model has been validated.

---

## 5. Pretraining ladder

Pretraining is separated into layers so its contribution can be measured.

### P0 — Current DINO control

Use the current frozen DINOv3 frontend and Dreamer implementation. This answers
whether the existing baseline can learn the selected game before introducing a
new video encoder.

### P1 — Generic LeVJEPA encoder

Initialize the causal video encoder from general internet-video pretraining.
Only perception is transferred. The RSSM, reward model, actor, and critic start
from scratch.

### P2 — Pixels2Play visual adaptation

Continue the LeVJEPA objective on gameplay clips from Pixels2Play while ignoring
actions. This adapts the visual representation to:

- HUDs and text overlays;
- first-person camera motion;
- rendered geometry and particle effects;
- small and fast-moving game objects;
- visual statistics unlike natural video.

The target title remains held out.

### P3 — Action-conditioned world-model pretraining

Use Pixels2Play's aligned keyboard and mouse actions to pretrain compatible
portions of the RSSM and perceptual-latent predictor on other games.

This stage trains:

- action embeddings;
- recurrent dynamics;
- prior/posterior transition structure;
- perceptual-latent prediction.

It does not train a target-game reward head, actor, or critic.

P3 requires an action representation compatible across source and target games.
For the first implementation, raw Pixels2Play controls may be quantized into the
same compact macro vocabulary used by the target adapter. A later factorized
action model can preserve the full keyboard-and-mouse space.

### P4 — Online target-game learning

At hour zero:

- load the selected perception and optional dynamics checkpoints;
- reset actor, critic, slow critic, reward head, continuation head, replay, and
  recurrent state;
- begin ordinary online Dreamer learning from the target game.

The pretraining comparison is therefore:

| Variant | Encoder | RSSM dynamics | Actor/critic |
|---|---|---|---|
| DINO control | generic DINO | random | random |
| LeVJEPA | generic video | random | random |
| P2P vision | generic + gameplay video | random | random |
| P2P world | generic + gameplay video | action-pretrained on other games | random |

---

## 6. Online interaction and learning loop

### 6.1 Real step

For every policy decision:

1. capture the current RGB frame;
2. update the causal video-encoder context;
3. produce the perceptual latent \(u_t\);
4. update the RSSM posterior from the previous belief and action;
5. sample or mode-select an action from the actor;
6. execute the action for the declared action-repeat interval;
7. observe reward, continuation, and the next frame;
8. append the transition to replay and the audit log;
9. schedule learner work according to accumulated training credit.

### 6.2 Learner update

- DreamerV3 behavioral reference:
  [danijar/dreamerv3](https://github.com/danijar/dreamerv3), revision
  `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.
- D3 runner ratio-scheduler reference:
  [danijar/elements](https://github.com/danijar/elements), revision
  `b781f900b6a3b5be3a9037fd1dbb3977ad70d219`.
- DINOv3 source reference:
  [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3),
  revision `6876159a11b4df116f30f667f8c9888617df0751`.
- Native Meganeura transcription source:
  [kvark/dinovision](https://github.com/kvark/dinovision), revision
  `dc35cdf1c7c910cdd93c5b5362846842ae469a21`.
- Meganeura graph/runtime dependency:
  [kvark/meganeura](https://github.com/kvark/meganeura), revision
  `3f2b91d95288d625a7616179604c33bba6472aaf`.
- Blade graphics dependency:
  [kvark/blade](https://github.com/kvark/blade), revision
  `95f5004fb02785a792c883a5312ca5ac37872a75` (the revision selected by
  Meganeura so the shared context types remain unified).
- DINO model:
  [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
  snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32`.

The intended trajectory is still “Dreamer inside, Kindle outside.” The inside
is now an implemented and testable baseline rather than a placeholder, while
its task-learning gate remains deliberately open.
