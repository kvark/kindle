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

- one GPU for vision, world-model learning, and behavior learning;
- actor inference at the game's chosen control rate;
- asynchronous or interleaved learner updates;
- no unbounded training debt.

The actor must continue meeting its control deadline while learning is enabled.
The learner may temporarily fall behind, but the backlog must not grow
monotonically throughout the run. Environment steps, replayed sequence steps,
imagined steps, wall-clock time, and GPU utilization are all reported.

---

## 2. Operational definition of “learn how to play”

| Area | Baseline |
|---|---|
| Replay | Uniform online replay, 100k-frame capacity, batch 16, length 64, one context frame, 1,024 valid-sequence warmup |
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
even small presets impractical to build. Kindle therefore uses 8-step truncated
BPTT, accumulates and averages gradients across all eight chunks, and retains
the full 64-step replay sample and all posterior imagination starts. This is an
implementation accommodation, not a claimed D3-equivalent gradient path.

For the selected game, collect:

World and behavior parameters live in separate optimizer sessions because their
static graphs execute in stages. Targets are formed before either update, the
replay-value representation gradient is retained, and D3's adaptive clipping is
applied independently to each parameter leaf.

### 2.2 Primary score

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

1. Uniform replay samples contiguous sequences with one preceding context frame.
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

A first objective is:

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

Validation snapshot (2026-08-24):

- formatting, Clippy, Rust/Python tests, serialized GPU learner canaries, and
  full-checkpoint DINO parity pass on the pinned stack;
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

These are implementation and failure-localization results, not evidence that
the baseline learns the environment. A pinned-source follow-up also found that
D3 waits for 1,024 eligible replay items, discards prefill-time train credit,
and evaluates warmup at optimizer step zero; Kindle had started as soon as one
sequence existed and used the next step's warmup rate. Those startup semantics
are now corrected and measured end to end. The tiny network remains a wiring
preset rather than an upstream size. Online return and falling scalar losses
are not sufficient.

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
