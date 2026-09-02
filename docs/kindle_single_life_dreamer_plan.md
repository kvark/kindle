# Kindle: 100-Hour Game Learning Plan

**Decision date:** 2026-09-01  
**Status:** revised near-term research target  
**Recommended repository path:** `docs/kindle_single_life_dreamer_plan.md`

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

The phrase must be converted into a game-specific metric before training begins.

### 2.1 Required baselines

For the selected game, collect:

1. a random or uniformly exploratory policy baseline;
2. the untrained Kindle policy at hour zero;
3. a novice-human baseline under the same scoring protocol;
4. optionally, a frozen Pixels2Play behavior-cloning policy as a practical upper
   comparison, not as Kindle initialization.

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

\[
S_{norm}=\frac{S_{agent}-S_{random}}{S_{novice}-S_{random}}.
\]

The exact threshold is fixed before the first full run. A reasonable initial
success bar is closing at least 25% of the random-to-novice gap, provided the
metric is well behaved.

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

The encoder should be causal in time: the representation at time \(t\) may use
current and previous frames but never future frames.

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

The implementation should expose two representation-prediction variants:

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

For each update:

1. sample contiguous replay sequences with burn-in/context;
2. reconstruct posterior states across the sequence;
3. update representation prediction, KL, reward, continuation, and replay-value
   losses;
4. start imagined trajectories from posterior states;
5. train actor and critic from imagined rewards and lambda returns;
6. update the slow critic;
7. synchronize inference parameters.

### 6.3 Evaluation windows

Every five hours, run a fixed evaluation window with:

- learning disabled;
- policy sampling mode fixed in advance;
- fresh natural episodes, never restored game states;
- the same observation and action path as training.

Evaluation time counts toward the 100-hour budget. Record score distributions and
video at hours 0, 5, 10, 25, 50, 75, and 100.

### 6.4 No mid-run search

A full 100-hour run uses a frozen configuration. Fatal implementation bugs may
terminate the run, but changing reward scales, learning rates, action mappings,
or model sizes produces a new experiment rather than silently continuing the
same one.

---

## 7. Reward contract

The first experiment uses game-native extrinsic feedback only.

Acceptable reward sources include:

- score deltas;
- win/loss events;
- level or checkpoint completion;
- lives lost;
- other outcomes already defined by the game.

The adapter may combine these into a documented scalar, but it should avoid
continuous hand-shaped hints such as distance to a hidden objective or privileged
enemy coordinates.

Intrinsic reward remains at zero in the primary run. If the selected game is so
sparse that no positive event is observed in the first 25 hours, that is a valid
failure of the chosen setup. A separate follow-up may add one predeclared
intrinsic signal, preferably epistemic disagreement or learning progress, but it
must not be introduced halfway through the primary run.

No target image or latent goal distance is used.

---

## 8. Replay and 100-hour storage

The current 100,000-transition replay is too short to represent a 100-hour run.
The revised experiment needs explicit lifetime storage.

### 8.1 Append-only audit archive

Store the complete interaction stream on disk:

- compressed RGB video or frame chunks;
- actions;
- rewards and boundary flags;
- episode and lifetime timestamps;
- model and policy versions;
- periodic diagnostics.

This archive is primarily for postmortem analysis and future re-encoding. It is
not necessarily sampled directly on every learner update.

### 8.2 Training replay

Use two sequence-preserving stores:

1. **Recent FIFO:** approximately 500,000 transitions, retaining the latest hours
   at full density.
2. **Lifetime reservoir:** fixed-size reservoir sampling over contiguous chunks,
   targeting another 500,000–1,000,000 transitions spread across the entire run.

A simple initial sampling mixture is 75% recent and 25% lifetime. This keeps the
agent responsive to its current state distribution while preventing early-game
situations from disappearing completely.

### 8.3 Representation storage

With a frozen encoder, replay can store perceptual latents rather than rerun the
video transformer for every learner sample.

Use packed `f16` or another measured compact format for:

- perceptual tokens;
- recurrent context when needed;
- action and boundary metadata.

Keep the raw compressed video archive because later encoder fine-tuning would
otherwise make old latent targets impossible to regenerate.

### 8.4 Checkpoint policy

Write periodic model, optimizer, normalizer, replay-index, and experiment-state
checkpoints for fault recovery and postmortem analysis.

Checkpoints must not be used to branch the environment, choose the best historical
policy, or retry a bad decision. A resumed run continues from the game's current
or next natural episode and is marked as resumed.

---

## 9. Experiment sequence

Running every variant for 100 hours immediately would waste compute. Use staged
elimination.

### Stage A — Two-hour systems smoke tests

Run every candidate configuration long enough to verify:

- finite world and behavior losses;
- stable recurrent state;
- encoder and actor meeting the control deadline;
- replay sampling correctness;
- checkpoint and log integrity;
- non-degenerate actions and posterior states.

No learning claim is made.

### Stage B — Ten-hour kill tests

Compare:

1. current DINO Dreamer control;
2. DINO plus reconstruction-free representation prediction;
3. generic LeVJEPA plus representation prediction;
4. Pixels2Play-adapted LeVJEPA plus representation prediction.

Terminate variants with clear numerical instability, frozen policies, invalid
reward plumbing, or severe throughput debt.

### Stage C — Twenty-five-hour pilots

Run the strongest two variants with fixed settings. Add action-conditioned P2P
world-model pretraining if its pipeline is ready.

The gate is not final competence. It is evidence of a directional learning curve
and a diagnosis of where remaining failure lies.

### Stage D — One 100-hour engineering run

Select the mainline architecture before starting. Complete the full run without
hyperparameter intervention and publish the entire curve, including regressions.

### Stage E — Scientific confirmation

Repeat the final configuration over at least three independent lifetimes with
identical hyperparameters and target-game initialization. Repeat the strongest
control when resources permit.

Do not report only the best seed.

---

## 10. Development milestones

### M0 — Freeze the benchmark contract

Deliver:

- selected game and immutable version;
- observation and action rates;
- action vocabulary;
- reward definition;
- random and novice-human baselines;
- success threshold;
- target-game exclusion record for pretraining;
- 100-hour run manifest format.

**Exit gate:** another person can reproduce the environment boundary and scoring
without reading Kindle internals.

### M1 — Validate the current DINO Dreamer control

Use the existing implementation unchanged except for the target-game adapter and
long-run logging.

**Exit gate:** a two-hour smoke test and ten-hour run complete with finite losses,
correct reward alignment, and bounded training debt.

### M2 — Replace feature reconstruction with latent prediction

Refactor `ObservationDecoder` into a perceptual-representation predictor while
keeping DINO and all behavior-learning code fixed.

Add diagnostics for:

- posterior representation error;
- prior one-step representation error;
- multi-step open-loop representation error;
- latent variance and collapse.

**Exit gate:** the DINO-CDP variant trains stably and is not materially worse than
the current DINO decoder on the native smoke environment.

### M3 — Introduce the LeVJEPA frontend

Add a `VisionEncoder` boundary supporting:

- the current DINO path;
- a LeVJEPA-style block-causal encoder;
- identical compact output shapes at the RSSM boundary;
- incremental temporal state or KV-cache handling;
- checkpoint provenance.

**Exit gate:** numerical inference tests pass, real-time latency is acceptable,
and the 10-hour target-game run does not exhibit perceptual or latent collapse.

### M4 — Gameplay-domain visual adaptation

Build a reproducible Pixels2Play clip pipeline for LeVJEPA continuation training,
with the target game excluded.

**Exit gate:** the adapted encoder improves held-out gameplay representation
probes or downstream 10-hour learning relative to generic-video LeVJEPA.

### M5 — Optional action-conditioned world pretraining

Pretrain compatible RSSM dynamics and representation prediction from
Pixels2Play image-action sequences without reward or policy training.

**Exit gate:** on the first hour of the held-out game, the pretrained model has
lower prior prediction error or faster reward-model adaptation than a randomly
initialized RSSM, without causing worse online control.

### M6 — Lifetime replay and run reliability

Implement:

- recent and reservoir sequence replay;
- packed latent storage;
- append-only video/action audit log;
- resumable checkpoints;
- automated periodic evaluation and video capture;
- a run dashboard or report generator.

**Exit gate:** a synthetic or game run survives at least 25 hours without memory
growth beyond the configured bound, missing logs, or unrecoverable learner drift.

### M7 — Complete the 100-hour run

**Exit gate:** the experiment reaches hour 100 and the predeclared success gate is
reported honestly, whether it passes or fails.

---

## 11. Diagnostics

### 11.1 Gameplay

- episode return and primary game metric;
- score distribution in frozen evaluation windows;
- survival time, progress, win/completion rate;
- action frequencies and action-transition entropy;
- behavior videos at fixed hours.

### 11.2 Perception and world model

- representation-prediction loss;
- prior/posterior KL and categorical entropy;
- one-, five-, and fifteen-step open-loop latent error;
- reward prediction calibration;
- continuation prediction calibration;
- perceptual-token variance and effective rank;
- zero-shot prediction error before target-game learning.

### 11.3 Actor and critic

- actor entropy and policy KL over time;
- imagined return distribution;
- critic error on realized returns;
- current versus slow-critic disagreement;
- replay-value loss;
- imagined-versus-real reward discrepancy.

### 11.4 Systems

- frame capture, encoder, actor, learner, and synchronization latency;
- 50th, 95th, and 99th percentile action latency;
- environment steps, replay steps, and imagination steps per second;
- learner debt;
- GPU memory and utilization;
- replay occupancy and sample-age distribution;
- checkpoint and archive sizes.

---

## 12. Failure interpretation

A failed 100-hour run is useful only if the failure can be localized.

| Observation | Likely problem | Next experiment |
|---|---|---|
| Perceptual targets omit visible score or critical objects | Encoder or token compression | Higher spatial resolution, intermediate tokens, or slow upper-layer adaptation |
| One-step prior error remains high | Action-conditioned dynamics | Action representation, RSSM capacity, or P2P dynamics pretraining |
| Latent prediction is good but reward prediction is poor | Reward cue absent or too sparse | Better game-native reward extraction; inspect encoder sensitivity to reward cues |
| Reward and value learn but policy stays near random | Actor optimization or action vocabulary | Verify imagined advantages, entropy, masks, and controllability |
| Imagined return rises while real play worsens | Model exploitation | Shorter imagination, stronger world training, uncertainty diagnostics |
| Early skill appears and later disappears | Replay or critic drift | Increase reservoir share; separate world-model and actor retention tests |
| P2P pretraining hurts | Domain mismatch or inherited action bias | Fall back to generic LeVJEPA or transfer only lower visual layers |
| Learner falls increasingly behind | Systems budget is invalid | Lower train ratio, compact tokens, smaller preset, or slower control rate |
| No positive reward occurs | Benchmark/reward mismatch | Report failure; run a separate intrinsic-exploration experiment |

The full 100-hour run is not the place for exploratory debugging. These failure
modes should first be exercised in the two-, ten-, and twenty-five-hour stages.

---

## 13. Explicit non-goals

The following are valuable but deferred until one game can be learned reliably:

- self-generated or fully intrinsic objectives;
- multi-game continual learning;
- indefinite lifetime memory;
- online visual-encoder plasticity;
- critic-free group-relative imagination;
- runtime planning, CEM, or MCTS;
- hierarchical options and semantic reflection;
- language-conditioned goals;
- public online multiplayer operation;
- physical robot control;
- irreversible-world safety guarantees.

The larger Kindle vision is not abandoned. It is gated on proving the smallest
complete learning loop first.

---

## 14. Required artifacts from the 100-hour experiment

A completed run must publish or retain:

- exact source revisions;
- complete model and environment configuration;
- pretraining checkpoint provenance and target-game exclusion record;
- action and reward adapter definitions;
- random and novice baselines;
- full time-series logs, not only summary metrics;
- frozen-evaluation results at fixed hours;
- behavior videos at hours 0, 5, 10, 25, 50, 75, and 100;
- final and periodic model checkpoints;
- replay/archive statistics;
- a short postmortem explaining what the agent learned and what it did not.

No best-checkpoint result should replace the hour-100 result.

---

## 15. Immediate implementation order

1. **Select and freeze the target game contract.** Architecture work without a
   fixed metric will drift.
2. **Run the current DINO Dreamer baseline for 2 and 10 hours.** Establish that
   game capture, reward alignment, replay, and actor learning are real.
3. **Refactor the world loss from `ObservationDecoder` to a latent
   representation-prediction head while keeping DINO fixed.** This isolates the
   reconstruction-free change.
4. **Add a `VisionEncoder` interface and LeVJEPA inference path with the same
   compact token shape.** Do not change RSSM and action modeling at the same
   time.
5. **Add packed long-run replay, audit video, and automated frozen evaluation.**
6. **Compare generic and Pixels2Play-adapted LeVJEPA in 10-hour kill tests.**
7. **Only then build action-conditioned P2P world-model pretraining.**
8. **Choose the final configuration before starting the 100-hour run.**

The first coding milestone is therefore not a controller, critic replacement, or
new intrinsic reward. It is a clean reconstruction-free representation target
inside the already implemented Dreamer loop, followed by a causal video frontend
that can carry useful priors into a held-out game.

---

## References

- Hafner et al., *Mastering Diverse Control Tasks through World Models*
  (DreamerV3), arXiv:2301.04104; Nature, 2025.
- Hauri and Zenke, *Dreamer-CDP: Improving Reconstruction-free World Models Via
  Continuous Deterministic Representation Prediction*, arXiv:2603.07083.
- Kuhn et al., *LeVJEPA: Efficient & Scalable Video Pretraining without the
  Heuristics*, arXiv:2608.27395.
- Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive
  Architecture from Pixels*, arXiv:2603.19312.
- Yue et al., *Pixels to Play: A Foundation Model for 3D Gameplay*,
  arXiv:2508.14295.

## Repository notes

- Current architecture: `README.md`
- Current Dreamer world loss: `kindle/src/dreamer/world.rs`
- Current broader plan path: `docs/kindle_single_life_dreamer_plan.md`
