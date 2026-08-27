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

That code pin and the released benchmark curves have different provenance.
The Atari-100K score file was bundled at upstream commit `2411f7d1` in April
2024; it predates the current agent/RSSM rewrite and the subsequent replay-
context previous-action fix. The score file is still useful historical result
evidence, but it is not an executable golden output for the May 2026 code pin.
Implementation-parity claims below refer to `e3f02248`; final score deltas label
the 2024 artifact separately, and a locally reproduced current-D3 curve remains
the strongest future matched reference.

That local reference is prepared with the standard upstream `size1m`, seed-zero,
100,000-step Pong run with batch 16x64, train ratio 256, full recurrent
backpropagation, and upstream's learned 64x64 RGB encoder. A config-only overlay
restores the released artifact's all-18-action, zero-no-op, 100,000-frame
episode wrapper and opts into deterministic environment seeding; the upstream
algorithm remains pinned at `e3f02248`. A CPU-only parameter initialization of
that exact preset and 18-action space reports 688,004 trainable parameters;
Kindle's `1M` run reports 721,888 world and 116,817 behavior parameters, or
838,705 total. The first upstream control is therefore preset-class matched,
not count matched: Kindle has 150,701 (21.9%) more trainable parameters. Keeping
the standard upstream preset avoids changing the reference architecture; an
exact-capacity control is warranted only if the result leaves capacity as a
plausible explanation. Its initial isolated Python 3.11 environment used the
requested JAX/JAXlib 0.4.33 and ALE 0.9.0 versions. A bounded CPU-only smoke
completed 1,200 Pong driver transitions and more than 100 finite learner
updates, including replay, checkpoint, policy, train, and report graph
compilation, with about 1.35 GiB observed process memory. That JAX wheel
predates the host RTX 5080: the queued CUDA run failed before taking an
environment step because its bundled CUDA 12.1 assembler does not support
compute capability 12.0, and redirecting it to the host CUDA 13.1 assembler
then exposed unsupported Blackwell code generation inside the old XLA/LLVM.
The failed startup is preserved as environment evidence, not a result. The
executable control therefore keeps the exact D3 source, config, Python 3.11,
ALE 0.9.0, and all non-JAX package versions, while upgrading only JAX, JAXlib,
and their CUDA plugin/compiler packages to 0.6.2 and NVCC 12.9.86. JAX 0.6 is
the first release line built with CUDA 12.8; an RTX smoke initializes all
688,004 parameters, compiles policy/train/report, saves a checkpoint, fills
replay, and emits finite learner losses. Its exact package freeze is pinned by
the score manifest. Upstream compares its
driver-transition counter directly with `run.steps`; importantly, that counter
includes the initial observation and each post-episode reset observation even
though those transitions execute no repeated ALE action. Its logger then
multiplies the counter by the Atari action repeat, so the smoke's final logged
step is 4,760 near its 1,200-transition limit, and the queued 100,000-transition
run reports nominal episode steps through 400,000 frames. This preserves native
D3 replay, scheduler, stopping, and logging semantics. Kindle likewise inserts
reset observations into replay but counts only executed actions in its
environment-step and frame budgets; on Pong the difference is roughly one
logical transition per completed episode and is too small to explain a learning
gap, but score comparisons disclose it rather than calling the counters exact
action parity. Kindle uses ALE 0.12.1, so the wrapper
mechanics are matched but trajectories are not byte-identical across the two
ALE revisions. A paired fixed-action trace is pixel-identical through decision
14 and first diverges at decision 15. Separate exact 100,000-action random
controls using the same three seeded action streams score -20.374 on ALE 0.9
versus -20.623 on Kindle's ALE 0.12 in the benchmark window, making the older
runtime about 0.249 Pong points easier under this control. Score summaries
therefore use runtime-specific random targets while retaining the same
historical D3 targets. The CUDA reference curve is queued only after the
immutable Kindle BPTT-8 and BPTT-64 comparisons and their diagnostics release
the RTX. The score CLI accepts that upstream format only when its pinned
manifest and post-success `RUN_COMPLETE` marker are present, then routes its
episodes through the same 350,000--400,000-frame, per-seed aggregation used for
Kindle logs.

| Area | Baseline |
|---|---|
| Replay | Fresh non-overlapping sequences FIFO before uniform fallback, 100k-frame capacity, batch 16, length 64, one context frame, 1,024 valid-sequence warmup |
| World BPTT | 8-step truncated chunks, gradient-averaged over the full length-64 batch |
| Train ratio | 32 replayed samples per real environment step by default; 256 for Atari-100K |
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
| Optimizer | Parameter-leaf AGC 0.3, RMS normalization, momentum, 4e-5, 1,000-step warmup |
| Initialization | Truncated fan-in normal; zero reward/value outputs; actor output scale 0.01 |

Meganeura implements DreamerV3's optimizer chain directly: parameter-leaf
adaptive gradient clipping at 0.3, RMS normalization, then momentum. This is
the exact convention at the pinned D3 source: `Agent._make_opt()` calls its
own `embodied.jax.opt.clip_by_agc()`, which flattens each parameter leaf before
computing the parameter and update norms. It does not call Optax's separate
rank-dependent unit-wise AGC helper. Meganeura also compiles a statically
unrolled recurrent graph. The production baseline uses 8-step truncated BPTT,
accumulates and averages gradients across all eight chunks, and retains the
full 64-step replay sample and all posterior imagination starts. The chunk
length remains configurable for targeted scaling checks. This is an
implementation accommodation, not a claimed D3-equivalent gradient path.

A fresh 1M/BPTT-16 check now narrows that accommodation further. The doubled
graph constructs successfully in 4.70 seconds, but its first learner update
panics in Blade with `descriptor pool count overflow` after peaking at only
518 MiB host memory. Blade's current upstream head is the revision pinned here.
Its Vulkan allocator grows descriptor-set pools from 65,536 directly to
1,048,576 sets, at which point the conservative inline-uniform byte budget
overflows `u32`. An isolated Blade build that doubles sub-pool capacity instead
completes the identical 1,120-step trajectory and all nine scheduled BPTT-16
learner updates with finite metrics, peaking at 517 MiB. This proves the current
limit is descriptor-pool bookkeeping rather than model-memory pressure.

The same isolated allocator fix now makes the full D3-equivalent 64-step 1M
graph practical too. It constructs in 97.7 seconds with 1.93 GiB peak host
memory and no swap. The corresponding 1,120-step Atari canary completes in
2:58 wall time, including construction, and executes all nine scheduled learner
updates with finite metrics. The measured run phase is 82.0 seconds and reports
0.110 learner updates/s including replay prefill, essentially the same short-run
rate as the patched BPTT-16 canary. Thus full recurrence is a viable controlled
1M follow-up. The allocator correction is now merged and reproducibly pinned by
Meganeura `e67ced75` to Blade `ae0f7ad1`. The already queued BPTT-64 curve stays
immutable on its pre-merge Kindle worktree (`7d5f0179`, local diff SHA-256
`a3b0da454e7483f2abf9d0eaff44ee33f3ad947293670857665bbec2e8617c8b`)
and allocator worktree (`95f5004f`, local diff SHA-256
`86a6f18432b80f469678f55a02d2b3b3dbb0e929d5246463c73995f01a46d14a`).
A source-tree comparison finds that its Kindle learner differs from the pinned
branch only by later read-only diagnostic sessions and revision constants; its
runner and extension are additionally guarded by SHA-256 hashes
`0680de555b4418fdd4a24ca7e8563fadb33070ca6a01330fa48ea11d50381b82`
and `22c8435fcff2c642fa8fcbc7e8bb9bcee8e50639616ad15a649f18d091c67f6b`.
The production default and active BPTT-8 curve remain unchanged; larger-preset
memory and the long BPTT-64 learning result remain unproven.

The prefill-amortized canary rate is not a useful long-run estimate. Its first
update arrives at 67.70 seconds after 1,086 interactions; the next eight updates
are spaced by 1.772 seconds on average, or 0.564 scheduled updates/s. BPTT-16 is
nearly identical at 1.741 seconds/update, while the active BPTT-8 curve averages
2.247 seconds/update from 80,000 onward because each logical update submits and
waits on eight chunk graphs. Holding the measured four-interaction cadence
projects roughly 12.2 hours after construction for a 100,000-interaction
BPTT-64 run on the measured AMD path. The queued RTX curve remains the
authoritative sustained measurement, but full recurrence is not assumed to be
slower merely because its differentiated graph is larger.

The conditional action-alias fallback is validated on that same graph path. A
six-action `published-minimal` 1M/BPTT-64 canary constructs in 96.4 seconds,
peaks at 1.88 GiB, and completes the identical 1,120 interactions and nine
finite learner updates. Its measured run phase is 82.0 seconds at 0.1097
updates/s. Reducing the action vocabulary therefore introduces no new backend,
memory, or throughput risk; it remains conditional on the full-action
BPTT-64 result rather than running ahead of that causal comparison.

The identical full-recurrence construction does not currently scale to the
12M preset on this host. A construction-only published-protocol Pong probe with
the patched Blade allocator, batch length 64, and BPTT length 64 reached a
guarded 12 GiB host-memory ceiling in 2.95 seconds, before emitting the run
header or taking an environment step. Raising that ceiling would consume the
headroom protecting the active run, so the controlled comparison stays at 1M.
Scaling a successful BPTT-64 result to 12M will first require graph
checkpointing/chunking, a shorter recurrence, or another memory-layout change;
the existing 12M throughput estimates below apply only to BPTT-8.

A bounded recurrence sweep locates the current 12M construction limit more
closely. BPTT-16 constructs on the integrated AMD GPU in 7.74 seconds with a
3.7 GiB host-memory peak, preserving the 64-state batch and splitting it into
four gradient-averaged chunks. BPTT-32 reaches an 8.6 GiB host-memory peak and
requests about 10.7 GiB of AMD buffer space, after which the driver rejects VM
mappings with `BO_VA (-12)` before a run header is emitted. That failure does
not unwind cleanly: the Python process remains in uninterruptible kernel sleep
with `SIGKILL` pending, retains about 10.7 GiB of GTT mappings, and prevents the
AMD render node from being enumerated. A memory-capped BPTT-16 learner retry
therefore stopped at device discovery before constructing an agent; it is not
evidence against BPTT-16. Driver recovery or a reboot is required before more
integrated-GPU checks. BPTT-16 remains the longest demonstrated 12M
construction on this device; its first learner update and sustained throughput
remain untested.

The default replay capacity is 100,000 frames instead of upstream D3's five
million. A compressed observation plus 12M-preset recurrent context is about
22.5 KiB in the current f32 representation, so the smaller default is already
roughly 2.3 GB. The current backend also trains in f32 rather than bfloat16.
Both are explicit resource accommodations to revisit with packed/f16 replay
and backend profiling.

World and behavior parameters live in separate optimizer sessions because their
static graphs execute in stages. Targets are formed before either update, the
replay-value representation gradient is retained, and D3's adaptive clipping is
applied independently to each parameter leaf.

The network-size presets `1M` through `200M` mirror upstream's width settings,
not exact parameter totals across the different observation encoders and
heads. Kindle defaults
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

Validation snapshot (updated 2026-08-27):

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
  per seed, so the sparse-game estimate remains noisy. Under the controlled
  `published-minimal` profile, three new Pong random seeds score -20.267,
  -20.786, and -20.286 over 15, 14, and 14 benchmark-window episodes, for a
  -20.446 seed mean. The score aggregator selects this profile-specific random
  target and omits the all-action D3 targets rather than presenting them as
  protocol-matched. The historical 2024 D3 artifact target
  is -4.537 on Pong and 2,895.24 on Private Eye, and zero on Freeway. These
  targets average each seed's episode scores over the same recorded-frame
  window, then average five seeds. An independent ICLR 2025 comparison also
  reports a 12M `DreamerV3XS` at -10 on Pong and 207 on Private Eye after the
  Atari-100K budget, giving the practical preset a more appropriate secondary
  target than the original 200M model. Its released benchmark metadata marks
  Dreamer as non-sticky/full-action, but the repository does not contain the XS
  run configuration or raw curves, so those rounded scores are supporting
  evidence rather than a protocol-matched acceptance gate. Pong and Private
  Eye remain the first dense and sparse Atari curves. The strict JSONL score
  aggregator reproduces the matched random means exactly, rejects the active
  curve before `run_end`, and requires complete window coverage and matching
  wrapper metadata before reporting target deltas;
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
- at the 20,000-step checkpoint, the same matched trajectory improves again:
  reward-event-vs-zero ROC-AUC reaches 0.814 posterior and 0.803 one-step
  prior, with negative-vs-rest AUC of 0.815 and 0.803. The normalized episode
  and interval records, including every action count and reward, have an
  identical SHA-256 across the 10,000-, 15,000-, and 20,000-step probes. The
  posterior negative-minus-zero prediction gap is now -0.02713 and the prior
  gap is -0.02425. However, all three rare positive targets are still predicted
  negative, so the event AUC partly reflects prediction magnitude rather than
  correct reward sign. Over the last 100 updates before the checkpoint,
  reconstruction loss averages 72.30, raw KL 2.525, replay reward MAE 0.04361,
  and policy entropy 2.8546. Online control remains random-level: 21 completed
  episodes average -20.333 with a best return of -18. A frozen sampled-policy
  probe likewise averages -20.667 over six episodes while retaining 2.871
  action entropy out of a 2.890 maximum. The separate argmax diagnostic chooses
  `UPRIGHT` for all 2,000 steps and scores -21 on both completed episodes. The
  curve therefore continued unchanged to the 25,000-step diagnostic rather
  than treating early negative-event separation as policy progress;
- at 25,000 steps, matched reward learning remains monotonic. Posterior and
  one-step-prior reward-event ROC-AUC reach 0.869 and 0.850, while their
  negative-vs-rest AUCs reach 0.870 and 0.852. Posterior negative-minus-zero
  separation grows to -0.07855 and prior separation to -0.06615. Positive
  targets now rank between zero and negative targets, but all three are still
  predicted negative, so correct reward sign remains unproven. The normalized
  trajectory hash is identical across all four 10,000--25,000-step probes.
  Final-100 reconstruction, raw KL, reward MAE, and policy entropy are 62.40,
  2.547, 0.04081, and 2.5429. In the corresponding 80,000--100,000
  emulator-frame window, Kindle averages -20.857 over seven online episodes
  versus -20.833 for pinned D3 seed zero. Kindle's repeated losses last 764
  decisions; the D3 artifact contains long stretches of the same -21 return
  every 765 decisions in this phase. Frozen sampled evaluation remains
  random-level at -20.667 over six episodes. The curve continues unchanged to
  the 30,000- and 35,000-step behavior discriminators;
- at 30,000 steps, the immutable checkpoint contains 7,229 learner updates and
  the matched trajectory hash still agrees exactly with every 10,000--25,000
  probe. Posterior and one-step-prior reward-event ROC-AUC rise again to 0.885
  and 0.877; negative-vs-rest AUC reaches 0.891 and 0.881. Posterior and prior
  negative-minus-zero prediction gaps widen to -0.14999 and -0.12180. The three
  positive events remain slightly below zero and all predictions remain
  negative, so reward-sign calibration is still not solved. Final-100
  reconstruction loss, raw KL, reward MAE, and policy entropy are 58.88, 2.421,
  0.03908, and 1.4975. In the exact 100,000--120,000 emulator-frame window,
  Kindle averages -20.5 over six online episodes versus -21.0 for pinned D3
  seed zero and -19.80 across its five seeds. Frozen sampled evaluation is
  likewise unchanged at -20.6 over five episodes, but its effective six-action
  entropy falls from 1.710 at 25,000 steps to 1.171: `RIGHTFIRE` aliases account
  for 59.9% of actions. World-model ranking is still improving while control
  remains random-level and increasingly concentrated, so the unchanged curve
  continues to the stronger 35,000-step recovery-versus-collapse discriminator;
- at 35,000 steps, the checkpoint contains 8,479 learner updates and the
  matched trajectory hash remains identical through all six probes. Posterior
  reward-event and negative-vs-rest ROC-AUC edge up to 0.891 and 0.893, while
  their one-step-prior counterparts dip slightly to 0.874 and 0.879. Posterior
  and prior negative-minus-zero gaps nevertheless widen to -0.23007 and
  -0.21156. All three matched positive targets remain predicted negative and
  below zero targets, so matched reward ranking is plateauing without correct
  sign calibration. Final-100 reconstruction loss, raw KL, reward MAE, and
  policy entropy are 56.02, 2.360, 0.03659, and 1.4545. The exact
  120,000--140,000-frame online window contains returns -18, -19, -20, and -21:
  its -19.5 mean is one point better than pinned D3 seed zero's -20.5, but below
  D3's -18.17 five-seed mean. Online effective actions also recover from the
  one-sided 30,000-step concentration to 43.6% `RIGHTFIRE` and 32.3%
  `LEFTFIRE`. Frozen sampled evaluation improves from -20.6 to -20.0 over four
  completed episodes, with 47.8% `RIGHTFIRE`, 35.7% `LEFTFIRE`, and 1.212
  effective-action entropy. On that policy's separate trajectory, five positive
  targets are finally predicted above zero targets on average and posterior
  positive-vs-negative AUC reaches 0.902, although the positive mean itself is
  still negative. This is the first modest evidence of learned Pong control,
  not a solved curve; checkpoints at 40,000 and 45,000 steps test it against
  D3's sharper next-phase improvement before the existing 50,000-step probe;
- at 40,000 steps, the checkpoint contains 9,729 learner updates and the
  matched trajectory hash remains unchanged. Posterior reward-event and
  negative-vs-rest ROC-AUC improve to 0.902 and 0.909, and their one-step-prior
  counterparts reach 0.889 and 0.894. Posterior and prior negative-minus-zero
  gaps widen again to -0.24611 and -0.22647, but the three positive targets
  remain predicted negative and below zero targets. Final-100 reconstruction
  loss, raw KL, reward MAE, and policy entropy are 53.29, 2.476, 0.03449, and
  1.2575. World-model reward discrimination is therefore still improving.
  Control is not: the exact 140,000--160,000-frame online window averages -20.0
  over five episodes, versus -18.67 for D3 seed zero and -15.13 across its five
  seeds. Online actions are almost entirely split between `RIGHTFIRE` at 44.3%
  and `LEFTFIRE` at 43.7%, with 1.111 effective-action entropy. The frozen
  endpoint policy regresses to -20.833 over six episodes, choosing those groups
  33.4% and 53.1% of the time with 1.103 entropy. Its sole positive reward ranks
  above negative and zero targets, but one event cannot establish calibration.
  The 35,000-step signal was not sustained: the reward model is viable while
  state-dependent control remains the active bottleneck. The unchanged curve
  continues through 45,000 and 50,000 steps before choosing a BPTT-length or
  capacity follow-up;
- a read-only open-loop probe on that 40,000-step checkpoint further localizes
  the control failure. Across 100 starts on a 5,000-action forced trajectory,
  posterior DINO reconstruction MSE is 0.01917. Prior MSE at horizons 1, 5, and
  15 is 0.01758, 0.02055, and 0.02344. Persistence is better at one step
  (0.01033), but degrades to 0.03105 and 0.04925, making the prior 0.662x and
  0.476x persistence at horizons 5 and 15. To reject a generic mean-prediction
  explanation, a paired unrelated-action rollout starts from the same latent
  state and reuses identical categorical random draws. Correct-action MSE is
  0.996x, 0.837x, and 0.714x that control at horizons 1, 5, and 15. All original
  measurements reproduce bit-for-bit in the paired run, and 98--100 valid
  samples remain at every horizon after excluding episode crossings. The world
  model therefore predicts action-conditioned visual change over imagination's
  full horizon; short-term precision is imperfect, but an action-blind prior is
  not the explanation for the failed endpoint policy;
- at 45,000 steps, the immutable checkpoint contains 10,979 learner updates
  and the matched trajectory hash remains identical through all eight probes.
  Posterior and one-step-prior reward-event ROC-AUC rise to 0.927 and 0.921;
  negative-vs-rest AUC reaches 0.934 and 0.926. The posterior
  negative-minus-zero prediction gap widens to -0.26386, while the prior gap
  stays near -0.22628. The prior now ranks the three positive events slightly
  above zero on average, but both means remain negative and the posterior still
  reverses that ordering. Final-100 reconstruction loss, raw KL, reward MAE,
  and policy entropy are 48.35, 2.561, 0.03307, and 1.3474. In the exact
  160,000--180,000-frame online window, returns of -20, -21, -20, -19, and -20
  average -20.0, versus -18.0 for D3 seed zero and -14.97 across its five
  seeds. Online effective-action entropy recovers to 1.293, with 35.7%
  `RIGHTFIRE`, 30.8% `LEFTFIRE`, and 27.9% `LEFT`, but the broader action mix
  does not improve return. Frozen sampled evaluation is worse: all six
  completed episodes score -21.0, five terminate in exactly 764 decisions,
  and no positive reward occurs. That policy's negative-event AUC is still
  0.968 posterior and 0.961 prior. Reward-state learning is therefore
  progressing while the actor/value path converges toward a repeatable losing
  strategy. The unchanged curve continues to the already scheduled 50,000-step
  discriminator;
- a paired behavior probe on the same 45,000-step checkpoint evaluates all 18
  candidate actions at 250 posterior states along the fixed random trajectory.
  Its one-step diagnostic target is predicted next reward plus predicted
  continuation times next critic value. The actor is directionally aligned
  with that target: mean within-state policy/value correlation is 0.426,
  pairwise ranking accuracy is 0.679, and the policy-weighted target exceeds
  the uniform-action mean by 0.01987. The greedy action is the one-step maximum
  on only 22.8% of states, but its mean gap to that maximum is just 0.01349.
  The critic's mean action span is itself only 0.06474, comprising a 0.01384
  reward span and 0.05994 next-value span. Its best raw action falls in a
  `LEFTFIRE` alias on 131/250 states, `LEFT` on 74, `RIGHTFIRE` on 28, and
  `RIGHT` on 12; the actor's greedy choices concentrate on those same effective
  groups. Thus the actor largely follows weak, alias-sensitive model values.
  On the identical first 1,000 trajectory decisions, 250-state probes at
  10,000, 25,000, 35,000, 40,000, and 45,000 training checkpoints show the
  one-step action span growing from 0.00007 through 0.01236, 0.03240, 0.04756,
  and 0.06428, while policy/value correlation changes from 0.111 through 0.246,
  0.233, 0.284, and 0.402. Alignment strengthens while game control fails.
  At 45,000 steps, mean values for the three environment-equivalent
  `LEFTFIRE` aliases alone differ by 0.02106, about one third of the entire raw
  action span. The critic is learning increasingly sharp but partly spurious
  action distinctions rather than staying flat.
  Together with the native control and source audit, this is evidence against
  a reversed policy gradient or stale actor synchronization and for improving
  temporal/value learning before changing the policy objective;
- an on-policy calibration probe makes that value failure explicit. Read-only
  value and probability queries reproduce the 45,000-step sampled trajectory
  exactly: all 18 action counts, six episode boundaries, five 764-decision
  episodes, the final 875-decision episode, and six -21 returns match the
  earlier probe bit-for-bit. Across the 4,695 states belonging to those fully
  terminated episodes, the critic predicts -3.205 on average while the realized
  horizon-discounted return is -5.906. Its optimism bias is 2.701, MAE is 3.083,
  RMSE is 3.457, and value/return correlation is only 0.045. The critic therefore
  neither calibrates the return level nor tracks which on-policy states precede
  losses, despite assigning increasingly sharp action preferences. This is the
  strongest evidence for testing full recurrent gradients before actor-loss
  changes;
- at 50,000 steps, the checkpoint contains 12,229 learner updates and the
  matched trajectory hash remains unchanged. Posterior and one-step-prior
  reward-event ROC-AUC improve to 0.943 and 0.937; negative-vs-rest AUC reaches
  0.953 and 0.943. Their negative-minus-zero gaps widen to -0.30350 and
  -0.27080, but all three positive events are again predicted below zero
  events. Final-100 reconstruction loss, raw KL, reward MAE, and policy entropy
  are 44.39, 2.613, 0.03175, and 1.0488. The exact
  180,000--200,000-frame online window averages -20.667 over returns -21, -20,
  -20, -21, -21, and -21, versus -16.25 for D3 seed zero and -13.42 across its
  five seeds. Online effective-action entropy is 1.266. Frozen sampled
  evaluation then scores -21.0 on six episodes that all terminate in exactly
  764 decisions; 64.0% of actions are `LEFT` aliases and 33.7% are `LEFTFIRE`,
  for only 0.767 effective-action entropy. No positive reward occurs, while
  negative-event AUC is still 0.977 posterior and 0.971 prior. This definitive
  50,000-step control failure selects full 64-step recurrent gradients as the
  next causal run. The current curve remains immutable and continues through
  its existing 75,000- and 100,000-step gates to measure whether the collapse
  eventually recovers;
- paired behavior and value diagnostics at the same 50,000-step checkpoint
  show that the collapse is increasingly critic-led. Across 250 states on the
  fixed forced-action trajectory, actor/one-step-value correlation rises from
  0.426 at 45,000 steps to 0.516, pairwise ranking accuracy rises from 0.679 to
  0.717, and greedy agreement with the one-step maximum rises from 22.8% to
  43.6%, even as policy entropy falls from 1.328 to 1.048. The actor assigns
  89.0% mean probability to the raw `DOWNLEFT` and `DOWNLEFTFIRE` aliases,
  whose greedy choices cover 247/250 states. The one-step action span remains
  small at 0.0621. On the separate sampled-policy trajectory, the calibration
  probe reproduces all 18 action counts and six -21 episodes of exactly 764
  decisions from the frozen endpoint evaluation. Across their 4,584 states,
  predicted values average -3.403 with standard deviation 0.132, whereas
  realized horizon-discounted returns average -5.949 with standard deviation
  2.222. Bias is +2.546, MAE 3.009, RMSE 3.383, and value/return correlation
  is -0.015. The actor is therefore becoming more faithful to a critic that
  has almost no useful temporal ranking. An extended probe preserves that
  trajectory and every prior metric bit-for-bit while also comparing each
  value with its realized one-step Bellman target. Those targets correlate
  0.634 with the critic; one-step bias is only +0.0139 and MAE is 0.0576, but
  that small optimism compounds over the 333-step horizon. On the same policy
  trajectory, actual reward averages -0.0274 while the one-step prior predicts
  -0.01955, so underestimating loss-event magnitude explains a substantial
  part of the long-horizon value bias. Matching 5,000-step probes at 10,000,
  25,000, 35,000, 40,000, 45,000, and 50,000 checkpoints preserve every
  available prior evaluation trajectory exactly. Their Bellman target
  correlations are -0.011, 0.241, 0.386, 0.357, 0.561, and 0.634, while
  value/realized-return correlations remain 0.017, 0.089, 0.015,
  -0.043, 0.045, and -0.015. Over the same sequence, policy entropy falls from
  2.890 to 0.956 and critic standard deviation grows from 0.00002 to 0.132.
  The critic therefore learns increasingly varied, locally consistent values
  without learning useful episodic ordering, and the actor specializes around
  them. Full recurrence remains the first controlled test, with reward
  calibration and capacity now explicit follow-up discriminators;
- through 60,000 steps, all 13 completed episodes after the 50,000-step
  checkpoint score -21 and terminate in exactly 764 decisions. Kindle averages
  -21.0 over both the 200,000--220,000-frame window (seven episodes) and the
  220,000--240,000-frame window (six episodes). Pinned D3 seed zero scores
  -14.5 and -17.25 in those windows, while its five-seed means are -12.10 and
  -12.92. A policy-distribution phase change has started but has not repaired
  behavior: final-100 learner entropy falls to 0.695 at 55,000 steps and
  rebounds to 1.183 at 60,000. In the final 1,000 interactions, effective
  action entropy is 1.191, split mainly across `LEFTFIRE` at 44.6%, `LEFT` at
  28.1%, and `RIGHTFIRE` at 24.1%. The broader action mix still loses every
  point, so the immutable curve continues to its 75,000-step evaluation rather
  than treating entropy recovery as control recovery;
- the atomic 75,000-step checkpoint is now archived. The six episodes ending
  from 70,000 through 75,000 interactions average -20.833 (range -21 to -20),
  and the twelve ending from 65,000 through 75,000 average -20.5 (range -21
  to -19). Over the complete 70,000--75,000 learner interval,
  reconstruction loss averages 36.44, raw KL 2.504, reward loss 0.0580,
  replay-value loss 1.434, policy entropy 1.375, and imagined return -3.834.
  The model losses continue to improve without an online control recovery;
  matched-random, sampled-policy, behavior, and value probes remain queued
  after the immutable 100,000-step endpoint so they do not contend with
  training for the RTX;
- the atomic 80,000-step checkpoint contains 19,729 learner updates. The six
  episodes ending from 75,000 through 80,000 interactions score -21, -20, -20,
  -19, -21, and -20, for a -20.167 mean. Across the 1,250 learner updates in
  that interval, reconstruction loss averages 35.287, raw KL 2.617, reward loss
  0.0575, replay-value loss 1.460, policy entropy 1.518, and imagined return
  -3.890. The 5,000 live actions have effective six-action entropy 1.367 and
  remain concentrated on `RIGHTFIRE` (39.7%), `LEFTFIRE` (28.9%), and `LEFT`
  (20.9%). This is another random-level late marker, not a benchmark score:
  the 350,000-frame score window starts at interaction 87,500;
- the atomic 85,000-step checkpoint contains 20,979 learner updates. The six
  episodes ending from 80,000 through 85,000 interactions score -19, -21, -20,
  -21, -21, and -21, averaging -20.5. Across the interval's 1,250 learner
  updates, reconstruction loss averages 33.798, raw KL 2.549, reward loss
  0.0536, replay-value loss 1.470, policy entropy 1.532, and imagined return
  -3.946. Effective action entropy is 1.414 over the six Pong transition
  groups; `LEFTFIRE` now receives 38.8%, `RIGHTFIRE` 24.3%, `RIGHT` 19.2%, and
  `LEFT` 15.3% of actions. The changing mix still loses essentially every
  point. At 340,000 emulator frames this is the final atomic checkpoint before
  the protocol score window, not evidence from inside that window;
- the atomic 90,000-step checkpoint contains 22,229 learner updates. The four
  episodes ending from 85,000 through 90,000 interactions score -21, -19, -18,
  and -20, averaging -19.5. Across the interval's 1,250 learner updates,
  reconstruction loss averages 32.472, raw KL 2.641, reward loss 0.0526,
  replay-value loss 1.478, policy entropy 1.337, and imagined return -3.909.
  Effective action entropy falls to 1.284: `RIGHT` receives 43.6% and
  `LEFTFIRE` 35.4% of the 5,000 actions, followed by `LEFT` at 10.5% and
  `RIGHTFIRE` at 7.2%. The interval sustains 1.772 environment decisions/s.
  The first three episode endpoints in the 350,000--400,000-frame score window
  are -18, -20, and -20 at frames 351,696, 356,492, and 360,248, for a
  provisional -19.333 mean. This incomplete-window value is useful live
  evidence but is not a protocol score; strict aggregation still waits for the
  100,000-step `run_end`;
- at environment step 94,254 and learner step 23,293, a system-wide OOM killed
  the BPTT-8 process. Kernel evidence shows the pressure was external to the
  experiment: a simultaneous container/Xorg burst filled all 8 GiB of swap,
  while Kindle's unit peaked at 2.1 GiB. The six uninterrupted score-window
  endpoints available before the kill are -18, -20, -20, -17, -15, and -19,
  averaging -18.167. They remain diagnostic only. The exact crash log is
  preserved separately with SHA-256
  `e87fa98f2642515050b3de07e1b7e45eee64f19611578202befee82e6d3bbb5d`;
  no completion event was fabricated;
- recovery resumes the real atomic 90,000-step checkpoint at learner step
  22,229 and appends an explicit checkpoint segment. Strict aggregation
  requires that the nested start preserve the environment, seed, protocol,
  runtime, DINO identity, parameter counts, and full agent config, and that its
  environment and learner counters match a prior checkpoint event. It closes
  the interrupted segment at 360,000 frames and discards the superseded
  post-checkpoint crash tail before combining episode endpoints, exposing the
  recovery count in the score summary. Because Kindle checkpoints do not yet
  persist replay, the resumed learner also refills 1,088 frames, providing
  1,024 eligible length-64 sequence starts, before updating at absolute
  environment step 91,086. The final result is therefore labeled checkpoint-
  recovered rather than uninterrupted; a winning configuration still requires
  an independent clean replication;
- the recovered atomic 95,000-step checkpoint contains learner step 23,208,
  only 979 updates after 90,000 because replay refill intentionally earns no
  deferred training credit. Its four resumed episodes score -20, -20, -20,
  and -19, averaging -19.75. Combining them with the two retained pre-recovery
  score-window endpoints gives a provisional recovery-aware mean of -19.5 over
  six episodes. Across the 979 resumed updates, reconstruction loss averages
  31.918, raw KL 2.614, reward loss 0.0482, replay-value loss 1.557, policy
  entropy 1.308, and imagined return -4.062. Effective action entropy is 1.286:
  `RIGHT` receives 41.1%, `LEFTFIRE` 34.0%, and `LEFT` 17.2% of the 5,000
  actions. The recovery segment has not reproduced the superseded crash tail's
  -17 and -15 episodes, another reason not to present either partial trajectory
  as the final score;
- the checkpoint-recovered BPTT-8 curve now has a real 100,000-step `run_end`.
  Strict recovery-aware aggregation retains the -18 and -20 episode endpoints
  before the 90,000-step checkpoint and combines them with the eight resumed
  endpoints, producing -18.9 across ten completed score-window episodes. This
  is 1.723 points above the ALE 0.12 matched-random target (-20.623), but 8.9
  points below the reported 12M D3 target (-10) and 14.363 below the historical
  200M artifact (-4.537). The final 5,000 interactions improve to -18.0 across
  four episodes (-19, -18, -17, and -18). Their 1,250 learner updates average
  reconstruction loss 31.612, raw KL 2.816, reward loss 0.0500, replay-value
  loss 1.626, policy entropy 0.977, and imagined return -3.588. Effective
  action entropy falls to 1.005, with `RIGHT` at 58.5% and `LEFTFIRE` at 30.5%.
  Across the full resumed 10,000 interactions, 2,229 updates average
  reconstruction loss 31.746, raw KL 2.727, reward loss 0.0492, replay-value
  loss 1.596, policy entropy 1.123, and imagined return -3.796. The missing 271
  updates relative to an uninterrupted interval are exactly the consequence of
  replay refill without deferred scheduler credit. The above-random endpoint
  is a real late-learning signal, but it is neither a D3-baseline pass nor a
  clean replication; the immutable BPTT-64 comparison and endpoint probes
  determine the next intervention;
- paired read-only endpoint probes confirm that the late improvement is visible
  off the training trajectory. The seed-100 matched-random stream is identical
  at both checkpoints and scores -20.4. On that same environment stream, the
  frozen sampled policy changes from -20.5 across six completed episodes at
  75,000 steps to -18.667 across three at 100,000, with a further -13 partial
  game still open after 1,176 decisions. The latter's smaller episode count
  makes this directional evidence rather than another benchmark score. Its
  critic is materially less wrong about realized discounted returns: value
  bias falls from +1.865 to +0.510, MAE from 2.501 to 1.277, and statewise
  value/return correlation rises from 0.014 to 0.204. One-step Bellman MAE
  improves from 0.0743 to 0.0657 and target correlation from 0.746 to 0.761.
  Actor alignment does not improve uniformly: probability on the model's best
  one-step action rises from 6.5% to 10.3% and greedy agreement from 3.2% to
  10.0%, but policy/value correlation falls from 0.294 to 0.204 and gain over
  uniform falls from 0.0133 to 0.0086. Thus late BPTT-8 learning improves
  temporal value calibration and control while leaving the 18-action actor
  only weakly aligned with small, alias-sensitive one-step differences. The
  full-recurrence control was released only after these probes completed;
- the first BPTT-64 launch exposed a runtime-protection mistake before its
  first checkpoint: it inherited `OOMScoreAdjust=200`, the same preference that
  made the kernel select BPTT-8 during unrelated host pressure. Its incomplete
  1,542-step/115-update startup log is preserved with SHA-256
  `b4dda8957203ddae4d72b371e4235c03a58216b0938244edf21764ee5a24f8ab`.
  With no checkpoint to migrate, the identical source-pinned command restarted
  from seed zero under a guarded launcher whose SHA-256 is
  `e40d4764fb60e68b04e1364ab8a6d7ffae77695da320b9483d5c5030022b74aa`.
  Removing timing-only fields makes both JSON streams bit-identical through
  environment step 1,138 (normalized SHA-256
  `e1cd30a50a0197e8611770e88252f16f306b05efb4fe13036e53735c30887570`),
  so the restart changes no trajectory or optimization input. The user manager
  imposes an effective process floor of +100 despite the unit's requested zero,
  still improving on +200; the waiting diagnostic and upstream workers now use
  the same protection. The protected restart constructs in 97.25 seconds,
  its first 50 updates are finite and average 2.248 seconds/update (1.780
  environment decisions/s), and it peaks initially at 1.96 GiB host and 3,267
  MiB device memory without unit swap;
- that protection does not make concurrent full-BPTT and eight-environment
  Quake collection safe while all 8 GiB of host swap remains occupied. The
  separate `mind-games` controller again grew to about 2.6 GiB, each of its
  eight `vkquake` processes to about 0.55 GiB plus Xorg, and host availability
  fell from 6.4 to 4.5 GiB while Kindle was active. Kindle therefore yielded
  voluntarily before the first checkpoint at environment step 2,018 and
  learner step 234; that incomplete log is preserved with SHA-256
  `4817b70c80a136a752f3608fdb1899b25a68928fb7732e7614330d9bddc9aee1`.
  A later clean restart similarly yielded at step 2,022/update 235 when a
  diagnostic repeat appeared, preserving SHA-256
  `8bd4de43e19364595d483c5b054ead9927126a35e3fb534a05a89de871a11a28`.
  All three stopped starts are timing-normalized and bit-identical through the
  first start's complete step 1,542 prefix (SHA-256
  `1e356a31b979902a58ccf7cd9485398d39682897fd88bc21a1b411a0317734ad`).
  None is a result or recovery segment: no checkpoint existed, and the
  authoritative curve restarts from seed zero. A narrow contention guard
  (SHA-256
  `02c958d42eed623d88d3c9ab7ece3ddd157031793ac0f67ba19d759d6bb3cf85`)
  watches only for that known external controller, freezes Kindle's dependent
  workers, and stops only Kindle; it never signals or changes `mind-games`.
  A one-shot resume worker (SHA-256
  `c1bcb053a4084f66ae953fca5214402746e59e6cb785d7bf4a4c2daacb1a8335`)
  now requires ten uninterrupted quiet minutes before relaunching the exact
  pinned command, thawing the queue, and arming the guard again;
- the guard later yielded once more at environment step 1,226/update 36 when
  the external controller reappeared. That pre-checkpoint log is preserved
  with SHA-256
  `739da93e6fbf861b2da47e7c6298e227f769e8eabf257d16a05c25edcb6490a7`.
  An hourly coordinator (SHA-256
  `97bda29acd12377ffc0b72fdeb73778d0b2b86305511f57967d7aa331869dddc`)
  closes the one-shot worker's stale edge case: it observes but never signals
  the external workload, preserves an inactive pre-checkpoint output, and
  rearms the same ten-minute quiet-window worker. After the competing goals
  were paused, it released a fourth seed-zero start. Removing only construction
  and event timing fields makes that authoritative start and the preserved
  1,226-step prefix bit-identical (SHA-256
  `46d817baba115db4f33afcf5b014274011a283ffb12a566b75fe320ad64c2dd8`).
  The resulting curve is uninterrupted through its archived 75,000-step
  checkpoint, which contains learner step 18,479. Over the final 100 updates,
  full BPTT-64 versus matched BPTT-8 has reconstruction loss 30.966 versus
  35.821, reward loss 0.03613 versus 0.05600, raw KL 2.158 versus 2.514,
  replay-value loss 1.398 versus 1.452, and policy entropy 1.754 versus 1.440.
  Effective six-action entropy over the final 5,000 interactions is likewise
  higher at 1.614 versus 1.252. This improved model fit and avoided the
  truncated run's sharper action collapse, but has not yet improved control
  convincingly: its five completed episodes average -20.4 versus -20.833 over
  six for BPTT-8, and its last-25,000-step mean is -20.607 versus -20.742.
  The protocol score window does not start until interaction 87,500, so the
  immutable run continues to 100,000 and the queued frozen-policy probes;
- the same uninterrupted full-BPTT curve now has an archived 85,000-step
  checkpoint at learner step 20,979, the final atomic checkpoint before the
  score window. Over interactions 80,000--85,000, its four completed episodes
  score -18, -17, -16, and -17, averaging -17.0, while matched BPTT-8 averages
  -20.5 across six episodes. The interval reward is -51 versus -122. Across
  the identical 1,250 updates, full BPTT-64 versus BPTT-8 averages
  reconstruction loss 29.626 versus 33.798, reward loss 0.03512 versus
  0.05356, raw KL 2.269 versus 2.549, replay-value loss 1.517 versus 1.470,
  and policy entropy 0.789 versus 1.532. Its effective Pong actions are 28.0%
  `NOOP`, 34.5% `LEFT`, and 29.8% `RIGHTFIRE`, with the remaining three groups
  totaling 7.7%. Effective-action entropy is 1.349 versus 1.414, but the
  conditional entropy spent among environment-equivalent aliases is only
  0.293 versus BPTT-8's 0.772. Thus the late improvement is accompanied by
  more consistent representatives for redundant actions rather than alias
  confusion. This is strong pre-window evidence that full recurrence changes
  control, but it remains diagnostic until the uninterrupted 100,000-step
  `run_end` closes the 350,000--400,000-frame score window;
- its atomic 90,000-step checkpoint is archived at learner step 22,229. Over
  interactions 85,000--90,000, full BPTT-64 completes two episodes at -17 and
  -18 (mean -17.5 and mean length 2,131), while matched BPTT-8 completes four
  at -21, -19, -18, and -20 (mean -19.5 and mean length 1,091). Total reward
  is -41 versus -91, and the five consecutive 1,000-step reward deltas are
  +14, +8, +10, +8, and +10, so the advantage is distributed across the
  interval rather than coming from one rally. Across the identical 1,250
  updates, full BPTT-64 versus BPTT-8 averages reconstruction loss 29.041
  versus 32.472, reward loss 0.03567 versus 0.05257, raw KL 2.358 versus
  2.641, replay-value loss 1.521 versus 1.478, policy entropy 0.599 versus
  1.337, and imagined return -3.061 versus -3.909. Full BPTT-64 concentrates
  its effective actions on `LEFT` (43.5%), `RIGHTFIRE` (39.5%), and `NOOP`
  (11.6%); its effective-action entropy is 1.185 versus 1.284, and its
  within-equivalent-group entropy is only 0.207 versus 0.700. The first
  episode endpoint in the protocol score window is -18 at emulator frame
  356,996, giving a provisional one-episode mean of -18.0. This remains live
  evidence, not a score: strict aggregation still requires complete coverage
  through frame 400,000 and a valid 100,000-step `run_end`;
- the uninterrupted curve's atomic 95,000-step checkpoint is archived at
  learner step 23,479. Over interactions 90,000--95,000, full BPTT-64
  completes two -18 episodes (mean length 1,981.5) and collects -43 reward.
  The recovery-aware BPTT-8 segment completes -20, -20, -20, and -19 episodes
  (mean -19.75 and mean length 978) and collects -96 reward. Full BPTT-64 wins
  all five 1,000-step reward slices by +16, +7, +12, +9, and +9. This is not
  an update-count-matched comparison: BPTT-8 deliberately performs only 979
  updates after refilling replay, versus the uninterrupted run's 1,250.
  Across those respective updates, full BPTT-64 versus recovered BPTT-8
  averages reconstruction loss 28.448 versus 31.918, reward loss 0.03557
  versus 0.0482, raw KL 2.414 versus 2.614, replay-value loss 1.592 versus
  1.557, policy entropy 0.546 versus 1.308, and imagined return -2.820 versus
  -4.062. Its within-equivalent-group action entropy is 0.152 versus 0.758,
  further weakening the six-action fallback. The first three score-window
  episode endpoints are now all -18, at emulator frames 356,996, 365,896, and
  372,848, for a provisional -18.0 mean. Strict scoring remains withheld until
  the uninterrupted 100,000-step completion event;
- the full-BPTT curve now has one uninterrupted 100,000-step `run_end`, one
  start, zero recoveries, 24,729 learner updates, and all 20 atomic snapshots.
  Its five score-window episode endpoints are -18, -18, -18, -19, and -16 at
  emulator frames 356,996, 365,896, 372,848, 381,484, and 391,924. The strict
  score is therefore -17.8. The unfinished episode at the 400,000-frame cutoff
  had reached -10 over 2,019 decisions and is correctly excluded. This beats
  Kindle ALE 0.12's matched-random mean by 2.823 points and the recovered
  BPTT-8 curve by 1.1 points, but remains 7.8 points below the reported 12M D3
  target and 13.263 below the historical 200M artifact. The final 5,000
  interactions are a fully update-matched late comparison: full BPTT-64 versus
  BPTT-8 completes -19 and -16 episodes (mean -17.5 and mean length 2,384.5)
  versus -19, -18, -17, and -18 (mean -18.0 and mean length 1,280.5), collects
  -32 versus -65 reward, and wins all five 1,000-step slices by +5, +9, +4,
  +9, and +6. Across 1,250 updates each, it averages reconstruction loss
  27.918 versus 31.612, reward loss 0.03488 versus 0.0500, raw KL 2.443 versus
  2.816, replay-value loss 1.640 versus 1.626, policy entropy 0.512 versus
  0.977, and imagined return -2.690 versus -3.588. Its effective-action
  entropy is 1.158 versus 1.005, while within-equivalent-group entropy falls
  to 0.102 versus 0.660. Thus full recurrence yields a clean, distributed
  improvement in model fit and control without evidence that action aliases
  are the remaining bottleneck. The final log SHA-256 is
  `5aef1d954be438f81084caa9bb8edb4544a0bf12e5733de11c0c1b3318cdaf4c`;
  the score artifact records the same source hash. Frozen-policy and value
  probes now run before the pinned upstream size1m D3 control;
- the phase comparison now puts Kindle behind the pinned D3 Pong artifact,
  rather than merely matching its random early regime. D3's five-seed episode
  means in 20,000-frame windows ending at 20,000, 40,000, 60,000, 80,000,
  100,000, 120,000, 140,000, 160,000, and 180,000 emulator frames are -20.40,
  -20.37, -20.93, -20.80, -20.63, -19.80, -18.17, -15.13, and -14.97
  respectively.
  These phase markers are diagnostic rather than size-matched acceptance gates:
  the active Kindle run is 1M with eight-step truncated BPTT and frozen DINO,
  while the published artifact uses the default D3 model and full recurrence.
  The 50,000-step diagnostics select BPTT length before capacity as the first
  controlled follow-up. That full-recurrence result is interpreted only beside
  the completed BPTT-8 curve and the queued, standard-size1m current-D3 run:
  a BPTT-64 improvement is replicated before scaling; a successful upstream
  1M run with a failed Kindle BPTT-64 run selects the frozen-vision interface
  as the next causal axis; failure of both 1M runs selects capacity rather than
  an encoder or policy change. The six-action `published-minimal` control is
  justified only if full recurrence produces useful temporal values while the
  all-action critic remains dominated by environment-equivalent aliases. It
  keeps the published profile's zero reset no-ops and 100,000-frame episode cap
  while changing only that redundant vocabulary;
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

- native visual GridWorld as an integration test;
- one locally reproduced pinned-upstream D3 curve at a resource-matched preset
  and wrapper protocol;
- at least one dense-reward Atari game;
- at least one sparse-reward Atari game;
- a small visual persistent world with endogenous rather than harness resets.

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
- Size-matched Atari comparison:
  [Drama ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/bc968adbdff4a2551649d464b83f264a-Paper-Conference.pdf)
  and [released code](https://github.com/realwenlongwang/Drama), revision
  `a50bd54c34e77d1d13e988a031733a47817098e2`.
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
  `e67ced7568ae051c0f1d9f20d67370d5019d2b58`.
- Blade graphics dependency:
  [kvark/blade](https://github.com/kvark/blade), revision
  `ae0f7ad1f05443bea121eb514fab2fc0b867a662` (the revision selected by
  Meganeura so the shared context types remain unified and full-length world
  BPTT does not overflow descriptor-pool growth).
- DINO model:
  [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
  snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32`.

The intended trajectory is still “Dreamer inside, Kindle outside.” The inside
is now an implemented and testable baseline rather than a placeholder, while
its task-learning gate remains deliberately open.
