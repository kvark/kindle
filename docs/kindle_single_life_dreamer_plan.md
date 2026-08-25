# Kindle: Dreamer baseline and single-life research plan

**Decision date:** 2026-08-23
**Status:** baseline implemented; environment-learning validation in progress

## Executive decision

Kindle now starts from DreamerV3 rather than from its previous collection of
policy, planning, credit, option, and reward experiments.

The immediate objective is narrow:

> Reproduce the DreamerV3 online data regime and learning algorithm on the
> Meganeura/Blade stack, using a frozen DINOv3 ViT-S/16 visual encoder.

This is intentionally a hybrid, not a literal copy:

- DreamerV3 supplies replay, recurrent state, world-model losses, imagination,
  actor/critic learning, and scaling presets.
- DINOv3 supplies a generic visual prior so the first experiment does not also
  need to learn vision from scratch.
- Kindle retains explicit seams for intrinsic reward and game adapters.
- Dreamer 4 remains a scaling reference, not the baseline.

The earlier Kindle architecture is not maintained side by side. Its useful ideas
remain in git history and should return only through controlled ablations against
the Dreamer baseline.

## 1. Research boundary

### 1.1 What “single life” means

The long-term project assumes one factual stream of external consequences:

- no cloning, rewinding, or restoring the external environment for training;
- game death/respawn is an ordinary transition when the game implements it;
- actual memories may be retained and replayed;
- counterfactual branches may be generated inside the learned world model;
- independent lifetimes remain necessary for scientific evaluation.

The current baseline supports this data model but does not yet solve lifelong
memory, safe policy deployment, self-directed goals, or irreversible exploration.

### 1.2 Baseline before novelty

The first scientific control is ordinary externally rewarded Dreamer. Intrinsic
reward channels exist, but their default scale is zero. Every Kindle extension
must answer a concrete ablation:

1. Does standard Dreamer learn the task?
2. Does the extension improve interaction efficiency, robustness, transfer, or
   retention at matched compute?
3. Which added mechanism caused the improvement?

This ordering is the main lesson of the pivot.

## 2. Why DreamerV3 rather than Dreamer 4

“Latest” and “best fit” are different questions.

DreamerV3 is the closer fit for Kindle's first implementation because it is an
online model-based reinforcement-learning agent:

- it trains continually from an online replay stream;
- its recurrent stochastic state handles partial observability;
- it learns reward and continuation together with dynamics;
- its actor and critic learn from short latent imagined trajectories;
- it has established small-to-large network presets and robust normalization.

Dreamer 4 explores a different axis: large offline video world models,
task-conditioned finetuning, behavioral cloning, and policy learning inside a
mostly fixed model. Its results are important evidence for scalable imagination,
but adopting that stack would add offline data, tokenizer/world-model
pretraining, and much larger compute to the first Kindle experiment.

Dreamer 4 becomes relevant later if the recurrent baseline hits a clear scaling
limit. VidTok is similarly deferred until video tokenization is itself the
experiment.

## 3. Visual perception decision

### 3.1 Frozen DINOv3 ViT-S/16

The baseline uses `facebook/dinov3-vits16-pretrain-lvd1689m`:

- 224×224 RGB input;
- 16×16 patches, producing a 14×14 patch grid;
- 384 channels;
- 12 transformer layers, 6 attention heads, and 4 register tokens;
- frozen inference in a session separate from every optimizer.

CLS and register tokens participate in faithful DINO inference but are removed
at the Dreamer boundary. The RSSM consumes dense spatial patch tokens; it owns
time and recurrent belief.

Arbitrary game resolutions are resized with deterministic aspect-preserving
letterboxing. Padding uses the ImageNet mean so it becomes approximately zero
after normalization.

### 3.2 Replay representation

Storing every 14×14×384 output would make online replay unnecessarily large.
The baseline therefore applies two frozen operations:

1. a seeded Rademacher projection from 384 to 64 channels;
2. fixed 2×2 average pooling from 14×14 to 7×7.

One replay observation is 7×7×64 = 3,136 floats. The seed and projection are
part of the architecture contract, not learned parameters. The Dreamer decoder
reconstructs this compressed DINO feature map rather than raw pixels.
Its final per-patch hidden width is 64: the 4–16 channels inherited by smaller
Dreamer pixel presets are appropriate before RGB output but impose an
unnecessary affine rank limit before a 64-channel feature target.

This is the largest deliberate departure from canonical DreamerV3 and must be
ablated later against full tokens, alternative pooling, selected intermediate
layers, and eventually learned pixel encoders.

## 4. Baseline contract

The behavioral reference is DreamerV3 revision
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`.

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

The network-size presets `1M` through `200M` mirror upstream. `12M` is the
default Dreamer model size; DINO's frozen parameters are additional.

## 5. Learning flow

For each real transition:

1. DINO encodes the new RGB frame.
2. The posterior updates the live recurrent state from the prior state and
   preceding action.
3. The frame, reward channels, flags, and recurrent context enter replay.
4. Scheduler credit accumulates according to D3's train ratio.

For each learner update:

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

Acting and learning are separate API calls. A runner may put them on different
threads later, but the first implementation does not yet provide policy
candidate promotion or a real-time scheduler.

## 6. Environment and reward seams

The model-facing environment contract is intentionally small:

- an RGB8 frame with its native width and height;
- a fixed categorical action vocabulary;
- an optional validity mask;
- separate extrinsic and intrinsic reward scalars;
- `terminated` versus `truncated` flags.

The distinction matters: both close a replay return trace, while only a true
terminal suppresses value bootstrapping.

Different games own their adapters. Atari can expose its discrete ALE action
set directly. A native game can render its own RGB frame. ARC needs a stable
categorical action vocabulary; coordinate-parameterized actions are not yet
represented by this baseline and should be added as an explicit model change.

The optional validity mask applies to live action selection. Imagination does
not yet predict future masks and therefore uses the complete vocabulary;
adapters must define invalid controls as benign/no-op transitions in this
baseline.

The reward seam is:

```text
combined = extrinsic_scale × extrinsic + intrinsic_scale × intrinsic
```

This keeps future curiosity, information-gain, homeostasis, or learned
preference experiments outside the D3 mechanics. It does not endorse any old
intrinsic reward formula.

## 7. Checkpoint and reproducibility contract

A model checkpoint contains:

- world-model parameters and optimizer moments;
- actor/value parameters and optimizer moments;
- slow value parameters;
- full configuration and pinned source/model revisions;
- fixed DINO projection seed and observation shape;
- learner/environment counters;
- return-normalizer state.

Replay and the live episode are excluded. Restoring begins between episodes
with empty replay. This choice keeps model checkpoints bounded and makes replay
persistence a separate future storage decision.

The DINO checkpoint is external and separately licensed. Kindle records its
expected repository and immutable snapshot revision but cannot prove that an
arbitrary file path contains those exact bytes.

## 8. What was removed

The pivot removes these from the active baseline:

- one-step deterministic latent models;
- PPO, GRPO, SIL, causal-credit, option, DIAYN, RND, outcome, coordinate, and
  planner variants;
- the frozen four-term scalar reward circuit;
- universal vector observation tokens and mixed continuous action machinery;
- custom visual pretraining and EfficientNet/V2-S paths;
- old ARC/Atari recipes, audit reports, stability logs, and parameter sweeps.

Removal is not a claim that every idea was wrong. It makes Dreamer the control
and prevents stale interfaces from silently constraining the new experiments.
Promising concepts return one at a time with a stated hypothesis and ablation.

## 9. Roadmap

### Phase 0 — Establish implementation confidence

Implemented in the initial pivot:

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

The world learner uses 8-step recurrent gradient chunks as described above;
the exit gate must measure whether longer chunks materially improve results
once the backend can compile them economically.

Exit gate:

- formatting, Clippy, unit tests, GPU learner canary, and DINO parity are green;
- one short native run and one Atari data-collection run produce finite metrics;
- checkpoint restore continues learning after replay refill;
- model/loss parameter counts and throughput are recorded for each preset.

Validation snapshot (updated 2026-08-25):

- formatting, Clippy, Rust/Python tests, serialized GPU learner canaries, and
  full-checkpoint DINO parity pass on the pinned stack;
- the Gymnasium/ALE adapter discovers 104 bundled games; Pong, Freeway, and
  Montezuma's Revenge reset to RGB observations, and repeated seeded 100-step
  Pong random controls produce identical actions and rewards;
- over 100,000 preprocessed interactions each, random Pong scores -2,177,
  -2,208, and -2,204 across seeds zero through two (mean -2,196.3), while
  random Freeway scores 0, 0, and 1 (mean 0.33). These are the matched external
  controls for the first dense and sparse Atari curves;
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

These are implementation and failure-localization results, not evidence that
the baseline learns the environment. A pinned-source follow-up also found that
D3 waits for 1,024 eligible replay items, discards prefill-time train credit,
and evaluates warmup at optimizer step zero; Kindle had started as soon as one
sequence existed and used the next step's warmup rate. Those startup semantics
are now corrected and measured end to end. The tiny network remains a wiring
preset rather than an upstream size. Online return and falling scalar losses
are not sufficient.

### Phase 1 — Externally rewarded baseline

Run D3-compatible evaluations before enabling intrinsic reward:

- native visual GridWorld as an integration test;
- at least one dense-reward Atari game;
- at least one sparse-reward Atari game;
- a small visual persistent world with endogenous rather than harness resets.

Measure environment steps, learner updates, wall time, replay memory, return,
world losses, continuation calibration, value calibration, actor entropy, and
multi-step latent prediction error. Use multiple independent seeds.

The goal is not immediate state-of-the-art performance. The gate is a credible,
reproducible learning curve whose failures can be located in perception,
dynamics, reward prediction, value learning, or behavior learning.

### Phase 2 — Perception ablations

Compare at matched interaction and learner compute:

1. fixed projected/pooled final-layer DINO patches;
2. full 14×14×384 patches;
3. alternative fixed projections and channel counts;
4. selected intermediate DINO layers;
5. a small learned Dreamer pixel encoder;
6. another frozen visual model only if it has a concrete hypothesis.

VidTok belongs here only if temporal tokenization becomes the hypothesis.

### Phase 3 — Intrinsic reward

Start with mechanisms whose semantics are testable:

- count- or density-based novelty in frozen observation space;
- ensemble disagreement or information gain in the RSSM;
- learning-progress reward that discounts irreducible noise;
- homeostatic/outcome channels when an environment exposes grounded signals.

Every experiment keeps the extrinsic-only run and random policy as controls.
Report task return separately from intrinsic return.

### Phase 4 — Continual memory and multiple games

Add only after single-game Dreamer is sound:

- fixed-capacity recent plus reservoir replay;
- task/environment identity at the adapter boundary;
- sampling that prevents inactive games from disappearing;
- separate tests for world-model retention and actor retention;
- actor rehearsal or distillation if the model remembers while behavior forgets;
- raw-frame anchor re-encoding to detect representation drift.

The first multi-game result should use compatible categorical action spaces.
Parameterized and continuous controls require an explicit distributional design.

### Phase 5 — Single-life safety and recoverability

Introduce:

- epistemic uncertainty separate from predicted observation noise;
- recoverability and resource outcome heads;
- risk-aware action selection;
- hard environment safety constraints;
- live/candidate policy synchronization and rollback;
- prequential scoring before each transition is learned.

External rollback remains forbidden; policy rollback is allowed because it
changes the controller, not history.

### Phase 6 — Planning, hierarchy, and scaling

Only after actor-only Dreamer is understood:

- selective runtime planning on high uncertainty or high consequence;
- planner-to-actor distillation;
- goal-conditioned critics and reusable skills;
- event-level memory and reflective proposal systems;
- Dreamer-4-style or VidTok world-model scaling if recurrence is the measured
  bottleneck.

## 10. Required ablations

At fixed real-interaction and compute budgets:

1. random policy;
2. D3 algorithm with frozen DINO and extrinsic reward;
3. the same agent with each perception alternative;
4. each intrinsic reward individually and in combination;
5. FIFO versus lifetime replay;
6. actor-only versus selective planning;
7. immediate policy synchronization versus candidate promotion;
8. recurrent baseline versus any later video-transformer model.

Claims about transfer or retention require an independent evaluation phase with
learning frozen, not training diagnostics alone.

## 11. Main risks

### Frozen features omit control-relevant details

DINO may underrepresent tiny sprites, exact coordinates, counters, or animation
phase. Dense patch tokens help, but projection and pooling can still erase
information. Diagnose this with probes and perception ablations before changing
the RSSM.

### Model exploitation

The actor can exploit errors in imagined trajectories. Keep rollouts short,
start from posterior states, prevent actor gradients into the world model, and
later add calibrated uncertainty.

### Curiosity traps

Prediction error rewards stochastic televisions and noise. Prefer reducible
uncertainty or learning progress, and always report external task performance.

### Replay cost

Even compressed observations are about 12.5 KiB each in f32; 100,000 frames are
about 1.25 GB before recurrent context and container overhead. Profile memory
before increasing capacity and consider f16 storage as a measured change.

### Architecture inflation

Kindle's old failure mode was accumulating plausible mechanisms without a strong
control. New modules enter only after the baseline is measured and only with a
falsifiable exit gate.

## 12. Pinned sources

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
