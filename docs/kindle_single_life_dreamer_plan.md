# Kindle: one actor learning to play

Updated 2026-09-12. This is the authoritative roadmap: direction, current evidence
and next decisions. Detailed protocols and measurements live in
[experiment reports](experiments/2026-09-05-kickoff.md) and their pinned
`runs/` artifacts. Working constraints remain in [AGENTS.md](../AGENTS.md).
Updating this roadmap does not change an experiment declaration.

## Direction

Build a Rust agent that improves through its own interactions, initially with
explicit game rewards and human guidance. Favor minimalism, expressiveness,
safety and speed. Keep learning and inference on Meganeura + Blade; Python is
for environment adapters, controls and analysis. Reuse mind-games rather than
building another game harness.

The order remains:

1. Preserve Pong's achieved initial-learning milestone; establish reliable
   final-policy competence across independent training seeds.
2. Learn across Atari, starting with Pong, Boxing, Freeway, Breakout and Qbert.
3. Accelerate playing **with training enabled** and test useful video/world
   pretraining, with its data and compute disclosed.
4. Use mind-games for vkQuake2 and TMNF, then a small GOG/Wine portfolio.
5. Transfer a gameplay actor to unseen titles and measure adaptation and retention.
6. Consider experience sharing only after strong single-actor GOG and transfer
   results.

Vectorized environments and batched live inference are implemented for **one
shared learner and policy**. They improve collection, not the number of learning
agents. Keep each stream's visual cache, belief, RNG and replay history
independent. Separate learner services and swarm infrastructure remain deferred.

## Current evidence and rollout videos

Pong is an Atari game; the missing result is reliable Atari breadth, not the
first Atari win. Current gameplay uses native LeVJEPA, not the historical DINO
frontend. Neither the five-game objective nor consistent Pong mastery is complete.

| Game | Completed frozen evidence | Reliability / next decision |
| --- | --- | --- |
| Pong | Old LeVJEPA 200k-action roots 0/1/2: means +10.2778 / +0.5 / +20.4651; wins 18/18, 7/12, 43/43 | Only root 2 passes the declared mastery gate. The old all-seed recipe fails; fresh longer confirmation is queued after Freeway. |
| Boxing | Fresh roots 1009/2017/3019: 123/123, 207/207 and 51/51 wins; means +83.8699 / +90.5845 / +83.5294. Untrained means −0.7222 / +0.8056 / +1.25. | Complete: all three paired learning gates pass at the declared recipe and budget. Full state, distinct initial parameters, replays and runtime evidence verified. |
| Freeway | Seed-0 hold64/hold1 pilots pass unassisted evaluation, means 31.0556 / 29.0278; untrained mean 0. Fresh root 1009 fails: mean 24.5833, only 16/36 qualifying rounds; its complete untrained control returns 0. | The pilot did not establish reliability. Root 2017 is training; finish the remaining declared roots and inspect the near-UP behavior before selecting any repair. Keep both gates unchanged. |
| Breakout | Seed-0 frozen mean 58.4583 versus untrained 0.9655; 0/24 trained and 0/29 control two-wall completions | Learned improvement, not competence. The [diagnostic](experiments/2026-09-11-breakout-diagnostic.md) finds ample reward coverage and a late return plateau. Stage a minimal-action comparison after the existing queue; retain the control and gate. |
| Qbert | Seed-0 final policy completes the first pyramid in 17/24 natural episodes, mean 3,754.17; untrained control 0/24, mean 125.00 | Both competence gates fail. Completed replay finds early misses and little post-bonus progress; prefer a separately declared bounded experience-budget comparison, not fresh-seed replication of this failed recipe. |

The [September 11 host-driver incident](experiments/2026-09-11-host-driver-incident.md)
stopped the runtime handoff: an unattended NVIDIA update left new user-space
libraries mismatched with the loaded kernel driver. Breakout finished training
on its original mappings, then the device guard failed before evaluation.
The trainer, logger and serial follower have exited. Preserve completed work
and the failure. The host now reports matching driver/library 595.91.07, and
[runtime requalification](experiments/2026-09-11-meganeura-runtime.md) passes.
The separately declared [continuation](experiments/2026-09-11-atari-continuation.md)
passes exact episode stopping and completes both failed paired pilots. The
[serial follower](experiments/2026-09-11-recovered-confirmations.md) has independently
reverified their complete raw evidence and started fresh Freeway confirmation;
Pong follows it. Preserve this completed continuation rather than restarting it.

Watch whole stream-zero evaluations, including failures and unfinished tails:

- [Boxing root 1009](../runs/boxing-confirmation-20260910.hTEDcu/seed1009-evaluation.mp4),
  [root 2017](../runs/boxing-confirmation-20260910.hTEDcu/seed2017-evaluation.mp4)
  and [root 3019](../runs/boxing-confirmation-20260910.hTEDcu/seed3019-evaluation.mp4);
  [scores, controls and replay checks](experiments/2026-09-10-boxing-confirmation.md).
- [Freeway hold64](../runs/freeway-persistence-learning-20260909.C0GoqT/hold64-evaluation.mp4)
  and [hold1](../runs/freeway-persistence-learning-20260909.C0GoqT/hold1-evaluation.mp4);
  both movies are **unassisted frozen evaluation**, unlike assisted training.
  [Fresh root 1009](../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed1009-evaluation.mp4)
  fails the unchanged competence gate; its [untrained control](../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed1009-untrained-evaluation.mp4)
  scores zero. See the [paired result and bounded diagnostic](experiments/2026-09-11-recovered-confirmations.md#freeway-root-1009-failed-frozen-gate).
- [Breakout trained](../runs/atari-driver-continuation-20260911.LR9yT3/breakout-evaluation.mp4)
  and [untrained](../runs/atari-driver-continuation-20260911.LR9yT3/breakout-untrained-evaluation.mp4);
  learned improvement, but neither completes both walls.
- [Qbert trained](../runs/atari-driver-continuation-20260911.LR9yT3/qbert-evaluation.mp4)
  and [untrained](../runs/atari-driver-continuation-20260911.LR9yT3/qbert-untrained-evaluation.mp4);
  initial-pyramid progress, not sustained competence. See the
  [complete paired result](experiments/2026-09-11-atari-continuation.md#completed-qbert-pilot-learning-without-competence).
- [Pong gameplay and world-model report](../runs/world-evaluation-20260908.Xzx3pN/report.html),
  plus the [common-recording comparison](../runs/common-world-report-20260909.O7nqqe/report.html).
  Forced cross-model recordings are diagnostics, not additional policy wins.

The [historical DINO controls](experiments/2026-09-05-self-learning.md) use final
100k-action models and 50k sampled frozen actions. They remain valid initial-
learning evidence: causal root 0 wins 28/28 frozen games with mean
+19.0714; root 1 wins 3/14 with mean −1.0714. Reconstruction root 0 also wins
28/28, mean +19.2143. These do not establish causal-objective superiority or a
matched DINO-versus-vector-LeVJEPA comparison. Persistent native GridWorld also
has three successful causal roots; it validates the learning loop, not Atari
breadth or the eventual native-game portfolio.

## Architecture: the pivot that landed and what is missing

The running design is a categorical Dreamer RSSM with a frozen video frontend
and **action-conditioned prediction before observation**. The predictor reads
the deterministic prior, not the posterior containing its target. This is the
JEPA-inspired change; reconstructing current frozen features from the posterior
remains the explicit reconstruction control.

~~~text
h[t]     = recurrent(previous belief, previous executed action)
u_hat[t] = predictor(h[t])
u[t]     = frozen_encoder(RGB history through t)
z[t]     = posterior(h[t], u[t])

(h[t], z[t]) --> reward / continuation / imagination --> actor / critic
~~~

Native LeVJEPA uses the causal prefix of each 16-arrival chunk, not a sliding
window or future frames. Chunk boundaries reset **perception only**; episode
boundaries also reset belief. Current-time spatial tokens, excluding clip CLS,
are projected/pooled to 7×7×64 F32 features. The pretrained 303.1M ViT-L encoder
is frozen and runs separately from the optimizer. Imagination uses RSSM priors,
not the encoder. Video perception and the recurrent action-conditioned belief
have different jobs.

| Component | Code / responsibility |
| --- | --- |
| Visual frontend | [vision](../kindle/src/vision/mod.rs), [LeVJEPA](../kindle/src/vision/levjepa.rs), [DINO](../kindle/src/vision/dinov3.rs): preprocessing, causal caches, fixed projection and actual weight identity |
| Agent and collection | [agent](../kindle/src/dreamer/agent.rs), [vector](../kindle/src/dreamer/agent/vector.rs): observe/act/scheduled learning, independent live streams, checkpoint integration |
| Learning | [world](../kindle/src/dreamer/world.rs), [networks](../kindle/src/dreamer/networks.rs), [behavior](../kindle/src/dreamer/behavior.rs): RSSM, prediction/reward/continuation/KL, imagined actor and two-hot critic |
| Experience and execution | [replay](../kindle/src/dreamer/replay.rs), [runtime](../kindle/src/dreamer/runtime.rs): contiguous sequences, reset masks, initialization, optimization and synchronization |

Keep the three objective controls distinct: reconstruction/future coefficients
0.25/0, 0.25/0.25 and 0/0.25. Match head structure, initialization, scalar
normalization and behavior settings. Reset observations are not predictable
transitions. Preserve full recurrence and corrected F32 gradients.

This is not yet action-conditioned world pretraining, end-to-end video learning
or cross-game skill transfer. Equal feature shapes do not make different
encoders' coordinates or old replay compatible. Keep the
[LeVJEPA provenance and numerical gates](experiments/2026-09-06-levjepa-pong.md).
The isolated [batched DINO candidate](https://github.com/kvark/kindle/blob/d909883032bb5a8e37ef199c6afbba64cb62db47/docs/experiments/2026-09-07-batched-dino.md)
is an unqualified matched-control candidate, not a frontend fallback.

### What to call the coupled loop

Use **adaptive execution** as project shorthand. The established umbrella is
online reinforcement learning; sustained adaptation and retention lead toward
[continual reinforcement learning](https://arxiv.org/abs/2307.11046).
This name does not claim a new learning algorithm.

In code, the actor is the policy network; the agent also includes perception,
world model, replay and critics. Keep observe, act and learn_scheduled explicit:
acting must not secretly update weights, especially during frozen evaluation.
A cheaper evaluation constructor is a local optimization, not a reason to split
the agent into services.

## Atari: acceptance before broader claims

Beating most of a declared Atari suite is a reasonable ambition, not guaranteed
by the Dreamer name. Our frozen-encoder implementation does not inherit published
[DreamerV3](https://danijar.com/project/dreamerv3/) results.

The active five-game objective requires **all three fresh training roots**
1009/2017/3019 to pass each game's final-checkpoint gate and beat separately
restored untrained controls. Initial/trained weights and actual seed provenance
must be verified. Their live seed ranges are disjoint under the current
`seed + stream` rule; adjacent model roots would reuse live RNG inputs.

| Game | Unchanged per-root frozen competence gate |
| --- | --- |
| Pong | ≥20 natural games, ≥90% wins, mean return ≥+15, no cutoffs |
| Boxing | ≥20 natural games, ≥90% wins, mean return ≥+50, no cutoffs |
| Freeway | ≥20 natural rounds, ≥90% reach 25 crossings, mean ≥25, no cutoffs |
| Breakout | ≥20 completed episodes, ≥90% clear both walls / reach 864 points |
| Qbert | ≥20 completed episodes, ≥90% complete the first pyramid **and** mean final score ≥15,000 |

For Breakout/Qbert, a task achieved before a later cutoff counts without calling
the episode natural; a cutoff without task completion is a failed task episode.
Keep every completed episode, including extras from faster streams, and report
partial tails separately.
The [task observers](experiments/2026-09-08-atari-task-observers.md) are post-hoc
evaluation, never policy inputs or training rewards. Scripted ROM fixtures,
random discovery, positive generic Atari scores and CPU tests are not Kindle wins.

### The immutable serial queue

The [original follower](experiments/2026-09-10-atari-serial-handoff.md) stopped
on the driver failure and remains terminal. The separately declared
[recovered-driver continuations](experiments/2026-09-11-recovered-confirmations.md)
preserve completed work and now enforce this order:

~~~text
Boxing: three fresh roots + final evaluations + untrained controls [complete]
  -> episode/runtime and recovered-driver qualification [complete]
  -> Breakout paired pilot [complete; competence failed]
  -> Qbert paired pilot [complete; competence failed]
  -> Freeway three-root confirmation [root 1009 failed pair; root 2017 training]
  -> longer-budget Pong three-root confirmation [queued]
  -> Breakout action-width hardware/synthetic diagnostic [queued; no learning]
~~~

Qbert retains its 200,004-action unassisted-training pilot. Freeway's fresh roots
each receive 200,004 actions with probability .5/hold64 exploration, followed by
75,000 unassisted frozen actions. Pong's fresh roots each receive 400,008 training
actions without overrides. Qbert and Pong use v4 four-episode-per-stream frozen
evaluation, cap 600,000. Every protocol keeps its separately restored untrained
controls, unchanged task gates and all outcomes. Cap exhaustion before an episode
target is incomplete. Pong is larger-exposure confirmation, not an isolated budget
ablation or a reinterpretation of the old failed campaign.

Keep each pinned native/Python package together; main's dependency update does
not switch these experiments. Do not restart old queues, edit active inputs or
manually launch successors. Each entrypoint requires actual predecessor exit and
complete raw evidence before GPU work. Valid competence failures remain failures;
integrity, incomplete-data or runtime-safety failures stop without retries.
Breakout's isolated [four-action candidate](experiments/2026-09-11-breakout-minimal.md)
has CPU-tested explicit replay/checkpoint schemas. Both future arms
must use its same qualified upstream backend; the old eighteen-action pilot is
historical context, not the matched control. Its compiled full-gradient/state
fixtures support a separately pinned, one-shot hardware/synthetic diagnostic
after the learning queue. No diagnostic GPU result
exists yet. Even a pass leaves native four-action initialization/restore, N6 pixel
replay and combined-memory checks before a paired learning declaration and then
fresh-root confirmation. The state/capture/ledger matrix, measured controller and
ten complete replay bindings are CPU-prepared, including real-ALE fixtures with
explicitly fake actors. This is not native qualification; no pixel GPU declaration
or follower exists. Require complete hardware/prerequisite proof before declaring
that gate. Keep comparisons within each action width exact; do not add exploration
assistance or a longer budget in the same comparison. Completed pilots and
scheduling never establish five-game success.

Count executed interactions, not vector ticks. Episode-reset observations can
advance replay warmup without earning action credit, so derive updates from the
complete source-matched ledger rather than copying Boxing's 49,651 everywhere.
Bind the final save, restored counters and evaluated model to that audited count.

### After the five-game result

Extend the diagnostic panel with Seaquest, Frostbite and Private Eye, then a
predeclared Atari-26 suite and, if useful, Atari-57. The proposed “most games”
target is ≥14 of Atari-26 meeting game-specific final competence thresholds using
the median across seeds; disclose all seed distributions and failures. Use valid
human-reference normalization where available, otherwise explicit task completion.
Improvement over random is a learning gate, not “beating” a game.

Independent per-game training tests algorithm breadth, not a general shared Atari
policy. Sequential transfer is separate. Pin game lists, budgets, wrappers,
reward definitions, seeds and final evaluation before results. Retain 100k
learning-curve readouts, but declare longer runs and their evaluation budgets
in advance; never extend only failing seeds or select a lucky checkpoint.

## Evaluate the world model separately from its policy

Yes: replay an actual match through a frozen model and ask what it predicted
**before consuming each target observation**. Compare reward, continuation and
future features with what actually occurred; keep posterior estimates separate.
Feature prediction is not imagined RGB, and lower loss is not a gameplay win.

The [completed first-match evaluation](experiments/2026-09-08-world-evaluation.md)
reproduces every recorded action and transition with zero updates. All three old
Pong models use action information in feature prediction. Own-policy errors alone
do not rank models on a common distribution.

The [common-input report](../runs/common-world-report-20260909.O7nqqe/report.html)
then conditions all three models on the same three recordings: 11,388 transitions
per model, 62 positive rewards, 29 negative rewards and only three terminals.
All three same-model diagonals reproduce prior results exactly; all six cross-model
runs pass input/state/memory checks. Each model predicts positive rewards and
features best on its own recording. Four of six cross-model all-frame prior reward
MAEs exceed the always-zero baseline, but the [squared-error supplement](experiments/2026-09-08-world-evaluation.md#september-12-the-zero-baseline-is-metric-specific)
finds every cross pair beats zero under MSE. Keep both metrics and event strata;
the MAE comparison alone does not imply an absence of useful reward signal.
Model 1 has the worst pooled positive error but the best negative error:
its world model is not uniformly worst.

This indicates limited cross-trajectory reward generalization, **not a proven
cause** of the policy failures. Another policy's logged return is not an unbiased
critic target. Preserve actual/unrelated-action, feature-persistence and zero-reward
baselines; report event counts and cache-reset strata. Fixed-stride forecasts can
miss sparse classes, cache resets can inflate persistence error, and event AUC
does not establish magnitude calibration or terminal accuracy.

The completed [motion and coverage diagnostics](experiments/2026-09-08-world-evaluation.md)
also argue against assuming numerical collapse or immediately enlarging perception.
Every trained belief contains useful motion information, without matching gameplay
ranking. In the first 80k actions, positive-free replay batches are 63.9% / 86.5% /
65.9% for roots 0/1/2. Root 1 is especially reward-starved, but coverage alone does
not explain all ranking differences. Repeated samples are not distinct experience.

Freeway supplies a sharper discovery result: the
[plain-policy pilot](experiments/2026-09-09-freeway-zero-signal.md) has zero reward
and zero reported absolute advantage in all 49,651 updates despite finite state
and declining prediction loss. The subsequent matched exploration pilot yields
unassisted competence. Do not generalize its held-action benefit to other games
or call the Pong diagnosis settled.

Next, use the new Pong confirmation's preselected **first four complete stream-zero
matches per final root**, without score filtering. The
[multi-match extractor](experiments/2026-09-10-multimatch-world-probe.md) passes CPU
accounting and independent ALE replay, but needs serial/vector, strict/forced,
same-model native forecast and GPU-safety checks before the common H1 comparison.
That GPU diagnostic is not scheduled by the learning follower. Keep current learning
arms fixed; require a new declaration before changing replay, exploration or losses.

## Runtime: uncapped, accelerated, and eventually free-running

| Mode | Current support |
| --- | --- |
| Uncapped step-driven playing and learning | Supported; no wall-clock pacing |
| Super-real-time playing **with learning** | R64 Boxing exceeds 1× in aggregate; selected R256 does not |
| Free-running play without time control | Required, not validated by step-driven Atari |
| Frozen inference | Supported; its speed is not training throughput |

The selected control is **LeVJEPA, 12M, N6/R256/B16/T64, full BPTT64,
microbatch 16, F32**. Six independent environments share one learner/policy.
Live visual/RSSM/policy inference and row-independent replay/head work are
batched without removing recurrence or mixing histories.

### Meganeura: current source and pinned runtimes

Main pins upstream `ce80e9cd`. The [September 12 preflight](experiments/2026-09-11-meganeura-runtime.md#september-12-upstream-preflight)
finds tip `3622e06f` differs only in four documentation/paper files: all runtime,
build and test inputs remain identical. No runtime fix is missing from main or
the current block-matmul/Breakout candidates. Recheck upstream before new backend
diagnosis; carry older candidates forward before testing them as current code.
The pinned runtime fixes generated matmul epilogues and includes the required LeVJEPA
frame-prefix attention/cache-alias corrections. Blade stays 0.9.0; minimum Rust
is 1.92. The [recovered-driver and backend qualification](experiments/2026-09-11-meganeura-runtime.md)
passes full production gradients, cache parity, complete optimizer/state/report/
trace comparisons and memory checks. Main's separate source-matched integration
also passes. No block-matmul or experimental tuning is enabled by this update.

The pixel-qualified upstream Atari package is native `abf4ae5d`, with matching
source `1e00e818`. Both warmed N6/R256 orders show only **0.6–0.9%** higher
throughput, still **0.573× aggregate real time** and about **0.0955× per stream**.
At least 3,303 MiB stays directly free. These short exact-state comparisons do
not establish training reliability or solve the runtime bottleneck.

The historical [Atari control](experiments/2026-09-09-meganeura-update.md#use-the-qualified-package)
remains `f6a2b6ad` / source `90b4763` / backend `4d45ba3a`; its prior update had
no measured speedup. It also reproduces archived old-driver results on the
recovered driver. Existing declarations and old checkpoints retain their pinned
packages and source-matched Python bundles; this source update does not switch them.

Main's Python accounting interface differs from the isolated Atari package.
Keep source-matched runners/auditors with each package; don't mix them. Historical
default extensions and executables remain intact, and old models still require
their original backend. Root's old release binaries contain the rejected
grouped-RSSM candidate: use the documented isolated package or rebuild, never
assume a source checkout identifies an existing executable.

### What actually costs time

The completed [fresh Boxing readout](experiments/2026-09-10-boxing-confirmation.md#completed-training-cost-readout)
covers 195,996 post-warmup actions: **8.47 actions/s, 0.5645× aggregate real time,
about 0.0941× per stream**, with zero final training debt in every measured window.
Time is 73.2% learning, 26.2% observation and under 1% emulator stepping.
Mean GPU activity is about 69%, not occupancy or calibrated idle-gap time.

A learner update costs roughly 345–346 ms: world training 161 ms, imagination
85 ms, posterior inference 59 ms, behavior training 19 ms and synchronization
about 20 ms. These are subdivisions of learner time, not extra elapsed costs.
At this fixed replay ratio, actual frame clocks leave only **140 ms/update**
for aggregate 1× if other work stays unchanged. World training alone exceeds that.
World synchronization is only 3.48% of wall time; removing it cannot solve the
throughput gap.

N6 retains ≥3,302 MiB directly free in qualified runtime and completed training
checks. The earlier fixed-learner N4/N6/N8 comparison found roughly 4,889 / 3,302 /
1,630 MiB directly free. N6 is about 1.2% slower than N8, chosen for safety,
not a speedup. N8 fails the **2,048 MiB directly measured free-memory gate**.
Total minus used omits driver reservations and cannot substitute for memory.free.
Record memory.reserved and coverage too; changed packages/configurations require
their own matching runtime evidence.

Prioritize world-training kernels/layout and recurrent handoffs, then perception.
[Device-resident imagination](experiments/2026-09-08-device-imagination.md) already
improved exact paired throughput by 12.9–13.3%; preserve its completed controls.
The [grouped-GRU candidate](experiments/2026-09-08-grouped-rssm-gates.md) fails exact
full learning from report 3 and is not adopted. The
[small-batch block-matmul](experiments/2026-09-10-block-matmul.md) and
[world-sync fan-out](experiments/2026-09-09-world-sync-fanout.md) candidates have CPU
evidence only. The block candidate now has an unchanged carry onto the qualified
upstream backend, avoiding a backend change in its future comparison. Neither
has a verified GPU speedup or may displace the fixed queue.

For every optimization, require production losses/all gradients, reset causality,
complete weights and optimizer moments from update 1, exact state/action traces,
direct-memory headroom and untraced AB/BA timing. Readback waits include unfinished
producer compute and transfers; they are not automatically GPU idle. Current
external captures resolve queue submissions, not individual kernels or calibrated
idle gaps. Serialize GPU work and avoid large CPU graph builds during training.

Lower replay ratios, smaller models and larger learner batches are separate
learning-compute ablations. Historical R64 Boxing reaches about 1.31× aggregate
real time and passes its one-seed gate, but its +51.55 mean has less margin than
R256's +92.49. Retain R64 and test learning quality across seeds before switching.
Report actual emulator frames, actions, updates, debt, cold construction and
end-to-end wall time. Aim for sustained >1× with retained quality; 2× is a stretch.

The completed [native AGC/BPTT ablations](experiments/2026-09-05-kickoff.md#native-agc-and-bptt-ablations)
retain control on a small saturated task, not proof that clipping or full BPTT
is unnecessary in 12M Atari. Keep the selected settings; any Atari objective,
representation, AGC or BPTT comparison needs its own matched declaration.

Use mind-games time control for accelerated development, then separately test
free-running play with arrival timestamps, executed action durations, observation
gaps and bounded training debt. Pausing a game is not that capability. Try measured
serial scheduling before actor/learner concurrency.

## Video and world pretraining

Loading a pretrained encoder is supported. A supported video-dataset/world-only
pretraining workflow is not. The main learning call also trains behavior; it is
not an offline world trainer. The isolated
[world-only candidate](https://github.com/kvark/kindle/blob/8196bd5666e9a79302ffd2d510b82509da965434/docs/experiments/2026-09-07-world-pretraining.md)
has CPU-tested world-only updates and strict fresh-runtime dynamics initialization,
but ingestion, GPU and adaptation/retention gates remain unfinished. Its initialized
checkpoints use format 4 for offline lineage; do not reinterpret ordinary format 3.

| Prior knowledge | Required data | Transfer being tested |
| --- | --- | --- |
| Generic video encoder | Disclosed video/model provenance | Perception; fresh dynamics and behavior |
| Gameplay visual adaptation | Source-title clips and held-out splits | Adapted perception, not policy skill |
| Action-conditioned world pretraining | Aligned RGB, executed controls, durations and boundaries | Compatible dynamics/predictor; fresh target behavior and declared head resets |
| Gameplay policy warm start | Rewarded source play or labeled action supervision | Source dynamics **and policy**, separately evaluated from dynamics-only transfer |

Start with aligned mind-games recordings and assess compatible
[Pixels to Play](https://arxiv.org/abs/2508.14295) data. Missing actions are not NOOP;
unlabeled video does not identify which control caused a transition. Missing reward
is not zero reward. Use explicit label masks, contiguous clips and reset boundaries;
disable unsupported actor/critic/replay-value targets. Inferred actions need their
own inverse-dynamics evaluation. Content-verify datasets and disclose offline
interactions and compute separately from target-game learning.

Compare fresh, encoder-only and encoder-plus-world initialization at equal target
budgets. Verify intended world parameters change while behavior does not. Declare
replay/recurrence, head, optimizer and normalizer resets; keep strict ordinary
restore checks. Useful pretraining means faster retained gameplay learning, not
just smaller feature error. Preserve causal/cache/encoder identity gates and
evaluate trained beliefs before choosing a larger visual grid or temporal window.

## After Atari: mind-games, vkQuake2/TMNF, then GOG/Wine

The inspected mind-games checkout (`d86cd439`) provides Podman, Chronolock,
game-only cgroup freezing, Dullahan capture, uinput and scalar/vector environment
contracts. Its Kindle submodule (`eb8af7fc`) and BatchAgent-based adapters are
historical, not the current Dreamer path. Reverify these revisions/APIs before
integration; respect mind-games' separate BC work.

Build one small RGB8/action/reward/boundary adapter. Keep engine observers outside
policy/world inputs, forward actual executed actions and both boundary causes,
and test frame alignment in frozen and learning modes. GPU capture exports are
not automatically zero-copy Blade inputs; optimize only a measured transfer cost.

- **vkQuake:** short integration/calibration reference, not another long campaign.
- **vkQuake2:** preferred next FPS target. No registered adapter/config was found
  in the inspected checkout; implement launch, clock, capture, input and reward/
  terminal hooks. Measure kills/deaths and level/objective completion, not movement.
- **TMNF:** audit existing checkpoint/finish/lap-time extraction and test held-out
  tracks. Speed without progress is not the task.
- **GOG/Wine:** choose a small existing-catalog panel, such as Broforce or Spelunky
  for 2D control and Project Warlock for FPS, subject to fresh launch/reward checks.
  A working launcher is not evidence Kindle can learn it.

A compact FPS vocabulary must permit movement/strafe, yaw/pitch aiming, firing,
interaction and needed combinations. Don't inherit navigation-only actions or
game-solving macros. Audit startup scripts and shaping: Broforce's inspected setup
includes scripted walking before handoff. Record assistance, menus, resets, input
overrides and reward-read failures explicitly. Natural deaths/respawns are allowed;
cloning or rewinding a live game for training is not.

## Rewards, skill transfer and the gate before swarms

Explicit game rewards remain primary. Keep game events, human guidance and intrinsic
reward separate; privileged coordinates/enemy state may feed external reward/task
observers, never policy inputs. Record assistance and the actual action executed.
The bounded visual-visitation bonus stays off in controls; its intrinsic-only
results do not establish useful competence. Any intrinsic experiment retains an
extrinsic-only arm and evaluates held-out dynamics and later guided adaptation.

A general FPS actor must carry behavior, not only perception. Reserve an unseen
shooter before source training/tuning; a held-out map is not a held-out title,
and vkQuake2 is no longer unseen once used for development. Audit source corpora
and disclose generic-pretraining overlap.

Compare three target-game arms under identical rewards, controls and budgets:
fresh dynamics/policy with the common encoder; transferred dynamics with fresh
policy; transferred dynamics **and compatible source policy**. Don't reset the
policy whose transfer is being tested. Declare action mappings and all head,
optimizer, normalizer, replay and belief resets/retentions. Measure frozen zero-shot
play, fixed-budget adaptation (for example 1k/10k/100k actions), final competence
and learning-curve area; then reevaluate source games for forgetting. BC and
demonstration-assisted comparisons are separate. Transfer alone is not a JEPA
advantage without matched objective/representation controls.

Before swarms, require predeclared competence on at least three GOG titles across
two control genres, multiple fresh training seeds, verified Wine execution where
used, whole-rollout videos, retained final policies, held-out cross-title adaptation
and a source-retention check. Report failures and negative transfer.

Long-lived operation also needs bounded replay/archive storage and atomic
complete-state recovery. Current checkpoints omit replay/RNG/live state and saves
are non-atomic; detecting a torn save is not fault-tolerant continuation.
Preserve representation/action/reward lineage and contiguous histories.

Only then compare two independent Kindles sharing immutable experience chunks
against isolated controls. Recompute receiving-agent belief, retain environment/
control/reward provenance and do not transfer latent coordinates or novelty blindly.
Larger swarms and shared optimizers remain out of scope.

## Research and implementation discipline

Keep one scientific variable per comparison and declare budgets, all seeds, final
checkpoints and evaluation before observing outcomes. Require actual state/encoder
identity, complete optimizer moments, whole action/reward/boundary accounting,
independent replay, videos and raw memory/timing evidence. A CPU test, training
win, one frozen seed or a completion flag is not a reliability result. Preserve
failures, interrupted runs, historical executables and corrected audit evidence.

The [recovered baseline work](experiments/2026-09-05-kickoff.md#recovery-and-audit)
includes full-recurrence row microbatching, F32 gradient safeguards and strict
score/provenance checks; retain it. The
[pinned upstream Dreamer control](experiments/2026-09-05-kickoff.md#upstream-control-provenance)
ran locally, but one seed and differences in encoder, precision, ALE and reset
accounting do not support an apples-to-apples superiority claim. Keep exact
historical pins in their manifests and the [native frontend](../kindle/src/vision/mod.rs),
not a moving label in the roadmap.

Boxing's confirmation, recovered-driver and latest-backend runtime qualification
and source integration are complete. Breakout and Qbert have complete paired
pilots, but both fail competence. Preserve their checkpoints, controls and videos;
do not spend fresh three-root confirmations on these unchanged failed recipes.
Monitor the declared Freeway confirmation and subsequent Pong run. Breakout's
bounded action-width diagnostic remains queued after them. Qbert's
[completed replay diagnostic](experiments/2026-09-12-qbert-diagnostic.md) separates
initial misses from low post-bonus progress: only 4.68% of training frames follow
the first pyramid, including its bonus animation. Prefer a separately declared
fresh 400,008-action dose test with a retained 200,004-action midpoint, keeping
the rest of the recipe fixed. No new GPU job is queued; stop at the declared
endpoint, retain failures, and confirm a successful choice on fresh roots.
Limited exposure is not a proven cause or a reason to relax the gates.
Separately qualify the multi-match
world-model diagnostic candidate when the shared GPU is available, without
changing or displacing pinned work. Optimize measured costs, not activity percentage.
Do not substitute another framework, larger perception, arbitrary long run or
concurrent learner for evidence of stronger single-agent behavior.

Keep this document decision-focused. Milestone chronology, command output,
tensor hashes and test counts belong in the linked experiment reports and
`runs/`, not repeated across roadmap sections. Delete superseded code and redundant
documentation when they have no current purpose, preserving unrelated changes.
