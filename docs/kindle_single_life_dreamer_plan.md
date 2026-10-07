# Kindle: a single actor that learns while acting

Updated October 6, 2026. This is the authoritative roadmap.
[PR31](https://github.com/kvark/kindle/pull/31) is the done/running/next dashboard;
[experiment reports](experiments/README.md) retain detailed evidence and failures.
[AGENTS.md](../AGENTS.md) gives working rules. No separate status document.

## Direction: CDP is the main path

Learn to play games through experience, using sparse explicit rewards and human
guidance where needed, and eventually useful intrinsic motivation. Keep learning
and inference native in Rust on Meganeura + Blade. Favor one effective actor,
small models and short experiments before larger budgets, concurrency or swarms.

**Adopt Dreamer-CDP as the main development architecture.** Keep Dreamer's RSSM,
replay and imagined actor/critic learning; train a small visual CNN jointly with
a deterministic latent predictor instead of reconstructing RGB. This advances
the JEPA bet on cheaper useful prediction. It does not require LeVJEPA, a frozen
video encoder, DINO or a detached visualization decoder.

The selected small recipe is Size1M, eight environments, B8/T16/H15/R32,
microbatch 8, replay 100000 and full BPTT. CDP has 804,785 trainable parameters,
including its CNN; the current exploration ensemble brings the whole learner to
1,057,201 parameters. There is no extra5.5M or303M frontend. Retain the
[qualified CDP loss and split learning rates](experiments/2026-10-04-cdp.md),
AGC and `ac_grads=false`. Change one factor at a time.

The CLI currently still defaults to learned-RGB reconstruction. **New main-path
experiment declarations must use `--cdp`.** This roadmap change does not silently
change runtime defaults. RGB remains the numerical/learning reference and a
fallback, not a mandatory extra arm in every exploration experiment. Causal
LeVJEPA is an optional future video/3D hypothesis, not a prerequisite for progress.

The causal model is: previous belief plus executed action produces a prior and
predicted embedding; the current frame's CNN embedding then forms the posterior.
The prediction target is detached and cannot enter its own prior. Encoder
gradients still arrive through world/task/value and temporal source paths.
Posterior state estimates are not forecasts.

## What is established

- **Fast iteration and a faithful control exist.** Shared/fused learning and a
  small batched recipe are complete. The original 12M 3x speed target remains an
  unmet, user-accepted stretch target. The small RGB port has
  [upstream numerical and three-seed learning evidence](results/2026-09-30-small-replication-learning.md).
  The old 45-run matrix is cancelled: 24 completed, 21 unstarted; never resume it.
- **Frozen Tiny did not earn its 2D cost.** The
  [three-seed comparison](results/2026-10-01-frontend-decision.md) found no clear
  pretraining advantage and 18% longer whole-agent time than RGB. The subsequent
  [joint-Tiny](results/2026-10-03-joint-tiny-learning.md) and
  [direct-policy-gradient](results/2026-10-03-policy-tiny-learning.md) early
  screens did not establish a useful benefit. Those results do not refute video
  priors or latent prediction. The unstarted posterior-Tiny ablation is deferred.
- **CDP improves the current Seaquest screen.** Three fresh paired seeds,
  200,000 actual actions / 49,939 updates per run: final online mean 543.6 versus
  RGB 318.1, paired difference +225.5 [62.8,330.4]. Every pair favors CDP, with
  35.9% less world-training time and 13.6% less wall time. The
  [complete result](results/2026-10-04-cdp-learning.md) includes all episodes,
  tails, curves, qualification failures and independent audits.
- **CDP's world representation is more useful, not solved.** Frozen probes find
  much more readable player state in the RSSM. Fifteen-step latent forecasts beat
  persistence, the constant training mean and unrelated actions. One-step
  persistence, zero-reward MAE and matched privileged coordinate persistence
  remain stronger. Sparse reward/terminal counts limit conclusions; there is no
  bullet/full-state sufficiency claim.

The controlled CDP/RGB comparison is one short game and three learner seeds,
not Atari-wide superiority or a reproduction of the authors' full benchmark.
The completed five-game exploration screen below now measures broader retained
learning, with important failures still unresolved.

## Current execution order — strategy reset

The [September strategy reset](strategy_reset_plan.md) remains the rationale:
iteration speed, useful representations, exploration/reward, video priors, then
real-time deployment. Phases 0–2 and the CDP follow-up are complete. The user's
October 4 direction puts CDP on the main path through the remaining stages.

### 1. Atari learning, exploration and budget (Phase 3)

The October6 goal is to reach **DreamerV3 quality on Boxing, Pong, Freeway,
Breakout and Qbert with CDP**, retaining the earlier budget comparison. All five remain in scope. Neither learning a subset nor completing a
queue establishes this goal.

**The first breadth screen is complete:** fifteen200k-action learners, including
all three retained Freeway cohorts. Frozen means are Boxing69.74, Pong−18.53,
Freeway24.14, Breakout4.36 and Qbert403.13. Four games improve over actual initial
controls in every seed; Pong improves in only one of three. All45 guards,720
selected natural frozen episodes, zero updates/cutoffs, exact saved tensors,
full trajectory replays and whole videos pass. [Results, curves and videos](results/2026-10-05-cdp-five-game-screen.md).

**No long-run DreamerV3 target is reached.** Keep the
[published references](results/2026-10-05-dreamerv3-five-game-reference.json)
separate: Atari57 uses200M frames; Atari100k uses400k frames and a different
nonsticky/minimal-action protocol. Our200k actions are about800k frames, not
less experience than Atari100k. At approximately the same early frame count,
published Atari57 is also weak on Pong/Breakout; our means are similar on
Boxing/Pong/Breakout, Freeway is ahead and Qbert somewhat behind. Different
online averaging windows and protocols prevent an exact parity claim.
Same-hardware/protocol RGB controls are required for a compute-saving claim.

**The longer Pong test is now complete:** three fresh500k-action learners
retain frozen scores−21.00/−2.375/−20.9167,7/72 wins. Only2017 improves;
the two failed seeds' KL drops further while reward separation stays weak.
Nine guards,144 selected natural episodes, unchanged tensors and full
replays/videos pass. [Result and curves](results/2026-10-06-cdp-pong-budget.md).
The online mean−15.27 is below the published−7.16 near2M frames, with protocol
differences retained. No further raw-cosine budget extension is declared.

**Frozen diagnosis is complete:** all four models preserve their346 tensors
over131,072 diagnostic actions. Every CNN retains readable ball/paddle state;
the two failed seeds' RSSMs do not reliably retain ball state or predict reward
events. Constant means nearly match their tiny cosine losses. Seed2017 retains
useful state and h15 forecasts. [Results and all controls](results/2026-10-06-cdp-pong-world.md).
This is not proof of complete encoder collapse. No privileged labels enter
learning; old-trained models are probed on newly qualified Meganeurab684ffd9.

**Centered CDP passes its three-seed early-learning screen:** the
[six fresh200k-action runs](results/2026-10-06-cdp-centered-learning.md) finish
October6 08:41 UTC. Frozen mean−8.4583 versus−18.6528 unchanged CDP; paired
gain+10.1944 [7.5,12.375]. Every centered seed improves over its actual initial
and matched control. All18 guards/288 selected natural episodes, exact tensors
and full replay/videos pass. Only4/72 centered matches are wins: not mastery.
Centering the training objective is the only changed mechanism; source CNN
gradients, capacity, rates, replay and exploration remain fixed. Promote
`--cdp --cdp-centered` as the next candidate, not the unchanged paper objective.
The ordinary `--cdp` control and CLI default remain unchanged.

**Paired world probes are complete:** six guards/exact346 tensors pass over
196,608 diagnostic actions, zero actor updates. All centered models retain
readable ball state and useful h15 negative-event forecasts; the raw2017
control remains stronger on some coordinate probes, and positive/terminal
evidence stays sparse. [Full result and controls](results/2026-10-06-cdp-centered-world.md).
One bounded
synthetic profile has completed; imagination alone has3,100 dispatches and
10.91ms ordinary GPU /13.28ms synchronized wall time. Kernel instrumentation
perturbs timing; do not turn its family shares into whole-agent percentages.
**The bounded throughput trial passes:** existing grouped B128 products reduce
ordinary synthetic network updates30.49->28.01ms (8.12%) and imagination
dispatches3,100->1,210 on the same updated Meganeurac6376542 backend. All14
guards, independent arithmetic/upstream checks and raw/centered production plus
frozen smokes pass. [Result and limits](results/2026-10-06-cdp-imagination-throughput.md).
Keep the change, without claiming a measured whole-game speedup or utilization.

**The five-game centered500k allocation is complete after evaluation repair:**
[result, curves and every whole video](results/2026-10-07-cdp-centered-five-game-budget.md).
All15 learners improve over actual initial controls; Boxing/Freeway pass the
historical gates in every seed, Pong remains unstable, Breakout/Qbert remain
weak. No long-run quality target is reached. Cost7.5M actions/1,874,085 updates /
16h24m7s training, plus all evaluation and earlier development compute. This is
longer centered-package learning, not a matched five-game loss/RGB ablation.
The repaired Breakout3019 natural cohort preserves its original cutoff and
exact207,288-action prefix; both CPU audit failures and extra collection remain
disclosed. All720 selected natural episodes and46 GPU guards pass.

**The frozen Breakout diagnosis now completes:** [result and limits](results/2026-10-07-cdp-breakout-world.md).
All six models remain exactly frozen. Ball state and h15 forecasts are weak,
but the pixel control is ill-conditioned: per-pixel whitening amplifies rare
changes to4,021 despite bounded RGB input. An F64 replay agrees with its saved
GPU predictions. The fixed-range2,048-update follow-up completes in5.01s with
no new actions/actor updates: paddle R².979, ball still near zero even in
training. This head is not an information ceiling. A fixed diagnostic color
centroid finds all483 eligible held-out open-playfield balls within.63/.55
source-pixel x/y RMSE; the GPU input retains those balls. No detector or RAM
enters the actor. The [existing12M preflight](experiments/2026-10-07-cdp-capacity.md)
passes195 finite updates and exact frozen restore with no new warnings.
Next: three fresh12M Breakout learners at500k actions each against the retained
1M controls, roughly12h total. Keep losses/exploration/native fixed and retain
the repaired natural-episode protocol. This tests whole-model capacity, not
isolated CNN capacity; no encoder redesign or automatic five-game extension.

The exploration mechanism is a small GPU-native action-conditioned ensemble.
It predicts detached CNN embeddings and subtracts each head's all-action mean
before disagreement, removing action-independent uncertainty. One actor uses
the bonus in imagination and replay-value targets; the world reward head stays
extrinsic-only. No second encoder, scripted action, game label, CPU feature
readback or external reward shaping enters learning. This is an adaptation
inspired by [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html), not
a reproduction of its separate exploration actor.

Freeway's zero-reward blocker is resolved across all three seeds without action
aids. [Its full200k result](results/2026-10-05-freeway-effects-200k.md) preserves
the stricter mastery limitation: only3019 passes. The earlier
[all-zero baseline](results/2026-10-05-cdp-freeway-exploration.md),
[soft/visual-target failures](results/2026-10-05-freeway-embedding-learning.md)
and [weak32k frozen result](results/2026-10-05-freeway-action-effects-frozen.md)
remain additional development compute. The200k result is not a matched
extrinsic-only ablation, and Freeway was used to develop the mechanism.

The stopped45-run CDP/RGB/Tiny comparison and cancelled historical12M matrix
remain stopped, with all completed/interrupted evidence retained. Venture,
Tiny comparisons and new representation matrices are deferred by the current
five-game goal. After breadth is established, a separately declared held-out
exploration task with fresh extrinsic controls can test generalization of the
bonus. Do not repeat Seaquest or tune old mastery gates merely to fill a queue.

### 2. Video priors for dynamics and behavior (Phase 4)

Build on the online CDP agent, rather than returning by default to a large frozen
frontend. First test a single world/dynamics initialization against fresh CDP at
equal online experience; later consider inferred actions or a behavior prior.
Action-free video and action-labelled recordings are different supervision.

Use whole-recording train/validation/test splits. Disclose corpus, offline
compute and same-title exposure; missing actions or rewards are not NOOP/zero.
World-only pretraining must not silently update actor/critic. Success is fewer
online interactions or less total time to useful retained gameplay learning,
not only a lower latent error. Confirm a promising method on a second game
before scaling it. Causal Tiny must earn any return through this comparison.

### 3. A useful native-game actor (Phase 5)

Use `/x/Code/mind-games`: **vkQuake2 first**, vkQuake only as the transport
reference, then **TMNF**. Recheck the current integration API before changes.
Complete the reward/terminal/action adapter and GameSession integration; measure
kills/objectives or track finishes, not just movement.

The target is game GPU -> GPU capture/preprocessing -> encoder/RSSM/policy ->
CPU input events. Qualify the current v4 real-producer ownership/reuse path
before relying on it; the old v3 vkQuake plumbing success is not v4 validation.
Sparse explicit game rewards and disclosed human guidance remain acceptable.

Start with one learner and batched/serialized acting. For free-running games
without time control, measure p50/p95/p99 capture-to-action latency, observation
gaps, actual action durations, effective replay ratio and learner debt.
Introduce an asynchronous actor/learner **only if these measurements require it**;
it is not a prerequisite for the next Atari experiment or first native-game
learning check.

### 4. GOG games, then held-out transfer and retention

Apply the same actor to a small predeclared GOG/Wine panel with explicit sparse
rewards. Menus, startup scripts and input overrides are assistance and must be
reported. Look for repeated learning success rather than one favorable clip.

Reserve an unseen shooter before tuning a general FPS actor. Compare a fresh
agent, transferred world with fresh policy, and transferred world plus policy
at matched adaptation budgets. Declare action mappings and optimizer,
normalizer, replay and belief resets. Report zero-shot behavior, adaptation
speed and forgetting on source games. A held-out map is not a held-out title.

### 5. Swarm learning, last

Only after strong single-actor native/GOG learning and measured transfer/
retention, test immutable experience sharing or several actors feeding one
learner. Keep independent causal histories and explicit data/weight ownership.
Do not build swarm infrastructure or a concurrent learner service now.

## Current game status

These are current sticky/full-action centered CDP500k results unless explicitly
marked historical. The common recipe is not a matched five-game comparison
against raw CDP or RGB. Frozen means cover three learner seeds and24 natural
episodes/model; streams/episodes are not independent learner replicates.
The [five-game report](results/2026-10-07-cdp-centered-five-game-budget.md) links every
trained/initial-control whole stream-zero video and all score/action/time curves.
Links into `runs/` require this workspace.

| Game | Current measured result | Unresolved question / historical gate | Evidence and whole videos |
| --- | --- | --- | --- |
| Seaquest | Three extrinsic CDP/RGB pairs: online543.6 vs318.1; CDP13.6% less wall time and35.9% less world-training time. Frozen world probes complete. | Controlled result on one title, not frozen policy mastery or five-game speed superiority. | [Learning and world report](results/2026-10-04-cdp-learning.md) |
| Boxing | Frozen77.69 vs1.00 initial; seeds74.46/71.63/87.00,72/72 wins. | All three pass the historical>=90% wins/mean>=50 gate. Published99.61 remains unmet. | [Current results and videos](results/2026-10-07-cdp-centered-five-game-budget.md#whole-rollout-videos) |
| Pong | Frozen+4.54 vs−20.40; seeds+2.92/+11.92/−1.21,54/72 wins. Every seed improves. | Still seed-sensitive. Published20.45 and historical>=90% wins/mean>=15 remain unmet. | [Current results and videos](results/2026-10-07-cdp-centered-five-game-budget.md#whole-rollout-videos); [earlier paired world report](results/2026-10-06-cdp-centered-world.md) |
| Freeway | Frozen30.26 vs0; seeds29.50/30.54/30.75. All72 rounds score27–32. | All three pass the original25-crossing gate, without action hints. Published33.40 remains unmet. | [Current results and videos](results/2026-10-07-cdp-centered-five-game-budget.md#whole-rollout-videos); [exploration evidence](results/2026-10-05-freeway-effects-200k.md) |
| Breakout | Frozen5.14 vs1.57; seeds4.00/3.50/7.92 after reviewed natural-cohort repair. Poor control; timeout retained. | Published381.81 and historical two-wall/864-point gate far away. Ball survives RGB64 on checked open-playfield frames; learned state/forecasts remain weak.12M preflight passes; three-seed capacity comparison next. | [Current results, repair and videos](results/2026-10-07-cdp-centered-five-game-budget.md); [world diagnosis](results/2026-10-07-cdp-breakout-world.md) |
| Qbert | Frozen993.75 vs191.32; seeds1057.29/892.71/1031.25. Only2/72 first pyramids. | Published193,220.77 and historical>=90% first pyramids/mean>=15,000 remain far away. | [Current results and videos](results/2026-10-07-cdp-centered-five-game-budget.md#whole-rollout-videos); [historical Tiny evidence](experiments/README.md) |

Historical Tiny results used nonsticky protocols and250k same-title video
observations (45k train+5k validation/game). Freeway also used a.5-probability
random action held for64 steps in training. Preserve them as historical, not
fresh/unassisted CDP or matched controls. Their gates are unchanged for interpreting
old claims, not perpetual development exit gates. Achievements before a cutoff
remain achievements without relabeling the episode natural; retain all tails.

## Runtime, speed and world-model checks

Eight small-recipe environments share batched perception/policy and one learner;
their recurrent states, resets, RNG and replay histories remain independent.
Uncapped, step-driven playing plus training already works. The completed five-game
CDP+exploration recipe on Meganeura592a2f5a achieves about8x aggregate /1x per-stream realtime.
Mean update31.62ms includes16.19ms imagination (51%),8.45ms world training (27%)
and4.33ms posterior work (14%). The older extrinsic-only CDP Seaquest comparison
used28.91ms updates; keep its settings distinct. GPU utilization is unmeasured,
not inferred from frame-clock ratios. Target measured whole-agent bottlenecks,
especially imagination, not just the now-cheaper world loss.
The October6 matched Pong comparison on Meganeurab684ffd9 averages32.11ms
raw /32.13ms centered updates and about28min per200k learner. Centering has no
material observed cost increase; it is not itself a throughput optimization.
The subsequent [grouped B128 trial](results/2026-10-06-cdp-imagination-throughput.md)
on Meganeurac6376542 reduces matched synthetic network updates30.49->28.01ms.
Whole-game cost for that changed native will be measured in the declared
five-game allocation; do not substitute the synthetic number for production.

Atari emulation/frame upload is a CPU-environment fallback; GPU preprocessing,
batched acting and resident replay collection are implemented. Native GPU
capture integration exists, but current v4 producer qualification remains open.
The [capture report](results/2026-10-02-matching-external-allocations.md) keeps
the allocation warning and historical v3 success separate. Only selected actions
leave the acting hot path; rewards, diagnostics, checkpoints and sampled learner
data may still cross the host. Some scalar targets/slow-critic work remains
host-side. Do not claim a completely host-free learner.

CDP uses one GPU resize from native Atari frames to RGB 64, not a downscale/
upscale detour. A future encoder must receive its intended native-detail input;
do not silently feed RGB 64-upscaled frames into LeVJEPA. Tiny's16-arrival chunk
reset is perception-only, not an environment or RSSM reset.

World-model checks accompany meaningful learning milestones, without becoming
a new qualification campaign. Forecast before observing targets; test held-out
real trajectories at short and longer horizons against persistence, constant
means, unrelated actions and zero-reward/always-continue controls. Report
representation spread, fitted-state readability, reward/terminal counts,
MAE/MSE and matched cohorts. Privileged observers are evaluation labels only.
Posterior estimates, predictable features and imagined RGB are different things.
Existing CDP probes and [older world reports](experiments/README.md) retain
negative results; do not select only successful trajectories.

## Execution discipline and evidence

- Screen small with one changed mechanism and at least three learner seeds.
  Report return versus actual interactions and wall time, seed-bootstrap
  uncertainty, configuration, every aid and all episodes/tails. A lower replay
  ratio, new loss or smaller model is a learning tradeoff, not an unchanged-
  learning speedup. Increase capacity or budget for evidence of a limitation,
  not automatically to12M.
- Frozen evaluation never updates model/optimizer tensors. Predeclare evaluation
  cohorts and report untrained controls when claiming competence. Development
  curves, numerical smokes and rollout videos alone are not mastery.
- Use the GPU and keep Meganeura/Blade current before diagnosing old bugs.
  Current qualified CDP/RGB pins are Meganeurac6376542/Bladee349cddf; the
  [grouped throughput qualification](results/2026-10-06-cdp-imagination-throughput.md)
  passes independent numerical and short production/restore checks, not a new
  learning comparison. Tiny retains its historicalb684ffd9 qualification.
  Retain original backend identities for old results.
  No repeated upstream learning replication without a relevant change.
- Serialize bounded native jobs under the [host guard](gpu_incident_response.md)
  in persistent systemd user services. Require the expected device and >=2 GiB
  sampled Vulkan budget headroom; this is not physical free or peak VRAM.
  Ordinary GPU/JAX initialization is allowed; separate NVML polling stays off.
  October4 permission allows GPU use unless wedged: record standalone allocation
  warnings without an approval gate. Actual API/numerical failures, hard faults
  and deadlines still stop the affected job for review. No blind retry, reset,
  driver change or host recovery.
- Preserve failures and compact JSON/Markdown reports; leave large artifacts in
  `runs/`. Keep current status in the PR, not in chronological roadmap appendices.
  Do not rebuild unchanged native code for docs or create a new pin/framework
  layer per experiment. Only the user merges.
- Checkpoints preserve weights/moments, not a complete replay/RNG/live-belief
  lifetime. Interrupted restore is not equivalent to uninterrupted learning.
  Natural game deaths/respawns are allowed; cloning/rewinding a live game for
  training is not. Full-lifetime recovery remains later work.

The initial CDP roadmap adoption was documentation-only. The user's subsequent
five-game comparison goal authorizes the separately declared training/evaluation.
CLI defaults remain unchanged; old artifacts retain their original scope.

Core implementation: [agent](../kindle/src/dreamer/agent.rs),
[vector collection](../kindle/src/dreamer/agent/vector.rs),
[world model](../kindle/src/dreamer/world.rs),
[networks](../kindle/src/dreamer/networks.rs),
[behavior](../kindle/src/dreamer/behavior.rs),
[replay](../kindle/src/dreamer/replay.rs).
