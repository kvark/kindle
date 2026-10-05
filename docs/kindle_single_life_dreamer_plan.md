# Kindle: a single actor that learns while acting

Updated October 5, 2026. This is the authoritative roadmap.
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
including its CNN; there is no extra 5.5M or 303M frontend. Retain the
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

This is one short game and three learner seeds, not Atari-wide superiority,
frozen policy competence or a reproduction of the authors' full benchmark.
The next step is learning with CDP, not another encoder-selection matrix.

## Current execution order — strategy reset

The [September strategy reset](strategy_reset_plan.md) remains the rationale:
iteration speed, useful representations, exploration/reward, video priors, then
real-time deployment. Phases 0–2 and the CDP follow-up are complete. The user's
October 4 direction puts CDP on the main path through the remaining stages.

### 1. Next: Atari learning and exploration with CDP (Phase 3)

**New October5 user goal:** unlock Boxing, Pong, Freeway, Breakout and Qbert
with CDP, and see whether DreamerV3 scores are reachable with less budget.
The [five-game budget study](experiments/2026-10-05-cdp-five-game-budget.md)
reuses all three Freeway200k cohorts and adds twelve fixed200k-action runs on
the other four games. Keep the qualified CDP/action-effects package unchanged;
frozen final/initial-control pairs and full videos measure retained learning.
Compare full Atari57/200M-frame and Atari100k/400k-frame references separately:
our200k actions are about800k frames. Published curve/protocol differences are
explicit, and compute savings need matched same-hardware controls. No claim
from simply being smaller. All five and the budget comparison remain in scope;
this screen is the first decision point, not presumed goal completion. Venture
is deferred. Neither historical queue nor the stopped45-run matrix restarts.

**Current result: Freeway's exploration blocker is resolved.** Three fresh
200k-action CDP seeds reach frozen means22.21/23.21/27.00; all72 candidate rounds
score19–30 versus zero in all72 initial-control rounds. [Results, curves and
whole videos](results/2026-10-05-freeway-effects-200k.md) pass nine guards, exact
frozen tensors, full replays and zero updates/cutoffs. Only one seed passes the
unchanged historical mastery gate; this is useful retained learning across all
three seeds, not full mastery or completed Phase3. Do not start a Freeway-only
gate-tuning campaign. The newly declared five-game screen above takes priority
over the previously proposed Venture comparison.

**Path to this result:** the user's October4 decision focused exploration on
unassisted Freeway.
The small disagreement mechanism below is implemented and
[qualified](results/2026-10-04-cdp-exploration-qualification.md); the declared
three-seed screen against extrinsic-only CDP is complete with zero real rewards
in both arms. The [result](results/2026-10-05-cdp-freeway-exploration.md) shows
nonzero intrinsic advantages but near-uniform policy entropy. The completed
[coverage/bonus diagnosis](results/2026-10-05-freeway-disagreement-diagnosis.md)
finds little coverage change, weak action contrast and no upward bonus preference.
The [soft-target test](experiments/2026-10-05-freeway-soft-disagreement.md) replaces
sampled one-hot ensemble targets with detached posterior probabilities: the
same expected regression gradient with less sampling noise. Qualification and
all three fresh seeds finish; [all still score zero](results/2026-10-05-freeway-soft-learning.md),
with almost unchanged bonus/coverage. The existing detached CNN embedding target,
where player position is more readable, is now [qualified](results/2026-10-05-freeway-embedding-qualification.md).
No new encoder: the wider256-output ensemble adds33,280 parameters and changes
natural target scale. Its [same-budget three-seed screen](experiments/2026-10-05-freeway-embedding-disagreement.md)
reuses completed soft/extrinsic controls. All three finish with
[zero rewards and unchanged coverage](results/2026-10-05-freeway-embedding-learning.md);
the larger bonus does not produce useful action selection. Skip frozen
competence evaluation. Six frozen diagnostic checks pass; state variation
overwhelms action contrast18–20x. An offline transform that removes each
predictor's all-action mean produces much clearer action preferences without
game-specific inputs. Test this [action-effects bonus](experiments/2026-10-05-freeway-action-effects.md)
at the same budget, not a further target/encoder change. That screen now finds
[2/1/1 real crossings](results/2026-10-05-freeway-action-effects-learning.md)
across the three seeds, versus zero in retained controls. Two seeds improve
late; the third loses early progress. Mean update cost rises to31.70ms (~25%
more). The [frozen comparison and videos](results/2026-10-05-freeway-action-effects-frozen.md)
now complete:14 crossings/72 candidate episodes versus1/72 extrinsic-trained
and0/72 untrained. All216 natural episodes have zero updates and exact saved
tensors. This small retained improvement supports one
[fresh200k-action follow-up](experiments/2026-10-05-freeway-effects-200k.md) per
seed, unchanged mechanism and new held-out evaluation seeds. That follow-up is
now complete with the strong retained result above. The32k cohort alone did not
unlock Freeway; all-zero controls and that weak result remain in the record.
No matched200k extrinsic superiority claim or automatic extension. Neither
Boxing nor the stopped matrix resumes now.

**Stopped for diagnosis, October4 at20:42 UTC:** the user requests investigation
before further zero-score runs. Seven Freeway pairs complete; CDP3019 is retained
as interrupted. The [diagnosis](results/2026-10-04-freeway-zero-reward-diagnosis.md)
finds zero reward discovery and zero policy advantages, while raw-ALE controls,
a bounded GPU reward pulse and the existing Seaquest positive control pass.
Published DreamerV3 Atari-100k Freeway traces also score zero. Do not infer a
CDP-specific regression or continue the matrix unchanged. The selected generic
exploration experiment measures first reward, replay coverage and advantages;
reward-bearing Boxing can separately test representation learning later. The full requested comparison
below remains unfinished, not replaced by a single easier game.

**Retained scope:** the user requests training CDP on the selected Atari
games and evaluating against Dreamer RGB and Tiny JEPA. This bounded
[three-method comparison](experiments/2026-10-04-cdp-atari-comparison.md) was started
before the intrinsic-reward experiment below. The user confirms **Boxing, Pong, Freeway,
Breakout and Qbert**: 45 small runs (three methods x three seeds per game),
each with frozen evaluation. Freeway runs first. Keep all methods extrinsic-only and
use fresh matched controls, not the old aided/non-sticky/large-model results.
Completed evidence and new rollout links are in the [five-game report](results/2026-10-04-cdp-atari-comparison.md);
the PR remains the running-status dashboard.

**Yes, Atari training is next.** Use the qualified small CDP agent to test
learning from sparse rewards, not to reopen the unchanged five-game mastery
queue or spend more days establishing the RGB baseline.

The authorized exploration experiment, before resuming the matrix, is
**unassisted Freeway**:

- **Selected mechanism, qualified:** small
  action-conditioned latent-disagreement heads, inspired by
  [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html). Reward
  uncertainty during imagined rollouts, with detached predictor inputs/targets,
  one existing actor and no second pixel encoder. This is a small
  adaptation, not a reproduction of Plan2Explore's separate exploration actor.
  The [reference implementation](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/expl.py)
  computes its bonus from current ensemble predictions during imagination.
  Compute the current bonus for replay-value targets too; do not train one
  critic against incompatible imagined and cached-replay reward definitions.
  The [new declaration](experiments/2026-10-04-cdp-freeway-exploration.md) uses
  four heads and coefficient1 through `--disagreement-scale 1`. The older
  visitation bonus is CPU-only and rejected by GPU collection; keep it disabled. Do not substitute
  raw CDP prediction error, a feature-readback workaround or scripted UP.
- Control: extrinsic-only CDP. Candidate: the same CDP agent plus **one**
  GPU-compatible intrinsic reward mechanism, kept in a separate reward channel.
- Use learner seeds 1009/2017/3019 and independent per-stream histories. No
  persistent random-action override, game-specific action aid, video pretraining
  or newly shaped external reward in this comparison.
- Keep the qualified sticky 0.25/full-action Atari protocol and small learning
  recipe. Start with32,768 aggregate actions/seed, not a
  mastery promise. Declare its exact settings, finite
  budgets and stop conditions before launching; qualify its numerical and cost
  behavior without a new training matrix.
- Measure first reward, extrinsic-return curves versus actual actions and wall
  time, seed variation, update cost and learner debt. Intrinsic return is never
  the game-performance metric. Evaluate predeclared frozen policies without
  updates and retain whole rollout videos, not selected successful episodes.

Before considering six full200k-action runs, use a32,768-action/8,131-update
screen per arm/seed using the same N8 small recipe, with reward discovery and
nonzero advantages as diagnostics. Qualify reset-safe action/target alignment,
detached gradients, zero-coefficient baseline equivalence, repeat-versus-novel
state behavior and whole-update overhead first. A completed screen is a decision
point, not an automatic extension or a mastery claim. The user selected this
exploration work before the unchanged three-method Boxing comparison.

Six learning runs completed at23:00 UTC on October4; all guard/counter/checkpoint
audits pass. Its predeclared all-zero branch skips frozen evaluation; the later
soft-target screen is a separately declared change, not its continuation.
Qualification smokes are excluded; nonzero intrinsic reward and
advantages are not Freeway reward discovery. The earlier8,135-update prose
was corrected to the launched/audited8,131; the action budget is unchanged.
Do not extend a weak run automatically or add a hyperparameter sweep.

After that decision, test the selected recipe on **one predeclared held-out
exploration game, planned as Venture**, at a matched finite budget. Seaquest
provides the existing learning reference; it need not be retrained unchanged.
Use fresh same-game extrinsic-only controls where required, not historical Tiny
controls. Keep RGB comparisons for an actual representation/backend question.

**Phase outcome:** evidence of unassisted sparse-reward learning across three
seeds and an improvement on a second task, with acceptable whole-agent cost.
A negative result identifies the next limitation; it does not authorize a
bigger queue. Investigate exploration, reward prediction, dynamics or capacity
according to the observed failure rather than changing all of them together.

Use the five-game comparison above to assess learning breadth and stability,
without repeating it unchanged. Breakout and sticky Pong remain useful
diagnostic checks, not perpetual release gates. Strong learning on many games
is the ambition; neither
a simple-looking game nor the name Dreamer guarantees mastery at 200k actions.
Separate per-game training measures algorithm breadth, not one transferable
multi-game policy.

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

Seaquest and Freeway include current CDP development results. The other four
rows retain historical recipes, **not CDP results**. New CDP-only200k screens
on those four are now declared; the old CDP/RGB/Tiny matrix stays deferred.
Historical gates remain
unchanged for interpreting old claims, not as exit gates for this development
study. Videos are
whole stream-zero evaluations with tails, while full multi-stream cohorts
determine the result. Links into `runs/` require this workspace; committed
[results](experiments/README.md) are the public summaries.

| Game | Measured result | Unchanged gate / next decision | Rollout |
| --- | --- | --- | --- |
| Seaquest (current development screen) | Three fresh small CDP/RGB pairs: online543.6 versus 318.1, paired+225.5 [62.8,330.4]; CDP uses13.6% less wall time. Frozen world diagnostics complete. | CDP is the main development path. No frozen policy competence or mastery gate claim. | [Learning curves and world report](results/2026-10-04-cdp-learning.md); no new policy-evaluation video |
| Boxing | Three roots pass: 123/123, 207/207, 51/51 wins; means +83.87/+90.58/+83.53; controls near zero | ≥20 natural matches, ≥90% wins, mean ≥+50, no cutoffs. Complete. | [1009](../runs/boxing-confirmation-20260910.hTEDcu/seed1009-evaluation.mp4), [2017](../runs/boxing-confirmation-20260910.hTEDcu/seed2017-evaluation.mp4), [3019](../runs/boxing-confirmation-20260910.hTEDcu/seed3019-evaluation.mp4) |
| Pong | Historical non-sticky roots pass 71/72 wins versus 0/76 controls. But root1009 with 25% sticky actions wins only2/24 equal-cohort matches, mean−7.1667; all 3/31, mean−8.3871. State/replay/video audit passes. | ≥20 natural matches, ≥90% wins, mean ≥+15, no cutoffs. Fixed-recipe pass; **robustness fails**. One stochastic-evaluation root, no new control pair. | [Sticky video](../runs/pong-sticky-evaluation-20260926.SxeHCw/seed1009.mp4), [new report](experiments/2026-09-26-gpu-pixels-and-pong-robustness.md), [historical videos/controls](experiments/README.md#current-pong-confirmation) |
| Freeway | **Current unassisted CDP:**200k-action seeds1009/2017/3019 frozen means22.2083/23.2083/27.0000. All72 rounds score19–30 versus0/72 initial controls: exploration unlocked. Historical aided Tiny means32.9167/31.6944/33.25. | ≥20 natural rounds, ≥90% reach25 crossings, mean≥25, no cutoffs. Current CDP passes only3019; not three-seed mastery. Historical Tiny passes only its aided/pretrained recipe. Next development decision is a held-out exploration task, not gate tuning. | [Current CDP and all videos](results/2026-10-05-freeway-effects-200k.md), [32k controls](results/2026-10-05-freeway-action-effects-frozen.md), [historical Tiny report/videos](../runs/tiny-freeway-confirmation-20260922.tij9QW/results.md) |
| Breakout | Complete matched Tiny comparison: four actions mean 10.9167 versus .875 control; eighteen mean 11.625 versus .93103. Both trained arms0/24 two-wall completions. Historical Large mean 30.7917 also fails; no demonstrated pretraining benefit in one Tiny seed. | ≥20 completed episodes, ≥90% clear both walls / reach864 points. Fewer actions did not repair this seed. Keep eighteen as reference; diagnose before another recipe. Historical Large four-action arm stays held. | [Complete comparison](../runs/breakout-minimal-comparison-20260926.xsQCaK/results.md), [four-action video](../runs/breakout-minimal-comparison-20260926.xsQCaK/a4/evaluate.mp4), [eighteen-action video](../runs/breakout-minimal-comparison-20260926.xsQCaK/a18/evaluate.mp4), [pretraining ablation](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md), [Large](../runs/breakout-action-pilot-20260920.kNeotb/results.md) |
| Qbert | Completed Tiny R64 seed 0: 3.2M final22/27 first pyramids (81.5%), mean 12,595.37; 1.6M midpoint24/24, mean 8,673.96; control0/24, mean 120.83. Complete state/replay/video checks pass. | ≥20 episodes, ≥90% first pyramids **and** mean ≥15,000. Final fails both thresholds; the first-episode probe misses its terminal and retains high values through a scoreless ending. | [Final](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/final.mp4), [midpoint](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/midpoint.mp4), [control](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/untrained.mp4), [report](../runs/qbert-r64-3m2-20260925.FrriIH/results.md), [world/policy diagnostic](../runs/qbert-hazard-probe-cpu-v2-20260926.GnWOvb/results.md) |

For Breakout/Qbert, an achievement before a later cutoff remains an achievement
without relabeling the episode natural. Retain every completed episode and tail.
Historical Freeway training used a .5-probability random action held for64 actions;
evaluation was unassisted. Its wins are not unaided exploration. Tiny also had
250k same-title random-play observations from Boxing/Pong/Freeway/Breakout/Qbert
(45k train +5k validation/game), unlike the fresh CDP/RGB comparison.

The five historical protocols were non-sticky with no reset no-ops; environment
seeds did not create varied starts in the checked action traces. Sticky Pong
already exposes that limitation. Do not relabel historical wins as robust
mastery or count episodes/streams as independent learner seeds.
[Human-normalized historical scores](results/2026-09-27-historical-scores.md)
are descriptive, not a matched modern benchmark.

## Runtime, speed and world-model checks

Eight small-recipe environments share batched perception/policy and one learner;
their recurrent states, resets, RNG and replay histories remain independent.
Uncapped, step-driven playing plus training already works. The CDP Seaquest runs
achieve8.74–8.76x aggregate real time, **1.093–1.096x per stream**. Full updates
average 28.91 ms, including 8.19 ms world training and 14.14 ms imagination. GPU
utilization is still unmeasured; these timings are not utilization percentages.
Optimize measured whole-agent bottlenecks, not just the now-cheaper world loss.

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
  Current qualified runtime pins are Meganeura592a2f5a/Bladee349cddf; the
  [backend refresh](results/2026-10-04-meganeura-main-qualification.md) passes
  independent numerical and short production/restore checks, not a new learning
  comparison. Retain original backend identities for old results.
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
