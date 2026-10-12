# Kindle: a single actor that learns while acting

Updated October 11, 2026. This is the authoritative roadmap.
[PR31](https://github.com/kvark/kindle/pull/31) is the done/running/next dashboard;
[experiment reports](experiments/README.md) retain detailed evidence and failures.
[AGENTS.md](../AGENTS.md) gives working rules. No separate status document.

## Performance pivot

**Current work: [backend training performance](backend_training_performance_plan.md).**
The user prioritizes all three remaining opportunities before more Kindle
training: cooperative convolution weight gradients, real mixed precision /
BF16 where supported, then measured fusion/layout/dispatch improvements.
The Freeway screen below is stopped at39,568 actions/9,831 updates on its first
learner; the partial checkpoint remains. No completed learner, later arm or
frozen cohort. Synthetic backend qualification continues, not game training.
The [first backend round](results/2026-10-11-backend-training-performance.md)
qualifies repeated-row addressing (2.54% lower CDP update latency and1.05 GiB
less imagination allocation); the convolution candidate has no whole-update
gain and stays draft. [Native BF16 dense learning](results/2026-10-12-bf16-learner.md)
now qualifies complete updates: CDP161.49->149.55ms (7.40%),
RGB71.33->66.47ms (6.81%). Auto remains default; convolution/grouped products,
masters/optimizer and sensitive reductions stay F32. World training regresses,
so mixed-precision coverage/performance is unfinished. Fresh JAX RGB FP32/BF16
24.35/12.42ms leaves a large gap. Initial complete-submission traces still miss
intermediate chunks; a timing-only repair now qualifies coverage with~2.7%
diagnostic overhead and unchanged reports/tensors. No learning or
SM-utilization claim, and no return to game training yet.

The user has selected a wall-clock performance round on the5080. The stopped
Pong queue stays cancelled. Optimize one useful learner before adding GPUs or
concurrent seeds; retain the current quality results as controls, not a mandate
to repeat a long mastery campaign.

1. Update Meganeura to latestd7a7293; Blade56f0565 is unchanged. The initial
   430f14b refresh adds two compiler-search fixes; five more Vulkan/packed-F16/
   convolution fixes land during this session. Preserve old runtime controls
   and retain the revisions used by each completed measurement.
2. Reuse each exploration head's state projection across all18 actions, then
   center in hidden space and project only the selected effect. Keep all-action
   centering, four heads, targets and coefficient unchanged. This is
   a new candidate under the user's revised arithmetic contract, not retroactive
   qualification of the rejected October7 result.
3. Expose checkpointed training precision: Auto, F32, F16-input products with
   F32 accumulators/master weights/optimizer/reductions. Do not describe this as
   BF16 support or half-storage throughout the learner. Check finite gradients,
   losses and checkpoints; judge adoption by learning and time-to-result, not
   bitwise trajectories or the old2e-5 end-to-end gate.
4. Compare B8, B16 and B32 at T16/R32, with full-batch centered CDP and no
   world microbatching. B32 consumes512 replay positions per update and updates
   every16 actions, versus128/every4 for B8. It reduces optimizer-step frequency
   and changes centering statistics; equal replay ratio is not identical
   learning. Stop sizes that violate the2GiB sampled headroom floor. Report
   time per replay position and whole-agent actions/s, not just update latency.
5. Profile matched native/JAX RGB work and native CDP/exploration separately,
   including compile time, ordinary complete updates, CPU/GPU gaps and hot
   kernels. Attribute differences in precision, model, exploration and batch;
   the old1M RGB comparison is not a current12M CDP comparison. Separate
   instrumented attribution from ordinary timing. Identify concrete Meganeura
   work rather than assuming that wider matrices are free.
6. The [performance result](results/2026-10-11-cdp-performance.md) selects
   B32/F16 for a now-stopped [three-seed paired Freeway screen](experiments/2026-10-11-cdp-performance-learning.md),
   200k actions per arm/seed against optimized B8/Auto. The synthetic replay
   throughput gain is2.66x; learning/time-to-score is not yet established.
   Keep all failures; no new five-game campaign or automatic extension.

Concurrent seeds can fill otherwise idle hardware without changing each seed's
batch, but do not inherently speed one learner and may compete for bandwidth
and memory. Wider batches are the first experiment. Neither idle capacity nor
room for multiple12M workers follows from a sampled VRAM estimate.

Cleanup so far: removed regenerable Rust debug builds from `target/debug` and
`python/target/debug`; filesystem free space549→612GiB (rounded). Rebuilding
restores them. Run evidence, checkpoints, videos and user-owned files remain.

## Preserved pivot boundary

**October11,20:25 UTC: training stopped at the user's request.** The existing
five-game goal is paused, not achieved. Nothing in that allocation will restart
automatically. The performance work above is newly authorized; the long-run
stages below are retained context, not an active training queue.

- **Preserved boundary:** Pong1009 stops gracefully at189,576 actions /
  47,333 updates after1h29m6s. Partial and actual-initial checkpoints, logs,
  declaration and source pins remain; saved partial hashes match the final log.
  Seeds2017/3019 and all evaluations are cancelled. There is no completed2M
  learner or new frozen result. See the [stop receipt](results/2026-10-11-cdp-pong-stop.json).
  A restore lacks replay/RNG/live belief and would not continue the same lifetime.
- **Reusable baseline:** qualified native CDP/RGB, eight batched environments,
  one GPU resize of native pixels,1M fast controls and12M quality candidates.
  The [game table](#current-game-status) links completed seed results and videos;
  Qbert's latest2M cohort and frozen diagnostics are complete. Preserve those
  controls rather than deleting evidence or changing their historical gates.
- **Unresolved:** reliable five-game quality, a matched RGB compute advantage,
  and transfer to native games. The interrupted Pong run settles none of these.
- **Decision to make for the pivot:** distinguish sample efficiency from wall
  time. Use matched-frame learning comparisons and time-to-score for any new
  study; keep the old long-run targets as historical references, not evidence
  that short-budget learning is broken. Current12M updates take about110ms and
  imagination, not world training alone, is the largest measured component.
  The performance round above supersedes this earlier stop boundary; the old
  long-budget queue remains cancelled and does not authorize concurrency.

The next work is the bounded performance round above, not an extension of Pong.

## Retained direction: CDP is the main path

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
**The three-seed12M capacity comparison now completes:**
[results, curves and whole videos](results/2026-10-07-cdp-capacity.md).
Frozen Breakout48.17/15.88/50.58, mean38.21 versus retained1M5.14;
paired+33.07 [12.38,44.17]. Every seed improves, but none reaches381.81.
The same500k-action recipe costs3.58x the training wall time:11h39m37s total.
All nine guards/144 natural frozen episodes/exact tensors/full videos pass;
three allocation warnings and all tails remain. No worker or automatic
extension remains for that Breakout cohort. This tests whole-model capacity,
not isolated CNN capacity;12M forecast quality remains untested. The other-game
follow-up is separate, below.

Use12M as the next quality candidate and1M as a fast engineering control.
The [projection-reuse trial](results/2026-10-07-cdp-exploration-throughput.md)
fails fixed numerical gates before timing and is not adopted. Original graphs
are restored; no tolerance relaxation or optimization sweep. The
[latest-backend control refresh](results/2026-10-07-meganeura-control-refresh.md)
passes20 guards/2,824 upstream comparisons and three train/frozen smokes.
Paired12M updates107.85->108.00ms, with exact first outputs,272 metric reports
and346 final tensors in the fixture. Adopt upstreamf104/native83be without a
speedup claim; this does not erase the stronger full-head failure.

**The five-game12M capacity comparison is complete:**
[Freeway result, combined comparison and all video links](results/2026-10-09-cdp-freeway-capacity.md).
The [four-game allocation](experiments/2026-10-07-cdp-12m-four-game-capacity.md)
finishes October9 at16:08 UTC in47h1m32s; its automatic CPU review finishes
16:12 UTC. With retained Breakout, fifteen500k learners cost7.5M actions /
1,874,085 updates /58h22m14s training,3.56x retained1M. All45 guards,
720 natural frozen episodes, exact saved tensors and30 whole replays/videos
pass. Every tail,35 excess episodes,52 allocation warnings and earlier
reporting/repair failures remain. No worker or automatic extension remains.

Capacity raises every paired reward mean on Breakout/Pong/Qbert, but Qbert's
first-pyramid successes still come only from3019. Boxing's gain interval
includes zero and its win rate was already72/72. Freeway regresses on average:
27.54 versus30.26, with crossing gates72/72 ->48/72. Keep the weak2017 seed,
not just1009's33.58. The game table below retains every result. This is not
universal capacity superiority, an exact same-binary ablation or tested12M
prior-forecast quality. Historical model/backend identities remain unchanged.

At approximately2M frames, the completed12M online means are ahead of released
DreamerV3 bins on Pong/Freeway/Breakout and close on Boxing/Qbert. Different
protocols/windows prevent a parity claim; all released seeds, fixed early bins
and the original last10% of200M targets remain in the
[combined data](results/2026-10-09-cdp-five-game-capacity.json).
All five quality targets and the RGB compute-saving question remain open.

**October9 backend qualified:** [refresh](results/2026-10-09-meganeura-refresh.md),
Meganeura31026833/Blade56f0565/native7311547d. All20 guards,2,824 upstream
comparisons and three train/frozen smokes pass. One paired12M timing measures
107.97->108.23ms, with exact first outputs,272 metric reports and346 final
tensors in this fixture. Adopt the fixes without a speedup or learning-cause
claim. Three allocation warnings and the corrected CPU provenance failure
remain; historical learning identities are unchanged.

**T64 does not advance:** [preflight](results/2026-10-09-cdp-sequence-preflight.md)
passes validity/frozen checks but saves only4.6% per replay position, below its
10% gate, with2.05GiB minimum sampled headroom. Keep T16; no sequence sweep.
The [one-shot cost capture](results/2026-10-09-cdp-imagination-cost.md) attributes
38.7% of instrumented imagination time to the four all-action ensemble first
projections. This is neither SM utilization nor a reason to retry the rejected
factorization or relax numerical gates.

**Qbert budget study complete:** [three fresh2M-action learners](results/2026-10-11-cdp-qbert-budget.md)
finish October11 at16:10:24 UTC on unchanged12M/T16. Frozen mean7,367.01
versus retained500k2,657.99, paired+4,709.03 [3,382.29,6,260.42]; all three
scores improve. First-pyramid completions0/0/24 ->20/22/20: more experience
helps the weak seeds, but3019 loses four successes. No seed meets the full
historical or long-run gate. Online6,545.50 is below the fixed published
8M-frame bin14,607.50, with protocol/model/window differences retained.

The cohort costs46h40m1s training,4.002x the retained500k cost. All nine guards,
144 natural frozen episodes, exact tensors and six whole replays/videos pass;
221,648 frozen actions, nine excess episodes, all tails and ten warnings remain.
Final CPU review passes in52.02s; no worker remains. Historical controls retain
their backend pins; this is not an exact same-binary budget ablation.
The [completed frozen forecast comparison](results/2026-10-11-cdp-qbert-forecasts.md)
finds useful h15 reward predictions in every2M seed: MAE/RMSE beat zero, mean,
persistence and unrelated actions; positive-event AUC.951/.967/.973. Later-stage
forecasting improves strongly over500k; survival remains unresolved with sparse
terminal evidence. Six guards/exact346 tensors pass,386,496 replayed actions,
zero updates,24m16.84s; original lifecycle failure/source and extra actions
remain. This is not universal world-model quality or a new competence result.
The [six-evaluation action-mode check](results/2026-10-11-cdp-qbert-action-mode.md)
is complete: greedy raises first-pyramid completions62/72→69/72, but mean
score falls7,367.01→6,953.47; paired−413.54 [−2,092.71,580.21]. All six guards,
144 natural episodes, exact tensors and whole videos pass;218,936 actions,
zero updates,3m32.63s total. Keep sampled evaluation as standard.
The [live-value check](results/2026-10-11-cdp-qbert-values.md) now compares
exact batched beliefs with full real discounted returns, including intrinsic
reward. Every critic beats zero/calibration mean; R².435/.927/.531. Bias differs
by seed/stage, not a shared collapse; the measured intrinsic fraction is<.03%.
Five guards/exact346 tensors/72 natural episodes/full traces and videos pass;
193,248 actions plus1,024 smoke, zero updates,88.87s capture. CPU review repairs
only wall-clock comparison; no GPU retry. No worker remains.
The [bounded measured-build check](results/2026-10-11-cdp-measured-build.md)
now completes. Adopt qualified ordinary Meganeura2c42b448/Blade56f0565/native
0f26ee63:21 guards,2,824 upstream comparisons, three train/frozen smokes and
exact first outputs/272 metric reports/346 final tensors pass. Updates remain
107.93→108.30ms; no speedup. Search's original logical-memory preflight fails;
one declared correction stops at value/bonus numerical tolerances. Both failed
guards/source bundles remain, with zero search optimizer updates/game actions;
ten total warnings and no GPU recovery. The test-only hook is removed. No
further search, tolerance relaxation or rejected-factorization retry.
**Pong2M budget study is stopped for the pivot:** the
[fixed declaration](experiments/2026-10-11-cdp-pong-budget.md) remains unchanged,
but its unstarted work is cancelled. Seed1009 runs October11 18:56:09–20:25:19
UTC and saves a partial checkpoint at189,576 actions/47,333 updates, zero debt.
No frozen evaluation runs;2017/3019 never start. The [stop receipt](results/2026-10-11-cdp-pong-stop.json)
retains counters, checkpoint identity and interruption limits. The service has
no workers or automatic restart. Its interrupted guard has no terminal verdict;
no new allocation warnings are logged. Keep the20.45 historical target and
completed500k controls; partial online progress is not a new competence result.
Do not automatically extend all five games, resume an old lifetime or start a
capacity/representation sweep. All five long-run targets and the RGB budget
claim stay open.

[Seed1009's500k prefix](results/2026-10-09-cdp-qbert-prefix.md) now matches the
retained run exactly: initial346 tensors,62,500 batched transitions,124,939
non-timing learner reports and671 episodes, online last50 mean1,809. The
CPU-only review does not establish universal backend parity, pixel equality,
final500k tensor equality or later/frozen competence. The completed2M evidence
above is separate from that earlier prefix check.

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

These are current sticky/full-action centered CDP500k results, except Qbert's
separately allocated2M follow-up. All five titles retain12M/1M500k controls;
the qualified backend refresh prevents an exact same-binary capacity ablation.
This is not a comparison against raw CDP or RGB. Frozen means cover three learner seeds and24 natural
episodes/model; streams/episodes are not independent learner replicates.
The [combined capacity report](results/2026-10-09-cdp-freeway-capacity.md#complete-five-game-conclusion)
links every game's trained/initial-control videos and score/action/time curves.
Links into `runs/` require this workspace.

| Game | Current measured result | Unresolved question / historical gate | Evidence and whole videos |
| --- | --- | --- | --- |
| Seaquest | Three extrinsic CDP/RGB pairs: online543.6 vs318.1; CDP13.6% less wall time and35.9% less world-training time. Frozen world probes complete. | Controlled result on one title, not frozen policy mastery or five-game speed superiority. | [Learning and world report](results/2026-10-04-cdp-learning.md) |
| Boxing |12M frozen85.26 vs1.96 initial; seeds73.17/92.33/90.29,72/72 wins. Paired gain over1M+7.57 [−1.29,20.71]; training costs3.57x. | Both sizes pass historical gates in every seed; larger-model advantage is inconsistent. Published99.61 remains unmet. | [Capacity results and videos](results/2026-10-09-cdp-boxing-capacity.md) |
| Pong |12M frozen+12.40 vs−20.21 initial; seeds+7.21/+12.96/+17.04,69/72 wins. All beat retained1M controls at500k actions; training costs3.57x. | Only3019 passes historical>=90% wins/mean>=15. Published20.45 remains unmet;12M forecasts untested. | [Capacity results and videos](results/2026-10-08-cdp-pong-capacity.md); [earlier1M paired world report](results/2026-10-06-cdp-centered-world.md) |
| Freeway |12M frozen27.54 vs0; seeds33.58/17.25/31.79. Retained1M30.26; paired−2.72 [−13.29,4.08], training costs3.57x. | Crossing gates fall72/72 ->48/72;2017 regresses. No universal capacity advantage. Published33.40 remains unmet by the cohort. | [Capacity results and videos](results/2026-10-09-cdp-freeway-capacity.md); [exploration evidence](results/2026-10-05-freeway-effects-200k.md) |
| Breakout |12M frozen38.21 vs1.56 initial; seeds48.17/15.88/50.58. All beat retained1M4.00/3.50/7.92 at500k actions; training costs3.58x. | Published381.81 and historical two-wall/864-point gate remain unmet. Capacity helps;12M forecasts remain untested. Earlier weak-state diagnosis applies to1M, not automatically to12M. | [Capacity results and videos](results/2026-10-07-cdp-capacity.md); [1M world diagnosis](results/2026-10-07-cdp-breakout-world.md) |
| Qbert | At2M actions,12M frozen7,367.01 vs200.69 initial; seeds5,172.92/5,812.50/11,115.63. All beat retained500k scores; first pyramids20/22/20 versus0/0/24, training costs4.002x. | Published193,220.77 and historical>=90% first pyramids/mean>=15,000 remain unmet.3019 loses four first-pyramid successes despite its higher score. Useful h15 rewards and live values in all seeds, with calibration errors; survival remains unresolved. | [Budget results and videos](results/2026-10-11-cdp-qbert-budget.md); [forecasts](results/2026-10-11-cdp-qbert-forecasts.md); [values](results/2026-10-11-cdp-qbert-values.md); [retained500k capacity](results/2026-10-08-cdp-qbert-capacity.md) |

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
The completed1M five-game allocation measures29.70ms/update and127 actions/s,
8.46x aggregate /1.06x per-stream realtime. The12M capacity follow-up measures
110.03ms/update,55.4% imagination/29.9% world training/9.2% posterior, about
2.38x aggregate /.30x per-stream realtime. This is a capacity/cost tradeoff,
not an unchanged-learning slowdown or measured GPU utilization. The shared
exploration state-projection trial was rejected at its numerical gates before
timing; the qualified latest-backend control retains the original graph.

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
  Current qualified CDP/RGB backend pins are Meganeura31026833/Blade56f0565; the
  [October9 control refresh](results/2026-10-09-meganeura-refresh.md)
  passes scoped independent and production/restore checks, with no measured
  speedup or new learning comparison. Nativee9ebbf5d adds only the qualified
  read-only value probe; learning evidence retains7311547d. Tiny retains its
  historicalb684ffd9 qualification.
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
