# Kindle: a single actor that learns while acting

This is the authoritative plan. [Current evidence and archive](experiments/README.md)
retain experiments and failures; [AGENTS.md](../AGENTS.md) gives working rules.
Keep the runtime small, comparisons controlled and results reproducible.
For a quick overview of done/in-progress/next work and why progress is costly,
start with the [PR status dashboard](https://github.com/kvark/kindle/pull/29).

## What exists

Native Rust/Meganeura/Blade implements a categorical Dreamer RSSM, sequence replay,
imagined actor/critic training and causal **LeVJEPA** perception. The frontend is
frozen during gameplay, not end-to-end JEPA training and not DINO. DINO remains
a historical control, not an automatic fallback. The new experiments use a
separately pretrained **5.49M causal ViT-Tiny/16**. Native numerical,
optimizer/restore, streaming, fit and noncollapse checks pass; the first bounded
4,096-update pretraining run completes with verified state and encoder exports.
Frozen probes retain useful position features but mixed motion results. Bounded
actor integration and matched-order throughput pass. The Tiny Freeway pilot
passes, but Breakout regresses and fresh-root reliability is incomplete. Tiny stays
opt-in; the 303M frontend remains the default pending broader evidence.

```text
previous belief + executed action -> deterministic prior -> predicted features
                                             |
RGB history through now -> frozen encoder -> posterior
                                             |
                              reward / continuation / imagination
                                             |
                                       actor + critic
```

The predictor sees the prior, never the posterior containing its target. LeVJEPA
uses causal prefixes of 16-arrival chunks, projected to 7×7×64 features. Chunk
boundaries reset perception only; episode boundaries also reset belief. Six
streams share batched inference and one learner while retaining separate visual
caches, recurrent state, RNG and causal replay histories.

Core code: [agent](../kindle/src/dreamer/agent.rs),
[vector collection](../kindle/src/dreamer/agent/vector.rs),
[world model](../kindle/src/dreamer/world.rs),
[networks](../kindle/src/dreamer/networks.rs),
[behavior](../kindle/src/dreamer/behavior.rs),
[replay](../kindle/src/dreamer/replay.rs),
[LeVJEPA](../kindle/src/vision/levjepa.rs).

Use **adaptive execution** as shorthand; the established category is online RL.
Keep observing, acting and scheduled learning explicit. Frozen evaluation must
never update weights. No concurrent learner service is needed for this phase.

## Current game status

The five-game goal requires each of fresh model roots **1009/2017/3019** to pass
its final-policy gate and beat a separately restored untrained control. Videos
are whole stream-zero evaluations, including unfinished tails, not selected wins.
Full multi-stream evaluations determine results.

| Game | Measured result | Unchanged gate / next decision | Rollout |
| --- | --- | --- | --- |
| Boxing | Three roots pass: 123/123, 207/207, 51/51 wins; means +83.87/+90.58/+83.53; controls near zero | ≥20 natural matches, ≥90% wins, mean ≥+50, no cutoffs. Complete. | [1009](../runs/boxing-confirmation-20260910.hTEDcu/seed1009-evaluation.mp4), [2017](../runs/boxing-confirmation-20260910.hTEDcu/seed2017-evaluation.mp4), [3019](../runs/boxing-confirmation-20260910.hTEDcu/seed3019-evaluation.mp4) |
| Pong | Three fresh roots 2017/3019/1009 pass: 24/24, 23/24, 24/24 frozen wins; means +20.5417/+17.4583/+20.0833. Controls 0/76 combined; zero updates/cutoffs. | ≥20 natural matches, ≥90% wins, mean ≥+15, no cutoffs. Complete on the fixed recipe; cross-root state/replay/video audit passes. | [2017](../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-evaluation.mp4), [3019](../runs/pong-block-confirmation-20260916.rBwdGF/seed3019-evaluation.mp4), [1009](../runs/pong-block-confirmation-20260916.rBwdGF/seed1009-evaluation.mp4), [controls](experiments/README.md#current-pong-confirmation) |
| Freeway | Fresh Tiny roots1009/2017 pass: final36/36 each, means32.9167/31.6944, versus controls0/36 each, mean0. Both complete pair/state/replay/video audits pass. Seed3019 training is running: **2/3 fresh-root pairs pass**. Seed0 is not a reliability root; historical Large fresh roots failed. | ≥20 natural rounds, ≥90% reach 25 crossings, mean ≥25, no cutoffs. Finish3019 and the cross-root state comparison on the fixed recipe; preserve the interrupted2017 attempt separately. | [Seed1009 final](../runs/tiny-freeway-confirmation-20260922.tij9QW/seed1009/final.mp4), [control](../runs/tiny-freeway-confirmation-20260922.tij9QW/seed1009/untrained.mp4), [seed2017 final](../runs/tiny-freeway-seed2017-replacement-20260922.12z27y72/seed2017/final.mp4), [control](../runs/tiny-freeway-seed2017-replacement-20260922.12z27y72/seed2017/untrained.mp4), [pair reports](../runs/tiny-freeway-confirmation-20260922.tij9QW/results.md) |
| Breakout | Pretrained Tiny: mean10.9167 versus .9310 control. Own initial Tiny:12.4583 versus1.10. Large:30.7917 versus .9655. All trained evaluations have0/24 two-wall completions; zero frozen updates/cutoffs. | ≥20 completed episodes, ≥90% clear both walls / reach864 points. Inspect forecasts and improve the small-encoder recipe; no pretraining benefit demonstrated in one seed. Four-action arm remains held. | [Pretrained Tiny](../runs/levjepa-tiny-breakout-20260921.ghJPWG/evaluate.mp4), [its control](../runs/levjepa-tiny-breakout-20260921.ghJPWG/untrained.mp4), [pretraining ablation and videos](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md), [Large](../runs/breakout-action-pilot-20260920.kNeotb/results.md) |
| Qbert | Pilot completes first pyramid in 17/24 episodes, mean 3,754.17; control 0/24, mean 125 | ≥20 episodes, ≥90% first-pyramid completion **and** mean ≥15,000. Test longer exposure and post-bonus coverage. | [Trained](../runs/atari-driver-continuation-20260911.LR9yT3/qbert-evaluation.mp4), [control](../runs/atari-driver-continuation-20260911.LR9yT3/qbert-untrained-evaluation.mp4) |

For Breakout/Qbert, a task completed before a later cutoff counts as achieved,
without relabeling that episode natural. Retain all episodes and partial tails.
Task observers are post-hoc evaluation, not privileged policy inputs or rewards.

## Immediate sequence

The [matched Pong campaign](../runs/pong-block-confirmation-20260916.rBwdGF/completed.json)
is complete. Together with Boxing this satisfies **two of five** game gates,
not general Atari competence. Preserve all completed writers and the fixed recipe.

1. Preserve the completed [Breakout Large reference](../runs/breakout-action-pilot-20260920.kNeotb/results.md):
   seed zero, eighteen actions, 200,004 fresh interactions, frozen evaluation,
   fresh initialization and frozen untrained control. All four phases and their
   audits complete; the competence gate fails. The [four-action hold](../runs/breakout-action-pilot-20260920.kNeotb/a4-train/HOLD.md)
   supersedes the unstarted half of the original pilot; do not remove it or
   claim a complete two-width comparison. All existing runtime checks and fixed
   inputs remain valid. No repeated qualification for unchanged binaries.
2. The compact causal-video JEPA candidate is pretrained and its bounded actor
   integration passes. The [matched-order cost check](../runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md)
   passes: 26–27% less total time, with exact same-arm state/trace repeats.
   The [full-budget Tiny comparison](../runs/levjepa-tiny-breakout-20260921.ghJPWG/README.md)
   completes all phases: 200,004 actions / 49,652 updates in 4.108h, frozen mean
   10.9167 versus .9310 control, no two-wall completions. Complete state/moments,
   common initialization and all replays pass. Large's matched-budget mean is
   30.7917: the smaller package is faster but weaker, not an adopted replacement.
   The [pretraining ablation](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md)
   completes the same learner seed/budget with Tiny's own original encoder weights:
   frozen12.4583 versus1.10 control, still no two-wall completion. It shows no
   pretraining benefit in this pilot; the target remains trained causal-video JEPA.
   The [15-step world check](../runs/tiny-world-horizon15-20260921.bKmiUF/results.md)
   now completes with exact one-step overlap: feature/reward forecasts beat their
   baselines, continuation does not. Four own-policy matches do not explain the
   score gap. Keep the pretrained Tiny frontend fixed for Freeway confirmation,
   with native learning, 12M RSSM, R256, feature contract and vocabulary unchanged.
   Include offline experience; do not infer a capacity limit from different
   pretrained packages or reliability from one paired seed.
3. The [fixed-Tiny Freeway pilot](../runs/tiny-freeway-exposure-20260921.ejKgSH/results.md)
   completes: seed0, 400,008 actions/99,652 updates in 7.969h. Frozen midpoint and
   final both pass, with means 30.50/33.03 and 36/36 qualifying rounds each;
   the untrained-policy control has mean 0 and 0/36. All complete state, replay
   and video checks pass. Both checkpoints already pass, so extra exposure is
   not shown necessary; they are one history, not independent roots.
   **Next confirm fresh roots 1009/2017/3019**, holding the encoder/runtime and
   400,008-action primary budget fixed, with retained 200,004-action midpoints.
   Keep Freeway's probability .5/hold64 training assistance and 75,000 unassisted
   frozen actions per arm, plus separately restored untrained-policy controls.
   Every native phase needs a separate declaration and review. The
   [fresh-root confirmation](../runs/tiny-freeway-confirmation-20260922.tij9QW/results.md)
   completes roots1009/2017's full pair, state/replay/video checks. Seed3019
   training is running; its frozen/control pair and cross-root state comparison
   remain. Two roots are not three-root reliability.
   Preserve the [interrupted seed2017 attempt](../runs/freeway-guard-interruption-20260922.2xt6tnn_/README.md)
   at45,714 actions after loss of its host guard. The user approved one fresh
   full-budget [replacement](../runs/tiny-freeway-seed2017-replacement-20260922.12z27y72/results.md);
   none of the interrupted experience is restored or counted in its budget.
   Replacement training,midpoint,final and control checks pass. Each new guard
   is owned by a persistent systemd user service with group cleanup,
   bounded lifetime and no automatic restart. Keep the qualified package fixed;
   no rebuild, automatic successor or undeclared retry.
   Historical Large pilots also passed before fresh roots failed. The
   [runner-owned history option](../runs/tiny-checkpoint-history-20260921.kVSoYg/results.md)
   passes default/history/restore checks with exact full state and trajectories,
   all-stream replay and 768 Python CPU tests; the native package is unchanged.
4. Then test Qbert with a continuous 400,008-action pilot and 200,004 midpoint;
   it remains unassisted, with four frozen episodes per stream and cap 600,000.
   The [Tiny Qbert protocol and CPU input checks](../runs/tiny-qbert-exposure-preparation-20260922.9c53nmwp/README.md)
   are prepared; no GPU phase is declared or queued. Keep the trained Tiny
   frontend, unchanged runtime and full Qbert gate. Compare added exposure within
   this pilot; the older Large result is historical context, not a matched arm.
   Revisit Breakout's prepared four-action comparison if still needed. Confirm
   successful changed recipes on all three fresh roots with controls.
   Do not replicate failures merely to keep the device occupied.

Pong's fixed recipe is N6, 12M/F32, B16×T64, full BPTT64, world microbatch 16,
replay ratio 256, learning rate 4e-5 / warmup 1000, AGC .3, reconstruction 0/future .25,
extrinsic rewards only, no exploration overrides, 400,008 actions. Both frozen
arms use sampled actions, environment root 100000, four complete episodes per
stream, hard cap 600,000, zero updates. Cap exhaustion is incomplete, not success.
Derive update counts from the ledger, not another run's counters.

The terminal xPz5ud queue and its reserved output stay untouched. Qualified native
`886bae68`, Meganeura `589d73ab`, Blade `2accfeee` and adapter `8dc0b98` stay fixed
for the completed campaign. The default editable extension is historical: select
the declared source-matched package. Cleanup does not change an active binary.

## Complexity and compute: what to change

Dreamer needs several interacting networks and a recurrent training loop, but
286 commits and 13K lines of experiment reports are not architectural necessities.
Keep investigations in the archive, production code for exercised features, and
compact evidence for decisions. Test coverage is not cruft merely because it
is larger than the implementation it protects.

This implementation also tests a costly departure from vanilla Dreamer: a frozen
303M video encoder beside the nominal 12M learner. Selected ratio 256 consumes about
102 million replay positions during a 400k-action run. Repeated learning and
frontend cost must earn their place through sample-efficiency comparisons.

Qualified small-batch block products deliver **27.3% higher throughput** with
exact full-state/action parity. Root 2017's 400,008-action training takes **10.94h**,
10.155 actions/s, **.677× aggregate real time** (about .113× per stream).
Wall time is 65.6% learning, 33.7% observing and .45% emulator stepping. These
stage wall times are not GPU utilization or calibrated idle intervals.

The new matched Tiny/Large AB/BA comparison reduces total time **26.0–26.9%**
and observation time **73.6–74.8%**, with exact same-arm state/trajectory repeats.
Tiny reaches only **.897–.902× aggregate real time** (~.150× per stream) at R256;
learning now accounts for about 87% of wall time. Mean learner step .257s includes
.092s world training and .086s imagination. More environment workers do not
directly remove this cost. [All four windows and stage clocks](../runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md).

Keep the active recipe fixed. The user-selected next change is the compact
causal encoder, followed separately by a matched replay-ratio ablation.
Historical R64 Boxing exceeds aggregate real time, but has only one successful
root and less score margin; it is not an adopted replacement. Retain AGC/full
recurrence unless an ablation supports changing them. Reconstruction/future
controls remain .25/0, .25/.25 and 0/.25. Backend fixes need numerical checks,
not weeks of blind training. Recheck upstream before diagnosing old bugs.

Uncapped step-driven playing/learning is supported; current R256 training is not
super-real-time. Free-running native gameplay without time control is required
but not validated by Atari. Measure arrival order, observation gaps, executed
action durations and training debt before introducing concurrency.

### Right-size the causal encoder

The [checkpoint accounting](../runs/model-sizing-20260920.kPIOWC/README.md)
finds **303,099,904** frontend parameters, **5,921,280** RSSM dynamics/prior/
posterior parameters and **10,281,233** optimizer-owned world+behavior parameters.
Our deterministic width 2048, hidden width 256 and 32×16 categorical state match
[DreamerV3's 12M preset](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml).
Its size labels refer to whole-agent presets, not RSSM counts. The disproportionate
component is the frontend, not an undersized accidental RSSM configuration.

Use **ViT-Tiny/16: 12 layers, width 192, three heads, MLP 768**, totaling
**5,486,592** encoder parameters. Preserve 224px inputs, 16-frame block-causal
chunks, independent stream histories and JL64/2×2 pooling to 7×7×64. Logical F32
KV storage falls from 588 to 55.125 MiB per stream (330.75 MiB for N6); these are
tensor sizes, not measured device peaks. Keep RSSM capacity and R256 fixed first.
Do not slice Large weights, substitute DINO/RGB or claim proportional speedup.

Tiny requires its own self-supervised video pretraining. Use the original
LeVJEPA multi-view invariance + SIGReg objective with causal token dropping and
an evaluation EMA, not an undisclosed teacher-distillation substitute. Declare
the corpus, train/held-out split, frame sampling, views, step budget, projector,
normalization and exported weight identity before training. Preserve original
token positions in masks/RoPE; patches cannot read the CLS sink or future frames.
Train on Meganeura/Blade; Python may prepare video batches and reference checks.

The [upstream recipe](https://github.com/MLO-lab/LeVJEPA) demonstrates Tiny
pretraining on unlabeled video; the inspected official released checkpoint is
Large. Native Tiny inference support passes CI. A separate
[native pretraining graph](../runs/levjepa-tiny-cpu-20260920.e4QkQH/README.md)
and native AdamW/GPU-EMA/save/restore/export paths pass 100 CPU tests. The
[independent reference and allocation preflight](../runs/levjepa-tiny-reference-20260920.XhB8T8/README.md)
cover every gradient and identify a 25.3→9.4 GiB B128 plan reduction from native
erf GELU. These are prepared numerical expectations and static allocations,
not GPU correctness, runtime fit, throughput or trained features. The native
Python interface and deterministic video adapter are staged. A separately
declared [fresh random-policy corpus](../runs/levjepa-tiny-atari-corpus-20260920.oTjdon/result.md)
contains 250,000 RGB64 observations / 999,112 emulator frames from the five games,
with whole-recording training/validation splits and no clips crossing resets.
The actual B128/V4 pilot measures **.45s native training + .80s batch preparation**
per step: data preparation is the measured bottleneck in this serial pipeline.
These are stage wall times, not GPU utilization. This is additional offline
experience, not reused evaluation
footage or an untrained gameplay control. Latest upstream RMSNorm fusion fixes
are included in staging. The [seven native GPU checks](../runs/levjepa-tiny-accuracy-20260921.qa3GqK/results.md)
now pass, including all 155 gradients and bitwise exact 763-tensor continuation.
Preserve the original device-selection/strict-trig failures and documented
primitive-bound revision; full-model and gameplay gates are unchanged. A
[32-update B128/V4 fit/timing pilot](../runs/levjepa-tiny-fit-20260921.oDmZVn/results.md)
and [frozen held-out noncollapse screen](../runs/levjepa-tiny-feature-check-20260921.wSQUJj/results.md)
complete. The first [fresh 4,096-update candidate](../runs/levjepa-tiny-pretrain-20260921.JaPZpW/results.md)
completes in 84.18 minutes: seed 743, B128/V4, peak LR 1e-4/warmup 128/cosine,
weight decay .04, EMA .99. All nine complete checkpoints and both encoder exports
verify. The separately declared [frozen quality comparison](../runs/levjepa-tiny-quality-20260921.ojeZgt/results.md)
completes against its own untrained export, with RGB/constant controls. All five
noncollapse screens pass; pooled Pong position mean R² improves .904→.938.
Motion is mixed: explicit-history mean R² falls .508→.333, with severe paddle
readout outliers despite lower median error. Preserve every target and the fixed
test split; do not retune this result away. These probes justify a bounded
downstream test, not adoption or a general JEPA advantage.
Gate adoption on numerical/causal and streaming/reset checks, held-out
feature variance/quality, measured N6 memory/time and frozen downstream learning
against its untrained control. Report all offline experience. If Tiny and Large
use different corpora, compare pretrained packages, not a pure size ablation.
Pretraining-only backend extensions do not automatically change the gameplay
runtime: Tiny exports use the existing frozen encoder operations. Prefer the
qualified gameplay backend for the first size comparison; if a newer runtime
is needed, qualify it separately and use it for both Large and Tiny controls.
The [prepared gameplay package](../runs/levjepa-tiny-gameplay-cpu-20260921.YF4BKS/README.md)
adds explicit Tiny selection and checkpoint identity on that unchanged backend.
All 754 Python and 93 Rust CPU checks pass. Its own
[dense-reference and exact N6/serial inference checks](../runs/levjepa-tiny-gameplay-gpu-20260921.X6TnAI/results.md)
now pass. [Bounded N6 pixel/save/restore](../runs/levjepa-tiny-gameplay-pixels-20260921.NQLh0I/results.md)
also passes: 3,840 actions / 611 updates per arm, exact frozen state and complete
replays; the new Large path matches its retained state and trajectories exactly.
Tiny selection is opt-in; Large remains the default until downstream evidence
supports replacement. Initial short-loop times are 194s Tiny versus 289s Large,
with observation cost 33s versus 125s and similar learner cost. These include
replay warmup and are not matched-order or steady-state throughput qualification.
The [post-warmup AB/BA check](../runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md)
now passes all cost and exact repeatability gates. Its terminal writers stay
unchanged. The full-budget Breakout pair completes, but Tiny's frozen mean
10.9167 regresses against Large's 30.7917. Keep Large as the default. A separate
same-architecture [encoder-initialization control](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md)
now completes: frozen mean12.4583 versus1.10 control, two-wall successes0/24
versus0/30. It exceeds pretrained Tiny by1.5417 points in this single paired
seed; no downstream pretraining benefit is demonstrated. This does not prove
a capacity limit, a reliable negative effect, or justify replacing the trained
causal-video JEPA target with random features.

## World-model evaluation and pretraining

Predict before each target observation in a recorded frozen match, then compare
features, reward and continuation with reality. Keep posterior estimates separate.
Use persistence, unrelated-action and zero-reward controls; report MAE **and** MSE,
positive/negative/terminal counts, and visual-cache/reset strata. Sparse all-frame
MAE or AUC alone cannot establish calibration. Another policy's logged return
is not an unbiased critic target.

Existing [own-policy forecasts](../runs/world-evaluation-20260908.Xzx3pN/report.html)
and [common-recording report](../runs/common-world-report-20260909.O7nqqe/report.html)
show action sensitivity but limited cross-trajectory generalization. This is not
a proven explanation for failed policies. For new Pong roots, preselect the first
four complete stream-zero matches without score filtering. The staged multi-match
probe needs current-source GPU forecast checks; CPU extraction alone is not that.
The [current-package readiness check](../runs/current-world-probe-cpu-20260921.KlMyMe/README.md)
passes 74 CPU tests and selects the first four Tiny trained/control matches.
Its [run-local wrapper](../runs/current-world-probe-wrapper-cpu-20260921.r8GrmS/README.md)
adds per-stage Vulkan memory checks and complete frozen-state comparison, with
39 CPU tests. The separate [one-step GPU report](../runs/tiny-world-one-step-20260921.U7yHOa/results.md)
now completes strict and forced replay: all1,126 actions/four matches, common
forecasts and input/reset identities match exactly; complete frozen state passes.
Scalar restore explicitly changes only collection-stream metadata6->1 and the
executed-action counter. Feature MSE .0001665 beats persistence .0002863; reward
MAE .00892 beats zero .03819. Continuation MSE .00439 is worse than always-continue
.00353, with only four terminals and22 positive rewards. These are useful
one-step measurements, not a causal explanation of the score gap. The separate
[15-step/all-origin check](../runs/tiny-world-horizon15-20260921.bKmiUF/results.md)
also passes, with 16,470 forecast targets, exact one-step overlap and unchanged
complete state. At horizon15, feature MSE is .0001860 versus .0010262 persistence;
reward MAE/MSE are .008938/.019244 versus .040187/.118692 zero prediction.
Continuation MSE remains worse (.005106 versus .003716). There are only22
positive rewards and four terminal targets at each horizon, not independent
new events. Recorded future actions condition these prior forecasts; this is
not counterfactual or imagined-policy validation. Preserve both terminal checks.
No current evidence justifies another encoder redesign before fresh-seed
Freeway confirmation; a continuation ablation remains a distinct later option.

Pretrained visual weights are supported; a video-dataset world-pretraining
workflow is not adopted. Start with aligned RGB, executed actions/durations and
boundaries from mind-games. Missing actions/rewards are missing labels, not
NOOP/zero. Compare fresh, encoder-only and encoder+world initialization at equal
target budgets. Keep actor/critic unchanged in world-only updates; declare resets
and offline lineage. Useful pretraining means faster retained gameplay learning,
not just lower feature error. No perception expansion before calibration/coverage.

## Beyond five Atari games

Beating most of a predeclared Atari suite is an ambition, not a consequence of
using Dreamer. Our variant does not inherit published DreamerV3 scores. Extend
to Seaquest/Frostbite/Private Eye, then Atari-26 with explicit gates, budgets and
seed distributions. Independent per-game training tests algorithm breadth, not
one transferable policy. Keep the pinned local upstream control and disclose
representation/precision/protocol differences.

Use `/x/Code/mind-games` for launch, time control, capture and input; recheck its
current Kindle API. Prefer **vkQuake2** next, with vkQuake only an integration
reference; then **TMNF** and a small **GOG/Wine** panel. Implement a small RGB8 /
executed-action / reward / boundary adapter, not another training stack. Measure
kills/objectives, track finishes and game completion, not motion alone. Explicit
sparse rewards and documented guidance remain acceptable. Menus, startup scripts
and overrides must not masquerade as autonomous learning.

Reserve an unseen shooter before tuning a general FPS actor. Compare fresh
dynamics/policy, transferred dynamics with fresh policy, and transferred dynamics
**plus policy** at matched budgets. Declare action mappings and optimizer,
normalizer, replay and belief resets. Measure zero-shot play, fixed-budget
adaptation and source-game forgetting; a held-out map is not a held-out title.

Only after strong multi-seed results on at least three GOG titles across two
genres and held-out cross-title adaptation/retention should two independent
Kindles share immutable experience chunks. Natural deaths/respawns are allowed;
cloning/rewinding a live game for training is not. Swarms and shared optimizers
remain later work. Intrinsic reward stays behind its existing seam, off in
controls, with extrinsic-only comparisons before adoption.

## Execution and evidence

Use the GPU; **NVML is temporarily disabled**. Serialize bounded direct native
jobs under the [host-only guard](gpu_incident_response.md), with actual device
assertions and >=2 GiB sampled Vulkan budget headroom. No recovery operation,
blind retry or quarantined candidate reuse. The four old Xid incidents remain
unexplained. Passing driver 580/no-NVML jobs is not a causal fix or safety proof.

Preserve declarations, failures and artifacts. Do not grow a new framework for
each check or rebuild qualified binaries for unchanged code. CI must install
declared test dependencies in a clean environment. Checkpoints preserve weights/
moments, not replay/RNG/live belief; interrupted resume is not equivalent lifetime
continuation. Atomic complete-state recovery and bounded storage remain later work.
