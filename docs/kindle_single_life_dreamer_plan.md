# Kindle: a single actor that learns while acting

This is the authoritative plan. [Current evidence and archive](experiments/README.md)
retain experiments and failures; [AGENTS.md](../AGENTS.md) gives working rules.
Keep the runtime small, comparisons controlled and results reproducible.
For a quick overview of done/in-progress/next work and why progress is costly,
start with the [status dashboard](../STATUS.md).

## What exists

Native Rust/Meganeura/Blade implements a categorical Dreamer RSSM, sequence replay,
imagined actor/critic training and causal **LeVJEPA** perception. The current
303M video frontend is frozen, not end-to-end JEPA training and not DINO. DINO
remains a historical control, not an automatic fallback.

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
| Freeway | Assisted seed 0 pilots pass unassisted evaluation; three fresh roots fail: means 24.58/22.78/22.47, only 16/36, 3/36, 5/36 qualifying rounds | ≥20 natural rounds, ≥90% reach 25 crossings, mean ≥25, no cutoffs. Test exposure, not unchanged failed replicas. | [Successful pilot](../runs/freeway-persistence-learning-20260909.C0GoqT/hold64-evaluation.mp4), [failed confirmation](../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed1009-evaluation.mp4) |
| Breakout | Pilot mean 58.4583 versus .9655 control, but 0/24 two-wall completions | ≥20 completed episodes, ≥90% clear both walls / reach 864 points. Test minimal action vocabulary; retain reward and competence gates. | [Trained](../runs/atari-driver-continuation-20260911.LR9yT3/breakout-evaluation.mp4), [control](../runs/atari-driver-continuation-20260911.LR9yT3/breakout-untrained-evaluation.mp4) |
| Qbert | Pilot completes first pyramid in 17/24 episodes, mean 3,754.17; control 0/24, mean 125 | ≥20 episodes, ≥90% first-pyramid completion **and** mean ≥15,000. Test longer exposure and post-bonus coverage. | [Trained](../runs/atari-driver-continuation-20260911.LR9yT3/qbert-evaluation.mp4), [control](../runs/atari-driver-continuation-20260911.LR9yT3/qbert-untrained-evaluation.mp4) |

For Breakout/Qbert, a task completed before a later cutoff counts as achieved,
without relabeling that episode natural. Retain all episodes and partial tails.
Task observers are post-hoc evaluation, not privileged policy inputs or rewards.

## Immediate sequence

The [matched Pong campaign](../runs/pong-block-confirmation-20260916.rBwdGF/completed.json)
is complete. Together with Boxing this satisfies **two of five** game gates,
not general Atari competence. Preserve all completed writers and the fixed recipe.

1. Finish qualifying the prepared Breakout four-action adapter against the same
   eighteen-action runtime. Gradient and same-width complete-state checks pass;
   zero-update initialization/restore, N6 pixels, replay and memory remain. Then
   declare a matched learning pilot changing only action vocabulary. Do not
   repeat old backend qualification unchanged.
2. Separately test Freeway and Qbert with continuous 400,008-action pilots and
   retained 200,004-action midpoints. Prefer runner-owned numbered saves, not a
   watcher. The staged history option needs bounded default/history/restore
   checks. Keep Freeway's probability .5 / hold64 training assistance and 75,000
   unassisted frozen actions; Qbert remains unassisted with four episodes per
   stream and cap 600,000. Midpoint/final are one history, not independent roots.
3. Confirm successful changed recipes on all three fresh roots with controls.
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
303M video encoder beside the 12M learner. Selected ratio 256 consumes about
102 million replay positions during a 400k-action run. Repeated learning and
frontend cost must earn their place through sample-efficiency comparisons.

Qualified small-batch block products deliver **27.3% higher throughput** with
exact full-state/action parity. Root 2017's 400,008-action training takes **10.94h**,
10.155 actions/s, **.677× aggregate real time** (about .113× per stream).
Wall time is 65.6% learning, 33.7% observing and .45% emulator stepping. These
stage wall times are not GPU utilization or calibrated idle intervals.

Keep the active recipe fixed. Next, prioritize a matched replay-ratio ablation
and cheaper representation control before a larger model or more orchestration.
Historical R64 Boxing exceeds aggregate real time, but has only one successful
root and less score margin; it is not an adopted replacement. Retain AGC/full
recurrence unless an ablation supports changing them. Reconstruction/future
controls remain .25/0, .25/.25 and 0/.25. Backend fixes need numerical checks,
not weeks of blind training. Recheck upstream before diagnosing old bugs.

Uncapped step-driven playing/learning is supported; current R256 training is not
super-real-time. Free-running native gameplay without time control is required
but not validated by Atari. Measure arrival order, observation gaps, executed
action durations and training debt before introducing concurrency.

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
probe still needs current-source GPU forecast checks; CPU extraction is not that.

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
