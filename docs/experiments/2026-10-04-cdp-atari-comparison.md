# CDP, Dreamer RGB and Tiny JEPA on the selected Atari games

The active user goal is to train CDP on the selected Atari games and evaluate
against Dreamer RGB and Tiny JEPA. This takes priority over the unstarted
intrinsic-reward comparison. CDP remains the main architecture. This is a new
small-model comparison, never a restart of the cancelled historical12M queue.

## Scope and preparation

The user confirms the original five: **Boxing, Pong, Freeway, Breakout and
Qbert**. Execute Freeway first, then Boxing, Pong, Breakout and Qbert. Venture
and intrinsic-reward work are deferred. This is45 fresh training runs:
five games x three methods x three seeds, not just the initial Freeway block.

Use three learner seeds1009/2017/3019 and three methods: learned-CNN CDP,
learned-CNN RGB Dreamer, and pretrained causal Tiny JEPA with a frozen encoder.
No joint-Tiny/direct-policy arm, intrinsic reward, action aid, reward shaping,
new pretraining or hyperparameter sweep. Tiny means the qualified5.5M pretrained
encoder, not random initialization or303M Large. It received250k same-title
random-play observations from the original five games (45k train +5k validation
per game); disclose this extra experience. Venture was not in that corpus.

All methods use native-detail observations and the qualified Size1M/N8/B8/T16/
H15/R32/microbatch8/replay100000 recipe, full18 actions, sticky.25, repeat4,
no reset no-ops and100000-frame artificial cutoffs. The existing CDP split rates
and cosine loss, RGB reconstruction, and Tiny future-feature MSE are distinct
packages, not one-factor objective or pretraining ablations. Per-stream causal
histories remain independent. No RGB64-upscaled JEPA input.

The new study uses the unchanged native library
`32353ffb5d4516aa9281e94004f7b7ca2f126c9d29bfd62e33c36f78c44f502a`,
Meganeura13b19d33/Bladee349cddf. October4 upstream recheck finds main6268ea5/
e349cddf unchanged. Old five-game checkpoints, Phase2 Tiny/RGB and the current
Seaquest result retain their original scope; they are not fresh matched controls
for these new games. No repeated CDP/RGB numerical qualification is needed for
unchanged code; their independent references and completed learning already pass.

## Bounded Tiny qualification, declared before launch

One excluded Freeway production smoke: pretrained frozen Tiny, seed1009,
1,024 actions /195 updates, then a separate frozen restore smoke after review.
Each native process has a120-second deadline and ordinary host guards, with
**no new allocation-warning allowance**. Assert RTX5080 and >=2GiB sampled
Vulkan budget headroom. Verify complete counters, finite checkpoint, zero debt,
the exact pretrained encoder identity and no encoder training. Restore must
act without learner updates; saved model/optimizer tensors must stay unchanged.
These are extra diagnostic actions, not part of any learning curve.

Runtime root: `runs/cdp-atari-comparison-20261004.KnROnx/`.
Guarded persistent services use Restart=no, KillMode=control-group and a hard
deadline. Serialize all GPU work; review any failure before another launch.
No NVML polling, automatic retry, driver recovery or local native rebuild.

## Fixed learning and evaluation declaration

The small comparison is200,000 actual actions /49,939 updates per
game/method/seed, with a60-minute hard deadline per training process. Final
checkpoint only; no test-selected checkpoint or automatic budget extension.
Method order is CDP/RGB/Tiny for1009, RGB/Tiny/CDP for2017 and Tiny/CDP/RGB for3019
within each game. Train and audit each final checkpoint before evaluating it;
review the complete pair before proceeding. Publish complete online score-vs-actions/time
curves, final last50 completed-episode means, all tails, paired seed-bootstrap
intervals, total/world-update time and aggregate/per-stream real time.

Each final model receives frozen sampled-policy evaluation on eight streams,
with environment seed=1,000,000,000+learner seed+stream*1,000,003 modulo2^32.
Use the same pixel/sticky/action protocol. Stop once every stream completes
three episodes, or at200,000 actual actions, with a30-minute wall deadline.
Primary frozen scores use the first three completed episodes per stream;
report natural versus truncated episodes separately and retain excess episodes
and tails. A capped incomplete cohort has no complete-cohort score; do not hide
it or silently extend the budget. This is a learning comparison, not a historical
mastery/competence gate or an untrained-policy comparison.

Export frozen final state separately and compare all saved model/optimizer
tensors against the source checkpoint. The runner now explicitly declares this
diagnostic checkpoint export for episode-limited evaluation; it still makes
zero learner calls/updates. This Python-only change does not change learning
or require a native rebuild. CPU tests cover full and capped frozen exports.
Retain complete stream-zero videos by exact CPU environment replay after GPU
evaluation; recording is not a GPU throughput measurement. Audit real actions,
rewards, boundaries, resets, frame counts, source identities and zero updates.

`summarize_atari_comparison.py ROOT --game GAME --method METHOD --seed SEED
--training-only --output PATH` audits the training result; omit `--training-only`
for its frozen pair. With no game/method/seed, it aggregates finished pairs and
optionally writes `--plot PATH`. It launches no GPU work. Capped frozen replays
require explicit `replay_atari.py --allow-capped-evaluation`; their incomplete
cohort status remains unchanged. Summaries resample learner seeds, not episodes.

Report all attempts and failures. The older ten-run cap applied
to upstream replication, not an implicit limit on this newly requested study.
No upstream/JAX replication, exploration experiment or swarm is added here.

## Preparation result

Both Tiny smokes pass:1,024 training actions/195 updates, then1,024 frozen actions/
zero updates with all241 saved tensors unchanged. Both ordinary guards/seals,
counter ledgers and checkpoint-finiteness checks pass; no new kernel warning.
The2,048 extra diagnostic actions are excluded from learning/evaluation results.
All1,093 CPU Python tests pass, including complete and capped frozen exports.
The subsequent completion-audit/replay tooling passes1,104 tests. Its first test
run exposed eight old CLI test stubs that did not accept the new explicit capped
evaluation keyword; updating those stubs resolves the failures. No native code,
learner math, training budget or GPU qualification changed.

Before the study the local disk had1.1GB free. Approved cleanup removed only
the two git-ignored Rust incremental-build caches under target/debug and
python/target/debug; free space is now51GB. Source, binaries, checkpoints,
logs and other experiment evidence were not removed. The caches are rebuildable.

## First launch: stopped during initialization

Freeway CDP seed1009 stops before recording any actions or updates on another
exact NVIDIA allocation warning. No successor/retry is launched; **0/45 runs
complete**. The declaration and all budgets remain unchanged. The scoped
host-only review, retained incomplete broad snapshot and pending startup-only
warning-policy request are in the [incident report](../results/2026-10-04-cdp-atari-initialization-stop.md).

The user subsequently authorizes proceeding unless the GPU is wedged. This
supersedes the proposed startup-only/count-limited exception. New ordinary
declarations record standalone allocation warnings without aborting; hard
faults, failed native calls, numerical errors and deadlines still fail the job.
Resume in fresh `runs/cdp-atari-learning-20261004.XLgCAlkq/`, preserving the
failed initialization and two excluded smokes above. Learning/evaluation
budgets, methods, seeds, native library and acting code are unchanged. Successful
stages can advance after the existing CPU audits; failed stages stop for review.

`run_atari_comparison.py` executes the declared45 pairs in that order. Its
explicit `--first-training-service` handoff waits for the already-running
Freeway CDP1009 service to finish, then requires its full training audit; it
does not restore/resume training or run another GPU job concurrently. Each
subsequent stage is train -> audit -> frozen evaluation -> whole-stream video/
exact replay -> full pair audit. Only successful audited pairs advance. Failed
stages stop the controller with a retained result and no retry. Per-game
boundaries emit aggregate JSON/curves; partial individual pairs remain available.
The controller has a48-hour overall deadline, with the original60/30-minute
native deadlines unchanged. CPU analysis/replay uses one CPU, a2GiB address-space
limit and inherited zero swap. No local compilation or native/acting-code change.

## October4, 20:42 UTC: user-directed stop and diagnosis

The user redirects work to investigating the zero scores. Seven Freeway pairs
are complete; CDP seed3019 is interrupted at126,408 actions/31,541 updates.
The partial checkpoint and accounting pass CPU audits but do not complete its
training budget. The remaining37 entries are unstarted. The controller and
native workers are stopped; its interrupted exit is not a GPU fault. No matrix
restart, replacement seed or budget extension is automatic.

The [diagnosis](../results/2026-10-04-freeway-zero-reward-diagnosis.md) finds
zero discovered rewards and zero policy advantages. Independent raw-ALE
controls and an excluded, bounded1,024-action/195-update GPU reward pulse pass.
Keep all completed/interrupted evidence and the original protocol; decide a
small exploration experiment before more large-scale comparisons. The five-game
objective remains incomplete.
