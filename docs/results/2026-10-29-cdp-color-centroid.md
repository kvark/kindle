# Native color-position supervision: learned target, unresolved gameplay benefit

The optional position loss passes numerical and bounded learning checks. A
subsequent fresh **200,000-action** Breakout learner finishes in **21m45s**, but
its full evaluation **fails the declared zero-truncation requirement**. Sampled
score is 14.375 versus control 13.208; the selected natural greedy cohort scores
20.125 versus 17.875, with one additional 25,000-action timeout. Both have zero
of 24 two-wall successes. These descriptive changes do not establish a gameplay
benefit or justify promoting the recipe. The original failed service and all
saved evidence remain; separate CPU closure finishes without repeating GPU work.

The earlier bounded learner completes **8,192 actions / 32,735 frames / 489
updates in 48.101s**. Its 1,024-action / 4,096-frame greedy frozen check takes
0.875s, makes zero updates and completes no natural episodes. That check's
experience is not reused by the full learner.

## Why this objective

The [fixed pixel and feature diagnostics](2026-10-29-cdp-breakout-representation.md#nonlinear-state-readouts-still-leave-substantial-position-error)
show precise ball positions available in RGB64 pixels, while tested linear and
nonlinear learned-feature readouts retain roughly 10–13 source pixels of error.
The [H5/R32 comparison](2026-10-29-cdp-breakout-h5-r32.md) lowers training imitation
error without improving frozen play. These motivate a direct position-learning
signal; they do not prove that position error causes the policy's weakness.

Keep centered CDP, the learned CNN, categorical recurrent world model and
imagined actor/critic. From each current replay image, the native GPU graph
computes red excess `max(R - 1.5 * max(G, B), 0)` in the fixed source-coordinate
region x[8,152), y[90,185). A detached normalized centroid supplies two targets.
Rows with mass at most 1e-6 contribute exactly zero. A training-only head with
one 256-unit hidden layer predicts those coordinates from the posterior state;
the squared-error loss trains the head and CNN/recurrent model. Its fixed weight
is 100, chosen before the bounded learner runs. The raw loss and valid fraction
are reported separately.

This adds an explicit game-specific color/region prior. No position input goes
to the policy, no emulator RAM supplies a learning target, and no CPU image
resize or inference side channel is added. Native max-pooled frames still receive
one GPU resize. The target and prediction refer to the same replay row. The
option defaults off; existing inference graphs omit its head. Fresh-run CLI
`--breakout-centroid-scale` rejects restore overrides and unsupported settings.

## Bounded learner and accounting

The recipe retains 12M/RGB64/Auto/N8/B16/T16/H15/R16, full18 actions, expert scale
100 with world gradients and corrective labels, all action weights 1, original
rates, seed1009, evaluation4000001009, 20% guided episodes and cap1920. The new
centroid head adds **656,386 trainable parameters**, bringing the total to
**16,990,739**. It adds five parameter tensors and ten Adam-moment tensors, so
the checkpoint contains **372 tensors**, versus the control's 357.

All 357 shared initial tensors and 47 pre-update semantic transition events
match the retained control. Every report is finite. The new hidden/output
weights and existing CNN, reward, actor and value weights change. All 372
tensors remain identical through frozen evaluation. Mean raw centroid loss is
0.098275, range 0.035631–0.498701; mean valid fraction is 0.816470, range
0.578125–0.964844. These are losses on sampled training replay, not held-out
position accuracy or evidence of task competence.

Training overrides 3,616 actions (44.140625%) and labels another 4,576 passively.
All 489 reports have expert supervision, with 117,028 labeled replay rows.
Complete recorded pixel/action/label replay and both host guards pass. Guided
online returns do not measure unaided competence. The frozen budget has no
completed natural episodes.

The earlier retained bounded control takes 47.623s at the same action/update
budget; this check takes 48.101s. Those are individual early timings, with a
different native/runner version and later state trajectories. They do not
isolate overhead or establish full-run speed. Linear extrapolation gives
19m34s per 200k actions, versus the control's measured full-run 22m19s, which
also illustrates why the early projection is not a guarantee.

## Validation and preserved failures

CPU checks pass 139 Kindle and five kindle-gym Rust tests, strict workspace and
binding Clippy, and 683 tests against the packaged Python runtime. Ten native
construction cases are deselected; the CPU suite attempts zero native
constructions. The first Rust test compilation fails on an inferred fixture
index type, then passes after explicit usize indices. Original logs remain.

Three isolated numerical fixture executions are recorded:

1. The first fails after 0.72s because the test uses `read_output(length)` as an
   output-index API. No numerical comparison completes.
2. Explicit indexed readback fixes that test defect. Targets, masks, losses and
   derivatives pass, but an exact-equality assertion rejects valid fraction
   0.49999997 instead of 0.5 after 0.72s.
3. A one-F32-epsilon tolerance for that mean passes. Per-row masks and absent
   gradients still require exact zeros; all original F64 target/loss/derivative
   tolerances, detached-target and unchanged-parameter checks remain. Production
   target/loss source is byte-identical across all three fixtures. All 83 pins
   and the final guard pass; neither failed check is relabeled as passing.

The bounded service starts **20:41:27 UTC October29**, source `3f31c99`. Both
GPU stages pass, but the CPU checkpoint audit fails at **20:42:25** because its
expected schema omits the ten new optimizer-moment tensors. A separately pinned
CPU-only repair adds their exact names/shapes and retains all original checks.
It completes **20:43:45**, preserving all 41 original and 21 repair pins. The
original service remains failed; completion status is
`complete_with_cpu_audit_repair`. No training or frozen stage repeats, and no
host reset/reload/reboot occurs. All evidence and original failures remain.

The new native source is `2974160`, SHA256
`e7ba0d4b208fa15be5790615034fb36ab727e4bd02b041114775b42b8e888e90`.
Runtime: `/x/Temp/kindle-color-centroid-20261029/runtime`. The retained `447fe50`
runtime is unchanged. Matching initial tensors/traces does not establish full
default-learning parity across native versions. The 2.9GiB disposable build
cache is released; wheels, all validation binaries and evidence remain. The
separate cache-release receipt preserves the original pinned build receipt.

[Protocol](../experiments/2026-10-29-cdp-color-centroid.md).
Root: `/x/Temp/kindle-color-centroid-20261029`.
Bounded evidence: `native-check/breakout-centroid100-check`.

- Numerical review: `eb0c4bef75bf820e895b32f1a16399faf8f732cdad9997a229b548a28ca9f136`.
- Bounded declaration: `1b14d88fd6b581676967ddef3a4fa1439c2fee07d8261fd33fd5612050cf95f5`.
- Repaired audit: `d57993217a59ab3feb91684a495c645d9dc0505ec8035c8e91382cc09df43d01`.
- Full training replay: `4cd348318e07402830112b718cbedeaa2730d25259393c4ddd49f1e52f146d74`.
- Completion: `f802657b468569d73aedc9b8ed7a9a5f3f28c3c52b33bf08e98d7161d530b7c3`.

No learner or GPU allocation is active at completion. The five-game goal remains
unfinished; check live state before any separately selected GPU work.

## Fresh 200k comparison selected

The [separate full-run protocol](../experiments/2026-10-29-cdp-breakout-centroid100-200k.md)
selected a fresh200k/12,477-update comparison under a30-minute training cap.
It retains the checked recipe and scale100,with100k recovery and fixed final/
actual-initial sampled and greedy24-episode assessments. No short-check experience
is reused. The original bounded outcome remains execution evidence only; no
full-run gameplay result is available at selection.

## Full 200k comparison fails the evaluation protocol

The fresh learner completes **200,000 actions / 799,500 frames / 12,477 updates
in 1,304.573s (21m45s)**. Retained control takes 1,339.198s (22m19s); this run
is 2.59% shorter. Native and runner versions, the extra head and encountered
trajectories differ, so this does not isolate a speedup or loss overhead. All
settings and the teacher remain as declared above; no checkpoint or scale is
chosen from intermediate results.

| Fixed evaluation | Own initial | Final | Retained control final | Final two-wall successes | Total final timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sampled actions | 1.417 | 14.375 | 13.208 | 0/24 | 0 |
| Greedy actions | 2.208 | 20.125 | 17.875 | 0/24 | 1 |

Means use the declared first three **natural** episodes per lane, 24 total.
The greedy score excludes its time-limited episode and cannot be treated as a
passing evaluation. Lane0/episode1 reaches 25,000 actions at score27, with a
20,819-action reward-free tail. Recorded actions include 16,555 NOOP,850 FIRE,
3,736 RIGHT and3,859 LEFT. Those counts do not identify why the policy stalled.
The assessment eventually records154 natural episodes and the timeout while
waiting for each lane's cohort. All four frozen arms total **244,312 actions /
976,892 frames / 172.214s**, with zero learner updates. The sampled arm meets
its own episode/accounting requirements; it does not retroactively pass the
four-arm declaration. No task gate passes.

Training overrides110,826 actions (**55.413%**) and passively labels89,174.
All12,477 reports are finite. Fixed50k-action windows show:

| Training action window | Mean raw position loss | Valid target fraction | Expert CE on labeled rows | Control expert CE |
| --- | ---: | ---: | ---: | ---: |
| 0–50k | .05547 | .8351 | 1.3583 | 1.3635 |
| 50–100k | .02095 | .8497 | .8687 | .8840 |
| 100–150k | .00758 | .8744 | .7535 | .7772 |
| 150–200k | .00349 | .8927 | .6837 | .6958 |

The raw auxiliary loss falls about94% between first and last windows. It is a
masked training-replay mean on changing states, not held-out coordinate error.
The direct target is learnable, but this alone does not show improved state
precision on the diagnostic cohort or better action choices near the paddle.
Expert fitting improves only modestly relative to the control. Guided selected
episodes average187.609, while unguided online episodes average5.519; these
changing-policy online groups are not substitutes for frozen competence.

### Validation, failure and closure

The original service starts **20:53:05 UTC October29**, source`b58d3a2`, with187
pins. Startup verifies372 finite initial tensors matching the bounded learner
and the correct first update at376 actions. The100k/6,227 recovery passes at
**21:04:40**. All five GPU stages finish with passing host guards; all372
tensors remain invariant through each frozen arm. All357 shared control initial
tensors match, and sampled/greedy actual-initial semantic traces reproduce all
800/795 events respectively. Auxiliary,CNN,reward,actor and value weights change.

Main fails at **21:18:18** when its CPU audit enforces the original
`truncated_episodes == 0` requirement for final greedy. Automatic postprocessing
records incomplete status at21:18:35; automatic collection preserves the failed
main and terminal observers at21:18:40. This is a protocol failure, not a GPU
fault or a CPU schema defect that can turn the comparison into a pass.

Separately pinned CPU-only closure starts21:20:54, preserves all187 original
and234 closure pins, reuses the two completed sampled replays, and finishes the
two missing greedy replays/videos and complete training pixel/action/label replay.
It verifies all372 tensors, numerical/config/counter/source checks and guards.
Audit completes21:24:43; exposure,curve and descriptive comparison complete
**21:25:40**,285.936s after closure selection. Final collection at **21:31:01**
checks the five postprocessing pins and all nine retained-control completion
files, including its exposure summary. The curve is visually inspected. No
native agent or environment is constructed by the closure collector, no GPU
stage repeats, and no host reset/reload/reboot occurs.

Closure status remains **`closed_protocol_failed`**, with
`comparison_accepted: false`. No passing original learning,postprocessing or
completion review is created. Ordinary frozen logs lack live pixel hashes;
replays verify the saved outcome/action/frame/reset trajectories. The original
failed service remains failed, all observers are terminal and no GPU allocation
is active at collection. Reused seed/cohort, explicit color/region priors and
changed native/runner limit causal and reliability claims.

Retain the control. No retry, cap increase, extension, confirmation or automatic
successor follows this life. A useful remaining diagnostic is whether the new
checkpoint improves held-out position precision and urgent action choices on
matched trajectories; the training losses above do not answer that question.
The five-game goal remains unfinished.

Full root:`/x/Temp/kindle-color-centroid-20261029/full`.
Trial:`breakout-12m-64-b16-r16-centroid100-expert100-200k-seed1009`.
Separate closure artifacts live in its`closure/`directory.

- Original declaration:`21c5a539eb81847da6cd10dbe16c1bf9a3889fae456af22ee899b51d4eb32df7`.
- Evidence review:`9a6d44497bc7fbc476e69fb220f0897bcde204075b5c9719d58db0878526bd34`.
- Descriptive comparison:`cb10634332f62e8567f0c1273a6fa75477d68f3db6f1ad22e897b6a76a97ae9e`.
- Closure completion:`41ce6517c95c9e70aed1920dd21ed3a450936fd9e578b0b33e8c978fb1294a37`.

## Subsequent held-out feature readouts

The [separate frozen diagnostic](2026-10-29-cdp-centroid-representation.md)
compares this final checkpoint on the exact saved control trajectories.
Recurrent linear ball x/y MAE improves12.878/10.918→9.267/8.064 source pixels
on555 shared rows, while near-paddle response-target MAE improves14.848→13.486.
Motion readouts also improve, but substantial position error remains and
nonlinear near-paddle target readouts do not improve. No new training or policy
evaluation occurs. This tests learned-feature accessibility, not the auxiliary
head itself or the actor's use of those features. The full learner's protocol
failure and lack of an established gameplay benefit remain unchanged.
