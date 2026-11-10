# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md` when
available. Only the user merges PRs; commits and pushes are allowed.

## Start here

- [Machine handover](docs/handover.md): source branches, build instructions,
  shared checkpoint/evidence package and unfinished work.
- [Roadmap](docs/kindle_single_life_dreamer_plan.md): results, priorities and gates.
- [PR31](https://github.com/kvark/kindle/pull/31): live status dashboard. Update its
  description when status changes; do not create a separate STATUS.md.
- [Learning diagnosis](docs/results/2026-10-29-cdp-learning-diagnosis.md):
  representation, forecasting, actor/critic and training-coverage evidence.
- [GPU incident](docs/results/2026-10-23-cdp-context64-gpu-fault.md):
  initialization failures on the original machine and their limits.

## Current work — November 10

Kindle now builds against **upstream Meganeura main** (`a0eff600`, Blade
`d9112e26`) instead of the October handover fork; see the
[switch report](docs/results/2026-11-10-meganeura-main-switch.md). The fork's
`MEGANEURA_COMMAND_CACHE`/`IMAGE_VIEWS`/`CONVOLUTION_*`/`COPY_COMPOSITION` flags
are inert. A one-method upstream patch (`Session::slot_buffer`) waits for the
user to push from `/x/Temp/kindle-meganeura-slot-buffer-20261110/`; until then
the retained-off actor-dynamics/expected-value snapshots cross the host.
Run every heavy job as a memory-capped `systemd-run --user` unit; a bare
`cargo test` peaked at 17.6 GB and the OOM killer took the session down.

The goal **prove the world model training on Breakout** is active and has
its first learner that beats the previous best at every horizon. Three
feedback-side screens ([16x more play](docs/results/2026-11-10-breakout-world-scale.md),
[96x96 frames](docs/results/2026-11-10-breakout-world-rgb96.md),
[palette-committed feedback](docs/results/2026-11-10-breakout-world-palette-feedback.md))
left eight-step development ball tracking at 38/107, 36/112 and 23/112 against
the 64x64 control's 53/107, and a [linear state probe](docs/results/2026-11-10-breakout-world-state-probe.md)
showed why: the spatial-history recipes never encode the ball in the recurrent
state, only in fed-back frames that fade. [Removing the image history](docs/results/2026-11-10-breakout-world-state-only.md)
makes the state learn the ball (prior-state readout 14/7 native pixels at one
step, 20/11 at sixteen, baselines 40/28); the 1M state-only recipe at 96x96
for 65,536 updates (74 minutes of native calls, 68 ms per update) reaches
observed 113/113, one-step 109/112, four-step 93/112, eight-step 70/112,
sixteen-step 46/115 and thirty-two-step 15/102 on the shared development
origins, against the control's 101/66/53/25/11, and predicts near-term brick
hits where the spatial recipes never did. Its bricks blur (balanced error
.32); a [fixed observed context frame](docs/results/2026-11-10-breakout-world-context-frame.md)
(`rgb_context_lag` 8) keeps the scene sharp with the ball still in the state
(1M, 16,384 updates: four-step 72/112, eight-step 32/112, brick error .011;
4M at 16,384: four-step 74/112, eight-step 30/112, brick error .010, its 65k
continuation stopped at that point and reproducible from `queue-96-context-4m.json`).
Whole agents can now train this objective: `world_pixel_loss` and
`world_prediction_length` (plain learned-RGB worlds, full BPTT) with runner
flags `--world-pixel-loss`/`--world-prediction-length`. The first
[Breakout agent with the state-only 96x96 world objective](docs/results/2026-11-10-breakout-agent-world96.md)
(seed 1009, 200k actions, 4M/96/B8/T32/R16, motion cross-entropy, 16-step
rollout, retained guidance, no CDP and no exploration bonus; 12,469 updates
in 31 minutes) is a negative result: frozen sampled 4.760 / greedy 4.785 on
cohort 4000022003 (initial 1.250 / 2.167) against retained CDP 1009's 94.625
greedy / 70.500 sampled and fresh CDP 22003's 75.250 / 44.708, with guided
online returns averaging 34.5 that the frozen policy does not carry. That
run was not matched: it also changed model size, resolution, batch geometry,
guidance labels, actor unimix, the dynamics rate and the replay-value scale
and dropped the action-effects bonus (protocol correction recorded). The
[matched follow-up](docs/results/2026-11-10-breakout-agent-world96-matched.md)
keeps the retained 12M/B16/T16 recipe (bonus, corrective labels, unimix,
rates) and swaps only the world objective (motion cross-entropy, 8-step prior
image rollout; runner now allows the bonus with a pixel world). Frozen on
cohort 4000022003: **64x64 greedy 151.080 / sampled 40.286** (12,477 updates
in 33 minutes, 143 ms per update, VRAM 9.5 GiB) and 96x96 greedy 114.625 /
sampled 38.231 (54 minutes, 17.6 GiB), against the retained CDP learner's
94.625 / 70.500 on the same cohort and initial controls near 2. The pixel
objective wins greedy by a wide margin and loses sampled; one seed, no
held-out qualification, 0 two-wall episodes. Root
`/x/Temp/kindle-breakout-world96-matched-20261110` (unit finished).
[Retained CDP plus the pixel objective as an auxiliary](docs/results/2026-11-10-breakout-agent-cdp-pixel-aux.md)
(`world_pixel_scale`, `--world-pixel-scale`) scores greedy 104.036 / sampled
61.692 at scale 1 (the most balanced learner; CDP loss still falls to .056)
and 68.292 / 47.667 at scale .1 (the objectives interfere); root
`/x/Temp/kindle-breakout-cdp-pixel-aux-20261110` (unit finished). Every one
of these is seed 1009 on cohort 4000022003; a second-seed confirmation
(seed 22003, against fresh CDP 22003's 75.250 / 44.708) is the next run.
First-run root
`/x/Temp/kindle-breakout-world96-agent-20261110` (unit finished). World-model root `/x/Temp/kindle-world-scale-20261110`; driver
`python/examples/breakout_world_scale.py` (`probe` mode runs the state readout
on any checkpoint). These are development screens on reused seed 41001 and
reused origins, not held-out qualification. CDP and the five-game campaign
stay parked.

The previous [cumulative-survival learner](docs/results/2026-11-10-breakout-world-survival-loss-learning.md)
remains frozen by systemd at the user's request at update 9,964/16,384
(`kindle-world-survival-loss-learning-20261110.service`, PID 2287733); it holds
about 1 GB of host memory and 0.5 GB of VRAM. Do not thaw, restart or stop it
without the user. Its fixed protocol, qualification and the closed RGB-event
reviews stay as recorded in their reports; its runtime is the fork's.

## Direction and retained results

Training is the priority. Backend optimization,upstream porting and JAX parity
are parked. The retained agent uses **centered CDP** with a jointly learned CNN,
categorical RSSM and imagined actor/critic. The isolated-world track
uses explicit pixel supervision; the denoiser still fails physics gates, with
bounded reward support retained as a development improvement. No pretrained DINO or Tiny
JEPA encoder is required. Explicitly select the intended objective; learned RGB
remains the CLI default.

The priority is compute-constrained success on this game set. Game-specific
model sizes,resolutions and learning settings are allowed; one universal recipe
is not required. Prefer useful frozen learning per GPU-hour. Start fromT16;
T64 is an experiment,not the default. Native32/48/64 geometry is supported.
Size1M works well for Boxing/Freeway but was much weaker for Breakout/Qbert.
Changing several settings to find a cheap useful recipe is allowed; disclose
them without claiming isolated causes or broad reliability.

Work on `phase2-levjepa-evaluation` and PR31. The branch includes the qualified
c8aeba28 implementation with portable dependencies. Do not switch to a separate
takeover branch or modify neighboring worktrees. Keep `runs/`,large data and
archived experiment ledgers out of Git.

The **five-game goal is unfinished**:

| Game | Retained evidence | Remaining limitation |
|---|---|---|
| [Boxing](docs/results/2026-10-25-cdp-cheap-boxing.md) |1M32:76.833/79.167,two learners with24/24 wins,in28m22s/28m13s|Both pass; broad reliability open|
| [Freeway](docs/results/2026-10-25-cdp-cheap-freeway.md) |Aided1M32:31.042/32.000,two learners with24/24 successes,in28m17s/28m50s|Aid off during evaluation; broad reliability open|
| [Pong](docs/results/2026-10-25-cdp-cheap-pong.md) |12M48/B16/R16:greedy15.208 on both learners,24/24 wins,in41m28s/41m31s|Greedy passes narrowly; sampled passes one of two|
| [Breakout](docs/results/2026-11-02-cdp-breakout-replay-value1.md) | Retain 200k sampled70.750/greedy143.667 in21m28s; [400k budget point](docs/results/2026-11-03-cdp-breakout-budget400k.md) scores92.792/173.708 in43m37s | [New cohort](docs/results/2026-11-06-cdp-breakout-seed22003.md):greedy94.625/75.250 andsampled70.500/44.708 on1009/22003; material seed/cohort variation. All0/24 two-wall; original864/90% gate unfinished |
| [Qbert](docs/results/2026-11-02-cdp-qbert-r20-seed18013.md) |Fixed240k/R20:greedy15,279.167/15,931.250 on16001/18013,24/24 first pyramids each,in25m58s/26m23s|One passing development learner andone fresh confirmation;sampled12,972.917/14,976.042 both fail15k. Earlier R24 failure remains; broad reliability open|

Published long-run scores are aspirations,not matched-budget comparisons.
Finite losses,short runs and backend speed do not establish mastery or CDP
superiority. Guided online returns and replay imitation losses are not frozen
competence. Keep detailed results in the roadmap and linked reports,not a
growing history in this file.

Useful Breakout diagnostic references:
- [Value consistency](docs/results/2026-11-02-cdp-breakout-value-consistency.md):
  sampled one-step RMSE.732/.941 versus complete-return6.682/6.691. Short targets
  use the same critic's bootstrap; terminal slices are outcome-conditioned.
  All32 diagnostic/73 source/912 active pins pass;6.366s CPU-only,original Qbert
  identity preserved. Import failure/repair retained; no new loss orlearner.
- [Encoder4e-5](docs/results/2026-11-01-cdp-breakout-encoder4e5.md):
  sampled/greedy56.292/104.667 versus25.292/42.625 at2e-5;21m28s versus22m27s.
  All227 pins,recovery,initial/frozen tensors,replays,guards and original
  services pass. Retain4e-5; earlier4M/expert10 rate failure remains relevant.
- [Representation and own-policy replay](docs/results/2026-11-01-cdp-encoder4e5-representation.md):
  ball/motion readouts improve but remain less precise than pixel diagnostics.
  All replacement action readouts trail the learned actor. Own episodes last
  25.36% longer and earn95.88% more reward/action; state distributions differ.
- [Teacher takeovers](docs/results/2026-11-01-cdp-breakout-recoverability.md):
  earlier control avoids108/120 recorded losses versus26/120 at the last
  near-descending state. This supports earlier-control investigation,not exact
  causal error attribution or optimal irrecoverability. CPU-only,293 pins pass.
- [Delayed-takeover states](docs/results/2026-11-01-cdp-breakout-takeover-states.md):
  onlytwo of21 earlier-only rescues project contact withinfive actions at delay16;
  14 already match the teacher command. Timing estimates/imperfect teacher do not
  isolate the cause. All480 joins/478 saved-branch recounts,nine inputs and524
  active Qbert pins pass;CPU-only2.766s,complete15:23:52 UTC November1. Do not rerun.

Completed negative results and protocol failures remain retained. Do not restart
completed learners,old queues,failed assessments or collectors; no unchanged
long extensions,broad seed campaigns or automatic successors. Consult the linked
protocol before a separately justified new experiment. Preserve original failures
when a separate CPU-only repair completes missing evidence; never relabel a
protocol-failed comparison as passed. Single-actor learning precedes swarms or
concurrent learners.

## Runtime and precision

Use **Auto** on the current AMD RX7900XT/RADV. BF16 and
`KINDLE_DEVICE_TARGETS` remain unavailable. Three paired early updates leave the
first reward projection unchanged underF16 while Auto changes it; derivatives
and mechanism remain unmeasured. Completed F16/Auto B2/T64 lives do not justify
extensions. [Precision evidence](docs/results/2026-10-25-cdp-amd-precision.md).

Current retained learning uses native447fe50/SHA256 prefix35ad0db and the exact
matching runner. Breakout retains `/x/Temp/kindle-breakout-corrective-20261028/runtime`;
current Qbert uses `/x/Temp/kindle-qbert-color-masks-20261101/runtime` with the
qualified faster,decision-equivalent teacher.
Each declaration pins exact native,runner,teacher and source bytes. Optional
posterior,temperature,centroid,open-loop and imitation-decay runtimes have
separate receipts; do not mix overlays. Preserve source-specific validation
failures and limits. No rebuild is needed for an exposed learning-rate change.

## Experiment practice

- Treat every declaration-pinned script as immutable, including CPU analysis.
  Write changes under a new filename and declare them separately.
- Use small development runs and meaningful checks; avoid turning preparation
  into another backend qualification project. Automate monitoring and audits.
- Keep batched inference with eight independent Atari environments. Report
  aggregate actions and emulator frames separately,plus wall-clock time.
- Retain config,seeds,source/native identity,curves and frozen natural-episode
  evaluations against actual-initial controls. Record aids,budgets,cutoffs and
  failures. Confirm promising changes across seeds before reliability claims;
  selected/reused development seeds are not held-out tests.
- Checkpoints omit replay,live belief,games and RNG. A restored extension is a
  warm start,not uninterrupted learning or an independent seed. Restore preserves
  architecture/geometry and rejects fresh-training overrides.
- Preserve native max-pooled frames and one GPU resize; verify small-object
  visibility. Do not reintroduce CPU downscale followed by upscale.
- Put compact results in docs/results and update PR31. Logs,weights,videos,
  build caches and profiling dumps belong outside Git. Keep useful checkpoint
  milestones; release disposable build caches after packaging.

## GPU operation

GPU work is authorized unless the device is genuinely faulted. Known allocation
warnings alone are not an approval gate. Use the existing host guard and stop on
real faults or numerical failures. No automatic driver reset/reload/reboot.
Separate NVML polling is unnecessary for the current experiment. Check live
process/service identity before work; an observer timeout is not a stopped job.

The current AMD host is separate from the original NVIDIA PMU incident. Older
595/615 drivers repeatedly halted during first-world initialization,before
checkpoint loading. An ASLR-disabled debugger success was not a fix; localization
does not establish a buffer,firmware or hardware cause. On that host,use the
prepared pre-submission capture after recovery,not another blind restore. The
ABI-specific GDB collector applies only to its original binary. On a new healthy
GPU,qualify bounded initialization/restore and record device/driver identity
without inheriting old boot assertions.

## Historical material

On this host, the original checkpoints remain in the verified local extraction
`/x/Temp/kindle-amd-20261025/handover`. The historical shared-transfer pathname
below is absent at the November10 cleanup check; see the cleanup report.

The shared transfer `/mnt/data/kindle-handover-20261025` contains the4096-action
context smoke and2M Breakout originals,matching all346 tensors. Other historical
game weights were not transferred. New AMD checkpoints remain locally retained.
Original archives moved to `/x/Artifacts/shared-drive-cleanup-20261025` on the
original machine; exact counters and paths are in the handover.

Old incident/optimization ledgers and detailed data remain on
`archive/phase2-before-cleanup-20261025`; current reports stay here.
Earlier expanded instructions remain in Git at34b1e74; the earlier experiment
ledger is preserved at52c2174 (and earlier113ecde/8b1ba49) and in the roadmap/reports. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
