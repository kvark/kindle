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

## Current work — November 2

The [fresh Breakout H30/R12 comparison](docs/experiments/2026-11-02-cdp-breakout-h30-r12-200k.md) remains active after its
verified **100,000-action / 4,670-update recovery**. Original main912196 /
invocation5bfcdae0db9f4682a7e9e6c72c98adcd continues unchanged; postprocessing,
progress and collection observers remain active. Launch07:31:11.304 UTC
November2/source538ef64,422 pins,30-minute training cap. No frozen outcome yet.

The original recovery service succeeds **07:42:47.665**. The separate reader
passes once **07:43:12.544**, in1.285s (1.304s wrapper):422 pins,357 finite tensors,
four retained files/206,067,161 bytes,4,705 finite reports through100,752 actions,
matching configuration and original process identity. ConfirmationSHA327621dd.
Do not repeat startup, recovery, qualification or completed stages.

Train200k/9,358 updates, then four final/actual-initial24-episode sampled/greedy
arms against70.750/143.667. Sampled stays primary; the90%-two-wall gate stays
fixed. H15/R16→H30/R12 changes rollout length,cadence,sampling,warmup andrelative
work. Keep12M64/Auto/N8/B16/T16/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/
expert100,sampled collection,exact runnercf4ea238/native447fe50/teacher.
No isolated causal claim or experience reuse. The bounded check is closed:
8,192 actions/367 updates/51.413s,382 pins,conservative23m00.725s projection;
its frozen1,024 actions have zero updates/no natural episodes. Do not repeat it.

Full CPU qualification and startup pass; terminal readerSHAdfee5cc5 is qualified.
After all five original services succeed and completion exists, run
`review_full_terminal.py terminal` once via `run_cpu.py full-terminal`.
The saved-log coverage readerSHA7d346078 qualifies07:40:38.302 in2.428s:
six configuration cases,52 diagnostic/422 active pins; five old boundary cases
are retained without repetition. After the full terminal summary, run
`review_score_coverage.py terminal` once via `run_cpu.py score-coverage-terminal`.
Reuse the completed control analysis; no new gameplay or GPU work. Score
fractions do not establish walls or isolate a coverage/horizon mechanism.

Root `/x/Temp/kindle-breakout-h30-r12-20261102/full`; service
`kindle-breakout-h30-r12-200k-20261102`. Check live state; no overlap,restart,
retry,cap increase,extension,confirmation or automatic successor. Preserve CPU
co-host durations without correcting training time. DeclarationSHAaa57e4a5;
coverage declarationSHA9d98d514. Breakout and the five-game goal remain unfinished.

Earlier allocations are closed. The completed
[lower-dynamics comparison](docs/results/2026-11-02-cdp-breakout-value1-dynamics1e4.md)
regresses both Breakout scores; retain dynamics4e-4 with replay-value1 and
behavior8e-5. Qbert's fixed240k/R20 recipe passes on its development learner and
[one fresh greedy confirmation](docs/results/2026-11-02-cdp-qbert-r20-seed18013.md).
Breakout remains below the two-wall gate. Do not repeat completed learners,
checks, replays, observers or readers; no automatic successor is selected.
The roadmap and linked reports retain detailed outcomes, failures and limits.

## Direction and retained results

Training is the priority. Backend optimization,upstream porting and JAX parity
are parked. Use **centered CDP** with a jointly learned CNN,categorical RSSM and
imagined actor/critic. No pretrained DINO or Tiny JEPA encoder is required.
Explicitly select CDP; learned RGB remains the CLI default.

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
| Boxing |1M32:76.833/79.167,two learners with24/24 wins,in28m22s/28m13s|Both pass; broad reliability open|
| Freeway |Aided1M32:31.042/32.000,two learners with24/24 successes,in28m17s/28m50s|Aid off during evaluation; broad reliability open|
| Pong |12M48/B16/R16:greedy15.208 on both learners,24/24 wins,in41m28s/41m31s|Greedy passes narrowly; sampled passes one of two|
| Breakout |Greedy collection:sampled78.542/greedy90.042 in21m45s;sampled collection:70.750/143.667 in21m28s|Retain the sampled gain andstronger greedy control separately;all0/24 two-wall,reused1009/cohort|
| Qbert |Fixed240k/R20:greedy15,279.167/15,931.250 on16001/18013,24/24 first pyramids each,in25m58s/26m23s|One passing development learner andone fresh confirmation;sampled12,972.917/14,976.042 both fail15k. Earlier R24 failure remains; broad reliability open|

Published long-run scores are aspirations,not matched-budget comparisons.
Finite losses,short runs and backend speed do not establish mastery or CDP
superiority. Guided online returns and replay imitation losses are not frozen
competence. Keep detailed results in the roadmap and linked reports,not a
growing history in this file.

Latest qualified Breakout evidence:
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

The shared transfer `/mnt/data/kindle-handover-20261025` contains the4096-action
context smoke and2M Breakout originals,matching all346 tensors. Other historical
game weights were not transferred. New AMD checkpoints remain locally retained.
Original archives moved to `/x/Artifacts/shared-drive-cleanup-20261025` on the
original machine; exact counters and paths are in the handover.

Old incident/optimization ledgers and detailed data remain on
`archive/phase2-before-cleanup-20261025`; current reports stay here.
Earlier expanded instructions remain in Git at34b1e74; the recent experiment
ledger is preserved at52c2174 (and earlier113ecde/8b1ba49) and in the roadmap/reports. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
