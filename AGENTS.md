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

## Current work — November 7

The active goal is **prove Breakout world-model training in isolation**. Actor/
critic and the five-game campaign are parked. Use the native pixel-supervised
CNN/categorical RSSM/RGB decoder as the inspectable reference; CDP is a later
efficiency comparison. No pretrained encoder or RAM/object/phase labels enter
model inputs/objectives. Full competence remains unproven.

The [motion learner](docs/results/2026-11-07-breakout-world-motion-attention.md)
solves most observed ball reconstruction; the [coverage comparison](docs/results/2026-11-07-breakout-world-coverage.md)
fixes waiting and improves contacts/ending. Retain both closed results, including
original count/comparison failures and the invalid FIRE90% pair-match gate with
separate repairs/reviews. Do not repeat any completed worker or collector.

The [joint predictive energy comparison](docs/results/2026-11-07-breakout-world-energy.md)
is now **closed**. At matched fresh8,192 updates,observed balls109/112 versus
108/112 and brick balanced error.207707 versus.223954 improve,but one-step
81/109 versus94/109 and eight-step4/107 versus14/107 regress. Native987.968s/
loop1,338.229s versus529.515/796.900s. Retain both models as different development
capabilities; no replacement or gameplay/seed-reliability claim.

Frozen history16/32,eight draws,84 groups finds useful ending learning: last-life
RIGHT97.182% versus50.211% (oracle100%); ending Brier.006129 versus.144566,passing.10.
Aggregate16/32 ball energy improves10.45%,failing the declared25% gate. Non-ending
group means pass5%,but65/304 origin-plan cases assign>5% ending probability,
maximum25.990%,versus0/304 and1.907% control. Precise tracking/reward remain weak.

Root `/mnt/data/kindle-breakout-world-energy-20261107`,native/source161839d9,
SHA8d5552e5,matching root/runtime.43 CPU world/27 affected/27 packaged Python checks,
strict Clippy,77/89/648 phase input pins,five guards/sealed audits,204 initial/
frozen tensors,18 initial/restore arrays and8,192 finite13-report rows pass.
Preserve initial CPU fixture/path failures. `closure-review.json` verifies closure.
The coverage trajectory control remains at its exact native/source3fa37b6d,
SHAeafe8359,runtime `/mnt/data/kindle-breakout-world-motion-20261107/runtime`.
Do not mix runtime overlays or objectives.

The energy result's original41/41 Python-count claim was incorrect: retained logs
show27/27. Its pinned receipt/protocol/publication remain preserved,with a separate
CPU-only count correction linked in the result. No old check or GPU phase reruns.

The [conditional score-function comparison](docs/results/2026-11-07-breakout-world-score-function.md)
is **closed and fails its learning comparison**. Its exact conditional gradient
passes CPU/native references and all13 first-update values match ST energy,but
fixed8,192 updates regress one-step71/109 versus81 ST/94 coverage; eight-step
4/107 versus4/14. Observed109/112 remains good. Frozen history16/32,eight draws,
84 groups regress primary ball energy.393950 versus.151784 ST/.169502 coverage;
NOOP8 false balls29/64 (45.31%) versus0; last-life RIGHT ending8.591% versus
97.182% ST/50.211% coverage (oracle100%); ending Brier.433024 versus.006129/.144566.
No extension,confirmation campaign or replacement. Retain both earlier capabilities.

Root `/mnt/data/kindle-breakout-world-score-function-20261107`,native/source0ab7fea7,
SHA0ebed9cd,matching root/runtime. Native990.401s/loop1,240.170s; source seed41001
and exact coverage data/plan reused,not confirmation.48 CPU world tests(22 GPU
ignored),51 selected/51 packaged Python checks,strict Clippy,83/95/858 phase pins,
five guards/sealed audits,204 initial/frozen tensors,18 initial/restore arrays
and8,192 finite13-report rows pass. Preserve fixture shape/Clippy failures and
previous energy Python-count correction. Execution passes; learning gates fail.

The [CPU frozen-feature diagnosis](docs/results/2026-11-07-breakout-world-readout-diagnosis.md)
is closed,with no GPU/emulator,new forecast or model update. Short-term motion is
linearly readable; current/future position readouts do not provide a useful new
head. Their generalization failure does not prove information absent. Eight
feature-input and30 action/source pins,63 fixed ridge coefficient/prediction
files remain in root/readout-diagnosis. No readout enters model training.

No Kindle GPU job or successor is active/selected. The next focused question is
actual gradient noise/loss/optimizer interference at a common frozen checkpoint,
before another recipe,larger architecture or longer budget. Separately declare
that diagnostic; no automatic learner,collector,old-model rerun or actor/critic.

Remaining requirements include precise causal1/4/8/16/32 forecasts,reward/
uncertainty calibration,removed-brick memory,late-game/tunnel/second-wall coverage,
untouched tests and fresh confirmation. The retained oracle has no above-top-
brick endpoints; detector-support limits and duplicate contexts remain.

The [prior-reward comparison](docs/results/2026-11-06-cdp-breakout-prior-reward.md#full200k-comparison) remains closed and inconclusive:
79.167/47.000 greedy/sampled versus matched75.250/44.708, with paired intervals
crossing zero, all0/24 two-wall. Retain prior reward off and posterior reward1
for the retained agent. Its original failures, repairs and closure remain intact.

Completed experiment details and their failures remain in the [roadmap](docs/kindle_single_life_dreamer_plan.md)
and [diagnosis](docs/results/2026-10-29-cdp-learning-diagnosis.md), with links to exact protocols and artifacts.
The expanded November3 working guide and handover remain in Git at
`0cf70be8d8cc9d4e6cab32bf1b8e97195fd1d0ab`. This condensation removes duplicated status prose only;
no checkpoint, log, runtime, protocol or experiment source is changed.
Keep this current-work section focused on the active allocation; replace its
status in place and link completed results instead of accumulating a job ledger.

## Direction and retained results

Training is the priority. Backend optimization,upstream porting and JAX parity
are parked. The retained agent uses **centered CDP** with a jointly learned CNN,
categorical RSSM and imagined actor/critic. The newly selected isolated-world
baseline uses pixel supervision as specified above. No pretrained DINO or Tiny
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
Earlier expanded instructions remain in Git at34b1e74; the earlier experiment
ledger is preserved at52c2174 (and earlier113ecde/8b1ba49) and in the roadmap/reports. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
