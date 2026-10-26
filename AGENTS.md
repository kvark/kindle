# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md` when
available. Only the user merges PRs; commits and pushes are allowed.

## Start here

- [Machine handover](docs/handover.md): source branches, build instructions,
  shared checkpoint/evidence package and the exact unfinished experiment.
- [Roadmap](docs/kindle_single_life_dreamer_plan.md): current results and priorities.
- [PR31](https://github.com/kvark/kindle/pull/31): live status dashboard. Update its
  description when status changes; do not create a separate STATUS.md.
- [GPU incident](docs/results/2026-10-23-cdp-context64-gpu-fault.md): evidence and
  limits of the repeated initialization failures on the original machine.

## Active direction

Training is the priority. Backend optimization, upstream porting and JAX parity
are parked. The selected architecture is **centered CDP**, with a jointly learned
CNN, categorical RSSM and imagined actor/critic. No pretrained DINO or Tiny JEPA
encoder is required. Explicitly select CDP; the CLI default remains learned RGB.
The user's current priority is compute-constrained success on this game set.
Game-specific model sizes, image resolutions and learning hyperparameters are
allowed; one universal recipe is not a prerequisite. Prefer useful frozen
learning per GPU-hour, with short development runs before long allocations.
T64 is an experiment, not the default for subsequent development. Start from
T16 and consider lower replay ratios and smaller models using the historical
capacity evidence. Size1M already worked well for Boxing and Freeway, but was
much weaker for Breakout and Qbert. Native32/48/64 image geometry is supported;
preserve one GPU resize from native frames and verify small-object visibility.
Changing several settings to find a useful cheap recipe is allowed; disclose
the changes without claiming an isolated causal effect or five-game reliability.
Work on `phase2-levjepa-evaluation` and PR31. This branch now includes the
qualified c8aeba28 implementation with portable dependencies. Do not switch to
a separate takeover branch. Keep `runs/`, large data and archived experiment
ledgers out of Git; old evidence lives on the explicitly named archive branch.

The five-game goal is unfinished. Compact AMD Boxing and aided Freeway pass on
two learners each, within 30 minutes per learner. Compact Pong wins most matches
but misses its mean-15 gate; Breakout and Qbert remain much weaker. Published
long-run scores are aspirations, not matched-budget comparisons. Do not claim
mastery or CDP superiority from short runs, finite losses or backend speed alone.
Keep detailed results in the roadmap and linked reports, not another ledger here.

Use Auto precision for current AMD learning. BF16 and `KINDLE_DEVICE_TARGETS`
remain unavailable. After three paired early updates, F16 leaves the first reward
projection unchanged while Auto changes it; raw derivatives and the mechanism
remain unmeasured. The completed F16/Auto B2/T64 lives do not justify extensions.
See `docs/results/2026-10-25-cdp-amd-precision.md` for the evidence and limits.

Current allocation: Qbert12M/RGB48/B16/R32 capacity/batch learning screen,
selected after bounded training/restore passes. Fresh1009/eval4000001009,
Auto/N8/T16/H15, coefficient1, full18, no aid,14,123,473 parameters. Fixed200k
aggregate actions/24,954 updates, first376,100k recovery and35-minute training
cap. Own actual initial/fixed final24-natural-episode evaluation; unchanged
15k/90%-pyramid gate and full config/counter/numeric/tensor/replay/guard audits.
Capacity and batch change together; no matched-initial claim across model sizes.
See `docs/experiments/2026-10-26-cdp-qbert-capacity-budget.md`; intended owner
`kindle-cheap-qbert-12m-48-b16-200k-20261026`. Check live state before GPU work.
Started10:41:58 UTC from`be6c942`; startup verifies first376,14,123,473 parameters
and346 initial tensors matching the same-shape check. Audits/analysis are automated.
The separate16,384-action check takes149.387s and preserves346 tensors on restore;
service exits10:35:06 UTC. It supplies execution/early cost, not competence.
No retry, cap increase, extension or automatic next allocation.

PongB16 completes500k/62,454 updates in38m45s,22.37% less training time than B8,
but frozen11.875 is below14.958; both24/24 wins and below mean15. All audits and
comparison pass; service exits10:28:51 UTC. No extension or confirmation.
Qbert500k/R16 remains weaker than its200k/R32 control, with0/24 pyramids.

Runtime/evidence: `/x/Temp/kindle-cheap-20261025`. Native `bfb8c31`, runner
`1ed07af`, CPU replay `e853818`. The active declaration pins exact runtime/script
hashes; the handover locates the build receipt. Native 32/48/64 geometry and the
intermediate 4M preset pass bounded training/frozen checks, including 64 restore.
The CLI exposes imagination length for fresh runs; defaults retain 15 and
restores reject overrides. Do not restart completed training, old queues, broad
seed campaigns or unchanged long extensions. Keep single-actor learning ahead
of swarms/concurrent learners.

## Experiment practice

- Use small development runs and meaningful checks; avoid turning preparation
  into another backend qualification project. Automate monitoring and audits.
- Keep batched inference with eight independent Atari environments. Report
  aggregate actions and emulator frames separately, plus wall-clock time.
- Retain explicit config, seeds, native/source identity, learning curves and
  frozen natural-episode evaluation against actual-initial controls. Record
  exploration aids, budgets, cutoffs and failed attempts. Confirm promising
  changes across seeds; selected/reused development seeds are not held-out tests.
- Checkpoints do not preserve replay, live belief, games or RNG. A restored
  extension is a warm start, not uninterrupted learning or an independent seed.
- Preserve visual detail: native max-pooled frames with one GPU resize. Do not
  reintroduce CPU downscale followed by upscale.
- Put compact results in docs/results and update PR31. Logs, weights, videos,
  build caches and profiling dumps belong outside Git. Keep only useful
  checkpoint milestones; release disposable build caches after packaging.

## GPU operation

GPU work is authorized unless the device is genuinely faulted. Known allocation
warnings alone are not an approval gate. Use the existing host guard and stop on
real faults or numerical failures. No automatic driver reset/reload/reboot.
Separate NVML polling is unnecessary for the current experiment.

Historical boot24f26399 on595.91.07 remained faulted. A later boot9461079a
was observed during handover; no GPU health check was performed by this task.
Repeated PMU halts occurred on595 and615; the latest captured wait was in the
first world's device-local initialization fills, before checkpoint loading.
This is localization, not proof of a buffer bug, firmware cause or bad hardware.
A debugger run with ASLR disabled once succeeded; that is not a fix.

On that host, use the prepared pre-submission capture after recovery, not another
blind restore. On a different healthy GPU, do not inherit the old boot/device
assertions or presume it is wedged. Start with bounded initialization/restore,
record the new device/driver, and stop if the fault reproduces. The ABI-specific
GDB collector is for the original binary only, not an arbitrary rebuild.

## Historical material

Original run/checkpoint archives were relocated to
`/x/Artifacts/shared-drive-cleanup-20261025` on the original machine. The4096-action
context smoke and2M Breakout originals are present and match all346 tensors in
the shared handover exports; see the handover for exact paths and counters.

The old incident/optimization ledgers and detailed experiment data remain on
`archive/phase2-before-cleanup-20261025`. Current result reports are retained here. Historical “next” paragraphs are not active instructions.
User-authored TASK.md and docs/cdp-claude-feedback.md are preserved as context;
subsequent CDP decisions and current measured results supersede their older
architecture/performance assumptions. Do not merge upstream PRs on the user's
behalf or modify neighboring worktrees during handover.
