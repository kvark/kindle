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

The five-game goal is unfinished. Boxing and Pong win consistently; aided
Freeway crosses consistently; Breakout and Qbert remain much weaker. Published
long-run scores are aspirations, not matched-budget comparisons. Do not claim
mastery or CDP superiority from short runs, finite losses or backend speed alone.

Completed learning test: Breakout B2/T64 versus retained B8/T16, keeping128 replay
positions/update and replay ratio32. The AMD F16 life finishes2M actions/499843
updates in11h40m. Frozen500k/final/initial scores are14.71/10.04/1.42, all0/24
two-wall successes, versus retained58.83/98.21/1.42. Full audits pass and the
service has exited. See `docs/results/2026-10-25-cdp-amd-takeover.md`.
Longer sequences also reduce independent sequences; precision, execution path
and hardware differ too. Do not attribute this result to context length alone.
Paired early updates now match initial tensors/experience but differ by precision:
after three updates F16 leaves the first reward projection unchanged, while Auto
changes654080 of655360 elements. Raw derivatives/mechanism remain unmeasured.
Auto's500k B2/T64 comparison is complete: frozen15.67 versus F16's14.71,
initial1.42; all0/24 two-wall successes. It takes2h53m with124843 updates; all
numeric, checkpoint, frozen and replay audits pass. This does not justify a
T64 extension. The next allocation uses the user's cheaper game-specific
recipes; see `docs/kindle_single_life_dreamer_plan.md`. BF16 and
`KINDLE_DEVICE_TARGETS` remain unavailable.
The cheap unassisted Freeway1M/B8/T16/R16 trial completes200k actions in8m07s
but freezes at0/24 crossings, matching its actual initial control. All audits
pass; see `docs/results/2026-10-25-cdp-cheap-freeway.md`.
The aided comparison also finishes: frozen0.25 versus initial0.00, six episodes
cross once but0/24 reach the25-crossing milestone; all audits pass. No extension.
Native32/48 RGB support and an intermediate4M preset now pass bounded GPU checks,
including unchanged64x64 checkpoint restore. Boxing1M/RGB32 completes200k in6m03s
but freezes at−3.875 versus initial0.00 (both10/24 wins); all audits pass.
Pong4M/RGB48 completes200k in10m22s: frozen−17.125 versus initial−20.375,
both0/24 wins; all audits pass. The short screens use only about one fifth of
the historical successful cohorts' updates, so capacity/resolution remain open.
Fresh Boxing seed2017,1M/RGB32/B8/T16/R32,500k actions now passes: frozen76.8333
versus initial2.875,24/24 wins,124939 updates in28m22s. All audits pass and the
service exits. Fresh confirmation seed4013 now also passes: frozen79.1667 versus
initial−0.625,24/24 wins,500k actions/124939 updates in28m13s. The recipe and cap
are unchanged; all audits pass and the service exits. Two useful cheap learners
do not establish broad seed reliability. See the cheap-Boxing result report.
Qbert4M/RGB48/R32 completes200k actions in19m45s: frozen809.375 versus180.2083,
0/24 first pyramids. Mean first-pyramid coverage improves5.83→16.63 of21 cubes,
but the task gate fails. All audits pass; no extension. See the cheap-Qbert report.
Breakout4M/RGB64/R32 completes200k in22m30s: frozen6.2083 versus1.375,0/24
two-wall successes, all audits pass. One selected episode waits12061 actions
after a lost life before choosing FIRE; CPU replay verifies this stall. See
`docs/results/2026-10-26-cdp-cheap-breakout.md`; no extension.
Breakout's1% mixture test also completes: frozen7.0833 versus initial1.375 in
22m29s, longest selected no-FIRE gap447 versus control12115. The original trained
weights with only frozen mixture score6.0833 and gap724 in9.8s, zero updates.
All three policies remain0/24 two-wall successes; all audits pass and services
exit. See `docs/results/2026-10-26-cdp-breakout-unimix.md`; no extension.
Compact aided Freeway1009,1M/RGB32/Auto/B8/T16/R32, completes500k actions and
124939 updates in28m17s: frozen31.0417 versus actual initial0.00,24/24 versus0/24
crossing successes. The5% persistent16-action aid is disabled in evaluation;
constant-UP CPU reference scores21.6667,0/24. All audits pass and the service
exits. Focused confirmation follows below; see the Freeway result report.
Resolution, ratio, budget and runtime change together.
Pong4M/RGB48/R32 completes500k actions/124939 updates in48m53s: frozen11.0
versus initial−20.375,23/24 versus0/24 wins. All audits pass and service exits;
the mean15 gate fails. No unchanged extension. See the cheap-Pong result report.
Freeway confirmation4013 also passes the fixed recipe: frozen32.000 versus
initial0.00,24/24 crossing successes,500k/124939 updates in28m50s. All audits
pass and service exits. Two compact learners now succeed within30 minutes each.
The retained Breakout2M policy's frozen AMD Auto check completes:67.5833,
0/24 two-wall successes,21,976 actions in14.127s and zero updates. All346 tensors
match the source, all audits pass and service exits. See
`docs/results/2026-10-26-cdp-retained-breakout-auto.md`; its warm start is declared separately below.
Qbert coefficient100 completes200k/49939 updates in20m12s: frozen809.375,
0/24 pyramids, matching the coefficient1 score but reducing mean coverage
16.625→13.708 of21 cubes. All audits pass and service exits. No extension.
Breakout H5 warm start completes50k new actions/12439 updates in8m02s but
regresses frozen67.5833→27.2083, both0/24 two-wall successes. Frozen source trace
matches H15 before learning; all audits pass and service exits. Retain the source,
no extension. See `docs/results/2026-10-26-cdp-breakout-short-imagination.md`.
Pong4013 confirmation completes500k/124939 updates in50m14s: frozen7.333 and
22/24 wins versus own initial−20.333 and0/24. All audits and curves pass; service
exits. Both compact Pong learners win most matches but miss mean15; no extension.
Six-action Qbert completes200k/49939 updates in18m21s: frozen605.208 versus own
initial148.958,0/24 pyramids, mean coverage10.167/21. Score/coverage are lower than
the full18 control. All audits/curves pass and service exits; no extension.
The separate CPU Breakout pixel follower scores11.333,0/24 two-wall successes,
with full replay. No reference controller or RAM enters learning.
Current allocation: fresh compact Breakout H5/500k,4M/RGB64/Auto/N8/B8/T16/
R32, native mixture.01, fixed45-minute cap. Horizon15→5 and experience200k→500k
change together. The earlier12M H5 warm start regressed; this is a fresh compact
compute/budget recipe, not a continuation or established horizon benefit. Retain
own initial/fixed final controls and the unchanged two-wall gate. See
`docs/experiments/2026-10-26-cdp-breakout-compact-h5-budget.md`; owner
`kindle-cheap-breakout-h5-unimix-500k-20261026`. Check live state before GPU work.
Started05:41:58 UTC, source`8910efe`. Full config/counters, finite initial weights
matching all346 control tensors and first updates pass;4,131,729 parameters.
Fixed evaluations/full audit follow. Runtime/evidence: `/x/Temp/kindle-cheap-20261025`.
CLI`1ed07af` exposes imagination length for fresh runs; existing recipes retain15
and restores reject overrides.263 vector-runner tests pass; native is unchanged.
Do not restart completed training, old queues, broad seed campaigns or unchanged
long extensions. Keep single-actor learning ahead of swarms/concurrent learners.

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
