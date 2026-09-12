# Recovered-driver qualification and latest Meganeura

The September 11 **15:52 UTC** direct remote recheck still resolves upstream main
to [`ce80e9cd`](https://github.com/kvark/meganeura/commit/ce80e9cd6056c230590b8b7e1eb9ffe9bbce08bc).
The existing isolated dependency-only candidate is therefore current. It fixes
generated matmul epilogues, invalidates affected compiled plans and includes
the required LeVJEPA cache corrections upstream. It does not automatically
enable Kindle's separate block-matmul candidate or experimental tuning options.

The [previous CPU qualification](2026-09-09-meganeura-update.md#september-10-follow-up-new-correctness-update-not-yet-adopted)
and original packages remain unchanged. A fresh work directory,
`runs/meganeura-runtime-20260911.Nnfxk4`, preserves the new inputs and results;
it never restarts the completed or failed queues.

## Separate the driver from the backend change

The [host incident](2026-09-11-host-driver-incident.md#observed-recovery-and-runtime-qualification)
is externally recovered: both kernel and NVML now report 595.91.07. Merely
seeing an idle GPU is not runtime qualification. Keep the same GPU UUID and
boot throughout each comparison, with no overlapping GPU-heavy work.

1. Run the original 18 hardware fixtures using the unchanged `4d45ba3a`
   control binaries on the recovered driver. Preserve full B16/T64 gradients,
   frame-prefix cache semantics, asymmetric resets, N4/N6/N8 feature parity,
   checkpoint restore, vector learning and executed-action overrides.
2. Run those same fixtures for `ce80e9cd`, plus its fused-clamp GPU regression.
   Build these fixtures from the pinned `1e00e818` candidate in a fresh target
   directory; do not overwrite old release binaries or packaged extensions.
3. Require exact complete learning-state and non-timing report comparisons
   from update 1, retaining optimizer moments even at zero learning rate.
   Compare the unchanged control with its archived old-driver result too.
4. Require matched N6/R256 pixel runs in AB/BA order, including full frozen
   state and action/reward/reset traces. Recheck actual direct free memory
   throughout every native phase (at least 2,048 MiB), not total minus used.
   Preserve the existing 98% timing regression guard; a speedup requires both
   warmed candidate/control ratios above 1.005.

Any numerical mismatch stops adoption and needs a separate diagnosis; a
correctness fix is not permission to relabel changed arithmetic as exact parity.
Driver and backend comparisons cannot be combined into an unexplained speedup.
Do not use the new backend to restore historical checkpoints by rewriting their
identity. Breakout's saved final model keeps its original `f6a2b6ad` executable.

## Completed runtime qualification

Preparation completed with one build job in a fresh target directory. The
manifest verifies preserved source/package/experiment pins and records boot,
kernel and driver identity. Forty-seven CPU checks pass for driver/evidence
validation, complete-state comparisons, the new pixel wrapper and raw auditing. Their
synthetic fixtures and archived-state rereads are not new GPU results.

The unchanged control's **18 GPU hardware tests** completed at **15:00:20 UTC**;
the upstream candidate's **19 tests** completed at **15:07:55**. Both pass full
production gradients and causal/cache/reset checks, with zero measured
N4/N6/N8 batched-versus-serial dense-feature error. The candidate also passes
the new fused-clamp fixture. Complete GPU logs retain at least **6,577 / 6,547
MiB directly free** respectively. These hardware groups do not establish
combined learner memory or throughput.

The separate synthetic comparison completed at **15:15:12 UTC**. The
one-update pair and both eight-update orders match all 241 tensors, optimizer
state, metadata and non-timing reports exactly across backends. The unchanged
control also exactly reproduces its archived old-driver eight-update result.
All six complete native windows pass memory/coverage checks. This synthetic
fixture omits the video frontend and is not a gameplay or speed result.

The N6 pixel gate ran **15:15:31–15:56:10 UTC** and completed normally. Its
actual imports are `f6a2b6ad` (control) and `abf4ae5d` (candidate). Every default
run completes 3,840 training actions / 610 updates, followed by 768 frozen
actions with zero updates. All four full checkpoints, optimizer moments,
non-timing learner reports and training/frozen action/reward/reset traces match
exactly across the two backend orders. The unchanged control also reproduces
the archived old-driver pixel result exactly. The comparison records the driver
header difference explicitly; it does not rewrite either raw header.

The candidate's separate Freeway override check completes the same budgets,
with probability .25 / hold64 during training and no overrides during frozen
evaluation. Executed actions, recurrent/replay accounting, complete state and
frozen restore checks pass. This short integration check is not learned Freeway
competence or an exploration ablation.

### Timing and memory

Each warmed window covers actions 2,304–3,840: **1,536 actual interactions and
384 learner updates**, with no training debt. Actual emulator-frame increments
determine the clocks. The unchanged recipe is N6/R256/B16/T64, full BPTT64,
microbatch 16, 12M and F32. CPU compilation was kept outside these windows.

| Order | Backend | Actions/s | Aggregate real time | Per stream |
| --- | --- | ---: | ---: | ---: |
| A1 | Control `4d45ba3a` | 8.5390 | 0.5693× | 0.09488× |
| B1 | Upstream `ce80e9cd` | 8.5881 | 0.5725× | 0.09542× |
| B2 | Upstream `ce80e9cd` | 8.6033 | 0.5736× | 0.09559× |
| A2 | Control `4d45ba3a` | 8.5227 | 0.5682× | 0.09470× |

Candidate/control ratios **1.005750 / 1.009460** pass the predeclared regression
and speedup gates. This is only **0.6–0.9%** in two short paired windows, not a
large or independently replicated throughput improvement. It does not solve
sub-real-time R256 learning or establish better training-seed reliability.
Whole-run rates include warmup without learning and are not steady-state speed.

All ten pixel native phases have complete sampled GPU coverage: at least
**3,303 MiB directly free** during training and **3,413 MiB** frozen, with
462 MiB directly reserved. Whole-phase activity includes startup and warmup;
it is neither kernel occupancy nor a calibrated idle-gap measurement.

The [independent raw audit](../../runs/meganeura-runtime-20260911.Nnfxk4/independent-audit.json)
reverifies all **1,364 input pins**, 47 CPU wrapper checks, both hardware groups,
six canary and ten pixel native windows, complete state/reports/traces, actual
command arguments and same-boot driver identity. Original experiment packages,
queues and results remain unchanged. Do not restart these completed gates.

## Source integration and package boundaries

Main's four dependency/identity files now adopt the qualified candidate. The
fresh source-matched integration completed normally at **16:01:47 UTC**:
workspace/Python formatting and Clippy, **92 Rust CPU tests**, **253 matched
Python tests** and **three focused GPU integration tests** all pass. Build,
wheel and actual import identity agree; source and packaged Python match. A
separate target and package directory leave historical extensions/binaries intact.

The [source-adoption recheck](../../runs/meganeura-runtime-20260911.Nnfxk4/source-adoption.json)
freshly rereads the full isolated runtime proof and main raw logs, test counts,
GPU windows, source/package identities and dependency files. Its 346 source/
artifact pins also bind the complete new checkpoints. It records an uncommitted
source snapshot over base `452521d`, not a claim that the old commit contains
this patch. The original runtime audit's `adopted: false` remains unchanged;
this later artifact records adoption.

The fresh main package is `runs/meganeura-runtime-20260911.Nnfxk4/main-package`,
native `1735b8a6`. It is source-matched integration evidence, **not** a long-run
Atari runtime qualification or a replacement for the isolated package below.

The fully pixel-qualified upstream Atari package is
`runs/meganeura-upstream-20260910.ERT7QD/package`, native `abf4ae5d`, with matching
source `1e00e818` at `/x/Code/.kindle-meganeura-upstream-20260910`. Keep its Python
sources, runner and auditor together. New experiments need their own declarations;
source adoption does not change the native or Python bundle of old declarations.

The original `f6a2b6ad` control is also qualified on this recovered driver. Old
checkpoints retain that backend and their source-matched Python bundle. In
particular, Breakout's missing final evaluation needs a separately declared
driver-aware continuation that preserves its completed training; the original
failed worker and follower remain terminal. No long learning or successor was
started by this qualification.

World-model diagnostics remain separate from the dependency update: the
[completed video reports](2026-09-08-world-evaluation.md) compare predictions
made before observations with actual outcomes using each model's original build.

## September 12 upstream preflight

These morning checks are historical. The **19:12 UTC recheck** discovers actual
runtime changes at `45991be1`, now staged in the separate
[Meganeura/Blade timing update](2026-09-12-meganeura-timings.md). Preserve all
completed evidence below; it qualifies ce80e9cd, not that newer dependency.

The fresh direct remote checks at **06:48–06:53 UTC** resolve main to
[`de7e6fcf`](https://github.com/kvark/meganeura/commit/de7e6fcf7ebec6fda003e3dae4a7f3e6f3a93169),
four commits beyond `ce80e9cd`. The [recorded comparison](../../runs/meganeura-upstream-recheck-20260912.jo1ZlX/result.json)
verifies that only four documentation/paper files changed. Every other tracked
path is identical, including runtime, shaders, build configuration and tests.
The `src` and `tests` tree IDs and `Cargo.toml` blob ID match exactly. All 25
source/checker pins are recorded; the shared backend checkout stays on its
original clean `megakernel-probe` branch.

The **07:40 UTC recheck** finds one further commit,
[`3622e06f`](https://github.com/kvark/meganeura/commit/3622e06fb27ff84efb04e29d1aab9493c2c1415b),
again changing only those four documentation/paper files. The
[new recorded check](../../runs/meganeura-upstream-recheck-20260912-0739.HyifGP/result.json)
reverifies all earlier source pins, every unchanged top-level object and the
full runtime/build/test diff against `ce80e9cd`. All **27 pins** pass. There is
no missing runtime fix at that check, package change, rebuild or GPU work. The original
check remains preserved; fetching the remote did not change the shared checkout.

| Kindle source | Actual runtime pin | Consequence |
| --- | --- | --- |
| Main, block-matmul candidate, Breakout minimal candidate at 07:40 UTC | `ce80e9cd` | All upstream runtime fixes at that check are present. No rebuild or new qualification follows from that documentation-only tip. |
| Active Atari control | `4d45ba3a` | Preserve its explicit historical package/checkpoint identity and already declared comparisons. Do not switch a live run. |
| Deferred grouped-RSSM and world-sync candidates | `a7e2efd9` | Carry candidate-only changes onto the current qualified backend before any new diagnosis. Their old evidence is not a current-backend result. |

Before another backend diagnosis or optimization, repeat the upstream comparison
and inspect relevant upstream fixes/regression tests first. Do not recreate a
fix already available upstream. A changed runtime needs an isolated source-matched
candidate and the existing numerical/state/memory/timing requirements; a changed
paper does not justify checkpoint-identity churn or another GPU queue. This
preflight changes no dependency, package, experiment or scheduled work and makes
no speedup claim.
