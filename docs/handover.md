# CDP handover — October25,2026

**Receiving-host update:** [AMD takeover](results/2026-10-25-cdp-amd-takeover.md).
RX7900XT/RADV cannot run the saved BF16 configuration. A labelled F16-input copy
passes restore with all346 tensors unchanged. The fresh2M B2/T64 life and audits
are complete: frozen500k/final/initial scores14.71/10.04/1.42, all0/24 two-wall
successes. Training took11h40m; that service has exited. Paired early updates
show precision-dependent differences, and Auto's short training/restore passes.
The fresh [500k Auto comparison](experiments/2026-10-25-cdp-amd-auto.md) also
completed: frozen15.67 versus F16's14.71, all0/24 two-wall successes, in2h53m.
All audits pass. Both services have exited; do not restart or extend these lives.
The user now prioritizes cheaper game-specific recipes; use the roadmap.
[Cheap Freeway screens](results/2026-10-25-cdp-cheap-freeway.md) finish200k actions
in about8 minutes but remain weak (unassisted/aided frozen0.00/0.25, initial0.00).
[Smaller native recipes](results/2026-10-25-cdp-cheap-runtime.md) add4M capacity
and32/48 image geometry; GPU training/frozen checks and retained64 restore pass.
The [200k Boxing screen](results/2026-10-25-cdp-cheap-boxing.md) completes in6m03s
but fails: frozen−3.875 versus initial0.00, both10/24 wins, all audits pass.
The [Pong4M/RGB48 screen](results/2026-10-25-cdp-cheap-pong.md) completes in10m22s:
frozen−17.125 versus initial−20.375, both0/24 wins, all audits pass.
Fresh Boxing2017,1M/RGB32/500k/R32 now passes: frozen76.8333 versus initial2.875,
24/24 wins,124939 updates in28m22s; all audits pass and the service exits. The fixed
recipe also passes fresh confirmation4013:79.1667 versus initial−0.625,24/24 wins,
500k/124939 updates in28m13s, all audits pass and service exits. Two useful learners
do not establish broad reliability. Details and curves are in the Boxing report.
[Qbert4M/RGB48/R32](results/2026-10-25-cdp-cheap-qbert.md) finishes200k in19m45s:
frozen809.375 versus180.2083,0/24 first pyramids. Mean coverage improves5.83→16.63
of21 cubes; all audits pass, but the task gate fails. No extension.
[Breakout4M/RGB64/R32](results/2026-10-26-cdp-cheap-breakout.md) finishes200k in
22m30s: frozen6.2083 versus1.375,0/24 two-wall successes, all audits pass. A CPU
replay verifies one12061-action delay before FIRE after a lost life. No extension.
[Breakout's1% native mixture](results/2026-10-26-cdp-breakout-unimix.md) completes
200k in22m29s: frozen7.0833 versus initial1.375, longest selected no-FIRE gap447
versus control12115. The baseline weights with only frozen mixture score6.0833
with gap724 in9.8s and zero updates. All three policies remain0/24 two-wall
successes. All audits pass and services exit; no extension.
[Compact aided Freeway](results/2026-10-25-cdp-cheap-freeway.md#compact-rgb32-with-the-successful-update-budget)
completes500k/124939 updates in28m17s: frozen31.0417 versus actual initial0.00,
24/24 versus0/24 crossing successes, aid disabled in evaluation. All audits pass
and service exits. Constant-UP CPU reference scores21.6667,0/24. The successful
1M/RGB32/R32 recipe's focused confirmation follows below; no isolated attribution.
[Pong500k/R32](results/2026-10-25-cdp-cheap-pong.md#r32-and500k-actions) completes
in48m53s: frozen11.0 versus initial−20.375,23/24 versus0/24 wins. All audits pass
and service exits; the mean15 gate fails. No unchanged extension.
[Freeway seed4013 confirmation](results/2026-10-25-cdp-cheap-freeway.md#fixed-recipe-seed4013-confirmation)
also passes:32.000 versus initial0.00,24/24 crossing successes,500k/124939 updates
in28m50s. All audits pass and service exits. Two compact learners now pass within
30 minutes each. The [retained Breakout2M frozen check](results/2026-10-26-cdp-retained-breakout-auto.md)
scores67.5833 with0/24 two-wall successes,21,976 actions in14.127s and zero updates.
All346 source/copied/frozen tensors match; all audits pass and service exits.
Only precision metadata changes to Auto; the warm start below is separately declared.
[Qbert coefficient100](results/2026-10-25-cdp-cheap-qbert.md#coefficient100-does-not-improve-the-fixed-budget-result)
completes200k/49939 updates in20m12s: frozen809.375,0/24 pyramids. Score matches
the control but mean coverage falls16.625→13.708 of21 cubes. All audits pass and
service exits; no extension.
The [H5 Breakout warm start](results/2026-10-26-cdp-breakout-short-imagination.md)
completes50k new actions/12439 updates in8m02s but regresses67.5833→27.2083,
both0/24 two-wall successes. The frozen source trace matched H15 before learning;
all audits pass and service exits. Retain the source; no extension.
The [Pong4013 confirmation](results/2026-10-25-cdp-cheap-pong.md#fixed-recipe-seed4013-confirmation)
completes500k/124939 updates in50m14s: frozen7.333 and22/24 wins versus own
initial−20.333 and0/24. All audits and curves pass; service exits successfully.
Both compact Pong learners win most matches but miss mean15. No extension.
The [six-action Qbert test](results/2026-10-25-cdp-cheap-qbert.md#six-action-choices-run-faster-but-reduce-frozen-coverage)
completes200k/49939 updates in18m21s: frozen605.208 versus own initial148.958,
0/24 pyramids. Coverage10.167/21 and score are lower than the full18 control.
All audits/curves pass and service exits; no extension. The separate CPU Breakout
pixel follower scores11.333 and a fixed two-frame interception variant20.292,
both0/24 two-wall successes with complete replay. Neither supplies training guidance.
The [compact Breakout H5 result](results/2026-10-26-cdp-breakout-compact-h5-budget.md)
retains a45-minute cutoff at487,736 actions/121,873 updates: frozen8.458 versus
initial1.375,0/24 two-wall successes. Its shared prefix costs18.06% less wall time
than H15, but the larger-budget recipe still fails the task. Numerical,
checkpoint/accounting and frozen replay checks pass; the original controller/
guard retain their time-budget failure. Both services exit; no extension.
The [Pong coefficient.01 primary](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-improves-the-fixed-budget-result)
completes500k/124939 updates in49m55s: frozen14.9583 and24/24 wins versus own
initial−20.375 and0/24. This improves the coefficient1 control's11.0/23 wins,
but still misses mean15. All audits/curves pass and its service exits; no extension.
The [coefficient.01 confirmation](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-seed4013-confirmation)
finishes500k/124939 updates in49m58s:14.0 and24/24 wins versus initial−20.333.
Its coefficient1 control scores7.333/22 wins. All audits/curves/comparisons pass;
service exits08:27:53 UTC. Both smaller-bonus seeds miss mean15; no extension.
The [four-action Breakout result](results/2026-10-26-cdp-breakout-minimal-actions.md)
completes200k/49939 updates in20m59s,6.64% less training time than full18.
Frozen9.542 versus own initial1.167 improves the full18 control's7.083, but
still0/24 two-wall successes. All audits/comparison pass; service exits09:01:55.
The [Qbert500k/R16 result](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-does-not-improve-the-result)
finishes in26m53s: frozen719.792 versus initial180.208,0/24 pyramids and14.125/21
coverage, below the200k/R32 control809.375/16.625. All audits/comparison pass;
service exits09:33:03 UTC. No extension or confirmation selected.
The [bounded B16 check](results/2026-10-25-cdp-cheap-pong.md#b16-bounded-check-passes-with-lower-early-wall-cost)
passes16,384 actions/2,002 updates and ordinary restore, using72.224s versus
94.054s for B8. All346 initial tensors match and restored tensors stay unchanged;
service exits09:38:03 UTC. The23.21% early cost reduction is not learning evidence.
The [full-budget PongB16 result](results/2026-10-25-cdp-cheap-pong.md#b16-at-the-fixed500k-budget)
completes500k/62,454 updates in38m45s,22.37% less training time than B8. Frozen
11.875 versus initial−20.375 retains24/24 wins but falls below B8's14.958 and
misses mean15. All four recoveries, frozen audits, curves and comparison pass;
service exits10:28:51 UTC. No extension or seed confirmation selected.

The [bounded Qbert capacity check](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-rgb48-bounded-check)
passes16,384 actions/65,481 frames/2,002 updates in149.387s, versus93.813s for
4M/B8. Actual14,123,473 parameters, all reports/config/counters/finite checks,
selected parameter changes and ordinary restore/346-tensor invariance pass.
Service exits10:35:06 UTC. This is execution/early-cost evidence, not competence.

The [completed Qbert capacity/batch screen](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-the-fixed200k-budget)
scores959.375 versus own initial165.625, with0/24 pyramids. At200k actions,
12M/B16 takes32m19s versus4M/B8's19m45s; coverage16.292/21 is below16.625/21.
All audits and completed comparison pass; service exits11:15:13 UTC. The score
increase does not justify63.70% more wall time for this compute-constrained
screen. No extension or confirmation selected.

The [completed Pong doubled-rate test](results/2026-10-25-cdp-cheap-pong.md#doubling-b16-learning-rates-lowers-the-frozen-score)
scores9.125 versus the B16 control's11.875, both24/24 wins and below mean15.
Training38m32s is essentially unchanged from38m45s. All audits, initial-tensor/
semantic-trace equality and completed comparison pass; service exits11:59:15 UTC.
Retain the original rates; no extension or seed confirmation selected.

The [completed Qbert training-aid test](results/2026-10-25-cdp-cheap-qbert.md#persistent-training-aid-does-not-improve-frozen-coverage)
scores678.125 versus809.375 without aid; mean cube coverage15.125 versus16.625,
both0/24 pyramids. Training20m17s versus19m45s, with10,039 verified overrides.
All audits and completed comparison pass; service exits12:24:16 UTC. Retain the
unaided recipe; no extension or confirmation selected.

The [completed PongH5 test](results/2026-10-25-cdp-cheap-pong.md#shorter-imagination-saves-time-but-lowers-the-frozen-score)
finishes500k/124,939 updates in38m07s,23.65% less training time than H15. Frozen
8.667 is below14.958, both24/24 wins and below mean15. All audits, matched initial
tensors/traces and comparison pass; service exits13:18:17 UTC. Retain H15; no
extension or conditional4013 confirmation is selected.

The [completed smaller-bonus Breakout test](results/2026-10-26-cdp-breakout-minimal-actions.md#smaller-bonus-lowers-the-frozen-score)
finishes 200k actions/49,939 updates in 21m07s, but lowers frozen score
9.542→5.042, both 0/24 two-wall successes. All audits and comparison pass after
copying the omitted, verified replay manifest and rerunning only the unchanged
CPU audit. The original service failure at 13:43:51 UTC and incomplete watcher
receipt are retained; no GPU work repeats. Retain coefficient 1; no extension.

The [Pong B16/R40 run is interrupted](results/2026-10-25-cdp-cheap-pong.md#host-restart-interrupts-b16r40)
after a host restart:265,440 complete logged actions/41,416 finite reports,
saved200k/31,192-update checkpoint. Its separate frozen assessment scores−9.708
versus initial−20.375, both0/24 wins. All frozen checks pass after a CPU-only
checker-import correction; original failures/unsealed tails remain. The500k
quality question is unresolved. No resumed training or extension.

The [completed Qbert H30 test](results/2026-10-25-cdp-cheap-qbert.md#longer-imagination-costs-more-and-lowers-frozen-coverage)
finishes200k actions/799,441 frames/49,939 updates in26m43s,35.34% more training
time than H15. Frozen582.292 versus809.375 and coverage9.583 versus16.625/21
are worse; both0/24 first pyramids. Retain H15. Final training/frozen/replay/
guard/comparison checks pass, including346 matched initial tensors and identical
semantic initial traces. The declared100k recovery was never written because the
pinned queue saved every200k. Original watcher failures remain; separate CPU
completion records `protocol_complete=false`. Service exits16:26:25 UTC. No GPU
repeat, extension or confirmation for H30.

The [completed six-action Pong test](results/2026-10-25-cdp-cheap-pong.md#six-actions-save-time-but-lower-frozen-quality)
finishes500k/124,939 updates in44m21s,11.17% less training time than full18.
Frozen10.375 versus own initial−20.042 is below full18's14.958; both24/24 wins
and below mean15. All numerical/config/counter/tensor/replay/guard checks,
four recoveries and curves/comparison pass; service exits17:28:27 UTC. Dimensions
and initial weights differ, and the control used the older kernel. Retain
full18; no extension or conditional4013 confirmation is selected.

The [bounded Breakout actor/critic check](experiments/2026-10-26-cdp-breakout-actor-critic-gradients.md)
passes4096 actions/963 updates in22.474s and1024 frozen actions in0.489s. All357
tensors preserve frozen invariance; all963 initial policy/value loss reports
are nonzero, CNN/reward weights change and346 shared initial tensors match the
control. Both guards pass; service exits17:34:23 UTC. This verifies execution,
not useful task learning or gradient correctness. Native unchanged; the exact
313-test CLI patch is applied as runner`fbdaee3`.

The [completed Breakout actor/critic test](results/2026-10-26-cdp-breakout-minimal-actions.md#actorcritic-gradients-give-a-modest-unconfirmed-gain)
finishes200k/799,187 frames/49,939 updates in20m41s, with frozen11.208 versus
9.542 and own initial1.167; both0/24 two-wall successes. All357-tensor/frozen,
100k recovery, numerical/replay/guard checks and matched346 shared initial
weights/semantic traces pass. Main service exits18:05:14 UTC. A separate CPU
comparison import failure is retained; corrected comparison passes without
repeating GPU work. This is a modest unconfirmed gain, with an older-kernel
control; no extension, seed confirmation or global default change is selected.

The [completed QbertRGB64 test](results/2026-10-25-cdp-cheap-qbert.md#rgb64-costs-more-and-lowers-frozen-quality)
finishes200k/799,444 frames/49,939 updates in22m20s,13.14% more training time
than RGB48. Frozen718.750 versus809.375 and coverage15.083 versus16.625/21 are
lower; both0/24 first pyramids. All numerical/config/counter/346-tensor/replay/
guard/100k recovery/curve/coverage/comparison checks pass; service exits18:39:49
UTC. Retain RGB48; no extension or confirmation. Geometry, parameter shapes,
initialization and old/new kernel/module limit attribution.

The [completed Pong greedy-action assessment](results/2026-10-25-cdp-cheap-pong.md#greedy-evaluation-does-not-improve-the-retained-pong-policy)
scores14.333 versus sampled14.958, both24/24 wins and below mean15. Own initial
greedy−20.833/0 wins. Final/initial77,640/20,288 actions and310,513/81,145 frames
in29.564/7.757s, zero learning,24 natural matches each. All346 frozen tensors,
source files, config/counters, complete replays/videos and both guards pass;
service exits18:44:25 UTC. Retain sampled evaluation; no greedy confirmation.
The retained learner's49m55s training cost remains. The five-game goal is unfinished.

The [bounded PongB16 actor/critic check](experiments/2026-10-26-cdp-pong-actor-critic-gradients.md)
passes4096/466 updates in16.126s and1024 frozen actions in0.436s. All357 tensors
remain unchanged,346 shared initial tensors match the control, all466 initial
policy/value reports are nonzero and both guards pass. Service exits18:51:00
UTC. Native/CLI unchanged; this is execution evidence, not useful learning.

The [completed PongB16 actor/critic comparison](results/2026-10-25-cdp-cheap-pong.md#b16-actorcritic-gradients-lower-frozen-quality)
finishes500k/62,454 updates in37m52s but lowers frozen11.875→−0.542 and wins
24/24→10/24. Own initial−20.375/0 wins. All357-tensor/replay/guard/recovery/
curve/comparison checks,346 shared initial tensors and semantic traces pass;
service exits19:36:47 UTC. Retain disabled gradients; no extension or confirmation.
The separate progress observer was reattached without restarting training.

The [CPU-only Qbert pixel rule and prepared adapter](results/2026-10-25-cdp-cheap-qbert.md#pixel-only-cpu-rule-clears-the-first-pyramid)
clear24/24 first pyramids, mean3764.583 and21/21 coverage. Collection and exact
replay each take about30s; no agent or learning. This hand-coded candidate
still misses mean15000 and does not establish unaided Kindle transfer.
The optional episode-guidance adapter passes CPU tests and full pixel/action
replay. The bounded native check completes4096/963 in22.541s and1024 frozen
restore in0.452s;512 verified overrides,346 matched initial/unchanged frozen
tensors, all finite reports and both guards pass. Its service fails19:40:14 UTC
only in the new CPU reader, which incorrectly rejects training intrinsic rewards.
A training-aware correction passes421 tests and the saved trajectory replay;
original failure retained, no GPU repeat.

The fresh Qbert pixel-guided comparison completes 200k actions/49,939 updates
in 20m13s, but lowers unaided frozen score 809.375→633.333 and first-pyramid
coverage 16.625→11.667/21; both have 0/24 pyramids. The 5% episode rule actually
overrides 23,428 actions (11.714%). All audits, 100k recovery, 346 matched initial
tensors/semantic traces, complete pixel/action replay and comparison pass;
service exits 20:07:01 UTC. Retain unaided; no extension or confirmation.
See [the result](results/2026-10-25-cdp-cheap-qbert.md#pixel-guided-training-lowers-unaided-frozen-quality).

Pong RGB32 completes500k/124,939 updates in42m19s,15.24% less training time
than RGB48, but lowers frozen14.958→2.292 and wins24/24→16/24. All numerical,
config/source/counter/346-tensor/replay/video/guard/recovery/comparison checks
pass; service exits21:02:34 UTC. Retain RGB48; no extension or confirmation.
Geometry/initialization, older control kernel and13.690s unintended integrated-
GPU inference during this run limit attribution. Full evidence is in cheap-Pong.

Qbert cap640 completes200k actions/799,403 frames/49,939 updates in20m00s.
Frozen750.000 improves whole-episode guidance633.333 but stays below unaided
809.375; all0/24 pyramids. Coverage16.208/21 versus11.667 and16.625. Actual
training overrides15,360 (7.680%);23 completed guided episodes clear a pyramid,
none of366 unguided episodes does. All audits,100k recovery,346 matched initial
and unchanged frozen tensors, semantic initial traces, full training pixel/
action replay and comparison pass; service exits21:30:21 UTC. Retain unaided.
No extension or confirmation follows capped Qbert. The applied
640-action option passes444 CPU tests; native unchanged. Use the matching
`runtime-qbert-guidance-cap` overlay. Prior isolation failures remain recorded.
See [the completed result](results/2026-10-25-cdp-cheap-qbert.md#capped-guidance-recovers-some-quality-but-remains-below-unaided-training).

Select a bounded Pong actor/critic-rate check after the completed Qbert result:
4M48/Auto/N8/B16/T16/H15/R32/coefficient.01/actor0/full18, original world/encoder/
dynamics rates and behavior rate8e-5 only. Keep4096 actions/466 updates, first376,
zero debt,90-second training cap and1024 frozen actions/60-second cap. All346
initial/frozen tensors, config/counters, selected parameter changes and both
host guards are required. CLI passes460 CPU tests; native unchanged. Owner
`kindle-cheap-pong-behavior-lr2-smoke-20261026`. No full allocation is selected;
review the bounded result first. See the cheap-Pong report's separate-rate check.

New boot`a8648671-8577-4f42-9b46-a4345d3ac297`, kernel7.0.0-38, AMD module
`9BDBC866F55BF296F1B7FA2`; same RX7900XT/Mesa26.0.8/native binary. Host scan,
512-action restore/346-tensor invariance and natural frozen evaluation pass.
No host recovery performed. Qbert's refreshed queue uses this identity; its H15
control used7.0.0-34, so record the kernel change as a comparison confound.
Check live service state before GPU work. Whole-episode and capped Qbert
guidance, Pong RGB32 and their postprocessing are complete. The bounded Pong
behavior-rate check is selected above; do not restart completed services.

Current native source`bfb8c31`; the runner now supports optional per-episode
Qbert guidance limits, SHA256
`50ba7759d4ff108d724e831d83fda69c9d4f4dce3da660cebf394f995cac98da`.
Use the `runtime-qbert-guidance-cap` Python overlay; frozen replay remains
`5043b7a` and the current guidance replay supports both bounded and whole-episode
kinds. The selected declaration pins exact source hashes. Native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Evidence, wheels and build receipt: `/x/Temp/kindle-cheap-20261025`;
`original-native` preserves the prior AMD binary. Both installed and source-tree
extensions now use the new native build; old completed queues are historical.
Use these reports for current host, precision, results and evidence details.
The historical handover steps below describe its origin.

The goal remains strong, reliable learning on **Boxing, Pong, Freeway, Breakout
and Qbert**, then native games and transfer. Training is the priority; another
backend optimization/JAX-parity campaign is not the next step.

Read this file, the [short roadmap](kindle_single_life_dreamer_plan.md), and the
[Breakout experiment](experiments/2026-10-23-cdp-breakout-context64.md). The
[PR31 description](https://github.com/kvark/kindle/pull/31) is the live dashboard.
Historical reports retain failed attempts and limitations; their old next steps
are not instructions to restart those queues.

## Source you actually need

The working branch is **`phase2-levjepa-evaluation`**, continuing PR31.
It now contains the qualified c8aeba28 library implementation directly, with
portable Git dependencies. There is no separate takeover branch to switch to.
The previous356-commit work history and detailed experiment corpus are retained
on [archive/phase2-before-cleanup-20261025](https://github.com/kvark/kindle/tree/archive/phase2-before-cleanup-20261025).
The working branch keeps current results, source and tests; no `runs/` tree,
raw experiment data or checkpoint history is committed.

| Dependency | Revision |
| --- | --- |
| [Meganeura](https://github.com/kvark/meganeura/tree/handover/cdp-qualified-20261025) | `423d373634a1b1f04ff623db2b419757f1e8d96d` |
| [Blade](https://github.com/kvark/blade/tree/handover/cdp-qualified-20261025) | `bbbc93b7d26b8196d9143944fe7d8a61d6da4583` |
| Naga fork | `13f582ddd2792befb46952498010248bd65818e6` |

These preserve the measured implementation, not a fresh upstream port. Only the
user merges PRs. Cargo fetches the dependency branches without sibling checkouts
or old absolute paths. The prior handover verified329 library source files
against the measured implementation and passed CPU workspace/Python compilation.
`tools/handover/verify_sources.py` and its manifest preserve the source receipt
and eight explicit cleanup differences: Rust formatting, actual dependency
revision reporting, and two stale CPU test repairs. Learning operations and
checkpoint architecture validation are unchanged. A separate Meganeura
portability repair records GPU commands directly on Metal/GLES, whose Blade
backends do not expose Vulkan secondary encoders; the Vulkan path is unchanged. The full CPU Rust suite passes
(127 tests;78 GPU tests ignored), as does formatting.
No GPU qualification or speed claim follows from CPU checks. Rebuilt binary
bytes can differ with paths, compiler and dependency metadata. Rebuild on the
new machine rather than importing the original Linux/CPython3.14 binary.

Example setup (Python3.11+; original environment was3.14.4, Rust1.98.0):

```sh
git clone --branch phase2-levjepa-evaluation https://github.com/kvark/kindle.git
cd kindle
python3 -m venv python/.venv
. python/.venv/bin/activate
python -m pip install -r tools/handover/requirements.txt
cd python
maturin develop --release --locked
cd ..
```

The requirements record the source machine's versions, rather than silently
upgrading the environment. Install the target's Vulkan driver and obtain Atari
ROMs separately; ROMs and the Python virtual environment are not in the transfer.
A release build is not a successful GPU initialization check.

The measured runtime enabled:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MEGANEURA_COMMAND_CACHE=1 KINDLE_DEVICE_MEANS=1
export MEGANEURA_IMAGE_VIEWS=1 MEGANEURA_CONVOLUTION_WORDS=1
export MEGANEURA_COPY_COMPOSITION=1 KINDLE_DEVICE_TARGETS=1
export MEGANEURA_CONVOLUTION_REUSE=1
```

Select the target GPU/ICD using that machine's actual configuration. Do not copy
`MEGANEURA_DEVICE_ID=0x2c02`,5080 name assertions, old driver versions or boot IDs
onto another GPU. Declare actual device/driver expectations for the host guard.
These flags retain the qualified implementation; capability fallbacks and
performance on other GPUs still need checking.

## Transfer package

Shared destination: **`/mnt/data/kindle-handover-20261025`**.
`manifest.json` identifies every included artifact by original path, size and
SHA256. `archive-review.json` records independent verification of all304 members.
The artifact archive is392,172,204 bytes (about374MiB); it contains no build caches.
`README.md` there lists package contents and source revisions. Source Git bundles
provide an additional fallback to the pushed branches.

Extract into a new local directory, after checking the archive SHA256 against
the manifest. Never extract over an existing run. The archive includes:

- `context64/frozen-checkpoint`: recovered867-update B2/T64 weights, and the
  successful frozen-restore review. See the recovery note below.
- `breakout-b8/2m-frozen`: recovered source-control export from the warm-start
  evaluation, learner counter499939 and environment counter2027152. All346
  tensors are finite. It is a frozen derivative of the2M control; the complete
  original training logs and initial/500k checkpoints are not included.
- `diagnostics`: retained PMU captures and prepared pre-submission collector.
- `original-runtime`: original native package, build receipt and example helpers.
  Historical scripts contain absolute paths and host pins: treat as evidence,
  not ready-to-run controllers on another machine.

**Originals located:** the shared-drive cleanup moved the runs to
`/x/Artifacts/shared-drive-cleanup-20261025` on the original machine; they were
not lost. Verified original checkpoints are:

- `kindle-cdp-breakout-context64-20261023.5OA15Iew/smoke/checkpoints/4096`:
  learner867, environment4096.
- `kindle-cdp-breakout-budget-20261021.wqYChxy8/checkpoints/2000000`:
  learner499939, environment2000000.

Both originals match all346 tensors in their corresponding shared-package
frozen exports exactly. The exports have advanced environment counters5120 and
2027152; preserve their metadata rather than relabeling them as original saves.
The shared package remains usable without transferring the whole local archive.
Use the saved originals when their original counters or full run evidence are
needed. Neither version preserves replay, games, RNG or live belief.

For historical artifact paths, look under the relocated run directory above.
The package is deliberately not a copy of every profiling dump or checkpoint.

## First action on the new machine

1. Build the preserved runtime and record source, compiler, Python, GPU and driver
   identity. Check available memory/disk. No backend port or optimization first.
2. Run a bounded initialization/frozen restore under `gpu_host_guard.py` or
   `run_gpu_queue.py`, using the recovered867-update checkpoint and an output
   directory created for this check. Use eight environments, native observation
   size, the published Atari protocol and sticky actions.25. Frozen evaluation
   must produce zero updates and preserve all346 tensors. A1024-action check
   suffices for this startup boundary; it is not a learning evaluation.
3. If initialization succeeds, finish the smoke review and run the declared
   fresh Breakout1009 B2/T64 life, with500k and2M checkpoints and the actual
   initial control. Eight streams;128 replay positions/update;ratio32;12M/BF16;
   centered CDP and action-effects coefficient1; no Freeway action aid. Expect
   first update at action632,124843 updates at500k,499843 at2M. Full life is
   fresh, not a continuation from the smoke or prior warm-start8M model.
4. Evaluate fixed milestones with24 natural sampled episodes each (three per
   stream), zero updates and full replay/videos. Compare to retained B8/T16
   scores58.83/98.21/1.42. Disclose different hardware/build, and the coupled
   sequence-length/independent-batch change. Review before any extension or seed
   replication. Stronger Qbert learning follows useful Breakout evidence.

See the existing runner's `--help` and the archived declarations for the complete
protocol. Do not launch the old absolute-path `run.py train` unchanged. Retain
ordinary guard checks, accounting and actual-initial/frozen comparisons; do not
turn relocation into another weeks-long qualification matrix.

## GPU fault scope

On the original host, ordinary restore repeatedly produced Xid62/PMU halt on
615.71.09 and595.91.07. One traced restore succeeded; an ASLR-preserving trace
later failed while waiting for the first world's device-local buffer fills,
before checkpoint loading/model execution. IOMMU errors followed30s later.
This localizes the wait, not the root cause. Neither hardware, driver/firmware
nor invalid runtime arguments are conclusively excluded.

No blind retries on a faulted device, and no automatic reset/reload/reboot.
The prepared collector records fill handles/capacities/offsets/sizes/values and
stops before Blade submit. Six CPU cases passed; native capture remains unrun.
Its ABI and symbols apply only to the original binary. On another healthy GPU,
start with the bounded restore above; do not inherit a historical machine-wide
stop or presume the failure reproduces there. Stop and collect evidence if it
actually faults. No training was launched during this handover.

## What not to redo

Do not restart Pong2M, the rejected B32 Freeway recipe, unchanged Breakout
warm-start extensions, the45-run old matrix, kernel geometry sweeps or JAX parity.
No concurrent/swarm learner project. Current five-game scores and limitations
are in the roadmap. User-authored `TASK.md` and `docs/cdp-claude-feedback.md`
are now tracked unchanged as historical intent/review, not current measurements.
