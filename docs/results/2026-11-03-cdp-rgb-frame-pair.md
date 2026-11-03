# Native RGB frame pairs: implementation qualification

The optional six-channel input passes implementation, legacy compatibility and
bounded execution/cost checks. Training8192 actions /489 updates takes51.357s,
7.69% more than the retained single-image check; padded200k projection is22m59s.
Conservative full-replay headroom is6.097GiB. Frozen1024 actions contain no natural
episodes, so gameplay benefit remains unmeasured. That qualification selected
no full learner; the separately selected comparison below is now active.
This work is part of the existing five-game goal; Breakout retains70.750 sampled/
143.667 greedy and its original864/90%-two-wall gate.
See the [fixed protocol](../experiments/2026-11-03-cdp-rgb-frame-pair.md).

## Fresh 200k comparison is active

The [separate full protocol](../experiments/2026-11-03-cdp-breakout-rgb-frame-pair-200k.md)
selects one fresh200k-action/12,477-update comparison, after bounded closure,
with a30-minute training cap. It launches **23:33:53.049 UTC November3**, source
`8ea3268158763bee64dac060ee7db5d8b4e66c37`, main PID1087230/invocation
`675fde529a5a4dc080adc6d655f2312b`. Root is `full/` beneath the package root;
unit `kindle-breakout-rgb-pair-200k-20261103`. Recovery observer PID1087236
finishes successfully after the100k verification; terminal observer PID1087233
remains active. No full frozen result is available.
Do not restart this learner or schedule overlapping GPU work.

Keep12M64/Auto/N8/B16/T16/H15/R16/context1,20% guidance/cap1920/all-state labels,
expert100/replay-value1 and base/encoder4e-5,dynamics4e-4,behavior8e-5. Sampled
collection and posteriors remain. Primary frozen evaluation is greedy, with
sampled secondary, final and actual-initial24-episode arms each. Compare against
retained143.667 greedy/70.750 sampled, using each model's own initial and actual
training time. The original864/90%-two-wall gate stays unchanged. No extension,
second learner or automatic successor is selected.

Full18 teacher policies have identical AST definitions and schedule methods;
the newer source adds optional four-action support. CPU checks cover2048
synthetic actions, ordinary episode selection/resets, feedback and bounded caps,
with no emulator/native construction. The first fixture incorrectly omitted a
required cap for the older bounded guidance kind; its source/error remain in
`full/teacher-preparation-failure.json`, and the corrected check passes0.509s.
This does not establish complete training equivalence or a causal frame-pair
effect: native, runner, teacher and schedule source hashes differ from control.

Changed helper qualification passes3.207s, including the complete comparison
entry point with synthetic outcomes and10 corrupted-header rejections. It also
checks accounting, own-initial comparison rules, replay argv/manifest paths and
null terminal journal messages. The five actual queued commands pass0.302s
without real native/emulator construction or GPU descriptors. All114 declaration
pins are verified. Historical source resolution preserves the exact old replay
script without rewriting its original declaration.

Startup passes all357 initial tensors against the paired bounded check,47
pre-update semantic events and first-update action376. It does not claim equality
with the three-channel initial model. The1.915s startup CPU audit remains in
training wall time. At the recorded live check,190 finite reports reach3416
actions and the original main/recovery/terminal identities remain active.
Require the read-only100k/6,227 recovery and full final/initial frozen assessments,
replays/videos, curves, exposure and comparison before drawing a result.

### The 100k recovery passes while the original learner continues

The main observer saves100,000 actions /6,227 updates and checks the read-only
checkpoint at23:45:51.707 UTC, taking0.310s. The independent observer passes at
**23:45:53.602 UTC November3** in0.343s (0.414s including its wrapper), with all
114 pins unchanged and357 finite tensors:280 world,66 behavior,11 slow value.
It verifies geometry, optimizer arrays, metadata hashes and0555-directory/0444-
file modes while preserving original main PID1087230/invocation675fde529a5a4dc080adc6d655f2312b.
Recovery contains weights/moments, not replay, games, live beliefs or full RNG.
The recovery service is terminal success; main and terminal collector remain live.

A subsequent observation reaches113,592 actions/7,077 updates, with replay length
100,000. The memory snapshot at113,600 actions records **5.019GiB** remaining
GPU budget. This is above the actual2GiB guard but below the bounded6.097GiB
projection; filling replay can coincide with other allocations. It is an observed
snapshot, not the final minimum or proof that all later work fits. The existing
memory guard remains active and no cap or recipe changes.

Two optional CPU status-readback attempts fail before producing their summaries:
one uses the run-end `learner_updates` key on a progress row; the other searches
for spaced JSON in compact JSONL. Their receipts are retained as
`full/live-observer-2339-failure.json` and `full/post-recovery-readback-failure.json`.
The corrected readback uses parsed rows and confirms the same active identity;
`full/post-recovery-observation.json` records its observation. These are separate
from the passing recovery service. No learner, observer or GPU stage restarts.
CPU co-host times remain in measured wall time without correction.

## Implementation and package

Production source `24e701bce7e65004a84f54fa29073fc4ef1f3355` adds default-off
`rgb_frame_pair`. Each arriving native pooled image is resized once on GPU,
then paired with the same stream's previous resized image, oldest first. The
first image after reset is duplicated. Missing arrivals preserve that stream's
private history. Replay stores both images together; restore retains geometry
and starts fresh live history. RGB reconstruction and color-centroid supervision
reject this option. Only the first convolution widens3→6, adding2,400 weights
at12M. Later widths stay fixed; RGB replay storage doubles.

The isolated package root is `/x/Temp/kindle-rgb-frame-pair-20261103`.
`build-receipt.json` records220 source pins, compiler/dependency identity and
separate build/test logs. It completes at23:04:52 UTC November3 with171 Kindle
and5 gym Rust CPU tests,993 packaged Python tests (10 deselected), strict
workspace/binding Clippy and a built wheel. No native agent is constructed by
the packaged Python CPU suite. Runtime is `runtime/`, native SHA256
`b451bc981a76a7ccc455ca22ffe637151d99353ea520fb9642cae217e8cc73e9`.
Original validation executable SHA256 is
`d3af6742847fa5b3a31a1e10f9a0832487ffb7fa92218edf5ae2440c03488600`.
The disposable build cache is released after bounded closure; the original build
receipt remains immutable and `packaging-final.json` records the release.

## Geometry passes; original derivative failure is preserved

The declared serial native queue passes the independent32/48/64 GPU resize/pair
fixture, including multiple streams, missing arrivals, resets, and CPU/resident
frame sources. The test takes0.75s; its host guard passes. It will not repeat.

The second fixture passes scalar F64 CNN outputs/loss but fails a derivative
assertion: GPU0.057106383 versus finite difference0.056595063837576225.
The difference0.000511319 exceeds the original tolerance0.000469785. The native
process exits101; there is no recorded GPU fault or host recovery. The original
queue stops, preserving its logs, declaration and `native-command.json` exit1.
The third replay/restore fixture remains unstarted at this point.

## Targeted derivative diagnosis

Test-only source `3d24bd20` adds an exclusive capture of parameters, gradients,
pixels and scalar/GPU outputs from the same seed103 six-channel fixture. It
changes no production source or packaged native binary. The original test source
is saved byte-exact as `derivative-diagnostic/rgb-tests-original.rs`, resolving
the sole changed path among the220 original source pins.

The diagnostic test binary builds successfully. Initial CPU preparation assumes
only one Cargo test executable, but package/workspace builds have different
hashes; the assertion stops before queue creation. A subsequent command fails
on the missing queue file. Both errors are retained in `preparation-failure.json`
and original queue logs. Corrected preparation selects the exact executable from
the successful build log; no build or GPU geometry stage repeats.

One new guarded derivative capture passes in0.80s, with no optimizer updates or
Atari actions. CPU analysis takes0.119s. Its independently implemented NumPy F64
CNN agrees with the Rust scalar outputs to2.89e-15 and GPU outputs to1.61e-6.
Five fixed perturbations,1e-3 through1e-7, cover the same five selected derivative
probes, including both input-image halves. The failing probe is first-layer
weight75, in the current-image half. At1e-4, the positive/negative perturbations
change two/one max-pool winners in the second layer. At1e-5 and1e-6 no winners
change, and the derivative converges to0.057106416561, within3.35e-8 of the GPU.
All five probes converge with no pool changes at those smaller steps.

The fixture correction uses1e-5 for paired input only, preserves1e-4 for the
legacy three-channel test, and leaves the acceptance tolerance unchanged.
This diagnoses the fixed fixture's discrepancy; it is not broad numerical or
gameplay evidence. Captured data, analysis, all failures and both test sources
remain under `derivative-diagnostic/`.

## Corrected native checks and legacy compatibility pass

Source `b58333ca` changes only the paired test perturbation and documentation.
The corrected derivative test passes in0.81s with its original tolerance; the
previously unstarted replay/learning/restore test passes in1.16s. The latter
performs one synthetic learner update, verifies paired replay remains unchanged,
checks learned CNN weights and Adam moments after restore, and checks frozen
invariance. Original GPU geometry is not repeated. All220 build source pins
resolve (one original test source is archived),13 remaining-check pins pass,
and all three successful host guards pass. The packaged native stays unchanged.

The first CPU summary wrapper incorrectly looks for full-result exit-code fields
in the compact guard-audit return and raises `KeyError`. A separate corrected
CPU review uses the audit's documented return and preserves the original source
and failure receipt. No GPU stage repeats. `native-qualified-review.json` closes
native checks as `complete_with_corrected_derivative_fixture`.

Legacy actual-initial and final greedy checkpoints each run1024 frozen actions
with the new runtime. Their128 transition records match the retained evidence
exactly apart from finite wall timestamps; all357 tensors in each remain
unchanged, with zero updates. Times are0.898s initial and0.889s final. All36
input pins and both host guards pass. The first CPU comparison rejects raw
config equality because the new runtime materializes the pre-existing default
`replay_value_bootstrap=imagined`, absent from old metadata. A separate CPU
correction normalizes only that exact field; every other config field still
must match, and `rgb_frame_pair` remains omitted. Original review/logs/failure
are preserved; no compatibility arm repeats. Evidence is under `legacy/`.

## Bounded cost and memory check passes

The native prerequisites and legacy compatibility pass. CPU bounded preparation
checks the locked configuration, cadence, guidance, command paths and independent
checkpoint schema, including both added first-layer Adam moments. It takes0.080s
with13 input pins. Only the three first-layer weight/moment shapes change2400→4800;
there are357 tensors and16,336,753 trainable parameters. Full100k replay contains
6×64×64 observation values plus2048+32×16 recurrent values per row, allF32:
10,854,400,000 bytes. The headroom check conservatively subtracts this entire
footprint from minimum observed bounded headroom, counting already allocated
replay pages twice. This is stricter than subtracting only remaining pages.

The fixed8192-action/32,713-frame learner completes489 updates in **51.357s**,
versus47.690s retained (+7.69%). Construction is separately2.333s. Padded200k
projection is **1,379.227s (22m59.227s)**, below the30min gate. Frozen1024 actions
take **0.877s**, with zero updates, no natural completed episodes and all357
tensors unchanged. All357 initial/final/frozen arrays are finite with independently
derived shapes. CNN/reward/actor/value weights change. Initial values are not
compared against the three-channel architecture.

All489 reports are finite and supervised, with117,025 repeated labeled replay
rows. Actual overrides are3,757/8,192 (45.8618%). Full recorded training pixels,
actions, labels and overrides reproduce in5.416s CPU replay; all42 bounded pins,
220 resolved build source pins and both host guards pass. No bounded failure or
stage repeat occurs. Native and runner differ from the retained447fe50 recipe;
input geometry, first-layer initialization and trajectories change, so this
historical timing comparison does not isolate input cost or imply gameplay gain.

Across3,098 memory snapshots, minimum observed GPU headroom is17,401,413,632
bytes. Deducting the entire full100k replay footprint leaves **6,547,013,632 bytes
(6.097GiB)**, above2GiB. This deliberately double-counts already allocated replay
pages; it is an early conservative projection, not a measured full-memory run.
Learning takes44.705s (87.05% of training), acting2.484s and environments2.538s.
Light source/document review during this bounded run is retained as a co-hosting
caveat; no wall-time correction is applied.

The controller PID1086637/sourceb58333ca finishes successfully at **23:23:19.862
UTC November3**,66.664s including guarded stages and CPU audits. Source and
runtime are unchanged; the controller is `bounded/run.py`, results under
`bounded/check`. No full learner, extension or automatic successor is selected.
After completion the995,602,644-byte apparent build cache is released under the
repository's packaging instruction. Wheels, runtime and all three validation/
diagnostic binaries remain pinned. Original build receipt and all failures
remain immutable. No GPU allocation remains active at closure; check live state
before any new work.

A full frozen gameplay comparison is the remaining question for this input.
The failed fixed action readout did not provide positive evidence for it, and
this short check establishes execution and cost only. A separately justified
comparison must keep the original Breakout gate and actual-initial controls.
