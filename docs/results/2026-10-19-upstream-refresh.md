# Latest upstream: recover fast paths, retain the JAX gap

October19,2026. [Refresh contract](../experiments/2026-10-19-upstream-refresh.md)
· [Dense-policy attribution](../experiments/2026-10-19-dense-policy-attribution.md).

The updated stack now reproduces the retained RGB/CDP synthetic updates and
performance: **RGB42.983ms, CDP40.741ms**. RGB remains **1.579× inclusive JAX**.
Two integration changes had bypassed substantial existing optimizations;
restoring them recovers our previous speed, not a new speedup. This is
**current upstream plus retained isolated extensions**, not stock Meganeura.
Keep native`d3ff4948` as the current-upstream isolated control and`e45b9c43`
as its historical anchor. No production promotion, Atari training or
learning-parity claim follows. [Compact evidence](2026-10-19-upstream-refresh.json).

## Complete-update comparison

Meganeura`c19cb1ca`, Blade`4845d652`, Naga`13f582dd`; RGB/CDP12M, B32/T16/H15,
explicit BF16 and all previously qualified backend/resident-target features.
Each arm runs16 warmups and256 measured complete updates. All samples remain.

| Complete update | Old qualified stack, fresh control | Refreshed ordinary dense policy |
| --- | ---: | ---: |
| RGB |42.912ms|57.108ms,33.08% slower|
| CDP |40.701ms|65.495ms,60.92% slower|

Control drift is−.660%/−.722%. RGB's refreshed ratio is2.098× caller-inclusive
JAX27.223ms; this failed candidate does not replace the43.197ms/1.59× anchor. CDP has
a different exploration workload and is not another numerator for RGB/JAX.

All272 paired metric reports, first targets and initial/first/final292 RGB and
346 CDP tensor sets match exactly. Each arm reproduces its own24-update prefix
and checkpoint restoration. Exact reproduction on these fixtures is stronger
than the declared finite-execution gate, but not an independent full-model oracle
or learning evidence. The predeclared performance gate fails.

The regression appears in every stage, especially CDP imagination/targets
17.200→32.834ms and RGB behavior3.416→6.546ms. These are complete host-stage
boundaries, including GPU waits, not individual shader latency or utilization.

## A substantial implementation difference

The qualified dense policy uses its64x64 shared/reuse implementation for eligible
shapes, with checked fallback. Upstream's ordinary typed path uses32x32 blocks.
RGB changes587 dense dispatch implementations; surrounding dispatches, products,
buffers and convolution selections remain unchanged. Blade/Naga also changed,
so this initial comparison alone cannot attribute the time difference.

The separately declared whole-policy comparison restores the old dense lowering
on the same new dependencies. All eight restored RGB/CDP program plans, constant
payloads and pipeline identities match the old control, including all272 reports,
first targets and initial/first/final tensor sets in the complete comparison.

| Complete update | Refreshed default, fresh control | Restore whole qualified dense policy | Versus original fresh control |
| --- | ---: | ---: | ---: |
| RGB |57.092ms|46.380ms,18.76% less|8.08% slower; recovery gate fails|
| CDP |65.353ms|40.844ms,37.50% less|.35% slower; recovery gate passes|

The fresh refreshed-default controls calibrate−.029%/−.218% against the preceding
comparison. The declared recovery gate is within5% of the original qualified
stack. Its RGB failure stays failed, including the original review assertion;
a separate CPU terminal audit completes both workloads without changing that
gate or repeating GPU work. No promotion. Restored RGB is1.704× inclusive JAX,
still worse than the retained qualified result.

Restoring one policy recovers a large part of the regression, not the original
JAX gap. The remaining RGB difference is concentrated in world training:

| Complete host-stage boundary | Original qualified control | Restored policy on new dependencies |
| --- | ---: | ---: |
| Posterior |7.043ms|7.171ms|
| Imagination and targets |7.662ms|7.668ms|
| World training and synchronization |24.790ms|28.125ms|
| Behavior training and synchronization |3.416ms|3.416ms|

World increases3.334ms; the corresponding CDP boundary is12.469→12.473ms.
These include host work/GPU waits, not kernel-only measurements. Do not dismiss
the increase as noise or trim samples.

## The remaining integration difference

The CPU audit finds204/205 RGB and224/225 CDP captured WGSL contents identical;
changed filename prefixes are not shader changes. One actual5x5/stride1 BF16
forward-convolution source differs. The new `Variant::Scalar` route bypasses
the retained vectorized/incremental generator, even with
`MEGANEURA_CONVOLUTION_WORDS=1`. The plan and pipeline labels did not change.
This route occurs at RGB world dispatches91/1680/1791 (first encoder, two decoder
products), but only CDP153 (first encoder). It is an integration omission,
not proof of an upstream compiler regression.

Both Naga libraries compile and offline-validate all251 modules from the complete
246-source union.250 outputs are byte-identical under fixed sequential bindings;
`rms_norm_grad_w` differs, including after ID compaction, with unchanged opcode
counts. Keep that difference: these are not captured driver modules or proof of
identical driver compilation. The first filename/source equality assertion stays
failed; the separate union audit includes both differing sources and drops none.

The [separate convolution-route comparison](../experiments/2026-10-19-convolution-route.md)
restores the existing generator without new math or selection. The complete
comparison confirms the effect:

| Complete update | Restored dense policy, fresh control | Also restore convolution route | Versus original qualified control |
| --- | ---: | ---: | ---: |
| RGB |46.350ms|42.983ms,7.26% less|.17% slower|
| CDP |40.803ms|40.741ms,.15% less|.10% slower|

Both original recovery gates pass. New controls calibrate−.065%/−.102% against
the previous comparison. RGB world training falls28.065→24.852ms; posterior
7.165→7.040ms. Thus this route accounts for the measured remaining integration
regression, not the original JAX gap. It is not a new convolution design or
proof of isolated vector-load versus indexing contributions.

All272 reports, first targets, initial/first/final292 RGB/346 CDP tensors,
checkpoint restores, own prefixes and full eight programs/constants/pipeline
identities match exactly. All205 RGB/225 CDP captured WGSL contents now match
the old control as multisets. First/last32 RGB means42.926/42.985ms and CDP
40.747/40.753ms retain all samples. Equality of these synthetic trajectories
does not establish independent full-model correctness or learning parity.

## What is still missing for JAX

Two distinctions matter before proposing another optimization campaign:

1. **The latest search work is not enabled by ordinary construction.** Kindle
   calls `build`, not `build_measured`. Furthermore, the upstream typed candidate
   domain offers checked and rectangular workgroup kernels; its larger
   `CooperativeTiled` candidate still rejects typed storage. Simply turning on
   search does not make our retained larger-reuse BF16 implementation available.
   The useful integration is one typed implementation family in the selector,
   with complete-program numerical qualification and held-out timing—not a new
   collection of per-shape overrides. No search is run in this refresh.
2. **Convolution feeding remains the major measured structural deficit.** The
   retained actual-model counter capture measures4.255× JAX convolution warp
   instructions,7.287× ALU instructions, despite fewer tensor instructions.
   Current source already has BF16, split-K, incremental indexing, packed views
   and cached recording. These ratios are historical counters, not new-compiler
   counters, latency fractions or a4× speed promise. They identify work outside
   the useful tensor arithmetic, not proof that one missing instruction fixes it.

The remaining gap is not explained by model size or slightly more arithmetic:
the audited reachable heavy arithmetic differs by1.40%. It also cannot all be
assigned to convolution: retained cached timing has18.060ms in convolution-only
intervals and21.850ms elsewhere. Recurrence/matrices and materialization remain
separate targets. These intervals are not additive attainable savings. See the
[actual-model counters](2026-10-17-model-counters.md),
[completion intervals](2026-10-17-boundary-timing.md), and
[recurrence contracts](2026-10-17-recurrence-contract.md).

Next integrate the already-qualified typed reuse family into the common backend
selection contract, preserving it as an incumbent before trying measured search.
Latest commits chiefly improve search and qualification; `build` does not run
them. Qualify complete programs and generated sources, not labels alone. Then
use the same full-update boundary to test whether measured selection actually
helps. This does not reopen rejected manual geometry/family sweeps. Convolution
feeding and non-convolution recurrence/materialization remain separate structural
targets; no evidence yet supports a single switch that removes the remaining
15.760ms. CDP lacks RGB reconstruction, so its optimization priorities differ.

## Corrections and evidence retained

The first integrated generator contained **our merge error**, not an upstream
bug: a legacy untyped early return intercepted new typed calls, reading BF16
buffers as F32 and discarding output storage. A sparse32x256x18 probe fails all
8,192 outputs. Restricting that branch to untyped calls repairs it.18 independent
CPU storage contracts,96 shader variants/offline SPIR-V checks,884,736 exact
sparse matrix values, finite first model outputs and unchanged292 tensors pass.

Earlier BF16-counter and duplicate create-new diagnostic-export failures remain.
The failed full update completed one world backward/optimizer before its invalid
target guard; it did not complete behavior training or advance the learner step.
The repaired diagnostic's controller separately failed on a missing inherited
timing-result budget field. Native initial budget assertion and data checks pass;
the missing export stays incomplete, with a separate CPU review and no GPU repeat.
Merge/import, shader-fixture, Rust path/borrow, Cargo package-path and CPU observer
failures remain with source and logs. No numerical tolerance was loosened.

The first guard stopped before a child on nine old October18 application-named
Xids. The separate exact cursor/message review checks absent PIDs at baseline;
every new hard fault remains fatal.82 guard tests pass. No driver recovery or
separate NVML polling was performed. This is not a hardware-health certification.

The convolution repair's initial CPU build selected the copied manifest's old
backend, and its source probe omitted the declared word-staging flag. Both
failures remain; the repaired manifest is checked against Cargo's actual receipt
and the probe uses the original flag. Source equality and Clippy pass before GPU
work. Its smoke-review logger appended to the predecessor CPU log; the original
prefix remains, and a separately versioned timing wrapper uses its own log.

Combined totals:26 guards,21 passing/5 failed,21 allocation warnings;
3,456 complete synthetic updates,one partial world update,19 standalone dense
calls,one additional posterior/imagined-forward diagnostic,zero actions.
Retained smoke controls are not counted twice. All failed starts and warnings
remain. Final CPU review passes2,464 source seals at06:09:49UTC; no worker remains.

Artifacts: `runs/upstream-refresh-20261019.KWEh8dEH`. Corrected new-default native
`01f0d42d`, dense-policy candidate`bf26b636`, fully restored current-upstream
control`d3ff4948` under `conv-route/`. Separate source trees and Cargo caches
preserve every earlier source/binary. Keep all six original backend/resident flags
from the sealed queues. User worktrees and untracked `TASK.md`/
`docs/cdp-claude-feedback.md` remain untouched. CI status belongs on PR31.
