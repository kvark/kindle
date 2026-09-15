# Recovery verification and an initialization-only control

Status: **health restored; one guarded initialization-only control passes**.
The user's reboot is observed as boot `efe90b23-aba2-4fcf-87c6-380ed64ba3a2`,
different from the [third incident](2026-09-14-pixel-initialization-incident.md).
Retained boot history begins this boot on September 14 at 14:07:36 UTC; the
verification and control execution occur on September 15.
The agent performs no reset, module/service change, driver change or reboot.

## Recovered host and current upstream

The [fresh capture](../../runs/gpu-recovery-20260915.jc4xSS/health/result.json)
checks the current kernel journal before its single bounded NVML request, then
checks the journal again. It finds no current-boot NVIDIA fault, matching loaded,
on-disk and reported driver **595.91.07**, kernel **7.0.0-31-generic**, the expected
RTX 5080 UUID/PCI identity, recovery action **None**, and **15,841 MiB directly
free**. The sample reports 0% activity, 33°C and 11.84 W. These are one recovery
observation, not runtime qualification or an idle-gap measurement. Host and
health seals independently verify 21 and nine pins; no unfinished child remains.

Fresh [upstream source captures](../../runs/gpu-recovery-20260915.jc4xSS/upstream/result.json)
find Meganeura **4f8c7689** and Blade **6ab5fcec**. Meganeura's `src`, Cargo files
and build-script paths are unchanged from 428fc2d; intervening changes concern
documentation and paper artifacts. Blade adds shader-validation capabilities
for packed/quantized float values, subgroup barriers, multisampling and cube
arrays. These are real graphics-source changes, not the earlier renderer-only
updates, but neither upstream diff supplies an observed initialization fix.
Keep them in scope for a separately identified latest candidate. Do not silently
apply them to the historical control or call documentation changes a runtime fix.

## The narrower control

The failed 0a98775/native 02b600a1 bundle and its block carry stay quarantined.
The new fixture starts from the qualified **ce80 / published Blade 0.9.0 control**.
It changes observability and the diagnostic endpoint, not allocation policy,
immediate Shared zeroing, learning arithmetic or cooperative-policy selection.

The fixture verifies the actual encoder hash before creating a device, loads
six-stream native LeVJEPA, keeps it resident, constructs all eleven production
CPU graphs in their original order, and initializes only the first world GPU
session. It then explicitly drops resources and exits. N6 / 12M / B16 / T64 /
full recurrence / R256 / seed7301 / causal loss match the original pixel windows.
There is no observation, action, D3 initialization, learned update, checkpoint,
restore or later world/behavior GPU session. This tests the relevant combined
initialization context, not a substitute tiny hardware workload.

Both layers emit flushed records: Blade's
[allocation observability](2026-09-14-vulkan-allocation-observability.md) records
actual allocator placement and buffer/bind boundaries; Meganeura additionally
records constant-upload boundaries and retains the three checked initialization
waits. The fixture checks the actual NVIDIA adapter/driver and device selection.
Instrumentation changes host timing. Allocation requests are not individual
`vkAllocateMemory` calls; memory handles may be reused, and the trace is not a
residency, lifetime-safety or hardware-fault-origin proof.

The versioned source is Kindle **ae7699ad**, Meganeura **9b9e7ee7** and shared
Blade **c96a9a87**; the diagnostic branches are committed and pushed.
They are instrumented historical-control revisions, not newly adopted upstream.
Both dependency locks and Kindle's reported identity must agree. Main stays ce80.

## CPU validation and preserved failures

The first [preparation](../../runs/combined-init-control-cpu-20260915.esQPVh/declaration.md)
stops before compilation because all-platform offline metadata needs an uncached
Android dependency. Its [Linux-only continuation](../../runs/combined-init-control-build-20260915.UYZCaF/declaration.md)
release-builds the fixture and passes 82 tests, but correctly fails the existing
Git-revision/dependency-lock assertion with local path overrides. Preserve both
failures, binaries, source paths and writers. No GPU test runs in either attempt.

The [versioned preparation](../../runs/combined-init-pinned-cpu-20260915.7cVANq/declaration.md)
uses explicit Git dependencies and matching workspace/Python locks, not an
identity-test skip or fallback. The fixture and Meganeura runtime source remain
byte-identical to the path preparation. CPU coverage is 83 Kindle tests,
twelve Blade tests and nine initialization trace/wait tests; all 23 Kindle GPU
tests remain ignored. Release formatting and Clippy cover the three components.
The private target uses one build job / 100% CPU quota, a 2 GiB host-memory cap
and zero swap. The mocked device-loss message belongs to a CPU wait-error test.
All sixteen command lifecycles, 754 inputs and 100 artifact/source pins re-audit
after the source commit. No cache artifact is relabeled as a fresh native build.

The separate [CPU reader review](../../runs/combined-init-control-runtime-20260915.kePI62/status.md)
tests eighteen synthetic traces and compares actual Cargo sources, including
WGSL, against the pinned source worktrees. It checks the complete pixel config
and preserves native/source depfile identities. Its directory name does not
declare a runtime job. No launcher, GPU result, Python wheel or adoption follows.

## Completed native control and independent reader

The separate [one-job declaration](../../runs/combined-init-control-20260915.2bZO4V/declaration.md)
binds 926 inputs and ten passing CPU launch/refusal checks. Its fresh upstream,
boot/kernel/driver/UUID, idle/memory and complete source/weight preflights pass.
At 05:41–05:42 UTC the sole native fixture completes both sessions, drops its
resources, and exits zero. The [guard](../../runs/combined-init-control-20260915.2bZO4V/guard/result.json)
passes and reaps PID 20382; it records no kernel fault or unfinished child.

The original controller then stops at its report reader's `wrong fixture inputs`
check. The fixture serializes typed F32 configuration through `serde_json::Value`,
which widens those values to F64; the checkpoint uses short F32 decimals. The
separate [read-only interpretation](../../runs/combined-init-control-audit-20260915.P7XIB3/interpretation.md)
preserves this failure and canonicalizes only the expected F32 representation.
Actual values are not rounded; even a one-F64-ULP change is rejected. Eight CPU
fixtures pass, then the complete unchanged native result verifies. This is no
numerical tolerance increase or successful rerun of the failed controller.

The [independent result](../../runs/combined-init-control-audit-20260915.P7XIB3/cpu/result.json)
verifies both exact complete alias plans: 632 frontend and 9,439 world physical
slots, their original interleaved Shared zeroing, 10,081 complete buffer/allocator
request pairs, 339 frontend and 14,306 world constant-upload pairs, every checked
wait and all 143,121 trace records. Actual memory types are recorded as 1 and 4;
no historical candidate placement is inferred from them. All 210 fresh health
samples pass, maximum gap **0.5134 s**, minimum directly free **4,977 MiB**.
There are exactly two initialized sessions and **zero actions or updates**.

The native invocation is completed and must never be restarted. Preserve the
original controller's failure and absent top-level result; use P7XIB3's
`audit_control.py --audit` for complete read-only verification. No new GPU query
or execution was needed for that correction. There is no follower or successor.

## Next boundary

The instrumented historical control can initialize the relevant combined
frontend/world context on the recovered host. This supplies a matched
observability baseline, not a cause or fix for the failed candidate. Before
another candidate execution, prepare a distinct, reviewed initialization
hypothesis on current upstream with the same observability; do not replay the
quarantined bundle. Retain exact source/artifact/weight/configuration identity,
fresh host/upstream checks, direct-child guard and all memory/device gates.

Never restart the failed pixel window, completed writers or the old queue. The
root-2017 hold still has its exact declared bytes. This control pass does not
qualify the latest candidate, explain the historical faults, establish
throughput or relax any learning gate. Full dependency qualification and matched
block-matmul throughput still precede the held Pong roots.
