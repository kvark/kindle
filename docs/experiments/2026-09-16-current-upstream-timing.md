# Current-upstream GPU qualification without NVML

The latest Meganeura candidate passes **all 23 native GPU tests**: four upstream
profiling tests and all nineteen original hardware tests, on the RTX 5080 /
driver 580.178.04, without NVML. It now also passes all three exact same-driver
full-state pairs and completes a seven-session native kernel profile without
changing the checkpoint. The older cross-driver anchor fails and is preserved
separately below. No throughput improvement,
driver/NVML causal fix or Atari win is claimed. The subsequent
[same-driver N6 runtime comparison](2026-09-16-native-memory-pixels.md) now also
passes exact repeated state, restore/override, native budget and non-regression
gates. It does not repair historical cross-driver/NVML gates.
Main remains ce80; throughput qualification still precedes the held Pong roots.

## Use upstream, remove the temporary adapter

The fresh source check found Meganeura **986f49a77776aafba5df2598a9e944d6b658122d**
and Blade **92553493a832d505e3d22b7ce3f7b0ac4fc9461d**. Meganeura now includes
the new completed-submission timing API and profiling windows for graphs larger
than Blade's timestamp capacity, alongside quantized-kernel updates. Blade
includes the signed calibration-delta fix. These are used directly.

The isolated source-matched candidate is:

| Component | Revision | Local additions |
| --- | --- | --- |
| Kindle | `9ec96cf` | Explicit session timing options; remove obsolete timestamp-capacity fallback |
| Meganeura | `589d73ab` | Retained allocation order/observations; fail-fast initialization/session/transfer/checkpoint/tuning waits |
| Blade | `2accfeee` | Retained allocation observations on upstream 92553493 |

The earlier local timing adapter, owned-duration implementation and Blade
timing-enabled accessor are not carried into this candidate. Learning math,
precision and cooperative-policy choices remain fixed. Timing-disabled contexts
are valid; an incomplete fence or device error must stop execution, not become
an empty timing result. GPU execution has no CPU fallback and makes no NVML
monitoring calls.

Sources are pushed on `exp/current-timing-20260916` (Kindle) and
`exp/kindle-current-timing-20260916` (Meganeura). They are not merged/adopted.
The shared Blade commit is retained on `exp/kindle-timing-20260916`.
This declaration launches native test binaries only. The isolated source tree's
historical Python profiler still contains NVML polling and must not be launched;
main's corrected profiler is separate. A future Python package needs matching
runners and the non-NVML monitoring policy, not an old scheduler copy.
The subsequent [native-memory pixel packages](2026-09-16-native-memory-pixels.md)
carry that corrected profiler and the same opt-in native budget getter in both
arms. Their CPU checks and all ten separately guarded N6 GPU phases now pass.

## Bounded native evidence

[VPMDem](../../runs/current-timing-runtime-20260916.VPMDem/declaration.md)
binds **868 inputs** and declares each of four timing tests and the original
nineteen hardware tests individually. There is no run-all mode or automatic
follower. Each result is reviewed before the next invocation; failures stop the
group. The existing direct-child host guard uses kernel logs and host identity,
not NVML. Timing tests have 120 seconds; hardware tests retain 1800 seconds.

| Completed test | PID | Result |
| --- | --- | --- |
| Windowed execution without timing | 56773 | Exact varied-input output parity; 3 complete sessions |
| Expected missing-timing error | 56931 | Subsequent execution stays unprofiled and matches reference; 2 sessions |
| One-window capture | 57106 | Complete sample and label attribution; 1 session |
| Multi-window capture | 57303 | Every dispatch sampled twice; exact output parity; 2 sessions |
| Production B16/T64/full-BPTT/F32 gradients | 57521 | 9 losses, 62 parameters, 51 nonzero gradients; 2 sessions |

All five children exit zero and are reaped, with passing host guards and no
recorded kernel fault. The gradient test's worst relative L2 error is
**0.0007456096368231737**, below the unchanged .003 outer gate. Its complete
292,932-record trace matches both retained ce80 allocation plans: 11,319/9,439
physical slots and 15,296/14,306 constant uploads. Native learning tolerances
are unchanged; this is not a gameplay benchmark.

The other eighteen hardware tests also pass: cached attention/reset behavior,
both numerical-tail regressions, matmul epilogues, device copies, checkpoint
round-trips, act/learn/restore, vector belief/policy/learning and override
causality, large loss reductions, LeVJEPA reference/batch parity and F16 clamp
bounds. Every native assertion retains its original inputs and tolerance.

The complete group has **237 session initializations, 666,377 trace records and
1,585 host checks**, maximum host-check gap 0.289187 seconds. These are host
checks, not GPU-health/utilization samples. All 23 children exit zero and are
reaped. The separate [complete read-only audit](../../runs/current-timing-completion-20260916.4I6PMB/capture/result.json)
reverifies all 868 inputs and every raw result, with a clean fresh post-audit
kernel check. It starts no GPU job. No recorded fault or unfinished child remains.

LeVJEPA matches all 37 causal reference frames with maximum relative L2
**8.47e-6** and maximum absolute error **0.00026131**. N2/N4/N6/N8 batched dense
features match serial execution exactly across reordered arrivals, observation
gaps and asymmetric resets; pooled features pass their original tolerances.
This is perception correctness, not N8 combined-learner memory qualification.

CPU preparation uses fresh copies of the earlier private targets, one CPU,
2 GiB and zero swap. All commands complete: **86 Kindle CPU tests** with 23 GPU
tests ignored, seven upstream profiler tests, nine initialization tests,
formatting and both release Clippy checks. Four reader tests reject missing or
skipped native execution, wrong devices/PIDs, incomplete initialization and
failed waits. Their separate retained check also passes. No original CPU writer
or completed GPU invocation is rerun.

For completed tests, the retained-file reader is:

```bash
python3 -B runs/current-timing-runtime-20260916.VPMDem/run.py audit timing-windows
python3 -B runs/current-timing-runtime-20260916.VPMDem/run.py audit hardware-14
```

`verify` rechecks input pins only. The complete reader is
`runs/current-timing-completion-20260916.4I6PMB/audit.py audit`; its `capture`
writer is completed and must not be rerun. Preserve the completed builds, seal
and all 23 invocations. **No run mode remains reusable in this group.**

## Preserved superseded timing work

[NxDACz](../../runs/timing-runtime-20260916.NxDACz/declaration.md) completes two
small GPU checks for the earlier 5f7e7cb/aeb554e6 implementation: timing disabled
(PID 53828) and enabled (PID 54060). Both verify two fills and two inference
steps; enabled timings identify the current first/second submission, at
2,592/2,144 ns. Its 845-input closure was sealed **after** the timing-off test,
not before it; both individual native commands/hashes were declared beforehand.
Use `audit.py off`, `on` or `verify` only. These passes do not qualify the later
upstream implementation.

Preserve the initial NxDACz build's foreign-package refusal: Cargo cannot test
a dependency's dev-dependencies through Kindle's workspace. Its Kindle build
passed before that refusal; a separate backend workspace build completed the
remaining checks. The old writer's `complete: false` remains unchanged.

## Next decisions

The N6 pixel/restore/override and matched non-regression comparison now passes
under its separately declared native budget gate. Its AB/BA ratios are
0.995114/0.994309, not a speedup. Proceed to the same-backend block-matmul
candidate, retaining completed inputs and comparisons. The profile identifies a
large split/concat dispatch burden, supporting the already staged block-matmul
comparison on this same backend. Retain ordinary untraced end-to-end timing as
the speed benchmark. A windowed replay profile is not GPU utilization or a
whole-Atari idle-gap measurement. Do not introduce a CPU execution fallback,
actor/learner separation or new monitoring infrastructure for this work.

The unchanged canary is freshly built in
`runs/current-timing-canary-cpu-20260916.CICPt6`, native SHA256
`497fd1475a7e532bce4f40e61edd2bdd4ac992af003115f75a0f9f56cd3da0c5`.
Its three-command CPU build completes in a copied private target; preserve the
writer/cache. Its GPU state comparisons now complete as described below.

GPU recovery action, utilization and directly free/reserved VRAM remain
unmeasured. Existing native Vulkan allocation-budget checks remain enabled;
they are not the historical NVML direct-memory gate. Declare a non-NVML memory
measurement explicitly for new comparisons rather than calling old readings
current; the completed N6 group does this with post-stage budget headroom.
Keep the same-backend block-matmul comparison ahead of
Pong, with unchanged game budgets and competence gates. Stop on a new fault;
no reset, reload, reboot or driver change is authorized.

## Same-driver complete state; historical anchor fails

The first [historical-anchor declaration](../../runs/current-timing-state-20260916.T8qQhs/declaration.md)
stops after its sole ce80 control, PID **64349**. The unchanged native executable
exits zero, its host guard passes, reports/checkpoint are complete and finite,
and no kernel fault is recorded. The exact comparison against its September 11
checkpoint fails: **90 optimizer-moment tensors differ**. All 95 non-moment
tensors, shapes, metadata and non-timing reports match. No candidate or later
window runs in T8qQhs; preserve its absent top-level success result and unused
declarations. Never rerun it.

This compares the same executable across driver **595.91.07 → 580.178.04**,
not two backends on the same driver. That distinction does not make the old
gate pass or prove the driver caused every difference. The separate
[terminal interpretation](../../runs/current-driver-state-20260916.QJKcT5/historical-result.json)
retains the full exact comparison and per-tensor errors. Most differences are
small, but do not describe all of them as insignificant rounding: the actor's
first-layer weight momentum at flat index 463568 is **19.5791378 versus
20.0183926**, absolute difference **0.4392548**. Both corresponding second-moment
entries are zero. This outlier already exists in the historical control;
its cause and effect on learning are not established. Keep it for a focused
optimizer numerical/stability investigation, not an unannounced epsilon or
learning-rule change during throughput qualification.

The source uses **LaProp**, despite retaining the checkpoint names `adam_m`
and `adam_v`: it normalizes each clipped gradient by its RMS estimate before
accumulating momentum. Consequently these are not raw-gradient Adam first
moments, and a raw-Adam moment/variance bound would be the wrong diagnostic.
The existing upstream LaProp test checks ordinary-scale gradients with epsilon
1e-8, not the retained Dreamer epsilon 1e-20 near variance underflow. A bounded
native test of that regime remains appropriate; no optimizer parameter is
changed in the pixel or throughput comparisons.

The new [same-driver diagnostic](../../runs/current-driver-state-20260916.QJKcT5/declaration.md)
pins **2,278 inputs**, reuses the completed 580 control after a fresh raw audit,
and performs five separately reviewed native invocations:

| Window | ce80 PID | Current candidate PID | Result |
| --- | --- | --- | --- |
| Update 1, control first | 64349, reused | 65643 | Exact complete state/reports |
| Update 8, control first | 66348 | 66880 | Exact complete state/reports |
| Update 8, candidate first | 67866 | 67351 | Exact pair and both same-backend repeats |

All comparisons retain **241 tensor entries and all 146 optimizer moments**,
complete checkpoint metadata except the declared backend identity, all
non-timing reports and the original 12M/B16/T64/full-BPTT64/R256 settings.
No numeric tolerance or tensor exclusion is introduced. The three candidate
runs have **33 complete initialization sessions / 439,227 trace records**.
Every native child exits zero and is reaped. Across the six windows, **1,472
host checks** pass, maximum gap **0.288709 s**, with no recorded kernel fault
or NVML query.

Both unchanged canaries expose Vulkan memory-budget snapshots. The new
declarations explicitly require at least 2 GiB of budget-minus-usage headroom
at four ordered snapshots. The minimum observed headroom is **10,021,634,048
bytes (9.33 GiB)**. This is not NVML directly free memory, a peak measurement
or combined N6 perception/learner qualification.

The short untraced update-3–8 medians are **360.402 / 361.393 ms** in the first
control/candidate pair and **362.438 / 360.104 ms** in candidate/control order.
These tiny synthetic windows do not establish a speedup or Atari throughput.
They motivate kernel profiling rather than longer learning runs on an assumed
improvement. Same-driver parity is established; the failed historical-anchor
gate remains failed, and full runtime/adoption and Pong gates remain open.

All QJKcT5 invocations are terminal. Reuse only `run.py audit NAME` or `verify`.
The separate profile preparation rechecks every raw state result before any
profiling execution; it does not rerun these canaries.

## Native kernel profile: reduce dispatch fragmentation

The separately declared [814vSr profile](../../runs/current-learner-profile-20260916.814vSr/declaration.md)
pins **2,430 inputs** and completes its sole native invocation, PID **68485**,
using the unchanged canary with GPU timestamps enabled. Five reader fixtures
pass; the independent retained-file audit reverifies all inputs, the raw guard,
all seven profiles, 11 complete initialization sessions and the final checkpoint.
Every one of **56,096 dispatches** has three attributed samples. All **241
tensors / 146 optimizer moments**, metadata and non-timing learner reports
remain exactly equal to the unprofiled candidate at update 1.

The helper retains eleven ordinary executions after two warmups per session.
It disables optimizer, gradient accumulation and clipping for these fixed-input
profiles; they are **not whole learner updates**. Ordinary GPU-pass and wall
medians are:

| Session | Dispatches | Profile windows/sample | Ordinary wall ms | Ordinary GPU-pass ms |
| --- | ---: | ---: | ---: | ---: |
| [Posterior](../../runs/current-learner-profile-20260916.814vSr/profiles/posterior.json) | 302 | 1 | 0.839 | 0.644 |
| [Actor/value](../../runs/current-learner-profile-20260916.814vSr/profiles/actor_value.json) | 27 | 1 | 0.322 | 0.259 |
| [Slow value](../../runs/current-learner-profile-20260916.814vSr/profiles/slow_value.json) | 14 | 1 | 0.274 | 0.232 |
| [World heads](../../runs/current-learner-profile-20260916.814vSr/profiles/world_heads.json) | 13 | 1 | 0.242 | 0.203 |
| [Transition](../../runs/current-learner-profile-20260916.814vSr/profiles/transition.json) | 284 | 1 | 2.439 | 2.083 |
| [World gradient](../../runs/current-learner-profile-20260916.814vSr/profiles/world_gradient.json) | 55,245 | 45 | 171.012 | 121.903 |
| [Behavior gradient](../../runs/current-learner-profile-20260916.814vSr/profiles/behavior_gradient.json) | 211 | 1 | 18.597 | 17.776 |

The world plan has **8,618 barrier groups** and **29,020 SplitA/SplitB/Concat
dispatches**. The posterior has 158 splits plus 39 concatenations out of 302
dispatches. This is a concrete reason to prioritize the staged small-batch
block-matmul graph: remove repeated slicing/concatenation without changing
recurrence, data, replay ratio or precision. Actual speed and complete-state
parity must still be measured; dispatch counts are not a speedup.

The world pass's ordinary wall/GPU median gap is approximately **49.1 ms**.
It includes work outside the timed GPU pass, such as command preparation,
submission and synchronization; it is not a calibrated 49.1 ms GPU-idle
measurement. The seven sessions also run at different frequencies in a learner
update, so summing their per-call medians is not total learner cost.

Instrumentation is substantial. The world capture uses **45 full replays per
sample**, giving **7,929.526 ms** of summed replay wall time versus 171.012 ms
for one ordinary execution. The stitched timestamp total is **336.282 ms**,
not the ordinary 121.903 ms GPU pass. Posterior instrumentation increases wall
time by **2.61×**. Per-dispatch family shares are diagnostic, not an unperturbed
cost breakdown or GPU utilization. No pipeline register/spill statistics are
returned; do not infer them from empty arrays.

The native child exits zero and is reaped. All **341 host checks** pass,
maximum gap **0.289187 s**, with no recorded kernel fault or NVML query.
Vulkan snapshot headroom remains at least **10,076,160,000 bytes**; snapshots
do not measure profile peak allocation or directly free memory. The complete
[result](../../runs/current-learner-profile-20260916.814vSr/result.json) retains
profile/checkpoint hashes. `run.py audit` and `verify` are read-only; the
preparation and sole GPU invocation are terminal and must never be rerun.
