# Current-upstream GPU qualification without NVML

The latest Meganeura candidate passes **all 23 native GPU tests**: four upstream
profiling tests and all nineteen original hardware tests, on the RTX 5080 /
driver 580.178.04, without NVML. No throughput improvement,
complete dependency qualification, driver/NVML causal fix or Atari win is claimed.
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

Complete full-state/moment, N6 pixel/restore/override and matched throughput
qualification. Use the new
profiler to identify expensive kernels, retaining ordinary untraced timing as
the speed benchmark. A windowed replay profile is not itself GPU utilization or
a whole-Atari idle-gap measurement.

The unchanged canary is freshly built in
`runs/current-timing-canary-cpu-20260916.CICPt6`, native SHA256
`497fd1475a7e532bce4f40e61edd2bdd4ac992af003115f75a0f9f56cd3da0c5`.
Its three-command CPU build completes in a copied private target; preserve the
writer/cache. It has not run on GPU. Use a new bounded declaration for state
and profiling work under the resumed GPU direction, not another approval wait.

GPU recovery action, utilization and directly free/reserved VRAM remain
unmeasured. Existing native Vulkan allocation-budget checks remain enabled;
they are not the historical NVML direct-memory gate. Declare a non-NVML memory
measurement explicitly for the new full-runtime comparison rather than calling
old readings current. Keep the same-backend block-matmul comparison ahead of
Pong, with unchanged game budgets and competence gates. Stop on a new fault;
no reset, reload, reboot or driver change is authorized.
