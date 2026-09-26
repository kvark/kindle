# Upstream control: a real protocol mismatch, corrected before a new run

The retained local Dreamer control is **not a matched non-sticky baseline**.
Its pinned Atari wrapper sets the repeat-action probability after loading the
ROM. ALE has already cached the default `.25`, even though reading the setting
afterward returns the requested `0`. Kindle configures it before loading.
This invalidates that control's non-sticky comparison claim, not Kindle's game
results or the recorded control measurements themselves.

The mechanism is explicit in ALE: [ROM loading constructs the environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/ale_interface.cpp#L104),
whose [constructor caches the repeat probability](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/environment/stella_environment.cpp#L71).
The setting getter reads configuration, not the cached runtime value.

## Reproduction and correction

CPU-only direct-ALE probes compare three construction orders at the same seed
and512 forced raw actions per game: early zero, late zero, early `.25`.
Both installed ALE0.9.0 and0.12.1 produce **late-zero trajectories identical to
early-.25**, and different from early-zero, in all five games. Each comparison
includes RGB hashes, rewards and game-over flags.

| Game | First differing raw action from early zero, zero-based; both versions |
| --- | --- |
| Pong | 3 |
| Breakout | 2 |
| Boxing | 2 |
| Freeway | 4 |
| Qbert | 462 |

ALE0.12.1 also rejects the pinned wrapper's bytes seed key / NumPy scalar API
call. The [control helper](../../python/examples/run_upstream_control.py) now
requires exactly two wrapper corrections in addition to the declared config:
convert that key/value to `str`/`int`, and set stickiness before ROM loading.
It rejects uncorrected legacy wrappers and any other source change. Model and
loss equations remain pinned to DreamerV3 `e3f02248`.

With those corrections and ALE0.12.1 in an isolated import overlay, the actual
Dreamer wrapper matches Kindle's explicit RGB64 wrapper for **20,480 actions /
20,523 total records**: 4,096 actions each in Pong, Breakout, Boxing, Freeway and
Qbert. Every resized RGB hash, reward, natural boundary and executed-frame count
matches. The original API failure, seed-only mismatch, sources and captures
remain preserved; no original writer was rerun.

The tests use no actor, learner, CUDA context or NVML. This is adapter parity,
not a new trained control or a JEPA efficiency result. Forced cutoff semantics
are not covered: upstream reports `is_terminal=is_last`, so any future comparison
must separately resolve or exclude artificial cutoffs. Its driver also counts
action-free reset records; equal configured steps do not mean equal experience.

Separate [CPU fixtures](../../runs/upstream-accounting-cpu-20260926.PjQxR0/result.json)
now reproduce both gaps using the actual corrected wrapper and pinned Driver:

- With toy three-action episodes, a1,000-step request yields1,000 records but
  only750 actions at N1. At N6, the ten-record driver blocks yield1,008 records,
  756 actions and252 action-free resets. Those fractions describe the toy,
  not Atari; they expose both reset counting and vector-block overshoot.
- At an eight-frame Pong cutoff, pixels and reward still match, but upstream
  reports terminal despite `ALE.game_over()==false`; Kindle reports truncation
  without termination. Equal pixels alone do not establish equal learning targets.

These fixtures construct no agent and do not even import JAX. The control still
needs corrected or explicitly reconciled action/update accounting and cutoff
targets before claiming matched learning. Retain the existing captures unchanged.

## Next control

The launcher now runs the native-bearing learner in-process with synchronous
environments, explicit precision and both NVIDIA telemetry collectors disabled.
That disables **application telemetry**, not every internal backend use of NVML.
The installed JAX0.6.2 CUDA plugin contains the NVML initialization/fabric-query
path. Its [pinned XLA client](https://github.com/openxla/xla/blob/3d5ece64321630dade7ff733ae1353fc3c83d9cc/xla/pjrt/gpu/se_gpu_pjrt_client.cc#L1390)
calls it for compute capability≥9 while constructing devices, including a
single-device client. Reading the installed binary confirms its fabric helper
calls `InitNvml`; no CUDA client or NVML query was executed in this inspection.
Disabling Dreamer's logging therefore does **not** qualify this stock plugin.

The user's subsequent September26 direction lifts the need for an NVML-free
initialization path. Ordinary JAX/CUDA initialization is permitted for a bounded
Dreamer sanity check; application polling stays disabled. No historical evidence
established NVML as the fault's cause. Do not fork the backend or install a shim
to avoid its normal initialization. Host recovery remains unauthorized.
A sanity check can verify real GPU learning before resolving every matched-budget
comparison detail, but must disclose actual action/update accounting and cannot
establish a JEPA architectural gain.

Require a separate bounded hardware/memory/accounting declaration. Match actual
interactions, N6/full18/F32/12M/B16/T64/R256 and replay
settings. Preserve RGB64 for the historical comparison; native-resolution JEPA
is a separate intervention. Report offline pretraining experience/cost as well.

Local raw evidence: [readiness root](../../runs/upstream-control-readiness-20260926.6FFIhm/README.md),
[ALE0.9 probe](../../runs/upstream-control-readiness-20260926.6FFIhm/sticky-order-0.9.0.json),
[ALE0.12 probe](../../runs/upstream-control-readiness-20260926.6FFIhm/sticky-order-0.12.1.json),
[corrected wrapper replay](../../runs/upstream-control-readiness-20260926.6FFIhm/nonsticky/wrapper-result.json).
These paths are local, not public artifact hosting. Those CPU readiness captures
did not declare a control GPU job or automatic successor; the later sanity
check below has its own declaration.

## Completed GPU sanity check

Following the user's revised NVML direction, the separately declared
[stock JAX/CUDA run](../../runs/upstream-sanity-20260926.8rbt0x/README.md)
completes on the RTX5080/580.178.04. DreamerV3e3f02248, N6/F32/12M/B16/T64/R256,
Pong RGB64/full18/nonsticky/no-noops, seed0; original model/loss/optimizer and
default5M replay capacity. Application NVIDIA collectors and the optional
JAX profiler are disabled; normal backend initialization is permitted.

The [independent raw audit](../../runs/upstream-sanity-20260926.8rbt0x/result.json)
verifies6,000 driver records = **5,990 actual actions +10 reset observations**,
**1,149 updates**,1,148 finite metric rows, all288 saved parameter/state entries
finite and changed, including dynamics weights. The native-bearing worker takes
290.04 seconds including initialization/compilation. All four completed episodes
are poor early policies (−21,−21,−21,−20), not learned Pong competence.

Eleven memory checkpoints pass: minimum Vulkan estimated budget headroom
6,703,939,584 bytes and minimum CUDA allocator limit-minus-use8,837,849,183 bytes.
These are distinct estimates, not physical free/peak VRAM. Guard exit is zero,
the child is reaped and no kernel fault is recorded. All106 declared inputs and
the final saved payload independently reverify. No host recovery or automatic
successor occurred. The first CPU test invocation lacked pytest in the preserved
reference environment; four standard-library unittest checks pass without
installing anything into it.

This establishes a working upstream GPU baseline path, **not** matched learning,
an isolated throughput comparison, NVML causality/safety or JEPA benefit. Resolve
actual-action/update scheduling, replay capacity and cutoff targets before a
long equal-budget comparison. The corrected environment protocol and normal
backend are ready; an NVML-free fork is unnecessary.
