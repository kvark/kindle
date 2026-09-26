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

## Next control

The launcher now runs the native-bearing learner in-process with synchronous
environments, explicit precision and both NVIDIA telemetry collectors disabled.
Require a separate bounded hardware/memory/accounting declaration before JAX/CUDA
execution. Match actual interactions, N6/full18/F32/12M/B16/T64/R256 and replay
settings. Preserve RGB64 for the historical comparison; native-resolution JEPA
is a separate intervention. Report offline pretraining experience/cost as well.

Local raw evidence: [readiness root](../../runs/upstream-control-readiness-20260926.6FFIhm/README.md),
[ALE0.9 probe](../../runs/upstream-control-readiness-20260926.6FFIhm/sticky-order-0.9.0.json),
[ALE0.12 probe](../../runs/upstream-control-readiness-20260926.6FFIhm/sticky-order-0.12.1.json),
[corrected wrapper replay](../../runs/upstream-control-readiness-20260926.6FFIhm/nonsticky/wrapper-result.json).
These paths are local, not public artifact hosting. No new control GPU job or
automatic successor is declared.
