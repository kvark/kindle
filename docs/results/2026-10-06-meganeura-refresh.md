# Meganeura October6 refresh qualifies for the current CDP path

Adopt Meganeura main`b684ffd9`; Blade remains`e349cddf`. The new main adds
shader-side read bounds, reduction/primitive gradient corrections and explicit
training-safe composite recognition over the previous qualified`592a2f5a`.
The user's separate Meganeura working branch is untouched; Kindle pins the
upstream commit rather than using a local path override.

The changes pass the exercised independent numerical and integration checks.
They are **not evidence that a backend bug caused the failed Pong seeds**,
nor new learning results. Completed500k models retain nativef4b6a5c7 and their
old backend identity; frozen probes will disclose the new runtime separately.

## Checks

- 114 CPU Rust tests pass (63 guarded GPU/fixture tests remain ignored by the
 ordinary CPU suite);1,161 Python tests at build. Workspace/Python formatting
 and strict Clippy pass. After the Pong-probe extension, all1,163 Python tests
 pass, including37 targeted probe tests.
- 18 guarded GPU processes and the independent CPU audit pass, with no new
 allocation/validation warnings, NVML polling or host recovery.
- Independent F64 CDP cosine values/1,024 derivatives and action-effects bonus
 values/invariance pass; repeated-state learning, replay/imagination alignment
 and exact zero-coefficient equivalence also pass.
- Four fixed synthetic updates per arm pass1,300 CDP and1,524 RGB upstream
 comparisons, including raw gradients, optimizer moments and EMA. Native
 pre-step weights align the gradient reference; this is not bitwise stochastic
 learning-trajectory parity. No tolerance is relaxed.
- CDP/RGB replay/re-encoding/restore and causal Tiny streaming fixtures pass.
- Each of CDP+action-effects, RGB and frozen-Tiny production paths completes
 1,024 actions/195 finite updates with zero debt, followed by1,024 frozen
 actions/zero updates. Exact saved tensors:346/292/241. All sampled Vulkan
 estimated-headroom checks pass; this is not physical free/peak VRAM.
- 3,072 smoke training actions plus3,072 frozen actions are excluded diagnostic
 compute. No unchanged learning matrix resumes.

Native:`683ccd223a49b5b246ae04c315ff2f7e452fc267668177e3aca82155fb7dcdfa`.
Only the dependency pin, both locks and reported backend revision change in
the production learner. Pong diagnostic tooling is a separate Python-only
extension; no native workaround or new production kernel is introduced.

CPU preparation uses one CPU,2GiB and zero swap. Native qualification is serial,
host-guarded, bounded and completed before frozen Pong diagnostics begin.
The guard's successful exits do not establish general hardware safety or
explain the historical driver incidents.

[Compact audit](2026-10-06-meganeura-refresh.json).
Artifacts:`runs/meganeura-refresh-20261006.7Edzu5C4`.
