# JAX preserves the optimizer state that native arithmetic loses

October21,2026. Follow-up to the [current-driver baseline](2026-10-21-driver615-baseline.md).
[Compact evidence](2026-10-21-optimizer-denormals.json).

The pinned upstream JAX optimizer passes **96 independent checks** over the same
16 explicit gradients and two sign-reversed updates. Its tiny second moments
remain nonzero, unlike the six failing native cases. This is a correctness
difference, not a profiler interpretation or a measured cause of poor game scores.

| First step, gradient3e-18 | Native control | Pinned JAX |
| --- | ---: | ---: |
| Stored variance |0|9.0000005e-39|
| Stored momentum |30.000008|0.09966714|
| Parameter, initially1 with test LR0.1 |−29.000004|0.90033287|

The test LR is deliberately large to expose the defect; it is not the production
learning rate. Both optimizers use epsilon1e-20 **outside** the square root.
Each reference uses its implementation's actual F32 constants: upstream computes
`1-beta` in Python before conversion, native subtracts F32 uniforms. This small
coefficient difference does not explain a zero variance or300× moment. Reference
recurrences use F64 arithmetic with F32 bias-correction denominators from the
captured JAX HLO. Original tolerances stay unchanged;18 negative-control checks
reject the native failure values. All119 source receipts pass.

## Why an FP32 preservation flag is not the repair

A properties-only query on the same RTX5080/driver615.71.09 reports
`shaderDenormPreserveFloat32=0`, while FP16 preservation is supported. The Vulkan
property determines whether the corresponding explicit preservation mode may be
used. [Vulkan specification](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFloatControlsProperties.html).
Thus latest Blade's FP16 preservation change cannot simply be extended to an
advertised FP32 mode on this device. The separate flush-to-zero support flag is
also false; that does **not** establish what the default shader arithmetic does.
The numerical probe, not that flag, establishes the lost variances.

Meganeura upstreama75c38e still contains the same variance update; its latest
fixes do not repair this case. The neighboring checkout has unrelated work and
is untouched. No Naga change, driver-bug claim, CPU optimizer fallback or epsilon
change follows.

## Scaled-variance shader passes the original GPU fixture

The isolated candidate evaluates tiny variance arithmetic at2^64 scale (gradient
at2^32), while retaining the ordinary path and the existing stored variance
format. Integer decoding/nearest-even encoding avoids arithmetic on subnormal
storage values. RMS uses the decoded, rounded variance; epsilon is unchanged.
No extra state buffer or CPU optimizer is introduced.

The actual incumbent optimizer shader and segment handling are reused, with
only the variance/RMS block replaced. Both shaders validate offline. One paired
GPU process runs the unchanged control and candidate for two sign-reversed
steps each. The control exactly reproduces all48 first-step saved values,
including its six expected failures. The candidate passes **96 F64 checks**;
all30 values in the ten unaffected first-step cases match control exactly.
For3e-18, candidate variance is8.9998842e-39, momentum0.09966780 and parameter
0.90033221. Its coefficient rounding remains native, not bit-identical JAX.

Independent CPU review verifies625 receipts and recomputes every candidate
reference. Two CPU preparation errors remain recorded: using Session memory
field names on Blade's type, and validating pre-binding shader IR before
assigning offline bindings. Corrections repeat no GPU work. GPU guard passes
with zero warnings; four standalone optimizer steps, zero learner updates/actions.
CPU review finishes07:15:35UTC, no worker remains.
Artifacts: `runs/optimizer-scaled-20261021.PYgm00jM`, including the32-line shared
helper, original/candidate shaders, probe, declaration, all raw values and review.

**Not yet qualified:** state transitions across subnormal/normal boundaries,
long sequences, wider optimizer settings, actual Session/cache integration,
complete-model behavior and performance. Ordinary finite fixtures here do not
establish behavior for arbitrary magnitudes or nonfinite inputs. No promotion.

## Accounting and next step

The JAX observation used one passing guard, zero new warnings, two standalone
optimizer steps, zero learner updates/actions and no separate NVML queries.
The service is terminal; CPU review passes without repeating GPU work.
Artifacts: `runs/optimizer-f32-20261021.YBh7h6AF`, including full values, compiled
HLO, float controls, independent `review.py`/`review.json` and retained receipts.

Across both observations: two passing GPU guards, six standalone optimizer
steps, zero learner updates/actions and zero new warnings. Next test state
boundaries/rounding and long sequences, then integrate the shared shader helper
into an isolated current-upstream Session and qualify checkpoint/cache behavior.
Keep ordinary variance storage, bias corrections, epsilon and model fixed.
Passing this original tiny fixture alone cannot qualify a full learner.

Production and Atari remain stopped. The measured performance baseline stays
48.281ms native /26.066ms inclusive JAX,1.85×; there is no new speed claim.
