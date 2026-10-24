# Explicit full-subgroup support is ready for upstream review

October21,2026. [Evidence](2026-10-21-full-subgroups-upstream.json).
[BladePR417](https://github.com/kvark/blade/pull/417), commit0af76fd on current
maineae0c6f, contains only the optional subgroup contract, tests and default
call-site migrations. It is unmerged; Kindle pins and production are unchanged.

`ComputePipelineDesc::full_subgroup_size` requests full compute subgroups of a
chosen width. `Capabilities::full_subgroups` advertises support and checked
geometry limits. Vulkan queries/enables the required features and attaches the
required-size chain/full-subgroup stage flag only for explicit requests.
Unsupported requests fail; Metal/GLES advertise no support. Existing callers
passNone and preserve defaults. No local-ID mapping assumption, Naga change,
convolution kernel or profiling code is imported. The contract follows the
[Vulkan requirements](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateInfo.html).

Validation:

- 22 CPU library tests and strict Vulkan library/tests Clippy pass; one existing
  ignored GPU unit test stays ignored. Formatting and the new offline WGSL test pass.
- Three RTX5080 tests pass: explicit full32 checks every one of128 invocations,
  subgroup width and active-lane count; default dispatch and manual barriers
  preserve their original results. Four dispatches, zero learner updates/actions.
- GLES passes the repository's normal CI lint policy with exactly the same
  warning as pristine upstream. Strict GLES Clippy remains failed on the existing
  `arc_with_non_send_sync` at unchanged`gles/egl.rs:415`; no suppression or
  unrelated fix. Metal/Web/Windows runtime behavior is not locally qualified.

Preserve the first GPU fixture failure: reserved WGSL identifier`active` caused
parsing to fail after initialization but before pipeline creation or dispatch.
Original source/binary/guard remain. The corrected`active_count` fixture gains
an offline parse/validation test before its separately declared execution.
Four guards total: one failed, three passed; zero allocation warnings/recovery/
NVML guard queries. GPU ends14:40:16UTC; final180-receipt audit14:43:07.
No local worker remains. Cross-platform[CI1192](https://github.com/kvark/blade/actions/runs/39619026064)
is still running at publication; MSRV/Format/Android/Web have passed, not the
entire workflow. Only the user merges.

Follow-up on October21: all12 jobs now report completed/success, including Linux,
Windows and macOS GPU jobs. The response is archived in
`runs/convolution-packed-port-20261021.DouXgWRZ/blade-ci.json`. No merge or pin.

The preceding[complete-update speedup](2026-10-21-convolution-reuse-cost.md)
remains39.740ms RGB/39.398ms CDP,1.525× inclusive JAX. Those measurements belong
to isolatedc8aeba28, not this new upstream dependency combination or Blade alone.

Next: audit/port the Meganeura prerequisites before the fixed reuse selector.
Current maina75c38e lacks the isolated`Conv2dGradientTyped` family and full-size
pipeline/cache wiring. A forward-only port would omit part of the measured
forward/dX/dW improvement. Keep the full family in scope, preserve compensation,
rounding and fallbacks, and do not import all experimental backend extensions.
No new geometry sweep, automatic pin/promotion or Atari restart. Five-game CDP
supremacy and the remaining JAX gap stay open.

Artifacts:`runs/full-subgroups-upstream-20261021.zgoBU9Zk`. Neighboring dirty
Blade/Meganeura checkouts were not modified; preparation uses separate worktrees
and a non-shared build cache. Approximately4.7GiB disk remains.
