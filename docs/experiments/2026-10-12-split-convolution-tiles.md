# Split-convolution tile fix

Declared October 12, 2026 before GPU execution. Follow the
[Nsight diagnosis](../results/2026-10-12-nsight-root-cause.md); no game training,
new profiler trials, tile sweep or adoption of rejected cooperative PR248.

Latest published Meganeura main is `20af998`: it adds GGUF mapping after the
reviewed layout/search changes. Blade main remains `7909fc1`. Neighboring
experimental branches and user edits stay untouched. Use isolated worktrees;
only the user merges.

Promote only the demonstrated 16-to-32 split weight-gradient change. Reuse the
existing convolution occupancy rule with the partition count: widen a 16-wide
producer only when 32-wide tiles can fill the existing 64-workgroup target.
Keep low-occupancy, already-32/64-wide, unsplit and budget-rejected cases intact.
Split counts, K stage, precision, partial storage and final reduction do not
change. Cover selection boundaries, ragged shapes, every partial derivative
against the existing F64 oracle, and optimizer/alias behavior. Do not relax
numerical gates. Existing serialized plans remain safe, though an old cached
plan retains its old selection; qualification builds fresh sessions.

Refresh the combined BF16/row-addressing branch onto published main first,
preserving current pins and historical evidence. Resolve source conflicts by
retaining both current upstream behavior and the qualified BF16 additions.
Keep a refreshed no-tile control separate from the candidate. No merge commits.

## Bounded checks

1. One-CPU, 2 GiB, zero-swap CPU preparation: format, scoped non-GPU selection /
   cache / precision tests, Clippy and builds. Do not run unfiltered backend tests
   as a CPU check; many construct a GPU context.
2. Serialized backend GPU tests, 180 seconds per process: independent F64 split
   derivatives including 16-wide input choices, and existing optimizer/alias
   checks. Refresh checks also retain the qualified BF16, convolution and
   row-addressing coverage. Require the expected RTX 5080 and at least 2 GiB
   sampled Vulkan budget headroom; retain warnings and stop on other failures.
3. One ordinary control/candidate pair each for RGB12M/B8/BF16 and
   CDP12M/B32/mixed: 16 warmup plus 256 measured updates, 600 seconds per process.
   Same source checkpoints, fixture RNG, configuration and no profiling layers.
   Compare all 272 numerical reports, first targets and all 292/346 initial,
   first and final tensors. Separate refresh effects from the tile effect;
   the tile pair must remain exact. Check exported plans against the declared
   shader/grid-only changes using bounded-memory analysis.
4. Only after qualification, publish the minimal upstream PR and pin the useful
   refreshed integration in both Kindle manifests/locks. Run relevant binding,
   revision and CI checks. No historical repin or claim of learning quality,
   ordinary SM utilization, complete BF16 backward, or closing the JAX gap.

Use the existing host guard and serial persistent systemd services with
`Restart=no`, `KillMode=control-group` and bounded deadlines. No CPU quota on
GPU workers; no compilation during matched timings. No NVML polling or driver
recovery. Artifacts: `runs/split-conv-tiles-20261012.exkYhrAF`.

## Preparation review

The refresh retains upstream's unified RoPE/attention parameters and measured-
search changes alongside BF16. Combined cache28 rejects the earlier BF16
cache27 encoding. The already-upstream hardware-row patch is dropped, including
its duplicate test discovered during source review, before compilation.

A reused CPU packaging script attempts to copy the unchanged backend binary
while its qualification process is running. The OS refuses with `Text file
busy`; the existing executable and successful GPU checks are unchanged. The
failed service/log remain. A dedicated builder with a new, write-once output
rebuilds only Kindle's accurate control revision label. Both final timing
binaries report their actual backend commits, and no build runs during timing.

The first CPU audit stops on a missing full-plan export: those files are tied
to the existing test harness's NVTX annotation option, which was disabled for
ordinary timing. Preserve the failed audit; no timing or numerical failure.
Declare four separate short plan-export runs on the same immutable binaries,
using existing annotation/short mode (16+8 synthetic updates each, 180 seconds
per process), with no injected profiler or GPU timestamps. Check their 24-report
prefixes and initial/first tensors against the ordinary runs, plus exact short
final tensors between control and candidate. Audit all ten exported plans.
Retain the extra96 synthetic updates; do not replace or repeat ordinary timings.

Upstream CI974 subsequently finds one stale shader-selection assertion in
`vision::conv2d_grad_weight_selected_splits` on both Metal and lavapipe: the
large split case now correctly selects32, while the low-occupancy case retains16.
Retain both failed logs. Correct those exact expectations, with no numerical
tolerance or runtime change. Build the oracle test and run that one check under
the same device/headroom guard, bounded to180 seconds; let upstream CI cover
the full portable suites. This test-only follow-up does not require repinning
the qualified production runtime or repeating complete-update timings.
