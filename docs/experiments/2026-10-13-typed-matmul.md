# Shared typed cooperative matmul

Declared October 13, 2026 before GPU execution. The user approved replacing
the dedicated BF16 matmul family with type-parameterized implementation and
shared scheduling. No Atari learning, precision-policy change or tuning sweep.

Control: qualified Meganeura `9e05731`, Blade `663afd2`, Naga `c9fd6a85a`.
Upstream rechecked: Meganeura main `20af998`, Blade main `7909fc1`, unchanged.
Use the isolated `refactor/typed-cooperative-matmul` worktree; preserve controls
and neighboring user work. Only the user merges.

## Scope

Make operand storage, multiplication precision and output storage explicit;
retain F32 accumulation. Share the checked cooperative loop, indexing,
transposition, partitioning and output handling across F32/F16/BF16. Keep
genuinely different algorithms and capability restrictions distinct. Precision
must be part of scheduling, pipeline and tuning identities. Do not let timing
silently change it, reinterpret packed buffers or round split partials early.
Retain specialized fused and hardware-specific paths where this checked loop
does not implement their contract. Remove the dedicated BF16 dense shader.

## Qualification

- One-CPU/2-GiB/no-swap preparation: format, CPU shader/type/selection/cache
  checks and Clippy; build immutable binaries before GPU work.
- Serialized GPU checks with the existing host guard, at most 180 seconds per
  process: typed matrix/partition oracles, transposes, ragged edges, addends,
  poisoned reuse, F32 master/reduced gradients, connected optimizer updates
  and unchanged convolution checks. Exercise supported F16/BF16 on the 5080;
  validate F32 shader generation/capability rejection if native F32 is absent.
  Use independent references and existing numerical gates, no relaxation.
- One ordinary matched pair each: RGB12M/B8/BF16 and CDP12M/B32/mixed,
  16 warmup plus 256 measured synthetic updates, at most 600 seconds/process.
  Same initial tensors, fixture RNG and all learning settings. Require all
  reports/targets/checkpoints finite and disclose any trajectory differences.
  This is a maintainability refactor: require no greater than 2% whole-update
  regression in either recipe; a speedup is not required. No timing repeats.
- Publish and pin only after qualification; update PR31 at the boundary.

Use persistent bounded systemd services, Restart=no, KillMode=control-group,
expected RTX 5080/driver580.178.04 and at least 2 GiB sampled Vulkan headroom.
Record allocation warnings; review failures before any follow-up. No NVML
polling, driver recovery, automatic training restart or concurrent GPU jobs.

Artifacts: `runs/typed-matmul-20261013.9OmuXOPr`.

## Qualification review

The first GPU check passes its initial F16 matrix comparison, then stops in
the CPU option validator: the test requested one timing pair, below the
tuner's existing minimum of four. Correct only that test setting; retain the
failed guard and immutable binary. Runtime and learner binaries are unchanged.
Run the same checks in a new bounded queue with the corrected test executable;
all ordinary timing pairs remain unstarted. No numerical gate is changed.

The corrected test then exposes a tuner integration error: an odd F16 operand
has a valid two-byte-aligned logical extent, but the new class builder rounded
that extent to four bytes and rejected the actual binding as too small. Keep
logical sizes exact and pad only private upload/readback allocations. A CPU
regression now collects the real compiled odd-sized F16 graph successfully.
Preserve this second failed guard. Rebuild the corrected backend and learner;
redeclare the same uncompleted checks and still-unstarted ordinary pairs with
new immutable executables. No precision, numerical tolerance or timing change.
