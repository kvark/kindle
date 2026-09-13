# Isolated latest-source block-matmul candidate

Use /x/Code/kindle/AGENTS.md, its authoritative
docs/kindle_single_life_dreamer_plan.md and /mnt/data/GUIDELINES.md for current
direction. This worktree is not an adopted runtime or a learning declaration.

Carry only networks.rs from the preserved f2e20af / 20b9b8a block candidate onto
7728d8d. Its serial batch-one path, large imagination batches, original GRU gates,
parameters and all tests remain unchanged; only small F32 block products select
the existing operator. This is source preparation, not a compiled package,
GPU declaration, parity result or speedup.

Keep Meganeura 0a98775 and shared git Blade f6f2729e identical to 7728d8d.
The exact upstream 428fc2d policy patch is included, but NativeF32 is not selected:
learner Auto and LeVJEPA Disabled remain. Flushed initialization traces, checked
waits, alias-order allocation and deferred host zeroing are unchanged.

The separately guarded production diagnostic in
/x/Code/kindle/runs/native-f32-alias-initialization-20260913.PDrNaR passes on
this backend with every native loss/gradient assertion and exact control
allocation plans. Its separate full hardware stage is in preflight. Those
dependency-only results do not qualify this block candidate. Finish all
dependency hardware/state/N6 pixel/restore/combined-memory gates first, then
separately qualify this exact block patch on the same backend with full
gradients, update-1/eight-update state and all optimizer moments, pixel traces
and matched AB/BA timing. Regression acceptance is not a speedup.

Preserve completed/failed attempts, original packages and the root-2017 hold.
No retries, run-all or automatic followers. Do not reset/reload/change drivers/
reboot without user approval. Guard one direct native GPU window at a time,
with actual device checks and at least 2048 MiB directly free. Inspect each
result before follow-up. Never run heavy CPU builds alongside GPU work;
use private caches and one core / 2 GiB / zero swap.

Do not enable tuning, skip initialization, change precision, batch/BPTT,
learning settings or profiling options alongside this block-only change.
Removing temporary initialization diagnostics also requires a separate
comparison; do not silently reuse this source identity.
