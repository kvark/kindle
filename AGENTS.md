# Isolated same-backend block-matmul candidate

Use /x/Code/kindle/AGENTS.md, its authoritative
docs/kindle_single_life_dreamer_plan.md and /mnt/data/GUIDELINES.md for current
direction. This worktree is not an adopted runtime or a learning declaration.

Carry only networks.rs from exp/block-matmul-conv-20260913 at 20b9b8a onto
the unchanged d62d356 allocation-order dependency candidate. Its parameters,
serial batch-one path, large imagination batches and GRU gates stay unchanged;
only small F32 block products use the existing upstream operator. This is a
source-only carry, not a compiled/tested package, GPU declaration or speedup.

Keep Meganeura 1c314b14 and git Blade f6f2729e exactly shared with d62d356.
Upstream still reports 75dfe901 / 68a23e49; all Blade native inputs outside the
unused renderer match f6f2729e. The original dependency candidate, its package,
hardware and complete-state declaration remain untouched.

The guarded production diagnostic in
/x/Code/kindle/runs/gpu-alias-order-runtime-20260913.3GAsGg passes with exact
control allocation plans and all original losses/gradients; its full hardware
group now passes all 19 checks. This does not prove root cause or qualify the
full runtime. Finish the dependency's complete-state and N6 pixel/restore/memory
gates, then separately declare the block candidate's original gradient, full-
state and N6 AB/BA gates on that same backend. Keep the >=2 GiB direct-free
margin and exact values/moments/traces. Regression acceptance is not a speedup.

Preserve completed/failed attempts, original packages and the root-2017 hold.
No retries or automatic followers. Do not reset/reload/change drivers/reboot
without user approval. Native tests remain serialized and guarded, with actual
device checks and at least 2048 MiB directly free. Never run heavy CPU builds
alongside GPU work; use private caches and one core / 2 GiB / zero swap.

Do not enable tuning, skip initialization, change precision, batch/BPTT,
learning settings or profiling options alongside this block-only change.
The temporary initialization diagnostics remain part of its actual source;
removing them requires explicit comparison, not silent identity reuse.
