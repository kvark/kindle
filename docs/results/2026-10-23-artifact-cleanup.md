# Artifact cleanup:164GiB reclaimed locally

At the user's request, remove inactive build caches and obsolete pre-pivot
checkpoint intermediates. No training/build processes are active during cleanup.
Local disk starts with366GiB free and finishes with529GiB free (rounded `df`
values). Measured free-space increases total175,786,864,640 bytes,163.7GiB.

| Removed | Count | Measured free-space increase |
| --- | ---: | ---: |
| Tagged Rust build caches inside local `runs/` |117 |168,115,712,000 bytes |
| Old top-level Pong history intermediate checkpoints |106 |7,671,152,640 bytes |

Caches are selected by valid Cargo cache markers and build-directory structure,
not merely a directory name. No symlink traversal or workspace-wide deletion.
Manifested device/inode checks validate exact targets before deletion. Unlinking
hardlinked cache files leaves copies outside those targets unchanged. Historical
cached executables may now need rebuilding from retained sources.

For the seven old `checkpoints/atari-pong-*-snapshots` histories, retain first
and final snapshots (13 distinct snapshots total); remove only intervening
steps. The plan retains each deleted checkpoint's metadata and existing
SHA256SUMS. **Deleted intermediate weights are not recoverable from Git**;
they were not moved to trash. Build caches can be regenerated.

Preserve all current CDP checkpoints, rollout videos, logs, reports, source
snapshots and the entire qualified runtime directory. Verify the native binary
and every source hash against its build receipt after cleanup. Independently
verify all saved tensor hashes in the pending B2/T64 smoke checkpoint at4096
actions/867 updates. No GPU calls, training restart, pin or numerical change.

Local `runs/` now uses694GiB and top-level `checkpoints/`6.8GiB. The shared
`/mnt/data` volume remains98% full,178GiB free; no unrelated shared data or
symlinked current run artifacts are removed or moved.

- [Exact cache deletion manifest](../../runs/artifact-cleanup-20261023.RE9G6xYv/cache-plan.json)
- [Cache result](../../runs/artifact-cleanup-20261023.RE9G6xYv/cache-result.json)
- [Checkpoint metadata, deletions and retained snapshots](../../runs/artifact-cleanup-20261023.RE9G6xYv/checkpoint-plan.json)
- [Checkpoint result](../../runs/artifact-cleanup-20261023.RE9G6xYv/checkpoint-result.json)

Future runs should release reproducible build caches once the required package
and source receipts are retained. Keep fixed evaluation checkpoints rather than
indefinite snapshot histories; preserve the checkpoint needed by unfinished work.
