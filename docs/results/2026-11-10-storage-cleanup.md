# Old runs, partial outputs and worktree cleanup

The requested cleanup increases available filesystem space by
**39,413,006,336 bytes (39.41 GB /36.71 GiB)** during the measured pass.
At completion `/x` has **316,198,625,280 bytes (316.20 GB /294.48 GiB)**
available. `/x/Temp` occupies164,863,770,624 bytes (153.54 GiB).
Other host work can change these capacity readings.

The removed paths account for40,778,244,096 allocated bytes. That sum is not
an isolated filesystem-recovery measurement: shared hardlinks and concurrent
host writes can make the available-space delta differ.

| Cleanup | Allocated bytes removed from the selected paths |
|---|---:|
|2,478 older logs, after verified lossless archival|33,631,059,968|
|Meganeura's unused Cargo build cache|6,928,257,024|
|Obsolete plotting virtual environment|143,499,264|
|Three old worktrees, including their disposable build outputs|75,427,840|

Also remove393 empty old output directories and prune two stale worktree
registrations whose working directories no longer exist. No large partial
download/core files are found in the inspected Kindle/Meganeura paths.

## Source worktrees and recovery

Retire these obsolete worktrees:

- `/x/Code/meganeura-low-precision`: retain branch
  `refactor/shared-low-precision`, its HEAD349a1f88402b6eff9dac9e66c07835b8232db180
  and all uncommitted/untracked source in a verified2,358,737-byte archive.
- Mind-Games' October11 `async-checkpoint-verification/recheck-source`:
  HEAD257f7ce7c4ce68b10a2f3ac669cb00eb44e2876e;2,744,060-byte source archive.
- Mind-Games' October23 `binding-candidate/source`:
  HEADacf531fdec1582c54cb7a20a6ba6b8ad26412021;5,073,748-byte source archive.

Before removal, verify every archived file's content/metadata and every symlink.
Restore all three into temporary independent clones and verify their exact Git
status, including untracked files and deletions. No selected tree has staged
changes; index files and patches are retained too. Remove the temporary
validation clones afterward. Source snapshots exclude only their root `.git`
pointer and regenerable `target` cache. Existing main repositories, their HEADs
and working changes remain unchanged. New `refs/archive/cleanup-20261110/*`
references protect every retired/detached HEAD from collection.

Prune the missing Meganeura `pr336.../engine` and Inferena `pr336.../runner`
registrations after preserving their HEAD references. Meganeura and Mind-Games
now each list only their main worktree; Inferena's stale registration is gone.
Keep `/x/Code/ommatidia/target/patched-naga`: Ommatidia's current Cargo manifest
still depends on it, and it contains uncommitted changes.

Recovery into a new destination, using the retained main Git repository:

```sh
python3 /x/Artifacts/kindle-storage-cleanup-20261110/restore_worktree.py meganeura-low-precision /x/Recovered/meganeura-low-precision
```

The helper also accepts `mind-games-recheck` and `mind-games-binding`. It
refuses an existing destination, verifies archive/source hashes and checks the
recovered Git status. A source-only extraction remains possible independently
of the Git repository; the standalone recovery clone normally shares the
retained repository's object store.

## Older logs and unused environment

The earlier [prepared archive](2026-11-10-temp-cleanup.md) is now used for the
user's explicit request to clean old runs. Recheck its SHA256 and every original
log's full SHA256, inode/size/mtime/permissions/ownership before removing any
original. The4,476,480,704-byte archive remains at
`/x/Artifacts/kindle-log-archive-20261110/older-logs.tar.zst`, SHA256
`f18850460717ee25ab1cba752e3de5c264309fea683d4f6c28738c60856e2d1d`.

Those2,478 selected older paths are removed; their content is recoverable.
Five other retained hardlinks remain and are excluded from the recovered-byte
sum. Relative to the prearchive baseline, net older-log savings are
29,154,574,336 bytes. This pass's larger log-path recovery reflects the fact
that the archive copy already existed before the pass started. Historical
experiment receipts are unchanged: restore a missing log before using a check
that expects its original pathname.

```sh
python3 /x/Artifacts/kindle-log-archive-20261110/restore.py --destination /x/Recovered/kindle-logs
```

All selected original paths can instead be restored with `--destination
/x/Temp`; existing destination files are never overwritten. Original content
hashes, metadata and the archive's preparation verification remain retained.

Remove the unused October25 `plot-venv`, retaining its Python configuration and
exact installed-package list as
`old-plot-environment-requirements.txt` in the new cleanup artifact directory.
Do not touch current Python environments or unique model/data assets.

## Paused learner preserved

All **280 distinct protected input hashes** pass before and after, covering
the current learner/CPU plan, qualified native package/closure and pending
loss-support audit. The original learner and both monitors retain their exact
PIDs/invocations and `FreezerState=frozen`. Last logged update remains9,964;
live optimizer age is not independently measured. No thaw, restore, RNG reset,
new GPU allocation or goal work occurs during cleanup. Shared checkpoints,
retained agent/world runtimes, models and unique datasets remain accessible.

The first normal status query on an old Mind-Games worktree fails because a
submodule path is a symlink. The separate `--ignore-submodules=all` inventory
and full filesystem/symlink backup preserve its state; no source deletion follows
the failed query. Backup, restoration and cleanup execution all complete once
without a retry. No source change or commit is made in neighboring repositories;
only worktree metadata, durable retention refs and selected derived outputs change.

Inventory, manifests, file hashes, source archives, recovery helper, restoration
checks, removal lists, original measurements and verification receipts remain at
`/x/Artifacts/kindle-storage-cleanup-20261110`.
