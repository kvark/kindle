# Temporary storage cleanup

The first cleanup pass recovers114,987,143,168 bytes (about115 GB) from
`/x/Temp`. Its allocated size falls302,190,182,400→187,203,252,224 bytes.
The root filesystem has about305 GB available after cleanup.

- Remove8,490 disposable compiler intermediates from ten Cargo caches:8.40 GB.
  Keep executable validation tools, packaged native libraries, source and receipts.
- Deduplicate1,977 byte-identical immutable checkpoint/package paths:106.59 GB.
  Complete SHA256 matches, ownership/permissions and extended attributes must
  agree before replacement. Duplicate artifacts now share hardlinks; retained
  evidence continues to be immutable. Every original artifact path remains.
- Recheck553 canonical SHA256 values after replacement and verify all4,392 large
  artifact paths, sizes, permissions and owners. Unique data remains unchanged.

Both retained Breakout/Qbert native runtimes still have SHA256
`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`.
No GPU work, learner restart, checkpoint rewrite or source change is used for
cleanup. Complete inventory, removed-cache list, deduplication map and verification
receipts remain at `/mnt/data/kindle-temp-cleanup-20261109`.

## Follow-up pass

A further pass recovers **16,864,657,408 bytes (16.9 GB)** while preserving all
7,155 inventoried artifact paths. Combined recovery across both passes is about
132 GB. After the follow-up, `/x/Temp` occupies 180,876,046,336 bytes (168.5 GiB)
and the root filesystem has about 301 GB (280.7 GiB) available. Other work
between passes means these are separate measurements, not a direct subtraction
from the first pass's remaining capacity.

- Merge 33 additional byte-identical checkpoint/package/log paths using hardlinks:
  1,522,356,224 allocated bytes reclaimed. Check full SHA256, permissions, owners
  and extended attributes before replacement; recheck 27 canonical hashes after.
- Replace allocated zero-filled checkpoint blocks with filesystem holes:
  15,342,301,184 bytes reclaimed across 313 unique checkpoint inodes. Full SHA256
  matches before and after; file sizes, inodes, permissions, owners, link counts
  and modification times remain unchanged. No checkpoint content is rewritten.
- The first zero-block pass stops on a read-only checkpoint. Its log and 310
  successful per-file receipts remain intact. A separate completion script
  rechecks those hashes, finishes three writable files and leaves three read-only
  files alone without changing their permissions.
- No remaining disposable Cargo intermediates are found. No unique checkpoints,
  logs, source, runtime packages, videos or evaluation evidence are deleted.
  Both retained Breakout/Qbert runtime hashes above still match.

The latest experiment directory,
`/x/Temp/kindle-world-plain-rgb1-20261109`, is excluded from both follow-up
operations. No process has an open file under `/x/Temp` at the zero-block
pass's start; its learner and observer have already finished successfully.
No GPU work or learner restart is performed for cleanup.

New inventory, scripts, full-content verification, duplicate map, per-file
allocation changes, original failure and completion receipts are retained at
`/x/Artifacts/kindle-temp-cleanup-followup-20261109`. Existing cleanup and
experiment receipts are unchanged. Hardlinked retained artifacts remain
immutable; filesystem holes read back as the original zero bytes.
