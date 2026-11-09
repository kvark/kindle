# Temporary storage cleanup

User-requested cleanup recovers114,987,143,168 bytes (about115 GB) from
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
