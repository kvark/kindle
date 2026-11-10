# Temporary storage cleanup — November 10

The requested cleanup recovers **670,523,392 allocated bytes (671 MB)** from
`/x/Temp`, reducing it from 194,189,500,416 to 193,518,977,024 bytes
(180.2 GiB). The root filesystem has about 288.6 GB (268.7 GiB) available
afterward. These measurements are separate from the earlier cleanup passes.

- Share storage for 32 byte-identical artifact paths, reclaiming 606,892,032
  allocated bytes. Full SHA256, ownership, permissions and extended attributes
  agree before replacement; all 19 canonical hashes are rechecked afterward.
  Every original artifact path remains accessible.
- Remove 3,066 regenerable Python bytecode files, reclaiming 63,631,360 allocated
  bytes. Each corresponding Python source file remains. No source, checkpoint,
  native runtime, validation executable, replay, video or experiment log is removed.
- Verify all 8,054 inventoried large artifact paths, including sizes, ownership
  and permissions, and 670 distinct current experiment/retained-runtime hashes.
  Both retained agent runtimes still match SHA256
  `35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`.

No process has an open temporary artifact at inventory time. No GPU allocation,
learner restart or experiment rerun is performed. Hardlinked evidence remains
immutable. Inventory, scripts, replacement/removal maps, verification and final
measurements remain in `/x/Artifacts/kindle-temp-cleanup-20261110`.

## Larger archival option

Most remaining space is retained experiment data. The initial unique-inode
inventory attributes 133.8 GB to checkpoints, 22.5 GB to JSONL and 12.5 GB to stdout.

There are 2,478 unique large log inodes in older experiment directories, using
33,678,077,952 allocated bytes. The proposed archival scope includes files of
at least 1 MiB with `.jsonl`, `.stdout` or `.log` suffixes in dated roots through
November 6, excluding all `kindle-world-*` roots. Models, replay, packages, source
and recent world-model work stay at their existing paths.

Three representative large-log samples compress 1,573,425,739→223,934,011 bytes
with Zstandard level 3. Applying that sample ratio to the proposed log set suggests
roughly 28.9 GB of potential savings; actual compression is not yet measured for
the full set. Temporary sample copies are released; their hashes, sizes and
source paths remain recorded.

The user has been asked whether older evidence may move into verified compressed
archives, requiring extraction before reuse. **That choice is pending; no original
log has been removed or archived in place.** The proposed file inventory is
`/x/Artifacts/kindle-temp-cleanup-20261110/log-archive-plan.json`.
Any archival pass must retain original content hashes, path/metadata mappings
and a restoration command, and verify decompressed contents before removing an
original. Existing experiment receipts are not rewritten to pretend archived
paths remain directly accessible.

## Verified archive prepared

The follow-up prepares and verifies the proposed archive without removing any
original logs. All **2,478 members**, covering **33,671,117,373 content bytes**
across 213 older experiment roots, pass full decompressed SHA256 and
size/permission/owner checks. Preparation and verification take 87.760 seconds.
The compressed archive is **4,476,480,704 bytes (4.48 GB)**. Five selected paths
share inodes with other retained paths; allowing for those remaining links and
the archive's allocation, removing the selected originals would reclaim
**29,154,574,336 allocated bytes (29.15 GB / 27.15 GiB) net**. No such removal
has been performed: the prepared copy currently uses additional space.

Archive, per-file content hashes/metadata, verification log and restoration
script are retained at `/x/Artifacts/kindle-log-archive-20261110`.
Archive SHA256 is
`f18850460717ee25ab1cba752e3de5c264309fea683d4f6c28738c60856e2d1d`.
To restore into a new directory, refusing to overwrite existing files:

```sh
python3 /x/Artifacts/kindle-log-archive-20261110/restore.py --destination /x/Recovered/kindle-logs
```

The same command with `--destination /x/Temp` restores the original log paths
after archival. It verifies the archive and restored content hashes and restores
recorded permissions, extended attributes and timestamps. Source hardlinks
restore as independent files. Checkpoints, replay, source, runtime packages and
all `kindle-world-*` roots remain outside this archive. The archive choice is
still pending; no additional space recovery is claimed.
