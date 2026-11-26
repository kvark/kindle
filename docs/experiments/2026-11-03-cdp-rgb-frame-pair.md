# Optional native two-image input for centered CDP

Select a default-off native prototype and bounded execution/cost check within
the unfinished five-game goal. Breakout retains20% guidance,12M64/B16/T16/H15/R16,
encoder/base4e-5, dynamics4e-4, behavior8e-5, replay-value1 andexpert100.
More guidance, larger capacity, longer replay context andthe fixed pixel-motion
action readout have not improved the primary retained control. The readout's
negative result supplies no positive evidence for this change; it also did not
test a CNN trained jointly on consecutive images. This prototype tests that
distinct input representation, without claiming motion is the causal bottleneck.

Add optional `rgb_frame_pair=true` for learned RGB/CDP only. Keep the default
serialized configuration andsingle-image path unchanged. Each stream supplies
its immediately previous andcurrent pooled native observation, oldest first,
as six centered CHW channels. Resize each arriving image once onGPU. Preserve
the previous resized image onGPU; reset duplicates the new episode's first
image into both halves. Subset arrivals must preserve other streams' histories,
andno pair may cross a reset. Both halves enter replay together andremain fixed
when the CNN updates. Restore preserves this architecture andbegins with fresh
live history, not checkpointed game/RNG/replay continuation.

The first convolution takes six channels; later CNN, RSSM andactor/critic widths
stay fixed. RGB reconstruction andcolor-centroid supervision are unsupported
with this option andmust reject before native construction. Python supplies
only an explicit fresh-training option andrecords preprocessing; native Rust,
Meganeura andBlade retain all preprocessing, learning andinference.

Before GPU work, check configuration/serialization/restore rejection, graph and
checkpoint shapes, derivatives through both image halves andCLI metadata onCPU.
Run relevant Rust/Python CPU tests andstrict Clippy. Build an isolated runtime;
pin sources, dependencies, compiler andbinary identity. Preserve the retained
runtime andoriginal failures. No neighboring worktree orlarge Git artifact.

Guarded GPU prerequisites are: independent resized-byte comparisons at32/48/64,
multiple streams, missing arrivals, episode resets andCPU/resident frame sources;
six-channel CNN values/derivatives against a scalar reference; replay immutability,
finite learning andfrozen restore with the saved pair architecture. Check old
single-image compatibility against retained initial/frozen evidence. These are
new implementation checks, not reruns of completed original experiment stages.

Only after prerequisites pass, separately prepare one fresh8192-action/489-update
Breakout check and1024-action frozen restore, seed1009/evaluation4000001009,
20% episode guidance/cap1920 andall-state labels. Use Auto/N8 andthe retained
settings above, native frames/oneGPU64 resize, full18/sticky.25/repeat4/noops0.
Save actual initial/final/frozen weights; derive six-channel shapes independently,
require finite reports andchanged CNN/reward/actor/value, full recorded training
pixel/action/label replay andfrozen tensor invariance. Compare runtime costs with
the retained8192-action check (47.690s), without an isolated speed claim.

Require training_seconds*200000/8192*1.1 <=1800 before any full learner. Report
construction separately. Pair replay stores twice as many RGB values: derive
the full100k replay footprint andrequire at least2GiB projected GPU headroom
after remaining replay pages, in addition to observed bounded headroom. An early
projection is not a full-memory orfull-learning result. Frozen1024 actions may
contain no natural episodes andcannot establish competence.

Stop on faults/nonfinite values; no overlap ordriver reset. No full200k learner,
extension, sweep orautomatic successor isselected here. Preserve the retained
70.750 sampled/143.667 greedy point andthe original864/90%-two-wall goal. A full
comparison requires separate evidence-based selection after bounded closure.
