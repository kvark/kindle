# Correct the demonstrated packed-layout dispatch limit

October14,2026. Complete; [result](../results/2026-10-14-packing-grid.md).
Original declaration follows the [B8/B32 result](../results/2026-10-14-batch-granularity.md).
This is a bounded implementation check, not an Atari allocation or batch
sweep. No GPU job for this correction had started at declaration.

Four RGB B32 convolutions lose the packed path because their64MiB packing
buffer requires65,536 X workgroups. The existing32Mi-element bound permits
that allocation, but one-dimensional admission rejects the grid. Meganeura's
existing `linear_grid` represents it as32,768×2×1. Use that helper and flatten
the packing shader's actual X/Y grid; do not add a new BF16-specific kernel.

Keep the packing element/memory cap, shared-memory gate, convolution tiles,
precision, reduction partitions, numerical tolerances and fallback conditions
otherwise unchanged. Preserve the separate reviewed128MiB logical gradient-
scratch setting in both controls. Work in a new isolated source snapshot,
retaining all completed batch artifacts and latest-upstream review.

Before GPU work, prove exact address coverage at the65,535/65,536 boundary,
padding/transposition correctness, bounds and unchanged B8 selections. Use all
supported packing storage types. The four B32 admissions must be explained
exactly, not enabled by broadening unrelated resource limits. Qualification
must test real two-dimensional dispatches, including nonconstant input, padded
channels, a second grid row and guard/tail behavior. Retain existing numerical
derivative checks for the packed family.

Then use finite short full-learner checks and one matched complete B32 timing
pair with identical source/model/optimizer/seed/batch settings except the grid
correction. Preserve B8's exact control. B32 now selects previously qualified
packed kernels: arithmetic association and later trajectories may differ,
and must be reported rather than hidden. Require independent numerical checks,
finite reports/checkpoints and exact initials; no learning-equivalence claim.
All packing costs must be included in the full update, with command caching
and the ordinary four-world-chunk schedule. Do not time just the pack kernel.

A measured whole-update benefit justifies keeping the correction isolated;
it does not close the full performance goal or justify a tile/grid sweep.
If it fails correctness, review before follow-up. If it offers no meaningful
whole-update improvement, retain the result and return to measured algorithm/
compiler costs. No new JAX benchmark or learning allocation is required here.

Declare exact commands/binary hashes after the CPU implementation review and
before GPU execution. Serial host-guarded persistent services, bounded deadlines,
expected5080 and>=2GiB sampled estimated Vulkan headroom; record standalone
allocation warnings. No NVML polling/recovery, blind retry, production change
or concurrent learner. Preparation one CPU/2GiB/no swap; GPU workers no quota.
Reserved artifacts: `runs/packing-grid-20261014.3ge6SKP4`.

Qualification is14 packing cases/28 executions: F32/F16/BF16, both storage
orders, changed inputs, the exact B32 allocation and a two-row partial tail.
Check every output element, padding zero and64 tail canaries. Reuse the existing
40 packed-convolution F64 cases/seven independent partial checks. CPU checks
cover both sides of the dispatch limit, the unchanged memory-bound rejection,
all24 typed shader variants and the two retained full plans: B8 identical,
exactly four new B32 packed convolutions/four two-dimensional packs.
Each numerical job is bounded300s; no game actions or optimizer updates.

After numerical review, run one24-update B8 candidate smoke and one24-update
B32 candidate smoke. Require B8 exact against the retained24-update control;
B32 exact initials, finite states/reports and reviewed changed selections.
Then one fresh B32 control/candidate pair,16 warmup+256 measured updates each,
900s per job. Retain if the qualified candidate reduces complete-update time
by at least3%; otherwise close this performance follow-up. This is592 total
full synthetic updates, no learning allocation or new JAX run. The initial CPU
test build's missing `ShaderEntry` clone and its source/log remain; no GPU job
had started at that failure.

The next CPU boundary test initially asserted legality of unrelated casts in
its artificial large graph. Standalone To/FromBF16 conversion still has a
one-dimensional launch in current upstream7574ad1; that is a separate limit,
not part of packing admission. Retain this failure/source and constrain the
admission assertion to the packing/convolution dispatches it tests. Check all
selected grids in the retained B8/B32 plans separately; do not silently send
the artificial graph to GPU or change conversion kernels in this trial.

A separately declared120-second, initialization-only `vulkaninfo` job confirms
the5080 advertises2,147,483,647 /65,535 /65,535 workgroups. Its larger X grids
are legal:65,535 is the portable minimum, not this device's X limit. Audit all
retained selected grids against those actual limits. The two-dimensional
packing correction removes an unnecessarily conservative selection rejection
while remaining portable; it is not recovery from an observed invalid5080
dispatch. Keep this additional passing guard and raw capability output.

Both24-update candidate smokes finish with passing guards. The first CPU
review fails because it zips dispatches by position, but packing legitimately
changes the scheduler's topological order. A separate read-only audit finds
identical output-buffer identities and only the four declared convolution
descriptor/grid changes; the other three sessions are unchanged. Retain the
failed reviewer/log. The corrected check matches by output identity, verifies
every producer precedes its consumers, checks exact packing provenance and
all unchanged dispatch fields, and admits only those four typed replacements.
No numerical tolerance, GPU source or test result changes; no GPU rerun.

The corrected smoke review passes. Both272-update ordinary timing jobs then
finish with passing guards. Final CPU review reaches its300-second deadline
while repeatedly stream-sorting complete plans. Retain that timeout and source;
replace only this costly canonicalization with whole parsed-plan canonical JSON,
including every field/value and the same legacy-empty-output normalization.
Run the CPU review with the same one-CPU/2GiB/zero-swap bounds and600-second
deadline. No GPU source, run, tolerance or selected result is changed/repeated.
