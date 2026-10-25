# The faster derivative qualifies, but covers one operation

October15,2026. [Declaration](../experiments/2026-10-15-convolution-qualification.md) ·
[Data](2026-10-15-convolution-qualification.json).

Fresh native/candidate CNN1 dX takes **2.14890→1.29280ms,39.84% less GPU time**;
queued wall time2.15014→1.29630ms,39.71% less. All GPU packs, scratch and
dependencies are included on every call. The fixed-address K32 proof clears
its20% complete-operation gate. This is not a complete-update improvement.

Broader BF16 qualification checks every output on three signed deterministic
distributions, including small wide-exponent gradients and sparse larger spikes.
All196,608 F64 comparisons pass unchanged gates: worst relativeL2=1.412e−6
against2e−5; no pointwise failures. K64/K32 outputs are also byte-identical in
these fixtures. Uploaded input bits are independently regenerated; changed-input
and restore checks pass. These are synthetic distributions, not captured learning
gradients or qualification of every shape/dtype.

The ordinary pair checks33,592,832 exact output values and128 retained F64
probes. Native plans/pipeline inventories reproduce their retained control;
its unchanged dW measurement is not a candidate dW result. Candidate scratch is
85,037,056 bytes. Across qualification: eight passing guards, zero warnings,
456 operation calls/930 shader dispatches, zero learner updates/game actions.

## Impact and next check

The actual current staged world plan has eight forward convolutions, seven
input gradients and eight weight gradients. The tested CNN1 dX appears **once**;
posterior/world labels alias the same plan and must not be double-counted.
No other learner session contains convolution. Its0.85610ms isolated saving is
only1.43% of the retained60.041ms complete update, before integration effects.
It cannot explain or close the remaining2.25× gap to JAX26.632ms.

The subsequent [fixed spatial-family check](2026-10-15-convolution-family.md)
rejects the shared replacement:12.833→18.131ms across actual forward/dX
occurrences. It supersedes the proposed compiler integration, not this isolated
numerical result. No tile sweep or large compiler project. Weight-gradient data feeding,
host targets and command-execution costs remain distinct. Production and Atari
stay unchanged; the full root-cause/performance goal remains open.

Artifacts: `runs/convolution-qualification-20261015.GqNMZ6qJ`. The publication
audit initially exceeded its2GiB CPU limit while parsing large plans; a second
full-tree jq attempt also exceeded it. Both failed services/sources/logs remain.
The corrected streaming audit passes without a GPU rerun or raised memory limit.
The original native queued-wall field included command recording. A separate
CPU audit subtracts the recorded.002317ms/call; the corrected39.71% gain still
passes. Original reports remain, without retiming or a changed gate.
