# Longer Qbert sequences lower frozen scores at slightly higher cost

Fresh B8/T32 completes **200,000 actions /799,710 emulator frames /12,469
updates in31m48s**. Greedy mean11609.375 is3.13% below the retained B16/T16
control's11984.375; both clear24/24 first pyramids. Sampled mean6120.833 is9.45%
lower than6759.375, with20/24 pyramids versus24/24. Training takes4.44% longer.
Both modes miss the original15000 mean gate; sampled also misses90% pyramids.
Retain B16/T16. No extension, confirmation or automatic successor is selected.

| Fresh12M48/R16/expert100 learner | Greedy mean / pyramids | Sampled mean / pyramids | Training |
| --- | ---: | ---: | ---: |
| Retained B16/T16 |11984.375 /24 of24|6759.375 /24 of24|30m27s|
| Candidate B8/T32 |11609.375 /24 of24|6120.833 /20 of24|31m48s|

Own actual-initial means are80.208 greedy and177.083 sampled, with zero pyramids.
All357 initial tensors and paired sampled/greedy initial semantic traces match
the control. Each frozen arm selects the first3 natural episodes per8 lanes=24,
with guidance/labels off and zero updates/truncations. Extra episodes and partial
tails remain in the evidence. Four arms total96,608 actions /386,285 frames /
58.074s. The control's later temperature.5 score12389.583 remains a separate
inference option; no temperature is selected for this candidate.

Sequence/world-backprop length16→32 and batch/world-microbatch16→8 preserve256
sampled replay positions per update andR16. Temporal gradients, sequence diversity
and sampling change together; behavior-target rows increase240→248. Warmup moves
376→504 actions, yielding eight fewer updates than the control's12,477. This
does not isolate a memory effect or establish that longer context never helps.
The reused4013 learner seed and evaluation cohort are development evidence,
not independent confirmation. Capacity, geometry, native447fe50, runner, teacher,
guidance and loss/rate settings remain fixed.

Actual training overrides are85,900 actions (42.950%);45 of197 episode starts are
selected for guidance. The44 completed selected episodes average15131.250 and
all clear a first pyramid;145 unguided episodes average2558.276 with63 pyramids.
These online returns include changing weights and teacher actions and are not
frozen competence. There are9,640 native actions after cap release. All12,469
reports carry supervision, totaling3,089,039 labeled replay rows; actor/world
raw expert losses differ by at most6e-7. Repeated replay rows are not unique
demonstrations. The inspected learning-curve plot shows similar aided online
curves despite the lower fixed-final frozen results.

All229 declared pins, exact config/source/counters,100k/6219 recovery,357 finite
initial and invariant frozen tensors, parameter changes, full training pixel/
action/corrective-label replay, frozen outcome replays/videos, host guards,
curves, coverage, exposure and completed comparison pass. Ordinary frozen logs
lack live pixel hashes; outcome replay is not direct live-image equality.
All four services and the terminal collector succeed. No failure, GPU repeat,
cap change or host recovery occurs. The.788s co-hosted source review remains
a timing caveat, with no wall-time correction.

Service `kindle-qbert-context32-200k-20261029` starts14:31:43 UTC October29,
source `07ce3dc`. Training ends15:03:38 and all GPU stages end15:05:03. The full
training replay finishes15:15:59; main succeeds then after44m16.442s, including
CPU audits. Postprocessing succeeds15:16:17 and collection15:17:05. Evidence
root `/x/Temp/kindle-qbert-context32-20261029/full` retains logs, weights, videos,
plots and receipts outside Git. No GPU allocation is active at completion.

- Declaration: `40c0f68ba3a836e56832c69a2efe370058b6085f5352d602dafb576b45851c07`
- Learning review: `959bf1281dd30f0e69ad933f010c8d48f86d1ea667284d6f343a375b075a9390`
- Comparison: `dfcce6d94f39ea3d6fb3be9ee1d2194bee61ea2bd32b2addd667bfb57b28dfe3`
- Completion: `4dc567bcc744515552b3f7f6438e022da8e38d69a71455ad90777887733d194b`

See the [full protocol](../experiments/2026-10-29-cdp-qbert-context32-200k.md).
The five-game goal remains unfinished.

## Bounded execution and cost check

Fresh 12M48/Auto/N8/B8/T32/H15/R16 completes **8,192 actions / 32,756 emulator
frames / 481 updates in 72.797s**. The subsequent 1,024-action frozen restore
takes .788s with 4,096 frames and zero updates. No natural frozen episodes
complete, so this check establishes execution and early cost, not task competence.

All 41 declared inputs remain unchanged. All 357 initial tensors match the
retained B16/T16 control; all 357 frozen tensors are finite and unchanged.
Config/source/counter checks, changed CNN/reward/actor/value weights, aligned
actor/world expert losses, complete training pixel/action/corrective-label replay
and both host guards pass. First update504 and final481 updates match the
independently derived schedule. All481 reports have nonzero supervision.
Actual overrides are4,391/8,192 (53.601%), not the20% episode selection rate.

The conservative200k projection is **1,955.004s (32m35s)**, using measured time
times200000/8192 times1.10. This passes the declared45min ceiling but is not a
guaranteed full-run duration. B16/T16→B8/T32 preserves256 replay positions per
update andR16 while changing temporal gradient length and sequence diversity.
Behavior-target rows rise240→248; later warmup produces eight fewer optimizer
updates at a fixed action budget. Capacity, native447fe50, runner, teacher and
supervision stay fixed. No isolated memory effect is claimed.

Service `kindle-qbert-context32-check-20261029` starts14:26:12 UTC October29,
source `c1ca09b`, and succeeds14:28:05 after1m53.315s including CPU replay.
Terminal collection passes14:28:28. No fault, failed stage, GPU repeat, cap
change or recovery occurs. Evidence remains outside Git under
`/x/Temp/kindle-qbert-context32-20261029/native-check`.

- Declaration: `4c81d61ba7d955d35a25d6d8ca4b2f22a1575e29ea7f401ed75b45ac18d71811`
- Review: `5e749eb457194c03366db1465fea88c04c367ea1e0834215826195da3ea666cd`
- Completion: `d28ff70c8c2d053fb058e861b34d998e3d8d5ae6caa29598027a6d3702cf0f2e`

See the [bounded protocol](../experiments/2026-10-29-cdp-qbert-context32.md).
After this check, the fresh200k life reported above was separately selected.
No bounded weights or experience were reused.
