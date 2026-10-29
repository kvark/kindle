# FIRE supervision exists, but sampled training avoids the long greedy stall

The CPU audit of both completed200k training logs finds no missing FIRE target
path. All collected states receive corrective labels, and **99.805%** of
eligible sampled replay rows contribute imitation in the centroid run. Its
23,317 FIRE labels represent11.66% of collected actions. The native target
builder aligns outgoing labels with the preceding state and excludes reset/end
boundaries and sequence tails. The same weighted loss trains the actor and,
with world gradients enabled, the representation through a frozen actor.

Training and frozen greedy behavior differ substantially. In the final50k
actions, the centroid learner follows **2081/2656 passive FIRE labels (78.35%)**.
Its longest consecutive passive FIRE-label run without any FIRE-bearing action
is **45 actions**, compared with the verified frozen12,828-action wait to serve.
Sampling, changing parameters and encountered states differ; this does not
identify which difference causes the frozen failure.

| Saved training metric | Retained control | Centroid100 |
| --- | ---: | ---: |
| FIRE labels, all200k states | 24,036 | 23,317 |
| FIRE labels without an override | 14,066 | 13,521 |
| Exact FIRE choices on those passive labels | 6181/14066 (43.94%) | 6282/13521 (46.46%) |
| Exact FIRE choices on late passive labels | 2138/2897 (73.80%) | 2081/2656 (78.35%) |
| Any FIRE-bearing action on late passive labels | 2152/2897 (74.28%) | 2098/2656 (78.99%) |
| NOOP on late passive FIRE labels | 293 | 184 |
| Longest passive FIRE-label run without FIRE-bearing action | 44 | 45 |
| Labeled sampled rows /eligible rows | 2,988,570 /2,994,480 | 2,988,645 /2,994,480 |

Neither training log contains an exact full input-pixel match to the363 unique
images in the timeout's12,250-action constant-playfield interval. This is a
strict overlap test that includes score, life display and paddle position;
zero matches does not establish perceptual novelty or absent serving examples.
The teacher's FIRE category also includes balls outside its crop, so label runs
must not be relabeled as verified waiting-to-serve intervals. Class-specific
sampled replay counts and gradient magnitudes were not logged.

The audit reads saved logs only: **2.668s CPU**, no environments, native model or
updates. Both full source guidance replays already verify pixel/action/label
identity. All16 inputs remain unchanged; counts match the prior exposure
receipts. Every50k window, passive/guided confusion matrix and the20 longest
runs per learner are retained at
`/x/Temp/kindle-breakout-fire-supervision-20261029`.

- Review: `8fe31a820715c1864dc4b087d927f0d2426c7bf624a426409b54413e22c74c24`.
- Completion: `0d752f3c027ecb1bb2390499802dd12f76afd23967055d391069cdc28677583c`.

## Selected bounded training adjustment

Select a single **FIRE weight1→4** check, retaining centroid100/expert100,
12M64/Auto/N8/B16/T16/H15/R16 and the existing teacher, rates and guidance.
The added `--guidance-fire-weight` option supplies the existing native class
weight by action meaning. Its default is1; frozen restore rejects an override.
Weighting affects both actor and world supervision and raises total imitation
magnitude; it is not merely a fixed-total redistribution. The native2974160
runtime and all model shapes stay unchanged.

All **699 CPU checks pass**, with10 GPU-constructor validation cases deselected
and zero native construction attempts. The16 new cases exercise defaults,
explicit1/4 weights, action-meaning lookup, world-gradient preservation,
invalid/nonfinite/product-overflow inputs and frozen override rejection.

[Fixed bounded protocol](../experiments/2026-10-29-cdp-breakout-fire-weight.md):
8192 fresh actions/489 updates, then1024 frozen actions, exact initial/frozen
tensors, full guided replay and host guards. The check has not yet run at this
revision. No full learner or automatic successor is selected. The original
centroid full learner remains `closed_protocol_failed`; this adjustment has no
measured gameplay benefit.
