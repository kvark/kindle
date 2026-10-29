# FIRE supervision exists, but sampled training avoids the long greedy stall

The separately selected [fresh200k comparison](../experiments/2026-10-29-cdp-breakout-fire-weight-200k.md)
starts22:43:07 UTC October29/sourcece86d72 under the30-minute cap, with100k
recovery and four frozen arms. All196 pins,372 bounded-matching initial tensors,
47 pre-update semantic events and427 finite reports through7208 actions pass
startup checks. Recovery at100k actions/6227 updates passes22:54:42 UTC, with
all372 tensors finite and6489 finite reports through104,200 actions. The recovery
observer succeeds; MainPID354340 and the remaining three observers are active.
No full frozen result is available; do not restart this learner. This is a fresh
experiment, with no bounded-check experience reused.

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

## The bounded FIRE-weight check passes execution and cost

The [fixed bounded check](../experiments/2026-10-29-cdp-breakout-fire-weight.md)
completes **8192 actions /32,736 frames /489 updates in48.059s**. Frozen restore
then completes **1024 actions /4,076 frames in.971s**, with zero updates and
all372 tensors unchanged. The short frozen arm has eight natural endings, all
with zero reward; this is not a gameplay gain. No long frozen evaluation occurs.

The conservative200k projection, with the predeclared10% allowance, is
**1290.650s (21m31s)**, below the30-minute gate. This is an early cost estimate,
not a measured full-run duration or speedup. Native2974160 and the16,990,739
parameter architecture remain unchanged.

All489 reports contain finite supervised and centroid metrics. The native
weight sums verify **20,312 repeated FIRE replay targets** among117,054 labeled
rows: `117054 + 3*20312 = 177990` total target weight. These counts follow from
the single changed class weight; they are not distinct demonstrations or gradient
norms. The maximum raw world/actor expert-loss difference is1.5e-6. All8192
collected states receive labels, including1124 FIRE labels and4261 passive
labels. Actual teacher overrides are3931/8192 (47.99%).

All372 initial tensors match the previous centroid check;357 shared tensors
match the retained control, and47 pre-update semantic events reproduce exactly.
CNN, reward, centroid, actor and value parameters change. The full CPU replay
verifies every training pixel, action, corrective label, reward, boundary, reset
and frame count. All142 pins, both host guards, checkpoint schemas and terminal
collection pass. The service starts **22:26:22 UTC October29**, source `c4bb03b`,
and succeeds **22:27:25**; collection succeeds **22:27:54**. No declared training
or frozen stage repeats, and no GPU allocation remains active at collection.

### Preserved preflight harness failure

Before the declared service, the first command preflight unexpectedly constructs
a native agent: resolving `kindle.default_config` repopulates lazy native exports
after the harness replaces `kindle.VectorAgent`. The fake frame then fails shape
validation at `begin_episodes`. The constructed-device receipt identifies the
integrated **AMD Ryzen5 9600X /RADV RAPHAEL_MENDOCINO**, not the intended RX7900XT.
This initialization was unguarded and is not described as CPU-only.

The original source, zero-step/zero-update memory receipt, empty run log and
failure are retained. No initial checkpoint, action or learner update occurs.
The process exits; a retained kernel-journal review finds no GPU fault/reset,
and no host reset is performed. The corrected harness resolves the lazy exports
first, replaces constructors in both public and native modules, and checks for
GPU device descriptors. Both corrected command boundaries pass without native
construction or real environments. The subsequent declared stages run on the
explicit RX7900XT under their normal guards.

Evidence: `/x/Temp/kindle-breakout-fire-weight-20261029`, with bounded artifacts
under `check/`. No full learner, coefficient sweep or automatic successor is
selected. The original centroid full learner remains `closed_protocol_failed`;
this adjustment still has no measured gameplay benefit.

- Bounded declaration: `c88778421b34165943749f0979c740b5461151a1d9ed7a0f04903d3e32354cf2`.
- Bounded review: `ccf30109cdb3855f23d265630b0ef8f11786ec5ba75d589f58fba5d9e6ca21de`.
- Completion: `4f9d01ec03e38a2f520c2103877d0e7a8d9c735f357e5a9167add9632b2576cc`.
