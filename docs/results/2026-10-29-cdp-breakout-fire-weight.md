# Stronger FIRE imitation completes without timeouts but does not improve Breakout

The fresh 200k-action comparison completes in **22m07s** with all audits passing.
Frozen sampled **12.375** and greedy **17.792** do not improve the retained
control's **13.208 / 17.875**; all have zero of 24 two-wall successes. All four
candidate evaluations finish without truncation. Retain the accepted control;
no weight sweep, extension or confirmation is selected. The previous centroid
learner's protocol failure remains unchanged. [Full results below](#full-comparison-completes-without-timeouts-but-does-not-improve-score).

## Existing FIRE supervision and the training/frozen gap

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
under `check/`. At bounded-check completion no full learner, coefficient sweep
or automatic successor was selected. The separate full comparison follows below.
The original centroid full learner remains `closed_protocol_failed`.

- Bounded declaration: `c88778421b34165943749f0979c740b5461151a1d9ed7a0f04903d3e32354cf2`.
- Bounded review: `ccf30109cdb3855f23d265630b0ef8f11786ec5ba75d589f58fba5d9e6ca21de`.
- Completion: `4f9d01ec03e38a2f520c2103877d0e7a8d9c735f357e5a9167add9632b2576cc`.

## Full comparison completes without timeouts but does not improve score

The [fixed fresh comparison](../experiments/2026-10-29-cdp-breakout-fire-weight-200k.md)
completes **200,000 actions / 799,434 frames / 12,477 updates in 1,327.241s
(22m07s)**. The retained control takes 1,339.198s (22m19s), a 0.89% difference
that does not establish an isolated speedup. The model remains
12M64/Auto/N8/B16/T16/H15/R16, with centroid scale100, expert scale100, world
gradients and the existing pixel teacher. Only exact FIRE's weight changes
1→4 relative to the previous centroid configuration. No prior experience is reused.

| Frozen actions | Own initial | Final | Accepted control final | Final two-wall successes | Total candidate timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sampled | 1.417 | 12.375 | 13.208 | 0/24 | 0 |
| Greedy | 2.208 | 17.792 | 17.875 | 0/24 | 0 |

Each mean uses the first three natural episodes per lane, 24 total. Sampled
score is 6.31% lower and greedy 0.47% lower than the accepted control. All four
final/actual-initial arms finish with zero updates and zero truncations:
**45,488 actions / 181,795 frames / 29.271s** in total. The zero-truncation
protocol passes, but the original 90%-two-wall task gate does not.

The previous centroid weight1 learner's sampled14.375 / selected-natural
greedy20.125 remain descriptive context: its greedy timeout still makes that
comparison protocol-failed. This new completed run does not repair the old result.
Relative to the accepted control, native/runner versions, the centroid head/loss,
extra656,386 parameters and FIRE weight differ. Weight4 also increases total
imitation magnitude. Reused seed1009 and evaluation lanes, changed trajectories
and these configuration differences prevent isolated causal or reliability claims.

### FIRE commands and training fit

| Frozen final diagnostic | Sampled | Greedy |
| --- | ---: | ---: |
| All recorded actions | 14,208 | 19,400 |
| All FIRE-bearing actions | 1,486 | 1,506 |
| Longest FIRE-free command interval, all recorded | 123 | 167 |
| Longest reward-free episode tail, all recorded | 56 | 44 |
| Selected natural-cohort actions | 11,699 | 15,723 |
| Selected FIRE-bearing actions | 1,247 | 1,264 |
| Longest selected reward-free episode tail | 56 | 40 |

These counts include all FIRE-bearing action aliases and preserve excess
completed episodes and partial segments. No interval approaches the prior
verified12,828-action stall, but command gaps also include normal ball play.
This is not a new pixel/life-based measurement of waiting to serve, a matched-state
intervention, or evidence that serving failures are eliminated generally.
The absence of a timeout here accompanies no score gain; pressing FIRE was a
concrete failure in the previous rollout, not a sufficient explanation or fix
for the overall performance gap.

All12,477 learner reports are finite. Repeated sampled supervision contains
**2,988,487 labeled rows**, including **399,577 FIRE targets**; total weight
**4,187,218 = 2,988,487 + 3×399,577**. World and actor raw expert losses agree
within1.5e-6. The collected trajectory has24,016 FIRE labels and82,657 passive
labels. Actual teacher overrides total117,343 (**58.6715%**). Guided completed
episodes average191.167, while unguided changing-policy episodes average4.440;
neither replaces frozen evaluation.

| Training action window | Raw centroid loss | Valid centroid fraction | Weight-normalized expert CE |
| --- | ---: | ---: | ---: |
| 0–50k | .057221 | .847048 | 1.295569 |
| 50–100k | .027911 | .847866 | .744572 |
| 100–150k | .012074 | .861734 | .641786 |
| 150–200k | .004900 | .883765 | .592494 |

The expert column divides summed cross-entropy by target weight mass. FIRE has
four times the weight, so these values are not the same unweighted-label metric
as the previous weight1 comparisons. The lower late scalar alone does not prove
better fitting on a common distribution, larger useful gradients or better
held-out action choices. The auxiliary loss remains a changing-replay metric;
no new representation capture or actor readout is included in this experiment.

### Verification and disposition

The main service starts22:43:07 UTC October29, source `ce86d72`, and succeeds
**23:08:19**. CPU postprocessing completes **23:08:40**, terminal collection
**23:08:43**. All five services are terminal success. No full-run failure, GPU
stage repetition, cap change or host recovery occurs. The earlier bounded
preflight failure remains preserved in its separate record above.

All196 pinned inputs and all ten terminal artifact hashes pass collection and a
subsequent read-only check. The100k/6,227-update recovery passes. All372 initial
tensors match the bounded learner;357 shared tensors match the accepted control.
The47 pre-update semantic events match the bounded run. Actual-initial sampled
and greedy traces reproduce all800/795 control semantic events. All372 tensors
remain invariant in every frozen arm, while CNN, reward, centroid, actor and
value weights change during training. Full training pixel/action/label replay,
all frozen replays/videos, five host guards, curves, exposure, comparison and
FIRE-command accounting pass. The generated plots are visually inspected.

Retain the accepted H15/R16 control. No unchanged extension, coefficient sweep,
confirmation learner or automatic successor is selected. The five-game goal
remains unfinished. No GPU allocation remains active; check live state before
any later work. Keep the.612s startup-check co-hosting caveat without subtracting
it from training time; lightweight source/status/documentation work also co-hosts.

Evidence: `/x/Temp/kindle-breakout-fire-weight-20261029/full`, trial
`breakout-12m-64-b16-r16-centroid100-fire4-expert100-200k-seed1009`.

- Learning review: `ddde22f17ee96069c77d24dbad8d9de55c0d72be7841b2075a6477135fd22021`.
- Comparison: `46f64810497c768d04fa23f2b35c70b71061c22a6a6b42af32194709c0cdf411`.
- FIRE accounting: `0f01a1f115f6e29433d7dc4029c8b8120626b5bcd820bb6cce637bafdc606133`.
- Completion: `61de8c14509b9ffbde758cbc31831be746aad7d7c2edba7bce2688bb888b4b31`.
