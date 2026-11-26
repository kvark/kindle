# Removing the residual correction raises point scores, with uncertain paired differences

The [frozen serving ablation](../experiments/2026-11-06-cdp-residual-actor-serving-ablation.md)
is complete. Zeroing the final residual correction raises greedy117.625→131.750
and sampled49.333→60.000, but both paired intervals span zero. Both means remain
below retained143.667/70.750 and all final cohorts have0/24 two-wall successes.
This does not establish a serving-time harm or benefit from the correction on
this reused cohort, or explain the full learner's regression. Keep residual actor
off for retained learning; no new learner or automatic successor is selected.

## Fixed checkpoint intervention

The original source is the completed residual learner's final200k checkpoint.
A separately identified copy sets only `behavior.actor.residual.weight` and
`.bias` to zero:46,080 weight values and18 bias values were originally nonzero.
All361 other tensors, including every optimizer moment, remain exactly equal.
Only `tensor_sha256.behavior` changes in metadata. Configuration still has
`actor_residual=true`; architecture, counters, world/base weights and training
history remain fixed. Source files stay byte-identical. The diagnostic copy is
never trained or resumed.

The two new frozen arms use the exact package7e0bb827/native3003151f and runner
from the completed residual comparison, with the same seed4000001009, full18
actions, sticky.25, repeat4, eight lanes and first3 natural episodes per lane.
Guidance, labels and learning are off. Commands differ from the source arms
only in restore/output/export paths. Completed combined-policy and actual-initial
assessments are reused; none is repeated. This is not a separately trained
base-only control.

| Measurement | Zero correction | Original combined policy | Retained control |
| --- | ---: | ---: | ---: |
| Greedy mean,24 natural episodes |131.750|117.625|143.667|
| Sampled mean,24 natural episodes |60.000|49.333|70.750|
| Two-wall successes, either mode |0/24|0/24|0/24|

The primary greedy difference against the combined policy is **+14.125**,
with paired eight-lane95% interval **[−27.252083,56.626042]**; six lanes improve.
Sampled difference **+10.666667** has interval **[−12.291667,43.083333]**; four
lanes improve. The10,000-resample intervals use seed0 and condition on reused
evaluation lanes, not independent training seeds. Descriptive differences from
retained are−11.916667 greedy [−72.710417,47.001042] and−10.750 sampled
[−32.125000,21.625000]. No interval establishes a new retained policy.

Removing the correction changes the subsequent trajectory and posterior draws
through its actions. The result therefore measures the full serving intervention,
not a per-state action-value advantage. The original training trajectory also
differs from retained. These results do not imply that training with policy loss
zero would reproduce the ablated scores, or establish a gradient-interference
cause. The original864/90%-two-wall and five-game goals remain unfinished.

## Execution and verification

CPU preparation passes in.915s wrapper/.845s internal, with351 unchanged source
pins, exact intervention/schema/hash checks and both path-only command changes.
There is no native import, model construction, emulator or update in preparation.
The selected declaration pins369 inputs. Existing package/CLI qualification is
reused; no rebuild or repeated native qualification occurs. Original qualification
fixture failures remain linked in the [source result](2026-11-06-cdp-residual-actor.md).

| New frozen arm | Aggregate actions | Emulator frames | Evaluation time | Construction | CPU audit/replay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sampled |33,944|135,735|21.681s|2.930s|11.857s|
| Greedy |40,104|160,383|25.834s|2.831s|14.197s|

Each arm has24 natural episodes, no truncations or learning updates, and363
invariant tensors. All action/outcome/frame/boundary replays, videos, host guards,
source-checkpoint hashes and369 pins pass. Frozen logs lack live image hashes;
replay does not establish unrecorded image equality. Aggregate new evaluation
work is74,048 actions/296,118 frames; no training work occurs.

Original controllerPID1451929/wrapperPID1451928 completes **10:31:27.438 UTC
November6**, source80f4559c, in86.268s wrapper/86.166s controller. Its terminal
exit is0 and source bytes remain unchanged. The independent read-only closure
finishes **10:33:13.169 UTC** in.519s: it recounts every selected episode and lane
score from saved replays, reproduces all four paired intervals, verifies70 sealed
files and369 pins, and confirms both original processes are terminal with no
Kindle service/render-device holder. No experiment stage fails or repeats; no
reset or successor allocation occurs.

Root `/x/Temp/kindle-residual-serving-ablation-20261106` retains the immutable
protocol/declaration, exact sources, derived checkpoint, queues, guard records,
replays/videos, original execution logs, comparison and closure receipts.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`67731b355328036ffcc6beab58d10bf935742cc125b353d414ee0bbafd725c6e`|
| Comparison |`262aebb4bd220c0d25de90430465c9abae812edaf0d9277c3e88aa719bb1aa82`|
| Completion |`4e6dbf2b6252bed201c3cc91832bc208446258346b0d5568abd1fbad2b3da872`|
| Original execution |`9ef42dc58b1fd106d77ea29d0ff654b5677d5442759f68eb8c1998dfacfe40d6`|
