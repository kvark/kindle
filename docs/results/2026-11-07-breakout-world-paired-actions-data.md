# Paired fitting action branches: collection complete; learning not assessed

The [paired-data protocol](../experiments/2026-11-07-breakout-world-paired-actions.md)
completed its first actual collection. **This is a dataset result, not a learned
improvement.** It adds 448 branches from 56 fitting-only parent histories: seven
categories, eight distinct source streams per category, four action plans and
two sticky RNG repeats. No development or oracle source selected the parents.
These are retained fitting families, not new independent emulator seeds; overlap
of identical histories across splits was not separately excluded.

The [action-gap audit](2026-11-07-breakout-world-action-gap.md) motivates this
intervention: on eight verified freeflight oracle histories, paddle commands do
not change the ball path through eight decisions, but the newer world model
renders a ball in 64/64 NOOP versus 41/64 RIGHT images. Existing fixed-command
support is nonzero but uneven and unpaired. New matched data can test whether
training support helps; it does not establish the cause or promise a remedy.

## Actual experience and cost

|Work|Aggregate actions|Actual emulator frames|Resets|
|---|---:|---:|---:|
|New stored futures|13,068|52,122|0|
|Initial source reconstruction|55,920|223,581|152|
|Sticky-state restore replay|447,360|1,788,648|1,216|
|Total executed|516,348|2,064,351|1,368|

Collection wall time is 228.179s (3m48s), excluding the failed launcher and CPU
preparation/review. Additional source/replay work is not independent experience.
The 32 shared prefix arrivals are duplicated across each parent's eight branches;
14,336 stored prefix arrivals must not be added to new action/frame counts.
Every branch stores its actual 1–32 future arrivals. Natural terminal arrival is
retained; no fake future, reset, padding or post-terminal image is stored.

|Category|New actions|Game-over branches|Short branches|Future length|
|---|---:|---:|---:|---:|
|serve|1,796|12|12|4–32|
|freeflight|2,048|0|0|32–32|
|contact|1,904|8|8|14–32|
|brick|1,936|16|16|24–32|
|wall|2,048|0|0|32–32|
|life_loss|2,048|0|0|32–32|
|ending|1,288|48|48|15–32|

## Native camera and source reconstruction

Reuse motion-rollout native source `85dbe150` and SHA256
`5b1128667e6436b9865c31dedcb50e22318a43250e884046b155e51dc955ad82`,
with its matching immutable runtime. The new Python collection driver does not
change or rebuild that native. Restore the coverage control's actual initial H0
world only as a native camera. Its one GPU resize matches all 1,792 saved prefix
RGB64 images exactly; all 204 weight/moment tensors are unchanged afterward.
Zero world predictions or learning updates occurred. RAM/phase labels annotate
and select data; they are never model inputs or loss targets.

Eight independent environments reconstruct source seeds/submitted transcripts,
including real episode resets. Every source arrival checks native image hash,
reward, terminal, position/lives and exact executed-frame counter. The selected
32-frame histories match before parent snapshots are captured. Qualified
`restore_branch` replays source commands to reconstruct ALE's omitted private
sticky-action state before installing the desired snapshot RNG; blind restore
is not substituted. Repeat0 shares the parent RNG across plans. Repeat1 uses
evolving RNG per plan; the common repeat index does not imply common random
numbers or independent samples. Parent snapshots are saved; per-branch desired
RNG snapshots were not separately serialized.

Nine CPU collector checks pass for plan meanings, reverse timing, short terminal
sequences, actual first-prefix actions, invalid reset/frame/episode handling and
native resize quantization. The repaired declaration pins 167 inputs. The guard
passes with no unfinished children or host recovery; telemetry is unmeasured,
not a broad hardware qualification. The collector hashes 677 output files.

## Preserved preparation and launcher failures

The parent selector writes all 112 prefix/transcript files, then strict JSON
serialization fails on absent-coordinate NaNs. The separate CPU-only repair
writes nulls and verifies those files unchanged, with no reselection or emulator
work. Preserve the original failed script/log and repair receipt.

The first launcher incorrectly resolves the virtual-environment Python symlink
and invokes base Python without NumPy. It fails before collector import; no
emulator or native world is constructed, and the collection output does not
exist. Preserve that failed guard. A separate CPU-only import check repairs the
interpreter invocation and verifies NumPy, native binding and collector imports
without constructing either system. A separately pinned queue runs only the
still-unstarted first collection. No completed collector or learner is repeated.

Root `/mnt/data/kindle-breakout-world-paired-actions-20261107`. The original
`collection-guards` remains failed; `unstarted-collection-guards` owns successful
execution. Parent preparation, source/native pins, raw branch files, snapshots,
logs and checks stay outside Git. No old failure is relabeled.

## Next learning check

The independent CPU dataset review passes in 96.815s: all 14,336 shared prefix
arrivals, future command/hash/frame/episode/terminal alignment and aggregate
counts match. All 167 input and 677 output pins pass. Of 448 branches, 285 have
distinct native-future/command byte strings; this is not an independence count.
All eight new freeflight parents have identical finite reference ball paths
across the four repeat0 plans through 16 decisions; three do through 32. This is
a RAM-reference check, not a substitute for full RGB/event qualification.
`collection-review.json` preserves every category and supported-prefix count.

Next separately declare the fixed
mixed sampler and one native full-shape preflight, then one fresh 8,192-update
H16 motion-cross-entropy learner. Keep the existing rates, native, objective,
initial seed and evaluation origins; the paired data and within-batch sampling
change together. Repeat-index pairing must not be described as common RNG for
repeat1. Initial/frozen tensor checks, restore and new-checkpoint-only branch
assessment remain separate gates. No unchanged extension, old collector rerun,
scalar/seed sweep or automatic successor. No learner is active yet.

All original quality gates remain, plus at least 90% rendered-ball presence in
each retained freeflight action plan at eight decisions. Presence alone cannot
qualify position or calibration. Precise all-horizon forecasts, event timing,
brick memory, late/tunnel/second-wall coverage, untouched families and fresh
confirmation still stand between this dataset and the full world-model goal.
Actor/critic and CDP comparisons remain parked.
