# Frozen paired-history diagnosis

The [data-scale result](../results/2026-11-09-breakout-denoising-sigma-data0p05.md)
learns one-step motion but loses balls during longer forecasts. Its actual-history
queries reset the inference seed at every arrival; ordinary rollouts advance it.
Select a new frozen diagnostic that removes this noise difference before choosing
another learner. This is neither a repeated assessment nor a training extension.
Root `/mnt/data/kindle-breakout-denoising-paired-history-20261109`.

Reuse only the current sigma_data.05 final checkpoint/native/runner, unchanged.
At source arrivals16..31, compare cached actual-history one-step predictions to
a new recursively generated-history branch. Call the same production one-step
API with the same draw seed at every arrival on the generated branch. Each
call starts the same native Gaussian stream of8×3×64×64 values; the cached actual
branch uses that same stream. Incoming actions, weights, coefficients, sampler
and batch geometry match. Only the past images differ after the first prediction.
The four seeds stay83001/83003/83009/83023. Resetting noise each step correlates
noise over time; this is a controlled diagnostic, not ordinary production rollout.

The generation function receives only the observed16-image prefix and submitted
future actions. Append its own predictions; no target image, object coordinate,
reward, ending or emulator state enters generation. The API's separately seeded
observed-denoising output never replaces an observed/predicted input. Require all
20 first-step output arrays bit-exact to the saved original first step; also
require each first prediction exact to the cached actual-history target16.
Then score all110 supported endpoints per draw, with missing predictions counted
wrong and18 unsupported endpoints explicit. Retain paired correctness transitions,
per-horizon ball/paddle/brick metrics, prediction/event arrays and source indices.
No threshold changes or fitting pass claim; original gates remain unfinished.

Pin script, checkpoint/config, cached actual/ordinary arrays, native, matching
runner, source archive, data, and completed previous closure/publication before
GPU allocation. No build or native math changes; reuse existing qualification.
Use one existing guarded queue with300s cap, AMD Auto, no driver reset/NVML.
Require terminal successful service, sealed guard and135 exact frozen tensors.
Preserve any original failure and use a separately declared repair if necessary.
No new training, emulator actions/frames, collection, extension or automatic
successor. Full physics/events/natural calibration, changed-brick memory,
late/tunnel/second-wall/above-top, untouched families and fresh confirmation still
remain required to prove the goal.

Completed: the [result](../results/2026-11-09-breakout-denoising-paired-history.md)
confirms real-history77/86/77/77 versus recursive40/41/35/33 of110 under matched
noise. All first-step/frozen/guard checks pass, service terminal success. No
learner or extension selected by this diagnostic; full goal remains unproven.
