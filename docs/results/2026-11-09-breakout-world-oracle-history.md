# Breakout: accurate recent history restores later edge predictions

The frozen action-feature model predicts the later surviving edge paths much
more accurately when supplied real recent history before each one-step query.
At arrivals16/32, LEFT and reversal reach64/64 within one RGB64 pixel under
both history lengths. Ordinary reversal gets0/64. This locates a sensitivity to
accumulated state/image-history errors; it does not isolate their components,
establish a deployable repair or prove the world model. Some collision-adjacent
one-step predictions still fail even with real recent history.

[Protocol](../experiments/2026-11-09-breakout-world-oracle-history.md),
[preceding edge-rebound/support audit](2026-11-09-breakout-world-edge-rebound.md).

## Frozen intervention

All four saved plans, histories16/32 and eight seeds983001+1009*i. Eighteen
fixed arrivals1–16,24,32 give1,152 native predict calls. Each query observes
only the immediately preceding real16/32 images and submitted actions, then
predicts one next arrival. Window starts use the original−1 reset convention.
Target and later images are excluded. Native max-pooled frames use the same
one GPU resize; all20 saved endpoint truth joins are exact. No CPU downscale.
All64 endpoint1 controls reproduce the original full RGB arrays exactly.

Actual histories change both posterior state and local decoder history. This
is assisted conditional prediction, unavailable to an original open-loop planner.
It also differs from a latent-only or image-history-only controlled intervention.
First-oracle-repeat trajectories are reused; eight lanes share one initial
visible history and can diverge under sticky controls. Draws and lanes are
correlated, not independent physical contexts or fresh confirmation.

## Ball accuracy

Within one RGB64 pixel of the first-oracle target; missing detections fail.
Entries are ordinary / assisted correct counts. Support is64 at every listed
NOOP/LEFT/reversal endpoint. RIGHT has64 at1/4 and no visible support at16/32.

|Plan/history|Arrival1|4|16|32|
|---|---:|---:|---:|---:|
|noop-h16|64/64|64/64|64/64|64/64|
|noop-h32|64/64|64/64|64/64|64/64|
|left-h16|64/64|64/64|9/64|60/64|
|left-h32|64/64|64/64|9/64|57/64|
|right-h16|64/64|64/64|unsupported|unsupported|
|right-h32|64/64|64/64|unsupported|unsupported|
|reverse-h16|64/64|64/64|0/64|0/64|
|reverse-h32|64/64|64/64|0/64|0/64|

At arrival8 every plan has zero detector-supported balls; the near-paddle
detector exclusion remains unchanged. Neither method reports a detection there.
RIGHT also has no supported ball at the later endpoints and neither method
hallucinates one. Absence is scored separately from accurate visible positions.

Assistance does not solve every local transition. At arrival9, NOOP/LEFT accuracy
is60/64 each with history16 and63/64,59/64 with history32. Reversal is0/40 at9,
8/8 at10,32/48 at11 and48/64 at12 under both histories; all supported predictions
from13 onward are correct on this slice. These collision-adjacent detector
failures include restricted-visibility effects and do not alone diagnose their
physical or rendering cause. Reversal paddle MAE at5 remains about2.30 pixels.
Every step, detection, error, posterior and event output is retained.

One-step ending probabilities range from0 to about.01482 across these means;
all corresponding true game-over labels are zero. They are conditional next-step
probabilities, not32-step cumulative risks or natural-frequency calibration.
They cannot replace any ending gate from the original assessment.

The reviewed draw0/parent0/history16 figure shows truth, ordinary rollout and
real-history one-step predictions for all four plans and five original endpoints.
No favorable draw, endpoint or checkpoint replaces a fixed score.

## Integrity and retained observer failure

Native work completes in64.214906s, including restore, native resize, predictions,
saves and input checks; this is not a kernel benchmark. Exact240 source/before/
after arrays, updates16,384, v25 metadata/action table, finite exports, all133
execution pins,64 control joins and20 truth joins pass. No update, collection,
old-model query, emulator action/frame, new build or host recovery occurs.
One host-only guard passes, with child exit0 and no unfinished children.

The original observer fails after native success because systemd clears the
invocation ID when unloading the completed transient service. Preserve its
failed service, watch_v1.py and journal traceback. A separately declared CPU
finish_v2.py verifies29 pins, the exact native invocation’s journal/controller
completion and successful guard exit, inactive/MainPID0/Resultsuccess state,
queue completion, frozen tensors and all original pins. Its guard audit verifies
nine sealed files. This completes missing evidence without relabeling the
observer as passed or repeating any native work. Plot inputs are separately
pinned and the resulting image is visually reviewed.

Root `/x/Temp/kindle-world-oracle-history-20261109`; native invocation
`c952149bfaf04a6f9cd58160ebf8a148`. Exact native1331b702/SHA15ea18b5 and matching
runtime-v2/runner stay retained. No continuation, collector or automatic
successor is selected by this protocol.

The broader action-feature assessment still fails at energy.224881 versus
required<=.127127, with296/304 nonending alarms. The goal remains active and
unproven. Full physical responses/events, brick memory, late/above-top coverage,
untouched families and fresh confirmation remain required.
