# Raw motion supports the fixed catch signal on teacher and retained-policy trajectories

All **1,476 teacher signals and 1,132 retained-policy signals** match unique
raw downward-to-upward reversals near the paddle under the fixed matching rule.
This includes all700 teacher signals left unresolved by the earlier filtered
contact reference. Every signal arrives with zero raw game reward. None occurs
within16 raw frames of the120 selected life losses in either cohort,or during
the first eight decisions of an episode.

The stricter spatial check supports1,294/1,476 teacher and961/1,132 retained-policy
signals. A separate post-hoc review finds every spatial exclusion involves a
paddle component touching the detector crop boundary. No matching rule or
threshold changes. These are pixel-based witnesses,not exact collision ground
truth or population precision estimates.

The evidence supports selecting a
[bounded one-point catch-reward integration check](../experiments/2026-11-04-cdp-breakout-catch-reward1-check.md).
No training aid has run yet,and no full learner is selected. Retain Breakout
70.750 sampled/143.667 greedy;the original864/90%-two-wall gate and five-game
goal remain unfinished.

## Fixed comparison

The [validation protocol](../experiments/2026-11-04-cdp-breakout-catch-validation.md)
keeps the [pooled catch detector](2026-11-04-cdp-breakout-catch-signal.md)
byte-for-byte unchanged. Its16 previously passed fixtures are retained.
The new raw-motion witness uses short positive and negative y-motion edges,
a near-paddle peak and bounded temporal gaps. Offline spatial support additionally
requires an uncropped paddle component and ball x within half-width plus6 pixels.
All thresholds and mutually unique matching windows were fixed before outcomes.

The teacher arm reads its saved raw centroids and paddle annotations; it repeats
neither the old replay nor pooled detector analysis. A new instrumented CPU replay
adds previously missing raw and pooled position observations to the retained
greedy policy's43,424 actions/173,660 frames. Every existing action,reward,
frame counter,boundary,reset and pooled input/output hash matches the saved
command-takeover annotations and original frozen replay. No native model is
constructed or invoked. The first three natural episodes per lane remain the
selected24 episodes; excess episodes and partial tails remain in the capture.

| Measurement | Teacher | Retained greedy policy |
|---|---:|---:|
| Selected actions / raw frames |42,303 /169,178|36,321 /145,251|
| Selected mean raw score |225.083|143.667|
| Pooled signals |1,476|1,132|
| Raw temporal reversals |1,494|1,166|
| Unique temporal matches |1,476|1,132|
| Temporal reversals without a signal |18|34|
| Spatially supported raw reversals |1,310|993|
| Unique spatial matches |1,294|961|
| Signals excluded by spatial rule |182|171|
| Spatially supported reversals without a signal |16|32|
| Ambiguous matches, either rule |0|0|
| Mean delay from incoming endpoint, spatial matches |5.364 frames|5.649 frames|
| 95th percentile delay, spatial matches |9 frames|9 frames|
| Signals with zero raw arrival reward |1,476|1,132|
| Signals within16 frames of a life loss |0|0|
| Signals in first eight episode decisions |0|0|

The original776 fitted-reference matches split into735 spatially supported
signals and41 spatial exclusions. Of the700 originally unmatched signals,
559 receive spatial support and141 are excluded. All retain temporal support.
The spatial rule therefore also excludes some independently fitted contacts;
its exclusions cannot be read as a count of false signals.

## What the spatial exclusions mean

The separate saved-output reader completes at16:20:33.403 UTC,after the original
analysis and closure. It is explicitly post-hoc and leaves the fixed result intact.
All182 teacher and171 retained-policy exclusions have crop-edge paddle components
at both raw endpoints. All have at least one observed endpoint within the fixed
half-width-plus6 margin. No exclusion involves a missing component or an
out-of-margin endpoint pair without crop contact.

This explains why the conservative spatial witness is incomplete. It does not
turn crop-edge detections into validated full paddle geometry. Shared red-color
heuristics,partial visibility,selected/reused cohorts and raw temporal uncertainty
remain limitations. The18/34 unmatched raw reversals also remain recorded; no
detector threshold is tuned to recover them. Neither corpus measures the future
training-state distribution or proves that rewarding catches improves brick
clearance. A catch reward could favor longer rallies; the proposed experiment
must still be judged on unmodified game scores with the aid off.

## Validation and closure

All12 new CPU fixtures pass,covering raw reversal/gap/reset rules,spatial boundaries,
matching ambiguity,native-import rejection and agreement between the small pooled
extractor and the retained teacher on synthetic frames. All58 declared pins remain
unchanged. Prior detector receipts and action/pixel/raw-data hash links verify.
Independent recount checks saved witness inequalities,spatial decisions,matching
graphs,arrival alignment,exclusion windows and totals without repeating capture
or detection.

The instrumented CPU replay takes44.848s,analysis1.405s and independent recount
.481s. The full controller finishes in46.947s at16:19:04.097 UTC November4,
source`e012286e`,with14 completion artifacts. All stages exit0;no failure,repair
or repeated completed stage occurs. The separate exclusion reader takes.099s
and verifies all14 original artifacts and58 pins. No learner update,GPU work,
checkpoint change or original-service change occurs. No running Kindle service
is observed before or after the controller. Check live state before GPU work.

## Evidence

Root `/x/Temp/kindle-breakout-catch-validation-20261104` retains the protocol
declaration,private scripts/fixtures,raw observations,signals,witness edge pairs,
both matching graphs,life windows,execution receipts and independent reviews.

| Artifact | SHA256 |
|---|---|
| Declaration |`d0b75605486514421b3096636e6b19dd023e51f7e3b97d7feeb2073bb7b11947`|
| Capture |`cdfcc54dc42e4b72427535dfc80d0ab75d6d1ba7b1bc6647241cad59211c2e53`|
| Analysis |`7d7d9256e68e2d69ff610b62425c049131579f5198377c6d40b712464e784622`|
| Independent recount |`1a290f64dc7a9f348b07b823645aec423f9701a2d49d72b8f59ddd193345c2d0`|
| Original completion |`0e73a6981945ed70dda8f540174f7d896a2bb16a89a36d6718ac6d5c8245c115`|
| Post-hoc exclusion review |`731c5ea1979376f07fa0cda1d755cec47b8f381118fd1debded58450fc14e774`|
