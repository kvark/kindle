# A causal pooled-pixel catch signal covers the filtered contact reference

The fixed detector matches all776 previously fitted contact candidates in the
selected24 teacher episodes, including all674 contacts accepted by the later
error-fit audit. It emits1,476 signals in total; the other700 have no candidate
in the incomplete reference. This establishes coverage of that reference,
not detector precision or a validated training reward. Every signal arrives
on a transition with zero raw game reward.

The [protocol](../experiments/2026-11-04-cdp-breakout-catch-signal.md) was fixed
before applying the detector. This is a CPU-only diagnostic within the existing
five-game goal. No reward coefficient, learner, new trajectory, native change
or GPU allocation is selected. Retain Breakout70.750 sampled/143.667 greedy;
the original864/90%-two-wall gate remains unfinished.

## Motivation and measurement

The [retained rescue forecasts](2026-11-04-cdp-breakout-rescue-forecasts.md)
have zero external reward in all240 nine-action branches, despite92 teacher
rescues. A positive paddle-catch event could supply a nearer outcome signal.
That hypothesis differs from the already rejected
[life-loss penalty](2026-11-03-cdp-breakout-life-penalty.md), and neither result
establishes that a catch reward improves learning.

Reuse the saved52,072-action/208,254-frame teacher trajectory, with its existing
pixel/action/debug replay receipt. Select the first three completed episodes
in each of eight lanes:42,303 actions/169,178 frames,mean score225.083 and
zero two-wall successes. This is older teacher-policy data, not the retained
student's trajectory distribution. The
[contact-error audit](2026-11-01-cdp-breakout-teacher-contact-errors.md) supplies
the existing filtered raw-frame fit candidates; no previous replay or analysis
is repeated.

The detector uses only causal pooled ball visibility, paddle visibility and
ball y. It arms on a downward movement of at least1 pixel at y≥170 and emits
once on upward movement of at least1 pixel at y≥150. Episode resets, missing
paddle, invalid coordinates and gaps longer than two decisions clear its state.
One missing observation may bridge a contact. A new descent is required for a
second signal. No teacher target, action, life count, future frame or emulator
state enters detection.

A signal observed before actionj is credited to the arrival of actionj−1.
Offline matching accepts only mutually unique neighbors within observed time
minus fitted time [−2,+12] raw frames. Thresholds and window are unchanged after
outcomes; ambiguous and unmatched rows remain in the evidence.

| Quantity | Result |
|---|---:|
| Selected episodes |24|
| Emitted signals |1,476|
| Fitted reference contacts matched |776/776|
| Accepted-error-fit subset matched |674/674|
| Signals without a reference neighbor |700|
| Ambiguous signals / contacts |0/0|
| Mean / median detection delay |5.550 /5.385 raw frames|
| Minimum /95th percentile /maximum delay |2.070 /9.063 /11.305 raw frames|
| Signals with zero raw arrival reward |1,476/1,476|

All eight lanes cover their references; per-lane reference counts are
106/60/78/106/124/123/70/109. Signals range37–93 per selected episode.
All source actions, including unselected episodes and partial tails, are read
once; they contain1,805 total signals. Only the declared24 episodes enter the
table and matching.

## Validation and limits

All16 CPU fixtures pass, covering causal prefixes, upward serves, reset,
duplicate prevention, missing observations, invalid coordinates, arrival reward
alignment and ambiguous matching. All23 declared input/source pins remain
unchanged. Prior action/replay/raw-centroid/contact hash links and eight saved
completion artifacts verify. Independent output recount confirms all776 matches,
matching-window edges, alignment and totals without repeating the detector.

Analysis takes0.435s and independent recount0.143s; the complete controller
takes0.615s, finishing16:07:08.888 UTC November4,source`ed155372`. Both subprocesses
exit0;18 completion artifacts are retained. No failure, repair or repeated
stage occurs. No ALE or native library is imported; zero learner updates and
no GPU work occur. No running Kindle service is observed before or afterward.
The test receipt records the reviewed successful unittest invocation; its
reported test-body time rounds to0.000s and is not a full-process timing.

The reference rejects many tracks for geometry or fitting reasons and is not
complete ground truth. The700 unmatched signals are unresolved, not700 proven
false positives. Conversely, full coverage of a selected contact subset does
not establish sensitivity over all contacts. Shared pixel detections, reused
teacher trajectories and temporal fitting uncertainty further limit inference.

This result supports investigating a catch signal, not adding it to training
yet. Outstanding questions are whether unmatched detections correspond to real
catches and whether the signal remains reliable on current-policy trajectories,
including misses and serves. Even a reliable event could reward long rallies
without improving brick clearance; task benefit requires a separately declared
learner comparison. No coefficient sweep or automatic successor is selected.

## Retained evidence

Root `/x/Temp/kindle-breakout-catch-signal-20261104` contains the immutable
declaration, private scripts/fixtures, source receipts, all signal/reference
rows and edges, analysis, independent recount, execution logs and closure.

| Artifact | SHA256 |
|---|---|
| Declaration |`1d50a1fef225b753983e3f690ee55996226a1a3f048a0fece3b6062be9970f16`|
| Review |`f7916f55f4047f0410d1894dae60434a91c0b67aa7f24b15979fade67ca5407a`|
| Events |`30cbaa097da1c541ee8c122da9caa464f1200c8b2953a3c4450648c808beb299`|
| Independent recount |`4c7015b8f37729e88da269cd85673c21f667d827282953663f14c891939debe1`|
| Completion |`af01ab73421bcc025eec5476f5375d251f5409b957a85a13484dfa7748e4e00c`|
