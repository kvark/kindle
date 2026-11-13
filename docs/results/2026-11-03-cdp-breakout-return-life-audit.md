# Later lives supply much of the fixed completed-return actor weight

Stopping external returns at the next lost life reduces positive actor targets
from **390 to143 of480**, removing **70.74% of their total clipped weight**.
This identifies substantial dependence on later-life rewards in the proposed
completed-return signal. It does not assign causal action credit, establish a
learning defect or show that a life-bounded loss improves gameplay.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-return-life-audit.md)
reuses the exact480 outgoing states, approximate combined values, normalizer
scales and actions from the [completed target audit](2026-11-03-cdp-completed-return-bonus-audit.md).
They are two fixed B16/T16 batches on the retained200k replay-value1 sampled
trajectory. They are not original online batches or states from the newly
completed [conservative-actor learner](2026-11-03-cdp-conservative-actor.md).

| Fixed subset | Rows | Positive targets, game-end → life-end | Clipped weight, game-end → life-end | Removed weight |
| --- | ---: | ---: | ---: | ---: |
| All selected states |480|390 →143|94.779770 →27.734618|70.74%|
| Within16 outgoing actions of a loss |60|30 →0|3.310854 →0|100%|
| Other selected states |420|360 →143|91.468916 →27.734618|69.68%|

The near-loss subset contributes only **3.49%** of the original total positive
weight. Its30 positive commands agree with the pixel teacher in28 cases;
agreement does not make an action optimal or show that the loss was still
avoidable. The analysis does not identify the earlier controllable contact or
the cause of any loss. Most removed weight lies outside this16-action subset.

Later-life rewards contribute46.99% of the sum of480 discounted external
returns. The mean of the480 individual later-life fractions is49.62%; all480
full returns are positive, and422 have some later-life reward. These are sums
over correlated, overlapping future windows, not unique reward fractions or
the proportion of online gradients attributable to later lives.

| Lane,60 states each | Positive targets, game-end → life-end | Removed weight |
| --- | ---: | ---: |
|0|60 →15|54.46%|
|1|60 →18|91.04%|
|2|50 →1|99.30%|
|3|49 →15|22.61%|
|4|30 →0|100%|
|5|60 →31|71.10%|
|6|36 →33|64.72%|
|7|45 →30|79.84%|

Positive targets remain in seven of eight lanes, but only one remains in lane2
and none in lane4. Unweighted teacher agreement among positive targets changes
52.82%→48.25%. Neither measure establishes better supervision. The fixed±.001
advantage perturbations leave all390/143/247 positive/remaining/lost counts
unchanged and removed-weight fractions between70.73697% and70.73869%. This is
a sensitivity check, not a proven bound on the recovered value error.

## Interpretation and design limit

The full-game return is a valid target for total game reward: losing a life
does not end the actual episode. Cutting at life loss changes that target.
Keeping the existing game-level, intrinsic-inclusive value as the threshold
then supplies a more conservative positive weight, rather than an estimate of
per-life advantage. A future actor objective would need to state that choice
explicitly and keep its auxiliary boundary separate from model continuation,
game termination, replay resets and frozen score accounting.

All rewards here are nonnegative. For the fixed value and positive scale,
`clip((G_life−V)/scale,0,1) ≤ clip((G_game−V)/scale,0,1)` holds. That arithmetic
does not show which actions caused later rewards or whether useful long-term
credit would be removed. The retained policy still scores70.750 sampled /
143.667 greedy withzero two-wall successes; no new gameplay score is produced.
The previous one-point life-loss penalty remains a separate failed learner.

This result supports considering explicit per-life credit design, without
selecting an implementation, coefficient, learner, extension or automatic
successor. Native learning remains unchanged. The original864/90%-two-wall
gate and the five-game goal remain unfinished.

## Replay, arithmetic and closure

One new CPU annotation replay covers35,552 recorded actions /142,169 raw frames,
26 completed natural episodes and partial tails. All rewards, frames, actions,
boundaries, resets, completed task outcomes and partial ledgers reproduce.
All eight complete input-pixel sequence hashes and512 selected input hashes
match the prior native capture. Raw-frame life counts record142 losses andzero
gains; no action contains both a loss and gain. Life observations never affect
actions. The old model capture, task replay and target audit are reused without
rerunning their stages.

The cut includes the entire action containing the first raw-frame decrement;
bonus-life gains alone do not cut it. Natural terminal rewards remain included,
while timeout and unfinished game segments stay ineligible. The original F32
discount is.9969969987869263. F64 backward/forward sums and later-life
decompositions agree within3.553e−14 before F32 rounding. All480 original rounded
full returns, outgoing actions and original positive weights reproduce exactly.

Sixteen CPU fixtures pass in.120s (.214s wrapper). Preparation resolves all
17/20/158 prior declaration entries, including one changed repository file
recovered from its exact Git blob; historical manifests remain unchanged.
All252 selected input pins pass throughout. Replay takes15.018s and completes
**18:52:31.984 UTC November3**; analysis takes.639s and completes18:52:32.739.
Independent saved-array weight/life-balance recount takes.273s and closes
**18:52:33.129**, source`0dc11ba4`. Wrappers take15.181/.764/.364s respectively.
There is no failed stage, retry, GPU context, model forward or learner update.

Root `/x/Temp/kindle-breakout-return-life-audit-20261103` is closed. It retains
every raw-frame life pair, selected row, arithmetic result, source resolution,
qualification and execution receipt. Do not repeat its completed stages. No
Kindle allocation is active at closure; check live state before future work.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`35b58d640c3bed9d0388d98fbbe61bb5772657ae3f1b1ad46f5f1d737bb876d4`|
| Annotation replay |`5e20140e9298b85c1570ea03561c03ac0d1cb592e222cae5bd1e7acfbebe5124`|
| Raw life ledger |`7409e5a647ff04983e6659e572819d1a3c7160476bbfedc7fe6de77ef645950d`|
| Analysis |`c5d5a6470f310fc46eb2d562e8af48eeb2a99e6b60c3e24ca0bdf6882b595300`|
| Selected arrays |`6b0c59a5ed83de0d96986d5a86ac60d006d980387a5f435c18ff70dea3a0ee8d`|
| Completion/recount |`231a554d9ba99bc0db9ed8e41ba38fa7e41cb3e363eccdc06935750cc5450c04`|
