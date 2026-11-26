# Retained teacher shots do not pass the outer-region design screen

The fixed offline screen fails. Shots projected toward the outer quarters have
higher aggregate post-catch reward per frame, but only **three of eight lanes**
favor them. The gain below score 216 is **17.349%**, short of the fixed 20%
requirement; the 216–431 band gains **27.101%**. All minimum-count requirements
pass. Retain the current teacher and native learning recipe. No controller,
collector, learner or automatic successor is selected.

This is a new outcome join on existing trajectories, not a repeat of the older
contact-direction analysis or the failed three-pixel aiming controller. The
[fixed protocol](../experiments/2026-11-06-cdp-breakout-shot-outcomes.md) joins
776 previously fitted contacts to their unique saved catch signals. A window
starts at that signal and ends at the next catch, life loss, episode end or
256 raw frames. Only complete action intervals contribute reward and frames.
The entry region projects the fitted outgoing line to y90 with side-wall
reflection; it is not an observed brick impact or a brick occupancy map.

| Score before window | Outer contacts | Middle contacts | Outer reward/frame | Middle reward/frame | Ratio | Fixed band screen |
|---|---:|---:|---:|---:|---:|---|
| Below 216 |368|298|.033493|.028541|1.173490|Fail|
| 216–431 |62|48|.067174|.052851|1.271007|Pass|
| 432 or more |0|0|—|—|—|No coverage|

The screen requires at least 30 nonempty windows per region in each of the first
two bands, at least 1.2 times the middle reward/frame in both bands and five
eligible lanes with a strictly higher outer rate. Every lane has at least ten
windows per region; only lanes 3, 4 and 5 favor the outer region. Criteria remain
unchanged. A positive pooled association alone does not select an aiming policy.

## Other declared descriptive measurements

| Absolute horizontal / upward slope | Contacts | Reward/frame | Mean window frames |
|---|---:|---:|---:|
| Below .4 |110|.026435|100.073|
| .4 to below .8 |407|.057509|81.346|
| .8 or above |259|.015190|119.208|

The middle slope band has the highest observed reward rate. This is not an
alternative passing screen or evidence for selecting an angle-control rule.
Ball speed, remaining brick configuration, contact location, score and earlier
play differ between groups. Changing contact offset also changes catch risk.
The previous [three-pixel aiming test](2026-10-26-cdp-breakout-pixel-reference.md)
remains negative; neither its policy nor collector was repeated.

All 776 windows are nonempty, covering **18,749 distinct recorded actions /
74,991 emulator frames and 2,664 reward**. No action is counted twice. End reasons
are 701 next catches, 65 life losses, ten frame caps and six episode ends; six
life-loss/episode-end ties explain why these counts sum above 776. Width data
retains all 674 previously accepted annotations (593 width16, 81 width12) and
102 unavailable/unstable annotations. All 1,476 saved catch signals bound windows,
including signals without an accepted raw contact fit.

The source's 24 selected natural episodes total 5,402 reward, mean225.083 and
maximum426, with zero two-wall successes. The analysis covers a selected subset
of successful catches, not all episode reward or all trajectories. Its outcome
windows are neither complete returns nor action values. Score bands are not
wall-completion events. Fitting error, detection error and unobserved brick
geometry prevent causal interpretation; the data do not reject every deliberate
shot-placement policy or establish a benefit for native learning.

## Validation and closure

Four synthetic fixture groups pass. All 30 artifacts sealed by three historical
completion receipts verify, as do source hashes and the complete 52,072-action /
208,254-frame ledger's episode totals. Every contact/signal join is unique and
consistent with its saved match. The independent reader checks all 776 reward
windows and all 33 output groups directly against the ledger, including the
fixed screen and duplicate-action exclusion.

Analysis takes .294689s and independent recount .255133s. The complete controller
finishes in **.769346s at 16:39:37.438 UTC November 6**, source
`687788a007561fa2367df7a1ef38ec2f41af82d7`. All 46 pins remain unchanged, both
subprocesses exit0 and before/after service checks find no active Kindle work.
No failed/repeated stage, emulator/native import, new Atari action, learner
update or GPU work occurs.

Evidence root: `/x/Temp/kindle-breakout-shot-outcomes-20261106`. The declaration,
original input receipts, exact scripts, per-contact `outcomes.json`, grouped
`review.json`, independent recount and completion manifest remain retained.
Do not rerun completed analysis or older contact collectors. Retain original
guidance and Breakout143.667 greedy/70.750 sampled; original864/90%-two-wall and
the five-game goal remain unfinished.
