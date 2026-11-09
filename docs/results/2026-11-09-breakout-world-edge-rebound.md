# Breakout: conditional edge rebounds fail despite fitting support

The closed CPU audit corrects the earlier life-loss/respawn interpretation. In the
challenge named `life_loss`, NOOP, LEFT and reversal rebound without losing a life.
The action-feature model often loses the rendered ball on these surviving paths.
This identifies a conditional transition error, not a demonstrated respawn defect.
All original failed scores and gates remain unchanged.

[Protocol](../experiments/2026-11-09-breakout-world-edge-rebound.md),
[frozen assessment](2026-11-09-breakout-world-action-features-branches.md),
[survival decomposition](2026-11-09-breakout-world-action-features-survival.md).

## Saved physical outcomes

All eight listed parents share one native history at both lengths16/32, also
when feature commands are included. Parent reference is ball(47,153), paddle8,
five lives. The eight oracle draws and parents are not independent contexts.

|Plan|Life decrements by32, out of64|Same-life upward reversals by16|Game-overs by32|
|---|---:|---:|---:|
|noop|0|64|0|
|left|0|64|0|
|right|64|0|0|
|reverse|0|64|0|

NOOP and LEFT have exactly equal saved positions, lives, rewards, terminals,
valid flags, native endpoint RGB, first-repeat full32 native sequences and
canonical RGB64 truth. This is a finite state-conditioned equivalence, not a
universal statement about the commands. None of the eight previously collected
paired fitting prefixes matches the challenge native16 history.

## Rendered forecast failures

Fractions of64 sampled model/lane images with a detected ball. All corresponding
NOOP/LEFT/reversal oracle balls are detected at16/32; all RIGHT balls are absent.
At8, the conservative detector excludes the near-paddle native y185 state even
when its reference is finite. Detector absence there is not evidence of life loss.

|Plan|H16 at16|H16 at32|H32 at16|H32 at32|
|---|---:|---:|---:|---:|
|noop|1.000000|1.000000|1.000000|1.000000|
|left|0.140625|0.953125|0.140625|0.906250|
|right|0.000000|0.000000|0.000000|0.000000|
|reverse|0.015625|0.031250|0.125000|0.140625|

The fixed draw0/parent0 grid covers all four plans and five endpoints; visual
review confirms faint/diffuse motion and missed rebound renderings. Reversal
also loses paddle-detector support at8, although it returns at16/32. That
absence does not by itself identify a decoder, recurrent-state or blur cause.
NOOP/LEFT H16 paddle pair MAE at16 is .362/.166 RGB64 pixels despite different
ball outcomes; small paddle errors are not established as the cause.

## Actual fitting support

The new masks use descending y150–160 approaches, paddle-reference8 subset,
last-life counterpart, actual nonterminal decrements and fixed post-decrement
age bins. All use real source metadata; no such labels enter the model or loss.
Values below are repeated plan exposures / unique(source,target) arrivals.

|Condition|Source|Observed|Imagined|First imagined|First, complete16 context|
|---|---:|---:|---:|---:|---:|
|approach|1635/1635|83576/1625|79556/1615|5026/1248|4838/1204|
|left_edge_approach|270/270|12532/269|10878/270|654/183|642/179|
|last_life_approach|427/427|49770/423|28464/427|3032/359|3032/359|
|nonterminal_life_drop|297/297|17656/296|15086/297|1100/227|1100/227|
|post_drop_1_4|1188/1188|75968/1183|62422/1188|4074/879|4074/879|
|post_drop_5_8|1188/1188|86496/1180|66304/1188|4212/897|4212/897|
|post_drop_9_16|2371/2371|182408/2355|144750/2371|10640/1830|10640/1830|
|post_drop_17_32|3831/3831|244702/3785|274988/3831|19654/3234|19654/3234|

The left-edge complete-context subset contains179 unique targets,139 distinct
native16 histories and142 histories plus feature commands. Raw NOOP/RIGHT/LEFT
contribute47/10/84 unique targets;51 targets have an observed same-life upward
reversal within16. Thus the data is not simply devoid of this condition. Exact
geometry and full submitted-plan support remain unproved. Subsequent controls
vary, so incoming-command counts do not estimate causal action effects. Both
compared learners used this same plan; support alone cannot explain their gap.

For actual post-decrement states, complete-context repeated exposure counts for
age1–4/5–8/9–16/17–32 are4,074/4,212/10,640/19,654. Finite-reference exposures
are2,314/2,570/7,584/10,004; submitted FIRE exposures238/122/322/2,308.
Absent references and all raw/physical commands remain separately tabulated
in the saved per-stream and aggregate results. These are not launch annotations.

## Integrity and next question

All159 input hashes pass before/after analysis and closure. Seven fixture groups
cover life/reset/terminal boundaries, discontinuities and horizon inclusivity.
Execution takes1.081s CPU. Constructors are trapped and Vulkan disabled. Zero
model queries, updates, emulator actions or frames; no failure or repair.
`result-v1.json`, `condition-masks-v1.npz`, `first-contexts-v1.json`, the reviewed
grid and closure remain at `/x/Temp/kindle-world-edge-rebound-20261109`.

The next separately declared diagnostic will supply true recent history before
each one-step forecast on these saved paths. It can distinguish successful
conditional predictions with assisted history from persistent local failures.
It changes posterior state and image history together; it cannot isolate one
component or prove a deployable repair. No new learner or collector is selected.
Full physical/event accuracy, brick memory, late/above-top coverage, untouched
families and fresh confirmation remain required. The goal stays unproven.
