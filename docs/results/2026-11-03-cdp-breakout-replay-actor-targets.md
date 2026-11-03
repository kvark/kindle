# Reusing replay targets would leave substantial model dependence in the actor

The current actor has no direct return-weighted update on executed replay
actions. It learns from imagined returns and teacher labels; replay returns
train the critic and, when enabled, the representation. This is a concrete
extension to investigate within the five-game goal, but simply copying the
critic's existing targets would retain imagined-return bootstraps.

The [fixed CPU audit](../experiments/2026-11-03-cdp-breakout-replay-actor-targets.md)
finds **241 positive advantages among480 saved states**. Removing only the
external rewards inside their short replay windows leaves **200 positive
states and83.32% of the proposed actor weight**. Imagined bootstraps, intrinsic
bonuses and value estimates are held fixed. Thus this particular construction
would not provide a clean test of learning from completed real outcomes.
It is not rejected as an algorithm; its gameplay benefit remains unmeasured.

| Target statistic | Existing replay target | Window external rewards removed |
| --- | ---: | ---: |
| Positive rows, of480 |241|200|
| Total normalized positive weight |7.176962|5.979909|
| Mean weight across all rows |.014952|.012458|
| Effective weighted rows |42.967|33.145|
| Weight on actions agreeing with teacher |64.75%|67.44%|
| Saturated weights |0|0|

Observed external rewards contribute to90 rows' targets; their removal changes
the advantage sign in41 originally positive rows. Overall executed-action
agreement with the teacher is53.125%; among positive rows it is54.36%.
Weighted agreement is higher because the weights are concentrated. Teacher
agreement neither identifies the best action nor proves useful gradients.

All eight lanes and both preselected batches are retained. Each lane has60
rows; their positive counts before/after removal are17/17,21/19,13/11,33/19,
48/37,44/39,26/26,39/32. Batch0 has111/104 positive rows and batch1 has130/96.
These are correlated states from one reused checkpoint/cohort, not480
independent trials or original training batches.

## Target construction and interpretation

The hypothetical weight is `min(1, max(0, (G_replay - V) / return_scale))`,
applied to the actually executed full-action-space action. No training
coefficient is chosen. The scales35.965408/35.686726 match the same sequential
normalization used for the saved imagined targets. All proposed weights are
below the cap, so clipping does not affect these results.

The audit decodes saved native two-hot targets and recovers posterior values
from the saved first-step weighted advantage identity. This requires no model
execution. Reconstructing the two normalizer updates agrees with the saved
final state to2.339e-7. The recovered values remain approximate because saved
targets are rounded. At the fixed ±.001 raw-advantage sensitivity range,
positive counts are241–243 and200–202; weight masses are7.170232–7.183718 and
5.974327–5.985517. Two rows are sign-ambiguous; the broad retention result stays.
That sensitivity range is not a proved numerical error bound.

For each nonterminal window, the subtracted external contribution obeys
`E_t = r[t+1] + gamma*lambda*E[t+1]`, with zero at its endpoint. Gamma is
.9969969987869263 and lambda is.949999988079071. The remaining target mixes
imagined continuation and intrinsic rewards; the audit does not separate them.
Nor does it remove reward information previously learned by the world model
or critic. The83.32% figure is a local arithmetic counterfactual, not a causal
fraction of learning attributable to imagination, or a measurement of all
external rewards in a completed episode.

Positive-advantage imitation has precedent in
[Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf),
which uses completed episodic returns, and return-weighted policy regression
in [AWR](https://xbpeng.github.io/projects/AWR/AWR_2019.pdf).
This proposed bootstrapped, clipped linear weighting is neither method exactly.
Their results do not establish that adding it to Kindle will improve Breakout.

The next design question is how to supply completed real-outcome targets with
consistent reward, intrinsic-bonus, boundary and replay accounting. That should
be resolved before selecting a native loss and its bounded gradient/cost check.
The source review and this audit do not select a coefficient, new learner or
automatic GPU allocation. The existing200k70.750/143.667 and400k92.792/173.708
sampled/greedy results remain, all zero of24 two-wall successes. The five-game
goal remains unfinished.

## Qualification and closure

Analysis completes **15:42:20.331 UTC November3**, source934053ae, in**.436s**
CPU (.514s wrapper). All17 fixtures pass before selection. All20 declared
inputs and158 original backward-stage pins remain unchanged. All512 incoming
actions/rewards match the frozen source log; all480 outgoing teacher labels
match native targets, with no terminal crossing in the fixed windows.

One independent terminal recount completes **15:43:34.993** in**.202s**
(.264s wrapper), reproducing the saved-row counts and weights, checking all
7,680 imagined action-target rows and positive trajectory weights, and finding
no running Kindle service. No failure, native import, model forward pass,
emulator action, GPU allocation or learner update occurs. No previous capture,
backward pass, analysis or reader is repeated. The original native447fe50
runtime, runner, teacher, checkpoint and gameplay results remain unchanged.

Root `/x/Temp/kindle-breakout-replay-actor-targets-20261103` is closed. It retains
the exact historical source excerpts, private analysis/tests, declaration,
all480 row arrays, qualification/execution receipts and terminal recount.
Do not repeat completed stages. Check live state before separately selected work.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`08353eb35e8eb7bc3376e25fe90b3ababbea0c69a6b768f3f64bb74db866ef8b`|
| Analysis |`2691948e60a755895933706f04fc359bd1fbe58759c920be9df6759e93be3f85`|
| Row arrays |`79ea3c9a83b323ea26564bf4079133c67c190ed625034c0834cd3eb663229e63`|
| Completion |`a70c76a2adfd6a2d4646903157d8ac43a61b94ab4630d312ee11b2a0525f5054`|
