# Higher continuation weight does not demonstrate better ending recognition

On the same24 recorded game endings, the continuation10 model's observed-state
prediction averages **.730320 versus .759636** for the retained scale1 model,
against target zero. Both models produce predictions below.5 on only **5 of24**
arrivals. The primary candidate-minus-control MSE difference is−.043530, with
paired lane95% interval **[−.212505,.123448]**. This does not demonstrate an
improvement on the fixed cohort, and substantial terminal error remains.

The completed learner's negative gameplay result remains unchanged:35.250
sampled/55.833 greedy versus70.750/143.667, all0/24 two-wall, at nearly identical
21m28s training cost. **Retain scale1.** No new learner, coefficient sweep, loss,
sampler or automatic successor is selected. The original864/90%-two-wall and
five-game goal remain unfinished.

## Same arrivals, separate outcomes

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-continuation10-arrivals.md)
reuses the completed control capture and exports only the candidate's states
on the exact43,424-action/173,660-frame trajectory. Native447fe50/SHA35ad0db,
the corrective runtime, all recorded actions, pixels, rewards, frame counts,
boundaries, resets and stream hashes remain fixed. No control capture is repeated.

| Arrival group and state | Scale1 mean | Scale10 mean | Scale1 MSE | Scale10 MSE | Paired MSE difference95% interval |
|---|---:|---:|---:|---:|---|
|24 game endings, observed posterior; target0|.759636|.730320|.644558|.601029|[−.212505,.123448]|
|24 game endings, one-step prior; target0|.952850|.938323|.911006|.884355|[−.062234,.005756]|
|96 intermediate life losses, observed posterior; target.996997|.979466|.961639|.003178|.010486|[−.000639,.017799]|
|96 intermediate life losses, one-step prior; target.996997|.979477|.949845|.001530|.009264|[.002367,.013707]|

Observed-terminal error decreases in10 cases and increases in14; five ofeight
lane means improve. Prior terminal error decreases in14 and increases in10,
but neither model's24 prior predictions falls below.5. Both terminal intervals
span zero. Intermediate-life prior error increases in63 cases, decreases in25
and ties within1e-6 in8; its paired interval is above zero. Intermediate-life
posterior error also rises on average, with an interval spanning zero.

Each interval uses5,000 fixed lane resamples, seed20261104. The cohort was chosen
from retained-policy life losses, not candidate predictions. It is not a sample
of candidate-policy failures or population calibration, and it does not establish
seed reliability. These findings do not identify why gameplay regresses. They
show no demonstrated terminal-recognition benefit from this whole-loss increase,
alongside worse prior predictions at continuing life losses on this cohort.

## Readout qualification and limits

Each model uses its own learned features and its own saved continuation head.
No head is applied to another model's latent coordinates. The candidate's
offline F64 reference matches all120 native prior outputs within **1.398e-7**,
below the fixed1e-4 limit. Posterior outputs remain reference estimates rather
than direct native posterior readbacks. Independent einsum/reduction arithmetic
reproduces all240 candidate reference outputs within1e-12.

All features have2048 deterministic entries and32 one-hot categorical groups
of16. Within-model prior/posterior deterministic vectors agree exactly; their
categorical draws differ. Shared-world gradients, learned representations,
state draws and training experience change across the two models. Neither a
head-only defect nor a transition-model or optimization mechanism is isolated.
Intermediate life losses retain their nonterminal target; no reward or boundary
semantics change.

## Execution and closure

Fifteen CPU tests pass, including independent scalar head arithmetic, extreme
values, invalid features/configurations, exact source joins, paired aggregation
on known outcomes and the actual entrypoint before model/emulator construction.
Preflight verifies634 baseline and761 candidate pins, prior artifacts and the
completed candidate checkpoint. All **841 execution pins** remain unchanged.

The single **50.752s** guarded capture advances43,424 actions/173,660 frames with
zero updates and all357 finite tensors invariant. All120 source endpoints and
repeated own predictions pass; diagnostic queries preserve live features,
action probabilities and counters. Full recorded pixel/transition/trajectory
parity and host guards pass. CPU analysis takes.313s on one low-priority core.

Source`91924e08`, mainPID1107057/invocation
`02c53395d00c44eda198cc09cab7fa78` starts04:25:08 UTC November4 and succeeds
**04:26:05.198**. Independent **4.224s CPU closure** completes
**04:27:01.813**, verifies the original service, head arithmetic, paired metrics,
all tensors and52 final artifact hashes. No preparation, capture, analysis or closure
failure, repair, tolerance relaxation or completed-stage repeat occurs.

Root `/x/Temp/kindle-breakout-continuation10-arrivals-20261104` is closed. No
learner or GPU allocation remains active; check live state before work. Do not
restart this capture, analysis or closure. The original learner regression and
its control remain unchanged.

| Artifact | SHA256 |
|---|---|
|Declaration|`7dee20598d576bc430fd02c4223d011a99f1e03e6ea56d58ab4019b4ea582c6c`|
|Frozen capture|`ec00833309d15c698ca8c5987b64e9e6247a855cf650a705cd9121f2ccbd250e`|
|Paired analysis|`bf4b4ce22a0dc5fcf2f78ef02edb02ca253bfed7a6a4c363d6a93e5514995e8b`|
|Main completion|`735177e92f9973fd7daf45deb15f864193d2e94150d0d3b074a66710b8bf4c45`|
|Independent terminal summary|`03f4f1137c361c65c5ad1a497ca843e52da49550356bba38fd370198913de8eb`|
