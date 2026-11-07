# Bounded Breakout check with a shorter return discount horizon

Select one fresh check changing native `horizon`333→100. This sets the F32
continuation discount from approximately.996997 to.99; imagination length
stays15. It changes continuation training targets, imagined/replay return
weighting and targets, and associated learned behavior together. It is not
an isolated critic intervention or a shorter imagination rollout.

The [retained critic diagnostic](../results/2026-11-02-cdp-breakout-replay-value-critic.md)
finds sampled complete-return RMSE9.231 versus7.855 for an earlier-episode
constant. Existing short action rankings do not establish a useful selector,
while the [rescue forecasts](../results/2026-11-04-cdp-breakout-rescue-forecasts.md)
have accurate near-zero immediate rewards on all240 zero-reward branches.
This motivates testing more local credit, not increasing immediate reward-loss
weight. Shorter discount may ease value fitting but also discard useful distant
rewards. No causal mechanism or gameplay improvement is established in advance.
The completed lambda1,imagined-value.3,H30/R12 and stronger-prior comparisons
remain negative; none tested this discount change. Do not repeat them.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,dynamics4e-4,behavior8e-5,lambda.95,policy/value/replay-value1,
continuation1,dynamics-KL1,posterior-KL.1,free-nat floors1,CDPfuture500,
disagreement1,expert100/unit weights/world gradients,all-state corrective labels,
20% episode guidance/cap1920,sampled collection/posteriors,actor-unimix.01,
replay100k,full18/sticky.25/repeat4/noops0. Native pooled RGB receives one GPU64
resize. Fresh seed1009/frozen4000001009; no weights or experience transfer.

Use native447fe50/SHA35ad0db and the exact corrective runtime. Add only
`--discount-horizon` to retained runnercf4ea238 and the corresponding canonical
CLI. Omission keeps333; explicit333 must match defaults;100 changes only
`config.horizon`. Reject nonintegers,values below2 or above33554430, and
restore/evaluation overrides before native construction. The upper limit keeps
the F32 discount below1. Both canonical/private runners must pass these checks;
all actual queued commands must reach intercepted constructors without an agent
or emulator. Retain exact source provenance for discount use in continuation,
CPU replay targets and GPU imagined targets. No native build or new graph rule.

Train8,192 actions/489 updates,first376,residual.5,within180s. Then1,024 greedy
frozen actions within60s,zero updates and357 invariant tensors. Require exact
357 initial tensors and47 pre-update events matching retained,finite weighted
losses,240-row label bounds,changed CNN/reward/continuation/actor/value,correct
config/counters,complete training pixel/action/label/override replay and guards.
Frozen natural episodes are not required at this execution-only budget.

Require time×200000/8192×1.1≤1800s. Report retained47.690s timing without claiming
a reliable speedup; this targets learning quality. Replay120s,audit180s,main600s,
collector650s. Stop actual faults/nonfinite values,preserve all failures,do not
repeat completed stages,reset the device,overlap GPU work or extend caps.
No discount sweep or automatic successor.

Only after the execution/cost gate passes may a separate fresh200k learner be
selected:30-minute training cap,100k/6,227 recovery,12,477 final updates and four
final/actual-initial sampled/greedy24-natural-episode arms. Greedy stays primary;
favor only higher raw score and own-initial gain/min with a positive paired lane
interval. Keep the original864/90%-two-wall gate. Reused1009/cohort and historical
timing are development evidence,not training-seed reliability.

Root `/x/Temp/kindle-breakout-discount100-20261104`; service
`kindle-breakout-discount100-check-20261104`. Preparation pending; no learner is
active at selection. Retain143.667 greedy/70.750 sampled. The five-game goal
remains unfinished. Check live state before launch.
