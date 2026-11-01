# Current Qbert action rankings on shared observation histories

**Running:** original service launches22:14:15 UTC November 1/source4aa6ac7.
Both128-action bounded captures pass; the full pair is active. No complete
ranking result is available yet. Preserve original main/collector identities.

The current RGB64 critics predict useful returns on their own policies, with
different later-play biases. Test the current models' action-dependent forecasts
on shared inputs before choosing a new loss or learner. This diagnostic belongs
to the existing five-game goal. No gameplay improvement or learner is selected.

## Fixed comparison

Use the retained learner10007 greedy trajectory, environment seed4000010007,
78,368 aggregate actions and 24 selected natural episodes. Compare frozen final
10007 and 12011 checkpoints on exactly the same recorded pixels/actions, using
the existing scalar re-observation path. An explicit CPU-only probe option
permits only checkpoint seed12011 to differ from the source config. Preserve
each model's own config and RNG. This controls input histories, not categorical
draws or latent states, and does not reproduce original vector-actor beliefs.

The unchanged selection rule chooses four evenly spaced valid stride16 origins
per phase per test episode: 128 total, 64 before and 64 after the first pyramid.
The first natural episode per lane is excluded from selection; the second and
third supply origins. Selection never inspects rewards or model predictions.

At every origin, each of 18 actions is held for four policy steps, then follows
the recorded continuation through 15 steps. A nineteenth plan uses the exact
recorded continuation as a replay control. This tests short open-loop plans;
the subsequent recorded actions may be unsuitable after a changed state. It is
not a deployment of planning or a full adaptive-policy value comparison. The
15-step forecast also exceeds the current H5 imagination horizon.

The CPU oracle forks a fresh process from the same emulator/RNG state for each
plan, preserving sticky-action memory. Require recorded-plan reward, boundary
and frame parity at every origin, reverse-order equality at one origin per lane,
unchanged parent state and exact full recorded transitions. Store complete
pooled-pixel trajectory hashes. No GPU context may exist in the oracle process.
There is one paired stochastic future per candidate, not an expected-return
estimate across independent futures. Do not repeat the October29 data or its
failed snapshot prototype.

## Qualification and execution

Before selecting GPU work, run CPU ranking/config/ledger tests, parse all four
actual capture commands without native construction, verify both 357-tensor
sources and the preceding completed critic inputs, and complete the single new
CPU oracle. Pin sources, runtime, checkpoints, original log/replay, selection,
oracle, queues and analysis. Require a clean committed tree and idle host.

Use unchanged native447fe50/SHA35ad0db from the RGB64 runtime. Run one128-action
check per model, each capped120s, with all19 forecasts at origin0. Require finite
forecasts, state/probability/counter invariance, exact repeated-plan predictions,
357 invariant tensors, zero updates and host guards. Both must pass before the
full captures. Then capture128 origins per model while replaying78,368 actions,
each capped600s. Require exact source transitions, identical CPU/native pixel
hashes, matching origins/plans, config/counters, finite outputs, all357 frozen
tensors and host guards. No earlier current-model forecast arrays exist; exact
repeat predictions and native invariants are checked without claiming parity
with the older RGB48 model. No native build or training update.

## Fixed analysis

Reuse the existing15-step scoring and tie handling: predicted external rewards
weighted by cumulative predicted continuation; actual rewards discounted with
F32 gamma, stopping at true termination/cutoff. Exclude the recorded control
from the18 candidate set. Report informative origins, pairwise agreement on
unequal actual outcomes, selected actual return, regret, gain over uniform and
reward MAE, overall and before/after the first pyramid. Ties receive mean actual
return and half pairwise credit. Scalar actor preference interpreted as a
four-step hold is descriptive, not the original adaptive policy's performance.

For each model and their paired difference, report mean gain by eight lanes and
a5,000-resample lane bootstrap using fixed seed20261029. Reused lanes and one
emulator future do not establish independent-learner reliability. Report both
models on these common inputs without attributing differences solely to weights
or explaining all of their own-policy score gap. Do not select a horizon, origin
subset, model or loss after inspecting outcomes.

Root `/x/Temp/kindle-qbert-rgb64-ranking-20261101`; main service
`kindle-qbert-rgb64-ranking-20261101`, collection service
`kindle-qbert-rgb64-ranking-collection-20261101`. Main cap2400s; collector2700s.
One original allocation only: no retries, cap increases, repeated stages,
extension, learner, loss change or automatic successor. The five-game goal
remains unfinished.

## CPU qualification complete

Preparation succeeds 22:13:29.395 UTC November 1, sourced08bcf2, in32.442s.
All53 ranking/config/forecast-ledger tests pass in.11s. Four actual CLI paths
parse before native construction; both357-tensor RGB64 schemas and64 preceding
critic pins verify. The60 preparation inputs are pinned. No native module,
queue, declaration or service is created during preparation.

The single shared CPU oracle replays78,368 actions and154,591 branch frames in
30.410s (30.747s wrapper), with128 selected origins andeight reverse-order
checks. All recorded-plan branches and full source transitions match; parent
states stay unchanged. Eight complete pooled-pixel hashes are retained for the
native comparison. No CPU failure or repeated oracle occurs. Do not repeat
successful preparation or the oracle.

Oracle SHA256 `3b5bbe0962a1453eca0ebab85878d95eceac7f287fb0cb53a9496bedb3de48ca`;
preflight `a57bdcb590f0caabcedbe2e6d1afc40db4f2ad68ef8c54cdae2ae8253a6f94b3`.


## Original launch and bounded captures

All64 selected pins pass; declaration SHA256
`a7a76524c86bf304e17f6d3a1063ea9e356db6ebb56478f988f670d9dfac70ee`.
Original main starts22:14:15.257 UTC November 1/source4aa6ac7, PID856594/
invocation315da71b11fb44e49522fc7bf72708fa. Original collectorPID856597/
invocation98afbe048fa74f398cfa851bbf02ce29 launches alongside.

The10007/12011 bounded captures pass128 actions each in3.458s/2.581s. Each
checks19 forecasts plus a repeated recorded plan, finite predictions, unchanged
live state/probabilities/counters, all357 invariant tensors and host guard,
withzero updates. No failure or repeated stage. The full pair proceeds under
the fixed original controller. Do not overlap another GPU allocation.
