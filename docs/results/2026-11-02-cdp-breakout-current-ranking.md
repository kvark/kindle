# Breakout short action rankings remain inconclusive

Neither retained model demonstrates a gain over uniform choice in the fixed
[shared-history comparison](../experiments/2026-11-02-cdp-breakout-current-ranking.md).
Only **8 of 128 origins** distinguish the candidate plans by actual 15-step
reward. Pairwise agreement is 63.24% / 66.67% for dynamics4e-4 / 1e-4 on 612 unequal
pairs, but selected actual reward is slightly below uniform for both models.
The conditional lane intervals include zero. This does not establish a useful
short-plan selector, a causal explanation of weak control, or a new loss.

Both models receive the same retained4e-4 greedy pixels/actions. At 128 fixed
origins, each of 18 candidate actions is held for four policy steps, followed by
the recorded continuation through 15 steps. The recorded 15-action plan is a
separate parity control. Origins are evenly spaced in the second and third
natural episodes per lane, without selecting by reward or branch outcomes.

| Metric | Dynamics4e-4 | Dynamics1e-4 |
| --- | ---: | ---: |
| Informative origins / total |8/128|8/128|
| Pairwise agreement on unequal outcomes |63.24%|66.67%|
| Forecast external-reward MAE |.871108|.849252|
| Actual reward of selected plan |1.027338|1.019756|
| Uniform actual reward |1.055113|1.055113|
| Best candidate actual reward |1.148455|1.148455|
| Regret |.121117|.128698|
| Gain over uniform |−.027775|−.035356|
| Gain lane95% interval |[−.065666,.007649]|[−.083287,.005062]|

The paired4e-4-minus1e-4 gain is **+.007581**, with lane95% interval
**[−.015094,.030325]**. These are 5,000 resamples over eight reused lanes with
fixed seed20261029, conditioning on one shared-RNG emulator future per plan.
They do not express uncertainty over independent learners or stochastic futures.
Actual ties are excluded from pairwise agreement; prediction ties receive half
credit and mean actual reward. All origin and lane results remain retained.

The recorded adaptive action sequence has mean 1.231551, higher than the best
four-step-hold candidate's 1.148455. The restricted candidate set cannot express
every recorded four-step sequence. Interpreting the scalar actor's preferred
first action as a four-step hold scores 1.080271 / 1.087852; these are not original
adaptive policy returns or a new gameplay evaluation.

## What this resolves and leaves open

The small difference in forecast MAE does not establish better decisions.
Only 6.25% of origins have distinct outcomes within this horizon, leaving little
action-sensitive evidence. We retain the fixed origins and budget; no alternative
subset, horizon or candidate family is chosen after observing these results.

This measures external reward over 15 steps using predicted preceding
continuations, which already include discount. It omits intrinsic reward and
terminal critic bootstrap. The [critic diagnostic](2026-11-02-cdp-breakout-current-critic.md)
shows weak sampled return calibration, but this new comparison neither confirms
nor rules out critic bootstrap as a control bottleneck. Long-term survival,
adaptive continuation and the complete actor training objective remain open.
The source trajectory belongs to4e-4; it is not a symmetric sample of both
models' own difficult states. Scalar posterior draws differ from those of the
original vector policy, despite identical raw histories and learner seed.

No learner, loss change or inference planner is selected by these findings.
No retry, origin replacement, extension or automatic successor is selected.
The five-game goal remains unfinished.

## Execution and evidence

The Python probe now permits an explicitly declared dynamics-rate difference
while requiring every other checkpoint config field to match. A separate
episode-uniform origin option preserves the old phase-balanced default. These
changes do not alter checkpoints or native learning/inference. Native447fe50/
SHA35ad0db and the qualified Breakout runtime remain unchanged; no build.

All 66 CPU origin/config/ranking/forecast-ledger tests pass. The first valid-rate
fixture omitted the required image format; its 65-pass/one-failure receipt and
original sources are retained, followed by a fixture-only correction. A separate
controller check incorrectly forbade importing ALE instead of constructing an
emulator. Its failure remains; the corrected check blocks construction, verifies
zero attempts, and passes ten direct-global screens and synthetic summary arithmetic.
Neither failure repeats a successful experiment stage or changes native code.

CPU preparation completes 00:15:02.990 UTC November 2 / source3d31ccd in 25.071s
(25.120s wrapper), checking four actual commands, both 357-tensor schemas,
227/272 learner pins and 73 preceding critic pins. The single CPU oracle takes
21.987s for 37,016 source actions and 153,132 branch frames. All 128 recorded-plan
checks, eight reversed orders, unchanged parent states and source trajectories
pass. Forked branches preserve sticky-action memory; no GPU context is constructed.

Bounded 128-action captures take 3.773s / 2.571s. Full captures take 57.933712s /
57.914495s: **74,032 recorded actions / 296,076 source emulator frames /
115.848207s**, zero updates. Each model evaluates 128 origins and 2,432 candidate/
control forecasts, plus 128 repeated recorded-plan checks. All eight pooled-pixel
hashes match the CPU oracle for each model. All 71 pins, 357 invariant tensors
per capture, config/counters, finite forecasts, unchanged live beliefs/probabilities
and host guards pass. There is no prior current-model forecast array to compare;
repeated forecasts do not establish an average over independent posterior draws.

Main PID871905 / invocation `9ed47a38d0d342ac836ed4b91ee019a1` launches 00:16:26.029 UTC
November 2 / sourcefc5032a and succeeds **00:18:33.552**. Original collector PID871908 / 
invocation`8299622f9c054bf3afc5bdb3f4a5bff9` writes completion at
**00:18:46.467** and is terminal success. Its journal has no separate resource-use
terminal timestamp; the complete payload and terminal systemd state are verified.
The .852s read-only summary (.870s wrapper) verifies all pins, completion files,
original identities and four capture reviews. No original capture, analysis or
collector fails or repeats. Both services are terminal and the host is idle at closure.

Root `/x/Temp/kindle-breakout-current-ranking-20261102` is complete; do not rerun.
Original gameplay scores, zero two-wall successes and training costs are unchanged.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `a94e57ecea32816c2dd0b822178220533bf76d75cb026c28eeae288412dc50c2` |
| CPU oracle | `27ed4a178b74478cf9c94f67a108e0e72a7504fd55425fcc13145e7ceaac1f00` |
| Comparison | `5ae212692e091276c0ce2277868182e3ef98ca740498972a8dedbe32c73605c6` |
| Original completion | `da43fb616cde0fbe53c9a6934e0ab440b691d38fb5fc3c9b8aaa5e15f90f9e13` |
| Read-only summary | `6fb6e8d020a380961c3083039a337ff3c9792c073e238087749e05c94c4fee1e` |
