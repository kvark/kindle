# Breakout ending forecasts with intervening observations

The completed [rescue forecasts](../results/2026-11-04-cdp-breakout-rescue-forecasts.md)
keep nearly full continuation mass through 24 recorded game endings. Within the
five-game goal, measure whether continuation still misses the terminal arrival
when the model receives the actual intervening observations. No learner,
objective change or automatic successor is selected.

Reuse all 120 recorded nine-action failure windows and the exact replay-value1
checkpoint. Do not choose new origins, rerun teacher branches or replace the
prior results. Keep the 24 natural game-over windows separate from 96
intermediate-life losses: the continuation target is zero only at game-over,
otherwise the original F32 gamma. Life counts remain diagnostic labels only.

One new frozen capture re-observes all 43,424 source actions / 173,660 frames
in the same scalar lane order as the completed probe, forcing recorded actions
and preserving resets. At each of the nine outgoing states in every window,
query the prior rollout for the remaining recorded sequence through the loss.
At the original start, reuse the saved nine-step forecast and query only its
first step as an exact parity check. Thus the new queries have lengths 1 at
offset zero and 8,7,6,5,4,3,2,1 after intervening observations. Record all
rewards, continuations and values, with 1,080 query records and 120 exact
origin-prefix/actor-probability checks. Repeat each final one-step query once.

Require every input/output pixel, action, reward, frame count, boundary, reset,
origin and full trajectory hash to match the saved source. Forecast queries
must preserve live features, probabilities and counters. Require finite outputs,
continuations in [0,1], zero updates, correct action counters and all 357 tensors
byte-identical afterward. Use native447fe50/SHA35ad0db and the same corrective
runtime; no native build. A private copy changes only the probe query block and
accounting, keeping the completed probe and all prior inputs untouched.

The primary comparison is the squared error of predicted **endpoint
continuation** on the 24 natural endings: saved nine-step endpoint versus the
new one-step forecast immediately before that same terminal arrival. Report
paired mean error difference and a fixed 5,000-resample interval over eight
lanes, seed20261104. No promotion threshold or training change follows directly.
Also report endpoint continuation means/errors at all nine remaining horizons,
with the 96 intermediate-life windows as a separate stratum. The full horizon
curve is a diagnostic, not a search for a new imagination length.

Report products of refreshed one-step continuation predictions and weighted
endpoint values descriptively. A product made after seeing future observations
is not a survival forecast available at the original start. Shorter forecasts
change available observations, rollout length and diagnostic categorical draws
together. Exact origin-prefix parity controls the initial state, not later
stochastic draws. This comparison cannot by itself isolate recurrent dynamics,
representation quality, the continuation head or target coverage. These are
outcome-conditioned states from one reused learner/cohort, not population
calibration or training-seed reliability. Original vector posterior draws remain
different from scalar re-observation.

Before launch, verify the prior completed capture and repaired manifest, all
prior pins, fixed windows, terminal targets, variable forecast lengths, reused
prefix matching, error/discount arithmetic and actual capture entrypoint using
CPU fixtures. Pin new sources, copied-source receipt, protocol and inputs.
Select one guarded capture on an idle host, capped at 180 seconds, with a
600-second controller cap. Stop on genuine host faults, nonfinite outputs or
failed checks; preserve failures without automatic retries. Complete separate
CPU analysis and terminal collection; do not hash a collector's unfinished logs.

Root `/x/Temp/kindle-breakout-ending-feedback-20261104`; service
`kindle-breakout-ending-feedback-20261104`. Retain 70.750 sampled /143.667 greedy
in 21m28s. The original 864/90%-two-wall gate and five-game goal remain unfinished.
