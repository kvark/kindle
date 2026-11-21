# Matched frozen contexts for action-path derivative diagnosis

Both captures in the [declared protocol](../experiments/2026-11-06-cdp-breakout-action-derivative-contexts.md)
are complete. The retained control and the failed dynamics-gradient candidate
each supply 120 latent states at the same saved rescue origins. This stage
captures inputs; it does not measure derivative quality or train a policy.

| Frozen checkpoint | Actions | Emulator frames | Origins | Wall time |
|---|---:|---:|---:|---:|
| Retained replay-value1, 200k | 43,424 | 173,660 | 120 | 50.591s |
| Actor dynamics coefficient .5, 200k | 43,424 | 173,660 | 120 | 50.498s |

The controller completes in 105.261s, or 106.730s including its wrapper and
initial pin verification, at **06:42:29.897 UTC November 6**. Total re-observation
is 86,848 actions / 347,320 emulator frames, with **zero updates and zero
multi-step forecast queries**. No stage failed or repeated. The CPU closure
finishes at 06:43:18.395 UTC; no Kindle service or render-device holder remains.

All 16 CPU tests, 537 execution pins, both host guards and all 357 finite,
byte-identical checkpoint tensors per model pass. Every source input/output
pixel, forced action, reward, frame count, boundary and reset matches the saved
history. Both complete stream summaries match the earlier forecast capture.
Each feature has 2,048 deterministic values and 32 one-hot groups of 16 classes;
all features/probabilities are finite, action probabilities normalize, and
repeated reads preserve states and counters exactly. Minimum reported native
budget headroom is 15,697,072,128 bytes for control and 14,958,870,528 for candidate.
These are budget reports, not a separate GPU telemetry measurement.

Both checkpoints use qualified native **744dd793**, SHA256
`876ed0c55a9dae95b414aea650d31bfa35544f866c71236da82f937dd50c1a23`.
The selected capture source is `e8cec160ff9bf80408a645037b669949266cd610`.
The retained control's original runtime differs; this common runtime has separate
default-off compatibility qualification. Configurations differ only by the
candidate's actor dynamics coefficient, after normalizing the existing imagined
bootstrap default. Coordinates remain off. The original branch/forecast failures
and their separate CPU repairs remain preserved; their completed stages did not run.

Root `/x/Temp/kindle-breakout-action-derivatives-20261106` retains the declaration,
queue, capture scripts, tests, source/native receipts, guarded output and closure.
`control/result.json` has SHA256
`66e7614160c6f7345a5d093cf662a23c538867c90b0a2ead8a47247cf1c627da`;
`candidate/result.json` has SHA256
`d09e5fe07e7bfa2992b80c4bc90639ea0141e2179d22df8a9ddb1576bbb8022b`.
Do not repeat the capture to recover these inputs.

The next useful question is whether moving from a recorded action plan toward
the saved teacher plan has a favorable model derivative, whether the complete
discrete plan change has a favorable predicted effect, and whether either agrees
with the already known short outcome. Those are three different quantities.
Existing branches provide 92 short rescues and 18 differing game-over outcomes,
but no complete alternative-policy returns. All short branch rewards are zero;
critic bootstrap and continuation must be treated explicitly.

A native probe still needs its own objective, surrogate, paired-draw convention
and numerical qualification. Fixed-plan derivatives omit actor feedback and its
optimizer, and scalar posterior draws differ from the original vector actor.
These outcome-conditioned origins on one retained-policy history cannot establish
population calibration or the cause of the full learner's regression. No
derivative GPU stage, new loss, learner, planner or coefficient sweep is selected
by this completed capture. Retain the original control; Breakout's 864/90%-two-wall
gate and the five-game goal remain unfinished.
