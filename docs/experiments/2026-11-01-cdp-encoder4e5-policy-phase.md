# Stronger Breakout policy: own trajectories and observed life losses

The encoder4e-5 learner improves greedy Breakout42.625→104.667, while fixed
state readouts improve and action evidence remains mixed. Select one CPU-only
replay of its saved final-greedy recording to measure the states and decisions
it actually produced. Record ALE life counts around each action to locate lost
lives. No native agent, model inference, new policy evaluation, learning or GPU
work occurs. The active Qbert learner continues unchanged.

Replay all37,016 recorded actions, preserving exact retained episode/task and
partial-tail outcomes, frame/reset counts and source identities. Select the
same first3 natural episodes per8 lanes=24; preserve excess episodes and tails.
Reuse the2e-5 policy's28,808 saved rows and metrics after exact reproduction,
without repeating its emulator replay. The two policies encounter different
states and episode lengths; this is not a paired-state or causal comparison.

Keep the exact phase, horizontal-action and aggregate functions and original
probe class. The replay function changes only its factory from the original
probe to a subclass that records before/after ALE lives, reward and episode
boundaries. It still returns the same environment result and leaves the original
pixel-teacher proposal/feedback order unchanged. Preserve all phases, full
confusion counts and per-lane metrics. Teacher labels are estimates, not optimal
action truth; horizontal agreement ignores other buttons.

For every selected life-loss event, retain the within-episode32-action window
ending at that action, or the available prefix for shorter episodes. Never cross
lanes or episodes and never look ahead. Report window indices, phase counts,
last visible/near-descending row, last near-descending action/teacher label,
teacher response-target/paddle gap and its age. Fixed age bins are0–4,5–8,9–16
and17–31 actions. Record terminal losses and life-count changes; bonus lives are
not losses. The old control has no per-action life counts, so this part is a
candidate-only description. It does not identify an action's causal effect.
Windows may overlap; missing detections are not verified serving states.

CPU preflight2.883s verifies263 input pins, prior196 analysis/227 learner inputs,
the four unchanged functions/classes and the replay function after only the
declared factory substitution. Original2e-5 aggregate and eight per-lane metrics
reproduce exactly. Eight pure fixtures cover boundaries, no lookahead,32-action
truncation, bonus/terminal lives and invalid transitions/selection. All276 active
Qbert inputs and PID736977/invocation48d20e8959f04654b6f5ff8932328d8d remain
unchanged. No native/emulator import occurs in preflight.

Run on one low-priority CPU core under90s, with native-module imports forbidden.
Require no GPU device descriptors, unchanged declared inputs and unchanged live
Qbert identity/pins before and after. Preserve replay-input hashes without
claiming parity with unavailable live pixel hashes. Retain CPU preflight/test/
replay co-host times without correcting Qbert training wall time. Stop on failure
and preserve evidence; never restart or disturb the active learner.

Root`/x/Temp/kindle-encoder4e5-policy-phase-20261101`. No architecture, loss,
replacement head or successor GPU allocation is selected by this diagnostic.

## Completed result

The23.375s CPU replay completes04:04:01 UTC November1/source3012999. All37,016
actions/148,038 frames reproduce retained outcomes;31,161 actions belong to the
selected24 natural episodes. Near-descending teacher agreement rises57.59%→
59.86% on each policy's own states; opposite-direction fractions fall and NOOP
fractions rise. Selected episodes last25.36% longer and earn95.88% more points
per action. All120 observed lost lives have a near-descending detection in the
preceding eight actions;86 last such decisions already match the teacher.
No action-level cause or counterfactual outcome is established.

All265 analysis/276 active Qbert pins, original process identity and replay
checks pass, with no native/GPU work, learner updates, failure or replay repeat.
Tests and preflight/replay co-host timing caveats remain. See the
[completed findings](../results/2026-11-01-cdp-encoder4e5-representation.md#own-policy-trajectories-improve-productivity-and-show-a-smaller-agreement-gain).
