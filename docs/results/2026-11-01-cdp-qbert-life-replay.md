# Qbert loses its lives after the first pyramid

The retained encoder2e-5 and encoder4e-5 greedy policies each lose all 96 lives
after completing the first pyramid across their fixed 24 evaluation episodes.
Neither records a life gain. The newer policy lasts 21.13% longer in actions,
but earns only 1.76% more total reward; aggregate reward per action falls 15.99%.
This locates the remaining failures in later play and separates longer survival
from efficient scoring. It does not identify a causal model or policy error.

| Fixed selected episodes | Encoder2e-5 | Encoder4e-5 |
|---|---:|---:|
| Natural episodes / first pyramids | 24 / 24 | 24 / 24 |
| Mean score | 13,785.417 | 14,028.125 |
| Actions in selected episodes | 42,622 | 51,630 |
| Median episode actions | 1,748 | 2,107.5 |
| Observed life losses / gains | 96 / 0 | 96 / 0 |
| Losses before or at first-pyramid completion | 0 | 0 |
| Losses after first-pyramid completion | 96 | 96 |
| Losses below episode action age 1,920 | 91 | 63 |
| Losses at or above episode action age 1,920 | 5 | 33 |

Every selected episode starts with four observed lives and ends with zero;
each lane contributes 12 losses. The newer policy shifts more losses to later
episode ages. The age split is descriptive: these frozen trajectories do not
measure the corresponding training-state coverage or establish that the
training guidance cap caused the failures. Both policies already lose lives
before that age. They visit different states, so this is not a matched-state
comparison or independent confirmation of a learning-rate effect.

The next useful question is which later-play decisions waste progress or lose
lives, and whether the learned representation, forecasts and action choices
can distinguish better alternatives on those states. This replay supplies
origins and histories for that investigation. It selects no new learner,
teacher, loss, intervention protocol or automatic successor.

## Exact replay and limits

The [fixed protocol](../experiments/2026-11-01-cdp-qbert-life-replay.md) replays
both complete saved greedy logs, including extra episodes and partial tails:
44,512 actions / 178,000 frames for the control and 60,648 / 242,553 for the
candidate. Every recorded reward, action-frame count, termination, truncation,
reset count, episode/task outcome and partial tail reproduces the original
replay. Only the existing first three natural episodes per eight lanes enter
the table; no new episode selection or gameplay evaluation occurs.

Life increases and decreases are observed separately at each raw frame, so a
gain followed by a loss within one action cannot cancel in the accounting.
All 21 cube-color bytes and their changes are retained, with 5,353 / 5,622
color-or-life events in selected control/candidate episodes. Colors also reflect
animation; these histories are not a generalized level detector. The existing
first-pyramid observer and success criterion remain unchanged.

Per-action reconstructed input/output pixel hashes, actions, rewards, lives,
ages and milestones are retained. Ordinary source frozen logs lack live pixel
hashes, so the reconstructed hashes cannot establish equality with unavailable
live images. RAM observations are post-hoc outputs and never policy inputs,
teacher labels or training targets. No model, teacher or GPU context is loaded;
the replay executes only previously recorded actions and performs zero updates.

Five pure CPU tests and the 3.036s input preflight pass. All 293 diagnostic inputs
and 265 active Breakout inputs remain unchanged; the original Breakout PID
742893 / invocation `eccf5a76eed6458babd75b872c1704e4` is preserved. Analysis
takes 49.272s (50.619s controller wall) on one low-priority CPU core and completes
05:07:23 UTC November 1, source `aa80855`. Both fixed caps pass. No failure,
retry or GPU work occurs. Retain CPU co-host timing as a caveat without
correcting Breakout training time.

Evidence remains at `/x/Temp/kindle-qbert-life-replay-20261101`, including full
action rows, raw-frame histories, per-lane losses, tests, declaration and
completion receipt. Source checkpoints and evaluations remain unchanged.

- Declaration: `74e0d89a48c9e5bfeac9290d565605acc0b3a94f35e5eb215f4b84567bb50675`.
- Review: `7a1fea080d26ea93f2e8618e0dab6161b0dedd11738cf38b4bc5b5ba076b9bfc`.
- Completion: `67f966e4727d8132881b485cb7eb012a51b4f1b979c37cb16cc708fff4a41017`.
