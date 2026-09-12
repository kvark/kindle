# Qbert: early misses, completion bonus and limited later-stage exposure

The completed seed-0 pilot **learns, but fails competence**. The next controlled
question is whether a bounded increase in continuous training experience improves
both initial-pyramid reliability and subsequent play. Prefer that dose comparison
before importing Freeway's exploration or Breakout's action-width intervention.
This is a direction for a separate declaration, **not a new GPU queue or a fix**.

The [paired pilot](2026-09-11-atari-continuation.md#completed-qbert-pilot-learning-without-competence)
remains unchanged: 17/24 initial pyramids, mean 3,754.1667, versus untrained
0/24 and mean 125. Both frozen runs have zero updates and no cutoffs. The existing
90% milestone and mean-15,000 criteria still fail; reliable Atari remains **1/5**.

## What the complete frozen recordings show

The [CPU diagnostic](../../runs/qbert-frozen-diagnostic-20260911.asgiON/summary.json)
reconstructs **all 20,232 trained and 8,616 untrained actions**, preserving every
reward, episode boundary, reset and actual emulator-frame increment. All 24
episodes and all tails in each run match the original task reports. The raw
stream-0 RGB hashes match the archived replay videos exactly. These are repeated
reconstructions of existing actions, not additional policy evaluations; original
policy-input pixels were not recorded for this comparison.

| Trained episode group | Observation |
| --- | --- |
| Seven initial-pyramid misses | Maximum initial coverage is 16–20/21 cubes. Four reach 20/21, then repeatedly lose lives at the same 1,300-point score. |
| Seventeen initial-pyramid completions | Every completion occurs at score 1,325. Thirteen arrive with three ALE lives, one with two, three with one. |
| Their completion bonus | All seventeen reproduce the existing actual-ROM fixture's 31 increments of +100 at exactly the same relative frames: +8, +13, …, +158. Score after the bonus is 4,425. |
| Subsequent play | Adds only 150–1,000 points, mean 460.2941 and median 450, before natural termination. Full returns are 4,575–5,425. |

Thus the roughly 4,900-point cluster mostly reflects the first completion and
its bonus, not sustained high-scoring play. Most successful initial pyramids
retain three lives, so arriving almost dead is not the whole explanation.
The low post-bonus gain is an observation, not proof of a particular planning,
representation, exploration or control defect.

The diagnostic does **not** implement a later-round/level counter. Display colors
and their changes are retained as raw observations; animation, cube reversions
and later objectives must not be turned into invented level counts. The existing
Qbert observer is unchanged. Cube colors and the raw falling flag use the
[pinned OCAtari extractor](https://raw.githubusercontent.com/k4ntz/OC_Atari/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram/qbert.py);
life changes use the actual
[ALE 0.12.1 life counter](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/8a8fafb1ac37796e3e8197a2e5727204fa1b926d/src/ale/games/supported/QBert.cpp).
Neither RAM nor these diagnostics enter Kindle's observations or rewards. A
falling flag alone is not a verified cause for every life loss.

Watch the whole stream, including failures and tails:

- [Trained Qbert rollout](../../runs/atari-driver-continuation-20260911.LR9yT3/qbert-evaluation.mp4).
- [Untrained Qbert control](../../runs/atari-driver-continuation-20260911.LR9yT3/qbert-untrained-evaluation.mp4).
- Diagnostic stills from stream 0: [20-cube stall](../../runs/qbert-frozen-diagnostic-20260911.asgiON/trained-stream0-episode0-frame1690-life_change.png),
  [first completion](../../runs/qbert-frozen-diagnostic-20260911.asgiON/trained-stream0-episode1-frame1754-first_pyramid.png),
  [240 frames later](../../runs/qbert-frozen-diagnostic-20260911.asgiON/trained-stream0-episode1-frame1994-first_plus_240.png).
  These illustrate already-counted episodes, not selected extra wins.

## Complete training exposure

The separate [training replay](../../runs/qbert-frozen-diagnostic-20260911.asgiON/training-exposure.json)
checks all **200,004 actions / 49,651 updates**, 345 completed episodes, six tails
and **799,510 actual emulator frames**. All earlier archived 120,024-action
prefix episode outcomes match. Every learner row remains checked by the complete
source-matched ledger auditor; only emulator reconstruction omits learner rows.

There are **7,024 distinct positive training reward events**. Every reported
replay batch contains positive rewards; only the first two updates have zero
absolute advantage. This is not Freeway's zero-discovery failure. It also does
not establish accurate held-out world forecasts or causal planning.

Only **31 completed training episodes** reach the first pyramid. Across complete
episodes and tails, **37,443 / 799,510 frames = 4.6832%** occur after that first
milestone, including the bonus animation. This measures training exposure, not
a later-level completion rate or a guarantee that more exposure is sufficient.

| Training action window | Completed episodes | Initial pyramids | Mean online return |
| --- | ---: | ---: | ---: |
| 100,020–120,024 | 33 | 2 | 997.7273 |
| 120,024–140,028 | 25 | 8 | 2,130.0000 |
| 140,028–160,032 | 28 | 2 | 1,400.8929 |
| 160,032–180,036 | 26 | 11 | 2,663.4615 |
| 180,036–200,004 | 26 | 8 | 2,341.3462 |

Windows assign whole episodes to their ending action coordinate, so earlier
rewards can precede the window. These changing-policy online results are uneven;
they are neither a monotonic learning curve nor frozen checkpoints. The final
window is 19,968 actions; the earlier windows are 20,004 each.

## Next comparison and stopping rule

After the fixed Freeway → Pong → Breakout hardware queue, separately declare
a **fresh continuous 400,008-action Qbert pilot**, retaining an immutable
200,004-action midpoint and the final checkpoint. Evaluate both with the same
unassisted v4 episode budget and same-seed untrained control. Preserve every
outcome and the original failed pilot; do not reopen its completed queue.

Change **only experience budget**: keep N6/R256, model, actual LeVJEPA encoder,
qualified native/source pair, optimizer, full BPTT, action space and rewards
fixed. Verify the actual reset-dependent update ledger at each save rather than
borrowing a counter from another game. The midpoint is the within-run dose
control, not an independent seed; any comparison to the old pilot additionally
needs exact package/driver/initialization and trajectory/state checks.

The decision checkpoint is the declared final model, not a selected high training
score. If it still fails either unchanged competence threshold, preserve the
failure and revisit the intervention; do not automatically extend again. If it
passes, separately confirm the selected recipe on fresh roots 1009/2017/3019
with untrained controls. No pilot or CPU diagnostic adds a reliable game.

## Evidence and limits

The [frozen diagnostics](../../runs/qbert-frozen-diagnostic-20260911.asgiON/result.json)
also exactly replay the preserved scripted positive fixture, its 240-frame
follow-through, and both negative fixtures. They do not regenerate the original
exclusive fixture outputs, write RAM, clone state or provide demonstration data.
The positive fixture's raw RAM prefix and complete replay-video RGB hash match.

There are **23 passing CPU tests**: eight diagnostic, eleven exposure and four
bonus-summary checks. Changed reward/timing, task results, frame/reset clocks,
RGB hashes and partial-visit false completions have negative checks. All **115
summary pins and 47 raw replay snapshots** reverify. The actual active cgroups
record one-core / 2 GiB / zero-swap limits; the frozen and training replays peak
at **43.75 / 43.40 MiB host memory**. These are not VRAM measurements. Their
CPU runs construct no Kindle agent and start no GPU work.

[Setup failures](../../runs/qbert-frozen-diagnostic-20260911.asgiON/development-notes.md)
are retained: the first test service lacked its working directory, and the first
training-analysis invocation guessed a nonexistent filename. Both failed before
replay; corrected invocations use explicit working directory and the declared
training path. The completed frozen result and all active inputs are unchanged.

Result identities:

- Frozen: `435a1248cee4bc6f4e9ba35c0261d12eda8956c9c941cbf361e061153eb4f725`.
- Training: `db1527760165aeefe7ec2f20ad3d3a1cb65a32f98a2e3d7e019e0447200d4116`.
- Summary: `882caa7b456c932353e2be58b307c5c774bb3d72c4cec711896b9f545cfb803a`.
