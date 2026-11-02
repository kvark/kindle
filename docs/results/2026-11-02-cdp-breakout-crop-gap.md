# Breakout crop dropouts follow the inferred crossings in the examined losses

The saved teacher trajectory does not support the hypothesis that its
missing-ball FIRE fallback interrupts control before the inferred paddle
crossing in the examined losses. All 103 losses with an available crossing
still have a visible ball at the preceding decision. Retain the teacher;
no controller change or new learner is selected by this diagnostic.

The [fixed protocol](../experiments/2026-11-02-cdp-breakout-crop-gap.md) joins
existing teacher decisions to previously fitted y187 crossings. It uses the
original first three natural episodes in each of eight lanes, without new
emulator work, fitting, filter changes or trajectory generation. Each join
selects the latest input frame strictly before the crossing and verifies that
its action interval spans the crossing. A late descending gap must follow a
projected, visible ball at y≥175 with a visible paddle; the fallback action is
full18 FIRE (1).

| Existing event group | Events | Available crossings | Visible ball at preceding decision | Late descending gap at preceding decision |
| --- | ---: | ---: | ---: | ---: |
| All selected losses | 120 | 103 | 103 | 0 |
| Earlier usable loss-geometry subset | 65 | 65 | 65 | 0 |
| Previously fitted successful contacts | 776 | 776 | 770 | 6 |
| Earlier usable success-geometry subset | 674 | 674 | 669 | 5 |

The 17 unavailable loss crossings remain excluded from timing claims. Among
the 103 available loss crossings, 98 have a qualifying gap start within eight
raw frames; all 98 start afterward, by 0.045–4.333 frames. The earlier usable
loss subset has 61 such starts, also all afterward. None occurs before or
exactly at its inferred crossing.

Successful contacts have 84 nearby starts: 78 afterward and six beforehand.
Those six have one missing-ball decision and precede the fitted crossing by
only 0.637 raw frames. The earlier usable subset retains five of them and
62 afterward. Across the selected episodes, 429 qualifying gap starts are
preserved, including those outside any contact's eight-frame neighborhood.

These are descriptive counts on one reused teacher trajectory. Successful
contacts and losses have different fit filters, and events within episodes
are correlated. Max-pooled observations and fitted pixel crossings are not
exact collision times. In particular, the subframe success offsets should
not be interpreted as precise physical contact ordering. This analysis
cannot show whether another action would rescue a loss, quantify unbiased
failure probabilities, or diagnose information in Kindle's learned state.
The earlier teacher contact-position errors and imperfect gameplay remain.

The analysis completes **10:28:57.759 UTC November 2**, source `04c30c6`, in
**0.387 seconds** (0.444-second wrapper). All **14 CPU fixture cases** and
**17 input pins** pass. The original **52,072 actions / 208,254 frames**
reconcile exactly. There are no failures, repeated analysis or replay stages,
native/emulator imports, GPU work or learner updates. No Kindle allocation
is active before or after the analysis. The five-game goal remains unfinished.

Evidence is retained under `/x/Temp/kindle-breakout-crop-gap-20261102`:

- `declaration.json`: `bb80c4d004c4babf6e06b2bfea333fe9004a5949eaec7d106b5761efaf42a725`.
- `review.json`: `6c6391eb2540d74caefb91df244b9fbae3154797e0c6843a3ca489e4679efd82`.
- `events.json`: `68bda8bdf4be1f8656875e4cd05cc9a78598fd0b834b4065f6d2c25ced0f43aa`.
- `gap-starts.json`: `2168a72dccd7ff35b524afd28bd44025d48831107ebe1347f42941853a376f4f`.
- `preflight-execution.json` and `analysis-execution.json` preserve both
  successful executions. Do not repeat them.
