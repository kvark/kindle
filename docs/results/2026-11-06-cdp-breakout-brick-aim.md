# Sparse-wall aiming fails the fixed teacher comparison

The [declared CPU candidate](../experiments/2026-11-06-cdp-breakout-brick-aim.md)
scores **208.458 versus 225.083**, a **7.386% decrease**, with only **1/8**
improved lane means. Both controllers remain **0/24 two-wall**. It fails the
required 10% mean improvement and five improved lanes. Keep the original
teacher; no fresh cohort, integration, threshold sweep or successor is selected.
These are hand-written controller returns, not learned Kindle competence.

The candidate preserves the original pixel tracking, unbiased landing cache,
five-step mean paddle response, edge reset and action cost. It adds a fixed
6-by-18 RGB occupancy grid and requests a three-pixel ball-contact offset toward
the remaining-cell centroid only with 1–36 detected cells, projected descent,
an interior landing and at least 16 pixels of centroid separation. The protocol
fixes every threshold. No emulator state enters the policy; task observation is
post hoc. Native model, guidance and training losses remain unchanged.

| Lane | Retained teacher mean | Sparse-wall aiming mean |
|---|---:|---:|
| 0 | 190.333 | 194.000 |
| 1 | 96.667 | 96.667 |
| 2 | 184.667 | 184.667 |
| 3 | 221.000 | 197.667 |
| 4 | 345.000 | 321.000 |
| 5 | 363.667 | 325.333 |
| 6 | 99.000 | 99.000 |
| 7 | 300.333 | 249.333 |

Each mean includes the first three natural episodes of its lane. All 24 selected
episodes finish without truncation. The eight environments reuse development
seed 4000001009 and its original lane seeds, with full 18-action space, sticky
probability .25, repeat four, no initial no-ops and native max-pooled RGB. The
retained control is reused from disk; it is not recollected. This selected cohort
does not establish broad reliability or isolate why the new rule regresses.

## Decision coverage

The selected episodes contain 39,749 decisions. The rule is eligible on 445 and
changes 191 first commands relative to the original cost at the same candidate
state. Fallback reasons are 18,014 without projected descent, 16,992 dense-wall,
3,930 missing-object, 219 centroid-near-landing and 149 edge-landing decisions.
No empty-wall fallback occurs. Eligibility does not guarantee a changed command.

| Detected occupied cells | Eligible decisions | Changed first commands |
|---|---:|---:|
| 1–12 | 19 | 8 |
| 13–24 | 117 | 51 |
| 25–36 | 309 | 132 |

These are comparisons within the candidate trajectory, not counterfactual
trajectory outcomes or causal credit for the score difference. Cell occupancy
is a fixed pixel heuristic, and the requested contact offset is not a calibrated
outgoing-angle model. This failure neither validates a general edge bias nor
rejects every policy that uses remaining bricks. Earlier fixed-edge aiming,
outer-entry and contact-risk failures remain retained in their linked reports.

## Execution and verification

Six CPU fixture groups pass, covering pixel thresholds, saved-image geometry,
steering gates/sign, unbiased-cache handling, invalid inputs and equivalence to
the original planner when disabled or ineligible. Preflight takes .245656s,
checks 47 pins, preserves the mean recurrence/cost AST and tracking bytes, and
constructs zero emulators with no native import. The two saved compressed PNGs
support grid geometry only. The earlier exploratory exact-color match found
zero cells and was rejected before qualification; its limitation is preserved
in the protocol and image provenance.

Before the first candidate action, all eight actual initial native frames pass
the declared 108-cell / 79.5-centroid check. One collection completes **43,176
aggregate actions / 172,667 emulator frames in 33.187557s**. Partial tails are
retained separately from the selected episodes. Complete **38.043074s** replay
matches every policy decision/debug record, input/output pixel hash, reward,
frame count, reset and task outcome. The controller finishes in **71.688512s**
at **19:16:42.795 UTC November 6** with all 48 declared pins unchanged.

The independent CPU recount passes at **19:16:50.885 UTC** in .210911s. It checks
all 48 pins and 31 completion artifacts, reconstructs all rewards/episodes and
decision counts, and independently reproduces lane means and the failed gate.
No test, collection, replay or recount fails or repeats. No native agent, learner
update or GPU work occurs.

Execution source `e52a684410a763a1fcc163437004f5cca9261549`; artifact root
`/x/Temp/kindle-breakout-brick-aim-20261106`. Exact source/ALE/ROM/adapter/control
identities, images, action ledger, video and receipts remain there. The original
controller and child are terminal; the publication receipt records live process
and render-device checks. Retain CDP scale 500 and learned Breakout 143.667
greedy / 70.750 sampled. Original 864/90% two-wall and the five-game goal remain
unfinished.
