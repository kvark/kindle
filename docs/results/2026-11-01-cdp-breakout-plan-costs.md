# Many Breakout disagreements have small model costs, but cost does not identify harmless actions

On the retained encoder4e-5 greedy trajectories,**7,059/12,521 horizontal
teacher disagreements (56.38%)** increase the teacher's best planning cost by
at most one unit. The median increase is.572. Near the paddle,the median is
1.483 and463/2,625 disagreements (17.64%) exceed16 units. Thus exact disagreement
mixes weak and strong preferences under this reference model.

This does not justify dropping or softening labels based on those costs. In the
retained120 first-command interventions,three of the six cases where only the
teacher preserves the life have cost gaps below one unit. Conversely,some large
model gaps lead to identical survival outcomes. The planning objective is an
imperfect proxy for actual control outcomes;neither a new loss nor a teacher
change is selected.

## Fixed trajectory results

The analysis reconstructs all34,429 planning rows among37,016 saved actions.
Every chosen plan,predicted paddle path,chosen cost and teacher label matches
exactly. The primary firstthree natural episodes per eight lanes retain31,161
actions,including28,873 planning rows and2,288 separately counted unplanned rows.
The existing greedy104.667/zero two-wall result is unchanged. No emulator replay,
new action branch,native forward pass or learning occurs.

For each of three first horizontal commands,the original objective is minimized
over its81 possible five-command continuations. “Excess cost” below is the best
cost for the recorded command minus the teacher's chosen minimum. It includes
squared predicted paddle error,and where applicable terminal-velocity cost,
plus command effort. Units are reference-model costs,not lost reward or physical
miss distances. Future commands are optimized independently;they are not native
policy continuations.

| Saved phase | Planning rows | Horizontal disagreements | Median excess cost on disagreements | At most1 unit | Above16 units |
|---|---:|---:|---:|---:|---:|
| All |28,873|12,521|.572|7,059 /56.38%|805 /6.43%|
| Ascending |13,334|5,675|.469|3,456 /60.90%|82 /1.44%|
| Far descending |8,048|3,831|.523|2,209 /57.66%|247 /6.45%|
| Near descending |6,582|2,625|1.483|1,130 /43.05%|463 /17.64%|
| Velocity unavailable |884|381|.335|257 /67.45%|13 /3.41%|
| Stationary/ambiguous |25|9|.257|7 /77.78%|0|

The median next-best first-command margin over all planning rows is.252,
versus2.097 near descending. All eight lanes have median executed-command
excess cost below one on their horizontal disagreements. Full means,quantiles,
fixed.02/1/4/16 thresholds and lane results remain in the saved review.
The long tails matter:mean excess cost on disagreements is9.139 overall and
17.557 near descending.

Exact command disagreements total12,577,versus12,521 horizontal disagreements.
Full18 commands use the unchanged actual-action feedback mapping. This mapping
does not equate FIRE with NOOP in serving states;those states lack planning
costs and remain outside this calculation.

## Relation to completed interventions

Reuse all120 first-command intervention outcomes without rerunning them. Two
origins lack planning data and stay recorded. The table uses all available
planning origins,including two exact disagreements with the same horizontal
command in the both-survive group.

| Matched original-loss endpoint | Total cases | Planning origins | Median executed excess cost | Planning origins at most1 unit |
|---|---:|---:|---:|---:|
| Both survive |102|101|.923|52|
| Only teacher-first survives |6|6|4.895|3|
| Only policy-first survives |6|6|1.885|2|
| Neither survives |6|5|2.410|2|

The both-survive group includes a cost gap2,082.899;the teacher-only group
includes.0749. These counterexamples rule out interpreting cost magnitude as a
perfect survival classifier on this corpus. They do not estimate a reliable
mapping from costs to outcomes:the groups are small,correlated and conditioned
on recorded losses. The changed first command also changes later feedback and
teacher state. No architectural cause or beneficial learning intervention follows.

## Execution and evidence

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-plan-costs.md)
completes in4.606s CPU /6.342s controller,closing **08:50:43 UTC November1**,
source`7d1ea69`. All331 analysis pins,265 replay/320 intervention inputs,and401
active Qbert inputs pass. Original QbertPID771374/invocation
`82a217b818e24b5cb7c10801a2b3f237` is preserved. Eight scalar fixtures over243
plans each pass in the3.229s preflight;maximum cost difference5.684e-14.
An independent JSON recount verifies group counts,medians,thresholds and every
saved first-command excess-cost subtraction. No failure,retry,emulator/native
import,GPU work or learner update occurs. Retain CPU co-host timing without
correcting Qbert training time.

Root`/x/Temp/kindle-breakout-plan-costs-20261101`,including all per-row costs,
intervention joins,source/AST pins,fixtures and receipts. This diagnostic is
complete;do not rerun it. Qbert H5/R24 continues under its existing observers.
The five-game goal remains unfinished;no automatic successor is selected.

- Declaration`597e8f53c19569b94e1ffd1e069ca811bc504219e7a82e3dc4c0161b201e5dc5`.
- Review`0fbb20ea6d5184a6fcc6cf32acbd0898e6dc1d86a24bae08fd4019dc85483176`.
- Cost rows`3fe723168d87ca0210c86d91f671d93e71419ba49008867d744c2a079a2511a9`.
- Completion`788c49c6bbba621b6dd0b9b55a129b0eeb494a4b25eef01862f91fffe7abdbbe`.
