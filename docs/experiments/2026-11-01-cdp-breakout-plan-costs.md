# Retained Breakout teacher planning-cost diagnostic

The completed [first-command interventions](../results/2026-11-01-cdp-breakout-first-disagreement.md)
find102 both-survive,six teacher-only,six policy-only andsix neither outcomes
on120 first disagreements in recorded loss windows. Exact teacher agreement
therefore does not measure optimal-action error. Select one CPU-only analysis
of whether the teacher's own objective strongly distinguishes its chosen first
command from the recorded learner's command. No new loss or learner is selected.

Use all37,016 saved encoder4e-5 greedy trajectory rows from the
[completed replay](../results/2026-11-01-cdp-encoder4e5-representation.md),
including partial tails. Primary summaries retain its firstthree natural
episodes per eight lanes=24,31,161 rows and greedy104.667/zero two-wall.
The prior emulator replay and120 completed interventions do not repeat.

Extract the unchanged production planner's AST from position initialization
through chosen first command,plus its exact243 five-command plans and calibrated
response coefficients. Inputs are saved paddle position/velocity,final response
target,collision horizon and the preceding actually executed horizontal command.
Respect episode resets. Require exact agreement with every saved chosen plan,
predicted paddle path,chosen cost and teacher label. Missing planning rows,
including serving states,remain counted separately. No native/emulator module
may load;no pixels,game state or GPU context are reconstructed.

For each first horizontal command NOOP/LEFT/RIGHT,take the minimum original
cost across its81 possible continuations. Report the next-best first-command
cost minus the chosen cost,and the executed horizontal command's best cost minus
the chosen cost. Map all18 actions with the unchanged feedback mapping;FIRE and
NOOP equivalence applies only to this horizontal planning calculation and does
not imply equivalent serving behavior. Future plans are optimized separately
for each first command;they are not native-policy continuations.

Report counts,means,quantiles0/25/50/75/90/95/99/100%,and fixed thresholds
.02/1/4/16 reference-cost units for all selected planning rows,each recorded
phase,and each lane. For the same120 first-disagreement origins,report these
quantities separately for the four already-recorded survival outcomes;retain
any origin without planning data. No fitting,threshold selection,confidence
filtering,new action branch or gameplay evaluation occurs.

The objective combines squared predicted paddle error,terminal-velocity cost
when beyond the planning horizon,and a.02 command-effort term. These are costs
under an imperfect teacher model,not realized reward/life-loss regret or a
validated equivalence class. Correlated rows and selected failure windows do
not establish causal architecture,loss or reliability effects. A small margin
would motivate caution about hard-label agreement;it would not establish that
softening labels improves learning.

Preflight3.229s passes eight independent scalar fixtures over243 plans each,
including stationary edges and fractional impacts,maximum cost difference
5.684e-14. All265 replay/320 intervention and401 active Qbert pins pass;native/
emulator import attempts arezero. Require original QbertPID771374/invocation
82a217b818e24b5cb7c10801a2b3f237 and its401 pins before/after. CPU analysis and
command caps60s/90s,one core/low priority;retain co-host timing without correction.
Root`/x/Temp/kindle-breakout-plan-costs-20261101`. No retry,automatic successor,
teacher change or GPU allocation. Qbert H5/R24 remains the only active learner.
