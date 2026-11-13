# Breakout's first early disagreement usually remains recoverable

At the first policy/teacher disagreement in each of120 retained loss windows,
teacher-first and recorded-policy-first each avoid108 losses at the original
loss endpoint when followed by the same feedback-aware teacher. Both survive
in102 cases. The first command changes survival in12 cases, split six in each
direction. This does not show an aggregate survival advantage for the teacher's
first command on this cohort.

| Paired outcome | Through original loss action | Through64 actions from disagreement |
|---|---:|---:|
| Both survive |102|86|
| Only teacher-first survives |6|10|
| Only policy-first survives |6|11|
| Neither survives |6|13|
| Teacher-first total |108|96|
| Policy-first total |108|97|

The matched endpoint uses the same remaining action count for each pair.
Teacher-first minus policy-first reward through that endpoint sums to+9 across
120 cases; reward is separate from survival. The64-action follow-up begins at
the new disagreement origin and is not the old window-origin64-action endpoint.
All individual trajectories, action pairs, phases and lane results are retained.

## What this narrows

The earlier [full-takeover diagnostic](2026-11-01-cdp-breakout-recoverability.md)
avoids108/120 losses from an earlier window origin, compared with26/120 from
the last near-descending observation. The new experiment keeps the original
policy command at the first disagreement and still avoids108/120 losses when
the teacher controls subsequent actions. Most such first choices therefore do
not make the original loss inevitable under this teacher.

The evidence supports examining sustained subsequent control and the states
the policy reaches, rather than treating the first disagreement as the decisive
error. It does not establish that all early choices are harmless:12 paired
survival outcomes differ at the matched endpoint. Nor does it prove a specific
representation, forecasting, actor-fitting, timing or coverage limitation.
Teacher continuations are imperfect; a changed first command can also change
subsequent teacher memory and commands. Both arms use this feedback controller,
not the native policy on counterfactual states.

Selection uses the first exact command disagreement in every one of the120
preselected early windows, without rescue-outcome, confidence or phase filtering.
All windows contain a disagreement; none is excluded. They come from the same
24 development episodes conditioned on observed losses. This is not a new game
score or independent reliability result. The source greedy score remains104.667
with0/24 two-wall successes. No new learner, loss or replacement policy is selected.

## Execution, preserved failure and evidence

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-first-disagreement.md)
first fails before replay under source23fe57f. Its list-equality assertion assumes
saved origins use selection order, but the saved file uses execution order.
All240 complete original origin records are identical by unique stream/episode/
step. The original command failure and309 pins remain preserved; no original
emulator replay, branch, review or completion output exists.

The separate CPU repair checks complete records by unique identity and rejects
duplicates or changed fields. Twenty-two CPU tests and3.456s preflight pass.
It retains the same120 origins, endpoints, branch semantics and180s/195s caps.
No GPU or completed replay stage repeats. Closure is explicitly
`complete_with_cpu_order_repair`; the original controller remains failed.

The39.271s analysis /40.843s controller completes06:24:01 UTC November1,
sourced139b43. All37,016 source actions/148,038 frames reproduce, including
pixel hashes, labels, lives, rewards, boundaries, task outcomes and partial tails.
Complete source phase rows and aggregate metrics match. All recorded suffixes
and6,905 retained teacher-suffix rows match. Process copies preserve sticky-action
memory; parent game serialization and teacher state remain unchanged. Eight
reversed branch-order checks pass. Counterfactual branches execute56,722 frames;
recorded/reversed validations execute17,916 additional frames.

All320 repair pins and317 active Qbert inputs remain unchanged. The original
Qbert learner retains PID755504/invocation`5fb0d30ffb5d47aca552009e586bb929`.
No native model, GPU context or learner update occurs in this diagnostic.
Tests, preflights, the1.005s original failed command and repaired analysis
co-host on CPU; preserve their timing without correcting Qbert training time.
The original Qbert learner continues with its existing recovery/frozen observers.

Root `/x/Temp/kindle-breakout-first-disagreement-20261101`; completed repair
under `order-repair`. Do not rerun either controller.

| Artifact | SHA256 |
|---|---|
| Original failed command |`517f9b794bff154b4aa12072b4848b8b6f437e7a719106ee3c3f499b97a75cb1`|
| Repair declaration |`7e4b91baa555303faa7fcdc50c52ca8ebc071d8de90f88dfcde885115885f1fb`|
| Repair review |`7a0266543e769bd9762ab726dc6cfdc71da19dc148b051329c933ec8b3f499db`|
| Complete branch records |`625fa1df61a1b3306403b92f8ad2549e87138eff5220c8687f3da8d27adba394`|
| Repair completion |`2969eb42fa65b0793c8bff1855089cf57873b9eec449b22115cc09cb0e226292`|
