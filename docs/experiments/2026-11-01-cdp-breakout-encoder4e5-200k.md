# Fresh200k Breakout comparison: encoder2e-5 versus4e-5

After the [bounded check](../results/2026-11-01-cdp-breakout-encoder4e5.md)
passes execution,restore,replay and the conservative20m59s cost projection,
select one fresh200k-action learner. The question is whether doubling the
encoder rate improves useful frozen learning at the same short action budget.
The retained2e-5 control scores25.292 sampled/42.625 greedy in22m27s,all0/24
two-wall successes. The earlier4M/expert10 rate4e-5 failure remains retained;
this distinct12M/B16/expert100 regime is not a general rate or causal claim.

Keep centered CDP/12M64/Auto/N8/B16/T16/H15/R16/expert100,base/actor4e-5,
dynamics4e-4,warmup1000,actor-unimix.01,action-effects coefficient1,full18,
sticky actions.25 and action repeat4. Change only encoder2e-5→4e-5.
Keep20% episode guidance/cap1920,all-state corrective labels,world imitation
gradients and all action weights1. Native447fe50,exact qualified runner and
pixel teacher are unchanged. No bounded or previous weights,replay,games,
belief or RNG are reused. Seeds1009/eval4000001009 remain development inputs.

Train200k actions/12,477 updates with a30-minute cap,first update376/residual
credit.5 and mandatory100k/6227 recovery audit. Require357 finite initial tensors
matching control and bounded check,47 matching pre-update transitions,finite
reports,changed CNN/reward/actor/value and exact source/config/counter accounting.
Verify all expert row/weight totals and aligned raw world/actor losses,complete
training input-pixel/action/teacher-label replay and all host guards.

Assess final and own actual initial checkpoints with both sampled and greedy
actions. Each arm has a300s/600k-action cap and selects the first3 natural
episodes per8 lanes=24. Guidance and labels are off; require zero updates and
truncations,357 invariant tensors,paired initial semantic traces,full recorded-
action/outcome/frame/reset replays and videos. Ordinary frozen logs do not carry
live pixel hashes. Preserve the90% two-wall gate; training/teacher scores do not
establish competence. Sampled is primary; retain greedy separately and report
both absolute means and own-initial-subtracted return per training minute.
Do not switch primary mode or select checkpoints after observing the result.

Main service cap4500s includes the fixed stages and CPU audit1200s; postprocess
5100s and collection5400s. Progress and recovery observers only inspect the
original process; they never start or restart GPU work. Collect curves,exposure,
paired comparison and terminal states. No retry,extension,cap increase,rate
sweep,confirmation or automatic successor follows this comparison. Stop on a
real GPU fault or nonfinite report; no driver recovery is authorized by this
protocol. A failed CPU closure preserves the original failure and GPU artifacts.

CPU preparation passes five actual CLI paths,24 audit/comparison config cases,
44 resolved paths,15 compiled controllers,8 header cases,4 observer-parser cases,
5 recovery-body cases,194 bounded/173 control pins and357 schema tensors in.959s.
An initial copied startup-header assertion still required2e-5. The CPU fixture
caught it before any full declaration or GPU stage; exact original scripts and
failure receipt remain under`full/preparation-failure`. Correcting the expected
rate and adding two wrong-rate rejection cases passes without native/environment
construction. No bounded GPU stage repeats.

Root`/x/Temp/kindle-breakout-encoder4e5-20261101/full`,trial
`breakout-12m-64-b16-r16-encoder4e5-expert100-200k-seed1009`,service
`kindle-breakout-encoder4e5-200k-20261101`. Recheck live state and host pins at
selection and launch; no full frozen result exists at declaration.
