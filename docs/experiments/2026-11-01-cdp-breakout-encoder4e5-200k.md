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

## Launch and startup

The learner launches03:07:11 UTC November1/source2f9af49,PID732902/invocation
`c69d91c610d24751bc334966ab8057b5`. Declaration pins227 inputs; SHA
`aae0c85102e3518464c312a5b93d4b030837798c9a78f7323db9c3b8bcb42db0`.
Startup verifies357 initial tensors matching control/check,47 matching pre-update
transitions,first updates376/392/408/424/440/456 and208 finite reports through
3704 actions. All five services are active with the original process identity.
The0.849s CPU startup command co-hosts with learning; retain this timing caveat
without wall-time correction. No full frozen score or gameplay gain is available.
Do not restart the learner or schedule another GPU job while it is active.

## Halfway recovery verified

The original recovery observer passes03:18:16 UTC November1 and exits success.
Its100k-action/6227-update checkpoint has357 finite tensors;the complete log
prefix has6447 finite reports through103,528 actions. The recorded102,400-action
progress takes653.209s. Original main PID732902/invocation remains active.

A.542s CPU confirmation at03:19:15 rechecks all227 declared inputs,original
process identity,terminal recovery success and the exact checkpoint metadata
identity. It preserves`recovery-metadata-100k.json` before the live final
checkpoint is replaced at the declared200k boundary. This retains metadata and
the existing tensor audit,not a separate copy of the100k tensor files. No GPU
work,stage repetition,cap change or wall-time correction occurs. Progress,
postprocessing and collection observers remain active;frozen competence is
still unmeasured.

Recovery review`0294fa0526ad22572633d16705eb89a627fa3dc63d1b3c0d4f44214f6da7dbb3`;
checkpoint metadata identity`3022afbb0fcd26027fb5153ed9c5bd39d5945f62bcc331a5a5725713b54003c4`;
confirmation`35ce8548c2d5ce8ea94206487b5a57eb1ec13359aef89fabf7b0ed8c3a69e2b7`.
