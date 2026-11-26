# Breakout H5/R24 with the retained encoder4e-5 recipe

Select one bounded execution and cost check before considering a fresh200k
comparison. The retained H15/R16 encoder4e-5 learner scores56.292 sampled and
104.667 greedy in21m28s,withzero two-wall successes. Temperature0.5 does not
improve on greedy. The next question is whether more world-model and imitation
updates at a similar training cost improve this stronger representation recipe.

The older Breakout H5/R32 test at encoder6e-6 lowered late replay CE but failed
to improve frozen play and cost50.12% more training time. That negative result
remains. Qbert's newer H5/R24 gain motivates testing the cheaper tradeoff but
does not establish that it transfers to Breakout. No isolated horizon mechanism
or reliability claim follows from changing horizon and replay ratio together.

## Fixed recipe and bounds

Keep centeredCDP/12M64/Auto/N8/B16/T16,world-microbatch16,replay100k,encoder/base
rate4e-5,dynamics4e-4,actor-unimix0.01,sampled training collection and sampled
posteriors. Keep expert100,unit weights for all18 actions,world imitation gradients,
all-state corrective labels,20% episode guidance capped at1920 actions,full18,
sticky0.25,repeat4,native max-pooling and one GPU resize to64. Native447fe50 and
exact runner6c1b69f from the retained encoder4e-5 learner stay unchanged;no build
or new native feature is required.

Change imagination length15 → 5 and replay ratio16 → 24. At200k actions,
scheduled world/expert updates rise12,477 → 18,715 and imagined transitions
roughly halve. Replay sampling,cadence and action-clock warmup also change.
Use reused learner1009/evaluation4000001009 as a development comparison.

Start fresh for8,192 aggregate actions /733 updates,first update376,residual
credit0.75. Follow with1,024 greedy frozen actions under the separate evaluation
seed. Require357 finite initial tensors matching the control,matching pre-update
semantic events,changed CNN/reward/actor/value weights,733 finite reports,
aligned expert losses and complete pixel/action/teacher-label replay. Frozen
restore must havezero updates and357 finite invariant tensors. Check actual
guidance overrides separately from the20% episode-selection probability.
This check does not establish natural-episode competence.

Caps:180s training,60s frozen,180s CPU audit and480s whole service. The fixed
conservative cost gate is training seconds ×200000/8192 ×1.10 ≤1800s. A passing
check permits a separate decision on one fresh200k learner with a30-minute
training cap;it does not launch or select that learner automatically. No check
weights,experience,RNG,games or belief may be reused. No retry,cap increase,
horizon/ratio sweep,extension or automatic successor. Stop on a real GPU fault
or nonfinite report;no driver recovery is part of this protocol.

## CPU qualification and selection

The0.636s CPU preflight (0.715s controller) passes at09:34:22 UTC November1,
sourcef483bc0. It exercises both actual CLI paths,15 actual config-boundary cases,
357 checkpoint-schema tensors and all227 retained control pins. It derives
8,192/733,100k/9340 and200k/18715 update schedules. Native and emulator
construction are forbidden;no native module loads. No preflight failure occurs.
Do not rerun this passing qualification.

Select only after the completed temperature assessment and its collector are
terminal success and the host is idle. Pin the retained control,earlier negative
H5/R32 evidence,new Qbert H5/R24 evidence,qualified sources and the terminal
identity checker. Collect the original bounded process once,including its
invocation identity and cost projection. A full learner remains unselected.

Root `/x/Temp/kindle-breakout-h5-r24-20261101/native-check`,trial
`breakout-12m-64-b16-h5-r24-encoder4e5-expert100-check`,service
`kindle-breakout-h5-r24-check-20261101`.

## Completed bounded check

The original service launches09:35:39 and succeeds09:36:47 UTC November1,
source1b4671b,PID777123/invocation0d189ef7da054ce0866a40c9b9a0cc4b.
Training8192 actions/32,723 frames/733 updates takes50.971s;frozen1024/4096frames/
.884s haszero updates/no natural episodes. All267 pins,357 matching/invariant
finite tensors,47 initial semantic events,full replay and guards pass. Actual
overrides3800=46.387%. Terminal collection succeeds09:37:06,zero failures/repeats.

Conservative200k projection1368.859s (22m49s) passes30min. The check is complete;
do not rerun stages or collection. A separately selected
[fresh200k protocol](2026-11-01-cdp-breakout-h5-r24-200k.md) uses no check experience.
See the [result](../results/2026-11-01-cdp-breakout-h5-r24.md). No full competence
is measured at bounded closure;check live state before work.
