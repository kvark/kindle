# Breakout encoder8e-5 passes execution and early cost checks

The fresh12M64/Auto/N8/B16/T16/H15/R16/expert100 check completes8192 actions/
32,725 emulator frames/489 updates in46.730s. Restored greedy evaluation takes
.862s for1024 actions/4096 frames,with zero updates and no natural completed
episodes. Conservative200k projection20m55s passes the30-minute cost gate.
This establishes execution and early cost,not task competence or a gameplay gain.

Only encoder4e-5→8e-5 changes from the retained [4e-5 control](2026-11-01-cdp-breakout-encoder4e5.md).
Keep base/actor4e-5,dynamics4e-4,warmup1000,actor-unimix.01,action-effects1,
full18 actions,all-state corrective labels,expert100/all action weights1/world
gradients,20% episode guidance/cap1920 and one GPU resize to64. Native447fe50,
the exact runner and pixel teacher remain unchanged. Seeds1009/eval4000001009
are reused development inputs; no earlier weights or experience enter this run.
The earlier4M/expert10 larger-rate failure remains relevant negative evidence.

All251 pins,357 control-matching initial tensors,47 pre-update semantic events,
357 finite/invariant frozen tensors,exact counters/configuration,finite reports,
changed CNN/reward/actor/value parameters,full training input-pixel/action/label
replay and both host guards pass. First update376/residual credit.5 match.
All489 reports are supervised;117,033 labeled replay rows equal the weight sum.
Maximum raw world/actor expert-loss difference is7e-7. Actual3843 overrides are
46.9116% of actions;4349 passive labels and4033 executed/teacher disagreements
are exposure counts,not unaided competence.

CPU preflight1.625s validates both actual CLI paths,14 actual audit-config cases,
357 schema tensors,the8192/100k/200k update schedule and227 retained control
pins,without native/environment construction. Exact runner bytes match control;
base/actor rates remain4e-5 and previous encoder rates are rejected. No build,
backend change or new numerical implementation is needed.

The [bounded protocol](../experiments/2026-11-01-cdp-breakout-encoder8e5.md)
waits for the original Qbert pipeline and separate terminal confirmation to
finish before selection. Service`kindle-breakout-encoder8e5-check-20261101`
launches04:43:09 UTC November1/source5e71622,PID742430/invocation
`81da06ef2a6f4976b86d30145310dbbb`,and succeeds04:44:13 after1m04.737s whole
wall. Original terminal collection passes04:44:36. No failure,GPU repeat,
cap change or host recovery occurs. Brief status/documentation work co-hosts;
retain the timing caveat without correction or an isolated speed claim.

Root`/x/Temp/kindle-breakout-encoder8e5-20261101/native-check`,trial
`breakout-12m-64-b16-r16-encoder8e5-expert100-check`.

| Artifact | SHA256 |
|---|---|
| Declaration |`2c3d567cd425d74395d94e5cffeb6bc83129bd5720a66d4f618be586efe6c290`|
| Review |`40000b5700be3e5cd457389acd2e9d90314b0e478bde00575c7b6449656d5513`|
| Guidance replay |`c2dc174cd14d175b1aa3e62700d8a61bc41661c8991a7d545d61b64c93539f80`|
| Completion |`368c0f382c691bb50ed28fbb67db8e40aca819c62b3457c807ceb20e62772871`|

No full learner,extension,rate sweep or automatic promotion is selected by this
check. A full comparison needs a separate decision against retained sampled
56.292/greedy104.667,with actual-initial controls and no check experience reuse.
No GPU allocation remains active at closure;check live state before work.
The five-game goal remains unfinished.

After completed bounded closure,a [separately selected fresh200k comparison](../experiments/2026-11-01-cdp-breakout-encoder8e5-200k.md)
will test frozen gameplay against retained56.292/104.667. It reuses no bounded
experience and retains the30-minute cap and original two-wall criterion.

## Full comparison launch and startup

The fresh200k learner launches04:49:29 UTC November1/source3897e74,with265
pinned inputs,mainPID742893/invocationeccf5a76eed6458babd75b872c1704e4. All four
observers attach;the30-minute training cap,100k/6227 recovery and fixed final/
actual-initial sampled/greedy24-episode arms remain unchanged. Startup passes
357 initial tensors matching control and bounded check,47 pre-update transitions,
first updates376/392/408/424/440/456 and267 finite supervised reports through
4648 actions. Its1.007s CPU check co-hosts with training;retain timing caveats
without correcting wall time. No full frozen outcome is available.

Declaration`5009887d92a4e26c2ebe5836ef3734ec6027e888113beb61e1fc1729cb18d23e`.
Do not restart this learner or schedule another GPU job while it remains active;
inspect the original service and observers before further work.
