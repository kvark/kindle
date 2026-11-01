# Qbert encoder4e-5 passes execution and early cost checks

The fresh12M48/Auto/N8/B16/T16/H15/R16/expert100 check completes8192 actions/
32,754 emulator frames/489 updates in64.165s. Restored greedy evaluation takes
.800s for1024 actions/4096 frames, with zero updates and no natural completed
episodes. Conservative200k projection28m43s passes the35-minute gate. This
qualifies execution and early cost; gameplay benefit remains unmeasured.

Only encoder2e-5→4e-5 changes from the retained cheaper Qbert recipe. Native
447fe50, the exact runner and pixel teacher remain unchanged. Keep base/actor
4e-5, dynamics4e-4, warmup1000, coefficient1, actor-unimix0, full18 actions,
all-state corrective labels, expert100/all action weights1/world gradients,
20% episode continuation guidance/cap1920 and one GPU resize to48. Seeds4013/
4000004013 are reused development inputs; no weights or experience are reused.
The Breakout4e-5 gain motivates this check but does not establish Qbert benefit
or a universal rate. Earlier weaker-capacity Breakout and combined Qbert H5/R32
negative results remain retained.

All257 pins,357 control-matching initial tensors,47 pre-update semantic events,
357 finite/invariant frozen tensors, exact config/counters, changed CNN/reward/
actor/value, full training input-pixel/action/teacher-label replay and both host
guards pass. First update376/residual credit.5 match. All489 reports are finite
and supervised;117,261 labeled replay rows equal the weight sum. Maximum raw
world/actor expert-loss difference is8e-7. Actual4,341 overrides are52.991% of
actions;3,851 passive labels and3,425 executed/teacher disagreements are exposure,
not unaided competence. The short frozen segment supplies no score conclusion.

CPU preflight.483s verifies both actual CLI paths,13 actual audit-config cases,
357 checkpoint-schema tensors, the8192/100k/200k update schedule and235 control
pins with zero native/environment construction. No production change or build
is needed. Brief read-only status checks co-host with the run; retain that timing
caveat without correction or an isolated backend-speed claim.

Service`kindle-qbert-encoder4e5-check-20261101` starts03:54:07 UTC November1,
sourcee2887ef,PID736567/invocation`839b3845da2f4cd9b8061c18c218fa95`, and succeeds
03:55:47 after1m40.686s. Original terminal collection succeeds03:56:06.
No failed stage, GPU repeat or host recovery occurs. No learner or GPU allocation
is active at bounded closure; check live state before further GPU work.

Root`/x/Temp/kindle-qbert-encoder4e5-20261101/native-check`,trial
`qbert-12m-48-b16-r16-encoder4e5-expert100-check`.

| Artifact | SHA256 |
|---|---|
| Declaration |`079915ec0781aa725ce0f6d75b58455e7ac3f80dd41ded7d91d21da31bfc07a9`|
| Review |`6788cf140d38906189fba1db6716ec1899b945d3566e384554e05c6f512403fb`|
| Guidance replay |`9d4a9042984c4bba824b384d220f73ee2c006f86c10556f276a2a991a492beee`|
| Completion |`6b30940a408198b11c41bdd00ab700f89eb35e1c4e0d9448dff13f878bcf1b99`|

The [bounded protocol](../experiments/2026-11-01-cdp-qbert-encoder4e5.md)
selects no automatic promotion. After its completed correctness/cost checks,
a [separate fresh200k comparison](../experiments/2026-11-01-cdp-qbert-encoder4e5-200k.md)
tests gameplay against retained greedy13,785.417/sampled9,987.500. No check
experience is reused. The five-game goal remains unfinished.
