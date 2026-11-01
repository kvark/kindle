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

## Separately selected full comparison is active

Fresh200k training launches03:58:23 UTC November1/source7f794b0 with276 pins,
mainPID736977/invocation`48d20e8959f04654b6f5ff8932328d8d`. The35-minute cap,
100k/6227 recovery and final/actual-initial greedy/sample24-episode assessments
are fixed. Native447fe50, exact runner and teacher remain unchanged. Startup
passes357 matching initial tensors,47 pre-update transitions, correct cadence
and94 finite reports through1880 actions. The.815s CPU check co-hosts with
learning; retain its timing caveat without wall-time correction. All four
observers attach at startup. No full frozen outcome is available; do not restart this
learner or schedule another GPU job while it is active.

The separate CPU Breakout own-policy analysis co-hosts with this learner:
preflight2.883s, eight pure fixtures and23.375s replay (24.583s controller wall).
It completes04:04:01 UTC with all276 inputs and the original Qbert process
unchanged, no native/GPU work and no updates. Its timing remains a caveat without
correcting this learner's training wall time; the retained co-host receipt links
the exact analysis completion. No new Qbert outcome is inferred.

### Halfway recovery passes

The original recovery observer passes04:12:29 UTC at100k actions/6227 updates,
with357 finite tensors and6353 finite reports in the complete102,024-action
log prefix. The observer is terminal success; the main learner, progress,
postprocessing and collection observers continue. Independent.607s CPU
confirmation checks all276 pins, original process identity, actual checkpoint
configuration and all three tensor-file hashes against the saved metadata.
The exact100k metadata is retained before the final checkpoint replaces it.
This verifies recovery, not frozen gameplay.

The separate [Breakout teacher-takeover analysis](2026-11-01-cdp-breakout-recoverability.md)
also co-hosts: preflight2.507s, tests.137s and36.589s analysis (37.941s controller
wall), ending04:14:23. All276 Qbert pins and its original process stay unchanged.
The co-host receipt preserves the analysis hash and timings without correction.

- Recovery review`949ab8d7dc7e9e2ea7b694622c993a5d17594330630fa98823d741c1b8051590`.
- Metadata`aed08d98e6efd576d05d367c7ab0aeb1e4d1075121f421f1c6634ab4b9de4c0c`.

## GPU stages complete; original CPU closure pending

Training finishes04:26:10 UTC November1 at200k actions/799,739 frames/12,477
updates in1661.025s (27m41s),below the35-minute cap. All four frozen stages
finish by04:27:41 with zero updates and truncations. Completed frozen outcome
replays select the fixed24 episodes: greedy14,028.125 and sampled11,522.917,
each24/24 first pyramids; own initials80.208/177.083 clear none. Both trained
means still miss15,000. Greedy improves six of eight lane means and sampled
seven,with different trajectories and reused development inputs.

The original training pixel/action/teacher-label replay remains active under
mainPID736977/invocation48d20e8959f04654b6f5ff8932328d8d. Postprocessing and
terminal collection still wait; no accepted full comparison is claimed yet.
No GPU stage repeats or new GPU job occurs. The separate Breakout8e-5 CPU
preparation takes1.625s after training/evaluation have finished; it does not
alter recorded training time. Its selection waits for this pipeline's complete
terminal evidence. Terminal-verifier preflight.041s used retained control
artifacts only and co-hosted with training; retain that caveat without correction.
