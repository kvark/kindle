# Breakout H5/R32 fits teacher labels better but does not improve frozen play

The fresh comparison completes **200,000 actions / 799,475 emulator frames /
24,954 updates in 33m30s**. Doubling world/expert updates while shortening
imagination lowers late training imitation error, but neither frozen action
mode improves the retained H15/R16 control. Training takes **50.12% longer**.
Retain H15/R16; no extension, second learner or automatic successor is selected.

| Fixed final-policy evaluation | H15/R16 control | H5/R32 candidate | Actual initial, both |
| --- | ---: | ---: | ---: |
| Sampled mean, primary |13.208|13.125|1.417|
| Greedy mean |17.875|14.333|2.208|
| Two-wall successes, either mode |0/24|0/24|0/24|
| Training updates |12,477|24,954|—|
| Training time |22m19s|33m30s|—|

Each frozen arm uses the first three natural episodes per eight lanes, with
zero guidance, labels, updates or truncations. Excess episodes and partial tails
remain recorded. Both candidate modes improve their actual initial controls,
but all miss the original 90%-two-wall gate. The four candidate evaluation arms
total 42,536 actions / 169,983 frames / 27.852s, separate from training time.

## Better replay-label fitting does not yield better task performance

Mask-corrected expert cross-entropy over the fixed action windows is:

| Training actions | H15/R16 | H5/R32 |
| --- | ---: | ---: |
| 0–50k |1.364|1.005|
| 50–100k |.884|.513|
| 100–150k |.777|.447|
| 150–200k |.696|.391|

Final-window error falls **43.80%**, and geometric-mean teacher-action
probability rises .499→.676. This is error on sampled training replay, not
held-out accuracy or greedy agreement. The lives encounter different states
and replay samples. H15→5/R16→32 changes horizon, cadence, sampling, action-clock
warmup and total work together; it does not isolate a causal mechanism.
The result does not establish that more optimization can never help, or that
representation precision alone causes the poor policy transfer.

Actual guidance overrides 109,667 actions (**54.8335%**), despite a 20% episode
selection probability; the remaining 90,333 actions receive passive corrective
labels. There are 24,954 supervised reports and 5,977,882 valid replay labels,
which include repeated samples. Maximum raw world/actor expert-loss difference
is 8e-7. Completed guided/unguided online means are 195.838/6.155 across 68/251
episodes. These use changing parameters, and guided returns include teacher
actions; neither substitutes for unaided frozen competence.

## Execution, audits and retained evidence

All **169 declared pins**, numeric/config/source/counter checks, exact update
cadence, changed CNN/reward/actor/value weights, **357 matching initial tensors**
and **357 invariant frozen tensors per arm** pass. Both actual-initial semantic
traces reproduce the retained control: 800 sampled and 795 greedy events.
Complete guided training input-pixel/action/label replay, frozen outcome/action/
frame/reset replays and videos, host guards, curves, exposure and comparison
pass. Ordinary frozen logs have no live pixel hashes; their replay does not
compare unavailable original images. Both saved plots are visually checked.

The 100k/12,454 recovery passes at **19:46:35 UTC** with 357 finite tensors and
12,738 finite reports through the complete 102,280-action prefix. Training
continues on the original process. Startup at 19:29:59 had already verified
357 bounded-matching initial tensors and the first updates at 376, 384, 392,
400, 408 and 416 actions. CPU preparation exercises five actual CLI validation
paths, 14 scripts, 44 paths and config/observer/recovery boundaries. A separate
plot-interpreter preflight reads the real comparator/control with ALE/native
imports forbidden, avoiding the previous RGB plotting dependency failure.

Source **86eb300**, service `kindle-breakout-h5-r32-200k-20261029`, starts
**19:29:30 UTC October29**, PID330763/invocation
`547224e0548644b089a790683a6c7c78`. Training ends 20:03:04; main succeeds
**20:05:58**, postprocessing **20:06:03**, progress **20:06:16** and automated
terminal collection **20:06:35**. Main service time is 36m27.782s including
setup, evaluation and audits. All services are terminal and successful; no GPU
stage repeats, host recovery or failed attempt occurs in this comparison.

The separate 9-pin terminal collector verifies the original four service
identities, using journal invocation IDs after successful unload, and invokes
the declared collector once. Its declaration is
`f4c53e67b2475d163538836a476e08aa309de4a176df0dfc12e483b91f9aec1f`.
Do not rerun completed collectors or restart this life. No GPU allocation is
active at completion; the five-game goal remains unfinished.

Native447fe50/SHA35ad0db, runner, teacher, 12M64/Auto/N8/B16/T16, full18,
actor-unimix .01, original rates and expert100/world gradients are unchanged.
Both lives start fresh with no reused weights, replay, belief, games, RNG or
bounded-check experience. Seed1009/evaluation4000001009 are reused development
inputs, so this is not independent learner confirmation. Scheduled imagined
transitions fall by one third while world/expert updates double. This is not
an equal-wall-time comparison or an isolated horizon/optimization test.

Two completed-control CPU diagnostics co-host on one low-priority CPU core:
.736s for the [timing summary](2026-10-29-cdp-learning-diagnosis.md#recorded-learning-work-dominates-breakouts-training-cost),
and 9.701s plus .161s for the [pixel precision and paired comparison](2026-10-29-cdp-breakout-representation.md#rgb64-pixels-retain-much-more-precise-ball-position-than-the-linear-feature-readouts-recover).
Neither reads the active trajectory to select its analysis; all169 inputs and
the original PID/invocation remain unchanged. Neither performs GPU work or
learning. Retain the co-hosting caveat without correcting training wall time.

[Fixed full protocol](../experiments/2026-10-29-cdp-breakout-h5-r32-200k.md).
Evidence: `/x/Temp/kindle-breakout-h5-r32-20261029/full`, trial
`breakout-12m-64-b16-h5-r32-expert100-200k-seed1009`. Logs, weights, videos and
plots remain outside Git.

- Declaration: `7865cc633e3a36102896a71bb384ea469fa7d0eeb7247f209742470d0b32958f`.
- Learning review: `0916670b081c21a78ba2a3024539c1515b76889cf14b900cea0d05bbe21d1207`.
- Recovery: `9a0bb5d25fe2d9f4855f00ffcbdb35b7f9d25ed3435d55e4d398149f1305fc06`.
- Comparison: `aa849d8b67385a9b52aa56553fbc4067affeb20706b3f0341d162d05e3661421`.
- Exposure: `b23dfb36aa19dfc5c070b1fd3058715a262915793360bfc734363a0d77578935`.
- Postprocessing: `f91beb8b8170299ba5736a80d7897bf319f69f8352e5faa1ff3b3822d4693c71`.
- Completion: `fd9c230d1f7fb636dca87a77c30fe6f4854489b0db8f107d342b3780cd334b73`.

## Bounded execution and early cost

The fresh 12M64/H5/R32/expert100 check completes **8,192 actions /32,717
emulator frames /978 updates in69.748s**. A1,024-action frozen restore takes
.895s with zero updates and no natural episodes. All357 initial tensors match
the retained H15/R16 control; all357 frozen tensors remain finite and unchanged.
CNN,reward,actor and value weights change, aligned expert reports remain finite,
and complete training pixel/action/corrective-label replay and host guards pass.

There are3,861 action overrides out of8,192 actions (47.131%), distinct from
the20% episode-selection probability. This is execution and early-cost evidence,
not natural-episode competence. The conservative200k projection is
**1,873.122s (31m13s)**, below the45-minute selection limit. No GPU stage repeats.

Source`60b505f`, service`kindle-breakout-h5-r32-check-20261029`, starts19:23:07
and succeeds19:24:32 UTC October29,1m24.966s including setup and audit.
Terminal collection succeeds19:25:09. All41 declared pins remain unchanged.
CPU preflight takes.270s: two actual CLI validation paths, two negative CLI
cases, nine actual config-boundary cases,357 schema tensors and132 retained
control pins; no native module or agent/environment construction occurs.

H15→5 and R16→32 change together. At200k actions, world/expert updates double
12,477→24,954 while scheduled imagined transitions fall by one third. Cadence,
sampling and action-clock warmup also change. This does not isolate fitting
difficulty, representation precision or horizon effects. Earlier weak H5
Breakout results remain; Qbert's H5/R32 benefit is not assumed to transfer.

The separately selected [fresh200k protocol](../experiments/2026-10-29-cdp-breakout-h5-r32-200k.md)
keeps this checked recipe, original two-wall gate and sampled/greedy evaluations
against actual-initial controls. No check experience or weights are reused.
The full learner has no measured task result at selection.

[Bounded protocol](../experiments/2026-10-29-cdp-breakout-h5-r32.md).
Root: `/x/Temp/kindle-breakout-h5-r32-20261029/native-check`, trial
`breakout-12m-64-b16-h5-r32-expert100-check`. Native447fe50/SHA35ad0db,
exact retained runner and pixel teacher are unchanged; no build occurs.

- Declaration: `72155bc757ec5b007d33a20cbc4178f995d18ade0852ba742c01d801ab922110`.
- Review: `dbfe787ed2a6fec92d743c01af042abe2a26887aa942078273e8f7f6cc6f99aa`.
- Guided replay: `673a45c43e282a6898db911d9c1406b54c960c73d6b9d8294fa0d3bf08ffea02`.
- Completion: `9965aed35f91f4975e142039b559c8122c2f1926e4c966dff379de1059601e1b`.
