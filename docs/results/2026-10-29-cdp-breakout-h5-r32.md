# Breakout H5/R32: bounded check passes; fresh full comparison is active

## Fresh200k learner is active

The full comparison starts **19:29:30 UTC October29/source86eb300** under
`kindle-breakout-h5-r32-200k-20261029`,PID330763/invocation
`547224e0548644b089a790683a6c7c78`. Its200k-action/24,954-update target has a
45-minute training cap,100k/12,454 recovery,and fixed final/actual-initial
sampled and greedy evaluations. No full-run task outcome is available yet.

Startup passes19:29:59 with169 declared pins unchanged,357 finite initial
tensors matching the bounded check,and first updates at376,384,392,400,408,416
actions. All369 reports in the completed3,328-action prefix are finite and
supervised. These are execution checks, not evidence of gameplay improvement.
Progress,recovery and postprocessing observers run independently and cannot
restart the learner. No prior weights or check experience are reused.

The full CPU preflight validates five actual commands,16 config cases,all14
scripts,44 paths and observer/recovery boundaries. A separate check with the
plotting interpreter reads the actual comparator/control and matches357 initial
tensors with ALE/native imports forbidden. Native447fe50,runner and teacher
remain unchanged. Full declaration SHA256:
`7865cc633e3a36102896a71bb384ea469fa7d0eeb7247f209742470d0b32958f`.
Evidence: `/x/Temp/kindle-breakout-h5-r32-20261029/full`, trial
`breakout-12m-64-b16-h5-r32-expert100-200k-seed1009`.
Check live state before GPU work; do not restart this selected life.

Terminal collection is also automated from19:35:03 UTC, under
`kindle-breakout-h5-r32-collection-20261029`. Its separate9-pin declaration
reuses the exact previously checked service-identity implementation and verifies
all four original service identities plus169 active-learning pins. It waits for
successful terminal training/recovery/progress/postprocessing, then invokes the
declared collector once; an existing valid completion is reused. Failed services
or an observation timeout do not restart or stop the learner. Unloaded successful
services are checked against their original journal invocation IDs. Evidence:
`/x/Temp/kindle-breakout-h5-r32-20261029/terminal-collection`, declaration
`f4c53e67b2475d163538836a476e08aa309de4a176df0dfc12e483b91f9aec1f`.

A separate.736s CPU summary at19:39:29 reconciles completed controls' recorded
stage/update timings; it does not read the active trajectory or change its
selection. All169 inputs and the original PID/invocation remain unchanged.
Retain this co-hosting caveat without a training wall-time correction. See the
[cost diagnosis](2026-10-29-cdp-learning-diagnosis.md#recorded-learning-work-dominates-breakouts-training-cost).

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
