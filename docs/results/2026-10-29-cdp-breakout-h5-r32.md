# Breakout H5/R32 passes its bounded execution and cost check

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
