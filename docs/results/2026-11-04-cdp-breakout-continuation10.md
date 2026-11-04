# Stronger continuation supervision passes execution and early cost checks

The fresh continuation10 check completes **8,192 actions /32,724 frames /
489 updates in46.764s**. A conservative200k projection is **20m55.877s**, below
the30-minute limit. Frozen restore completes1,024 actions in.872s with zero
updates and all357 tensors unchanged. This establishes execution and early cost;
gameplay benefit and terminal calibration remain unmeasured.

The [fixed bounded protocol](../experiments/2026-11-04-cdp-breakout-continuation10.md)
raises only existing world continuation BCE scale1→10. Both terminal and
nonterminal targets, including their shared-world gradients, receive the higher
weight. It is not class balancing, and optimizer adaptation/clipping prevents
inferring a tenfold parameter step. Intermediate life losses remain nonterminal.
The earlier H3 supervision regression is retained; H3 stays off.

Native447fe50/SHA35ad0db, the corrective runtime, CDP12M64/Auto/N8/B16/T16/
H15/R16/replay-value1/expert100, rates, teacher,20% episode guidance/cap1920 and
all-state labels remain fixed. The new `--continuation-scale` option is for fresh
training only; omitted defaults and restored configurations remain unchanged.
The qualified private runner differs from cf4ea238 by exactly ten added lines,
matching the canonical change. No native build or runtime overlay occurs.

Both canonical and private runners pass **34 focused CPU checks**, including
invalid F32 inputs, restore/evaluation rejection, omitted/explicit defaults and
exact requested-config changes. Actual training and restore commands stop at
mocked construction boundaries; schema, replay argv, cadence, loss-sum rejection
cases and the original terminal collector also pass CPU preflight.

All **699 current pins and338 retained pins** pass, with one exact historical
replay-source resolution. All357 initial arrays and47 pre-update semantic events
match the retained control. Every489 report is finite and its weighted world
loss includes continuation10; independent recount has maximum absolute sum error
.000113436. CNN, reward, continuation hidden/output, actor and value parameters
change. Cadence starts at376 actions and ends with residual credit.5.

Full training pixel/action/label replay passes, including22 completed episodes
and3,733 actual teacher overrides. Guided returns are not unaided competence.
Both GPU stages and host guards pass without recovery. Historical bounded
control time is47.689684s; the candidate is1.94% lower, but this single historical
timing comparison does not establish a speedup. Learner wrapper time is40.413s;
these timers do not isolate GPU kernels.

Source`d5777b14`, original mainPID1103277/invocation
`1bf8e5c290a64498bffb1419675d2326` starts03:41:11 UTC November4. Main and original
collector succeed **03:42:20.787 /03:42:24.382**. Independent **5.398s CPU closure**
completes **03:43:00.161**, recounts all489 reports and47 semantic events, rechecks
all initial/frozen tensors, original service identities, replays and guards.
All84 final artifact hashes verify. No experiment preparation, GPU, audit or
collector failure, repair, cap change or completed-stage repeat occurs.

Root `/x/Temp/kindle-breakout-continuation10-20261104` is closed. Do not repeat
its training, assessment, replay or collectors, or transfer its experience.
A [separate fresh200k comparison](../experiments/2026-11-04-cdp-breakout-continuation10-200k.md)
is selected because execution and projected cost pass. Its actual state must be
checked before GPU work. Retain70.750 sampled/143.667 greedy until full results;
the original864/90%-two-wall and five-game goal remain unfinished.

| Artifact | SHA256 |
|---|---|
| Bounded declaration |`f45310b5855fabe60f17857e8edac981f762f0ed8e347b401e31490765f1e1a2`|
| Bounded review |`c2215977ddfafd74ec230bc7b659f1472e88560d537082cd562b028213743d9e`|
| Full training replay |`e8e7596db03d375d0f2bbf89b0a41045550008c7d506e8892a75b8eeab0304b8`|
| Main completion |`82cc134ed53b0fe99282615ada73b65f235edea387ca9e8ba1435831fe3dd765`|
| Independent closure |`32d6c88dfc3a919829dd4b741bd23085037a30a686cf69265c60b7dd9abbc758`|
| Qualified private runner |`74af89412b43a7521cc0f7d9b7a8c5f9e715534545f0fecf60110778f1e251cb`|
