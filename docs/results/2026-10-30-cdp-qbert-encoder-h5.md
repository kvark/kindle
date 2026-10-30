# Qbert H5/R32 with encoder2e-5 passes the bounded check

The fresh 12M48/Auto/N8/B16/T16/H5/R32/expert100 check completes 8,192 aggregate
actions / 32,751 emulator frames / 978 updates in **95.153s**. Its 1,024-action
frozen restore takes **.798s**, with 4,096 frames, zero updates and no natural
completed episodes. The conservative 200k projection is **42m35s**, below the
45-minute gate. This establishes execution and early cost; gameplay benefit
from combining H5/R32 with the moderate encoder rate remains unmeasured.

The [protocol](../experiments/2026-10-30-cdp-qbert-encoder-h5.md) combines two
separately useful Qbert changes. Relative to the retained H5/R32 control, only
encoder6e-6→2e-5 changes as a learner setting. Native447fe50, the Qbert teacher,
the qualified optional-rate runner and the 14,123,473-parameter architecture
remain unchanged. The runner's Breakout corrective-label plumbing is disabled
here; its exact diff against the older H5 control remains retained. Initial
matching does not prove complete cross-version learning parity.

Seed4013/evaluation4000004013 are reused development inputs. No weights,
replay, games, belief, RNG or prior check experience are reused. Relative to
the cheaper H15/R16/encoder2e-5 point, horizon, sampling, update cadence,
warmup duration and total work change together. No isolated horizon effect,
reliable combination benefit or full learner is established by this check.

All **288 declared inputs**, 357 control-matching finite initial tensors,
47 pre-update semantic events, 357 invariant frozen tensors, exact config and
cadence, finite reports, changed CNN/reward/actor/value, full training
pixel/action/corrective-label replay and both host guards pass. First update
is376, then every8 actions, with zero residual credit. All978 reports have
supervision; 234,563 labeled replay rows equal their weight sum. Maximum raw
world/actor expert-loss difference is8e-7. There are6,411 NOOP and1,781 hop
labels,3,702 passive labels and2,588 actual/teacher disagreements. Actual
4,490 overrides are54.810% of actions; guided returns are not frozen competence.

CPU preflight passes in.855s: two actual command paths,14 config cases,six
collector accounting/cost cases,all229 H5 control and235 completed moderate-rate
pins,357 schema tensors and the978/12,454/24,954 schedule. No native module,
agent or environment is constructed during preflight. No new build is needed.

The service starts **02:30:36 UTC October30**, source `8b9936a`, PID378268,
invocation `cac67bba1731491f9bb9b8518034bd1c`. It succeeds **02:32:47** after
2m11.460s; terminal collection succeeds **02:33:13**. Independent confirmation
rechecks all288 pins and five completion artifacts. No failure, GPU repetition
or host recovery occurs. Brief read-only status inspection co-hosts with the
check; no wall-time correction or isolated backend-speed claim follows.

Root `/x/Temp/kindle-qbert-encoder-h5-20261030/native-check`, trial
`qbert-12m-48-b16-h5-r32-encoder2e5-expert100-check`. All work is terminal and
no learner or GPU allocation remains active at closure. A full comparison must
be selected separately with fresh experience and fixed final/actual-initial
evaluations. No retry, cap increase, extension, rate sweep or automatic
promotion follows this bounded declaration. The five-game goal is unfinished.

| Artifact | SHA256 |
|---|---|
| `declaration.json` | `47e794bc8173ac21dc56fe3df4f72e53a8b0cd10eb0e4dfa6118b23b6162ee40` |
| `review.json` | `f56588b6098933d6b649e30666dcc9d6bc5f8e216c5f0a55bf04346e6eb8023d` |
| `guidance-replay.json` | `e38d17390c81502ba422f4ea41ffe3ebb71752ff3445b2584f4029025af5e605` |
| `completion-review.json` | `7b03856705eb8695b44c1e509ba4e2f61e4d9bc0b319e7e7c95d4d86bdcc75df` |
