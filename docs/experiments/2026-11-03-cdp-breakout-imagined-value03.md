# Breakout: bounded imagined-value weight0.3 check

Select one fresh bounded check after the actor/critic representation-gradient
comparison closes. That change lowers both frozen scores, so retain gradients
off. The current replay-value1 control improves gameplay over replay-value.3,
yet its sampled critic RMSE9.231 still exceeds the earlier-episode constant7.855.
Test a different balance between the two existing critic losses before any
full learner: only `loss_scales.value` changes1→.3; replay-value stays1.

The primary [NanoDreamer implementation](https://github.com/benyebai/NanoDreamer/blob/main/nanodreamer/agent.py)
weights its real-replay critic loss1 and imagined critic loss.3. Its architecture,
environment, replay sampling and targets differ; this motivates a coefficient
choice, not a matched baseline or an expectation of transfer. Pin the fetched
source (Git blob2b59b4ee9c5a91caffe35de2aba26fa38e59073a) with its config and
replay source. The blog page is unavailable through the web reader; the repository
is the inspected primary source. Do not substitute secondary summaries for code.

Native447fe50 already scales the imagined value objective separately from the
replay-value objective. This scales both imagined return fitting and its slow-value
regularizer. Actor loss, expert loss and direct replay-value representation
weight stay fixed. Kindle's replay lambda targets still use imagined-return
bootstraps; this does not replace them with complete observed returns or repair
a demonstrated bug. Downstream critic changes can alter actor targets and play.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/lambda.95/
discount333/encoder-base4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100,
unit expert action weights/world gradients/all-state corrective labels, sampled
collection/posteriors, actor-unimix.01, action-effects1, replay100k, warmup1000,
20% episode guidance/cap1920, seed1009/evaluation4000001009, full18/sticky.25/
repeat4/noops0/native pooled RGB/one GPU64 resize. Preserve raw rewards and
real boundaries. Native447fe50/SHA35ad0db and teacher remain unchanged; add
only a fresh-CDP `--imagined-value-scale` option to the retained runnercf4ea238.
Omission preserves defaults; reject nonpositive/nonfinite/F32-unrepresentable
values, evaluation overrides, restore overrides and non-CDP use on CPU before
native construction. Test exact resulting config and unchanged defaults.

Require8,192 actions/489 updates, first update376 and residual.5 within180s;
save actual initial/final weights and evaluate1,024 frozen greedy actions within
60s, guidance/labels off, zero updates. Verify exact config except the declared
loss field,357 control-matching initial and invariant frozen tensors,47 matching
pre-update semantic events, finite reports, changed CNN/reward/actor/value,
full guided pixel/action/label replay and existing host guards. Preserve the
357-tensor/16,334,353-parameter schema. Qualify actual commands, exact config,
cadence, schema and source paths on CPU before declaration and launch.

Use180s audit/480s main/600s original collector caps. No overlap,retry,extension,
cap increase or repeated successful stage. Conservative200k cost projection is
training seconds×200000/8192×1.10; require≤1800s before separately selecting
any full200k/12,477-update comparison. A later full learner must use fresh
experience,100k recovery and fixed final/actual-initial sampled/greedy24-natural-
episode arms against70.750/143.667, all0/24 two-wall, with the unchanged90% gate.
Do not transfer bounded experience or automatically launch a full learner.

Root `/x/Temp/kindle-breakout-imagined-value03-20261103`; runtime remains
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No native build, optimizer
change, teacher change, coefficient sweep or confirmation is selected. Reused
development seed/cohort and coupled imagined-return/slow-value weighting limit
mechanism and reliability claims. The five-game goal remains unfinished.

## Preparation and preserved fixture failure

The private runner adds12 lines and deletes none fromcf4ea238; candidate SHA
`76440d32d8792a800afe34fa68c45c0073f6d1d7103b609864f6046fac11bd4b`.
The inspected native source confirms separate imagined/replay value scales.
Fetched NanoDreamer source contents match their Git blob hashes. No native
implementation or build changes.

The first CLI fixture fails08:08:26.093 UTC November3 in13.203s: resolving
`kindle.default_config` lazily repopulates module exports and replaces an
earlier constructor stub. The retained-runner fixture therefore constructs
one native agent, then rejects its dummy frame before completing episode
initialization. No learning update or policy evaluation occurs. This is not
a CPU-only pass. Preserve the original source, receipt and kernel journal;
the captured interval contains resume messages but no explicit reset/fault/hang
line. No host recovery is performed and no successful experiment stage repeats.

The correction binds a separate explicit namespace proxy to both runner modules;
CPU config defaults may import the native extension, while both agent constructors
are blocked. It passes08:10:30.991 in.107s: ten invalid-command cases,six constructor
boundaries,two unchanged legacy modes and two coefficient cases. The actual
pre-update reader matches the previously qualified reader exactly, so its five
semantic fixtures are retained without repetition. The original failed receipt
and correction remain pinned.

Bounded preflight passes08:10:45.969 in1.030s (1.065s wrapper): both actual
commands,20 exact-config cases,first376/489/6,227/12,477 cadence with residual.5,
357 tensor shapes and338 retained control pins. No native module,agent or emulator
is constructed in this preflight. Native runtime and teacher remain unchanged.
Source generation succeeds once; its initial receipt beside the reused wrapper
is copied byte-for-byte into the new package without repeating generation.
