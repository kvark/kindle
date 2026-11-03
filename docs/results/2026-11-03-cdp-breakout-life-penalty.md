# Optional Breakout life-loss feedback

The separate Python overlay passes CPU qualification. It subtracts one training
point per observed net lost life while preserving raw rewards, game scores,
actual termination/truncation flags and policy observations. Frozen evaluation
refuses shaping. The aid defaults off; its gameplay benefit is unmeasured.
The [bounded protocol](../experiments/2026-11-03-cdp-breakout-life-penalty.md)
selects one fresh 8,192-action check after qualification, not a full learner.

The runner derives from retained `cf4ea238`; only the declared reward handling
changes. Native `447fe50` / SHA `35ad0db` is copied byte-for-byte, with no build
or native implementation change. The retained runtime, teacher and all 338
control pins remain unchanged. The new runtime is
`/x/Temp/kindle-breakout-life-penalty-20261103/runtime`; its source and packaging
receipt are retained beside it. Do not edit the retained control runtime.

All **57 CPU tests** pass at **06:00:51.313 UTC November 3**, source `c6341db`,
in 2.102s including pin checks (2.145s wrapper). They exercise the actual runner
with synthetic environments and an agent stub, including separate raw/training
reward accounting, loss/gain/multiple-loss/game-over/truncation/reset cases,
F32 conversion, malformed input rejection, actual CLI refusals, and exact
default-off semantic agreement with the retained runner. The changed auditor
rejects corrupt reward contracts and stored rewards. The independent replay
reader queries environment life counts and separately computes penalties;
its synthetic tests reject altered initial/transition/reset counts and rewards.
All 365 qualification pins, including the 338 control pins, remain unchanged.
No native module, actual emulator or GPU is used in these tests.

Bounded preflight passes at **06:03:31.852 UTC**, in 1.972s (2.005s wrapper):
both actual commands, nine native-config assertions, five pre-update comparison
cases, expected 376-action first update / 489 updates / .5 residual credit, and
the 357-tensor initial schema. Pre-update comparison preserves raw environment,
action, teacher and intrinsic-reward fields while normalizing only the declared
extrinsic training change. The full auditor checks that training change.

Two CPU failures remain preserved. The first synthetic fixture retained expert
imitation without declaring guidance; the ordinary auditor correctly rejected
it. The corrected fixture disables imitation in that synthetic case. The first
bounded preflight omitted `.safetensors` from a schema-component filename; only
that reader was repaired. Production reward, runner, auditor and replay sources
are identical before and after both repairs. No GPU stage has failed or repeated.

At this qualification milestone, no native check or full learner has run. The
next bounded check keeps 12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/
behavior8e-5/dynamics4e-4/replay-value1/expert100, sampled collection and replay100k.
Require real ALE life/reward replay, finite training, unchanged frozen weights
and the fixed cost gate before any full-run selection. Retained frozen scores
70.750 sampled / 143.667 greedy and zero two-wall successes remain unchanged.

Evidence root `/x/Temp/kindle-breakout-life-penalty-20261103`:

| Artifact | SHA256 |
| --- | --- |
| CPU review | `b97fdbb00091ed394fc06493809616c50058c338beafd5178be96450d02a3a90` |
| Runtime receipt | `cc257a731477f337fbada8f4e6bc380e29efc26b9cf25e6b053aa3e83784d662` |
| Bounded preflight | `ce0818216b92f76d97883f6413014b98974a14642c21ce18fbeaa60eddd1873c` |
| Runner | `9ff2552b1c59fe7e45db156d9bba571737d3a3eb34d2ac073136f2bf92ce244a` |
| Reward helper | `e61a067c8ca6d9df400788a2ae25ec5d2bac69eadc5199f31186d98e2dc5b76a` |
| Vector auditor | `47a3a4ee206eba485ae8e6999b50204615fe9a45ee2bb6f0df93eac45cee0d20` |
| Independent replay reader | `cac4c9a91f34c23791d343ef67fc8acada0df879b08fbef52668555cd2677ae0` |
