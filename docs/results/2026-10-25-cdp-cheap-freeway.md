# Cheap Freeway development results

The compact RGB32/R32 recipe passes on two learners after500k actions each:
**31.042/32.000 frozen crossings,24/24 task successes each, in28m17s/28m50s**.
Both actual-initial controls score0.00. The second seed keeps the selected recipe
and budget fixed. All audits pass. This is useful focused confirmation, not
broad seed reliability. The two earlier RGB64/R16 screens below fail.

## Short RGB64 screens

The first cheap configuration executes quickly but fails its frozen learning
test. The fresh, unassisted seed1009 uses centered CDP/action-effects1,
Size1M (1,057,201 trainable parameters), Auto, N8/B8/T16/H15, replay100k/R16,
full BPTT and microbatch8. Native max-pooled frames receive one GPU64 resize.
Full18 actions, sticky.25 and repeat4 remain unchanged.

| Allocation | Actions | Emulator frames | Updates | Training wall | Frozen final / initial | Task successes |
|---|---:|---:|---:|---:|---:|---:|
| Unassisted |200000|799992|24970|486.58s|0.00 /0.00|0/24|
| Training action aid |200000|799992|24970|491.09s|0.25 /0.00|0/24|

Training averages411.03 aggregate actions/s, with first update at248 and zero
debt. It sees one crossing in96 completed episodes; the last50-episode mean is
zero. Final and actual-initial frozen arms each complete24 natural episodes,
49152 actions and196600 emulator frames, with zero learner updates. The existing
Freeway gate (mean at least25, success fraction at least.9, no truncations)
fails. Neither fast execution nor finite losses demonstrate useful learning.

All24970 reports are strictly finite. Configuration, counters, exact initial
identity, all346 frozen tensor equalities, full CPU action replay, videos and
all host guards pass. The service exits successfully. The action/frame/time
curve independently reconciles completed episodes and updates. GPU telemetry
has18 valid samples (median73% across training and frozen jobs) and one initial
sysfs read unavailable with EBUSY; this is not a training-only utilization figure.

Native source`f4bb2ca`, SHA256
`6cb2677f2bc33f0aaced090ed7b6f328533057893672434c6b30c0ef08dec7fd`,
RX7900XT/RADV, runner`769bdef`. Full evidence, declarations, checkpoints and
curve artifacts: `/x/Temp/kindle-cheap-20261025/freeway-1m-200k`.
Review SHA256`193a56c81d46ef291086abd5a3e5496bdae03ad269f6433ad7d2cec2aead1c83`.

This is one reused development seed, with capacity, sequence geometry and
replay ratio changed from the earlier12M/T64 work. It neither measures an
isolated speedup nor establishes five-game reliability. The follow-up adds the
historically used training action aid; see the amended declaration. Neither
short configuration passes its frozen learning gate.

## Aided comparison

The only recipe change is5% persistent-uniform16-action blocks during training,
disabled in frozen evaluation. This fresh run reuses seed1009 after its failed
unassisted result. All346 actual-initial tensors, full learner config and native
library match the unassisted run. It finishes in491.09s (8m11s),407.26 aggregate
actions/s, with zero debt. The aid ledger verifies10048 overridden actions.

Learning improves late:27 crossings occur in96 completed training episodes,
with three more in partial tails;13 completed episodes have positive returns.
The last50-episode mean is0.50. Frozen evaluation retains only a small gain:
mean0.25, six of24 episodes cross once, and0/24 reach the25-crossing task
milestone. Actual initial remains0.00. Both frozen arms again complete49152
actions/196600 emulator frames, without truncation or learner updates.

All24970 finite reports, action-aid accounting, checkpoint/counter/config checks,
exact frozen tensors, full CPU replays/videos and host guards pass. The service
exits successfully. Artifacts and independently reconciled action/frame/time
curves: `/x/Temp/kindle-cheap-20261025/freeway-1m-200k-aided`.
The action aid was selected on a reused development seed. This is a small gain
at a cheap budget, not consistent crossing behavior or seed reliability. No
extension followed this screen; the later compact recipe is reported below.

## Compact RGB32 with the successful update budget

The [declared fresh test](../experiments/2026-10-26-cdp-cheap-freeway-update-budget.md)
keeps seed1009 and the same training aid, while changing RGB64→32, replay ratio
16→32, actions200k→500k and the qualified native runtime. Size1M now has 834,673
parameters. Centered CDP/action-effects1, Auto, N8/B8/T16/H15, full BPTT/microbatch8,
replay100k, full18/sticky.25/repeat4 and one GPU resize from native frames remain.
Native actor_unimix is0. All346 initial tensors match the same-seed RGB32 smoke;
this run retains its own actual-initial frozen control.

Training completes **500,000 aggregate actions /1,999,960 emulator frames /
124,939 updates in 1,697.484 seconds (28m17s)**,294.55 actions/s. First update248,
zero debt,240 natural training episodes, final online last50 mean26.62. The aid
ledger verifies24,756 overridden actions. The30-minute cap is unchanged; the
250k recovery checkpoint passes its finite/config/counter audit and is replaced
by the fixed500k final. No checkpoint selection or extension occurs.

Frozen evaluation disables the aid and selects the first three natural episodes
per lane,24 per arm, using seed4000001009. Final mean **31.0417**, range29–32,
with **24/24 episodes reaching25 crossings**, versus initial0.00 and0/24. The
existing mean≥25/90%-success gate passes. Each arm uses49,152 actions and196,600
emulator frames; final/initial wall time is22.363/22.263s. There are no excess
episodes, nonempty partial tails, truncations or learner updates. All346 tensors
remain unchanged in each frozen export.

All124,939 numeric reports, complete configuration/source/counters, initial and
final checkpoints, aid accounting, full CPU replays/videos and three host guards
pass. The service exits successfully. Independently reconciled action/frame/time
curves and a three-recipe comparison by actions, updates and wall time are
retained. Resolution, ratio, experience budget and native runtime change together;
the result does not isolate the benefit of smaller images or more updates.

A separate CPU reference always chooses UP under the same evaluation protocol
and environment seeds. It averages **21.6667**, range21–23, with0/24 task
successes, using49,152 actions/196,600 frames in16.291s. It constructs no agent and
performs no learning. The trained score exceeds this simple fixed policy; that
alone does not establish visual reactivity or explain the learned strategy.

Source at launch`541c90b`, native`bfb8c31`, runner`bcef410`, RX7900XT/RADV.
Native SHA256`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Artifacts: `/x/Temp/kindle-cheap-20261025/freeway-1m-32-500k-r32-aided`, including
`learning-review.json`, `curve/`, `budget-comparison/` and fixed checkpoints.
Review SHA256`4f72aa5686df726ac32d02f1d591f2c762fc3f539bb269219b0e1df6257f0036`.
The CPU reference is in sibling`freeway-always-up-reference`; review SHA256
`fd85fab155917f17b87c8c3b564f102bb31f1f1a1e1c31c41d18e6740c395b14`.

This establishes a useful compact aided-training recipe on one reused development
seed. The focused confirmation below tests the selected recipe on another learner.

## Fixed-recipe seed4013 confirmation

The [declared confirmation](../experiments/2026-10-26-cdp-compact-freeway-confirmation.md)
changes learner/exploration seed1009→4013 and evaluation seed4000001009→4000004013.
The complete native configuration otherwise matches:1M/RGB32/Auto,834,673
parameters, N8/B8/T16/H15/R32, centered CDP/action-effects1 and actor_unimix0.
The5% persistent-uniform16-action aid remains training-only. The CLI adds an
optional imagination-length argument; this run omits it and retains15.

Training completes **500,000 aggregate actions /1,999,960 emulator frames /
124,939 updates in1,730.358 seconds (28m50s)**,288.96 actions/s, within the same
1800-second cap. First update248, zero debt,240 natural training episodes and
final online last50 mean28.94. The aid ledger verifies24,656 overrides. The250k
recovery checkpoint passes its finite/config/counter audit before fixed500k
replaces it. There is no favorable-checkpoint selection or extension.

Unaided frozen final scores **32.000**, range29–33, with **24/24 episodes reaching
25 crossings**, versus actual initial0.00 and0/24. The existing gate passes.
Each arm selects all24 natural episodes,49,152 actions/196,600 frames;
final/initial time26.648/26.839s. No excess episodes, nonempty partial tails,
truncations or updates occur. All346 tensors remain unchanged per arm.

All124,939 numeric reports, full source/config/counters, aid accounting, finite
checkpoints, full CPU replays/videos and three host guards pass. All346 initial
tensors match the same-config Boxing4013 initial checkpoint; this is an
initialization check, not pretraining or a reused Freeway evaluation. The service
exits successfully. Independently reconciled action/frame/time curves and a
two-seed comparison are retained. Each seed's online and frozen outcomes remain
separate;48 evaluation episodes are not48 independent learners.

Source at launch`1ed07af`, native`bfb8c31`, runner`1ed07af`, same RX7900XT/RADV
and native SHA256 as the primary run. Artifacts:
`/x/Temp/kindle-cheap-20261025/freeway-1m-32-500k-r32-aided-seed4013`.
Review SHA256`140815f4660d11b621d407fcb12c446def3842f5e1d0da599731cfa6987d3344`.
The fixed compact recipe now succeeds on two learners within30 minutes each.
That supports moving the next learning allocation to an unresolved game;
broader reliability and five-game completion remain open.
