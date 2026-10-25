# Cheap Freeway development results

The first cheap configuration executes quickly but fails its frozen learning
test. The fresh, unassisted seed1009 uses centered CDP/action-effects1,
Size1M (1,057,201 trainable parameters), Auto, N8/B8/T16/H15, replay100k/R16,
full BPTT and microbatch8. Native max-pooled frames receive one GPU64 resize.
Full18 actions, sticky.25 and repeat4 remain unchanged.

| Allocation | Actions | Emulator frames | Updates | Training wall | Frozen final / initial | Task successes |
|---|---:|---:|---:|---:|---:|---:|
| Unassisted |200000|799992|24970|486.58s|0.00 /0.00|0/24|

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
isolated speedup nor establishes five-game reliability. The next short test
adds the historically used training action aid; see the amended declaration.
