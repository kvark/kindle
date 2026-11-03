# Four-action guided Breakout cuts early training time by13.75%

A separately selected [fresh200k comparison](../experiments/2026-11-03-cdp-breakout-minimal-guided-200k.md)
launches20:37:08 UTC/source1917a4de after closure of the bounded work below.
Startup passes357 tensors/47 events; recovery andfour frozen arms remain pending.
No full gameplay result isavailable. Do not restart oroverlap the active learner.

The fresh four-action check completes **8192 actions/32,728 frames/489 updates
in41.133s**, versus47.690s for the closed full18 check: **13.75% less training
time**. Its1.1-padded200k projection is18m24.645s, below30min. This establishes
execution and early cost; no full learner or gameplay improvement is selected.

| Clock | Retained full18 check | Guided minimal4 check |
| --- | ---: | ---: |
| Logged training, including final checkpoint |47.690s|41.133s|
| Learning stage |40.495s|34.621s|
| Native construction, recorded separately |2.404s|2.939s|
| Training plus construction |50.094s|44.072s|

The last row is12.02% lower; it is not the entire service wall time. The main
service takes57.950s including guarded stages and CPU replay/audit. This is one
short comparison against a retained measurement, with different trajectories
and possible cache effects from the preserved preflight incident below. It
does not isolate a kernel speedup or predict final score per minute.

All118 declared inputs, exact configuration/cadence,357 finite initial/final
tensors and357 invariant frozen tensors pass. CNN, reward, actor and value
weights change. All489 reports contain supervision:117,046 repeated labeled
rows;3,740/8,192 actions are actually overridden. Full32,728-frame training
pixel/action/label replay passes, including21 natural episodes. The1024-action
frozen restore takes.918s/4,081 frames withzero updates andfive natural episodes;
this short assessment is not the declared24-episode competence test.

The original service succeeds **20:21:17.038 UTC November3**, source`11023a19`,
PID1067123/invocation`a6ee66bf44214bfebfb9a607a20fd9ef`. Both guarded stages
run once and pass. The original CPU terminal collector fails because systemd
clears `InvocationID` after termination. A separate.135s CPU repair verifies
the original controller's invocation in the journal and closes **20:22:44.473**
as `complete_with_cpu_terminal_repair`. The original failure remains; no learner,
frozen stage, replay or main audit repeats. No Kindle allocation remains active.

## Why test four actions here?

The retained Breakout policy already restricts its greedy choices to NOOP,
FIRE, RIGHT and LEFT. None of43,424 recorded greedy actions, including all
36,321 actions in the selected24 natural episodes, uses another output.
Removing the other outputs would preserve those recorded greedy decisions.
Fresh four-action learning and sampled trajectories remain different questions.

| Retained trajectory | Recorded actions | Outside the four-action vocabulary |
| --- | ---: | ---: |
| Guided training |200,000|3,516 (1.758%)|
| Sampled frozen, all recorded |35,552|297 (.835%)|
| Sampled frozen, selected24 episodes |30,567|258 (.844%)|
| Greedy frozen, selected24 episodes |36,321|0|

These are vocabulary counts, not ineffective-action rates: the excluded Atari
combinations are not assumed equivalent to NOOP. Training contains86,117 actual
overrides. Logs, episode lengths, terminal counters and saved review endpoints
reconcile; twelve inputs remain unchanged. The.747s CPU review adds no model
forward, emulator replay or learner update. Retained scores remain70.750 sampled
and143.667 greedy, each0/24 two-wall.

The native disagreement bonus evaluates every action through four prediction
heads before centering the selected action's effect. For the saved12M geometry,
the matrix multiply-add count per state falls8,388,608→4,718,592 when18 actions
become four. This43.75% reduction applies only to those matrix operations;
normalization, activations, reductions, launch cost and all other learner work
are excluded. Measured early cost is reported above. Parameter removal is
only21,518/16,334,353 (.132%).

The [bounded protocol](../experiments/2026-11-03-cdp-breakout-minimal-guided.md)
selects the completed8192-action cost check using the existing minimal-action adapter
and an explicit translation around the unchanged pixel teacher. Source tests
pass258 cases; repaired packaged command qualification passes161 cases with
zero agent construction and GPU descriptors. Native447fe50 remains unchanged.

The original command preflight accidentally constructs a real native agent
because lazy metadata loading replaces its mock. It stops before any emulator
step or update; its unguarded initialization, zero-update checkpoint and failure
are preserved. The repaired CPU check does not erase that incident. The intended
guarded check uses the separate `bounded-check` directory and passes. No full
learner or gameplay gain is selected. See the protocol for the boundary repair.

Evidence: `/x/Temp/kindle-breakout-minimal-guided-20261103`, including
`action-review.json`, source snapshots, original failure and repaired preflight.
The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`c28ae2645ade84309957563a616361e26e87f13cf5f7c63044efc162eaf1baf8`|
| Bounded review |`d7dfa57d5aecb42ff71547a2ccf35281bb9d060bea30a2b0113a61adf7008496`|
| Main completion |`3b645538906782ce14c432724a55511ff9666be3ea3c79e755d29871564b6618`|
| Training replay |`a707908dd89edd4b286c6e2de4bb8c77f8520335d96c392933997fb6980c5da5`|
| Repaired terminal closure |`b01ebd935ef1293f101f57b51466fc1e57b4211f29e866ac32469d69f26d2a30`|
