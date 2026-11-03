# Four actions target exploration work, not current greedy mistakes

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
are excluded. Actual training speed remains unmeasured. Parameter removal is
only21,518/16,334,353 (.132%).

The [bounded protocol](../experiments/2026-11-03-cdp-breakout-minimal-guided.md)
prepares a fresh8192-action cost check using the existing minimal-action adapter
and an explicit translation around the unchanged pixel teacher. Source tests
pass258 cases; repaired packaged command qualification passes161 cases with
zero agent construction and GPU descriptors. Native447fe50 remains unchanged.

The original command preflight accidentally constructs a real native agent
because lazy metadata loading replaces its mock. It stops before any emulator
step or update; its unguarded initialization, zero-update checkpoint and failure
are preserved. The repaired CPU check does not erase that incident. The intended
guarded check has a separate output directory; no full learner or gameplay gain
is selected. See the protocol for the exact boundary repair and limits.

Evidence: `/x/Temp/kindle-breakout-minimal-guided-20261103`, including
`action-review.json`, source snapshots, original failure and repaired preflight.
The five-game goal remains unfinished.
