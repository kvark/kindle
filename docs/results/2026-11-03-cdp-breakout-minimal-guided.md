# Four-action guided Breakout trains faster but scores lower

The fresh200k comparison cuts training time **13.29%**, but greedy frozen score
falls **36.31%** and initial-subtracted return per minute falls **29.20%**.
**Retain the full18-action recipe.** Four actions fail both the declared primary
replacement rule and the separate cheaper-budget rule. The original mean864 /
90%-two-wall gate and five-game goal remain unfinished.

## Full 200k comparison

| Measurement | Retained full18 | Guided minimal4 |
| --- | ---: | ---: |
| Training time |21m27.910s|18m36.686s|
| Native construction, recorded separately |2.404s|2.321s|
| Learning stage |1133.927s|965.508s|
| Sampled frozen mean |70.750|55.958|
| Greedy frozen mean, primary |143.667|91.500|
| Actual-initial sampled / greedy |1.417 /2.208|1.250 /4.667|
| Sampled initial-subtracted return/min |3.230|2.940|
| Greedy initial-subtracted return/min |6.590|4.666|
| Two-wall successes, either final mode |0/24|0/24|

The paired lane-bootstrap greedy return difference is−52.167,95% interval
[−94.708,−16.499]; the gain/min difference is−1.925 [−3.907,−.192]. Both favor the
retained full18 recipe on this reused evaluation cohort. These10,000 resamples,
seed0, quantify lane variation, not training-seed reliability. Sampled return
falls20.91%; sampled gain/min falls8.99%, with difference−.291 [−.852,.362].
The faster candidate therefore fails the separately fixed cheaper-budget rule
as well as the primary replacement rule. No confirmation or extension follows.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-minimal-guided-200k.md)
keeps12M64/Auto/N8/B16/T16/H15/R16/context1,encoder4e-5,dynamics4e-4,behavior8e-5,
policy1/critic1/replay-value1,disagreement1,expert100 andsampled collection.
It starts fresh atseed1009/evaluation4000001009,with no transferred experience.
The full18 teacher is unchanged internally; the explicit adapter translates
NOOP/FIRE/RIGHT/LEFT. Native447fe50 is unchanged. Action dimensions,
initialization, uniform mixture and action-effect centering change together;
this does not isolate an output-layer or exploration mechanism. Actual initial
scores differ, so each learner uses its own control in gain/min accounting.

The learner completes200,000 actions /799,580 emulator frames /12,477 updates
inside the30min training cap. Most measured saving lies in the learning stage;
acting62.578s andemulation62.662s are close to retained63.126s/62.988s. Timings
compare one fresh candidate with a historical control,not a paired kernel
benchmark. Parameters fall only21,518 (.132%) to16,312,835. CPU observer time
remains in training wall time; no correction is applied.

All169 current and338 retained inputs verify,with the declared exact historical
replay-source resolution. All357 actual-initial tensors match the bounded
minimal4 check and47 pre-update semantic events match. Full18 initial equality
is explicitly not claimed. The read-only100k/6,227 recovery passes independently,
including weights and moments; it does not preserve replay,games,beliefs orall
RNG state for exact continuation. CNN,reward,actor andvalue parameters change.

Full training pixel/action/label replay passes. Each of four frozen arms selects
the first three natural episodes per eight lanes,24 total,with no truncations,
zero updates andall357 tensors unchanged. Full recorded-action replays,videos,
host guards,curves,exposure andcomparison pass. The four arms total83,296 actions /
333,044 frames /62.439s. The online curve is visually checked and remains clearly
labeled as online rather than frozen performance.

Actual overrides total84,232/200,000 (42.116%); all12,477 updates include expert
labels,with2,989,756 repeated labeled replay rows. Late label CE is.567343 versus
retained.560514. The candidate reaches a maximum training score406 andhas only
1,222 native actions at pre-action scores216–431,with none at432+. These are
changing-policy exposure and fitting counts; neither they nor aided returns
identify a causal architecture limitation or establish unaided competence.

The original main succeeds **20:59:55.332 UTC November3**, launchsource`1917a4de`,
PID1067864/invocation`62b85df574b7486e961cd488cd3c9137`. The original terminal
collector succeeds **21:00:26.026** in2.408s. Main,recovery andcollector all
succeed; no full-run stage fails,repeats orchanges its cap. The bounded preflight
andCPU collector failures below remain preserved. No Kindle allocation remains
active atclosure; check live state before new work. No automatic successor is
selected. Evidence is under`/x/Temp/kindle-breakout-minimal-guided-20261103/full`.
The fixed protocol bytes andall original declarations remain unchanged.

| Full-run artifact | SHA256 |
| --- | --- |
| Declaration |`29952d0284f3bd25e7150547bb158f613fddef99f9c88340a47c10b48c4e82ac`|
| Learning review |`277c4afcd91f94f9fbeceb1eb940c7eb4e63cf157e33325fb2040872bbf3797b`|
| Training replay |`4cff05ae689d5dfaa9e657a84e306b79a712576f4e464f0f53082178296fb2cb`|
| Comparison |`02b0af2872e8653d7d30b2931badb0f86940b8a98295f59b3a258d9c5e4f0317`|
| Main completion |`36aaa1aa9df2265bb6e8477f5cffb06837d21974a28c46d7d437c99fe5388d57`|
| Terminal closure |`df923d19776453a485638f67b3a7eda92af5dd9bd3f6ce6caafbd6ac85c19b9b`|

## Bounded execution and cost check

The fresh four-action check completes **8192 actions/32,728 frames/489 updates
in41.133s**, versus47.690s for the closed full18 check: **13.75% less training
time**. Its1.1-padded200k projection is18m24.645s, below30min. This short check established
execution and early cost; the full comparison above addresses gameplay.

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
guarded check uses the separate `bounded-check` directory and passes. That
qualification did not establish gameplay benefit; the full comparison above
closes the separately selected learner. See the protocol for the boundary repair.

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
