# Fresh Qbert seed check for the cheaper passing recipe

The [completed greedy-training learner](../results/2026-11-01-cdp-qbert-greedy-training.md)
passes the primary gate with 15,515.625 / 24 first pyramids after 27m33s.
Its sampled mean is 11,989.583 / 24. All original audits, collection and the
independent terminal confirmation pass. This is reused development seed4013;
select one fresh learner to test whether the same recipe clears the unchanged
gate on another seed. This work belongs to the existing five-game goal.

Fix learner seed8039 and evaluation4000008039 before observing new outcomes.
A semantic scan of 61 retained Qbert declarations found neither seed in their
seed fields; this is a statement about the examined evidence. No prior weights,
replay, belief, game state, RNG or experience enters the learner. There is no
sampled-training control for8039: this checks the recipe across seeds, not an
isolated replication of the greedy-collection improvement.

Keep centered CDP, 12M48, Auto precision, N8/B16/T16/H15/R16, encoder/base/actor
rates 4e-5, dynamics rate 4e-4, expert scale100, all 18 action weights1,
world imitation gradients, all-state corrective labels, 20% episode continuation
guidance capped at1920 actions, actor-unimix0, full18 actions, sticky probability
.25 and repeat4. Keep max-pooled native frames and one GPU resize to48.
Live training actions are greedy; categorical posteriors and imagined actor
sampling remain unchanged. Only learner/evaluation seeds change from4013.

Use native447fe50 and runtime
`/x/Temp/kindle-breakout-corrective-20261028/runtime`, with the exact qualified
runner SHA256 `aba4300d0cd1a7d1a8d2d0c71de4a0cad90ec84b2019060fef4e6ef2db501a13`.
Model, losses, teacher and preprocessing are unchanged. The completed identical
recipe supplies execution and cost evidence; no new build or repeated bounded
GPU qualification is needed.

Train 200,000 aggregate actions / 12,477 updates under2100s (35 minutes).
Require first update376, residual credit.5 and a verified100k /6227 recovery.
No checkpoint selection, cap increase, retry, extension, seed campaign or
automatic successor follows. Assess the fixed final and actual-initial
checkpoints with sampled and greedy actions: first three natural episodes in
each of eight lanes, 24 per arm. Guidance/labels off, zero updates/truncations.
Greedy remains primary; sampled results remain separate. Preserve excess
episodes and partial tails. Each arm has300s and the existing600k-action ceiling.
The original mean15,000 /90%-first-pyramid gate remains unchanged.

Report both modes, own-initial-subtracted gain per training minute, first-pyramid
counts, actual guidance exposure and total cost alongside4013. Two learners
cannot establish broad reliability. Aided online curves and replay fit are not
frozen competence. Preserve CPU co-host timing without wall-time correction.

Require357 finite initial tensors with the same schema as4013 and record
seed-dependent differences. Do not require initial values or semantic trajectories
to match across seeds. Each frozen arm must preserve all357 tensors. Verify
exact configuration, native/runner/teacher hashes, counters, finite reports,
CNN/reward/actor/value changes, raw expert losses and labeled-row weights.
Complete training pixel/action/label replay, frozen outcome replays/videos,
host guards, curves, coverage, exposure and comparison. Ordinary frozen logs
lack live image hashes; their replay establishes recorded outcomes, not direct
live-image equality.

The CPU preflight passes in2.179s: five actual CLI paths,26 configuration cases,
eight training-mode cases,13 startup-header cases,11 recovery cases,four observer
parser cases,49 paths,15 compiled scripts,357 schema tensors and all305 bounded /
317 preceding full-run pins. No native model or emulator is constructed.
Fresh-seed command, header and recovery checks reject4013 where8039 is required.
The comparison and startup audits deliberately record initial differences instead
of requiring same-seed equality. Exact source provenance is retained privately.

Launch only after the original4013 services and terminal confirmation pass,
all pinned inputs remain unchanged and no Kindle GPU allocation is active.
The training replay has1200s, enclosing audit1800s and whole main service5400s.
Progress, recovery, postprocessing and terminal-collection observers have bounded
deadlines and never start or restart GPU stages. Preserve failures; a CPU repair
must not repeat completed training, frozen assessment, replay or plotting.

Root `/x/Temp/kindle-qbert-greedy-seed8039-20261101/full`, trial
`qbert-12m-48-b16-r16-encoder4e5-greedy-expert100-200k-seed8039`, service
`kindle-qbert-greedy-seed8039-200k-20261101`. At protocol selection, no8039
learner has launched and no result exists. Inspect live state before work;
do not schedule another GPU job while this learner is active.
