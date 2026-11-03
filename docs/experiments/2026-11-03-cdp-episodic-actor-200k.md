# Fresh200k comparison of completed-return actor learning

The [bounded pair](../results/2026-11-03-cdp-episodic-actor.md) establishes
active completed-return targets and4.36% additional early training cost,with
no gameplay result. Select one fresh200k-action bonus-off control and one fresh
200k-action bonus-off scale100 candidate, in that order, with no overlap or
experience/weight transfer. Reuse the qualified e3b02163 runtime and bounded
runner without rebuilding. No new coefficient or seed is selected after seeing
either full outcome.

Both arms retain the bounded recipe:12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/
R16/context1,encoder4e-5/behavior8e-5/dynamics4e-4,policy1,critic1/1,constant
actor/world expert100,all-state corrective labels,unchanged teacher,20%episode
overrides/cap1920,lambda.95/horizon333,unimix.01/entropy.0003,replay100k,warmup1000,
sampled collection/posteriors,full18/sticky.25/repeat4/noops0,native RGB and one
GPU64 resize. Model disagreement is off in both. Only the candidate adds
`replay_actor_scale=100`. All other optional objectives and crop remain off.

Each arm gets200k aggregate actions/12,477 updates and a30-minute training cap.
Retain an immutable100k/6,227 recovery and final checkpoint. Validate checkpoint
schema,finite arrays,counters and hashes; no warm-start extension or cap increase.
First update stays376. Require matched261 initial arrays,matched common
pre-update transitions and full sampled/greedy actual-initial controls.

After each training phase, freeze the final and actual-initial checkpoints for
24 natural episodes per mode: three per lane,eight lanes,seed4000001009. Training
seed1009 is reused development evidence. Keep the original per-episode frame
limit and record every truncation; a failed assessment is not silently retried
or replaced. Require zero updates and all261 frozen tensors unchanged,full
recorded-action replays/videos and original task accounting. A maximum600k
actions per frozen arm is an execution cap,not an invitation to select episodes.

Greedy is the primary development comparison; sampled is reported in parallel.
Compare absolute means,task successes and actual-initial-subtracted return per
training minute against the paired bonus-off control and retained action-effects1
70.750/143.667 in21m27.910s. Report all eight paired lane means and a fixed10,000-
resample lane-bootstrap interval with seed0. Do not change evaluation mode or
gate after observing results. The original Breakout mean864/90% two-wall gate
and five-game requirements remain unchanged. Preserve useful budget tradeoffs;
do not claim an isolated benefit versus the older runtime/action-effects1 arm.

For the candidate,report completed/positive replay-row exposure,weight sum,raw
loss and total-loss accounting throughout. Keep repeated replay exposure
separate from unique experience. Verify training pixel/action/teacher-label
replay,override counts,curves and later-score coverage. Teacher-guided online
returns are not frozen competence. Require host guards and exact source/runtime/
teacher/ROM pins. Validate each completed phase before the next GPU phase.

Prepare and pin controllers,queues,schemas,validation and deadlines before
launch. Preserve original failures; repair CPU analysis separately without
repeating successful GPU stages. Root
`/x/Temp/kindle-episodic-actor-20261103/full`. Check live state before launch.
No extension,confirmation,third learner or automatic successor follows this
pair. It tests one reused seed/cohort and cannot establish broad reliability.
The five-game goal remains unfinished.
