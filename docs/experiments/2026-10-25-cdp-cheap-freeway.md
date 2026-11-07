# Cheap Freeway development trial

The user authorizes game-specific recipes and prioritizes useful learning per
GPU-hour. Historical1M Freeway frozen mean30.26 supports trying small capacity;
it is not a matched AMD control. The completed12M/B2/T64 Breakout precision
comparison remains weak and is not extended.

One fresh seed1009, centered CDP/action-effects1, Size1M, Auto precision,
N8/B8/T16/H15, full BPTT/microbatch8, replay100k and ratio16. Full18 actions,
sticky.25, repeat4, native max-pooled frames and one GPU64 resize. No action aid.
The actual trainable count is1057201 parameters. Only one learner runs.

Budget200000 aggregate actions, expected24970 updates, first at248. Training
has a1800-second host-guard cap, including construction. Save the actual initial
and fixed final checkpoints; no best-checkpoint selection or automatic extension.
A timeout remains a failed/cut-off attempt, not an excuse to increase the cap.

Evaluate final and actual initial at seed4000001009, first three natural sampled
episodes per lane (24 selected per arm), zero updates. Each frozen arm has a
600k-action/300-second cap. Audit exact tensor equality and all action accounting;
replay every collected action on CPU, keep stream0 videos, and apply the existing
Freeway task-success criteria. CPU replay/audit cap1200 seconds; service cap3660
seconds includes construction and both frozen caps.

Before this allocation, a separate4096-action smoke produces16384 emulator frames
and482 updates in8.912s (459.58 aggregate actions/s), with zero debt. Its1024-action
frozen restore preserves all346 tensors exactly. Finite reports, configuration,
checkpoint counters and both host guards pass. This measures a cheap execution
path, not competence. The full200k run starts fresh rather than extending the smoke.

Native library remains SHA256
`6cb2677f2bc33f0aaced090ed7b6f328533057893672434c6b30c0ef08dec7fd`, source`f4bb2ca`,
on RX7900XT/RADV. The runner now rejects invalid numeric reports directly
(`769bdef`;325 relevant CPU tests pass). BF16-only resident targets stay disabled.
No native rebuild, driver operation or kernel change is part of this trial.

Artifacts: `/x/Temp/kindle-cheap-20261025/freeway-1m-smoke` and `freeway-1m-200k`.
Controller: `run_freeway_200k.py` in that artifact root. Exact commands, config,
source/native identities, controller and smoke-review hashes are declared before
launch. Service owner: `kindle-cheap-freeway-200k-20261025`.

Review frozen performance and the complete wall/action/frame curve before any
replication or changed recipe. This changes capacity, sequence geometry and
replay ratio together. It tests whether a cheap configuration is useful, not
which change caused a speed or score difference. One reused development seed
does not establish reliability or solve the five-game goal.

## Follow-up: persistent exploration

The unassisted trial is complete: frozen final and initial both score0/24
crossings after8m07s training; all audits pass. See the result report.
The follow-up changes only training exploration: probability.05, persistent
uniform16-action blocks (`persistent-uniform-v1`), disabled in both frozen arms.
The same seed1009 is selected after seeing its failure, so it is not held out.
All other config, fresh200k-action budget, caps, actual-initial control and
24-episode frozen selection remain identical. It is a fresh life, not a warm
start. No automatic extension or replication is queued.

Artifacts: `/x/Temp/kindle-cheap-20261025/freeway-1m-200k-aided`.
Controller: `run_freeway_200k_aided.py`; service owner:
`kindle-cheap-freeway-aided200k-20261025`. Native runtime remains unchanged.
A separate low-priority, two-core CPU build may prepare a future intermediate
model preset during this allocation; it cannot replace the installed runtime
until training, frozen checks and replay have all exited.
