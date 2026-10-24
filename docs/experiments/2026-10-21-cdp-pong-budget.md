# Pong: return to longer-budget learning

Fresh seed1009 only,2M actions/499,939 updates; review before allocating more.
This replaces the interrupted October11 allocation, not its results. Backend
optimization/upstream porting are parked and are not training gates.

Use qualified c8aeba28 native code,12M centered CDP/action-effects1,
N8/B8/T16/H15/R32, microbatch8, replay100k, BF16. No action overrides, game
hints, reward shaping or RAM inputs. Full18 actions, sticky.25/repeat4,
native pixels with one GPU64 resize. Model/loss/rate settings retain the
unassisted October21 Freeway recipe. This differs from historical Pong's
backend/precision; do not attribute historical differences solely to budget.

Save actual-initial weights and500k/1M/1.5M/2M snapshots. After training,
evaluate the predefined500k,2M and initial policies: sampled actions,
eight streams, first three natural rounds per stream, evaluation seed4000001009.
Each evaluation allows600k actions/1800s and exports its frozen checkpoint.
Audit exact346 tensors, zero updates, finite checkpoints, complete replay and
whole stream-zero videos. Retain excess episodes, cutoffs and unfinished tails;
any cutoff prevents the intermediate pass. Do not select the best checkpoint.

Training deadline14400s; whole service24000s,12GiB/no swap/no restart.
CPU reviews use one CPU/2GiB/no swap. Host-only guard records allocation
warnings; real faults/API/numerical errors stop. No NVML polling or recovery.
Fresh NFS artifacts avoid the nearly full local disk. No checkpoint resume.

The historical intermediate gate is mean≥15 and≥90% wins, not five-game
supremacy. The published Pong reference20.4455 is a much longer-budget curve,
not a matched short-budget control. Compare500k→2M within this life and retain
the actual-initial control. A single seed cannot establish reliability.

Service: `kindle-cdp-pong-budget-1009-20261021`.
[Declaration](../../runs/cdp-pong-budget-20261021.SNxyTFk7/declaration.json).
[Automated controller](../../runs/cdp-pong-budget-20261021.SNxyTFk7/run.py).
