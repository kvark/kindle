# 4M context-frame world with a 32-step rollout for 65,536 updates

The [4M 32-step-rollout world at 32,768 updates](../results/2026-11-11-breakout-world-context-h32.md)
is the best Breakout world on the held-out corpus at thirty-two steps
(28/116) and was still improving at its last point; the [1M model at
65,536](../results/2026-11-11-breakout-world-context-1m-h32.md) showed capacity
limits the long horizon. This run is the 4M model's 65k budget point under
the same objective: a fresh life of `context-config-96-4m-h32.json` with
`--steps 65536 --eval-every 8192 --dense 128 --prediction-length 32
--pixel-loss motion_event_rollout` on `corpus/fit96` and `corpus/dev96`.
Deterministic: its first four points reproduce the 32k run; new information
starts at 40,960 updates. Output `train-96-context-4m-h32-65k`, unit
`kindle-world-context-h32-65k-20261112.service`, direct launch under a 24G
memory cap without the host guard; about sixteen hours at 870 ms per update.
The final checkpoint is scored on the held-out seed 777001 corpus with
`probe`. Read against the 32k point (development 110/96/64/41/31, held-out
109/89/63/42/28) and the 1M 65k point (112/95/63/42/24, held-out
110/88/69/45/18). One run; development screen plus one held-out corpus.
