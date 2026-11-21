# 12M context-frame world with a 32-step rollout

Under the 32-step training rollout, the 4M model beat the 1M model at every
long horizon (held-out thirty-two steps 28/116 at 32k updates against the 1M
model's 18/116 at 65k), and the 4M model's 65k run was still improving
(37/116). Capacity matters for the long horizon under this objective; this run
asks how much. It trains the 12M preset (the retained agent's size) with the
same objective for 32,768 updates: `context-config-96-12m-h32.json` is the 4M
32-step config with `model_size` changed, and the command is otherwise the
32k run's (`--steps 32768 --eval-every 8192 --dense 128 --prediction-length
32 --pixel-loss motion_event_rollout` on `corpus/fit96` and `corpus/dev96`).
Output `train-96-context-12m-h32`, unit
`kindle-world-context-12m-h32-20261112.service`, direct launch under a 24G
memory cap without the host guard; cost per update is expected near twice the
4M model's 870 ms. The final checkpoint is scored on the held-out seed 777001
corpus with `probe`. Read against the 4M 32k point (development 110/96/64/41/31,
held-out 109/89/63/42/28) at equal updates and against the 4M 65k point
(112/99/75/61/32, held-out 110/90/72/51/37) at roughly equal native time. One
run; development screen plus one held-out corpus.
