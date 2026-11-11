# 1M context-frame world with a 32-step training rollout for 65,536 updates

The [4M run with a 32-step rollout](../results/2026-11-11-breakout-world-context-h32.md)
showed that supervising thirty-two-step images lifts thirty-two-step ball
tracking (28/116 held-out against 19/116) at 870 ms per update, which puts a
65k budget at sixteen hours. The 1M context-frame world matched the 4M one
closely at sixteen steps in the 16-step-rollout series (1M at 16,384: 72/32
at four and eight steps against 4M's 74/30). This run asks whether the
cheaper model carries the 32-step objective to a 65k budget in a workable
time, and whether thirty-two-step tracking keeps rising with updates under
that objective.

## Recipe

`context-config-96-1m-h32.json`: the 1M context-frame config
(`context-config-96.json`: 96x96, B8, context lag 8, 1M, Auto, seed 41001)
with `batch_length` 48 and `world_backprop_length` 48. Command:
`breakout_world_scale.py train --steps 65536 --eval-every 8192 --dense 128
--prediction-length 32 --pixel-loss motion_event_rollout` on `corpus/fit96`
and `corpus/dev96`. Output `train-96-context-1m-h32`, unit
`kindle-world-context-1m-h32-20261111.service`, launched directly under a 24G
memory cap without the host guard. The final checkpoint is scored on the
held-out seed 777001 corpus with `probe`.

## Reading

Against the 4M 32-step run at 32,768 updates (development 110/96/64/41/31;
held-out 109/89/63/42/28) and the 1M 16-step run at 16,384 (107/72/32/23/11
at one to thirty-two on the development origins): thirty-two-step counts
above 31/102 at 65k would say the objective keeps paying with budget and
that 1M is enough; counts at or below the 4M 32k point would say capacity
limits the longer horizon. One run on reused seed and origins plus one
held-out point.
