# 4M context-frame world with a 32-step training rollout

The [4M context-frame world](../results/2026-11-10-breakout-world-context-frame.md)
is evaluated to thirty-two steps but trained with a sixteen-step prior image
rollout (`--prediction-length 16` in a 32-step window: sixteen observed steps,
sixteen imagined). Its thirty-two-step ball tracking stalls at 16% to 22%
(19/116 held-out) while its sixteen-step tracking reaches 37% to 46%. This
run supervises the images the metric asks for: a 32-step rollout after a
sixteen-step observed prefix, in a 48-step window.

## Recipe

`context-config-96-4m-h32.json`: the 4M context-frame config with
`batch_length` 48 and `world_backprop_length` 48; everything else unchanged
(96x96, B8, context lag 8, 4M, Auto, seed 41001, learning rate 4e-4). Command:
`breakout_world_scale.py train --config ... --steps 32768 --eval-every 8192
--dense 128 --prediction-length 32 --pixel-loss motion_event_rollout` on the
same `corpus/fit96` and `corpus/dev96`. Output `train-96-context-4m-h32`,
unit `kindle-world-context-h32-20261111.service` (launched directly under a
24G memory cap without the host guard, as the 65k run was). Each update
sees 48 frames per sequence instead of 32 and rolls out twice as far, so
updates are not compute-matched to the 32k point of the 16-step run; the
comparison is at equal update counts with the cost disclosed.

## Reading

Against the 16-step run at 32,768 updates (dense development 111/92/66/46/15
at one to thirty-two steps) and at 65,536 (111/97/72/53/22), a clear gain at
sixteen and thirty-two steps with the short horizons held would say the
long-horizon limit was an objective-horizon limit; parity would say it is
model or data capacity. The final checkpoint is also scored on the held-out
seed 777001 corpus with `probe`. Development screen on reused seed and
origins; one run.
