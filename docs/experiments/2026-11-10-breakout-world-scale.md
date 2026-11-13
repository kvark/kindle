# Breakout world model: corpus scale before more knobs

Every isolated Breakout world learner so far trained on about 33k arrivals for
16,384 B8/T32 updates, roughly 128 passes over the same frames. Eight-step ball
tracking is equally poor on fitting and development origins (86/176 versus
52/107), so the long-horizon failure is not explained by overfitting alone, but
no run has tested whether the recipe learns the dynamics when it sees enough
distinct play. This experiment changes only the amount of experience.

## Data

`python/examples/breakout_world_scale.py collect` records eight lanes with the
same fixed pixel teacher plus random 4–12 command blocks (probability .025) as
the November 7 corpus, sticky .25, repeat 4, native max-pooled frames and one
GPU resize, storing only the 64x64 frames, actions, rewards, terminals, RAM
positions and lives. Fitting: 65,536 ticks per lane (524,288 aggregate actions),
seed base 251001. Development: 4,096 ticks per lane (32,768 actions), seed base
291001. Seeds do not overlap each other or the November 7 corpora.

Collection runs as a plain memory-capped `systemd-run --user` unit rather than
under `gpu_host_guard.py`: the guard's 3 s `journalctl` kernel probe times out
during the 6.4 GB lane-file write burst and killed a complete first collection
(retained under `guards-v2/`; an earlier attempt failed on the learner
constructor, retained under `guards-failed-v1/`). The GPU only resizes frames
during collection; the training job keeps the guard.

## Learner

Unchanged early physical-feature recipe: 1M/RGB64/B8/T32/full BPTT/Auto/seed
41001, RGB100, KL1/.1, free nats 1, rates 4e-4, spatial decoder/fusion/action,
LEFT/RIGHT/FIRE action table, `motion_event_rollout` with prediction length 16.
16,384 updates, so the learner sees each fitting arrival about eight times
instead of 128. Evaluation every 2,048 updates on the development natural,
contact and ending groups (four draws) and 128 dense origins per split (one
draw), with the existing ball/paddle/brick/event metrics.

Native stack: Meganeura main `a0eff600`/Blade `d9112e26`, built from the
working tree after the dependency switch; the exact source hash and timing are
recorded in the result report. Timing is not comparable to the fork's
1,841 s/16,384 updates until measured on the same stack.

## Reading the result

Compare development ball-within-1px at horizons 1/4/8/16/32 against the
closed early-feature control (99/109, 66/109, 53/107, 25/114, 11/95) and its
fitting counterpart. A large development gain with fitting roughly unchanged
means the corpus was the limit; parity on both means the model or objective is,
and capacity/latent determinism come next. Either way this is a development
screen on reused seed 41001, not a held-out qualification.
