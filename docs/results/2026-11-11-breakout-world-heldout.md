# Held-out qualification of the two best standalone Breakout worlds

[Protocol](../experiments/2026-11-11-breakout-world-heldout.md). Fresh corpus
`/x/Temp/kindle-world-scale-20261110/corpus/heldout96`: seed 777001 (lane
seeds 777001 to 784064, never used before), 8,192 actions, 32,761 emulator
frames, 16 episodes, rewards 23 to 43 per lane, collected 09:43 UTC on
November 11 in 4.1 s. Outputs `heldout/context-4m-65k` and
`heldout/state-only-1m-65k` (probe mode, 128 dense origins per split, origin
draws seeded as in training, no selection). Side-agnostic ball tracking
within 1.5 px at 96x96; the development columns repeat each checkpoint's
score on the reused November 7 origins.

| Checkpoint | Split | Observed | 1 | 4 | 8 | 16 | 32 | Brick error at 8 |
|---|---|---|---|---|---|---|---|---|
| Context frame 4M, 65,536 | **held-out** | 115/115 | 112/115 | 93/117 | 72/113 | 41/111 | 19/116 | .009 |
| Context frame 4M, 65,536 | development (reused) | 113/113 | 111/112 | 97/112 | 72/112 | 53/115 | 22/102 | .007 |
| Context frame 4M, 65,536 | fitting | 114/114 | 114/123 | 96/125 | 74/118 | 39/109 | 19/116 | .004 |
| State-only 1M, 65,536 | **held-out** | 115/115 | 109/115 | 87/117 | 65/113 | 38/111 | 19/116 | .223 |
| State-only 1M, 65,536 | development (reused) | 113/113 | 109/112 | 93/112 | 70/112 | 46/115 | 15/102 | .32 |
| State-only 1M, 65,536 | fitting | 114/114 | 108/123 | 92/125 | 66/118 | 38/109 | 17/116 | .192 |

Linear state probe on the held-out origins (development RMSE, native pixels;
ball x / ball y / paddle x): context frame, encoder 7.5 / 8.9 / 12.4,
posterior 15.4 / 5.6 / 8.6, prior at 1 step 16.4 / 5.2 / 9.2, 8 steps 23.2
/ 7.0 / 11.5, 16 steps 18.7 / 9.0 / 14.2, 32 steps 33.8 / 15.1 / 16.2;
state-only, encoder 9.3 / 10.3 / 10.8, posterior 11.9 / 7.4 / 10.0, prior at
1 step 13.7 / 6.1 / 10.1, 8 steps 18.9 / 7.8 / 11.2, 16 steps 18.1 / 10.0 /
14.0, 32 steps 35.4 / 14.8 / 17.0.

## Reading

The development screens were honest at short and medium horizons and
optimistic at long ones. On the fresh corpus the context-frame world keeps
its one-, four- and eight-step counts (112, 93, 72 against 111, 97, 72) and
loses at sixteen and thirty-two (41/111 and 19/116, that is 37% and 16%,
against 46% and 22% on the reused origins). The state-only world shows the
same shape (87 and 65 at four and eight; 38 and 19 at sixteen and
thirty-two). The fitting-corpus scores sit at the held-out level, not above
it, so neither world memorized its training corpus; the long-horizon
optimism belongs to the reused development origins specifically. On
held-out data the context-frame world leads the state-only world at every
horizon up to sixteen steps (112, 93, 72, 41 against 109, 87, 65, 38), ties
at thirty-two, and renders the bricks correctly (.009 against .223). The
64x64 control has no held-out point because its episodes cannot be
reproduced at 64x64 from this collector.

This is one held-out corpus of 16 episodes with one origin draw per split;
it qualifies the ordering and the short-horizon counts, not the exact
long-horizon numbers. Thirty-two-step ball tracking at 16% to 19% on either
corpus is the standing limitation of the Breakout world model.
