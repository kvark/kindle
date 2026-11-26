# Life loss as a training terminal on the clean centered-CDP Breakout agent

The goal is now **make Kindle learn Breakout well**. The [clean centered-CDP
learner](../results/2026-11-15-breakout-clean-cdp-96.md) holds the ball in
its state on seed 1009 (linear probe 6 px) and still plays a median of 28 to 38 per game:
68 of its 96 frozen episodes score 50 or less. Its numbers match the Dreamer
family's published Breakout weakness, not a Kindle defect: at 100k actions
it plays 32 to 40 with sticky actions and 18 actions, where DreamerV3
reports 31.

| Agent, Atari 100k Breakout | Score | Training episode ends on life loss |
|---|---:|---|
| DreamerV3 | 31.0 | no (`lives: unused`) |
| IRIS | 83.7 | yes (`done_on_life_loss: True`) |
| DIAMOND | 132.5 | yes (`done_on_life_loss: True`) |
| BBF | 370.6 | yes (`terminal_on_life_loss = True`) |
| EfficientZero | 414.1 | yes (`episode_life=True`) |

Scores are from the DIAMOND and BBF papers' tables; the life handling was
read from each project's published configuration on November 15. The four
agents that play Breakout well at this budget all end the training episode
when a life is lost, and evaluate on whole games. The clean recipe follows
DreamerV3 and does not. Without that boundary a missed ball costs nothing
at once: play resumes after a FIRE press, and only the fifth loss ends the
episode. The value difference between catching and missing is then small
against the return scale, and smaller still once a few tunnelling episodes
inflate it, which is where the seed-1009 run lost its sharp policy (scale 20
to 93, entropy .5 to 1.3 nats, online returns 124 to 59). The agents differ
in other ways (search, replay ratio, resets), so this is a hypothesis.

Kindle's two earlier life-loss aids were different interventions on the
guided recipe: a [one-point penalty](../results/2026-11-03-cdp-breakout-life-penalty.md)
(worse) and [life-bounded auxiliary actor returns](../results/2026-11-03-cdp-life-return-actor.md)
(neutral). Neither made the loss a terminal for the world model's
continuation head and the critic.

**Change.** `--life-loss-terminal` in the runner, no native change. When
the emulator's lives counter drops during an action and the game is not
over, the stored transition is marked terminated and the stream begins a
new stored episode from that same frame (`begin_episodes`, so the recurrent
state resets as at a game start). Rewards are unchanged. The emulator keeps
playing the same game, logged episodes and returns stay whole games, and
evaluation never uses the flag (the runner rejects it with `--evaluate` or
`--restore`). The log audit accounts for the extra stored frame and rejects
undeclared or malformed life terminals.

This uses a game-provided signal during training, as the four agents above
do. It is not a label, guidance, hand feature or reward change, and the
policy never sees it. It does leave DreamerV3's default protocol, which is
disclosed here and in every result.

**Arms.** All: published protocol (18 actions, sticky .25, repeat 4, no
no-op starts, 8 environments), 12M, 96x96 RGB, B16/T16, replay 100k, actor
unimix .01, behavior rate 8e-5, replay value 1.0, `--actor-critic-gradient`,
`--cdp --cdp-centered --encoder-learning-rate 4e-5`, 400k actions,
checkpoints kept every 100k. No guidance, labels, imitation,
demonstrations, motion weighting, context frame or bonus.

| Arm | Seed | Train ratio | Life terminal | Updates | Expected time | Control |
|---|---:|---:|---|---:|---|---|
| `M22-cdp-life-ratio64-96-s22003` | 22003 | 64 | yes | 100k | 3.2 hours | seed-22003 run |
| `M10-cdp-life-ratio64-96` | 1009 | 64 | yes | 100k | 3.2 hours | seed-1009 run |
| `K-cdp-life-ratio16-96-s22003` | 22003 | 16 | yes | 25k | about 1 hour | arm L |
| `L-cdp-ratio16-96-s22003` | 22003 | 16 | no | 25k | about 1 hour | |

The two M arms are the clean CDP recipe (`queue-7.json`, `queue-8.json`)
with only the flag added, one per existing control seed. The controls
differ widely: [seed 1009](../results/2026-11-15-breakout-clean-cdp-96.md)
learned the ball (within 8 px .51 to .67) and plays 71 to 91, [seed
22003](../results/2026-11-15-breakout-clean-cdp-96-seed22003.md) never did
(.08 to .12) and plays 34 to 59. Seed 22003 runs first: it is the case the
hypothesis says the boundary should repair, and its online curve can be
read against the control's plateau (19 to 23 per game through 250k
actions) long before the run ends. K and L are a fast matched pair at a
quarter of the updates on the same seed; they measure whether ratio 16 is
a usable screening recipe for this objective.

Runtime: Kindle built on Meganeura main `e3215fa2` (the repin restores the
on-device parameter snapshot, which these arms do not use; the Meganeura
source change is the 10-line `slot_buffer` accessor). Runner and audit from
this commit, copied to the root with `sources.sha256`; probe script
unchanged from the clean runs. Root `/x/Temp/kindle-breakout-life-20261115`,
queue `queue-9.json`, guards `guards-9`, unit
`kindle-breakout-life-20261115`, 15 GiB memory cap, run in the order M22,
M10, K, L (the queue stops on any failure), about 8.7 hours.

**Measures.** Frozen whole-game play, sampled and greedy, three natural
episodes per lane on cohorts 4000777001 and 4000022003: at 100k and 400k
for the M arms, at 400k for K and L. The linear state probe (seed 5550001)
at every kept checkpoint for the M arms and at 400k for K and L. Online
whole-game returns, policy entropy and return scale per 50k actions.
Frozen evaluations get an hour each, because a learner that plays well
produces long episodes.

**Reading.** Means and medians per cell, 24 episodes each. Play is bimodal
in both controls, so medians carry the comparison; pooled medians over the
96 episodes at 400k are 27 for seed 22003 and 30 for seed 1009.

- Representation: the boundary repairs the failed seed if M22's probe
  reads the ball within 8 px above .4 at 300k to 400k (control .08 to
  .10).
- Play, per seed: the boundary helps if the M arm's pooled median at 400k
  is at least twice its control's (54 for seed 22003, 60 for seed 1009)
  and every cell mean is above the control's.
- It helps the recipe if both seeds meet the play criterion. One seed of
  two is unresolved and needs a further seed, not a claim.
- K against L: the boundary helps at ratio 16 if K's median exceeds L's in
  at least three of four cells and its pooled median is at least 1.5 times
  L's. Ratio 16 is a usable screen if that verdict agrees with M22's.
- If the play criterion fails on both seeds, life handling is not what
  separates Dreamer-style agents from the others on Breakout, and the next
  step is the use of the model (decision-time planning on existing
  checkpoints) or the return normalizer, each declared separately.
- A probe on seed 1009 clearly worse than its control's (within 8 px
  under .4 at 300k to 400k) would mean the reset costs representation and
  is reported whatever the score.

One run per arm on development cohorts and reused seeds: a screen, not a
reliability claim.
