I agree it's solid progress. But the learning isn't slow for the amount of experience it gets, and CDP was never likely to crush Atari. What looks slow comes from comparing 2M-frame runs with targets DreamerV3 reached after 200M frames. The real bottleneck is wall-clock time, and I estimate close to half of each update goes to the exploration bonus.

**What's genuinely better since PR #29:**
- Comparisons now use three seeds against matched controls, with untrained-model baselines.
- The frozen LeVJEPA encoder was dropped because the evidence didn't support it.
- Freeway now learns with no exploration aid, thanks to the action-effects bonus.
- Centering the CDP loss rescued Pong, which failed on two of three seeds with the plain cosine loss.
- The 12M model clearly helps Breakout, Pong and Qbert.

## Is learning slow?

Not compared with DreamerV3 at the same experience. Below are online scores against DreamerV3's released Atari57 curves. Upstream uses the same replay ratio (32) and sticky actions, but a ~200M-parameter model, about 12× yours.

| Game | Kindle 12M at 2M frames | DreamerV3 at 2M frames | DreamerV3 at 8M frames | DreamerV3 at 200M (the PR's target) |
|---|---|---|---|---|
| Boxing | 92.0 | 89.8 | 98.7 | 99.6 |
| Pong | 11.9 | −7.2 | 14.9 | 20.4 |
| Freeway | 23.1 | 2.1 | 12.4 | 33.4 |
| Breakout | 37.6 | 12.7 | 81.9 | 381.8 |
| Qbert | 2,600 | 2,665 | 14,608 | 193,221 |

- **Qbert is the one clear gap.** At 2M actions (8M frames) it scores 6,546 online, against DreamerV3's 14,608 at the same point.
- **The targets are 50M-action numbers.** That's 100× the 500k-action runs. At the current speed, reaching that budget takes about 16 days per seed per game.

## Should CDP crush Atari?

No, for three reasons:

- **The paper doesn't claim it.** Dreamer-CDP was tested only on Crafter, using DreamerV3's ~200M-parameter XL configuration. It claims parity with reconstruction-based Dreamer: 16.2% vs 14.5%, not statistically significant. Its pitch is dropping the decoder without losing performance. Your Seaquest result (+71% score, 36% less world-training time) is already a good outcome for it.
- **Atari is the worst case for it.** Nearly every Atari pixel is game state, so reconstruction wastes little. The risk of dropping reconstruction is losing tiny objects. Your 1M Breakout probes read the paddle from the learned state well, but the ball poorly, even though the frames contain it. Latent prediction should pay off on cluttered 3D scenes, your eventual target.
- **Agents that crush Atari early pay in compute per sample.** BBF does it with a high replay ratio and network resets, EfficientZero with tree search. DreamerV3's own Atari100k recipe uses replay ratio 256, 8× yours. The biggest gain in this PR also came from model size (1M→12M took Breakout from 5 to 38), not from the loss. All of these levers cost GPU time.

## Wall clock

You read it right. Each 2M-action seed takes about 56,000 s (15.5 h, ~36 actions/s), and the three Qbert seeds ran one after another for 46h40m. Almost all of it is the learner: one update of about 108 ms every 4 actions. That splits into ~61 ms for imagination, ~33 ms for world-model training and ~10 ms for the posterior pass.

- **The exploration ensemble is about half.** For each of the 1,920 imagined states in an update, it evaluates 4 heads for all 18 actions.
  - Its first layer alone is 91 billion multiply-adds, measured at 20.5 ms.
  - On Qbert, intrinsic reward is under 0.03% of the return measured at the end of training.
  - A rewrite that cuts its arithmetic by 63% is mathematically exact, but was rejected. Its output differed from the original F32 result by 4.9e-4 against a 2e-5 tolerance, likely because the backend picked a different kernel. The no-retry rule then closed it.
- **The native stack is ~3× slower than JAX on the same recipe.** On your machine, upstream DreamerV3 ran the 1M recipe at 387 actions/s against 125 native, both in F32. Upstream normally trains in bf16; Meganeura has no bf16 path, and the 2e-5 tolerance rules out reduced precision anyway.
- **The T64 test changed the wrong axis.** The posterior and world-training passes step through the sequence one timestep at a time, with only 8 sequences per batch. Longer sequences add steps, so cost grew linearly. More sequences per batch would add width instead, which is nearly free on these small kernels.

## What I'd do

1. **Set targets at matched budgets.** Use DreamerV3's curve at the same frame count (for example, 8M frames) as the bar, not the 200M-frame endpoint.
2. **Speed it up, roughly in order of payoff for the effort:**
   - **Make exploration cheap.** Compute the state part of the first layer once per head instead of once per action, fixing the kernel choice so the tolerance check compares like with like. Alternatively, score only the sampled action or a subset of states, or turn the bonus off for games that don't need it.
   - **Use wider batches and update less often.** For example, 32 sequences of 16 steps with one update per 16 actions keeps replay ratio 32 and moves toward upstream's batch size. It changes the number of optimizer steps, so it needs one learning check.
   - **Run seeds concurrently.** A 12M learner leaves ~11 GiB of the RTX 5080's memory free, so two or three seeds fit. This needs relaxing the one-heavy-GPU-job rule that came from the September GPU incidents, which is your call.
   - **Allow reduced precision.** Accept it when the 3-seed learning curves match, instead of applying the 2e-5 numerical tolerance.

   My rough estimate is 2–3× per learner from the first two together. Running seeds concurrently would add more on top.
3. **Consider stopping the Pong 2M run.** It will most likely confirm what Qbert showed, that more experience helps. It also occupies the GPU until about October 13, time the speed work needs.

Process is still heavy: about 64% of the PR's 46k added lines are documentation. The rejected exact rewrite is the clearest case of a gate costing more than it protects against.

