# Breakout agents without hand-made relevance priors

Every Breakout learner so far was told what matters by hand. Pixel worlds
weight changed pixels up (motion cross-entropy), standalone worlds receive
hand-coded fire/left/right action features and copy scenery from a real
context frame, and every agent was guided by the scripted pixel teacher:
takeovers in 20% of episodes, teacher labels on every visited state and an
imitation loss of scale 100 routed into the world state. The retained
learners still read the ball from their state poorly (linear error 9 to 13
source pixels against .06 for a fixed pixel detector), score far below the
teacher, and never clear two walls. This screen removes all of those priors
and asks what the representation learns from pixels, reward and value alone,
and whether value gradients and offline-style teacher demonstrations make the
ball and paddle part of the state.

**Clean recipe.** Plain DreamerV3 learned-RGB agent: published protocol (full
18 actions, sticky .25, no no-op starts), eight environments, 12M preset,
64x64 RGB from native frames with one GPU resize, squared-error
reconstruction decoded from the model state only, actions as ordinary learned
inputs, no prior-image rollout, no exploration bonus, no guidance or labels.
Retained generic settings: B16/T16/R16, replay 100k, actor unimix .01,
behavior learning rate 8e-5, D3 replay-value loss 1.0, 400k actions, seed 1009.

| Arm | Change from the clean recipe |
| --- | --- |
| A clean | none (the replay-value critic loss reaches the representation, as in D3) |
| B ac_grads | `--actor-critic-gradient`: imagined actor and value losses also train the starting posterior states |
| C demonstrations | the edge-correcting pixel teacher plays each lane's first six episodes in full (`--breakout-demonstration-episodes 6`, about 83k of the 400k actions), then never acts or labels again; its actions are cloned at scale 100 with the gradient routed into the world state (`--guidance-world-gradient`), fading from update 6,000 to 11,000 as demonstrations leave the 100k replay |
| D demonstrations + ac_grads | C and B together |
| E no value gradient | `--no-replay-value-gradient`: no value loss reaches the representation |

The teacher's own play under this protocol scores a mean 202 (median 139) over
16 episodes, 1,724 actions per episode. Its hand-coded perception never
reaches the learner: the learner sees only frames, actions and rewards, as it
would from a human. The demonstration arms spend about a fifth of the budget
watching rather than playing; budgets are reported as total actions.

**Measures.** Frozen sampled and greedy play on cohorts 4000777001 and
4000022003 (three natural episodes per lane, 24 episodes). A linear probe of
the posterior model state on held-out teacher play
(`python/examples/breakout_agent_probe.py`, seed 5550001, 1,536 actions per
lane; every checkpoint observes the same trajectories; ridge on lanes 0-3,
penalty on 4-5, scored on 6-7): ball x/y and paddle x mean absolute error in
source pixels, the share of ball rows read within 8 px, and ball velocity R².
Each arm's untrained initial checkpoint is probed as its own floor. RAM
positions label the probe only.

**Reading.** A is the reference for everything clean; it is expected to play
poorly at this budget (unguided 4M CDP Breakout scored 9.5 to 11.2 at 200k).
The probe answers the representation question directly: C over A says
demonstrations put the ball into the state, B over A and D over C say value
gradients into the representation help, E against A says whether D3's
default value path matters at all. Read probe gains of a few pixels as
noise until repeated. Previous guided learners on 4000777001 (greedy /
sampled at 400k: pixel world 223.0 / 89.8, CDP 192.0 / 90.6) are context,
not controls. One seed per arm; development screen, not qualification.
Root `/x/Temp/kindle-breakout-clean-20261115`.
