# Physical action features for the Breakout world

The closed half-rate16k learner still fails all three development gates and
misses many fitting forecasts. Existing conditional support evidence also shows
sparse or absent clean-launch examples for several full18 action names. Select
a different question: can known controller semantics reduce the action-learning
burden without changing images, dynamics capacity or acceptance gates?

Root `/x/Temp/kindle-world-action-features-20261109`. First select only a bounded
CPU/emulator semantics check. No learner, GPU allocation, old-model query,
completed collector repeat or automatic successor is selected by this protocol.

## Proposed representation

Use three Boolean command features: LEFT, RIGHT and FIRE, derived only from the
submitted action name. Pad the remaining15 channels with zero so input and
parameter shapes stay unchanged. The18 full actions form six candidate classes:
NOOP,FIRE,RIGHT,LEFT,RIGHTFIRE,LEFTFIRE, each with three vertical variants.
Keep simultaneous movement+FIRE; the four-action minimal set alone does not
prove all other actions redundant. This uses game/controller knowledge, not
object/RAM labels, future images, reward labels or a pretrained encoder.

Official ALE paddle input handling applies horizontal movement and FIRE
separately; pin the inspected source and verify behavior in the installed
ALE0.12.1 and the actual Breakout ROM before implementing the feature table.
[Paddle input source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/4f0ee906488d1b6521f21cc61943022c5db0d496/src/ale/environment/ale_state.cpp),
[minimal-set definition](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Breakout.cpp).
Source inspection is not proof that the installed binary is from that commit.

## New paired semantics check

Use eight independent retained fitting lanes in four fixed parent categories:
the existing clean-wait prefixes, the existing contact clip prefix endpoints,
one arrival before the first positive reward, and one arrival before the first
life loss in each original fitting stream. Preparation records the exact
indices and seeds before any emulator execution. Do not select on model errors,
replace unsupported parents or claim untouched generalization.

For every parent execute all18 constant16-command plans, six within-class
cycling plans and one mixed canonical/mixed-alias pair:26 plans×32 parents.
Every rollout starts in a fresh environment, reset with the same recorded seed
and replaying the exact original raw-action prefix. Do not rely on cloning
private sticky-action state. Before each branch require the retained last16
available native-image hashes, source rewards/terminals and parent lives to
match. Use native max-pooled RGB, repeat4/sticky.25/full18/no random reset NOOPs;
no image resizing or GPU camera.

Compare every alias/cycling plan to its canonical control, and the mixed pair,
at every arrival until natural game-over or16. Require exact native RGB and RAM
hashes, reward, terminal/truncation, lives and executed-frame counts. Keep all
mismatches, including the first divergent step. Also report pairwise differences
among the six canonical controls; clamped/idle equality at one parent cannot
merge physically distinct commands globally. Identity comparisons are not
independent controls. Save source-replay/new action and emulator-frame counts,
resets, every planned/observed transcript and source/ROM/package identities.

One CPU-only subprocess, cap600s, no automatic retry. Pin scripts, source data,
wrapper, ALE/Gym binaries/version, ROM, parent list, plans and inspected official
source before execution. Trap Kindle constructors; no native model construction,
optimizer, forecasts, checkpoint writes or changes to old evidence. Finite
matched outcomes over these parents are not a universal semantic proof or a
world-model pass. Full physics/events/brick memory,late/tunnel/second-wall/
above-top,untouched families and fresh learner confirmation remain required.

Only after reviewing the semantics evidence may a separate native feature-table
implementation/qualification and bounded learning protocol be selected. It must
default off, persist its action representation, preserve ordinary raw action
traces, apply consistently during learning/observed inference/prior inference,
and retain original full-game gates. No benefit is established yet.
