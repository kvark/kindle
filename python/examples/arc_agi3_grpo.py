"""ARC-AGI-3 joint multi-game training with state-locked GRPO rollouts.

Built 2026-05-07 to break the 1-game-at-L2-per-50k-steps wall hit by
arc_agi3_multi.py. The core idea:

  - Use the underlying ARC game's pickle-able state to do BACK / state
    save/restore (`copy.deepcopy(env._game)` works — verified).
  - Run K parallel forks per game (25 games × K forks = 25K lanes).
    All K forks of game g start from the same state at the beginning
    of each macro step.
  - Every M micro-steps, score the K forks of each game by running
    return; copy the best fork's state to all K forks of that game
    (state synchronization). This is the BACK operation amplified.
  - kindle's standard cross-lane advantage normalization happens to
    work IN OUR FAVOR here: K forks from the same starting state see
    diverse outcomes; the value head learns the per-state return
    distribution; advantages naturally sort the K forks. This is a
    GRPO-flavored learning signal without needing per-game grouping
    in the advantage code.

Works on top of the K_lowent_recon10 recipe + goal_bonus + sil_event_filter
+ whatever else proved promising in earlier sweeps. The BACK forks just
multiply effective sample efficiency on rare-reward states.

Defaults: K=4 forks/game, M=20 micro-steps/macro-step, total lanes
= 25 × 4 = 100. Throughput ~50 env/s aggregate at this scale.
"""
from __future__ import annotations

import argparse
import collections
import copy
import sys
import time


def main() -> int:
    try:
        from arc_agi import Arcade
        from arcengine import GameAction, GameState
    except ImportError as exc:
        print(f"missing arc_agi toolkit: {exc}", file=sys.stderr)
        return 1
    import numpy as np
    import kindle

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20000,
                        help="Total macro-steps. Each macro step advances all "
                        "lanes M micro-steps. So total env-steps per game = "
                        "K · steps · M.")
    parser.add_argument("--forks-per-game", "-K", type=int, default=4,
                        help="Number of parallel forks per game (K). All K "
                        "forks of game g start from the same state at the "
                        "beginning of each macro step.")
    parser.add_argument("--macro-len", "-M", type=int, default=20,
                        help="Micro-steps per macro step (M). After M steps, "
                        "the harness scores all K forks of each game by "
                        "macro-window return and copies the best fork's "
                        "underlying game state to all K forks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--log-every-macro", type=int, default=50,
                        help="Diagnostic line every N macro steps.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--watchdog-threshold", type=float, default=1e6)
    parser.add_argument("--levels-reward-scale", type=float, default=10.0)
    parser.add_argument("--encoder", choices=["mlp", "cnn", "cnn_dqn"],
                        default="cnn_dqn")
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--rnd-alpha", type=float, default=2.0)
    parser.add_argument("--reward-surprise", type=float, default=5.0)
    parser.add_argument("--reward-homeostatic", type=float, default=0.1)
    parser.add_argument("--use-ppo", type=int, default=1)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-beta", type=float, default=0.01)
    parser.add_argument("--use-sil", type=int, default=0)
    parser.add_argument("--sil-loss-coef", type=float, default=0.5)
    parser.add_argument("--sil-event-filter", type=int, default=0)
    parser.add_argument("--recon-loss-coef", type=float, default=10.0)
    parser.add_argument("--recon-visual-target", type=int, default=1)
    parser.add_argument("--goal-bonus", type=float, default=10.0)
    parser.add_argument("--use-grpo", type=int, default=0,
                        help="Enable kindle's cross-batch advantage "
                        "normalization. Pairs naturally with state-locked "
                        "K-fork rollouts because forks from the same state "
                        "give clean within-batch comparison.")
    parser.add_argument("--visit-counts-max", type=int, default=10000,
                        help="Cap on the per-lane visit-counts HashMap "
                        "(novelty-bonus working memory). 0 = unbounded "
                        "(legacy). At latent_dim=256 the latent grid is "
                        "effectively unbounded, so the unbounded HashMap "
                        "grows ~1 KB/step/lane indefinitely. Recommended "
                        "10000 (10 MB/lane) for joint runs. Default 10000.")
    parser.add_argument("--game-prefixes", default=None,
                        help="Comma-separated game-id prefixes to include. "
                        "Default: all 25 games.")
    parser.add_argument("--exclude-prefixes", default=None,
                        help="Comma-separated game-id prefixes to EXCLUDE. "
                        "Applied after --game-prefixes. Useful for "
                        "leave-K-out cross-game pretraining: pretrain "
                        "with --exclude-prefixes a,b,c,d,e then resume "
                        "with --game-prefixes a,b,c,d,e to test transfer.")
    parser.add_argument("--novelty-bonus-scale", type=float, default=0.0,
                        help="Script-level frame-hash novelty bonus: each "
                        "step the agent receives bonus = scale / sqrt(count) "
                        "where count is the per-lane visit count of the "
                        "post-step frame. Fed via set_extrinsic_reward, "
                        "stacked with the level-completion goal_bonus. "
                        "0 disables (default). Try 0.5–2.0 for sparse-reward "
                        "puzzle games where the policy can't find first events.")
    parser.add_argument("--novelty-cap", type=int, default=50000,
                        help="Max distinct frame hashes per lane tracked for "
                        "the novelty bonus. Beyond this, the bonus stays at "
                        "scale/sqrt(novelty_cap+1) — small but not zero.")
    parser.add_argument("--explore-sync-novelty", type=int, default=0,
                        help="Extend the K-fork BACK sync to fall back on "
                        "frame-hash novelty when no fork hit an extrinsic "
                        "event in the macro window. Forks sync to the most-"
                        "novel fork's game state instead of letting them "
                        "diverge freely. Turns the GRPO sync into a Go-"
                        "Explore-style frontier expansion: forks become "
                        "parallel branches at the highest-novelty waypoint "
                        "discovered so far. Default 0 = disabled.")
    parser.add_argument("--archive-cap", type=int, default=0,
                        help="Per-game persistent frontier archive size. "
                        "When > 0, the macro sync saves the highest-novelty "
                        "fork's full game state (copy.deepcopy of _game) into "
                        "a per-game ring of capped size. On episode reset, "
                        "with probability --archive-reset-prob the lane "
                        "restores from a random archive entry instead of "
                        "env.reset() — the real Ecoffet Go-Explore mechanism. "
                        "0 = disabled (default).")
    parser.add_argument("--archive-reset-prob", type=float, default=0.5,
                        help="Probability of resetting an ended episode to a "
                        "random archived state instead of true env.reset(). "
                        "Only fires when --archive-cap>0 and the per-game "
                        "archive is non-empty. Default 0.5.")
    parser.add_argument("--planner-horizon", type=int, default=0,
                        help="Enable kindle's model-based planner. The "
                        "trained world model rolls forward K=planner-horizon "
                        "steps for each of `planner-samples` random action "
                        "sequences (per lane), picks the trajectory with "
                        "highest latent-visit-count novelty score, and "
                        "queues the actions to override the policy. 0 = off. "
                        "Try 5-10 for sparse-reward puzzle games where the "
                        "policy gradient can't find first events.")
    parser.add_argument("--planner-samples", type=int, default=32,
                        help="Number of random action sequences sampled per "
                        "planning call (per lane). Default 32.")
    parser.add_argument("--planner-policy-mix", type=float, default=0.0,
                        help="Policy-guidance mix for the planner's per-step "
                        "action sampling. 0.0 = pure uniform random shooting "
                        "(default; best for cold-start rare-event discovery). "
                        "1.0 = pure policy-guided (better for fine-tuning a "
                        "converged policy). Mix in between trades exploration "
                        "for policy-consistency.")
    parser.add_argument("--planner-policy-temperature", type=float, default=1.0,
                        help="Temperature for the policy-guided sampler when "
                        "planner_policy_mix > 0. T>1 flattens (more "
                        "exploration); T<1 sharpens. Useful counterweight to "
                        "a peaked policy. Default 1.0.")
    parser.add_argument("--planner-use-mcts", type=int, default=0,
                        help="Replace random-shooting CEM planner with MCTS "
                        "tree search. Uses kindle's wm_mcts_session for "
                        "expansion + latent-visit-count novelty for leaf "
                        "scoring + UCB1 for selection. Output: most-visited "
                        "root path (depth planner_horizon).")
    parser.add_argument("--mcts-simulations", type=int, default=64,
                        help="MCTS simulations per planning call (per lane).")
    parser.add_argument("--mcts-c-puct", type=float, default=1.4142,
                        help="UCB1 exploration constant. Default sqrt(2).")
    parser.add_argument("--stagnation-reset", type=int, default=0,
                        help="Press GameAction RESET (in-game level "
                        "restart, levels_completed preserved) on a lane "
                        "after N consecutive steps without a NOVEL frame "
                        "(per-lane frame-hash). No novelty for that long "
                        "is the signature of a doomed state (e.g. a "
                        "depleted step budget where nothing new can "
                        "happen) — the situation where human players "
                        "press RESET (~20x/run in the tu93 demos). "
                        "Harness-level: does not dilute the action "
                        "space or planner rollouts. 0 = off. Try 30-60.")
    parser.add_argument("--nav-action", type=int, default=0,
                        help="Add a learned NAVIGATION action slot: it "
                        "executes the action whose learned avatar-"
                        "displacement best matches the first BFS step "
                        "from the avatar to the nearest must-disappear "
                        "target (pickup), else toward the avatar's "
                        "relational exit-partner. Turns maze motor-"
                        "control (unsolvable from a pooled obs) into one "
                        "primitive the policy chooses WHEN to use; route "
                        "+ avatar + targets + action effects are all "
                        "learned from the agent's own experience. "
                        "Requires --rule-goal-coef>0 for target rules. "
                        "0 = off.")
    parser.add_argument("--fog-nav", type=int, default=0,
                        help="Fog-of-war navigation: accumulate a "
                        "persistent per-episode map (revealed walls + "
                        "explored mask) and route on it — to the exit "
                        "color when visible/remembered, else to the "
                        "nearest unexplored FRONTIER to reveal more maze. "
                        "Replaces single-frame BFS (which plans through "
                        "fog that turns to walls). Requires --nav-action. "
                        "0 = off.")
    parser.add_argument("--nav-force-min-level", type=int, default=2,
                        help="Only force fog/nav on the frontier when the "
                        "frontier level (levels_completed) is >= this. "
                        "Keeps L1/L2 bootstrapping via random+archive (which "
                        "form the rules fog-nav needs); force only at the "
                        "deep frontier where navigation is the bottleneck.")
    parser.add_argument("--nav-force-frontier", type=float, default=0.0,
                        help="Probability of FORCING the nav action on a "
                        "lane currently at the frontier (deepest-seen) "
                        "level whenever nav resolves there. The frontier "
                        "level has no reward signal yet (never won), so "
                        "the policy chooses nav only at its prior (~6%); "
                        "without commitment it cannot complete a multi-leg "
                        "pickup->...->exit route under the step budget. "
                        "Generic exploration assist (route is computed "
                        "generically), analogous to --frontier-random-"
                        "steps but goal-directed. 0 = off.")
    parser.add_argument("--reset-action", type=int, default=0,
                        help="Expose GameAction RESET (id 0) as a "
                        "learnable discrete action slot. RESET restarts "
                        "the CURRENT level (levels_completed preserved) "
                        "and is legal in every ARC game but never "
                        "advertised in available_actions — without it "
                        "the agent cannot escape doomed in-level states "
                        "(e.g. depleted step budgets). Human recordings "
                        "use it ~20x per run. 0 = off.")
    parser.add_argument("--reset-action-min-step", type=int, default=0,
                        help="Mask the RESET slot until ep_step >= N "
                        "(safety knob against early-exploration reset "
                        "spam). Default 0 (always available).")
    parser.add_argument("--restore-value-rank", type=int, default=0,
                        help="Rank frontier-level archive restores by "
                        "the win-classifier value V recorded at "
                        "admission time: prefer restoring states that "
                        "look like winning trajectories instead of "
                        "uniform choice (which, with depth-biased "
                        "admission, prefers doomed late-budget states). "
                        "Samples 12 candidates, picks max V. 0 = off.")
    parser.add_argument("--rule-goal-coef", type=float, default=0.0,
                        help="Rule-inference goal shaping: learn the "
                        "start->win object transformation per color from "
                        "the agent's OWN level completions, transfer the "
                        "averaged rule to the frontier level as a "
                        "predicted win configuration, and add potential-"
                        "based reward coef*(phi_prev - phi_cur) where phi "
                        "= object-config distance to the prediction. "
                        "Active only at the game's frontier level. "
                        "0 = off. Try 0.5-2.0.")
    parser.add_argument("--rule-goal-restore", type=int, default=0,
                        help="Rank frontier-level archive restores by "
                        "rule-goal proximity instead of uniform choice "
                        "(samples 12 candidates, picks the closest to "
                        "the predicted win configuration). 0 = off.")
    parser.add_argument("--object-click-slots", type=int, default=0,
                        help="Extend the action space with N extra "
                        "discrete slots meaning 'ACTION6-click the i-th "
                        "largest object on screen' (connected-component "
                        "centroids — generic visual salience, no game "
                        "knowledge). Replaces the uniform-random click "
                        "coordinate lottery (1/4096 per click) with a "
                        "learnable object choice. Slots are masked when "
                        "ACTION6 is unavailable or fewer objects exist. "
                        "0 = off. Max 11 (MAX_ACTION_DIM headroom).")
    parser.add_argument("--plasticity-interval", type=int, default=0,
                        help="(P5) Shrink-and-perturb the encoder/WM/"
                        "recon params every N steps (p <- 0.999p + "
                        "0.01*std*noise by default): plasticity "
                        "maintenance for late-arriving novelty. 0 = off. "
                        "Try 25000-100000.")
    parser.add_argument("--plasticity-shrink", type=float, default=0.999)
    parser.add_argument("--plasticity-noise", type=float, default=0.01)
    parser.add_argument("--surprise-ring-capacity", type=int, default=0,
                        help="(P2) Surprise frame-replay ring size (CNN "
                        "modes). Rows whose WM error spikes above "
                        "--surprise-replay-mult x lane EMA are stored "
                        "(frame+action+latent, ~33KB each) and replayed "
                        "with priority sampling at replay_ratio cadence "
                        "— extra encoder/WM gradient on novel elements "
                        "the moment they appear. 0 = off. Try 1024.")
    parser.add_argument("--surprise-replay-mult", type=float, default=1.5,
                        help="(P2) Ring admission threshold multiple "
                        "over the lane's WM-error EMA. Default 1.5.")
    parser.add_argument("--planner-sigma-alpha", type=float, default=0.0,
                        help="(P3) Information-seeking planner bonus: "
                        "add alpha * mean learned sigma per rollout step "
                        "to the trajectory score — steer toward "
                        "transitions the WM cannot yet predict (the new "
                        "elements at an unsolved level). Requires "
                        "--wm-stochastic 1. 0 = off.")
    parser.add_argument("--planner-sigma-horizon", type=float, default=0.0,
                        help="(P3) Sigma-budgeted adaptive horizon: stop "
                        "trusting a rollout once cumulative mean sigma "
                        "exceeds this budget; queue only the trusted "
                        "prefix. sigma is sigmoid-bounded so each step "
                        "adds (0,1); try 2-3. Requires --wm-stochastic 1. "
                        "0 = off.")
    parser.add_argument("--replan-surprise-mult", type=float, default=0.0,
                        help="(P4) Clear a lane's queued planner actions "
                        "when its per-step WM prediction error exceeds "
                        "this multiple of the lane's running EMA — the "
                        "world diverged from what the plan assumed, so "
                        "stop executing it blind. 0 = off. Try 2.0-3.0.")
    parser.add_argument("--planner-noise-sigma", type=float, default=0.0,
                        help="GRAM-style stochastic WM rollout: per-element "
                        "N(0,σ²) noise added to each WM step's z_next inside "
                        "the planner. Each of the `planner-samples` rollouts "
                        "diverges by both action choice AND latent "
                        "perturbation. 0 (default) = deterministic. Try "
                        "0.01–0.2 for multi-trajectory exploration. With "
                        "--wm-stochastic, acts as scale × learned σ.")
    parser.add_argument("--wm-stochastic", type=int, default=0,
                        help="Enable the heteroscedastic σ-head in the WM. "
                        "σ is trained to predict |z_target − z_hat| and "
                        "used by the planner to scale per-element noise "
                        "(requires --planner-noise-sigma > 0).")
    parser.add_argument("--wm-sigma-loss-coef", type=float, default=0.5,
                        help="Coefficient on the σ-head regression loss in "
                        "the main WM loss. Default 0.5.")
    parser.add_argument("--planner-rnd-alpha", type=float, default=0.0,
                        help="Blend factor for RND novelty in the planner's "
                        "trajectory score. score = visit_count_score + alpha * "
                        "rnd_reward(z). RND varies continuously across latent "
                        "space (unlike visit_count which is mostly ≈1 at "
                        "256-dim), so even small alpha makes the score "
                        "discriminating. Try 1.0-10.0 with rnd_alpha=2.0 on.")
    parser.add_argument("--planner-goal-alpha", type=float, default=0.0,
                        help="Goal-conditioned planner: blend factor for "
                        "max cos-sim to past win-state latents. Agent "
                        "saves the latent at every level-event into a per-"
                        "env goal archive (FIFO, cap --goal-states-cap). "
                        "Planner adds alpha * max-cos-sim(predicted_z, "
                        "goal_archive) to trajectory score. Emergent goal-"
                        "directed planning: navigate WM rollouts toward "
                        "discovered win regions. Try 1.0-5.0; 0 = off.")
    parser.add_argument("--goal-states-cap", type=int, default=100,
                        help="Per-env capacity of the goal-state archive. "
                        "FIFO eviction when full. Default 100.")
    parser.add_argument("--value-head-train-coef", type=float, default=0.0,
                        help="Enable value-head training (>0). Trains a "
                        "separate MLP V(z) on Monte-Carlo discounted "
                        "return-to-go from every completed episode. "
                        "Losses contribute baseline shape, wins contribute "
                        "peaks — the smooth interpolation gives the planner "
                        "a goal gradient across latent space, not just at "
                        "exact past win-state points. 0 = off.")
    parser.add_argument("--planner-value-alpha", type=float, default=0.0,
                        help="Blend factor for V(z_step) in planner score. "
                        "Per-step `alpha * V` added to each candidate "
                        "trajectory. Independent of --value-head-train-coef: "
                        "can train V without using it (cold-start), use a "
                        "pre-trained V without re-training, or both. 0 = off.")
    parser.add_argument("--value-head-gamma", type=float, default=0.99,
                        help="Discount factor for value-head return-to-go.")
    parser.add_argument("--value-head-buffer-capacity", type=int, default=10000,
                        help="Replay buffer cap for (latent, R_to_go) samples.")
    parser.add_argument("--value-head-train-batch", type=int, default=32,
                        help="Per-step value-head training batch size.")
    parser.add_argument("--goal-states-her-prob", type=float, default=0.0,
                        help="HER relabel probability: on each failed "
                        "episode end (zero extrinsic events), push the "
                        "terminal latent into goal_states queue with "
                        "this probability. NOTE: pushes terminal "
                        "(GAME_OVER) latents which mislead the win-"
                        "region cos-sim scorer. Default-off; needs a "
                        "separate non-win-region consumer to be useful.")
    parser.add_argument("--value-head-grad-to-encoder", type=int, default=0,
                        help="Allow V loss to backprop through encoder. "
                        "Default 0: V trains atop frozen-relative-to-V "
                        "encoder (R_to_go ≈ 0 dominates and would "
                        "collapse representations otherwise). Set 1 to "
                        "experiment with encoder shaping.")
    parser.add_argument("--bc-planner-synthetic-r", type=float, default=0.0,
                        help="BC-from-planner: synthetic R_to_go pushed "
                        "into sil_buffer for each planner-chosen first "
                        "action. Policy learns to clone planner. Closes "
                        "policy-planner gap so the executor matches the "
                        "discoverer. 0.3-1.0 recommended. 0 = off.")
    parser.add_argument("--planner-change-alpha", type=float, default=0.0,
                        help="Affordance bonus: planner step score gets "
                        "+alpha * ||z_step - z_prev|| (predicted latent "
                        "change magnitude). Drives planner toward actions "
                        "that DO SOMETHING. WM is trained cross-game/level "
                        "so this signal transfers to novel levels where "
                        "state-specific signals (value head, goal cos-sim) "
                        "have no data. 0.1-1.0 recommended. 0 = off.")
    parser.add_argument("--value-buffer-cross-game", type=int, default=0,
                        help="Pool win/loss buffers across all games "
                        "(env_ids) into a single shared queue. "
                        "Classifier learns generic 'winning' features "
                        "across games. Trade-off: forgoes the per-game "
                        "stratification bias-fix. Useful for horizontal "
                        "transfer (game->game): a novel game still gets "
                        "classifier signal from latents that look like "
                        "generic wins. 0 = off (default, per-game). "
                        "1 = on (cross-game).")
    parser.add_argument("--goal-states-cross-game", type=int, default=0,
                        help="Pool goal_states (win-latents) across all "
                        "games. Planner's cos-sim to wins draws from "
                        "shared pool. 0 = per-game (default). 1 = "
                        "cross-game. Pairs with --value-buffer-cross-game.")
    parser.add_argument("--subgoal-k", type=int, default=0,
                        help="Online k-means centroid count over the "
                        "goal_states pool. 0 (default) = disabled. "
                        "Centroids are abstractions over raw wins "
                        "(region-of-winning vs specific past wins); "
                        "designed for vertical (L1->L2) and horizontal "
                        "(game->game) transfer because they average out "
                        "instance-specific details. 4-16 reasonable.")
    parser.add_argument("--subgoal-lr", type=float, default=0.05,
                        help="Online k-means update rate. Centroid is "
                        "pulled toward new win sample by lr. 0.05 = "
                        "slow average; 0.5 = chases recent wins. Higher "
                        "values favor specificity; lower favor generality.")
    parser.add_argument("--planner-subgoal-alpha", type=float, default=0.0,
                        help="Planner score += alpha * max cos_sim(z_step, "
                        "centroid). Use ALONGSIDE planner_goal_alpha (raw "
                        "win cos-sim) — both signals add. 0 = off. Try "
                        "1.0-3.0 alongside goal_alpha=1.0-2.0.")
    parser.add_argument("--confidence-mode", type=int, default=0,
                        help="Confidence-weighted planner scoring. C rises "
                        "on extrinsic events, falls on WM surprise above "
                        "threshold. Exploit terms (value, goal, subgoal) "
                        "scaled by C; explore terms (change, rnd) scaled by "
                        "(1-C); visit_count always on. 0 = off (static "
                        "weights, default). 1 = on.")
    parser.add_argument("--confidence-win-increment", type=float, default=0.02,
                        help="C += this on every extrinsic event "
                        "(sil_ep_event_count incrementing).")
    parser.add_argument("--confidence-novelty-drop-rate", type=float,
                        default=0.005,
                        help="C -= rate * (visit_novelty - threshold) when "
                        "visit-count novelty > threshold.")
    parser.add_argument("--confidence-novelty-threshold", type=float, default=0.3,
                        help="Visit-count novelty (1/sqrt(visit+1)) above this "
                        "counts as 'novel'. 0.3 ~= visit_count<10. Bounded in "
                        "[0,1]; visit-count chosen over surprise because "
                        "surprise magnitude depends on latent_dim.")
    parser.add_argument("--win-trail-cap", type=int, default=0,
                        help="Full-trajectory archive: per-lane env-state "
                        "snapshot trail size. On extrinsic-event step, "
                        "the trail is flushed to the archive — every "
                        "state along the winning trajectory becomes a "
                        "restorable frontier. 0 = off (default).")
    parser.add_argument("--win-trail-stride", type=int, default=5,
                        help="Micro-step interval between trail "
                        "snapshots. Smaller = denser trail (more env "
                        "deepcopies, more memory). Default 5.")
    parser.add_argument("--progress-change-coef", type=float, default=0.0,
                        help="Term 1: persistent configurational delta "
                        "coefficient. Per-step reward = coef * (number "
                        "of coarse cells that have stably differed from "
                        "ep-start across recent persistence window). "
                        "Dense gradient for 'I caused real change.' "
                        "0 (default) = off. Typical 0.001-0.01.")
    parser.add_argument("--progress-diversity-coef", type=float, default=0.0,
                        help="Term 2: per-episode entropy growth "
                        "coefficient. Per-step reward = coef when the "
                        "current coarse-grid hash is new to this "
                        "episode, else 0. Rewards trajectory diversity "
                        "without lifetime saturation. 0 = off. "
                        "Typical 0.01-0.05.")
    parser.add_argument("--progress-persistence-window", type=int, default=5,
                        help="How many recent steps a cell must "
                        "persistently differ to count for Term 1. "
                        "Default 5. Filters transient UI animation.")
    parser.add_argument("--progress-grid-size", type=int, default=8,
                        help="Coarse-grid side length for progress "
                        "signals (8 = 64×64 frame → 8×8 cells). "
                        "Cell-level diffs filter pixel noise.")
    parser.add_argument("--object-token", type=int, default=0,
                        help="Use object-level features as obs token "
                        "instead of 8x8 pixel pool. Top 8 objects "
                        "(color, position, size, area, holes) + 8 "
                        "global stats = 64-dim token. NEGATIVE result "
                        "on tu93 (2026-05-16): loses spatial precision "
                        "needed for navigation. Prefer --hybrid-token.")
    parser.add_argument("--hybrid-token", type=int, default=0,
                        help="Hybrid v2: 4x4 pixel pool (16) + 6 "
                        "objects×7 features (42) + 6 globals = 64 "
                        "dims. Keeps spatial precision while adding "
                        "layout-invariant object structure. Default 0 "
                        "= flat 8x8 pixel pool.")
    parser.add_argument("--wm-kstep-k", type=int, default=1,
                        help="WM k-step training: when >1, build "
                        "wm_kstep_session that trains WM to be accurate "
                        "at depth k. Forces WM accuracy at planner-"
                        "rollout depth, fixing compounding error. "
                        "Recommended 5-10. Default 1 (off).")
    parser.add_argument("--wm-kstep-batch", type=int, default=32,
                        help="k-step training batch size.")
    parser.add_argument("--wm-kstep-train-prob", type=float, default=0.0,
                        help="Per-observe probability of running a "
                        "k-step training step. 0.5-1.0 recommended.")
    parser.add_argument("--wm-kstep-loss-coef", type=float, default=1.0,
                        help="LR scale for k-step training relative to "
                        "base learning_rate. Default 1.0.")
    parser.add_argument("--planner-action-repeat", type=int, default=1,
                        help="Each planner step's action is repeated "
                        "this many times in the WM rollout AND committed "
                        "to the planner queue. Effective atomic reach "
                        "per planner call = planner_horizon × repeat. "
                        "Default 1 (no repeat). Try 3-5 for L3+ pushes.")
    parser.add_argument("--num-options", type=int, default=1,
                        help="L1 hierarchical options: number of "
                        "discrete options (skills). 1 (default) = "
                        "L0-only, no options. ≥2 activates kindle's "
                        "option-policy session, per-lane goal table, "
                        "DIAYN-style option distinguishability bonus. "
                        "Options last `--option-horizon` steps each. "
                        "Effective planner reach = horizon × option-"
                        "horizon. 4-8 recommended.")
    parser.add_argument("--option-horizon", type=int, default=5,
                        help="How many env steps each option lasts "
                        "before re-selection. Default 5. With "
                        "num_options=8 and option_horizon=5, agent "
                        "picks one of 8 macros that each run for 5 "
                        "atomic actions.")
    parser.add_argument("--per-option-heads", type=int, default=0,
                        help="When 1, each option gets its own "
                        "policy-head MLP. Otherwise options share "
                        "a single trunk + per-option bias.")
    parser.add_argument("--option-entropy-beta", type=float, default=0.01,
                        help="Entropy bonus on option-choice policy. "
                        "Encourages distinct option usage.")
    parser.add_argument("--skill-replay-prob", type=float, default=0.0,
                        help="At plan_and_queue time, with this prob, "
                        "prefix planner_queue with a random captured "
                        "action sequence from a prior winning level "
                        "transition. Learned macros / options. 0 (def) "
                        "= disabled. 0.2-0.5 recommended.")
    parser.add_argument("--skill-capture-len", type=int, default=20,
                        help="How many recent actions to keep per "
                        "lane (rolling). On level-up, the last N "
                        "of these are appended to the game's skill "
                        "bank. Should be ≥ typical level-demo length.")
    parser.add_argument("--skill-bank-cap", type=int, default=100,
                        help="Max captured action sequences per game.")
    parser.add_argument("--archive-level-weight-base", type=float, default=1.0,
                        help="Exponential base for level-weighted "
                        "archive restore. 1.0 = uniform (default). "
                        "2.0: L1=2 L2=4 L3=8 → L3 gets ~57% of "
                        "restores when L1/L2/L3 all populated. Push "
                        "for higher-level attempts.")
    parser.add_argument("--frame-diff", type=int, default=0,
                        help="(8) Encoder gets 2 channels: current frame + "
                        "(current - prev) frame. Highlights what changed "
                        "last step. May help WM predict action effects. "
                        "0 = single channel (default). 1 = on. Doubles "
                        "first conv layer cost.")
    parser.add_argument("--trace-lane", type=int, default=-1,
                        help="(4) Per-step trace of one lane to "
                        "/tmp/aff_runs/trace_lane_{N}.csv with columns "
                        "macro,micro,ep_step,action,levels,ext_reward."
                        " -1 = off (default).")
    parser.add_argument("--frontier-random-steps", type=int, default=0,
                        help="(3) After restoring from a high-score "
                        "archive entry at level >= 2, force pure random "
                        "actions for this many steps. Bypasses planner/"
                        "policy which haven't trained on this region. "
                        "Recommended 30-100. 0 = off.")
    parser.add_argument("--stagnation-random-after", type=int, default=0,
                        help="(7) Adaptive random escape: if a lane has "
                        "had no extrinsic event for this many steps "
                        "WITHIN current episode, force fully-random "
                        "actions for the rest of the episode. Prevents "
                        "lanes from idling at L2 frontier when planner "
                        "exploits dead-end regions. 0 = off.")
    parser.add_argument("--frontier-archive-on", type=int, default=0,
                        help="(2) Go-Explore L2 frontier archive: track "
                        "each lane's deepest-progress state in the current "
                        "episode (snapshotted on every level event). On "
                        "episode reset, push to the per-level archive under "
                        "the achieved level. Future episodes can restore "
                        "from these high-water marks instead of fresh L1 "
                        "starts. Targets L2-completion bootstrap. 0 = off.")
    parser.add_argument("--cell-archive", type=int, default=0,
                        help="(P1) Novelty-cell archive admission: within "
                        "a game's FRONTIER level (deepest level reached), "
                        "snapshot the env whenever a never-before-seen "
                        "coarse-grid cell configuration appears, and add "
                        "it to the per-level archive. Gives Go-Explore "
                        "stepping stones INSIDE unsolved levels — the "
                        "win-trail/level-event paths only seed levels "
                        "that have already been completed. Scores rank "
                        "below win trails so wins displace stepping "
                        "stones once a level is solved. 0 = off.")
    parser.add_argument("--cell-archive-min-interval", type=int, default=10,
                        help="Min micro-steps between cell-archive "
                        "admissions per lane (deepcopy throttle). "
                        "Default 10.")
    parser.add_argument("--cell-seen-cap", type=int, default=100000,
                        help="Max tracked cell hashes per (game, level). "
                        "When full, no further novel-cell admissions for "
                        "that level. Default 100000.")
    parser.add_argument("--max-lvl-bump-bonus", type=float, default=0.0,
                        help="One-shot bonus added to ext_reward the FIRST "
                        "time a lane's max_levels_seen advances (per delta). "
                        "Multiplied by goal_bonus inside agent. Makes "
                        "*reaching a new level* much more attractive than "
                        "accumulating intermediate progress rewards. "
                        "Recommended 100-1000 (vs typical level reward ~5). "
                        "0 = off (default).")
    parser.add_argument("--episode-steps-per-level", type=int, default=0,
                        help="Per-lane max_episode_steps = base + this * "
                        "max_levels_seen[lane]. Higher levels get more "
                        "exploration budget per episode, since L2/L3 "
                        "completion likely requires longer trajectories "
                        "than L1. 0 = uniform budget (default).")
    parser.add_argument("--level-reward-scale", type=float, default=0.0,
                        help="Scale level rewards by level reached: "
                        "reward = delta * (1 + scale * (level - 1)). "
                        "L0→L1: 1.0; L1→L2 at scale=0.5: 1.5; L2→L3: "
                        "2.0. Pushes the planner & classifier toward "
                        "higher levels in already-unlocked games. 0 "
                        "(default) = flat binary reward.")
    parser.add_argument("--progress-empowerment-coef", type=float, default=0.0,
                        help="Term 3: per-lane empowerment from "
                        "planner rollouts (cross-sample variance of "
                        "step-0 z_next). High = different first "
                        "actions diverge state. Updated at planner "
                        "cadence; spread across following N micro "
                        "until next planner call. Normalized to "
                        "median-of-batch before scaling. 0 = off.")
    parser.add_argument("--eval-mode", type=int, default=0,
                        help="When 1, configure for inference: heavy "
                        "archive use, near-deterministic planner "
                        "policy, disable all auxiliary training "
                        "(SIL, BC pushes, value-head training, win-"
                        "trail snapshots, classifier replay). Keeps "
                        "planner + value-head consumption + encoder "
                        "forward, so the agent still acts via the "
                        "trained policy & planner. Pair with --load-"
                        "state to evaluate a trained checkpoint.")
    parser.add_argument("--eval-archive-reset-prob", type=float, default=0.95,
                        help="Eval-mode archive_reset_prob override. "
                        "0.95 = almost always start from a known "
                        "frontier (winning route). Default 0.95.")
    parser.add_argument("--eval-policy-temperature", type=float, default=0.05,
                        help="Eval-mode planner_policy_temperature "
                        "override. Lower = closer to argmax. Default "
                        "0.05.")
    parser.add_argument("--visit-count-proj-dim", type=int, default=0,
                        help="Random-projection dim for visit-count hashing. "
                        "When >0, projects the latent through a fixed random "
                        "matrix instead of truncating. Preserves L2 distance "
                        "across the full latent (Johnson-Lindenstrauss). "
                        "Strictly more informative than --visit-count-dims.")
    parser.add_argument("--visit-count-dims", type=int, default=0,
                        help="Latent-dim truncation for the visit-count "
                        "novelty hash. 0 = use all dims (default). At "
                        "latent_dim=256 the unbounded grid makes "
                        "visit_count ≈ 1 always — uniform novelty score "
                        "across planner candidates. Setting to 8 truncates "
                        "to first 8 dims, giving ~3^8=6.6k cells so "
                        "revisits are common and the novelty signal becomes "
                        "informative.")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--load-state", default=None)
    parser.add_argument("--pretrained-cnn", default=None,
                        help="Path to a wm.safetensors-format file with "
                        "pretrained encoder.conv{1,2,3} + encoder.fc1 "
                        "weights. Injected into wm_session AFTER load-state "
                        "(so pretrained CNN overrides any random init). "
                        "Use pretrain/train_dae.py to produce.")
    parser.add_argument("--encoder-lr-mult", type=float, default=1.0,
                        help="LR multiplier on the encoder.* parameters "
                        "of wm_session. 1.0 = normal (default). 0.0 = "
                        "freeze (useful with --pretrained-cnn so the "
                        "recon/WM loss doesn't drift the encoder away "
                        "from the pretrained init). Try 0.1 for "
                        "soft freeze.")
    parser.add_argument("--save-archive", default=None,
                        help="(6) Pickle per-game per-level archive to "
                        "this path on exit. Use with --load-archive to "
                        "curriculum-resume a new agent at high-level "
                        "states the previous agent reached.")
    parser.add_argument("--load-archive", default=None,
                        help="(6) Load pickled archive at start. Lets a "
                        "fresh agent skip L1 search and start training "
                        "directly at L2 states reached by a prior agent.")
    args = parser.parse_args()

    if args.eval_mode:
        # Eval-mode overrides: heavy archive use, near-deterministic
        # policy, all auxiliary training off. The agent keeps its
        # trained policy + planner + value-head consumption but does
        # no more SIL / BC / classifier / trail updates.
        args.archive_reset_prob = args.eval_archive_reset_prob
        args.planner_policy_temperature = args.eval_policy_temperature
        args.use_sil = 0
        args.bc_planner_synthetic_r = 0.0
        args.win_trail_cap = 0
        args.value_head_train_coef = 0.0
        args.goal_states_her_prob = 0.0
        print(
            f"[eval-mode] archive_reset_prob={args.archive_reset_prob} "
            f"planner_policy_temperature={args.planner_policy_temperature} "
            f"use_sil=0 bc=0 win_trail=0 value_train=0"
        )

    arcade = Arcade()
    all_envs = arcade.get_environments()
    if args.game_prefixes:
        prefixes = [p.strip() for p in args.game_prefixes.split(",") if p.strip()]
        env_infos = [e for e in all_envs if any(e.game_id.startswith(p) for p in prefixes)]
    else:
        seen: set[str] = set()
        env_infos = []
        for e in sorted(all_envs, key=lambda x: x.game_id):
            key = e.game_id[:4]
            if key in seen:
                continue
            seen.add(key)
            env_infos.append(e)
    if args.exclude_prefixes:
        ex = [p.strip() for p in args.exclude_prefixes.split(",") if p.strip()]
        env_infos = [e for e in env_infos
                     if not any(e.game_id.startswith(p) for p in ex)]
    n_games = len(env_infos)
    K = args.forks_per_game
    M = args.macro_len
    n_lanes = n_games * K
    print(f"games: {n_games}, forks/game: {K}, lanes: {n_lanes}, "
          f"micro/macro: {M}, total env-steps: {args.steps * M * n_lanes:,}")

    # Build n_games × K envs. Lane index layout: lane_index = game_idx * K + fork_idx.
    envs: list = []
    obs_list: list = []
    avail_actions: list[list[int]] = []
    win_levels_per: list[int] = []
    for g_idx, e in enumerate(env_infos):
        for _ in range(K):
            env = arcade.make(e.game_id)
            obs = env.reset()
            envs.append(env)
            obs_list.append(obs)
            avail_actions.append(list(obs.available_actions) or [1])
            win_levels_per.append(int(obs.win_levels))

    # Synchronize fork-0 state across all K forks of each game at start.
    for g_idx in range(n_games):
        base_state = copy.deepcopy(envs[g_idx * K]._game)
        for k in range(1, K):
            envs[g_idx * K + k]._game = copy.deepcopy(base_state)

    NUM_ACTIONS = 7
    action_by_value = {int(a.value): a for a in GameAction}
    # Object-click slots extend the discrete action space: slot
    # NUM_ACTIONS + i = "ACTION6 at the centroid of the i-th largest
    # object". The policy/planner treat them as ordinary actions; the
    # mask gates them on ACTION6 availability + object existence.
    _reserved = (1 if args.reset_action else 0) + (1 if args.nav_action else 0)
    click_slots = max(0, min(int(args.object_click_slots), 18 - NUM_ACTIONS - _reserved))
    total_actions = NUM_ACTIONS + click_slots
    nav_slot = None
    if args.nav_action:
        nav_slot = total_actions
        total_actions += 1
    reset_slot = total_actions if args.reset_action else None
    if reset_slot is not None:
        total_actions += 1
    if click_slots:
        from object_features import object_centroids as _obj_centroids

    # Rule-inference goal shaping (see --rule-goal-coef). All state is
    # per-game; rules recompute lazily when new win pairs land.
    rule_goal_on = args.rule_goal_coef > 0.0 or args.rule_goal_restore
    if rule_goal_on:
        from object_features import (
            object_set as _objset,
            extract_color_rules as _extract_rules,
            predict_win_stats as _predict_win,
            config_distance as _config_dist,
            extract_relation_rules as _extract_rel,
            relation_distance as _rel_dist,
            nearest_target_distance as _waypoint_dist,
            bfs_target_distance as _bfs_dist,
            bfs_first_step as _bfs_first,
            fog_nav_step as _fog_nav,
            color_stats as _cstats,
        )

    agent_kwargs = dict(
        obs_dim=64,
        num_actions=total_actions,
        batch_size=n_lanes,
        env_ids=list(range(n_lanes)),
        seed=args.seed,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        policy_loss_watchdog_threshold=args.watchdog_threshold,
        use_adam=True,
        adam_eps=args.adam_eps,
        grad_clip_norm=args.grad_clip_norm,
        reward_homeostatic=args.reward_homeostatic,
        reward_surprise=args.reward_surprise,
        rnd_reward_alpha=args.rnd_alpha,
        entropy_beta=args.entropy_beta,
        use_ppo=bool(args.use_ppo),
        ppo_clip_eps=args.ppo_clip_eps,
        use_sil=bool(args.use_sil),
        sil_loss_coef=args.sil_loss_coef,
        sil_event_filter=bool(args.sil_event_filter),
        recon_loss_coef=args.recon_loss_coef,
        recon_visual_target=bool(args.recon_visual_target),
        extrinsic_reward_alpha=(args.goal_bonus if args.goal_bonus > 0
                                else (1.0 if args.novelty_bonus_scale > 0 else 0.0)),
        use_grpo=bool(args.use_grpo),
        advantage_normalize=bool(args.use_grpo),  # required by use_grpo
        visit_counts_max=args.visit_counts_max,
        planner_horizon=args.planner_horizon,
        planner_samples=args.planner_samples,
        planner_policy_mix=args.planner_policy_mix,
        planner_policy_temperature=args.planner_policy_temperature,
        planner_use_mcts=bool(args.planner_use_mcts),
        mcts_simulations=args.mcts_simulations,
        mcts_c_puct=args.mcts_c_puct,
        planner_noise_sigma=args.planner_noise_sigma,
        replan_surprise_mult=args.replan_surprise_mult,
        surprise_ring_capacity=args.surprise_ring_capacity,
        plasticity_interval=args.plasticity_interval,
        plasticity_shrink=args.plasticity_shrink,
        plasticity_noise=args.plasticity_noise,
        surprise_replay_mult=args.surprise_replay_mult,
        planner_sigma_alpha=args.planner_sigma_alpha,
        planner_sigma_horizon=args.planner_sigma_horizon,
        wm_stochastic=bool(args.wm_stochastic),
        wm_sigma_loss_coef=args.wm_sigma_loss_coef,
        planner_rnd_alpha=args.planner_rnd_alpha,
        planner_goal_alpha=args.planner_goal_alpha,
        goal_states_cap=args.goal_states_cap,
        value_head_train_coef=args.value_head_train_coef,
        planner_value_alpha=args.planner_value_alpha,
        value_head_gamma=args.value_head_gamma,
        value_head_buffer_capacity=args.value_head_buffer_capacity,
        value_head_train_batch=args.value_head_train_batch,
        goal_states_her_prob=args.goal_states_her_prob,
        value_head_grad_to_encoder=bool(args.value_head_grad_to_encoder),
        bc_planner_synthetic_r=args.bc_planner_synthetic_r,
        planner_change_alpha=args.planner_change_alpha,
        value_buffer_cross_game=bool(args.value_buffer_cross_game),
        goal_states_cross_game=bool(args.goal_states_cross_game),
        subgoal_k=args.subgoal_k,
        subgoal_lr=args.subgoal_lr,
        planner_subgoal_alpha=args.planner_subgoal_alpha,
        confidence_mode=bool(args.confidence_mode),
        confidence_win_increment=args.confidence_win_increment,
        confidence_novelty_drop_rate=args.confidence_novelty_drop_rate,
        confidence_novelty_threshold=args.confidence_novelty_threshold,
        planner_action_repeat=args.planner_action_repeat,
        wm_kstep_k=args.wm_kstep_k,
        wm_kstep_batch=args.wm_kstep_batch,
        wm_kstep_train_prob=args.wm_kstep_train_prob,
        wm_kstep_loss_coef=args.wm_kstep_loss_coef,
        num_options=args.num_options,
        option_horizon=args.option_horizon,
        per_option_heads=bool(args.per_option_heads),
        option_entropy_beta=args.option_entropy_beta,
        visit_count_dims=args.visit_count_dims,
        visit_count_proj_dim=args.visit_count_proj_dim,
    )
    n_channels = 2 if args.frame_diff else 1
    if args.encoder == "cnn_dqn":
        agent_kwargs.update(
            encoder_kind="cnn_dqn", encoder_channels=n_channels,
            encoder_height=64, encoder_width=64,
        )
    elif args.encoder == "cnn":
        agent_kwargs.update(
            encoder_kind="cnn", encoder_channels=n_channels,
            encoder_height=64, encoder_width=64,
        )
    agent = kindle.BatchAgent(**agent_kwargs)
    if args.load_state:
        agent.load_state(args.load_state)
        print(f"loaded prior state from {args.load_state}")
    if args.pretrained_cnn:
        agent.load_wm_checkpoint(args.pretrained_cnn)
        print(f"loaded pretrained CNN from {args.pretrained_cnn}")
    if args.encoder_lr_mult != 1.0:
        agent.set_wm_lr_multiplier("encoder.", args.encoder_lr_mult)
        print(f"set encoder LR multiplier = {args.encoder_lr_mult}")

    # Helpers ---
    _obj_tok = None
    _hybrid_tok = None
    if args.object_token or args.hybrid_token:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from object_features import object_token as _obj_tok, hybrid_token as _hybrid_tok
        if args.hybrid_token:
            print("[obs] using HYBRID token: 4x4 pixel pool (16) + 6 objects×7 (42) + 6 globals = 64 dims")
        elif args.object_token:
            print("[obs] using OBJECT-LEVEL token (8 objects + 8 globals = 64 dims)")

    # Token cache keyed by frame hash: ARC frames are mostly static
    # and K forks of a game often share frames, so the expensive
    # object/hybrid token extraction (scipy connected components per
    # call) hits the cache the vast majority of steps. Pixel-pool mode
    # is cheap but caches too (free win). Shared across lanes on
    # purpose; bounded by a hard clear at 60k entries (~15 MB).
    _token_cache: dict = {}

    def preprocess_pooled(frame_arr):
        import hashlib as _hl
        key = _hl.blake2b(
            frame_arr.astype(np.uint8).tobytes(), digest_size=8
        ).digest()
        tok = _token_cache.get(key)
        if tok is not None:
            return tok
        if args.hybrid_token:
            tok = _hybrid_tok(frame_arr.astype(np.int32)).tolist()
        elif args.object_token:
            tok = _obj_tok(frame_arr.astype(np.int32), k=8).tolist()
        else:
            arr = frame_arr.astype(np.float32) / 15.0
            tok = arr.reshape(8, 8, 8, 8).mean(axis=(1, 3)).flatten().tolist()
        if len(_token_cache) > 60000:
            _token_cache.clear()
        _token_cache[key] = tok
        return tok

    def homeo_for(frame_arr, new_levels, win_levels):
        remaining = max(0, win_levels - new_levels)
        return [
            {"value": float(remaining) * args.levels_reward_scale,
             "target": 0.0, "tolerance": 0.0},
            {"value": 1.0 - float(np.unique(frame_arr).size) / 16.0,
             "target": 0.0, "tolerance": 0.1},
        ]

    # MAX_ACTION_DIM in kindle's adapter — kindle's policy emits 18-wide
    # logits regardless of the 7 ARC-AGI-3 game actions; we'll mask the
    # unused slots to -inf so the policy never samples them.
    MAX_ACTION_DIM = 18
    mask_buf = np.ones((n_lanes, MAX_ACTION_DIM), dtype=np.float32)

    # Per-lane object-centroid cache for the click slots. Recomputed
    # only when the frame hash changes (ARC frames are mostly static,
    # so this is nearly free despite scipy label calls).
    lane_centroids = [[] for _ in range(n_lanes)]
    lane_centroid_hash = [None] * n_lanes
    # (nav-action) resolved base game-action per lane for this micro-
    # step (computed in update_action_mask, consumed by map_action).
    lane_nav_action = [None] * n_lanes

    def refresh_lane_objects():
        for lane_i in range(n_lanes):
            o = obs_list[lane_i]
            if not o.frame:
                continue
            frm = np.asarray(o.frame[0])
            h = frame_hash(frm)
            if h == lane_centroid_hash[lane_i]:
                continue
            lane_centroid_hash[lane_i] = h
            lane_centroids[lane_i] = _obj_centroids(
                frm.astype(np.int32), click_slots
            )

    def update_action_mask():
        """Rebuild per-lane action masks from current avail_actions and push
        them to kindle. Slots whose ARC GameAction.value (idx+1) is in
        avail_actions[lane] stay at 1.0; object-click slots (NUM_ACTIONS+i)
        stay at 1.0 when ACTION6 is available AND object i exists;
        everything else is forced to 0.0."""
        if click_slots:
            refresh_lane_objects()
        mask_buf.fill(0.0)
        for lane_i in range(n_lanes):
            for v in avail_actions[lane_i]:
                idx = int(v) - 1
                if 0 <= idx < MAX_ACTION_DIM:
                    mask_buf[lane_i, idx] = 1.0
            if click_slots and 6 in avail_actions[lane_i]:
                for i in range(min(click_slots, len(lane_centroids[lane_i]))):
                    mask_buf[lane_i, NUM_ACTIONS + i] = 1.0
            if (reset_slot is not None
                    and level_step[lane_i] >= args.reset_action_min_step):
                mask_buf[lane_i, reset_slot] = 1.0
            # (nav-action) resolve + gate the nav slot for this lane.
            if nav_slot is not None:
                lane_nav_action[lane_i] = None
                o = obs_list[lane_i]
                g_idx = lane_i // K
                gr = game_rules[g_idx]
                avc = avatar_color_of(g_idx)
                if (args.fog_nav and o.frame and avc is not None
                        and last_levels[lane_i] >= args.nav_force_min_level):
                    # Fog-of-war path, ISOLATED to the deep frontier (L3+):
                    # L1/L2 must bootstrap via the proven random+archive
                    # path (engaging fog-nav there lets the policy lean on
                    # it and starves the bootstrap -> evt stays ~0). Explore
                    # the maze (building a map) until the exit reveals, then
                    # route to it. Rule-free (exploration needs only avatar).
                    nav_a = fog_nav_resolve(
                        g_idx, lane_i, np.asarray(o.frame[0]).astype(np.int32),
                        avc)
                    if nav_a is not None and nav_a in avail_actions[lane_i]:
                        lane_nav_action[lane_i] = nav_a
                        mask_buf[lane_i, nav_slot] = 1.0
                    elif nav_a is not None:
                        nav_gate["not_avail"] += 1
                elif not (o.frame):
                    nav_gate["no_frame"] += 1
                elif gr is None:
                    nav_gate["no_gr"] += 1
                elif avc is None:
                    nav_gate["no_avc"] += 1
                else:
                    frm = np.asarray(o.frame[0])
                    frm_colors = set(np.unique(frm).tolist())
                    # Must-disappear targets (pickups) that are STILL
                    # PRESENT this frame. Using the rule set unfiltered
                    # would keep targeting already-collected colors (absent
                    # from the frame) — BFS then finds no path AND the
                    # exit-leg never fires, so the agent strands itself
                    # after the last pickup. Multi-leg routing falls out of
                    # this naturally: each collected pickup drops from the
                    # present set, so nav retargets the next nearest one,
                    # then the exit once none remain.
                    tgt = {c for c, r in (gr["color"] or {}).items()
                           if not r["survives"] and c != avc
                           and c in frm_colors}
                    if not tgt and gr["rel"]:
                        # exit leg: target the avatar's relational partner
                        # color (where it must end up). Prefer a SMALL,
                        # distinctive partner — the exit/goal is a compact
                        # object, whereas the maze walls (color 2 in tu93,
                        # ~thousands of cells across dozens of components)
                        # are a structural color the relation rule
                        # spuriously pairs with the ever-present avatar.
                        # Routing to walls strands the avatar adjacent to
                        # them (observed: stuck at one cell for 27 steps).
                        fi3 = frm.astype(np.int32)
                        npix = fi3.size
                        bg3 = int(np.bincount(fi3.flatten()).argmax())
                        # Always exclude the avatar's OWN sprite colors (the
                        # relation rule trivially pairs the core with its
                        # adjacent body -> "routes to itself", never moves).
                        sprite = avatar_set_of(g_idx) or {avc}
                        # Score each present relational partner. PREFER a
                        # small, distinctive object (the exit/goal), but
                        # NEVER strand: rank by (is-structural, is-bg, cells)
                        # and always pick the best available. Over-filtering
                        # to "decline" broke L2 completion (the agent could
                        # no longer reach the exit after the pickup).
                        cand = []
                        for (c1, c2) in gr["rel"]:
                            if avc not in (c1, c2):
                                continue
                            partner = c2 if c1 == avc else c1
                            if partner not in frm_colors or partner in sprite:
                                continue
                            cells = int(np.sum(fi3 == partner))
                            structural = 1 if cells > 0.03 * npix else 0
                            is_bg = 1 if partner == bg3 else 0
                            cand.append((is_bg, structural, cells, partner))
                        if cand:
                            cand.sort()
                            tgt = {cand[0][3]}
                    nav_a = nav_action_for(g_idx, frm, avc, tgt)
                    if (_NAV_DUMP and last_levels[lane_i] >= 2
                            and len(_nav_dump) < 24):
                        _nav_dump_calls[0] += 1
                        # sample every 3rd L3 decision: dense enough to see
                        # within-episode avatar movement + pickup collection.
                        if _nav_dump_calls[0] % 3 == 0:
                            all_md = sorted(
                                int(c) for c, r in (gr["color"] or {}).items()
                                if not r["survives"] and c != avc)
                            present_md = [c for c in all_md if c in frm_colors]
                            fi2 = frm.astype(np.int32)
                            md_cells = {int(c): int(np.sum(fi2 == c))
                                        for c in all_md}
                            ay, ax = np.argwhere(fi2 == avc).mean(axis=0) \
                                if (fi2 == avc).any() else (-1, -1)
                            _spr = avatar_set_of(g_idx) or {avc}
                            pdist = _bfs_dist(fi2, avc, set(tgt)) if tgt \
                                else None
                            ppos = {}
                            for c in present_md:
                                cc = np.argwhere(fi2 == c)
                                if cc.size:
                                    ppos[int(c)] = [int(cc.mean(0)[0]),
                                                    int(cc.mean(0)[1])]
                            _eff = avatar_effects(g_idx)
                            eff_tbl = {}
                            if _eff is not None:
                                _e, _en = _eff
                                eff_tbl = {int(a): [round(_e[a][0], 4),
                                                    round(_e[a][1], 4),
                                                    int(_en[a])]
                                           for a in _e}
                            _stp = _bfs_first(fi2, _spr, set(tgt)) \
                                if tgt else None
                            _nav_dump.append({
                                "lvl": int(last_levels[lane_i]),
                                "ep_step": int(ep_step[lane_i]),
                                "lane": int(lane_i),
                                "all_md": all_md,
                                "present_md": present_md,
                                "md_cells": md_cells,
                                "av_pos": [int(ay), int(ax)],
                                "pickup_pos": ppos,
                                "sprite": sorted(int(c) for c in _spr),
                                "bg": int(np.bincount(fi2.flatten()).argmax()),
                                "frame": fi2.copy(),
                                "exit_leg": len(present_md) == 0,
                                "tgt": sorted(int(t) for t in tgt),
                                "bfs_step": (list(_stp) if _stp else []),
                                "nav_a": (int(nav_a) if nav_a is not None
                                          else -1),
                                "eff": eff_tbl,
                                "bfs_ok": nav_a is not None,
                            })
                            if len(_nav_dump) >= 24:
                                import pickle as _pk
                                with open("/tmp/l3runs/nav_l3_dump.pkl",
                                          "wb") as _df:
                                    _pk.dump(_nav_dump, _df)
                                print("[nav-dump] wrote 24 L3 records")
                                _nav_dump.clear()  # rolling: latest 24
                    if nav_a is not None and nav_a in avail_actions[lane_i]:
                        lane_nav_action[lane_i] = nav_a
                        mask_buf[lane_i, nav_slot] = 1.0
                    elif nav_a is not None:
                        nav_gate["not_avail"] += 1
        agent.set_action_masks(mask_buf.reshape(-1))

    def map_action(idx, lane):
        # RESET slot: restart the current level (lc preserved). An
        # in-game action — no episode boundary, so the WM learns the
        # (deterministic) reset dynamics and the planner can plan
        # through restarts.
        if reset_slot is not None and idx == reset_slot:
            return action_by_value[0], None
        # NAV slot: the learned base action routing the avatar one BFS
        # step toward the nearest target. Resolved in update_action_mask
        # (and only unmasked when resolvable); fall back to a valid
        # action if the frame moved since.
        if nav_slot is not None and idx == nav_slot:
            nonlocal nav_uses
            na = lane_nav_action[lane]
            if na is not None and na in avail_actions[lane]:
                nav_uses += 1
                return action_by_value[na], None
            aa = avail_actions[lane]
            return action_by_value[aa[0] if aa else 1], None
        # Object-click slot: ACTION6 aimed at the i-th largest object's
        # centroid. The mask guarantees ACTION6 availability and object
        # existence at mask-build time; if the frame changed since, fall
        # back to the last known centroid (still a salient point).
        if idx >= NUM_ACTIONS:
            i = idx - NUM_ACTIONS
            cents = lane_centroids[lane]
            aa = avail_actions[lane]
            assert 6 in aa, (
                f"click-slot escape: lane {lane} sampled click slot {i} "
                f"but ACTION6 not in avail_actions={aa}"
            )
            if i < len(cents):
                y, x = cents[i]
            else:
                y, x = rng.randrange(64), rng.randrange(64)
            a = action_by_value[6]
            a.set_data({"x": int(x), "y": int(y)})
            return a, {"x": int(x), "y": int(y)}
        # Kindle now masks invalid actions before sampling, so `idx`
        # should always map to an action in `avail_actions[lane]`.
        # Assert defensively; if this fires, something else broke
        # (mask not propagated, or the mask was empty).
        target = idx + 1
        aa = avail_actions[lane]
        assert target in aa, (
            f"action mask escape: lane {lane} sampled idx={idx} "
            f"(target={target}) not in avail_actions={aa}; "
            f"masking via agent.set_action_masks must be in sync"
        )
        a = action_by_value[target]
        if a.is_complex():
            x = rng.randrange(64)
            y = rng.randrange(64)
            a.set_data({"x": x, "y": y})
            return a, {"x": x, "y": y}
        return a, None

    import random
    rng = random.Random(args.seed)

    # Per-lane frame-hash visit counts for the novelty bonus. Each lane
    # has its OWN dict — we don't share across lanes because the same
    # visual frame in different games means different things, and at
    # focused-single-game runs the lanes are different forks of the same
    # game that should each get credit for their own discoveries.
    import hashlib
    novelty_counts = [dict() for _ in range(n_lanes)]

    # Per-GAME shared frame-hash counts for the Go-Explore-style fork
    # sync selector. All forks of the same game contribute to one
    # shared dict, so the "most-novel fork in this macro" is judged
    # against the agent's GLOBAL exploration history across all forks
    # of that game. Always tracked (cheap); only consumed by the sync
    # when --explore-sync-novelty is on.
    explore_counts = [dict() for _ in range(n_games)]

    def frame_hash(frame_arr):
        # 8-byte blake2b of the raw uint8 pixel grid is plenty of bits.
        return hashlib.blake2b(frame_arr.astype(np.uint8).tobytes(),
                               digest_size=8).digest()

    # CNN visual buffer.
    if args.encoder in ("cnn", "cnn_dqn"):
        frame_mv = agent.visual_obs_memoryview()
        frame_buf = np.frombuffer(frame_mv, dtype=np.float32).reshape(
            n_lanes, n_channels, 64, 64)
    else:
        frame_buf = None
    # (8) Frame-diff: keep prev frame per lane to compute delta.
    prev_frame = (
        np.zeros((n_lanes, 64, 64), dtype=np.float32)
        if args.frame_diff else None
    )

    # Per-lane state.
    last_levels = [int(o.levels_completed) for o in obs_list]
    new_max_lvl_bump = [0] * len(obs_list)  # per-lane delta when max_lvl advances
    # Go-Explore L2 frontier: per-lane "deepest state in current episode".
    # Updated on every extrinsic event during an episode. Captures the
    # FURTHEST progress reached so far in this episode. On episode reset,
    # pushed to the per-level archive (under the entry's level), giving
    # future episodes the option to restore from the agent's high-water
    # marks rather than just L1-completion states.
    #
    # Each entry: {"env": deepcopy, "obs": ..., "score": cumulative_reward,
    #              "levels": current_level, "ep_step": ...,
    #              "avail_actions": ...}
    deepest_state = [None] * len(obs_list)
    deepest_score = [0.0] * len(obs_list)
    frontier_archive_adds = 0  # diagnostic
    # (3) Frontier-restore random burst: per-lane counter of remaining
    # forced-random steps after restoring from a deep archive entry.
    frontier_random_left = [0] * len(obs_list)
    # (7) Adaptive stagnation random: per-lane "steps since last event".
    # Resets on extrinsic event or episode boundary.
    stagnation_steps = [0] * len(obs_list)
    # (5) Per-episode max-level reached. Reset on episode boundary,
    # tracks the highest level any fork reached within current episode.
    # Aggregated to a histogram for diagnostic on log macros.
    ep_max_lvl = [0] * len(obs_list)
    # Histogram of completed-episode max-levels. Bin index = level.
    completed_ep_max_lvl_hist = [0] * 10
    # (4) Per-step lane trace file handle.
    trace_fp = None
    if args.trace_lane >= 0 and args.trace_lane < n_lanes:
        trace_path = f"/tmp/aff_runs/trace_lane_{args.trace_lane}.csv"
        trace_fp = open(trace_path, "w")
        trace_fp.write("macro,micro,ep_step,action,levels,ext_reward,frontier_rand_left\n")
        print(f"[trace] writing per-step trace of lane {args.trace_lane} to {trace_path}")
    levels_events = [0] * n_lanes
    max_levels_seen = [int(o.levels_completed) for o in obs_list]  # lifetime max per lane
    # Per-lane per-level event counter: events_by_level[lane][level]
    # = number of times this lane transitioned INTO this level.
    # E.g. events_by_level[i][1] = times reached L1, [i][2] = times
    # reached L2. Lets us see depth distribution, not just max.
    events_by_level = [collections.defaultdict(int) for _ in range(n_lanes)]
    # (L3 metrics) Per-GAME count of micro-steps spent playing each
    # level (= levels_completed while stepping). Leading indicator for
    # frontier work: events at the frontier level lag; dwell time
    # leads.
    steps_at_level = [collections.defaultdict(int) for _ in range(n_games)]
    ep_step = [0] * n_lanes
    ep_count = [0] * n_lanes

    # Per-fork macro-window cumulative reward (used for best-fork
    # selection at sync time).
    macro_return = [0.0] * n_lanes
    # Per-fork macro-window cumulative novelty score (sum of
    # 1/sqrt(explore_counts[hash]+1) per step). Used as a fallback
    # sync selector when no fork hit an extrinsic event AND
    # --explore-sync-novelty is on.
    macro_novelty = [0.0] * n_lanes

    # Per-game-per-LEVEL persistent frontier archive (Go-Explore +
    # level-stratified). archive[g_idx] is a dict {level: [entries]}.
    # Each level keeps its own queue capped at --archive-cap. On
    # restore, we pick a level uniformly from those with entries,
    # then a random entry. Once a higher level is reached even once,
    # subsequent restores have a real chance of starting there —
    # bootstrapping per-level memorization independently.
    archive = [{} for _ in range(n_games)]
    archive_uses = 0  # diagnostic: how many times we restored from archive
    archive_adds = 0  # diagnostic: how many entries added

    # (P1) Novelty-cell admission state. cell_seen[g_idx][level] is the
    # set of coarse-grid hashes ever observed at that game-level; a
    # hash NOT in the set marks a state worth archiving as a stepping
    # stone (only at the game's frontier level — completed levels are
    # already seeded by win trails). cell_last_add throttles deepcopy
    # frequency per lane in absolute micro-steps.
    cell_archive_on = args.cell_archive != 0 and args.archive_cap > 0
    cell_seen = [dict() for _ in range(n_games)]  # level -> set(bytes)
    cell_last_add = [-(10 ** 9)] * n_lanes
    cell_archive_adds = 0  # diagnostic

    # Rule-goal state. rule_pairs[g][lvl] = recent (start_objset,
    # win_objset) pairs for completed levels; game_rules[g] = cached
    # per-color rules aggregated across levels (None until >= 2 pairs).
    # ep_start_objset anchors the prediction at the CURRENT level's
    # start (reset on episode boundary AND on each level event).
    # lane_phi = previous potential for the shaping difference.
    rule_pairs = [collections.defaultdict(
        lambda: collections.deque(maxlen=50)) for _ in range(n_games)]
    game_rules = [None] * n_games
    rules_dirty = [False] * n_games
    # Steps since the lane's CURRENT level started (reset on level
    # event, episode reset, and RESET presses). Gates the RESET slot:
    # a restart only makes sense after a meaningful fraction of the
    # attempt has been spent.
    level_step = [0] * n_lanes
    # (stagnation-reset) per-lane counter of consecutive steps without
    # a novel frame-hash; cleared on novelty, level events, episode
    # boundaries, and fired RESETs.
    stale_steps = [0] * n_lanes
    stagnation_resets = 0  # diagnostic
    # Best (lowest) rule-goal potential reached this level per lane:
    # improving toward the predicted win clears the staleness clock —
    # retracing KNOWN ground toward a goal is not stagnation. Reset
    # at rule anchors (level start).
    lane_best_phi = [None] * n_lanes
    # (rule v3 / nav-action) Controllability detection via ACTION
    # CORRELATION. For each color that is a single connected blob (n==1)
    # in both frames, EMA the centroid displacement (dy,dx) produced by
    # each base action 1..NUM_ACTIONS. The avatar is the color whose
    # per-action directions are most DISTINCT (cardinal spread) — its
    # motion depends on which action was pressed. This rejects
    # deterministic drifters (e.g. tu93's budget bar shrinks left every
    # step regardless of action: high move-count but zero distinctness)
    # that a naive "most-moving color" heuristic mistakes for the avatar.
    # The chosen avatar's per-action EMA IS the inverse-dynamics table
    # the nav slot routes with — no game knowledge anywhere.
    lane_prev_cstats = [None] * n_lanes
    # color_act_eff[g][c][gid] = [dy, dx] EMA ; _n[g][c][gid] = count
    # (used for AVATAR DETECTION via per-action distinctness).
    color_act_eff = [collections.defaultdict(dict) for _ in range(n_games)]
    color_act_eff_n = [collections.defaultdict(collections.Counter)
                       for _ in range(n_games)]
    # sprite_act_eff[g][gid] = [dy, dx] EMA of the SPRITE-MASK centroid
    # (all avatar colors). Used for NAV ROUTING: the single-cell core
    # jitters within the re-orienting sprite (same action -> +dy then
    # -dy), so its per-component EMA cancels to ~0 and nav can't route;
    # the full-sprite centroid is stable. (Detection still uses the
    # per-color table; this needs the sprite known, which detection
    # supplies.)
    sprite_act_eff = [dict() for _ in range(n_games)]
    sprite_act_eff_n = [collections.Counter() for _ in range(n_games)]
    lane_prev_sprite_cen = [None] * n_lanes
    nav_uses = 0  # diagnostic
    nav_gate = collections.Counter()  # diagnostic: why nav didn't fire
    import os as _os
    _NAV_DUMP = bool(_os.environ.get("NAV_DUMP"))
    _nav_dump = []  # captured L3 nav-decision records when NAV_DUMP set
    _nav_dump_calls = [0]
    _FOG_DUMP = bool(_os.environ.get("FOG_DUMP"))
    _fog_dump = []
    _fog_dump_calls = [0]
    _avatar_cache = [None] * n_games   # (color_or_None, set, stamp)
    _avatar_calls = [0] * n_games

    def _distinctness(effs, effs_n):
        """Mean pairwise distance between the per-action unit directions
        of a color (only actions with >= 8 samples). High => the color
        moves a different way per action (avatar-like)."""
        units = []
        for gid, n in effs_n.items():
            if n < 8:
                continue
            ey, ex = effs[gid]
            mag = (ey * ey + ex * ex) ** 0.5
            if mag < 1e-9:
                continue
            units.append((ey / mag, ex / mag))
        if len(units) < 3:
            return 0.0, len(units)
        s, npair = 0.0, 0
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                dy = units[i][0] - units[j][0]
                dx = units[i][1] - units[j][1]
                s += (dy * dy + dx * dx) ** 0.5
                npair += 1
        return s / npair, len(units)

    def _recompute_avatar(g_idx):
        _avatar_calls[g_idx] += 1
        cached = _avatar_cache[g_idx]
        if cached is not None and (_avatar_calls[g_idx] - cached[2]) < 64:
            return cached
        best_c, best_score = None, 0.0
        scores = {}
        for c, effs in color_act_eff[g_idx].items():
            score, nready = _distinctness(effs, color_act_eff_n[g_idx][c])
            if nready >= 3:
                scores[c] = score
                if score > best_score:
                    best_score, best_c = score, c
        # require genuine action-correlation (drifters score ~0; clean
        # cardinal avatars score ~1.5). Threshold well clear of noise.
        avatar = best_c if best_score > 0.9 else None
        # Hysteresis: tu93's avatar sprite spans a few co-moving colors
        # (4/9/14) with near-equal distinctness, so the argmax flickers
        # between them frame-to-frame, churning the effect table and BFS
        # start cell. Keep the previous avatar if it still scores within
        # 85% of the new best — only switch on a clear improvement.
        prev = cached[0] if cached is not None else None
        if (prev is not None and prev in scores and scores[prev] > 0.9
                and scores[prev] >= 0.85 * best_score):
            avatar = prev
        # The avatar SPRITE = ALL co-moving action-correlated colors
        # (distinctness > 0.9). The primary avatar (cleanest cardinal,
        # for the effect table + BFS start) is one cell of it, but the
        # sprite's other colors box it in — BFS must treat the whole
        # sprite as passable or it can never leave the start cell.
        av_set = {c for c, s in scores.items() if s > 0.9}
        if avatar is not None:
            av_set.add(avatar)
        result = (avatar, av_set, _avatar_calls[g_idx])
        _avatar_cache[g_idx] = result
        return result

    def avatar_color_of(g_idx):
        return _recompute_avatar(g_idx)[0]

    def avatar_set_of(g_idx):
        return _recompute_avatar(g_idx)[1]

    def avatar_effects(g_idx):
        av = avatar_color_of(g_idx)
        if av is None:
            return None
        # Prefer the STABLE sprite-centroid inverse-dynamics once it has
        # >= 3 calibrated actions; the per-color core table jitters and
        # cancels to ~0 (see sprite_act_eff comment). Fall back to the
        # core table while the sprite table is still warming up.
        sn = sprite_act_eff_n[g_idx]
        if sum(1 for a in sn if sn[a] >= 8) >= 3:
            return sprite_act_eff[g_idx], sprite_act_eff_n[g_idx]
        return color_act_eff[g_idx][av], color_act_eff_n[g_idx][av]

    def action_effects_ready(g_idx):
        # The avatar is only selected once its per-action effects are
        # calibrated, so a non-None avatar IS ready by construction.
        return avatar_color_of(g_idx) is not None

    def nav_action_for(g_idx, frm, av_c, targets):
        """Base action (GameAction value) whose learned avatar
        displacement best matches the BFS first step toward the nearest
        target. None when no reachable target / effects uncalibrated."""
        if av_c is None:
            nav_gate["no_avc"] += 1
            return None
        if not targets:
            nav_gate["no_tgt"] += 1
            return None
        eff = avatar_effects(g_idx)
        if eff is None:
            nav_gate["ae_notready"] += 1
            return None
        effs, effs_n = eff
        # Pass the whole avatar sprite as passable (the core color is
        # boxed in by its own body colors otherwise — see analyze_dump).
        av_set = avatar_set_of(g_idx) or {av_c}
        step = _bfs_first(frm.astype(np.int32), av_set, set(targets))
        if step is None:
            nav_gate["no_bfs"] += 1
            return None
        sy, sx = step
        best_a, best_dot = None, -1e9
        for a, n in effs_n.items():
            if n < 8:
                continue
            ey, ex = effs[a]
            mag = (ey * ey + ex * ex) ** 0.5
            if mag < 1e-6:
                continue
            dot = (sy * ey + sx * ex) / mag  # cosine-ish (step is unit)
            if dot > best_dot:
                best_dot, best_a = dot, a
        # require the best action to actually point the right way
        if best_dot > 0.3:
            nav_gate["resolved"] += 1
            return best_a
        nav_gate["low_cos"] += 1
        return None

    def action_for_step(g_idx, step):
        """Pick the base action whose learned (sprite-centroid) effect
        best matches a desired (dy,dx) step. Shared by BFS and fog-nav."""
        if step is None:
            return None
        eff = avatar_effects(g_idx)
        if eff is None:
            return None
        effs, effs_n = eff
        sy, sx = step
        best_a, best_dot = None, -1e9
        for a, n in effs_n.items():
            if n < 8:
                continue
            ey, ex = effs[a]
            mag = (ey * ey + ex * ex) ** 0.5
            if mag < 1e-6:
                continue
            dot = (sy * ey + sx * ex) / mag
            if dot > best_dot:
                best_dot, best_a = dot, a
        return best_a if best_dot > 0.3 else None

    # --- Fog-of-war map state (per lane), reset on level/episode change.
    # tu93 only ever shows ~15% of the maze; unexplored fog reads as the
    # SAME color as floor and walls reveal on approach, so single-frame
    # BFS plans through fog. We accumulate a persistent map and route on
    # it: explore frontiers until the exit is revealed, then go to it.
    # Maps are PERSISTENT per (game, level), NOT per-episode: the maze at
    # a given level is deterministic (verified: identical across all human
    # demos), so accumulating revealed+bumped walls over many attempts
    # progressively maps the whole level and lets the agent route to the
    # exit even though a single ~48-step attempt can't cover it. Generic
    # spatial memory (remember the maze you've seen), NOT path-memorizing.
    gl_wall = {}                          # (g_idx, lvl) -> bool 64x64
    gl_expl = {}                          # (g_idx, lvl) -> bool 64x64
    gl_goal = {}                          # (g_idx, lvl) -> exit centroid
    lane_prev_av = [None] * n_lanes       # prev avatar centroid (fog-nav)
    lane_prev_step = [None] * n_lanes     # prev fog-nav (dy,dx) issued
    lane_stuck = [0] * n_lanes            # consecutive no-move count
    fog_uses = 0  # diagnostic

    def fog_nav_resolve(g_idx, lane_i, frm, avc):
        """Accumulate the persistent map from this frame and return the
        base action toward the exit (if visible/remembered) else toward
        the nearest unexplored frontier. frm is int32 64x64."""
        nonlocal fog_uses
        H, W = frm.shape
        # PERSISTENT per-(game,level) map (accumulates across attempts).
        key = (g_idx, int(last_levels[lane_i]))
        if key not in gl_wall:
            gl_wall[key] = np.zeros((H, W), dtype=bool)
            gl_expl[key] = np.zeros((H, W), dtype=bool)
        wmap = gl_wall[key]
        emap = gl_expl[key]
        gr = game_rules[g_idx]
        sprite = avatar_set_of(g_idx) or {avc}
        bg = int(np.bincount(frm.flatten()).argmax())
        tgtcolors = set()
        if gr:
            tgtcolors = {int(c) for c, r in (gr["color"] or {}).items()
                         if not r["survives"] and c != avc}
        # WALLS = LARGE structural colors only. A color is a wall if it
        # is non-floor, non-avatar, non-target AND covers many cells
        # (>16). Small/rare non-floor colors (the exit marker, items) are
        # NOT walls — otherwise, BEFORE rules exist, the exit color is
        # treated as a wall and the agent routes AROUND it, never wins,
        # never forms rules (deadlock). Generic: structure is big, goals
        # are small.
        counts = np.bincount(frm.flatten(), minlength=16)
        wall_colors = {int(c) for c in range(len(counts))
                       if counts[c] > 16 and c != bg
                       and c not in sprite and c not in tgtcolors}
        wmap |= np.isin(frm, list(wall_colors)) if wall_colors else \
            np.zeros((H, W), dtype=bool)
        emap |= (frm != bg)                       # non-floor = revealed
        avm = np.isin(frm, list(sprite))
        avcells = np.argwhere(avm)
        for (y, x) in avcells:                     # reveal radius around avatar
            emap[max(0, y - 6):y + 7, max(0, x - 6):x + 7] = True
        # BUMP DETECTION: walls can be INVISIBLE — a cell may be a wall
        # in-game yet render as floor-color (fog), so BFS routes into it,
        # the avatar bumps and does NOT move, but the map never learns the
        # wall -> permanent stuck at a junction (observed: 26+ steps frozen
        # at one cell). If the avatar didn't move since the last fog step,
        # the cell one step beyond it in that direction is a wall: mark it.
        if len(avcells):
            ay = int(avcells[:, 0].mean())
            ax = int(avcells[:, 1].mean())
            pv = lane_prev_av[lane_i]
            ps = lane_prev_step[lane_i]
            moved = pv is None or abs(ay - pv[0]) + abs(ax - pv[1]) > 2
            if moved:
                lane_stuck[lane_i] = 0
            elif ps is not None:
                # require 2 consecutive no-moves before marking a wall
                # (one stall could be a policy step, not a real bump);
                # the map persists, so false walls are costly.
                lane_stuck[lane_i] += 1
                if lane_stuck[lane_i] >= 2:
                    dy, dx = ps
                    by, bx = ay + dy * 5, ax + dx * 5  # one cell-step beyond
                    wmap[max(0, by - 1):by + 2, max(0, bx - 1):bx + 2] = True
        else:
            ay = ax = -1
        # TARGET (exit): the rule-known must-disappear color when known;
        # else any small/rare non-floor color present (candidate exit) so
        # the agent BEELINES to it pre-rules and bootstraps the win.
        if tgtcolors:
            tmask = np.isin(frm, list(tgtcolors))
        else:
            cand = (frm != bg) & (~avm) & (~np.isin(frm, list(wall_colors)))
            tmask = cand
        if tmask.any():
            cc = np.argwhere(tmask)
            gl_goal[key] = (int(cc[:, 0].mean()), int(cc[:, 1].mean()))
        elif gl_goal.get(key) is not None:
            gy, gx = gl_goal[key]
            tmask = np.zeros((H, W), dtype=bool)
            tmask[gy, gx] = True
        lane_prev_av[lane_i] = (ay, ax) if ay >= 0 else None
        step = _fog_nav(wmap, emap, avm,
                        tmask if tmask.any() else None)
        lane_prev_step[lane_i] = step
        if step is None:
            nav_gate["fog_nostep"] += 1
            return None
        a = action_for_step(g_idx, step)
        if a is None:
            nav_gate["fog_lowcos"] += 1
            return None
        nav_gate["fog_resolved"] += 1
        fog_uses += 1
        if _FOG_DUMP:
            _fog_dump_calls[0] += 1
            if _fog_dump_calls[0] % 5 == 0 and len(_fog_dump) < 200:
                exit_seen = bool(tgtcolors) and \
                    bool(np.isin(frm, list(tgtcolors)).any())
                avc2 = np.argwhere(avm)
                _fog_dump.append({
                    "lane": int(lane_i), "ep_step": int(ep_step[lane_i]),
                    "av": [int(avc2[:, 0].mean()), int(avc2[:, 1].mean())]
                    if len(avc2) else [-1, -1],
                    "explored": int(emap.sum()), "wall": int(wmap.sum()),
                    "mode": "exploit" if tmask.any() else "explore",
                    "exit_seen": exit_seen,
                    "goal": list(gl_goal[key]) if gl_goal.get(key) else None,
                })
                if len(_fog_dump) >= 200:
                    import pickle as _pk
                    with open("/tmp/l3runs/fog_dump.pkl", "wb") as _df:
                        _pk.dump(_fog_dump, _df)
                    print("[fog-dump] wrote 200 L3 fog records")
        return a
    ep_start_objset = [None] * n_lanes
    lane_pred = [None] * n_lanes      # predicted win stats for this episode
    lane_phi = [None] * n_lanes
    lane_objset_hash = [None] * n_lanes
    lane_objset_cache = [None] * n_lanes
    rule_shaping_total = 0.0  # diagnostic
    rule_pairs_count = 0      # diagnostic

    def lane_objset(i, frm):
        h = frame_hash(frm)
        if h != lane_objset_hash[i]:
            lane_objset_hash[i] = h
            # cap 64 (not the library default 16): tu93 has ~32 wall
            # segments + floor + budget bar that crowd out the small
            # (1-cell) avatar under a 16-object area cap, making it
            # invisible to avatar detection and rule inference alike.
            lane_objset_cache[i] = _objset(frm.astype(np.int32),
                                           max_objects=64)
        return lane_objset_cache[i]

    def refresh_rules(g_idx):
        all_pairs = []
        for lvl_pairs in rule_pairs[g_idx].values():
            all_pairs.extend(lvl_pairs)
        if len(all_pairs) >= 2:
            game_rules[g_idx] = {
                "color": _extract_rules(all_pairs),
                # v2: pairwise-relational win constraints — arrangement
                # rules ("c1 ends at offset (dy,dx) from c2") that hold
                # across wins regardless of starting layout.
                "rel": _extract_rel(all_pairs),
            }
        else:
            game_rules[g_idx] = None
        rules_dirty[g_idx] = False

    def reset_rule_anchor(i, frm):
        """Anchor the rule prediction at a fresh level start for lane
        i. Called on episode boundary and after each level event."""
        g_idx = i // K
        if rules_dirty[g_idx]:
            refresh_rules(g_idx)
        ep_start_objset[i] = lane_objset(i, frm)
        lane_phi[i] = None
        lane_best_phi[i] = None
        rules = game_rules[g_idx]
        if rules:
            lane_pred[i] = {
                "pred": _predict_win(ep_start_objset[i], rules["color"]),
                "rel": rules["rel"],
            }
        else:
            lane_pred[i] = None
    if args.load_archive:
        try:
            try:
                import cloudpickle as _pkl
            except ImportError:
                import pickle as _pkl
            with open(args.load_archive, "rb") as _f:
                loaded = _pkl.load(_f)
            # Only restore entries for games we have envs for.
            if isinstance(loaded, list) and len(loaded) >= n_games:
                for g in range(n_games):
                    if isinstance(loaded[g], dict):
                        archive[g] = loaded[g]
                total = sum(len(lst) for ag in archive for lst in ag.values())
                print(f"[archive] loaded {total} entries from {args.load_archive}")
        except Exception as _exc:
            print(f"[archive] load failed: {_exc}; starting empty")

    # #2 Full-trajectory archive: per-lane rolling trail of env snapshots
    # captured at micro-step granularity. When an extrinsic event (level
    # event) fires, the whole trail flushes into the per-game archive —
    # so EVERY state along the winning trajectory becomes a restorable
    # frontier, not just the macro-sync snapshot. Trail is cleared on
    # episode boundary (without an event). Disabled when --win-trail-cap=0.
    win_trail_on = args.win_trail_cap > 0
    win_trails = (
        [collections.deque(maxlen=args.win_trail_cap) for _ in range(n_lanes)]
        if win_trail_on else None
    )
    trail_step_counter = [0] * n_lanes
    trail_archive_adds = 0  # diagnostic

    # Action-skill bank (per game): action sequences captured from
    # winning level transitions. Recent action history is kept in a
    # deque per lane and snapshotted on level-events. At planner time
    # with probability --skill-replay-prob a random captured sequence
    # is pushed onto the planner_queue, prefixing the planner-chosen
    # actions. Acts as a learned macro / option mechanism — compresses
    # action sequences into reusable templates the agent can recombine.
    skills_on = args.skill_replay_prob > 0.0
    skill_bank = [collections.deque(maxlen=args.skill_bank_cap) for _ in range(n_games)]
    recent_actions = [collections.deque(maxlen=args.skill_capture_len) for _ in range(n_lanes)]
    skill_replays = 0  # diagnostic

    # Intrinsic progress signals (Term 1 + Term 2 + Term 3).
    # Term 1: persistent configurational delta from episode-start.
    # Term 2: per-episode coarse-state entropy growth.
    # Term 3: planner-rollout empowerment.
    # All reset on episode boundary except empowerment (updated each
    # planner call, lives until the next).
    progress_on = (args.progress_change_coef > 0.0
                   or args.progress_diversity_coef > 0.0
                   or args.progress_empowerment_coef > 0.0)
    last_empowerment = [0.0] * n_lanes
    # Per-env (game) running mean of empowerment so values are normalized
    # WITHIN each game before being used as reward. Without this,
    # responsive games (sp80) hog empowerment magnitude and slower
    # games (tu93) get effectively zero reward — see 2026-05-15
    # cross-game encoder bias finding.
    emp_per_game_mean = {}  # env_id -> running mean
    emp_per_game_ema = 0.05  # EMA factor
    pwin = max(1, args.progress_persistence_window)
    pgs = max(2, args.progress_grid_size)  # coarse grid side
    ep_start_cells = [None] * n_lanes      # 2D coarse grids
    recent_cells = [collections.deque(maxlen=pwin) for _ in range(n_lanes)]
    ep_unique_cells = [set() for _ in range(n_lanes)]
    progress_total = 0.0  # diagnostic: cumulative progress reward

    def coarse_grid_from_frame(frame_2d, side):
        # frame_2d: (H, W) numpy array (typically 64×64).
        # Average-pool into (side, side) then round to int.
        H, W = frame_2d.shape
        bh, bw = H // side, W // side
        if bh == 0 or bw == 0:
            return None
        h_used = bh * side
        w_used = bw * side
        view = frame_2d[:h_used, :w_used].reshape(side, bh, side, bw)
        cells = view.mean(axis=(1, 3))
        return np.round(cells).astype(np.int32)

    t0 = time.time()
    last_log_t = t0
    syncs = 0

    for macro_step in range(args.steps):
        # Reset macro-window accumulators at start of macro window.
        for i in range(n_lanes):
            macro_return[i] = 0.0
            macro_novelty[i] = 0.0

        # Run M micro-steps.
        for micro_step in range(M):
            # Reset any lanes that hit terminal/forced limit.
            for i in range(n_lanes):
                obs = obs_list[i]
                state = obs.state
                # (4) Per-lane episode budget scales with max_lvl reached.
                # Lanes that have progressed further get more steps per
                # episode to find completion-length action sequences.
                lane_budget = (
                    args.max_episode_steps
                    + args.episode_steps_per_level * int(max_levels_seen[i])
                )
                need_reset = (
                    state in (GameState.NOT_PLAYED, GameState.GAME_OVER)
                    or state is GameState.WIN
                    or ep_step[i] >= lane_budget
                )
                _g_sr = i // K
                _frontier_sr = max(
                    max_levels_seen[_g_sr * K:(_g_sr + 1) * K]
                )
                _g_events_sr = sum(
                    levels_events[_g_sr * K + kk] for kk in range(K)
                )
                if (args.stagnation_reset > 0
                        and _g_events_sr >= 20
                        and stale_steps[i] >= args.stagnation_reset
                        and last_levels[i] >= _frontier_sr
                        and state not in (GameState.NOT_PLAYED,
                                          GameState.GAME_OVER,
                                          GameState.WIN)):
                    # In-game level restart: the lane produced nothing
                    # novel for N steps — the doomed-state signature
                    # (e.g. depleted budget). Restart the LEVEL
                    # (levels_completed preserved), keep the episode.
                    try:
                        obs_r = envs[i].step(action_by_value[0])
                    except Exception:
                        obs_r = None
                    if obs_r is not None and obs_r.frame:
                        obs_list[i] = obs_r
                        if list(obs_r.available_actions):
                            avail_actions[i] = list(obs_r.available_actions)
                        last_levels[i] = int(obs_r.levels_completed)
                        agent.mark_boundary(i)
                        level_step[i] = 0
                        stale_steps[i] = 0
                        stagnation_resets += 1
                        if rule_goal_on and obs_list[i].frame:
                            reset_rule_anchor(
                                i, np.asarray(obs_list[i].frame[0])
                            )
                        if win_trail_on:
                            win_trails[i].clear()
                if need_reset:
                    if ep_step[i] > 0:
                        ep_count[i] += 1
                        # (5) Record completed-episode max-level.
                        lvl_bin = min(ep_max_lvl[i], len(completed_ep_max_lvl_hist) - 1)
                        completed_ep_max_lvl_hist[lvl_bin] += 1
                        ep_max_lvl[i] = 0
                    # (7) Reset stagnation counter on episode reset.
                    stagnation_steps[i] = 0
                    # (2) Go-Explore frontier flush: if this episode
                    # reached a high-progress state, push it to the
                    # per-level archive under its level. Restoring
                    # from these high-water marks lets future episodes
                    # CONTINUE exploration from L2's interior rather
                    # than always starting fresh at L1.
                    if args.frontier_archive_on and deepest_state[i] is not None:
                        g_idx_d = i // K
                        ent = deepest_state[i]
                        ent_lvl = int(ent["levels"])
                        gar = archive[g_idx_d].setdefault(ent_lvl, [])
                        cap = args.archive_cap
                        if cap > 0:
                            if len(gar) < cap:
                                gar.append(ent)
                                frontier_archive_adds += 1
                            else:
                                worst_idx = min(
                                    range(len(gar)),
                                    key=lambda j: gar[j]["novelty"],
                                )
                                if ent["novelty"] > gar[worst_idx]["novelty"]:
                                    gar[worst_idx] = ent
                                    frontier_archive_adds += 1
                        deepest_state[i] = None
                        deepest_score[i] = 0.0
                    # Archive-reset: with prob, restore from a random
                    # archived frontier state instead of a fresh env
                    # reset. Real Ecoffet Go-Explore: "return then
                    # explore" — start episodes from interesting
                    # discovered waypoints, not always from level-0.
                    g_idx = i // K
                    used_archive = False
                    # Pick a level from those with entries. Weighted by
                    # `level_num + 1` so higher levels are favored —
                    # restoring at L3 is what we want, since L3 attempts
                    # are precious (each one is a chance to actually
                    # break into L4). Uniform weighting (the previous
                    # behaviour) wasted episode budget on L1/L2 returns
                    # when those levels were already saturated.
                    levels_with = [
                        lvl for lvl, lst in archive[g_idx].items() if lst
                    ]
                    if (args.archive_cap > 0
                            and levels_with
                            and rng.random() < args.archive_reset_prob):
                        # Exponential weighting by level — higher levels
                        # get exponentially more restores. Default base
                        # 2: L1=2, L2=4, L3=8, L4=16. With L1,L2,L3
                        # populated → L3 gets 8/14 ≈ 57% of restores.
                        base = args.archive_level_weight_base
                        weights = [base ** float(lvl) for lvl in levels_with]
                        chosen_lvl = rng.choices(levels_with, weights=weights, k=1)[0]
                        gar = archive[g_idx][chosen_lvl]
                        entry = gar[rng.randrange(len(gar))]
                        # Rule-goal restore ranking: at the frontier
                        # level with rules available, sample a few
                        # candidates and restore the one closest to
                        # the predicted win configuration (predictions
                        # anchored at each entry's own frame — exact
                        # for absolute-target rules, neutral for
                        # relative ones).
                        game_events = sum(
                            levels_events[g_idx * K + kk] for kk in range(K)
                        )
                        if (args.restore_value_rank
                                and game_events >= 20
                                and rng.random() < 0.5
                                and chosen_lvl == max(
                                    max_levels_seen[g_idx * K:(g_idx + 1) * K])
                                and len(gar) > 1):
                            # V-rank only once the classifier has real
                            # win labels (>= 20 events) — before that
                            # stored V is noise and max-of-12 selection
                            # concentrates restores on garbage entries
                            # (verified: 12x slower bootstrap). 50/50
                            # mix with uniform keeps frontier diversity.
                            # Prefer restoring states the win classifier
                            # scored as winning-trajectory-like at
                            # admission. With depth-biased admission the
                            # uniform choice prefers doomed late-budget
                            # states; V is the learned antidote.
                            best_v, best_e = None, entry
                            for _cand in range(min(12, len(gar))):
                                e2 = gar[rng.randrange(len(gar))]
                                v2 = e2.get("v", 0.0)
                                if best_v is None or v2 > best_v:
                                    best_v, best_e = v2, e2
                            entry = best_e
                        if (args.rule_goal_restore and rule_goal_on
                                and game_rules[g_idx]
                                and chosen_lvl == max(
                                    max_levels_seen[g_idx * K:(g_idx + 1) * K])):
                            best_d, best_e = None, entry
                            for _cand in range(min(12, len(gar))):
                                e2 = gar[rng.randrange(len(gar))]
                                if not e2["obs"].frame:
                                    continue
                                fr2 = np.asarray(e2["obs"].frame[0])
                                os2 = _objset(fr2.astype(np.int32))
                                gr = game_rules[g_idx]
                                d2 = _config_dist(
                                    os2,
                                    _predict_win(os2, gr["color"]),
                                )
                                if gr["rel"]:
                                    d2 = 0.5 * d2 + 0.5 * _rel_dist(
                                        os2, gr["rel"]
                                    )
                                if best_d is None or d2 < best_d:
                                    best_d, best_e = d2, e2
                            entry = best_e
                        envs[i] = copy.deepcopy(entry["env"])
                        obs_list[i] = entry["obs"]
                        last_levels[i] = entry["levels"]
                        avail_actions[i] = list(entry["avail_actions"])
                        ep_step[i] = entry["ep_step"]
                        archive_uses += 1
                        used_archive = True
                        # (3) Trigger frontier-restore random burst when
                        # restoring at a level >= 2 (skip L1 restores).
                        # We want exploration at the unknown frontier.
                        if (args.frontier_random_steps > 0
                                and entry["levels"] >= 2):
                            frontier_random_left[i] = args.frontier_random_steps
                    if not used_archive:
                        obs_list[i] = envs[i].reset()
                        avail_actions[i] = list(obs_list[i].available_actions) or avail_actions[i]
                        last_levels[i] = int(obs_list[i].levels_completed)
                        ep_step[i] = 0
                    level_step[i] = 0
                    agent.mark_boundary(i)
                    if rule_goal_on and obs_list[i].frame:
                        reset_rule_anchor(
                            i, np.asarray(obs_list[i].frame[0])
                        )
                    if win_trail_on:
                        win_trails[i].clear()
                    if progress_on:
                        if obs_list[i].frame:
                            frm = np.asarray(obs_list[i].frame[0], dtype=np.float32)
                            ep_start_cells[i] = coarse_grid_from_frame(frm, pgs)
                            ep_unique_cells[i] = set()
                            if ep_start_cells[i] is not None:
                                ep_unique_cells[i].add(ep_start_cells[i].tobytes())
                        else:
                            ep_start_cells[i] = None
                            ep_unique_cells[i] = set()
                        recent_cells[i].clear()

            pooled = [preprocess_pooled(np.asarray(o.frame[0], dtype=np.float32))
                      for o in obs_list]
            # NOTE (2026-06-10): frames are now written to the visual
            # slot just before agent.observe() below, from new_obs_list
            # (the POST-action frames), matching Agent::observe's
            # post-action convention. Writing the PRE-action frames
            # here (the old behaviour) made the encoder lag the world
            # by one step: act() and the planner ran on the previous
            # frame's latent, and the WM trained on misaligned
            # (frame, action) pairs.

            # Push current per-lane action masks to kindle BEFORE act();
            # avail_actions can change after each env.step() so we
            # refresh every micro-step.
            update_action_mask()
            # Run the model-based planner. No-op when planner_horizon == 0
            # OR when every lane's queue still has actions to play. When
            # the queue is empty, the planner samples planner_samples
            # random valid-action sequences of length planner_horizon,
            # rolls them through the WM, scores each by latent-visit-
            # count novelty, and queues the best one. act() then plays
            # the queued action instead of sampling from the policy.
            if args.planner_horizon > 0:
                agent.plan_and_queue(total_actions)
                if progress_on and args.progress_empowerment_coef > 0.0:
                    raw_emp = agent.empowerment()
                    # Per-game normalization: divide each lane's
                    # empowerment by the EMA of its game's recent
                    # values. Ratio 1.0 = "average for this game".
                    # Result is comparable across games regardless of
                    # the game's intrinsic action-effect magnitude.
                    last_empowerment = [0.0] * n_lanes
                    for i, e in enumerate(raw_emp):
                        if e <= 0:
                            continue
                        env_id = env_infos[i // K].game_id
                        prev = emp_per_game_mean.get(env_id, e)
                        new_mean = (1.0 - emp_per_game_ema) * prev + emp_per_game_ema * e
                        emp_per_game_mean[env_id] = new_mean
                        # Normalize, clip ratio to [0, 3] so a single
                        # outlier can't dominate.
                        if new_mean > 1e-8:
                            last_empowerment[i] = min(e / new_mean, 3.0)
                        else:
                            last_empowerment[i] = 0.0
            # Skill replay: prefix planner_queue with a random captured
            # action sequence per lane, with given probability. The agent
            # then COMMITS to that sequence before any new planning.
            if skills_on and args.planner_horizon > 0:
                for i in range(n_lanes):
                    if rng.random() >= args.skill_replay_prob:
                        continue
                    g_idx = i // K
                    if not skill_bank[g_idx]:
                        continue
                    seq = rng.choice(list(skill_bank[g_idx]))
                    agent.queue_actions(i, list(seq))
                    skill_replays += 1
            actions = agent.act(pooled)

            # (3) Frontier-restore random override. When a lane was just
            # restored from a high-score archive entry, force pure random
            # actions for the next N steps. Reasoning: at deep L2 frontier
            # states the planner/policy have NEVER trained on this region
            # (the value/goal/change signals point at L1-known territory),
            # so the planner exploits backward. A random burst forces
            # exploration of the L2 frontier neighborhood.
            if args.frontier_random_steps > 0 or args.stagnation_random_after > 0:
                actions = list(actions)
                for li in range(n_lanes):
                    use_random = False
                    if frontier_random_left[li] > 0:
                        use_random = True
                        frontier_random_left[li] -= 1
                    elif (args.stagnation_random_after > 0
                          and stagnation_steps[li] >= args.stagnation_random_after):
                        use_random = True
                    if use_random:
                        avail = avail_actions[li]
                        # avail holds 1-based action values; the agent's
                        # action index is value-1 (see map_action).
                        actions[li] = avail[rng.randrange(len(avail))] - 1

            # (nav) Frontier-nav commitment. On a lane at the deepest-seen
            # level (the unsolved frontier, e.g. L3) where nav resolved
            # this step, force the nav action with prob p. The frontier
            # has no reward gradient yet, so the policy under-uses nav
            # there; committing to the computed route is what completes a
            # multi-leg pickup->exit run within budget. Generic assist.
            if (nav_slot is not None and args.nav_force_frontier > 0.0):
                actions = list(actions)
                for li in range(n_lanes):
                    if lane_nav_action[li] is None:
                        continue
                    g_idx_f = li // K
                    frontier_f = max(
                        max_levels_seen[g_idx_f * K:(g_idx_f + 1) * K]
                    )
                    # Only force on a DEEP frontier (>= nav_force_min_level).
                    # L1/L2 must bootstrap with the proven random+archive
                    # mechanisms — forcing fog-nav there starves them (they
                    # win the early levels that form the rules fog-nav needs)
                    # and the agent stalls at L0 (evt=0). Force where
                    # navigation is the actual bottleneck: L3+.
                    if (last_levels[li] >= frontier_f
                            and frontier_f >= args.nav_force_min_level
                            and rng.random() < args.nav_force_frontier):
                        actions[li] = nav_slot

            # Record action history per lane for skill capture on
            # level-up below.
            if skills_on:
                for i in range(n_lanes):
                    recent_actions[i].append(int(actions[i]))

            new_obs_list = []
            homeo_list = []
            new_pooled = []
            level_deltas = [0] * n_lanes
            lane_values = agent.values()
            for i in range(n_lanes):
                # #2 Trail snapshot: capture state BEFORE the step. If
                # this step turns out to be a winning one, the trail
                # holds the lead-up states for archive flush below.
                if win_trail_on and trail_step_counter[i] % args.win_trail_stride == 0:
                    pre_obs = obs_list[i]
                    pre_state = pre_obs.state.name if hasattr(pre_obs.state, "name") else str(pre_obs.state)
                    if pre_state not in ("GAME_OVER", "WIN", "NOT_PLAYED"):
                        win_trails[i].append({
                            "env": copy.deepcopy(envs[i]),
                            "obs": pre_obs,
                            "levels": last_levels[i],
                            "ep_step": ep_step[i],
                            "avail_actions": list(avail_actions[i]),
                            "v": float(lane_values[i]),
                        })
                trail_step_counter[i] += 1

                ga, ad = map_action(int(actions[i]), i)
                try:
                    obs_new = envs[i].step(ga, data=ad)
                except Exception:
                    obs_new = None
                # local_wrapper.step() may catch internal IndexError /
                # AttributeError from the game module and return None
                # (or some games return None to signal an unrecoverable
                # internal state). Treat that as a forced reset.
                if obs_new is None:
                    obs_new = envs[i].reset()
                    avail_actions[i] = list(obs_new.available_actions) or avail_actions[i]
                    agent.mark_boundary(i)
                    if rule_goal_on and obs_new.frame:
                        reset_rule_anchor(i, np.asarray(obs_new.frame[0]))
                    if win_trail_on:
                        win_trails[i].clear()
                new_obs_list.append(obs_new)
                if list(obs_new.available_actions):
                    avail_actions[i] = list(obs_new.available_actions)
                new_levels = int(obs_new.levels_completed)
                d = new_levels - last_levels[i]
                level_deltas[i] = d
                if d > 0:
                    levels_events[i] += d
                    macro_return[i] += float(d)  # cheap proxy for "did good things happen"
                    # (7) Reset stagnation counter on any extrinsic event.
                    stagnation_steps[i] = 0
                    # (5) Track per-episode max level reached.
                    if new_levels > ep_max_lvl[i]:
                        ep_max_lvl[i] = new_levels
                    # (1) One-shot max_lvl-bump bonus: when this event
                    # advances max_levels_seen for this lane, mark it for
                    # an extra reward injection below. This makes the
                    # FIRST L2 win, FIRST L3 win, etc. much more
                    # attractive than intermediate progress rewards —
                    # the agent learns to *pursue level transitions*
                    # rather than saturate at partial progress.
                    new_max_lvl_bump[i] = 0
                    if new_levels > max_levels_seen[i]:
                        new_max_lvl_bump[i] = new_levels - max_levels_seen[i]
                        max_levels_seen[i] = new_levels
                    # (2) Go-Explore frontier: this is a high-progress
                    # moment in the current episode. Snapshot the state.
                    # Score = cumulative_reward proxy (current level).
                    # On episode reset this snapshot will be added to
                    # the archive so future episodes can restore from
                    # the FURTHEST point this lane has reached.
                    if args.frontier_archive_on:
                        # Score = base + level + ep_step. Including ep_step
                        # means DEEPER moments in an L2 episode displace
                        # SHALLOWER ones — the archive grows toward
                        # L2-completion-vicinity rather than saturating
                        # at L2-start replays.
                        score_now = (
                            500.0
                            + 100.0 * float(new_levels)
                            + float(ep_step[i] + 1)
                        )
                        if score_now > deepest_score[i]:
                            deepest_score[i] = score_now
                            deepest_state[i] = {
                                "env": copy.deepcopy(envs[i]),
                                "obs": obs_new,
                                "novelty": score_now,
                                "levels": int(new_levels),
                                "ep_step": int(ep_step[i] + 1),
                                "avail_actions": list(obs_new.available_actions or [1]),
                                "v": float(lane_values[i]),
                            }
                    # Tally per-level. The delta is usually 1; for
                    # delta>1 (rare multi-level jump) credit each
                    # transitioned level once.
                    for lvl_into in range(last_levels[i] + 1, new_levels + 1):
                        events_by_level[i][lvl_into] += 1
                    # #2 Trail flush: a winning step just happened.
                    # Push the entire trail into the per-game archive
                    # with a strong novelty score (1500 + macro_return)
                    # so these entries rank above ordinary frontier
                    # snapshots and survive eviction longer.
                    if win_trail_on and args.archive_cap > 0:
                        g_idx = i // K
                        flush_score = 1500.0 + float(d) * 1000.0
                        for entry in win_trails[i]:
                            e = dict(entry)
                            e["novelty"] = flush_score
                            # Bucket by the LEVEL THIS ENTRY WAS AT (so
                            # restoring this entry resumes that level).
                            entry_lvl = e.get("levels", 0)
                            gar = archive[g_idx].setdefault(entry_lvl, [])
                            if len(gar) < args.archive_cap:
                                gar.append(e)
                                archive_adds += 1
                                trail_archive_adds += 1
                            else:
                                worst_idx = min(
                                    range(len(gar)),
                                    key=lambda j: gar[j]["novelty"],
                                )
                                if e["novelty"] > gar[worst_idx]["novelty"]:
                                    gar[worst_idx] = e
                                    archive_adds += 1
                                    trail_archive_adds += 1
                        win_trails[i].clear()
                    # Skill capture: snapshot the recent action
                    # sequence to the game's skill bank — these are
                    # actions that led to a level-event so they're
                    # candidate macros for similar future situations.
                    if skills_on and recent_actions[i]:
                        g_idx_sk = i // K
                        seq = tuple(recent_actions[i])
                        skill_bank[g_idx_sk].append(seq)
                    # NEW (2026-05-17): snapshot env state AFTER the
                    # level-up step → push to archive[g][new_level].
                    # This IS the new level's starting state. Without
                    # this, the new-level archive is empty until the
                    # new level is itself beaten — chicken/egg that
                    # blocked tu93 from progressing past L2 despite
                    # 1374 L2 wins in LONG2. Now every L2 win seeds
                    # an L3-start state.
                    if args.archive_cap > 0:
                        g_idx_lu = i // K
                        try:
                            entry_obs = obs_new  # the freshly observed level state
                            state_name_lu = (
                                entry_obs.state.name
                                if hasattr(entry_obs.state, "name")
                                else str(entry_obs.state)
                            )
                            if state_name_lu not in ("GAME_OVER", "WIN", "NOT_PLAYED"):
                                lu_entry = {
                                    "env": copy.deepcopy(envs[i]),
                                    "obs": entry_obs,
                                    "novelty": 2500.0 + 500.0 * float(new_levels),
                                    "levels": int(new_levels),
                                    "ep_step": int(ep_step[i] + 1),
                                    "avail_actions": (
                                        list(entry_obs.available_actions)
                                        if hasattr(entry_obs, "available_actions") and entry_obs.available_actions
                                        else list(avail_actions[i])
                                    ),
                                }
                                gar_lu = archive[g_idx_lu].setdefault(int(new_levels), [])
                                if len(gar_lu) < args.archive_cap:
                                    gar_lu.append(lu_entry)
                                    archive_adds += 1
                                else:
                                    # Highest priority: replace worst
                                    worst_idx = min(
                                        range(len(gar_lu)),
                                        key=lambda j: gar_lu[j]["novelty"],
                                    )
                                    if lu_entry["novelty"] > gar_lu[worst_idx]["novelty"]:
                                        gar_lu[worst_idx] = lu_entry
                                        archive_adds += 1
                        except Exception:
                            pass  # archive seeding is best-effort
                else:
                    # (7) No level event this step → stagnation grows.
                    stagnation_steps[i] += 1
                last_levels[i] = new_levels
                steps_at_level[i // K][new_levels] += 1
                level_step[i] = 0 if d > 0 else level_step[i] + 1
                if d > 0:
                    stale_steps[i] = 0
                if (reset_slot is not None
                        and int(actions[i]) == reset_slot):
                    level_step[i] = 0
                # Rule-goal: a level event means obs_list[i] (pre-step)
                # was a WIN state for the level anchored at
                # ep_start_objset. Record the (start, win) pair and
                # re-anchor at the new level's first frame.
                if (rule_goal_on and d > 0
                        and obs_list[i].frame and obs_new.frame):
                    g_idx_r = i // K
                    if ep_start_objset[i] is not None:
                        pre_frm = np.asarray(obs_list[i].frame[0])
                        rule_pairs[g_idx_r][new_levels - d].append(
                            (ep_start_objset[i], lane_objset(i, pre_frm))
                        )
                        rules_dirty[g_idx_r] = True
                        rule_pairs_count += 1
                    reset_rule_anchor(i, np.asarray(obs_new.frame[0]))
                if not obs_new.frame:
                    # Defensive fallback for empty-frame Observations
                    # (rare; primarily seen when restoring archive
                    # entries from terminal states — now prevented at
                    # archive-add time). Skip frame handling for this
                    # micro-step but still advance ep_step.
                    new_pooled.append([0.0] * 64)
                    homeo_list.append(
                        homeo_for(
                            np.zeros((64, 64), dtype=np.float32),
                            new_levels,
                            win_levels_per[i],
                        )
                    )
                    ep_step[i] += 1
                    continue
                frame = np.asarray(obs_new.frame[0], dtype=np.float32)
                new_pooled.append(preprocess_pooled(frame))
                homeo_list.append(homeo_for(frame, new_levels, win_levels_per[i]))
                ep_step[i] += 1

            # Always update per-game-shared explore_counts for the
            # Go-Explore sync selector, AND accumulate macro_novelty
            # per fork. Cheap; runs regardless of novelty_bonus_scale.
            do_explore_sync = bool(args.explore_sync_novelty)
            level_event = args.goal_bonus > 0
            novel_event = args.novelty_bonus_scale > 0.0
            ext_vec = [0.0] * n_lanes
            for i in range(n_lanes):
                g_idx = i // K
                # Some envs transiently return a frameless obs (e.g.
                # right after an internal error); skip novelty
                # accounting for those steps.
                if not new_obs_list[i].frame:
                    continue
                h = frame_hash(np.asarray(new_obs_list[i].frame[0]))
                # Per-game shared count for the Go-Explore selector.
                gcounts = explore_counts[g_idx]
                gc = gcounts.get(h, 0) + 1
                if gc == 1 and len(gcounts) < args.novelty_cap:
                    gcounts[h] = 1
                elif h in gcounts:
                    gcounts[h] = gc
                # (stagnation-reset) novel frame for this game clears
                # the lane's staleness clock.
                stale_steps[i] = 0 if gc == 1 else stale_steps[i] + 1
                macro_novelty[i] += 1.0 / (gc ** 0.5)
                # Build the per-step extrinsic-reward signal. Goal-bonus
                # fires on level events (binary 0/1). novelty bonus
                # fires per-step using the per-lane (legacy) count
                # so existing-bonus behavior is preserved bit-for-bit
                # when --explore-sync-novelty is off.
                r = 0.0
                if level_event and level_deltas[i] > 0:
                    # Reward proportional to the NEW level achieved,
                    # not just binary "any level event." This biases
                    # the planner and classifier toward pushing higher
                    # levels in already-unlocked games, not just
                    # repeatedly winning level 1.
                    new_total = last_levels[i]  # this is the post-update value
                    if args.level_reward_scale > 0:
                        r += float(level_deltas[i]) * (
                            1.0 + args.level_reward_scale * float(new_total - 1)
                        )
                    else:
                        r += float(level_deltas[i])
                    # (1) One-shot max_lvl-bump bonus.
                    if new_max_lvl_bump[i] > 0 and args.max_lvl_bump_bonus > 0:
                        r += args.max_lvl_bump_bonus * float(new_max_lvl_bump[i])
                        new_max_lvl_bump[i] = 0
                if novel_event:
                    lane_counts = novelty_counts[i]
                    lc = lane_counts.get(h, 0) + 1
                    if lc == 1 and len(lane_counts) < args.novelty_cap:
                        lane_counts[h] = 1
                    elif h in lane_counts:
                        lane_counts[h] = lc
                    r += args.novelty_bonus_scale / (lc ** 0.5)
                ext_vec[i] = r
            if level_event or novel_event:
                agent.set_extrinsic_reward(ext_vec)
            # (4) Per-lane trace write.
            if trace_fp is not None:
                li = args.trace_lane
                trace_fp.write(
                    f"{macro_step},{micro_step},{ep_step[li]},"
                    f"{int(actions[li])},{last_levels[li]},"
                    f"{ext_vec[li]:.3f},{frontier_random_left[li]}\n"
                )

            # Rule-goal potential shaping (Term 4): at the frontier
            # level, reward movement toward the rule-predicted win
            # configuration. Potential-based (phi difference), so it
            # cannot create reward loops.
            rg_vec = None
            if rule_goal_on and args.rule_goal_coef > 0.0:
                rg_vec = [0.0] * n_lanes
                for i in range(n_lanes):
                    if not new_obs_list[i].frame:
                        continue
                    g_idx_r = i // K
                    frm = np.asarray(new_obs_list[i].frame[0])
                    cur_os = lane_objset(i, frm)
                    # (v3) Controllability statistics: which color moved?
                    cs_now = _cstats(cur_os)
                    cs_prev = lane_prev_cstats[i]
                    a_taken = int(actions[i])
                    # (nav) STABLE sprite-mask centroid for routing effects.
                    spr_cen = None
                    _spr_now = avatar_set_of(g_idx_r)
                    if _spr_now:
                        _m = np.isin(frm.astype(np.int32), list(_spr_now))
                        if _m.any():
                            _ys, _xs = _m.nonzero()
                            spr_cen = (float(_ys.mean()) / 63.0,
                                       float(_xs.mean()) / 63.0)
                    spr_prev = lane_prev_sprite_cen[i]
                    if (cs_prev is not None and nav_slot is not None
                            and a_taken < NUM_ACTIONS):
                        # (nav-action) Learn per-color, per-action inverse
                        # dynamics: for every single-blob (n==1) color that
                        # exists in both frames, EMA the centroid
                        # displacement this base action produced. The
                        # avatar is later picked as the color whose
                        # per-action directions are most distinct
                        # (avatar_color_of) — so we DON'T need to know the
                        # avatar first (no chicken-and-egg). n==1 excludes
                        # the maze, floor, and segmented UI.
                        ga_id = a_taken + 1
                        for c, st in cs_now.items():
                            pv = cs_prev.get(c)
                            if pv is None or st["n"] != 1 or pv["n"] != 1:
                                continue
                            dy = st["cy"] - pv["cy"]
                            dx = st["cx"] - pv["cx"]
                            # Reject TELEPORTS: a single grid move is small
                            # (~0.02-0.05 normalized); level transitions,
                            # episode resets and archive RESTORES jump the
                            # avatar across the frame (>0.15). Those huge
                            # deltas poison the EMA (alpha=0.2 can't shrug a
                            # 0.5 jump), collapsing the per-action
                            # directions to one diagonal and destroying the
                            # cardinal signal avatar detection relies on.
                            # The live run restores constantly, so this cap
                            # is essential (a low-restore standalone barely
                            # shows the corruption).
                            d = abs(dy) + abs(dx)
                            if 0.003 < d < 0.15:
                                cur = color_act_eff[g_idx_r][c].get(ga_id)
                                if cur is None:
                                    color_act_eff[g_idx_r][c][ga_id] = [dy, dx]
                                else:
                                    cur[0] = 0.8 * cur[0] + 0.2 * dy
                                    cur[1] = 0.8 * cur[1] + 0.2 * dx
                                color_act_eff_n[g_idx_r][c][ga_id] += 1
                        # sprite-centroid effect (stable; primary for nav)
                        if spr_cen is not None and spr_prev is not None:
                            sdy = spr_cen[0] - spr_prev[0]
                            sdx = spr_cen[1] - spr_prev[1]
                            sd = abs(sdy) + abs(sdx)
                            if 0.003 < sd < 0.15:
                                scur = sprite_act_eff[g_idx_r].get(ga_id)
                                if scur is None:
                                    sprite_act_eff[g_idx_r][ga_id] = [sdy, sdx]
                                else:
                                    scur[0] = 0.8 * scur[0] + 0.2 * sdy
                                    scur[1] = 0.8 * scur[1] + 0.2 * sdx
                                sprite_act_eff_n[g_idx_r][ga_id] += 1
                    lane_prev_sprite_cen[i] = spr_cen
                    lane_prev_cstats[i] = cs_now
                    if lane_pred[i] is None:
                        continue
                    frontier_r = max(
                        max_levels_seen[g_idx_r * K:(g_idx_r + 1) * K]
                    )
                    if last_levels[i] < frontier_r:
                        # Below the frontier the real wins teach
                        # directly; shaping is for the unsolved level.
                        lane_phi[i] = None
                        continue
                    lp = lane_pred[i]
                    # Combined potential: v1 per-color prediction +
                    # v2 relational-arrangement violation, averaged
                    # over whichever components exist.
                    # (v3) Waypoint potential: with a detected avatar
                    # and must-disappear target colors, phi becomes the
                    # sequential-leg distance (nearest remaining target,
                    # then the exit relation) — a monotone navigation
                    # gradient the color-mean forms cannot give.
                    phi = None
                    av_c = avatar_color_of(g_idx_r)
                    gr_r = game_rules[g_idx_r]
                    if av_c is not None and gr_r:
                        tgt = {
                            c for c, r in (gr_r["color"] or {}).items()
                            if not r["survives"] and c != av_c
                        }
                        # v3.1: wall-aware path distance is the correct
                        # maze potential; straight-line is the fallback
                        # when no target is reachable by BFS (or none
                        # remain — exit leg via the relational form).
                        if tgt:
                            phi = _bfs_dist(frm, av_c, tgt)
                        if phi is None:
                            wd = _waypoint_dist(
                                cur_os, av_c, tgt, gr_r["rel"] or None
                            )
                            if wd is not None:
                                phi = wd
                    if phi is None:
                        parts = []
                        if lp["pred"]:
                            parts.append(_config_dist(cur_os, lp["pred"]))
                        if lp["rel"]:
                            parts.append(_rel_dist(cur_os, lp["rel"]))
                        if not parts:
                            continue
                        phi = sum(parts) / len(parts)
                    if lane_best_phi[i] is None or phi < lane_best_phi[i] - 0.01:
                        lane_best_phi[i] = phi
                        stale_steps[i] = 0
                    if lane_phi[i] is not None:
                        r = args.rule_goal_coef * (lane_phi[i] - phi)
                        rg_vec[i] = r
                        rule_shaping_total += abs(r)
                    lane_phi[i] = phi
                if not progress_on:
                    agent.set_intrinsic_progress(rg_vec)

            # Intrinsic progress signals (Term 1 + Term 2).
            if progress_on:
                prog_vec = [0.0] * n_lanes
                for i in range(n_lanes):
                    if not new_obs_list[i].frame:
                        continue
                    if ep_start_cells[i] is None:
                        # Lazy initialize from first observed frame.
                        frm = np.asarray(new_obs_list[i].frame[0], dtype=np.float32)
                        ep_start_cells[i] = coarse_grid_from_frame(frm, pgs)
                        if ep_start_cells[i] is not None:
                            ep_unique_cells[i].add(ep_start_cells[i].tobytes())
                        continue
                    frm = np.asarray(new_obs_list[i].frame[0], dtype=np.float32)
                    g = coarse_grid_from_frame(frm, pgs)
                    if g is None:
                        continue
                    # Term 2: per-episode entropy growth — reward
                    # whenever the current coarse-hash is new in this
                    # episode.
                    if args.progress_diversity_coef > 0.0:
                        gh = g.tobytes()
                        if gh not in ep_unique_cells[i]:
                            ep_unique_cells[i].add(gh)
                            prog_vec[i] += args.progress_diversity_coef
                    # Term 1: persistent configurational delta.
                    # A cell counts if it has been different from
                    # ep_start across the entire recent window.
                    if args.progress_change_coef > 0.0:
                        recent_cells[i].append(g)
                        if len(recent_cells[i]) == pwin and ep_start_cells[i] is not None:
                            persistent = np.ones_like(ep_start_cells[i], dtype=bool)
                            for rc in recent_cells[i]:
                                persistent &= (rc != ep_start_cells[i])
                            cnt = int(persistent.sum())
                            prog_vec[i] += args.progress_change_coef * cnt
                    # Term 3: empowerment (refreshed each planner call).
                    if args.progress_empowerment_coef > 0.0:
                        prog_vec[i] += args.progress_empowerment_coef * last_empowerment[i]
                    # Term 4: rule-goal potential shaping.
                    if rg_vec is not None:
                        prog_vec[i] += rg_vec[i]
                    progress_total += prog_vec[i]
                agent.set_intrinsic_progress(prog_vec)

            # (P1) Novelty-cell archive admission: whenever a lane at
            # its game's FRONTIER level (deepest reached — i.e. the
            # unsolved one) shows a coarse-cell configuration never
            # seen at that game-level, snapshot the env into the
            # per-level archive. Win trails and level-event snapshots
            # only seed levels that have been completed at least once;
            # this is the within-level stepping-stone path that lets
            # restore-from-archive decompose a 70+-action level into
            # planner-sized hops. Scores rank below level-event
            # snapshots (500+) and win trails (1500+), so real wins
            # displace stepping stones once the level is solved.
            if cell_archive_on:
                abs_micro = macro_step * M + micro_step
                for i in range(n_lanes):
                    obs_i = new_obs_list[i]
                    if not obs_i.frame:
                        continue
                    if (abs_micro - cell_last_add[i]
                            < args.cell_archive_min_interval):
                        continue
                    g_idx = i // K
                    lvl = last_levels[i]
                    frontier = max(max_levels_seen[g_idx * K:(g_idx + 1) * K])
                    if lvl < frontier:
                        continue
                    frm = np.asarray(obs_i.frame[0], dtype=np.float32)
                    cg = coarse_grid_from_frame(frm, pgs)
                    if cg is None:
                        continue
                    seen = cell_seen[g_idx].setdefault(lvl, set())
                    ch = cg.tobytes()
                    if ch in seen or len(seen) >= args.cell_seen_cap:
                        continue
                    seen.add(ch)
                    score = 100.0 * float(lvl) + float(ep_step[i] + 1)
                    ent = {
                        "env": copy.deepcopy(envs[i]),
                        "obs": obs_i,
                        "novelty": score,
                        "levels": int(lvl),
                        "ep_step": int(ep_step[i] + 1),
                        "avail_actions": list(obs_i.available_actions or [1]),
                        "v": float(agent.values()[i]),
                    }
                    gar = archive[g_idx].setdefault(lvl, [])
                    if len(gar) < args.archive_cap:
                        gar.append(ent)
                        cell_archive_adds += 1
                        cell_last_add[i] = abs_micro
                    else:
                        worst_idx = min(
                            range(len(gar)),
                            key=lambda j: gar[j]["novelty"],
                        )
                        if score > gar[worst_idx]["novelty"]:
                            gar[worst_idx] = ent
                            cell_archive_adds += 1
                            cell_last_add[i] = abs_micro

            if frame_buf is not None:
                for i, o in enumerate(new_obs_list):
                    if not o.frame:
                        # Frameless obs (transient env hiccup): keep the
                        # previous frame in the slot for this lane.
                        continue
                    cur = np.asarray(o.frame[0], dtype=np.float32) / 15.0
                    frame_buf[i, 0] = cur
                    if args.frame_diff:
                        # Channel 1 = signed delta. Centered at 0 so
                        # encoder sees +/- changes symmetrically.
                        frame_buf[i, 1] = cur - prev_frame[i]
                        prev_frame[i] = cur
            agent.observe(new_pooled, list(actions), homeostatic=homeo_list)
            obs_list = new_obs_list

        # END OF MACRO WINDOW: synchronize forks per game.
        # Primary criterion: highest macro_return (extrinsic events).
        # When no fork hit an event AND --explore-sync-novelty is on,
        # fall back to highest macro_novelty (Go-Explore-style frontier
        # expansion). All forks then BRANCH from the same "interesting"
        # state next macro.
        for g_idx in range(n_games):
            base = g_idx * K
            # 1. Look for an event-bearing fork (return-based sync).
            best_k = 0
            best_r = macro_return[base]
            for k in range(1, K):
                if macro_return[base + k] > best_r:
                    best_r = macro_return[base + k]
                    best_k = k
            sync_reason = "return" if best_r > 0 else None
            # 2. Fallback: novelty-based sync when no event.
            if sync_reason is None and bool(args.explore_sync_novelty):
                best_n_k = 0
                best_n = macro_novelty[base]
                for k in range(1, K):
                    if macro_novelty[base + k] > best_n:
                        best_n = macro_novelty[base + k]
                        best_n_k = k
                # Sync if at least one fork has positive novelty AND
                # there's spread across forks (avoid syncing when all
                # forks are equivalently novel — would just flicker).
                if best_n > 0:
                    spread = best_n - min(
                        macro_novelty[base + k] for k in range(K)
                    )
                    if spread > 1e-6:
                        best_k = best_n_k
                        sync_reason = "novelty"
            if sync_reason is None:
                continue
            best_state = copy.deepcopy(envs[base + best_k]._game)
            for k in range(K):
                if k == best_k:
                    continue
                envs[base + k]._game = copy.deepcopy(best_state)
                obs_list[base + k] = obs_list[base + best_k]
                last_levels[base + k] = last_levels[base + best_k]
                avail_actions[base + k] = list(avail_actions[base + best_k])
                ep_step[base + k] = ep_step[base + best_k]
                agent.mark_boundary(base + k)
            syncs += 1
            # Archive-add: only save non-terminal env states. Terminal
            # states (GAME_OVER / WIN) can't be stepped from — restoring
            # to one yields empty-frame Observations. Also skip if
            # ep_step is too close to max (would immediately retrigger
            # forced reset, wasting a slot).
            if args.archive_cap > 0 and sync_reason is not None:
                best_obs = obs_list[base + best_k]
                best_state_name = (
                    best_obs.state.name
                    if hasattr(best_obs.state, "name")
                    else str(best_obs.state)
                )
                is_terminal = best_state_name in ("GAME_OVER", "WIN", "NOT_PLAYED")
                near_terminal = (
                    ep_step[base + best_k] >= args.max_episode_steps - 5
                )
                if not is_terminal and not near_terminal:
                    score = macro_novelty[base + best_k] + 1000.0 * macro_return[base + best_k]
                    entry_lvl = int(last_levels[base + best_k])
                    entry = {
                        "env": copy.deepcopy(envs[base + best_k]),
                        "obs": obs_list[base + best_k],
                        "novelty": float(score),
                        "levels": entry_lvl,
                        "ep_step": int(ep_step[base + best_k]),
                        "avail_actions": list(avail_actions[base + best_k]),
                    }
                    gar = archive[g_idx].setdefault(entry_lvl, [])
                    if len(gar) < args.archive_cap:
                        gar.append(entry)
                        archive_adds += 1
                    else:
                        worst_idx = min(range(len(gar)), key=lambda j: gar[j]["novelty"])
                        if score > gar[worst_idx]["novelty"]:
                            gar[worst_idx] = entry
                            archive_adds += 1

        if args.log_every_macro and macro_step > 0 and macro_step % args.log_every_macro == 0:
            now = time.time()
            real_t = now - last_log_t
            sps_macro = args.log_every_macro / max(1e-3, real_t)
            env_sps = sps_macro * M * n_lanes
            last_log_t = now
            total_evt = sum(levels_events)
            eps_sum = sum(ep_count)
            # Per-game event tally (count by fork-0 of each game for brevity)
            per_game_evt = []
            for g_idx, e in enumerate(env_infos):
                # Sum across K forks for total per-game events
                total = sum(levels_events[g_idx * K + k] for k in range(K))
                if total > 0:
                    per_game_evt.append(f"{e.game_id[:4]}:{total}")
            evt_str = ",".join(per_game_evt) or "(none)"
            d0 = agent.diagnostics()[0]
            archive_info = ""
            if args.archive_cap > 0:
                total_archive = sum(
                    len(lst) for ag in archive for lst in ag.values()
                )
                trail_tag = f",t{trail_archive_adds}" if win_trail_on else ""
                if args.frontier_archive_on:
                    trail_tag += f",f{frontier_archive_adds}"
                if cell_archive_on:
                    trail_tag += f",c{cell_archive_adds}"
                if rule_goal_on:
                    trail_tag += (
                        f" rg={rule_pairs_count}p/{rule_shaping_total:.0f}"
                    )
                if args.stagnation_reset > 0:
                    trail_tag += f" sr={stagnation_resets}"
                if nav_slot is not None:
                    trail_tag += f" nav={nav_uses}"
                    if args.fog_nav:
                        trail_tag += f" fog={fog_uses}"
                    if nav_gate:
                        g0 = game_rules[0] if 0 < len(game_rules) else None
                        av0 = avatar_color_of(0)
                        ar0 = action_effects_ready(0)
                        top_g = nav_gate.most_common(3)
                        trail_tag += (" navg[" + ",".join(
                            f"{k}:{v}" for k, v in top_g)
                            + f"|av0={av0} ready0={ar0}]")
                    # Per-level archive occupancy of game 0 — at a
                    # glance: are stepping stones accumulating at the
                    # frontier level?
                    lvl_occ = ",".join(
                        f"L{lvl}:{len(lst)}"
                        for lvl, lst in sorted(archive[0].items())
                    )
                    if lvl_occ:
                        trail_tag += f" arc0=[{lvl_occ}]"
                # (5) Episode max-level histogram: counts of completed
                # episodes per max-level. Helps see if any episodes are
                # reaching deeper levels even when total events flat.
                ep_hist_nonzero = [
                    (lvl, n)
                    for lvl, n in enumerate(completed_ep_max_lvl_hist)
                    if n > 0
                ]
                if ep_hist_nonzero:
                    trail_tag += " maxL=" + ",".join(
                        f"{lvl}:{n}" for lvl, n in ep_hist_nonzero
                    )
                prog_tag = f" prog={progress_total:.1f}" if progress_on else ""
                # (L3 metrics) dwell-time-per-level for game 0.
                tl = ",".join(
                    f"{lvl}:{n}"
                    for lvl, n in sorted(steps_at_level[0].items())
                )
                if tl:
                    prog_tag += f" tL=[{tl}]"
                archive_info = (
                    f" arc={total_archive}/{args.archive_cap * n_games}"
                    f"({archive_uses}u,{archive_adds}a{trail_tag}){prog_tag}"
                )
            conf_tag = ""
            if args.confidence_mode:
                try:
                    confs = agent.confidence()
                    conf_min = min(confs) if confs else 0.0
                    conf_max = max(confs) if confs else 0.0
                    conf_mean = sum(confs) / len(confs) if confs else 0.0
                    conf_tag = f" C={conf_mean:.2f}[{conf_min:.2f}-{conf_max:.2f}]"
                except Exception:
                    pass
            print(
                f"macro={macro_step:>5} micro_total={macro_step * M:>6} "
                f"eps={eps_sum:>4} evt={total_evt:>3} "
                f"({evt_str}) syncs={syncs:>4}{archive_info} | "
                f"wm={float(d0['loss_world_model']):.2f} "
                f"pi={float(d0['loss_policy']):.1f} "
                f"ent={float(d0['policy_entropy']):.2f}{conf_tag} | "
                f"{sps_macro:5.1f} macro/s × {M} × {n_lanes} = {env_sps:6.0f} env/s"
            )

    elapsed = time.time() - t0
    print()
    print(f"--- Per-game summary ({n_games} games, K={K} forks each) ---")
    print(f"{'game':<6} {'eps':>5} {'evt':>4} {'max_lvl':>7} {'win':>4}")
    grand_evt = 0
    games_with_evt = 0
    for g_idx, e in enumerate(env_infos):
        ev = sum(levels_events[g_idx * K + k] for k in range(K))
        eps = sum(ep_count[g_idx * K + k] for k in range(K))
        max_lvl = max(max_levels_seen[g_idx * K + k] for k in range(K))
        if ev > 0:
            games_with_evt += 1
        grand_evt += ev
        # Aggregate per-level counts across forks
        by_lvl = collections.defaultdict(int)
        for k in range(K):
            for lvl, cnt in events_by_level[g_idx * K + k].items():
                by_lvl[lvl] += cnt
        per_lvl_str = " ".join(
            f"L{lvl}:{by_lvl[lvl]}" for lvl in sorted(by_lvl) if by_lvl[lvl] > 0
        )
        # (L3 metrics) dwell time per level for this game.
        dwell_str = " ".join(
            f"t{lvl}:{n}" for lvl, n in sorted(steps_at_level[g_idx].items())
        )
        print(f"{e.game_id[:4]:<6} {eps:>5} {ev:>4} {max_lvl:>7} {win_levels_per[g_idx * K]:>4}  {per_lvl_str}  {dwell_str}")
    print()
    print(f"games with ≥1 event: {games_with_evt}/{n_games}")
    print(f"total events (across all forks): {grand_evt}")
    print(f"sync count (forks rejoined to best): {syncs}")
    print(f"throughput: {args.steps * M * n_lanes / max(1e-3, elapsed):.0f} env/s "
          f"({elapsed:.0f}s total)")

    if args.checkpoint_dir:
        agent.save_state(args.checkpoint_dir)
        print(f"saved trained agent to {args.checkpoint_dir}")
    if args.save_archive:
        try:
            # cloudpickle handles unusual module names (e.g.
            # arc_agi_3.tu93-0768757b which has a hyphen and breaks
            # standard pickle's importable-module assumption).
            try:
                import cloudpickle as _pkl
            except ImportError:
                import pickle as _pkl
            with open(args.save_archive, "wb") as _f:
                _pkl.dump(archive, _f, protocol=_pkl.DEFAULT_PROTOCOL if hasattr(_pkl, 'DEFAULT_PROTOCOL') else 4)
            total = sum(len(lst) for ag in archive for lst in ag.values())
            print(f"[archive] saved {total} entries to {args.save_archive}")
        except Exception as _exc:
            print(f"[archive] save failed: {_exc}")
    if trace_fp is not None:
        trace_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
