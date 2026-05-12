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
    args = parser.parse_args()

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

    agent_kwargs = dict(
        obs_dim=64,
        num_actions=NUM_ACTIONS,
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
        planner_rnd_alpha=args.planner_rnd_alpha,
        planner_goal_alpha=args.planner_goal_alpha,
        goal_states_cap=args.goal_states_cap,
        visit_count_dims=args.visit_count_dims,
        visit_count_proj_dim=args.visit_count_proj_dim,
    )
    if args.encoder == "cnn_dqn":
        agent_kwargs.update(
            encoder_kind="cnn_dqn", encoder_channels=1,
            encoder_height=64, encoder_width=64,
        )
    elif args.encoder == "cnn":
        agent_kwargs.update(
            encoder_kind="cnn", encoder_channels=1,
            encoder_height=64, encoder_width=64,
        )
    agent = kindle.BatchAgent(**agent_kwargs)
    if args.load_state:
        agent.load_state(args.load_state)
        print(f"loaded prior state from {args.load_state}")

    # Helpers ---
    def preprocess_pooled(frame_arr):
        arr = frame_arr.astype(np.float32) / 15.0
        return arr.reshape(8, 8, 8, 8).mean(axis=(1, 3)).flatten().tolist()

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

    def update_action_mask():
        """Rebuild per-lane action masks from current avail_actions and push
        them to kindle. Slots whose ARC GameAction.value (idx+1) is in
        avail_actions[lane] stay at 1.0; everything else (including the
        slots beyond NUM_ACTIONS) is forced to 0.0."""
        mask_buf.fill(0.0)
        for lane_i in range(n_lanes):
            for v in avail_actions[lane_i]:
                idx = int(v) - 1
                if 0 <= idx < MAX_ACTION_DIM:
                    mask_buf[lane_i, idx] = 1.0
        agent.set_action_masks(mask_buf.reshape(-1))

    def map_action(idx, lane):
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
            n_lanes, 1, 64, 64)
    else:
        frame_buf = None

    # Per-lane state.
    last_levels = [int(o.levels_completed) for o in obs_list]
    levels_events = [0] * n_lanes
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

    # Per-game persistent frontier archive (real Go-Explore). Each
    # entry: {'game': deepcopy(env._game), 'obs': obs_after_save,
    # 'novelty': macro_novelty_at_save, 'levels': levels_completed}.
    # Bounded by --archive-cap; evicts lowest-novelty when full.
    archive = [[] for _ in range(n_games)]
    archive_uses = 0  # diagnostic: how many times we restored from archive
    archive_adds = 0  # diagnostic: how many entries added

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
                need_reset = (
                    state in (GameState.NOT_PLAYED, GameState.GAME_OVER)
                    or state is GameState.WIN
                    or ep_step[i] >= args.max_episode_steps
                )
                if need_reset:
                    if ep_step[i] > 0:
                        ep_count[i] += 1
                    # Archive-reset: with prob, restore from a random
                    # archived frontier state instead of a fresh env
                    # reset. Real Ecoffet Go-Explore: "return then
                    # explore" — start episodes from interesting
                    # discovered waypoints, not always from level-0.
                    g_idx = i // K
                    used_archive = False
                    if (args.archive_cap > 0
                            and archive[g_idx]
                            and rng.random() < args.archive_reset_prob):
                        # Replace the lane's env with a fresh deepcopy
                        # of the archived wrapper. Restores all
                        # wrapper-internal state along with _game.
                        entry = archive[g_idx][rng.randrange(len(archive[g_idx]))]
                        envs[i] = copy.deepcopy(entry["env"])
                        obs_list[i] = entry["obs"]
                        last_levels[i] = entry["levels"]
                        avail_actions[i] = list(entry["avail_actions"])
                        ep_step[i] = entry["ep_step"]
                        archive_uses += 1
                        used_archive = True
                    if not used_archive:
                        obs_list[i] = envs[i].reset()
                        avail_actions[i] = list(obs_list[i].available_actions) or avail_actions[i]
                        last_levels[i] = int(obs_list[i].levels_completed)
                        ep_step[i] = 0
                    agent.mark_boundary(i)

            pooled = [preprocess_pooled(np.asarray(o.frame[0], dtype=np.float32))
                      for o in obs_list]
            if frame_buf is not None:
                for i, o in enumerate(obs_list):
                    frame_buf[i, 0] = np.asarray(o.frame[0], dtype=np.float32) / 15.0

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
                agent.plan_and_queue(NUM_ACTIONS)
            actions = agent.act(pooled)

            new_obs_list = []
            homeo_list = []
            new_pooled = []
            level_deltas = [0] * n_lanes
            for i in range(n_lanes):
                ga, ad = map_action(int(actions[i]), i)
                try:
                    obs_new = envs[i].step(ga, data=ad)
                except Exception:
                    obs_new = envs[i].reset()
                    avail_actions[i] = list(obs_new.available_actions) or avail_actions[i]
                    agent.mark_boundary(i)
                new_obs_list.append(obs_new)
                if list(obs_new.available_actions):
                    avail_actions[i] = list(obs_new.available_actions)
                new_levels = int(obs_new.levels_completed)
                d = new_levels - last_levels[i]
                level_deltas[i] = d
                if d > 0:
                    levels_events[i] += d
                    macro_return[i] += float(d)  # cheap proxy for "did good things happen"
                last_levels[i] = new_levels
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
                h = frame_hash(np.asarray(new_obs_list[i].frame[0]))
                # Per-game shared count for the Go-Explore selector.
                gcounts = explore_counts[g_idx]
                gc = gcounts.get(h, 0) + 1
                if gc == 1 and len(gcounts) < args.novelty_cap:
                    gcounts[h] = 1
                elif h in gcounts:
                    gcounts[h] = gc
                macro_novelty[i] += 1.0 / (gc ** 0.5)
                # Build the per-step extrinsic-reward signal. Goal-bonus
                # fires on level events (binary 0/1). novelty bonus
                # fires per-step using the per-lane (legacy) count
                # so existing-bonus behavior is preserved bit-for-bit
                # when --explore-sync-novelty is off.
                r = 0.0
                if level_event and level_deltas[i] > 0:
                    r += 1.0
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
                    entry = {
                        "env": copy.deepcopy(envs[base + best_k]),
                        "obs": obs_list[base + best_k],
                        "novelty": float(score),
                        "levels": int(last_levels[base + best_k]),
                        "ep_step": int(ep_step[base + best_k]),
                        "avail_actions": list(avail_actions[base + best_k]),
                    }
                    gar = archive[g_idx]
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
                total_archive = sum(len(a) for a in archive)
                archive_info = (
                    f" arc={total_archive}/{args.archive_cap * n_games}"
                    f"({archive_uses}u,{archive_adds}a)"
                )
            print(
                f"macro={macro_step:>5} micro_total={macro_step * M:>6} "
                f"eps={eps_sum:>4} evt={total_evt:>3} "
                f"({evt_str}) syncs={syncs:>4}{archive_info} | "
                f"wm={float(d0['loss_world_model']):.2f} "
                f"pi={float(d0['loss_policy']):.1f} "
                f"ent={float(d0['policy_entropy']):.2f} | "
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
        # We don't track max_lvl per game in this harness; could.
        if ev > 0:
            games_with_evt += 1
        grand_evt += ev
        print(f"{e.game_id[:4]:<6} {eps:>5} {ev:>4} {'?':>7} {win_levels_per[g_idx * K]:>4}")
    print()
    print(f"games with ≥1 event: {games_with_evt}/{n_games}")
    print(f"total events (across all forks): {grand_evt}")
    print(f"sync count (forks rejoined to best): {syncs}")
    print(f"throughput: {args.steps * M * n_lanes / max(1e-3, elapsed):.0f} env/s "
          f"({elapsed:.0f}s total)")

    if args.checkpoint_dir:
        agent.save_state(args.checkpoint_dir)
        print(f"saved trained agent to {args.checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
