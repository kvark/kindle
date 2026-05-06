"""ARC-AGI-3 *joint* multi-game trainer (2026-05-05).

Tests the generality hypothesis: train a single kindle agent across all
N games concurrently rather than per-game. ARC-AGI-3's hidden-test
games are not knowable in advance, so a useful agent must navigate
unfamiliar games — single-game training is the wrong protocol.

Design:
  - One BatchAgent with `batch_size=N_games` (one ARC env per lane).
  - Unified action space: kindle outputs 0..6; we map idx → ARC
    GameAction value 1..7. Each game receives the action as-is; if
    the game's `available_actions` doesn't include that value, the
    env-side wrapper either treats it as no-op or errors (we catch
    the error and substitute the first available action).
  - Each lane carries its own homeostatic signal (distance-to-win
    levels + frame entropy). The shared policy must learn behaviours
    that generalize across game dynamics.
  - In-line monitoring: per-lane level-events counter logged each
    K steps. Final summary tabulates per-game progress.

Usage (typical):
  python python/examples/arc_agi3_multi.py --steps 50000 \\
    --adam 1 --adam-eps 1e-4 --grad-clip-norm 0.5 \\
    --encoder cnn_dqn --lr 1e-5 --watchdog-threshold 1e6
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--steps", type=int, default=50000,
                        help="Total training steps (each step advances all lanes once).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--watchdog-threshold", type=float, default=1e6)
    parser.add_argument("--levels-reward-scale", type=float, default=10.0)
    parser.add_argument("--encoder", choices=["mlp", "cnn", "cnn_dqn"], default="cnn_dqn")
    parser.add_argument("--adam", type=int, default=1)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--rnd-alpha", type=float, default=2.0,
                        help="RND curiosity weight per lane. 'full' preset uses 2.0.")
    parser.add_argument("--reward-surprise", type=float, default=5.0)
    parser.add_argument("--reward-homeostatic", type=float, default=0.1)
    parser.add_argument("--use-ppo", type=int, default=0,
                        help="Enable PPO clipped surrogate (1) instead of A2C (0). PPO "
                        "clips the policy update ratio so L_pol can't blow past the "
                        "watchdog — the failure mode that left A2C uniformly random.")
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-beta", type=float, default=0.01,
                        help="Entropy bonus weight; raise to keep exploration alive when "
                        "the policy is reward-sparse.")
    parser.add_argument("--use-sil", type=int, default=0,
                        help="Enable Self-Imitation Learning: replay high-return "
                        "trajectories as positive examples. The remedy for a sparse-"
                        "reward setting where each game produces only 1 success in 50k "
                        "steps — single events become persistent training signal.")
    parser.add_argument("--sil-loss-coef", type=float, default=0.5)
    parser.add_argument("--recon-loss-coef", type=float, default=0.0,
                        help="Weight on the WM-session obs reconstruction loss. >0 "
                        "forces encoder to retain enough info to reconstruct the "
                        "obs token. Standard anti-collapse pressure. Try 1.0.")
    parser.add_argument("--recon-visual-target", type=int, default=0,
                        help="When 1 (and recon_loss_coef > 0 and CNN encoder), "
                        "target the raw 64×64 visual_obs (4096-d, rank ~30 globally) "
                        "instead of the 8×8 pooled obs (rank ~8). Higher rank target "
                        "forces higher-rank z. Adds ~1M decoder params.")
    parser.add_argument("--policy-z-layer-norm", type=int, default=0,
                        help="Apply LayerNorm to z before the policy head (1) or not "
                        "(0). Equalizes per-dim amplitudes so within-game state "
                        "variation can compete with the much-larger between-game "
                        "centroid offset in the policy gradient.")
    parser.add_argument("--policy-z-layer-norm-scale", type=float, default=1.0,
                        help="Constant multiplier after the policy-z LayerNorm. "
                        "Default 1.0 leaves z at unit-std (which empirically killed "
                        "policy commitment). Try 10–40 to recover signal magnitude "
                        "while keeping per-dim equalization.")
    parser.add_argument("--num-options", type=int, default=0,
                        help="DIAYN-style L1 options. >=2 enables option-conditional "
                        "policy heads (per_option_fc2 if --per-option-heads else "
                        "shared trunk + per-option bias). The agent picks options via "
                        "the option session and the policy is conditioned on the "
                        "selected option. Now compatible with PPO (added 2026-05-06). "
                        "0 disables.")
    parser.add_argument("--per-option-heads", type=int, default=0,
                        help="When num_options>=2: 1 = per-option fc2 (each option "
                        "gets its own [hidden,action] matrix), 0 = shared trunk + "
                        "per-option bias.")
    parser.add_argument("--goal-bonus", type=float, default=0.0,
                        help="Extrinsic reward pulse applied on each positive "
                        "level_completed delta. Routed via kindle's "
                        "set_extrinsic_reward + extrinsic_reward_alpha — adds a "
                        "one-shot reward bump on top of the sustained homeo "
                        "level-distance signal. The kindle reward circuit can't "
                        "currently express 'I solved a puzzle' beyond the homeo "
                        "term; this pulse is the simplest way to amplify the rare "
                        "level-completion event.")
    parser.add_argument("--game-prefixes", default=None,
                        help="Comma-separated game-id prefixes to include (e.g. "
                        "'cd82,sp80,r11l'). Default: all 25 games from the corpus.")
    parser.add_argument("--val-prefixes", default=None,
                        help="Comma-separated game-id prefixes held out for "
                        "post-training validation. After --steps training on the "
                        "non-val games, the agent is checkpointed, a fresh val "
                        "agent is instantiated with the held-out games, the "
                        "checkpoint is loaded into it, and the val agent runs "
                        "--val-steps env steps with lr=0 (no training) to measure "
                        "transfer.")
    parser.add_argument("--val-steps", type=int, default=10000,
                        help="Steps to run on val games (lr=0, no training). Default 10k.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Optional directory to save the trained agent state "
                        "to (after training, before any val pass). Useful for "
                        "post-hoc analysis.")
    args = parser.parse_args()

    # --- Discover all games ---
    arcade = Arcade()
    all_envs = arcade.get_environments()
    if args.game_prefixes:
        prefixes = [p.strip() for p in args.game_prefixes.split(",") if p.strip()]
        env_infos = [e for e in all_envs if any(e.game_id.startswith(p) for p in prefixes)]
    else:
        # Take the first env per unique game-id 4-char prefix.
        seen: set[str] = set()
        env_infos = []
        for e in sorted(all_envs, key=lambda x: x.game_id):
            key = e.game_id[:4]
            if key in seen:
                continue
            seen.add(key)
            env_infos.append(e)

    # Train/val split. If --val-prefixes is set, those games are pulled
    # OUT of the training set and saved for a post-training val pass.
    val_set = set()
    if args.val_prefixes:
        val_set = {p.strip() for p in args.val_prefixes.split(",") if p.strip()}
        val_env_infos = [e for e in env_infos if any(e.game_id.startswith(p) for p in val_set)]
        env_infos = [e for e in env_infos if not any(e.game_id.startswith(p) for p in val_set)]
        if not val_env_infos:
            print(f"warning: --val-prefixes matched no games", file=sys.stderr)
        else:
            print(f"train/val split: {len(env_infos)} train games, {len(val_env_infos)} val games")
            print(f"  val games: {[e.game_id[:4] for e in val_env_infos]}")
    else:
        val_env_infos = []

    n_games = len(env_infos)
    if n_games == 0:
        print("no games matched", file=sys.stderr)
        return 1

    # --- Instantiate envs (one per lane) ---
    envs = []
    obs_list = []
    avail_actions: list[list[int]] = []  # per-lane available_actions (mutable)
    win_levels_per: list[int] = []
    for e in env_infos:
        env = arcade.make(e.game_id)
        obs = env.reset()
        envs.append(env)
        obs_list.append(obs)
        avail_actions.append(list(obs.available_actions) or [1])
        win_levels_per.append(int(obs.win_levels))

    print(
        f"joint training: {n_games} games × {args.steps} steps = "
        f"{n_games * args.steps:,} env steps total"
    )
    for i, e in enumerate(env_infos):
        print(
            f"  lane {i:>2} {e.game_id[:4]:<4} "
            f"avail={avail_actions[i]} win={win_levels_per[i]}"
        )

    # --- Unified action mapping ---
    # kindle outputs idx 0..NUM_ACTIONS-1; we map idx → GameAction.value (idx+1).
    # ARC-AGI-3's full simple-action space is 1..7.
    NUM_ACTIONS = 7
    action_by_value = {int(a.value): a for a in GameAction}

    # --- Build the shared agent ---
    agent_kwargs = dict(
        obs_dim=64,
        num_actions=NUM_ACTIONS,
        batch_size=n_games,
        env_ids=list(range(n_games)),
        seed=args.seed,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        policy_loss_watchdog_threshold=args.watchdog_threshold,
        use_adam=bool(args.adam),
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
        recon_loss_coef=args.recon_loss_coef,
        recon_visual_target=bool(args.recon_visual_target),
        policy_z_layer_norm=bool(args.policy_z_layer_norm),
        policy_z_layer_norm_scale=args.policy_z_layer_norm_scale,
        extrinsic_reward_alpha=args.goal_bonus if args.goal_bonus > 0 else 0.0,
    )
    if args.num_options >= 2:
        agent_kwargs["num_options"] = args.num_options
        agent_kwargs["per_option_heads"] = bool(args.per_option_heads)
    if args.encoder == "cnn":
        agent_kwargs.update(
            encoder_kind="cnn",
            encoder_channels=1,
            encoder_height=64,
            encoder_width=64,
        )
    elif args.encoder == "cnn_dqn":
        agent_kwargs.update(
            encoder_kind="cnn_dqn",
            encoder_channels=1,
            encoder_height=64,
            encoder_width=64,
        )
    agent = kindle.BatchAgent(**agent_kwargs)
    print(
        f"agent: {NUM_ACTIONS} actions, latent={args.latent_dim}, "
        f"hidden={args.hidden_dim}, encoder={args.encoder}, lr={args.lr}, "
        f"adam={bool(args.adam)} eps={args.adam_eps} grad_clip={args.grad_clip_norm}"
    )

    # --- Helpers ---
    def preprocess_pooled(frame_arr: np.ndarray) -> list[float]:
        """64×64 → 8×8 mean-pool → 64-dim float in [0, 1]."""
        arr = frame_arr.astype(np.float32) / 15.0
        pooled = arr.reshape(8, 8, 8, 8).mean(axis=(1, 3))
        return pooled.flatten().tolist()

    def preprocess_visual(frame_arr: np.ndarray) -> np.ndarray:
        """64×64 int → flat 4096-dim float in [0, 1]."""
        return (frame_arr.astype(np.float32) / 15.0).flatten()

    def homeo_for(
        frame_arr: np.ndarray,
        new_levels: int,
        win_levels: int,
    ) -> list[dict]:
        remaining = max(0, win_levels - new_levels)
        levels_term = {
            "value": float(remaining) * args.levels_reward_scale,
            "target": 0.0,
            "tolerance": 0.0,
        }
        uniq = float(np.unique(frame_arr).size) / 16.0
        entropy_term = {
            "value": 1.0 - uniq,
            "target": 0.0,
            "tolerance": 0.1,
        }
        return [levels_term, entropy_term]

    def map_action(kindle_idx: int, lane: int, rng) -> tuple[GameAction, dict | None]:
        target_value = kindle_idx + 1  # 0..6 → 1..7
        aa = avail_actions[lane]
        # If the game doesn't have that action, fall back to first available.
        if target_value not in aa:
            target_value = aa[0]
        a = action_by_value[target_value]
        if a.is_complex():
            x = rng.randrange(64)
            y = rng.randrange(64)
            a.set_data({"x": x, "y": y})
            return a, {"x": x, "y": y}
        return a, None

    # --- Per-lane training state ---
    import random
    rng = random.Random(args.seed)

    last_levels = [int(o.levels_completed) for o in obs_list]
    levels_events = [0] * n_games  # per-game total level events
    ep_step = [0] * n_games
    ep_count = [0] * n_games
    ep_levels_at_end: list[list[int]] = [[] for _ in range(n_games)]
    ep_lens: list[list[int]] = [[] for _ in range(n_games)]

    # Initial visual_obs upload (CNN encoders only).
    if args.encoder in ("cnn", "cnn_dqn"):
        frame_mv = agent.visual_obs_memoryview()
        frame_buf = np.frombuffer(frame_mv, dtype=np.float32).reshape(
            n_games, 1, 64, 64
        )
    else:
        frame_buf = None

    t0 = time.time()
    last_log_t = t0

    for step in range(args.steps):
        # Reset any lane that needs it.
        for i in range(n_games):
            obs = obs_list[i]
            state = obs.state
            need_reset = state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or (
                state is GameState.WIN
            ) or ep_step[i] >= args.max_episode_steps
            if need_reset:
                if ep_step[i] > 0:
                    ep_count[i] += 1
                    ep_levels_at_end[i].append(int(obs.levels_completed))
                    ep_lens[i].append(ep_step[i])
                obs_list[i] = envs[i].reset()
                avail_actions[i] = list(obs_list[i].available_actions) or avail_actions[i]
                last_levels[i] = int(obs_list[i].levels_completed)
                ep_step[i] = 0
                agent.mark_boundary(i)

        # Build observation batch.
        pooled_batch = [preprocess_pooled(np.asarray(o.frame[0], dtype=np.float32)) for o in obs_list]
        if frame_buf is not None:
            for i, o in enumerate(obs_list):
                frame_buf[i, 0] = np.asarray(o.frame[0], dtype=np.float32) / 15.0

        # Act.
        actions = agent.act(pooled_batch)

        # Step each env.
        new_obs_list = []
        homeo_list = []
        new_pooled_batch = []
        level_deltas = [0] * n_games
        for i in range(n_games):
            game_action, action_data = map_action(int(actions[i]), i, rng)
            try:
                obs_new = envs[i].step(game_action, data=action_data)
            except Exception as exc:
                # Hard error: substitute reset state, log, continue.
                if step < 10:
                    print(f"  lane {i} step error: {exc}", file=sys.stderr)
                obs_new = envs[i].reset()
                avail_actions[i] = list(obs_new.available_actions) or avail_actions[i]
                agent.mark_boundary(i)
            frame = np.asarray(obs_new.frame[0], dtype=np.float32)
            new_obs_list.append(obs_new)
            if list(obs_new.available_actions):
                avail_actions[i] = list(obs_new.available_actions)
            new_levels = int(obs_new.levels_completed)
            delta = new_levels - last_levels[i]
            level_deltas[i] = delta
            if delta > 0:
                levels_events[i] += delta
            last_levels[i] = new_levels
            new_pooled_batch.append(preprocess_pooled(frame))
            homeo_list.append(homeo_for(frame, new_levels, win_levels_per[i]))
            ep_step[i] += 1

        # Goal-completion extrinsic bonus: per-lane 1.0 on positive level
        # transitions, 0 otherwise. Applied via kindle's extrinsic-reward
        # path (multiplied by extrinsic_reward_alpha at observe-time).
        if args.goal_bonus > 0:
            ext = [1.0 if d > 0 else 0.0 for d in level_deltas]
            agent.set_extrinsic_reward(ext)

        # Observe (single batched call — this is where training happens).
        agent.observe(new_pooled_batch, list(actions), homeostatic=homeo_list)
        obs_list = new_obs_list

        if args.log_every and step > 0 and step % args.log_every == 0:
            now = time.time()
            sps = args.log_every / max(1e-3, now - last_log_t)
            agg_sps = sps * n_games  # aggregate env steps/sec
            last_log_t = now
            total_events = sum(levels_events)
            total_eps = sum(ep_count)
            # Per-lane events: print only the games that have any events
            evt_str = ",".join(
                f"{env_infos[i].game_id[:4]}:{levels_events[i]}"
                for i in range(n_games) if levels_events[i] > 0
            ) or "(none)"
            d0 = agent.diagnostics()[0]
            print(
                f"step={step:>6} eps={total_eps:>4} lvl_events={total_events:>3} "
                f"({evt_str}) | "
                f"wm={float(d0['loss_world_model']):.3f} "
                f"pi={float(d0['loss_policy']):.3f} "
                f"ent={float(d0['policy_entropy']):.2f} | "
                f"{sps:5.0f} step/s × {n_games} = {agg_sps:6.0f} env/s"
            )

    elapsed = time.time() - t0
    sps = args.steps / max(1e-3, elapsed)

    print()
    print("--- Per-game summary ---")
    print(f"{'game':<6} {'eps':>4} {'evt':>4} {'mean_lvl':>9} {'max_lvl':>7} {'win':>4} {'mean_len':>8}")
    for i, e in enumerate(env_infos):
        mean_lvl = (
            sum(ep_levels_at_end[i]) / len(ep_levels_at_end[i])
            if ep_levels_at_end[i] else 0.0
        )
        max_lvl = max(ep_levels_at_end[i]) if ep_levels_at_end[i] else 0
        mean_len = sum(ep_lens[i]) / len(ep_lens[i]) if ep_lens[i] else 0.0
        print(
            f"{e.game_id[:4]:<6} {ep_count[i]:>4} {levels_events[i]:>4} "
            f"{mean_lvl:>9.2f} {max_lvl:>7d} {win_levels_per[i]:>4d} "
            f"{mean_len:>8.1f}"
        )
    total_evt = sum(levels_events)
    games_with_events = sum(1 for x in levels_events if x > 0)
    print()
    print(f"games with ≥1 level event: {games_with_events}/{n_games}")
    print(f"total level events: {total_evt}")
    print(f"throughput: {sps:.1f} step/s ({sps * n_games:.0f} env/s; {elapsed:.0f}s total)")

    # --- Save checkpoint (always when val pass is requested, optional otherwise) ---
    ckpt_dir = args.checkpoint_dir
    if val_env_infos and ckpt_dir is None:
        import tempfile
        ckpt_dir = tempfile.mkdtemp(prefix="kindle_ckpt_")
    if ckpt_dir is not None:
        agent.save_state(ckpt_dir)
        print(f"\nsaved trained agent to {ckpt_dir}")

    # --- Validation pass on held-out games ---
    if val_env_infos:
        n_val = len(val_env_infos)
        print()
        print(f"=== validation pass ({n_val} held-out games, {args.val_steps} steps, lr=0) ===")

        # Build val envs.
        val_envs = []
        val_obs_list = []
        val_avail_actions: list[list[int]] = []
        val_win_levels_per: list[int] = []
        for e in val_env_infos:
            env = arcade.make(e.game_id)
            obs = env.reset()
            val_envs.append(env)
            val_obs_list.append(obs)
            val_avail_actions.append(list(obs.available_actions) or [1])
            val_win_levels_per.append(int(obs.win_levels))

        # Fresh agent with the same architecture but val-game count.
        val_kwargs = dict(agent_kwargs)
        val_kwargs["batch_size"] = n_val
        val_kwargs["env_ids"] = list(range(n_val))
        val_agent = kindle.BatchAgent(**val_kwargs)
        val_agent.load_state(ckpt_dir)
        val_agent.set_all_learning_rates(0.0)
        print(f"loaded trained weights into val agent ({n_val} lanes), lr=0")

        if args.encoder in ("cnn", "cnn_dqn"):
            val_frame_mv = val_agent.visual_obs_memoryview()
            val_frame_buf = np.frombuffer(val_frame_mv, dtype=np.float32).reshape(
                n_val, 1, 64, 64
            )
        else:
            val_frame_buf = None

        val_last_levels = [int(o.levels_completed) for o in val_obs_list]
        val_levels_events = [0] * n_val
        val_ep_step = [0] * n_val
        val_ep_count = [0] * n_val

        def val_map_action(idx, lane):
            target = idx + 1
            aa = val_avail_actions[lane]
            if target not in aa:
                target = aa[0]
            a = action_by_value[target]
            if a.is_complex():
                x = rng.randrange(64)
                y = rng.randrange(64)
                a.set_data({"x": x, "y": y})
                return a, {"x": x, "y": y}
            return a, None

        t0v = time.time()
        for vstep in range(args.val_steps):
            for i in range(n_val):
                obs = val_obs_list[i]
                state = obs.state
                if (state in (GameState.NOT_PLAYED, GameState.GAME_OVER)
                        or state is GameState.WIN
                        or val_ep_step[i] >= args.max_episode_steps):
                    if val_ep_step[i] > 0:
                        val_ep_count[i] += 1
                    val_obs_list[i] = val_envs[i].reset()
                    val_avail_actions[i] = list(val_obs_list[i].available_actions) or val_avail_actions[i]
                    val_last_levels[i] = int(val_obs_list[i].levels_completed)
                    val_ep_step[i] = 0
                    val_agent.mark_boundary(i)

            pooled_batch = [preprocess_pooled(np.asarray(o.frame[0], dtype=np.float32)) for o in val_obs_list]
            if val_frame_buf is not None:
                for i, o in enumerate(val_obs_list):
                    val_frame_buf[i, 0] = np.asarray(o.frame[0], dtype=np.float32) / 15.0

            actions = val_agent.act(pooled_batch)

            new_obs_list = []
            homeo_list = []
            new_pooled_batch = []
            for i in range(n_val):
                ga, ad = val_map_action(int(actions[i]), i)
                try:
                    obs_new = val_envs[i].step(ga, data=ad)
                except Exception:
                    obs_new = val_envs[i].reset()
                    val_avail_actions[i] = list(obs_new.available_actions) or val_avail_actions[i]
                    val_agent.mark_boundary(i)
                new_obs_list.append(obs_new)
                if list(obs_new.available_actions):
                    val_avail_actions[i] = list(obs_new.available_actions)
                new_levels = int(obs_new.levels_completed)
                delta = new_levels - val_last_levels[i]
                if delta > 0:
                    val_levels_events[i] += delta
                val_last_levels[i] = new_levels
                frame = np.asarray(obs_new.frame[0], dtype=np.float32)
                new_pooled_batch.append(preprocess_pooled(frame))
                homeo_list.append(homeo_for(frame, new_levels, val_win_levels_per[i]))
                val_ep_step[i] += 1

            val_agent.observe(new_pooled_batch, list(actions), homeostatic=homeo_list)
            val_obs_list = new_obs_list

        velapsed = time.time() - t0v
        print(f"\n--- Val per-game summary ({args.val_steps} steps, lr=0) ---")
        print(f"{'game':<6} {'eps':>4} {'evt':>4} {'win':>4}")
        for i, e in enumerate(val_env_infos):
            print(f"{e.game_id[:4]:<6} {val_ep_count[i]:>4} {val_levels_events[i]:>4} "
                  f"{val_win_levels_per[i]:>4d}")
        val_total = sum(val_levels_events)
        val_breadth = sum(1 for x in val_levels_events if x > 0)
        print(f"\nval games with ≥1 event: {val_breadth}/{n_val}")
        print(f"val total level events: {val_total}")
        print(f"val throughput: {args.val_steps / max(1e-3, velapsed):.1f} step/s "
              f"({velapsed:.0f}s total)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
