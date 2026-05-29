"""Genetic algorithm over action sequences at L2-archive starts.

Loads a pickled per-game-per-level archive (produced by harness with
--save-archive). For each L2 entry, evolves a population of action
sequences to maximize cumulative extrinsic events. If any sequence
fires a max_lvl=3 advance (= L2 completion), saves it.

This is a direct attack on the L2-completion sequence: instead of
random shooting via planner, use evolution to find a coherent 30-50
action sequence. No model, no value head — just envs as black-box
evaluators.
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

# PyO3 GameAction doesn't implement value-based lookup, breaking pickle
# round-trip. Patch _missing_ before any unpickle attempt.
try:
    from arcengine import GameAction as _GameAction
    if not hasattr(_GameAction, "_pickle_missing_patched"):
        @classmethod
        def _missing_(cls, value):
            for m in cls:
                if m.value == value:
                    return m
            return None
        _GameAction._missing_ = _missing_
        _GameAction._pickle_missing_patched = True
except ImportError:
    pass


def evaluate_sequence(env_snapshot, obs, action_by_value, avail_actions,
                       sequence, max_steps=None):
    """Run a sequence of action indices in the env starting from snapshot.
    Returns (cum_levels_completed, max_lvl_reached, terminated_early).
    """
    try:
        from arcengine import GameAction, GameState
    except ImportError:
        return 0, 0, True
    env = copy.deepcopy(env_snapshot)
    start_level = int(obs.levels_completed)
    max_lvl = start_level
    rewards = 0
    avail = list(avail_actions)
    n = max_steps if max_steps else len(sequence)
    for i in range(min(n, len(sequence))):
        if not avail:
            break
        # Clamp index into avail
        a_idx = int(sequence[i]) % len(avail)
        target = int(avail[a_idx])
        action = action_by_value[target]
        try:
            if action.is_complex():
                obs = env.step(action, data={
                    "x": int(np.random.randint(0, 64)),
                    "y": int(np.random.randint(0, 64)),
                })
            else:
                obs = env.step(action)
        except Exception:
            break
        if obs is None:
            break
        new_level = int(obs.levels_completed)
        if new_level > start_level:
            rewards += (new_level - start_level)
        if new_level > max_lvl:
            max_lvl = new_level
        start_level = new_level
        avail = list(obs.available_actions) or avail
        if obs.state.name in ("GAME_OVER", "WIN"):
            break
    return rewards, max_lvl, False


def ga_search(env_snapshot, obs, action_by_value, avail_actions,
              pop_size=50, seq_len=40, generations=20, mutation_rate=0.1,
              rng=None):
    """Genetic algorithm. Returns (best_sequence, best_score, best_max_lvl).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Each sequence = array of action indices (0..len(avail)-1)
    n_actions = len(avail_actions)
    pop = [rng.integers(0, n_actions, size=seq_len) for _ in range(pop_size)]
    best_score = -1
    best_max_lvl = int(obs.levels_completed)
    best_seq = pop[0]
    for gen in range(generations):
        scores = []
        max_lvls = []
        for seq in pop:
            r, ml, _ = evaluate_sequence(env_snapshot, obs, action_by_value,
                                          avail_actions, seq)
            scores.append(r)
            max_lvls.append(ml)
            if r > best_score or (r == best_score and ml > best_max_lvl):
                best_score = r
                best_max_lvl = ml
                best_seq = seq.copy()
        # Tournament selection + crossover + mutation
        new_pop = [best_seq]  # elitism
        while len(new_pop) < pop_size:
            i, j = rng.integers(0, pop_size, size=2)
            parent_a = pop[i] if scores[i] >= scores[j] else pop[j]
            i, j = rng.integers(0, pop_size, size=2)
            parent_b = pop[i] if scores[i] >= scores[j] else pop[j]
            xover = int(rng.integers(1, seq_len))
            child = np.concatenate([parent_a[:xover], parent_b[xover:]])
            mut_mask = rng.random(seq_len) < mutation_rate
            child[mut_mask] = rng.integers(0, n_actions, size=int(mut_mask.sum()))
            new_pop.append(child)
        pop = new_pop
        if gen % 5 == 0:
            print(f"  gen {gen}: best_score={best_score} best_max_lvl={best_max_lvl}")
    return best_seq, best_score, best_max_lvl


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="/tmp/aff_runs/archive_curr_tu93.pkl")
    p.add_argument("--target-level", type=int, default=2,
                   help="Search from entries at this level (typically 2 = L2).")
    p.add_argument("--out", default="/tmp/aff_runs/ga_winners.pkl")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--max-entries", type=int, default=20,
                   help="Search at most this many L2 archive entries.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    try:
        from arcengine import GameAction
    except ImportError as exc:
        print(f"missing arcengine: {exc}", file=sys.stderr)
        return 1
    action_by_value = {int(act.value): act for act in GameAction}

    print(f"loading archive: {a.archive}")
    try:
        import cloudpickle as _pkl
    except ImportError:
        _pkl = pickle
    with open(a.archive, "rb") as f:
        archive = _pkl.load(f)

    # archive is list of per-game dicts: archive[g][level] = [entries]
    n_games = len(archive)
    print(f"loaded {n_games} games")

    rng = np.random.default_rng(a.seed)
    winners = []  # list of (game_idx, entry_idx, sequence, score, max_lvl)
    breakthroughs = 0  # count of sequences that reached level > target_level

    for g_idx, game in enumerate(archive):
        if a.target_level not in game:
            continue
        entries = game[a.target_level]
        print(f"\n=== game {g_idx}: {len(entries)} L{a.target_level} entries ===")
        for ent_idx, entry in enumerate(entries[:a.max_entries]):
            print(f"--- entry {ent_idx} (ep_step={entry.get('ep_step', '?')}) ---")
            env_snap = entry["env"]
            obs = entry["obs"]
            avail = entry["avail_actions"]
            best_seq, best_score, best_max_lvl = ga_search(
                env_snap, obs, action_by_value, avail,
                pop_size=a.pop, seq_len=a.seq_len, generations=a.gens,
                rng=rng,
            )
            print(f"  → best_score={best_score} max_lvl={best_max_lvl}")
            if best_max_lvl > a.target_level:
                breakthroughs += 1
                print(f"  *** BREAKTHROUGH: L{a.target_level} → L{best_max_lvl} ***")
            winners.append({
                "game_idx": g_idx,
                "entry_idx": ent_idx,
                "sequence": best_seq.tolist(),
                "score": int(best_score),
                "max_lvl": int(best_max_lvl),
                "start_level": int(obs.levels_completed),
            })

    print(f"\n=== TOTAL: {breakthroughs} breakthroughs out of {len(winners)} entries ===")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(winners, f)
    print(f"wrote winners to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
