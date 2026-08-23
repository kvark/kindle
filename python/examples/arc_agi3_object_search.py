"""Object-centric state search for cloneable local ARC-AGI-3 games.

This is a planning harness, not a claim that policy-gradient learning solves
ARC. It turns visible connected components into candidate coordinates, probes
one-step outcomes to remove duplicate/no-op actions, and breadth-first searches
the remaining visual state graph for an increase in ``levels_completed``.

When the candidate actions are visually involutive and pairwise commutative
(as in toggle puzzles), only increasing-index action combinations are searched.
That removes redundant permutations without embedding game-specific rules.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

StateT = TypeVar("StateT")
ObsT = TypeVar("ObsT")
ActionT = TypeVar("ActionT")


class TransitionRejected(RuntimeError):
    """A local simulator rejected one cloned state/action branch."""


@dataclass(frozen=True)
class SearchResult(Generic[StateT, ObsT, ActionT]):
    state: StateT | None
    observation: ObsT | None
    path: tuple[ActionT, ...]
    expanded: int
    generated: int
    seen: int

    @property
    def found(self) -> bool:
        return self.state is not None


@dataclass(frozen=True)
class ArcAction:
    value: int
    x: int | None = None
    y: int | None = None

    def label(self) -> str:
        if self.x is None:
            return f"A{self.value}"
        return f"A{self.value}({self.x},{self.y})"


def visual_state_key(frame: np.ndarray, crop_border: int = 1) -> bytes:
    """Hashable visual state, optionally excluding a volatile HUD border."""
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D frame, got shape {arr.shape}")
    crop = max(0, int(crop_border))
    if crop and 2 * crop < min(arr.shape):
        arr = arr[crop:-crop, crop:-crop]
    return arr.tobytes()


def breadth_first_reward_search(
    root_state: StateT,
    root_observation: ObsT,
    actions_for: Callable[[StateT, ObsT], Sequence[ActionT]],
    transition: Callable[[StateT, ActionT], tuple[StateT, ObsT]],
    state_key: Callable[[ObsT], Hashable],
    score: Callable[[ObsT], int],
    terminal: Callable[[ObsT], bool],
    *,
    max_states: int,
    max_transitions: int = 0,
    max_depth: int = 0,
    ordered_actions: bool = False,
) -> SearchResult[StateT, ObsT, ActionT]:
    """Find the shortest path whose observation increases external score.

    ``actions_for`` is evaluated at every expanded state so moving objects,
    changing controls, and newly exposed targets can enter the search. When
    ``ordered_actions`` is true, the provider must instead return the same
    ordered sequence at every state; this is valid for commuting involutions,
    where each action occurs at most once.
    """
    root_score = score(root_observation)
    root_key = state_key(root_observation)
    seen = {root_key}
    queue = deque([(root_state, root_observation, (), 0)])
    expanded = 0
    generated = 0

    while (
        queue
        and expanded < max_states
        and (max_transitions <= 0 or generated < max_transitions)
    ):
        state, observation, path, next_index = queue.popleft()
        expanded += 1
        if max_depth and len(path) >= max_depth:
            continue
        actions = actions_for(state, observation)
        begin = next_index if ordered_actions else 0
        for index in range(begin, len(actions)):
            if max_transitions > 0 and generated >= max_transitions:
                break
            action = actions[index]
            generated += 1
            try:
                child, child_observation = transition(state, action)
            except TransitionRejected:
                continue
            child_path = path + (action,)
            if score(child_observation) > root_score:
                return SearchResult(
                    child, child_observation, child_path,
                    expanded, generated, len(seen),
                )
            if terminal(child_observation):
                continue
            key = state_key(child_observation)
            if key in seen:
                continue
            seen.add(key)
            child_next = index + 1 if ordered_actions else 0
            queue.append((child, child_observation, child_path, child_next))

    return SearchResult(None, None, (), expanded, generated, len(seen))


def commuting_involutions(
    root_state: StateT,
    root_observation: ObsT,
    actions: Sequence[ActionT],
    transition: Callable[[StateT, ActionT], tuple[StateT, ObsT]],
    state_key: Callable[[ObsT], Hashable],
    *,
    max_actions: int = 32,
) -> bool:
    """Empirically verify binary, pairwise-commuting visual transitions."""
    if len(actions) > max_actions:
        return False
    root_key = state_key(root_observation)
    one_step: list[tuple[StateT, ObsT]] = []
    for action in actions:
        try:
            child, observation = transition(root_state, action)
            _restored, restored_observation = transition(child, action)
        except TransitionRejected:
            return False
        if state_key(restored_observation) != root_key:
            return False
        one_step.append((child, observation))

    for i, action_i in enumerate(actions):
        state_i, _ = one_step[i]
        for j in range(i + 1, len(actions)):
            state_j, _ = one_step[j]
            try:
                _, obs_ij = transition(state_i, actions[j])
                _, obs_ji = transition(state_j, action_i)
            except TransitionRejected:
                return False
            if state_key(obs_ij) != state_key(obs_ji):
                return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game-prefixes", default="ft09")
    parser.add_argument("--max-levels", type=int, default=0,
                        help="Stop after this many completed levels; 0 = win.")
    parser.add_argument("--max-states", type=int, default=50000,
                        help="Maximum expanded states per level.")
    parser.add_argument("--max-transitions", type=int, default=100000,
                        help="Maximum cloned transitions per level; 0 = unlimited.")
    parser.add_argument("--max-depth", type=int, default=0,
                        help="Maximum actions in one level solution; 0 = unlimited.")
    parser.add_argument("--max-proposals", type=int, default=256)
    parser.add_argument("--crop-border", type=int, default=1,
                        help="Ignore this many border pixels in visual dedup keys.")
    parser.add_argument("--commutativity-limit", type=int, default=32)
    parser.add_argument("--require-levels", type=int, default=0,
                        help="Exit 2 unless every selected game reaches this level.")
    parser.add_argument("--require-games-reached", type=int, default=0,
                        help="Exit 2 unless this many games reach at least L1.")
    parser.add_argument("--require-max-level", type=int, default=0,
                        help="Exit 2 unless at least one game reaches this level.")
    parser.add_argument("--save-demonstrations", default=None,
                        help="Write successful state/action pairs as JSON for "
                        "action/coordinate policy distillation.")
    parser.add_argument(
        "--demonstration-encoding",
        choices=("hybrid", "object"),
        default="hybrid",
        help="64-value state token stored in demonstrations.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from arc_agi import Arcade
        from arcengine import ActionInput, GameAction, GameState
    except ImportError as exc:
        print(f"missing ARC-AGI toolkit: {exc}", file=sys.stderr)
        return 1

    from object_features import (
        hybrid_token,
        object_candidates,
        object_centroids,
        object_token,
    )

    demo_encoder = hybrid_token if args.demonstration_encoding == "hybrid" else object_token
    demo_encoding_name = (
        "hybrid_token_v2"
        if args.demonstration_encoding == "hybrid"
        else "object_token_k8"
    )

    arcade = Arcade()
    prefixes = [p.strip() for p in args.game_prefixes.split(",") if p.strip()]
    environments = [
        info for info in arcade.get_environments()
        if any(info.game_id.startswith(prefix) for prefix in prefixes)
    ]
    if not environments:
        print(f"no environments match {prefixes}", file=sys.stderr)
        return 1

    results: list[tuple[str, int, int, int]] = []
    demonstrations: list[dict] = []
    for info in environments:
        wrapper = arcade.make(info.game_id)
        observation = wrapper.reset()
        if observation is None or not observation.frame:
            print(f"{info.game_id}: reset returned no frame", file=sys.stderr)
            results.append((info.game_id, 0, 0, 0))
            continue
        game = wrapper._game
        if game is None:
            results.append((info.game_id, 0, 0, 0))
            continue

        action_by_value = {int(a.value): a for a in GameAction}

        def obs_key(obs) -> tuple[int, str, bytes]:
            frame_key = b""
            if obs.frame:
                frame_key = visual_state_key(obs.frame[-1], args.crop_border)
            state_name = getattr(obs.state, "name", str(obs.state))
            return int(obs.levels_completed), state_name, frame_key

        def transition(
            state,
            proposal: ArcAction,
            action_by_value=action_by_value,
        ):
            child = copy.deepcopy(state)
            action = action_by_value[proposal.value]
            data = {}
            if proposal.x is not None:
                data = {"x": int(proposal.x), "y": int(proposal.y)}
            try:
                out = child.perform_action(
                    ActionInput(id=action, data=data), raw=True,
                )
            except RuntimeError as exc:
                raise TransitionRejected(str(exc)) from exc
            return child, out

        def proposals_for(
            obs,
            action_by_value=action_by_value,
        ) -> list[ArcAction]:
            available = [int(a) for a in obs.available_actions]
            coords: list[tuple[int, int]] = []
            if obs.frame:
                coords = [
                    (x, y) for y, x in object_centroids(
                        np.asarray(obs.frame[-1]), args.max_proposals,
                    )
                ]
            if not coords:
                coords = [(32, 32)]
            proposals = []
            for value in available:
                action = action_by_value.get(value)
                if action is None:
                    continue
                if action.is_complex():
                    proposals.extend(ArcAction(value, x, y) for x, y in coords)
                else:
                    proposals.append(ArcAction(value))
            return proposals

        def settle_pending_level(state, obs):
            if not getattr(state, "_next_level", False):
                return state, obs
            available = proposals_for(obs)
            if not available:
                return state, obs
            # ARC consumes the next action while applying a deferred level
            # transition. The proposal's identity/coordinate has no effect.
            return transition(state, available[0])

        total_expanded = 0
        executed_path: list[ArcAction] = []
        target = (
            min(int(observation.win_levels), args.max_levels)
            if args.max_levels > 0 else int(observation.win_levels)
        )
        while int(observation.levels_completed) < target:
            game, observation = settle_pending_level(game, observation)
            level = int(observation.levels_completed)
            raw_proposals = proposals_for(observation)
            root_key = obs_key(observation)
            unique_outcomes: dict[Hashable, ArcAction] = {}
            immediate = None
            noops = 0
            rejected = 0
            for proposal in raw_proposals:
                try:
                    child, child_obs = transition(game, proposal)
                except TransitionRejected:
                    rejected += 1
                    continue
                if int(child_obs.levels_completed) > level:
                    immediate = SearchResult(
                        child, child_obs, (proposal,), 1, 1, 1,
                    )
                    break
                key = obs_key(child_obs)
                if key == root_key:
                    noops += 1
                    continue
                unique_outcomes.setdefault(key, proposal)

            actions = list(unique_outcomes.values())
            ordered = commuting_involutions(
                game, observation, actions, transition, obs_key,
                max_actions=args.commutativity_limit,
            ) if actions else False
            print(
                f"{info.game_id} L{level}: raw={len(raw_proposals)} "
                f"basis={len(actions)} noops={noops} rejected={rejected} "
                f"commuting_involutions={ordered}",
                flush=True,
            )
            if ordered:
                # The empirically verified commutative fast path intentionally
                # keeps one root-relative action basis. General state graphs
                # regenerate object/action proposals at every node.
                actions_for = lambda _state, _obs, actions=actions: actions
            else:
                actions_for = lambda _state, obs: proposals_for(obs)
            result = immediate or breadth_first_reward_search(
                game,
                observation,
                actions_for,
                transition,
                obs_key,
                lambda obs: int(obs.levels_completed),
                lambda obs: obs.state in (GameState.GAME_OVER, GameState.WIN),
                max_states=args.max_states,
                max_transitions=args.max_transitions,
                max_depth=args.max_depth,
                ordered_actions=ordered,
            )
            total_expanded += result.expanded
            if not result.found:
                print(
                    f"{info.game_id} L{level}: no reward within "
                    f"{result.expanded} expanded/{result.generated} generated/"
                    f"{result.seen} seen states",
                    flush=True,
                )
                break
            # Replay the short winning path from this level's root to retain
            # the exact pre-action visual state paired with each coordinate.
            # Search itself stores only paths to keep its queue compact.
            demo_game = game
            demo_observation = observation
            for proposal in result.path:
                frame = np.asarray(demo_observation.frame[-1], dtype=np.int32)
                params = None
                object_index = None
                candidate_rows = object_candidates(frame, args.max_proposals)
                object_points = [(y, x) for y, x, _features in candidate_rows]
                if proposal.x is not None:
                    params = [
                        2.0 * float(proposal.x) / 63.0 - 1.0,
                        2.0 * float(proposal.y) / 63.0 - 1.0,
                    ]
                    object_xy = [(x, y) for y, x in object_points]
                    try:
                        object_index = object_xy.index((proposal.x, proposal.y))
                    except ValueError:
                        pass
                next_demo_game, next_demo_observation = transition(
                    demo_game, proposal,
                )
                demonstrations.append({
                    "game_id": info.game_id,
                    "level": level,
                    "observation": demo_encoder(frame).tolist(),
                    "action_index": proposal.value - 1,
                    "available_action_indices": [
                        int(action) - 1
                        for action in demo_observation.available_actions
                    ],
                    "action_parameters": params,
                    "object_index": object_index,
                    "object_candidate_count": len(candidate_rows),
                    "object_candidate_features": [
                        features.tolist()
                        for _y, _x, features in candidate_rows
                    ],
                    "reward": max(
                        0,
                        int(next_demo_observation.levels_completed)
                        - int(demo_observation.levels_completed),
                    ),
                })
                demo_game = next_demo_game
                demo_observation = next_demo_observation
            game = result.state
            observation = result.observation
            assert game is not None and observation is not None
            executed_path.extend(result.path)
            print(
                f"{info.game_id} L{level}: solved in {len(result.path)} actions; "
                f"expanded={result.expanded} generated={result.generated} "
                f"path={' '.join(a.label() for a in result.path)}",
                flush=True,
            )

        reached = int(observation.levels_completed)
        win_levels = int(observation.win_levels)
        results.append((info.game_id, reached, win_levels, total_expanded))
        print(
            f"{info.game_id}: reached {reached}/{win_levels}; "
            f"solution_actions={len(executed_path)} expanded={total_expanded}",
            flush=True,
        )

    games_reached = sum(reached > 0 for _, reached, _, _ in results)
    max_level = max(reached for _, reached, _, _ in results)
    print(
        f"summary: games_reached={games_reached}/{len(results)} "
        f"max_level={max_level} total_expanded="
        f"{sum(expanded for _, _, _, expanded in results)}",
        flush=True,
    )

    if args.save_demonstrations:
        output_path = Path(args.save_demonstrations)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "kindle-arc-policy-demonstrations-v1",
            "observation_encoding": demo_encoding_name,
            "coordinate_range": [-1.0, 1.0],
            "samples": demonstrations,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved {len(demonstrations)} demonstrations to {output_path}")

    gate_errors = []
    if args.require_levels > 0:
        failed = [name for name, reached, _, _ in results if reached < args.require_levels]
        if failed:
            gate_errors.append(
                f"gate failed: require_levels={args.require_levels}; "
                f"below gate={','.join(failed)}"
            )
    if games_reached < args.require_games_reached:
        gate_errors.append(
            f"gate failed: games reached={games_reached}, "
            f"required={args.require_games_reached}"
        )
    if max_level < args.require_max_level:
        gate_errors.append(
            f"gate failed: max level={max_level}, required={args.require_max_level}"
        )
    if gate_errors:
        print("\n".join(gate_errors), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
