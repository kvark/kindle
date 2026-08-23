"""Distill ARC object-search actions into Kindle's persisted action heads.

Input comes from ``arc_agi3_object_search.py --save-demonstrations``. The
world model remains frozen during this supervised phase. Kindle's end-to-end
policy encoder learns action identities from direct masked labels. A variable-
set pointer scores exported object features for complex actions; older corpora
without those labels fall back to the CPU coordinate MLP. Optional evaluation
executes deterministic heads on local games without clone search.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SUPPORTED_DEMONSTRATION_FORMATS = {
    "kindle-arc-coordinate-demonstrations-v1",
    "kindle-arc-policy-demonstrations-v1",
}
_SUPPORTED_OBSERVATION_ENCODINGS = {"object_token_k8", "hybrid_token_v2"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "demonstrations",
        nargs="+",
        help="one or more compatible object-search demonstration JSON files",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--coord-lr", type=float, default=0.002)
    parser.add_argument("--coord-freeze-mse", type=float, default=0.001)
    parser.add_argument("--policy-lr", type=float, default=0.0005)
    parser.add_argument("--policy-check-interval", type=int, default=100)
    parser.add_argument("--coord-hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--candidate-pointer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "score the current variable object set when demonstrations "
            "contain object_candidate_features"
        ),
    )
    parser.add_argument(
        "--pointer-epochs",
        type=int,
        default=None,
        help="pointer updates; default min(--epochs, 30000)",
    )
    parser.add_argument("--pointer-lr", type=float, default=0.0005)
    parser.add_argument("--pointer-hidden-dim", type=int, default=128)
    parser.add_argument("--pointer-hidden-dim-2", type=int, default=64)
    parser.add_argument(
        "--eval-pointer-max-candidates",
        type=int,
        default=256,
        help="maximum live object candidates scored per complex action",
    )
    parser.add_argument(
        "--require-pointer-accuracy",
        type=float,
        default=0.0,
        help="fail unless pointer demonstration accuracy reaches this fraction",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument(
        "--load-weights-dir",
        default=None,
        help="initialize from a saved distillation checkpoint",
    )
    parser.add_argument("--evaluate-game-prefixes", default=None)
    parser.add_argument("--eval-max-steps", type=int, default=100)
    parser.add_argument(
        "--eval-snap-max-objects",
        type=int,
        default=64,
        help=(
            "project learned complex-action coordinates onto the nearest of "
            "this many visible object representatives; 0 disables projection"
        ),
    )
    parser.add_argument(
        "--eval-trace",
        action="store_true",
        help="print each deterministic evaluation action and coordinate",
    )
    parser.add_argument(
        "--eval-object-motion-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="extrapolate two matching click displacements before projection",
    )
    parser.add_argument("--require-eval-level", type=int, default=0)
    parser.add_argument(
        "--require-eval-levels",
        default=None,
        help=(
            "comma-separated prefix:level overrides, for example "
            "ft09:2,vc33:2"
        ),
    )
    parser.add_argument("--require-eval-games-reached", type=int, default=0)
    return parser.parse_args()


def _mse(predictions, targets, active) -> float:
    errors = [
        (px - tx) ** 2 + (py - ty) ** 2
        for (px, py), (tx, ty), enabled in zip(predictions, targets, active)
        if enabled
    ]
    return float(sum(errors) / max(1, len(errors)))


def _accuracy(predictions, targets) -> float:
    return sum(
        int(prediction == target)
        for prediction, target in zip(predictions, targets)
    ) / max(1, len(targets))


def _pointer_sample(sample: dict) -> bool:
    """Whether a demonstration fully labels a variable candidate choice."""
    candidates = sample.get("object_candidate_features")
    object_index = sample.get("object_index")
    return (
        sample.get("action_parameters") is not None
        and object_index is not None
        and isinstance(candidates, list)
        and bool(candidates)
        and 0 <= int(object_index) < len(candidates)
    )


def _load_demonstrations(paths: list[str | Path]) -> tuple[str, list[dict]]:
    """Load compatible corpora and remove exact repeated state labels.

    The policy observes game identity, the 64-value token, and the availability
    mask. If that complete model-visible state has different labels in two
    files, merging would create an impossible supervised target, so fail early
    instead of silently training on contradictory rows.
    """
    observation_encoding = None
    samples = []
    labels_by_state = {}
    sample_index_by_state = {}
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") not in _SUPPORTED_DEMONSTRATION_FORMATS:
            raise ValueError(f"{path}: unsupported demonstration format")
        encoding = payload.get("observation_encoding")
        if encoding not in _SUPPORTED_OBSERVATION_ENCODINGS:
            raise ValueError(f"{path}: unsupported observation encoding")
        if observation_encoding is None:
            observation_encoding = encoding
        elif encoding != observation_encoding:
            raise ValueError(
                f"{path}: observation encoding {encoding!r} does not match "
                f"{observation_encoding!r}"
            )
        file_samples = payload.get("samples", [])
        if not file_samples:
            raise ValueError(f"{path}: demonstration file has no samples")
        for sample in file_samples:
            available = tuple(sorted(int(index) for index in sample.get(
                "available_action_indices", (sample["action_index"],)
            )))
            state_key = (
                str(sample["game_id"]),
                tuple(float(value) for value in sample["observation"]),
                available,
            )
            parameters = sample.get("action_parameters")
            label = (
                int(sample["action_index"]),
                None if parameters is None else tuple(float(v) for v in parameters),
            )
            previous = labels_by_state.get(state_key)
            if previous is not None and previous != label:
                raise ValueError(
                    f"{path}: conflicting labels for a model-visible state in "
                    f"{sample['game_id']}: {previous!r} versus {label!r}"
                )
            if previous is None:
                labels_by_state[state_key] = label
                sample_index_by_state[state_key] = len(samples)
                samples.append(sample)
            else:
                # Search caps can truncate the same deterministic candidate
                # set differently across corpus files. Retain the richer row
                # without duplicating its policy/coordinate supervision.
                index = sample_index_by_state[state_key]
                prior_candidates = samples[index].get(
                    "object_candidate_features", ()
                )
                new_candidates = sample.get("object_candidate_features", ())
                if len(new_candidates) > len(prior_candidates):
                    samples[index] = sample
    assert observation_encoding is not None
    return observation_encoding, samples


def _parse_level_requirements(specification: str | None) -> list[tuple[str, int]]:
    requirements = []
    if not specification:
        return requirements
    for entry in specification.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            raise ValueError(
                "--require-eval-levels entries must use non-empty prefix:level"
            )
        prefix, raw_level = (part.strip() for part in entry.split(":", 1))
        if not prefix:
            raise ValueError("--require-eval-levels prefix must not be empty")
        try:
            level = int(raw_level)
        except ValueError as exc:
            raise ValueError(
                f"--require-eval-levels invalid level {raw_level!r}"
            ) from exc
        if level < 0:
            raise ValueError("--require-eval-levels levels must be non-negative")
        requirements.append((prefix, level))
    return requirements


def main() -> int:
    args = _parse_args()
    try:
        level_requirements = _parse_level_requirements(args.require_eval_levels)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not np.isfinite(args.policy_lr) or args.policy_lr < 0.0:
        print("--policy-lr must be finite and non-negative", file=sys.stderr)
        return 1
    if not np.isfinite(args.coord_lr) or args.coord_lr < 0.0:
        print("--coord-lr must be finite and non-negative", file=sys.stderr)
        return 1
    if not np.isfinite(args.coord_freeze_mse) or args.coord_freeze_mse < 0.0:
        print("--coord-freeze-mse must be finite and non-negative", file=sys.stderr)
        return 1
    if not np.isfinite(args.pointer_lr) or args.pointer_lr <= 0.0:
        print("--pointer-lr must be finite and positive", file=sys.stderr)
        return 1
    if args.pointer_epochs is not None and args.pointer_epochs < 0:
        print("--pointer-epochs must be non-negative", file=sys.stderr)
        return 1
    if not 0.0 <= args.require_pointer_accuracy <= 1.0:
        print("--require-pointer-accuracy must be in [0, 1]", file=sys.stderr)
        return 1
    try:
        import kindle
    except ImportError as exc:
        print(f"missing kindle extension: {exc}", file=sys.stderr)
        return 1

    try:
        observation_encoding, samples = _load_demonstrations(args.demonstrations)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"invalid demonstrations: {exc}", file=sys.stderr)
        return 1

    game_ids = sorted({sample["game_id"] for sample in samples})
    env_id_for = {game_id: index for index, game_id in enumerate(game_ids)}
    observations = [sample["observation"] for sample in samples]
    action_indices = [int(sample["action_index"]) for sample in samples]
    pointer_active = [
        args.candidate_pointer and _pointer_sample(sample)
        for sample in samples
    ]
    pointer_samples = [
        sample for sample, enabled in zip(samples, pointer_active) if enabled
    ]
    targets = [
        tuple(sample["action_parameters"] or (0.0, 0.0))
        for sample in samples
    ]
    active = [
        sample["action_parameters"] is not None and not pointer_enabled
        for sample, pointer_enabled in zip(samples, pointer_active)
    ]
    env_ids = [env_id_for[sample["game_id"]] for sample in samples]
    available_indices = [
        int(action_index)
        for sample in samples
        for action_index in sample.get(
            "available_action_indices", (sample["action_index"],)
        )
    ]
    num_actions = max(action_indices + available_indices) + 1
    action_masks = np.zeros((len(samples), kindle.MAX_ACTION_DIM), dtype=np.float32)
    for row, sample in zip(action_masks, samples):
        for action_index in sample.get(
            "available_action_indices", (sample["action_index"],)
        ):
            if 0 <= int(action_index) < num_actions:
                row[int(action_index)] = 1.0
    if any(not row.any() for row in action_masks):
        print("demonstration contains an empty action mask", file=sys.stderr)
        return 1
    if any(not action_masks[row, action_index] for row, action_index in enumerate(action_indices)):
        print("demonstration action is absent from its availability mask", file=sys.stderr)
        return 1

    config = {
        "obs_dim": 64,
        "num_actions": num_actions,
        "batch_size": len(samples),
        "env_ids": env_ids,
        "seed": args.seed,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "coord_action_alpha": 1.0 if any(active) else 0.0,
        "coord_hidden_dim": args.coord_hidden_dim,
        "coord_lr": args.coord_lr,
        "learning_rate": 0.0,
        "lr_policy": args.policy_lr,
        "use_grpo": False,
        "value_loss_coef": 0.0,
        "end_to_end_encoder": True,
        "recon_loss_coef": 0.0,
        "reward_pred_loss_coef": 0.0,
        # The dedicated end-to-end encoder trains through the delayed rollout
        # path; two repeated passes are enough to satisfy that graph contract.
        "n_step": 2,
        "rollout_length": 2,
        "entropy_beta": 0.0,
        "entropy_floor": 0.0,
    }
    agent = kindle.BatchAgent(**config)
    if args.load_weights_dir:
        agent.load_weights(args.load_weights_dir)
    agent.set_action_masks(action_masks.reshape(-1))
    initial_actions = agent.act(observations, deterministic=True)
    initial_accuracy = _accuracy(initial_actions, action_indices)
    initial = agent.coord_means(observations)
    initial_mse = _mse(initial, targets, active)
    loss = 0.0
    policy_frozen_epoch = None
    coord_frozen_epoch = None if any(active) else 0
    epochs_run = 0
    check_interval = max(1, args.policy_check_interval)
    for epoch in range(1, max(0, args.epochs) + 1):
        epochs_run = epoch
        if policy_frozen_epoch is None:
            agent.train_policy_supervised(observations, action_indices)
        if coord_frozen_epoch is None:
            loss = agent.train_coord_head_supervised(
                observations, targets, active_mask=active,
            )
        if epoch % check_interval == 0:
            agent.set_action_masks(action_masks.reshape(-1))
            if policy_frozen_epoch is None:
                checkpoint_actions = agent.act(observations, deterministic=True)
                checkpoint_accuracy = _accuracy(checkpoint_actions, action_indices)
                if checkpoint_accuracy == 1.0:
                    agent.set_lr_policy(0.0)
                    policy_frozen_epoch = epoch
            if coord_frozen_epoch is None:
                checkpoint_coords = agent.coord_means(observations)
                checkpoint_mse = _mse(checkpoint_coords, targets, active)
                if checkpoint_mse <= args.coord_freeze_mse:
                    coord_frozen_epoch = epoch
            if policy_frozen_epoch is not None and coord_frozen_epoch is not None:
                break
    agent.set_action_masks(action_masks.reshape(-1))
    trained_actions = agent.act(observations, deterministic=True)
    final_accuracy = _accuracy(trained_actions, action_indices)
    trained = agent.coord_means(observations)
    final_mse = _mse(trained, targets, active)
    print(
        f"samples={len(samples)} games={len(game_ids)} epochs={epochs_run} "
        f"action_accuracy={initial_accuracy:.1%}->{final_accuracy:.1%} "
        f"policy_frozen_epoch={policy_frozen_epoch} "
        f"coord_frozen_epoch={coord_frozen_epoch} "
        f"coord_mse={initial_mse:.6f}->{final_mse:.6f} coord_loss={loss:.6f}"
    )
    if final_accuracy < 1.0:
        errors = [
            (
                sample["game_id"][:4],
                int(sample.get("level", 0)),
                target,
                prediction,
            )
            for sample, target, prediction in zip(
                samples, action_indices, trained_actions,
            )
            if target != prediction
        ]
        print(f"action_errors={len(errors)} examples={errors[:20]}")

    pointer = None
    pointer_initial_accuracy = 0.0
    pointer_result = None
    if pointer_samples:
        try:
            from candidate_pointer import CandidatePointer

            pointer_path = (
                Path(args.load_weights_dir) / "candidate_pointer.npz"
                if args.load_weights_dir
                else None
            )
            if pointer_path is not None and pointer_path.exists():
                pointer = CandidatePointer.load(pointer_path)
            elif args.load_weights_dir and max(0, args.epochs) == 0:
                raise ValueError(
                    f"missing candidate-pointer checkpoint {pointer_path}"
                )
            else:
                pointer_tasks = sorted({
                    str(sample["game_id"]) for sample in pointer_samples
                })
                candidate_dim = len(
                    pointer_samples[0]["object_candidate_features"][0]
                )
                pointer = CandidatePointer(
                    pointer_tasks,
                    context_dim=64,
                    candidate_dim=candidate_dim,
                    hidden_dim=args.pointer_hidden_dim,
                    hidden_dim_2=args.pointer_hidden_dim_2,
                    seed=args.seed,
                )
            pointer_initial_accuracy = pointer.accuracy(pointer_samples)
            pointer_epochs = (
                min(max(0, args.epochs), 30_000)
                if args.pointer_epochs is None
                else args.pointer_epochs
            )
            if pointer_epochs > 0 and pointer_initial_accuracy < 1.0:
                pointer_result = pointer.fit(
                    pointer_samples,
                    epochs=pointer_epochs,
                    learning_rate=args.pointer_lr,
                )
            else:
                pointer_result = {
                    "epochs": 0,
                    "accuracy": pointer_initial_accuracy,
                    "loss": 0.0,
                    "last_loss": 0.0,
                }
        except (KeyError, TypeError, ValueError, OSError) as exc:
            print(f"candidate-pointer failure: {exc}", file=sys.stderr)
            return 1
        print(
            f"pointer_samples={len(pointer_samples)} "
            f"pointer_epochs={pointer_result['epochs']} "
            f"pointer_accuracy={pointer_initial_accuracy:.1%}->"
            f"{pointer_result['accuracy']:.1%} "
            f"pointer_loss={pointer_result['loss']:.6f}"
        )

    weights_dir = args.weights_dir
    if weights_dir:
        agent.save_weights(weights_dir)
        if pointer is not None:
            pointer.save(Path(weights_dir) / "candidate_pointer.npz")
        metadata = {
            "format": "kindle-arc-policy-distill-v1",
            "demonstration_encoding": observation_encoding,
            "demonstration_files": [str(path) for path in args.demonstrations],
            "game_env_ids": env_id_for,
            "agent_config": config,
            "epochs": epochs_run,
            "initial_mse": initial_mse,
            "training_mse": final_mse,
            "initial_action_accuracy": initial_accuracy,
            "training_action_accuracy": final_accuracy,
            "policy_frozen_epoch": policy_frozen_epoch,
            "coord_frozen_epoch": coord_frozen_epoch,
            "candidate_pointer": None if pointer_result is None else {
                "samples": len(pointer_samples),
                "epochs": pointer_result["epochs"],
                "initial_accuracy": pointer_initial_accuracy,
                "training_accuracy": pointer_result["accuracy"],
                "loss": pointer_result["loss"],
                "hidden_dim": args.pointer_hidden_dim,
                "hidden_dim_2": args.pointer_hidden_dim_2,
                "learning_rate": args.pointer_lr,
            },
        }
        metadata_path = Path(weights_dir) / "distill.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"saved distilled weights to {weights_dir}")

    if (
        pointer_result is not None
        and pointer_result["accuracy"] < args.require_pointer_accuracy
    ):
        print(
            "gate failed: pointer accuracy="
            f"{pointer_result['accuracy']:.1%}, required="
            f"{args.require_pointer_accuracy:.1%}",
            file=sys.stderr,
        )
        return 2

    if not args.evaluate_game_prefixes:
        return 0
    evaluation_weights_dir = weights_dir or args.load_weights_dir
    if not evaluation_weights_dir:
        print(
            "--evaluate-game-prefixes requires --weights-dir or "
            "--load-weights-dir",
            file=sys.stderr,
        )
        return 1

    try:
        from arc_agi import Arcade
        from arcengine import GameAction, GameState
        from object_features import (
            hybrid_token,
            object_candidates,
            object_centroids,
            object_token,
            project_to_object_centroid,
            project_with_object_motion,
        )
    except ImportError as exc:
        print(f"missing ARC-AGI toolkit: {exc}", file=sys.stderr)
        return 1

    arcade = Arcade()
    encode_observation = (
        hybrid_token if observation_encoding == "hybrid_token_v2" else object_token
    )
    prefixes = [
        prefix.strip()
        for prefix in args.evaluate_game_prefixes.split(",")
        if prefix.strip()
    ]
    selected = [
        info for info in arcade.get_environments()
        if any(info.game_id.startswith(prefix) for prefix in prefixes)
    ]
    reached_levels = []
    for info in selected:
        required_level = max(
            [args.require_eval_level]
            + [
                level
                for prefix, level in level_requirements
                if info.game_id.startswith(prefix)
            ]
        )
        if info.game_id not in env_id_for:
            print(f"{info.game_id}: no demonstrations/embedding; skipped")
            reached_levels.append((info.game_id, 0, required_level))
            continue
        eval_config = dict(config)
        eval_config["batch_size"] = 1
        eval_config["env_ids"] = [env_id_for[info.game_id]]
        eval_agent = kindle.BatchAgent(**eval_config)
        eval_agent.load_weights(evaluation_weights_dir)
        wrapper = arcade.make(info.game_id)
        observation = wrapper.reset()
        click_history = []
        action_by_value = {int(action.value): action for action in GameAction}
        for step in range(args.eval_max_steps):
            if observation is None or not observation.frame:
                break
            if observation.state in (GameState.GAME_OVER, GameState.WIN):
                break
            if (
                required_level > 0
                and observation.levels_completed >= required_level
            ):
                break
            available = [int(action) for action in observation.available_actions]
            frame = np.asarray(observation.frame[-1], dtype=np.int32)
            token = encode_observation(
                frame
            ).tolist()
            mask = np.zeros(kindle.MAX_ACTION_DIM, dtype=np.float32)
            for value in available:
                mask[value - 1] = 1.0
            eval_agent.set_action_masks(mask)
            action_index = eval_agent.act([token], deterministic=True)[0]
            data = {}
            raw_data = None
            value = action_index + 1
            if action_by_value[value].is_complex():
                candidate_rows = object_candidates(
                    frame,
                    max_n=max(1, args.eval_pointer_max_candidates),
                )
                if (
                    candidate_rows
                    and pointer is not None
                    and info.game_id in pointer.task_index
                ):
                    pointer_index = pointer.choose(
                        token,
                        info.game_id,
                        [features for _y, _x, features in candidate_rows],
                    )
                    y, x, _features = candidate_rows[pointer_index]
                    raw_data = {"x": x, "y": y}
                    data = raw_data
                    click_history.clear()
                else:
                    x_norm, y_norm = eval_agent.coord_means([token])[0]
                    x = max(0, min(63, round((x_norm + 1.0) * 0.5 * 63.0)))
                    y = max(0, min(63, round((y_norm + 1.0) * 0.5 * 63.0)))
                    raw_data = {"x": x, "y": y}
                    if args.eval_snap_max_objects > 0:
                        if args.eval_object_motion_prior:
                            x, y = project_with_object_motion(
                                frame,
                                x,
                                y,
                                click_history,
                                max_n=args.eval_snap_max_objects,
                            )
                        else:
                            x, y = project_to_object_centroid(
                                frame,
                                x,
                                y,
                                max_n=args.eval_snap_max_objects,
                            )
                    data = {"x": x, "y": y}
                    click_history.append((x, y))
            else:
                click_history.clear()
            if args.eval_trace:
                suffix = "" if raw_data == data else f" raw={raw_data}"
                if raw_data is not None:
                    nearest = sorted(
                        object_centroids(
                            frame,
                            max_n=max(1, args.eval_snap_max_objects),
                        ),
                        key=lambda point: (
                            (point[1] - raw_data["x"]) ** 2
                            + (point[0] - raw_data["y"]) ** 2
                        ),
                    )[:5]
                    suffix += f" nearest={[(x, y) for y, x in nearest]}"
                print(f"{info.game_id} step={step} action=A{value} data={data}{suffix}")
            previous_level = int(observation.levels_completed)
            observation = wrapper.step(action_by_value[value], data=data)
            if (
                observation is not None
                and int(observation.levels_completed) > previous_level
            ):
                click_history.clear()
        reached = 0 if observation is None else int(observation.levels_completed)
        reached_levels.append((info.game_id, reached, required_level))
        print(f"{info.game_id}: distilled policy reached {reached} level(s)")

    if not selected:
        print(f"no environments match {prefixes}", file=sys.stderr)
        return 1
    failed = [
        f"{game_id}<{required}"
        for game_id, reached, required in reached_levels
        if reached < required
    ]
    if failed:
        print(
            "gate failed: required evaluation levels not reached; "
            f"below gate={','.join(failed)}",
            file=sys.stderr,
        )
        return 2
    games_reached = sum(reached > 0 for _, reached, _ in reached_levels)
    if games_reached < args.require_eval_games_reached:
        print(
            f"gate failed: eval games reached={games_reached}, "
            f"required={args.require_eval_games_reached}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
