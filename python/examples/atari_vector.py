"""Independent Atari streams, one native batched Dreamer learner.

--steps counts aggregate executed actions, not vector ticks or per-env actions.
CPU environment stepping is synchronous; GPU inference is batched. Terminal
observations are consumed before individually resetting completed environments.
Learned RGB is the 2D default; --encoder-checkpoint selects frozen causal JEPA.
"""

import argparse
import inspect
import json
import math
import os
import signal
import struct
import sys
import time
from pathlib import Path

import ale_py
import gymnasium as gym

import kindle
import atari
from kindle._vector_audit import EPISODE_EVALUATION_PROTOCOL, VECTOR_PROTOCOL, episode_summary, require_numbers
import kindle._pixel_guidance as guidance_schedule_module
import kindle._qbert_guidance as guidance_module
import kindle._breakout_guidance as breakout_guidance_module
from kindle._breakout_guidance import BreakoutGuidance
from kindle._qbert_guidance import BOUNDED_GUIDANCE_KIND, GUIDANCE_KIND, QbertGuidance
import kindle._exploration as exploration_module
from kindle._exploration import EXPLORATION_KIND, EXPLORATION_PROTOCOL, PersistentExploration
from atari import (
    ATARI_ACTION_REPEAT, ATARI_PROTOCOLS, DreamerAtariPreprocessing,
    checkpoint_identity, sha256_file,
)


def require_gpu_budget(snapshot, minimum_bytes):
    if (not isinstance(snapshot, dict) or set(snapshot) != {"usage_bytes", "budget_bytes"}
            or any(type(value) is not int or value < 0 for value in snapshot.values())):
        raise ValueError("invalid native GPU memory budget")
    if snapshot["budget_bytes"] == 0:
        raise ValueError("native GPU memory budget is unsupported")
    if snapshot["budget_bytes"] - snapshot["usage_bytes"] < minimum_bytes:
        raise ValueError("native GPU memory budget headroom below declared minimum")


def positive_f32(value):
    try:
        value, = struct.unpack("=f", struct.pack("=f", value))
    except OverflowError:
        return False
    return math.isfinite(value) and value > 0


def require_gpu_device(snapshot, expected):
    if snapshot.get("device_name") != expected or snapshot.get("is_software_emulated") is not False:
        raise ValueError(f"expected native device {expected}, got {snapshot}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", help="opt into causal JEPA (frozen unless encoder-training=joint); omitted: jointly learned RGB")
    parser.add_argument("--encoder", choices=("levjepa", "levjepa-tiny"),
                        help="fresh default: levjepa-tiny; restore default: recorded checkpoint kind")
    parser.add_argument("--encoder-training", choices=("frozen", "joint"),
                        help="Tiny-only experiment: native pixel replay with phase-aligned causal re-encoding")
    parser.add_argument("--actor-critic-gradient", action="store_true",
                        help="train posterior representations from initial imagined actor/value losses (upstream ac_grads)")
    parser.add_argument("--cdp", action="store_true",
                        help="learned CNN with cosine feature prediction instead of RGB reconstruction; CDP split learning rates")
    parser.add_argument("--image-size", type=int, choices=(32, 48, 64),
                        help="learned RGB/CDP GPU resize side; fresh default64, restore uses checkpoint geometry")
    parser.add_argument("--cdp-centered", action="store_true",
                        help="CDP ablation: center cosine predictions/targets over the full replay batch")
    parser.add_argument("--disagreement-scale", type=float, default=0.0,
                        help="CDP exploration: current latent disagreement in imagined and replay returns; zero disables the ensemble")
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atari-protocol", choices=ATARI_PROTOCOLS, default="published")
    parser.add_argument("--observation-size", choices=("native", "64"),
                        help="fresh default: native max-pooled RGB; restore requires an explicit choice")
    parser.add_argument("--sticky-actions", type=float, choices=(0.0, 0.25), default=0.0,
                        help="ALE action repeat probability; declare .25 separately for robustness evaluation")
    parser.add_argument("--model-size", default="12m")
    parser.add_argument("--training-precision", choices=("auto", "f32", "f16", "bf16", "mixed"), default="auto",
                        help="bf16 uses native reduced dense/convolution storage; mixed uses F16-input/F32-storage world gradients with BF16 elsewhere; masters and optimizer stay F32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--world-microbatch-size", type=int)
    parser.add_argument("--imagination-length", type=int,
                        help="fresh imagined rollout steps; omitted: native default, restore keeps checkpoint length")
    parser.add_argument("--train-ratio", type=float, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.00004,
                        help="fresh base rate; CDP encoder/dynamics rates scale proportionally from default4e-5")
    parser.add_argument("--behavior-learning-rate", type=float,
                        help="fresh actor/critic rate; omitted: native default follows the base rate")
    parser.add_argument("--actor-unimix", type=float, default=0.0,
                        help="native policy uniform mixture in [0,1); fresh default0, restore keeps checkpoint policy")
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="fresh training only: save the actual zero-experience weights for frozen controls")
    parser.add_argument("--checkpoint-every", type=int, default=20000)
    parser.add_argument("--checkpoint-history", action="store_true",
                        help="preserve each save in CHECKPOINT/<run-actions> instead of replacing the last save")
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--probe-behavior", action="store_true",
                        help="frozen diagnostic: log live critic values and scaled action-effects bonuses")
    parser.add_argument("--episodes-per-env", type=int,
                        help="frozen restore only: stop after this many natural episodes per stream; timeouts do not count, steps is a hard cap")
    parser.add_argument("--min-gpu-budget-headroom-mib", type=int,
                        help="check native budget after each GPU stage; write OUTPUT.gpu-memory.jsonl")
    parser.add_argument("--exploration-probability", type=float, default=0.0)
    parser.add_argument("--exploration-hold", type=int, default=16)
    parser.add_argument("--qbert-guidance-probability", type=float, default=0.0,
                        help="fresh Qbert only: probability of selecting an episode for pixel guidance")
    parser.add_argument("--qbert-guidance-policy", choices=("first-pyramid", "continuation"),
                        help="fresh guided Qbert only; continuation requires an explicit action cap")
    parser.add_argument("--qbert-guidance-max-actions", type=int,
                        help="release each selected episode to the learner after this many actions")
    parser.add_argument("--qbert-corrective-labels", action="store_true",
                        help="fresh continuation imitation: label all learner states and give the teacher action feedback")
    parser.add_argument("--breakout-guidance-probability", type=float, default=0.0,
                        help="fresh Breakout/full18 only: probability of pixel-guided episodes")
    parser.add_argument("--breakout-guidance-max-actions", type=int,
                        help="required action cap for each selected Breakout episode")
    parser.add_argument("--breakout-corrective-labels", action="store_true",
                        help="fresh Breakout imitation: edge-corrected teacher labels all states with actual-action feedback")
    parser.add_argument("--guidance-imitation-scale", type=float,
                        help="fresh guided training only: native supervised actor loss on expert actions")
    parser.add_argument("--guidance-noop-weight", type=float,
                        help="relative NOOP target weight for expert imitation; default1")
    parser.add_argument("--guidance-world-gradient", action="store_true",
                        help="fresh imitation only: route expert loss through the frozen actor into world state")
    args = parser.parse_args()
    corrective_labels = args.qbert_corrective_labels or args.breakout_corrective_labels
    if args.num_envs <= 0 or args.steps <= 0 or args.steps % args.num_envs:
        parser.error("steps must be a positive multiple of num-envs")
    if args.report_every <= 0 or args.checkpoint_every <= 0:
        parser.error("report/checkpoint intervals must be positive")
    if args.checkpoint_history and args.checkpoint is None:
        parser.error("checkpoint-history requires --checkpoint")
    if args.initial_checkpoint and (args.restore or args.evaluate):
        parser.error("initial-checkpoint requires fresh training")
    if args.world_microbatch_size is not None and args.world_microbatch_size <= 0:
        parser.error("world-microbatch-size must be positive")
    if not positive_f32(args.learning_rate):
        parser.error("learning-rate must be finite and positive in F32")
    if args.behavior_learning_rate is not None:
        if not positive_f32(args.behavior_learning_rate):
            parser.error("behavior-learning-rate must be finite and positive in F32")
        if not positive_f32(args.behavior_learning_rate / args.learning_rate):
            parser.error("behavior-learning-rate ratio must be finite and positive in F32")
    if args.imagination_length is not None and args.imagination_length <= 0:
        parser.error("imagination-length must be positive")
    if args.min_gpu_budget_headroom_mib is not None and args.min_gpu_budget_headroom_mib <= 0:
        parser.error("GPU budget headroom must be positive")
    if args.greedy and not args.evaluate:
        parser.error("greedy actions are only supported for frozen evaluation")
    if args.probe_behavior and not args.evaluate:
        parser.error("behavior probing requires frozen evaluation")
    if args.episodes_per_env is not None:
        if args.episodes_per_env <= 0:
            parser.error("episodes-per-env must be positive")
        if not args.evaluate or not args.restore:
            parser.error("episodes-per-env requires frozen restore")
    if not math.isfinite(args.exploration_probability) or not 0 <= args.exploration_probability <= 1:
        parser.error("exploration probability must be in [0, 1]")
    if not math.isfinite(args.qbert_guidance_probability) or not 0 <= args.qbert_guidance_probability <= 1:
        parser.error("Qbert guidance probability must be in [0, 1]")
    if not math.isfinite(args.breakout_guidance_probability) or not 0 <= args.breakout_guidance_probability <= 1:
        parser.error("Breakout guidance probability must be in [0, 1]")
    if args.breakout_guidance_max_actions is not None:
        if not 0 < args.breakout_guidance_max_actions < 2**32:
            parser.error("Breakout guidance max-actions must be a positive u32")
        if not args.breakout_guidance_probability:
            parser.error("Breakout guidance max-actions requires positive guidance probability")
    if args.breakout_guidance_probability:
        if args.breakout_guidance_max_actions is None:
            parser.error("Breakout guidance requires an explicit max-actions cap")
        if args.evaluate:
            parser.error("frozen evaluation must not use Breakout guidance")
        if args.exploration_probability or args.qbert_guidance_probability:
            parser.error("Breakout guidance cannot combine with another action aid")
        if (args.environment != "ALE/Breakout-v5" or args.atari_protocol != "published"
                or args.observation_size not in (None, "native") or args.encoder_checkpoint is not None):
            parser.error("Breakout guidance requires Breakout/full18/native pixels/learned RGB")
    if args.guidance_imitation_scale is not None:
        if not positive_f32(args.guidance_imitation_scale):
            parser.error("guidance imitation scale must be positive finite F32")
        if not (args.qbert_guidance_probability or args.breakout_guidance_probability) or args.evaluate:
            parser.error("guidance imitation requires fresh guided training")
    if args.guidance_noop_weight is not None:
        if args.guidance_imitation_scale is None:
            parser.error("guidance NOOP weight requires imitation")
        if not positive_f32(args.guidance_noop_weight) or args.guidance_noop_weight > 1:
            parser.error("guidance NOOP weight must be positive F32 and at most1")
    if args.guidance_world_gradient and args.guidance_imitation_scale is None:
        parser.error("guidance world gradient requires imitation")
    if args.breakout_corrective_labels and (not args.breakout_guidance_probability
                                            or args.guidance_imitation_scale is None):
        parser.error("Breakout corrective labels require Breakout guidance and imitation")
    if args.qbert_corrective_labels and (args.qbert_guidance_policy != "continuation"
                                         or args.guidance_imitation_scale is None):
        parser.error("Qbert corrective labels require continuation guidance and imitation")
    if args.qbert_guidance_policy is not None and not args.qbert_guidance_probability:
        parser.error("Qbert guidance policy requires positive guidance probability")
    if args.qbert_guidance_policy == "continuation" and args.qbert_guidance_max_actions is None:
        parser.error("Qbert continuation guidance requires an explicit max-actions cap")
    if args.qbert_guidance_max_actions is not None:
        if not 0 < args.qbert_guidance_max_actions < 2**32:
            parser.error("Qbert guidance max-actions must be a positive u32")
        if not args.qbert_guidance_probability:
            parser.error("Qbert guidance max-actions requires positive guidance probability")
    if args.qbert_guidance_probability:
        if args.evaluate:
            parser.error("frozen evaluation must not use Qbert guidance")
        if args.exploration_probability:
            parser.error("Qbert guidance cannot combine with persistent exploration")
        if (args.environment != "ALE/Qbert-v5" or args.atari_protocol != "published"
                or args.observation_size not in (None, "native") or args.encoder_checkpoint is not None):
            parser.error("Qbert guidance requires Qbert/full18/native pixels/learned RGB")
    if args.exploration_hold <= 0:
        parser.error("exploration hold must be positive")
    if not math.isfinite(args.actor_unimix) or not 0 <= args.actor_unimix < 1:
        parser.error("actor-unimix must be finite and in [0, 1)")
    if args.evaluate and args.exploration_probability:
        parser.error("frozen evaluation must not use exploration overrides")
    training_options = {"--model-size", "--batch-size", "--batch-length", "--world-microbatch-size", "--train-ratio", "--learning-rate", "--exploration-probability", "--exploration-hold", "--encoder-training", "--actor-critic-gradient", "--replay-capacity"}
    training_options.add("--qbert-guidance-probability")
    training_options.add("--qbert-guidance-max-actions")
    training_options.add("--qbert-guidance-policy")
    training_options.add("--qbert-corrective-labels")
    training_options.update(("--breakout-guidance-probability", "--breakout-guidance-max-actions", "--breakout-corrective-labels"))
    training_options.update(("--guidance-imitation-scale", "--guidance-noop-weight", "--guidance-world-gradient"))
    training_options.add("--image-size")
    training_options.add("--actor-unimix")
    training_options.add("--imagination-length")
    training_options.add("--behavior-learning-rate")
    training_options.add("--cdp")
    training_options.add("--cdp-centered")
    training_options.add("--disagreement-scale")
    training_options.add("--training-precision")
    if args.restore and any(arg.split("=", 1)[0] in training_options for arg in sys.argv[1:]):
        parser.error("training overrides require a fresh run; restore uses checkpoint config")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must fit an unsigned 32-bit integer")
    if args.output.exists() or (args.checkpoint and args.checkpoint.exists()):
        parser.error("output and checkpoint must be fresh paths")
    if args.initial_checkpoint and (args.initial_checkpoint.exists() or args.initial_checkpoint == args.checkpoint):
        parser.error("initial-checkpoint must be a separate fresh path")
    memory_path = args.output.with_suffix(".gpu-memory.jsonl")
    if args.min_gpu_budget_headroom_mib is not None and memory_path.exists():
        parser.error("GPU memory output must be a fresh path")
    if (args.exploration_probability or args.qbert_guidance_probability or args.breakout_guidance_probability) and "action_overrides" not in inspect.signature(kindle.VectorAgent.act).parameters:
        parser.error("persistent exploration requires native vector action overrides")
    if args.guidance_imitation_scale is not None and "expert_actions" not in inspect.signature(kindle.VectorAgent.act).parameters:
        parser.error("guidance imitation requires native expert-action support")
    if corrective_labels and "expert_labels" not in inspect.signature(kindle.VectorAgent.act).parameters:
        parser.error("corrective labels require native passive-label support")
    if args.guidance_world_gradient and not getattr(kindle.VectorAgent, "SUPPORTS_EXPERT_WORLD_GRADIENT", False):
        parser.error("guidance world gradient requires native world-gradient support")
    if args.restore and args.encoder:
        metadata = json.loads((args.restore / "metadata.json").read_text())
        if (metadata.get("perception") or {}).get("kind") != args.encoder:
            parser.error("encoder selection differs from restored checkpoint")
    if args.restore and args.observation_size is None:
        parser.error("restore requires --observation-size; checkpoints do not record Atari preprocessing")
    args.observation_size = args.observation_size or "native"
    learned_rgb = args.encoder_checkpoint is None
    if args.image_size is not None and not learned_rgb:
        parser.error("image-size requires the learned RGB/CNN frontend")
    if args.cdp and not learned_rgb:
        parser.error("CDP requires the learned CNN")
    if args.cdp:
        rate_scale = args.learning_rate / 4e-5
        cdp_rates = dict(encoder_learning_rate=6e-6 * rate_scale,
                         dynamics_learning_rate=4e-4 * rate_scale)
        if not all(positive_f32(rate) for rate in cdp_rates.values()):
            parser.error("CDP learning rates must be finite and positive in F32")
    if args.cdp_centered and (not args.cdp or args.world_microbatch_size not in (None, args.batch_size)):
        parser.error("centered CDP requires --cdp and full-batch world training")
    if not math.isfinite(args.disagreement_scale) or args.disagreement_scale < 0:
        parser.error("disagreement-scale must be finite and non-negative")
    if args.disagreement_scale and not args.cdp:
        parser.error("disagreement requires fresh CDP")
    if args.replay_capacity <= 0:
        parser.error("replay-capacity must be positive")
    if args.encoder_training and (learned_rgb or args.encoder == "levjepa"
                                 or args.observation_size != "native" or args.batch_length != 16):
        parser.error("encoder-training requires a Tiny checkpoint, native frames and batch-length16")
    if args.encoder_training and not any(arg.split("=", 1)[0] == "--replay-capacity" for arg in sys.argv[1:]):
        parser.error("encoder-training requires explicit replay-capacity; native pixel replay is large")
    if learned_rgb and (args.encoder is not None or args.observation_size != "native"):
        parser.error("learned RGB consumes native frames for one GPU resize; frozen --encoder requires --encoder-checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = ATARI_PROTOCOLS[args.atari_protocol]
    gym.register_envs(ale_py)
    environments = []
    output = args.output.open("x")
    memory_output = None

    def emit(event):
        event["unix_time"] = time.time()
        print(json.dumps(event, separators=(",", ":"), allow_nan=False), file=output, flush=True)

    try:
        if args.min_gpu_budget_headroom_mib is not None:
            memory_output = memory_path.open("x")

        def check_memory(stage, run_step):
            if memory_output is None:
                return
            before = time.perf_counter()
            snapshot = agent.gpu_memory_budget
            elapsed = time.perf_counter() - before
            record = dict(stage=stage, run_step=run_step, learner_step=agent.learner_step,
                          pid=os.getpid(), unix_time=time.time(), query_seconds=elapsed,
                          minimum_headroom_bytes=args.min_gpu_budget_headroom_mib * 1024**2,
                          memory=snapshot)
            if stage == "constructed":
                record.update(protocol="kindle-native-memory-budget-v1", device=agent.gpu_device)
            print(json.dumps(record, separators=(",", ":"), allow_nan=False),
                  file=memory_output, flush=True)
            require_gpu_budget(snapshot, record["minimum_headroom_bytes"])

        env_seeds = [(args.seed + stream * 1_000_003) % 2**32 for stream in range(args.num_envs)]
        initial = []
        for seed in env_seeds:
            env = DreamerAtariPreprocessing(gym.make(
                args.environment, frameskip=1, repeat_action_probability=args.sticky_actions,
                full_action_space=protocol.full_action_space,
            ), noop_max=protocol.noop_max, max_episode_frames=protocol.max_episode_frames,
                screen_size=None if args.observation_size == "native" else 64)
            environments.append(env)
            initial.append(env.reset(seed=seed)[0])
        actions = int(environments[0].action_space.n)
        config = kindle.default_config(actions, args.model_size)
        config.update(seed=args.seed, batch_size=args.batch_size, batch_length=args.batch_length,
                      training_precision=args.training_precision,
                      video_encoder=args.encoder_training, replay_capacity=args.replay_capacity,
                      actor_critic_gradient=args.actor_critic_gradient, actor_unimix=args.actor_unimix,
                      world_backprop_length=args.batch_length,
                      world_microbatch_size=(args.batch_size if args.world_microbatch_size is None else args.world_microbatch_size),
                      train_ratio=args.train_ratio, learning_rate=args.learning_rate,
                      disagreement_bonus=args.disagreement_scale > 0,
                      intrinsic_reward_scale=args.disagreement_scale,
                      learning_rate_warmup=1000, agc=0.3)
        if args.imagination_length is not None:
            config["imagination_length"] = args.imagination_length
        if args.behavior_learning_rate is not None:
            config["behavior_learning_rate"] = args.behavior_learning_rate
        if args.guidance_imitation_scale is not None:
            weights = [1.0] * actions
            noop = environments[0].action_meanings.index("NOOP")
            weights[noop] = args.guidance_noop_weight if args.guidance_noop_weight is not None else 1.0
            config["expert_imitation"] = dict(scale=args.guidance_imitation_scale, action_weights=weights)
            if args.guidance_world_gradient:
                config["expert_imitation"]["world_gradient"] = True
        if learned_rgb:
            config["observation_kind"] = f"rgb{args.image_size or 64}"
            config["loss_scales"].update(reconstruction=1.0, future_prediction=0.0)
            if args.cdp:
                config["loss_scales"].update(reconstruction=0.0, future_prediction=500.0)
                config.update(**cdp_rates, cdp_centered=args.cdp_centered)
        else:
            config["loss_scales"].update(reconstruction=0.0, future_prediction=0.25)
        exploration = (PersistentExploration(dict(kind=EXPLORATION_KIND,
            probability=args.exploration_probability, hold_actions=args.exploration_hold,
            seed=args.seed), args.num_envs, actions) if args.exploration_probability else None)
        if args.qbert_guidance_probability:
            guidance_config = dict(kind=GUIDANCE_KIND,
                probability=args.qbert_guidance_probability, seed=args.seed)
            if args.qbert_guidance_max_actions is not None:
                guidance_config.update(kind=BOUNDED_GUIDANCE_KIND, max_actions=args.qbert_guidance_max_actions)
            if args.qbert_guidance_policy == "continuation":
                guidance_config["kind"] = guidance_module.CONTINUATION_GUIDANCE_KIND
            if args.qbert_corrective_labels:
                guidance_config["kind"] = guidance_module.CORRECTIVE_GUIDANCE_KIND
            exploration = QbertGuidance(guidance_config, args.num_envs, actions)
        if args.breakout_guidance_probability:
            exploration = BreakoutGuidance(dict(kind=(breakout_guidance_module.CORRECTIVE_GUIDANCE_KIND
                if args.breakout_corrective_labels else breakout_guidance_module.GUIDANCE_KIND),
                probability=args.breakout_guidance_probability, seed=args.seed,
                max_actions=args.breakout_guidance_max_actions), args.num_envs, actions)
        guidance = isinstance(exploration, (QbertGuidance, BreakoutGuidance))
        selected_guidance_module = breakout_guidance_module if isinstance(exploration, BreakoutGuidance) else guidance_module
        current_frames = initial
        construction = time.perf_counter()
        restored = checkpoint_identity(args.restore) if args.restore else None
        if learned_rgb:
            agent = (kindle.VectorAgent.restore_rgb(str(args.restore), args.num_envs) if args.restore
                     else kindle.VectorAgent.learned_rgb(args.num_envs, config))
        else:
            agent = (kindle.VectorAgent.restore(str(args.restore), args.encoder_checkpoint, args.num_envs)
                     if args.restore else kindle.VectorAgent(args.encoder_checkpoint, args.num_envs, config,
                         **(dict(encoder=args.encoder) if args.encoder else {})))
        if expected := os.environ.get("KINDLE_EXPECT_DEVICE_NAME"):
            require_gpu_device(agent.gpu_device, expected)
        if agent.config["action_count"] != actions:
            raise ValueError("checkpoint action vocabulary differs from environment")
        construction = time.perf_counter() - construction
        check_memory("constructed", 0)
        ids = list(range(args.num_envs))
        starting_actions, starting_updates = agent.environment_step, agent.learner_step
        started = time.perf_counter()
        agent.begin_episodes(ids, initial)
        check_memory("initialized", 0)
        initial_checkpoint = None
        if args.initial_checkpoint:
            agent.save_checkpoint(str(args.initial_checkpoint))
            initial_checkpoint = checkpoint_identity(args.initial_checkpoint)
        exploration_header = (dict(exploration=exploration.config,
            **(dict(guidance_schedule_sha256=sha256_file(guidance_schedule_module.__file__)) if guidance else {}),
            exploration_sha256=sha256_file(selected_guidance_module.__file__ if guidance else exploration_module.__file__)) if exploration else {})
        vector_protocol = (EPISODE_EVALUATION_PROTOCOL if args.episodes_per_env else
                           EXPLORATION_PROTOCOL if exploration else VECTOR_PROTOCOL)
        emit(dict(event="run_start", protocol=vector_protocol, environment=args.environment,
                  **(dict(initial_checkpoint=initial_checkpoint) if initial_checkpoint else {}),
                  num_envs=args.num_envs, steps=args.steps, seed=args.seed, environment_seeds=env_seeds,
                  **(dict(checkpoint_history=True) if args.checkpoint_history else {}),
                  **(dict(evaluation_episodes_per_stream=args.episodes_per_env) if args.episodes_per_env else {}),
                  **(dict(frozen_checkpoint_export=True) if args.evaluate and args.checkpoint else {}),
                  policy_seed_rule="config.seed + stream (wrapping u64)",
                  atari_protocol=args.atari_protocol, action_repeat=ATARI_ACTION_REPEAT,
                  noop_max=protocol.noop_max, max_episode_frames=protocol.max_episode_frames,
                  full_action_space=protocol.full_action_space, sticky_actions=args.sticky_actions,
                  observation_size=args.observation_size, observation_shape=list(initial[0].shape),
                  **(dict(learned_rgb_preprocessing=f"single GPU Pillow-equivalent antialiased bilinear resize to {agent.config['observation_kind'][3:]}x{agent.config['observation_kind'][3:]}, per-axis RGB8 rounding; CHW /255-0.5") if learned_rgb else {}),
                  action_meanings=list(environments[0].action_meanings), ale_py_version=ale_py.__version__,
                  mode=("evaluate_greedy" if args.greedy else "evaluate_sample") if args.evaluate else "train",
                  **(dict(behavior_probe="live-posterior-v1") if args.probe_behavior else {}),
                  config=agent.config, model_provenance=agent.provenance, gpu_device=agent.gpu_device,
                  cpu_worker_threads=agent.cpu_worker_threads,
                  trainable_parameter_counts=agent.trainable_parameter_counts,
                  restored_checkpoint=restored, starting_environment_step=starting_actions,
                  starting_learner_step=starting_updates, agent_construction_seconds=construction,
                  native_extension_sha256=sha256_file(kindle._native.__file__),
                  runner_sha256=sha256_file(__file__), wrapper_sha256=sha256_file(atari.__file__),
                  **exploration_header))
        stop = False

        def request_stop(_signum, _frame):
            nonlocal stop
            stop = True

        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        episode_returns = [0.0] * args.num_envs
        episode_lengths = [0] * args.num_envs
        episode_counts = [0] * args.num_envs
        natural_episode_counts = [0] * args.num_envs
        total_rewards = [0.0] * args.num_envs
        completed = []
        timing = dict(act=0.0, environment=0.0, observe=0.0, learn=0.0, reset=0.0, checkpoint=0.0)
        last_report = last_checkpoint = 0
        run_actions = 0

        def save():
            before = time.perf_counter()
            checkpoint = args.checkpoint
            if args.checkpoint_history:
                checkpoint = checkpoint / str(run_actions)
                checkpoint.mkdir(parents=True)
            agent.save_checkpoint(str(checkpoint))
            check_memory("checkpoint", run_actions)
            emit(dict(event="checkpoint", run_step=run_actions, learner_step=agent.learner_step,
                      identity=checkpoint_identity(checkpoint)))
            timing["checkpoint"] += time.perf_counter() - before

        def progress(event):
            elapsed = time.perf_counter() - started
            frames = [env.executed_action_frames for env in environments]
            exploration_counts = (dict(overridden_actions=exploration.overridden_actions)
                                  if exploration else {})
            if guidance:
                exploration_counts.update(guidance_episodes=exploration.episode_counts)
            return dict(event=event, run_step=run_actions, vector_ticks=run_actions // args.num_envs,
                        environment_step=agent.environment_step, learner_step=agent.learner_step,
                        elapsed_seconds=elapsed, actions_per_second=run_actions / elapsed,
                        executed_action_frames=frames,
                        reset_noop_frames=[env.reset_noop_frames for env in environments],
                        emulator_resets=[env.emulator_resets for env in environments],
                        aggregate_simulated_wall_ratio=sum(frames) / 60 / elapsed,
                        per_stream_simulated_wall_ratio=[n / 60 / elapsed for n in frames],
                        total_rewards=total_rewards, episode_counts=episode_counts,
                        **(dict(natural_episode_counts=natural_episode_counts) if args.episodes_per_env else {}),
                        partial_returns=episode_returns, partial_lengths=episode_lengths,
                        replay_len=agent.replay_len, training_debt=agent.training_debt,
                        stage_seconds=timing, **exploration_counts)

        try:
            for tick in range(1, args.steps // args.num_envs + 1):
                if stop:
                    break
                before = time.perf_counter()
                overrides = (exploration.actions(current_frames) if guidance else
                             exploration.actions() if exploration else None)
                override_argument = "expert_actions" if args.guidance_imitation_scale is not None else "action_overrides"
                if corrective_labels:
                    selected = agent.act(greedy=args.greedy, action_overrides=overrides,
                                         expert_labels=exploration.expert_labels)
                else:
                    selected = (agent.act(greedy=args.greedy, **{override_argument: overrides})
                                if exploration else agent.act(greedy=args.greedy))
                if overrides is not None and any(forced is not None and selected[stream] != forced
                                                for stream, forced in enumerate(overrides)):
                    raise ValueError("native action override was not honored")
                if corrective_labels:
                    exploration.feedback(selected)
                timing["act"] += time.perf_counter() - before
                check_memory("act", run_actions)
                behavior = {}
                if args.probe_behavior:
                    before = time.perf_counter()
                    values, bonuses = agent.behavior_predictions()
                    behavior = dict(behavior_values=values, behavior_intrinsic=bonuses)
                    timing["diagnostics"] = timing.get("diagnostics", 0.0) + time.perf_counter() - before
                    check_memory("behavior_probe", run_actions)
                before = time.perf_counter()
                results = [env.step(action) for env, action in zip(environments, selected)]
                timing["environment"] += time.perf_counter() - before
                frames, rewards, terminated, truncated, _ = map(list, zip(*results))
                current_frames = frames
                before = time.perf_counter()
                stored_rewards = agent.observe(ids, frames, rewards, terminated, truncated)
                timing["observe"] += time.perf_counter() - before
                check_memory("observe", run_actions + args.num_envs)
                run_actions += args.num_envs
                emit(dict(event="transition", vector_tick=tick, run_step=run_actions,
                          actions=selected, rewards=rewards, stored_rewards=stored_rewards,
                          terminated=terminated, truncated=truncated,
                          executed_action_frames=[env.executed_action_frames for env in environments],
                          **behavior,
                          **(dict(guidance_frame_sha256=exploration.frame_sha256) if guidance else {}),
                          **(dict(action_overrides=overrides) if exploration else {}),
                          **(dict(expert_labels=exploration.expert_labels) if corrective_labels else {}),
                          **(dict(expert_action_mask=[action is not None for action in
                                 (exploration.expert_labels if corrective_labels else overrides)])
                             if args.guidance_imitation_scale is not None else {})))
                before = time.perf_counter()
                reports = [] if args.evaluate else agent.learn_scheduled()
                timing["learn"] += time.perf_counter() - before
                check_memory("learn", run_actions)
                for report in reports:
                    require_numbers(report)
                    emit(dict(event="learner", run_step=run_actions, report=report))
                resets = []
                for stream in ids:
                    episode_returns[stream] += float(rewards[stream])
                    total_rewards[stream] += float(rewards[stream])
                    episode_lengths[stream] += 1
                    if terminated[stream] or truncated[stream]:
                        result = dict(event="episode", stream=stream, run_step=run_actions,
                                      stream_step=tick, episode=episode_counts[stream],
                                      episode_return=episode_returns[stream], episode_length=episode_lengths[stream],
                                      elapsed_seconds=time.perf_counter() - started,
                                      terminated=bool(terminated[stream]), truncated=bool(truncated[stream]))
                        emit(result)
                        completed.append(result)
                        episode_counts[stream] += 1
                        natural_episode_counts[stream] += bool(terminated[stream] and not truncated[stream])
                        episode_returns[stream] = 0.0
                        episode_lengths[stream] = 0
                        resets.append(stream)
                before = time.perf_counter()
                if resets:
                    reset_frames = [environments[stream].reset()[0] for stream in resets]
                    agent.begin_episodes(resets, reset_frames)
                    for stream, frame in zip(resets, reset_frames):
                        current_frames[stream] = frame
                    check_memory("reset", run_actions)
                    if exploration:
                        exploration.reset(resets)
                    emit(dict(event="reset", run_step=run_actions, streams=resets))
                timing["reset"] += time.perf_counter() - before
                if args.checkpoint and run_actions - last_checkpoint >= args.checkpoint_every:
                    save()
                    last_checkpoint = run_actions
                if run_actions - last_report >= args.report_every:
                    event = progress("progress")
                    emit(event)
                    print(f"{run_actions}/{args.steps} actions; {event['actions_per_second']:.2f} actions/s; "
                          f"{agent.learner_step - starting_updates} updates; debt={agent.training_debt:.3f}", flush=True)
                    last_report = run_actions
                if args.episodes_per_env and min(natural_episode_counts) >= args.episodes_per_env:
                    break
            if args.checkpoint and run_actions != last_checkpoint:
                save()
            event = progress("run_end")
            check_memory("finished", run_actions)
            reason = "budget_complete"
            if args.episodes_per_env:
                reason = "episode_budget_complete" if min(natural_episode_counts) >= args.episodes_per_env else "action_cap_reached"
            event.update(reason="interrupted" if stop else reason,
                         **episode_summary(completed),
                         learner_updates=agent.learner_step - starting_updates)
            emit(event)
            print(json.dumps(event, indent=2), flush=True)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    finally:
        output.close()
        if memory_output is not None:
            memory_output.close()
        for env in environments:
            env.close()


if __name__ == "__main__":
    main()
