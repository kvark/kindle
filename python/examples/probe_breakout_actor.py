"""Reference actor probabilities on saved features; frozen native qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from probe_breakout_actions import evaluate
from probe_breakout_representation import advance, sha, validate_corpus, validate_capture_config, write

PROTOCOL = 'kindle-breakout-actor-reference-v1'
EPSILON = 1e-4
PROBABILITY_ATOL = 1e-4


def actor_probabilities(features, parameters, unimix):
    """F64 reference for the three Linear/RMSNorm/SiLU actor layers."""
    value = np.asarray(features, dtype=np.float64)
    keys = {f'behavior.actor.layer{i}.{suffix}' for i in range(3)
            for suffix in ('weight', 'bias', 'norm.weight')}
    keys.update({'behavior.actor.out.weight', 'behavior.actor.out.bias'})
    if (value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all()
            or set(parameters) != keys or not np.isfinite(unimix) or not 0 <= unimix < 1):
        raise ValueError('invalid actor features, parameters or unimix')
    weights = {key: np.asarray(v, dtype=np.float64) for key, v in parameters.items()}
    if any(not np.isfinite(v).all() for v in weights.values()):
        raise ValueError('nonfinite actor parameter')
    for index in range(3):
        prefix = f'behavior.actor.layer{index}.'
        weight, bias, norm = [weights[prefix+key] for key in ('weight', 'bias', 'norm.weight')]
        if (weight.ndim != 2 or weight.shape[0] != value.shape[1] or not weight.shape[1]
                or bias.shape != (weight.shape[1],) or norm.shape != bias.shape):
            raise ValueError('invalid hidden actor shape')
        value = value @ weight + bias
        value = value / np.sqrt(np.square(value).mean(axis=1, keepdims=True) + EPSILON) * norm
        value = value * np.exp(-np.logaddexp(0., -value))
    weight, bias = weights['behavior.actor.out.weight'], weights['behavior.actor.out.bias']
    if (weight.ndim != 2 or weight.shape[0] != value.shape[1] or not weight.shape[1]
            or bias.shape != (weight.shape[1],)):
        raise ValueError('invalid actor output shape')
    logits = value @ weight + bias
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    probability = (1.-unimix) * probability + unimix / probability.shape[1]
    if not np.isfinite(probability).all():
        raise ValueError('nonfinite actor result')
    return probability


def load_actor(checkpoint):
    from safetensors import safe_open
    with safe_open(checkpoint/'behavior.safetensors', framework='np') as source:
        return {key: source.get_tensor(key) for key in source.keys() if key.startswith('behavior.actor.')}


def policy_metrics(probabilities, teacher, lanes, phases):
    p, teacher = np.asarray(probabilities), np.asarray(teacher)
    if (p.shape != (len(teacher), 18) or not np.isfinite(p).all() or (p <= 0).any()
            or not np.allclose(p.sum(1), 1., atol=1e-10, rtol=0)):
        raise ValueError('invalid full-action probabilities')
    prediction = p.argmax(1)
    result = evaluate(teacher, prediction, lanes, phases)
    masks = dict(all=np.ones(len(teacher), bool), near_descending=phases == 'near_descending')
    masks.update({f'lane{lane}': lanes == lane for lane in np.unique(lanes)})
    for name, mask in masks.items():
        target, chosen, values = teacher[mask], prediction[mask], p[mask]
        assigned = values[np.arange(len(target)), target]
        directional = np.isin(target, [3, 4])
        opposite = ((target == 3) & (chosen == 4)) | ((target == 4) & (chosen == 3))
        result[name].update(mean_teacher_probability=float(assigned.mean()) if len(target) else None,
            teacher_cross_entropy=float(-np.log(assigned).mean()) if len(target) else None,
            entropy=float(-(values*np.log(values)).sum(1).mean()) if len(target) else None,
            directional_teacher_count=int(directional.sum()), opposite_direction_count=int(opposite.sum()),
            directional_noop_count=int((directional & (chosen == 0)).sum()),
            fire_teacher_count=int((target == 1).sum()),
            fire_noop_count=int(((target == 1) & (chosen == 0)).sum()),
            non_teacher_action_count=int((~np.isin(chosen, [0, 1, 3, 4])).sum()))
    return result


def check_native(corpus_path, checkpoint, manifest, prior_smoke, output, *, compare_centroid100=False):
    import ale_py
    import gymnasium as gym
    import kindle
    from atari import DreamerAtariPreprocessing, checkpoint_identity
    from audit_atari import read_run
    from check_atari_adapter import rom_identity
    from fit_atari_probes import checked_memory
    from probe_fixed_latents import assert_frozen_tensors
    from probe_policy_forecasts import validate_checkpoint_config
    from replay_atari import verify_replay_identity

    corpus = json.loads(corpus_path.read_text())
    validate_corpus(corpus)
    source = read_run(Path(corpus['source_log']))
    header = source['start']
    assert source['sha256'] == corpus['source_log_sha256'] and header == corpus['source_header']
    assert source['end'] == corpus['source_end'] and not source['accounting']['updates']
    config = json.loads((checkpoint/'metadata.json').read_text())['config']
    validate_capture_config(header['config'], config, compare_centroid100=compare_centroid100)
    assert config['actor_unimix'] == .01 and config['action_count'] == 18
    gym.register_envs(ale_py)
    verify_replay_identity(header, json.loads(manifest.read_text()), rom_identity(header['environment']))
    old = json.loads((prior_smoke/'result.json').read_text())
    assert old['status'] == 'complete' and old['smoke'] and old['diagnostic_actions'] == 128
    assert old['corpus_sha256'] == sha(corpus_path) and old['checkpoint'] == checkpoint_identity(checkpoint)
    assert len(old['files']) == 1
    previous = old['files'][0]
    assert sha(prior_smoke/previous['file']) == previous['sha256']
    with np.load(prior_smoke/previous['file'], allow_pickle=False) as data:
        previous_features = {k: v.copy() for k, v in data.items()}
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    agent = kindle.Agent.restore_rgb(str(checkpoint))
    validate_checkpoint_config(config, agent.config)
    assert sha(kindle._native.__file__) == old['native_sha256']
    updates, actions = agent.learner_step, agent.environment_step
    checked_memory(agent)
    environment = DreamerAtariPreprocessing(gym.make(header['environment'], frameskip=1,
        repeat_action_probability=header['sticky_actions'], full_action_space=header['full_action_space']),
        noop_max=0, max_episode_frames=header['max_episode_frames'], screen_size=None)
    features, probabilities = [], []
    pixels = hashlib.sha256()
    frame, _ = environment.reset(seed=header['environment_seeds'][0])
    agent.begin_episode(frame)
    records = corpus['streams'][0][:128]
    try:
        for index, row in enumerate(records):
            digest = hashlib.sha256(frame.tobytes(order='C')).hexdigest()
            assert digest == row['input_pixel_sha256']
            pixels.update(bytes.fromhex(digest))
            before = agent.latent_feature, agent.environment_step, agent.learner_step
            probability = agent.posterior_action_probabilities()
            assert before == (agent.latent_feature, agent.environment_step, agent.learner_step)
            assert probability == agent.posterior_action_probabilities()
            if index in previous_features['indices']:
                position = int(np.flatnonzero(previous_features['indices'] == index)[0])
                assert np.array_equal(previous_features['posterior'][position], np.asarray(before[0],np.float32))
                assert np.array_equal(previous_features['cnn'][position], np.asarray(agent.encoded_observation,np.float32))
            features.append(before[0]); probabilities.append(probability)
            frame, _ = advance(agent, environment, row, last=index == 127, action_count=18)
        assert environment.executed_action_frames == previous['frames'] == 512
        assert environment.emulator_resets == previous['resets']
        assert pixels.hexdigest() == previous['input_hash_sequence_sha256']
    finally:
        environment.close()
    assert agent.learner_step == updates and agent.environment_step-actions == 128
    native = np.asarray(probabilities, np.float64)
    reference = actor_probabilities(features, load_actor(checkpoint), config['actor_unimix'])
    np.savez_compressed(output/'probabilities.npz', features=np.asarray(features,np.float32), native=native, reference=reference)
    difference = float(np.abs(native-reference).max())
    assert difference <= PROBABILITY_ATOL, difference
    assert np.array_equal(native.argmax(1), reference.argmax(1))
    agent.save_checkpoint(str(output/'frozen-after'))
    counts = assert_frozen_tensors(checkpoint, output/'frozen-after')
    assert sum(counts.values()) == (372 if compare_centroid100 else 357)
    checked_memory(agent)
    result = dict(status='complete',protocol=PROTOCOL,seconds=time.monotonic()-started,
        source_sha256=sha(__file__),native_sha256=sha(kindle._native.__file__),gpu_device=agent.gpu_device,
        corpus_sha256=sha(corpus_path),checkpoint=checkpoint_identity(checkpoint),
        diagnostic_actions=128,frames=512,actor_updates=0,readback_checks=128,
        previous_feature_rows=len(previous_features['indices']),unchanged_tensors=counts,
        probability_atol=PROBABILITY_ATOL,max_absolute_probability_error=difference,
        greedy_agreement=128,probabilities_sha256=sha(output/'probabilities.npz'))
    write(output/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','checkpoint','manifest','prior_smoke','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--compare-centroid100',action='store_true')
    args=parser.parse_args()
    check_native(args.corpus,args.checkpoint,args.manifest,args.prior_smoke,args.output,
                 compare_centroid100=args.compare_centroid100)
