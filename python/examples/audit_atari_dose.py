"""Read complete continuous Freeway/Qbert exposure studies; never launches an agent.

Midpoint and final belong to one training history. Runtime/guard and observer
lifecycles remain separate prerequisites, not certified by this artifact reader.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

import retain_atari_checkpoint as retain
import audit_atari as matches
import audit_atari_tasks as tasks

require, digest, read = retain.require, retain.digest, retain.read
STAGES = (200004, 400008)
EVALUATION_SEED = 100000
RECIPE = dict(action_count=18, model_size='size12_m', batch_size=16, batch_length=64,
    world_backprop_length=64, world_microbatch_size=16, train_ratio=256.0, learning_rate=0.00004,
    extrinsic_reward_scale=1.0, intrinsic_reward_scale=0.0, visitation_bonus=False)
SHARED = ('environment', 'atari_protocol', 'action_repeat', 'full_action_space',
    'noop_max', 'max_episode_frames', 'sticky_actions', 'action_meanings', 'ale_py_version',
    'config', 'model_provenance', 'native_extension_sha256', 'runner_sha256', 'wrapper_sha256',
    'trainable_parameter_counts', 'gpu_device', 'cpu_worker_threads', 'policy_seed_rule')


def verify_pins(pins):
    require(bool(pins), 'missing input pins')
    for path, expected in pins.items():
        require(digest(path) == expected, 'changed input: ' + path)


def read_training(path, expected_header, expected_actions, source):
    run = matches.read_run(path)
    header, end, accounting = run['start'], run['end'], run['accounting']
    require({key: value for key, value in header.items() if key not in retain.RUNTIME_HEADER}
        == {key: value for key, value in expected_header.items() if key not in retain.RUNTIME_HEADER},
        'declared complete-training header changed')
    retain.check_training_header(header, expected_actions)
    require(header['steps'] == accounting['actions'] == end['run_step'] == expected_actions
        and accounting['budget_complete'] and accounting['updates'] > 0
        and end['event'] == 'run_end' and end['reason'] == 'budget_complete', 'complete training budget missing')
    source = Path(source)
    require(header['runner_sha256'] == digest(source / 'atari_vector.py')
        and header['wrapper_sha256'] == digest(source / 'atari.py'), 'changed source-matched runner')
    return run


def check_frozen_protocol(training, evaluation, stage):
    start, header, accounting = training['start'], evaluation['start'], evaluation['accounting']
    game = start['environment']
    require(game in ('ALE/Qbert-v5', 'ALE/Freeway-v5'), 'unsupported exposure study')
    require(stage['event'] == 'checkpoint' and 0 < stage['run_step'] <= training['accounting']['actions'],
        'invalid checkpoint stage')
    require(header['mode'] == 'evaluate_sample' and 'exploration' not in header
        and 'exploration_sha256' not in header and header['num_envs'] == 6
        and accounting['updates'] == 0 and accounting['budget_complete'],
        'stage evaluation must be sampled, unassisted and frozen')
    if game == 'ALE/Qbert-v5':
        require(header['protocol'] == matches.EPISODE_EVALUATION_PROTOCOL
            and header['steps'] == 600000 and header['evaluation_episodes_per_stream'] == 4
            and accounting['episode_budget_complete'] and accounting['evaluation_episodes_per_stream'] == 4
            and 0 < accounting['actions'] <= 600000 and accounting['actions'] % 6 == 0
            and evaluation['end']['reason'] == 'episode_budget_complete'
            and len(evaluation['end']['episode_counts']) == 6
            and min(evaluation['end']['episode_counts']) >= 4, 'incomplete or changed frozen episode budget')
    else:
        require(header['protocol'] == retain.VECTOR_PROTOCOL
            and 'evaluation_episodes_per_stream' not in header and 'episode_budget_complete' not in accounting
            and header['steps'] == accounting['actions'] == evaluation['end']['run_step'] == 75000
            and evaluation['end']['reason'] == 'budget_complete', 'incomplete or changed frozen action budget')
    require(header['seed'] == EVALUATION_SEED
        and header['environment_seeds'] == [EVALUATION_SEED + stream * 1000003 for stream in range(6)],
        'changed evaluation seeds')
    require(header['starting_environment_step'] == stage['run_step']
        and header['starting_learner_step'] == stage['learner_step'], 'wrong restored checkpoint-stage counters')
    for key in SHARED:
        require(header[key] == start[key], 'stage evaluation identity differs: ' + key)
    require(header['unix_time'] > training['end']['unix_time'], 'frozen evaluation precedes complete training')


def check_saved_stage(training, stage, evaluation, checkpoint, schema, encoder):
    check_frozen_protocol(training, evaluation, stage)
    checkpoint = Path(checkpoint)
    restored = evaluation['start']['restored_checkpoint']
    require(restored is not None and Path(restored['path']).resolve() == checkpoint.resolve(),
        'undeclared restored checkpoint path')
    require(all(restored[key] == stage['identity'][key] for key in ('metadata_sha256', 'tensor_sha256')),
        'restored checkpoint is not the declared stage')
    state = retain.verify_state(checkpoint, stage['identity'], training['start'], stage, schema, encoder)
    return dict(path=str(checkpoint), identity=stage['identity'], finite_complete_state=state,
                checkpoint_stage_actions=stage['run_step'], checkpoint_stage_updates=stage['learner_step'])


def check_retained_stage(training, directory, action):
    directory = Path(directory)
    record = read(directory / 'retained.json')
    require(record['protocol'] == 'kindle-atari-settled-checkpoint-retention-v1'
        and record['complete_training'] is False and record['learner_called'] is False
        and record['source_modified'] is False and record['prefix_expected_error'] == 'missing run_end',
        'wrong retained checkpoint evidence')
    verify_pins(record['pins'])
    prefix = retain.checkpoint_event(training['path'], action)
    require(Path(record['source_log']).resolve() == Path(training['path']).resolve()
        and record['source_header'] == training['start'] == prefix['header']
        and record['source_checkpoint_event'] == prefix['event']
        and record['identity'] == prefix['event']['identity']
        and record['source_prefix_bytes'] == prefix['bytes']
        and record['source_prefix_sha256'] == prefix['sha256'], 'retained stage does not belong to complete training')
    archive = directory / 'training-prefix.jsonl'
    require(digest(archive) == prefix['sha256'], 'retained prefix bytes differ')
    retain.checked_prefix(archive)
    checkpoint = directory / 'checkpoint'
    require(Path(record['archived_checkpoint']).resolve() == checkpoint.resolve(), 'retained checkpoint escaped its archive')
    retain.verify_files(checkpoint, prefix['event']['identity'])
    return record, prefix['event'], checkpoint


def score_replayed(evaluation, replay_path):
    replay = read(replay_path)
    tasks.check_replay(evaluation, replay)
    video = replay['video']
    require(video is not None and video['stream'] == 0 and video['fps'] == 60
        and video['frames'] == evaluation['end']['executed_action_frames'][0]
        and digest(video['path']) == video['sha256'], 'missing or changed whole-stream video')
    return dict(score=tasks.score_tasks(evaluation['start']['environment'], replay['episodes']),
        replay=dict(path=str(replay_path), sha256=digest(replay_path), video=video, partial=replay['partial']),
        evaluation={key: evaluation[key] for key in ('path', 'sha256', 'accounting', 'end')})


def audit_stage(training, directory, evaluation_path, replay_path, *, action, schema, encoder):
    record, event, checkpoint = check_retained_stage(training, directory, action)
    evaluation = matches.read_run(evaluation_path)
    saved = check_saved_stage(training, event, evaluation, checkpoint, schema, encoder)
    result = score_replayed(evaluation, replay_path)
    result.update(checkpoint=saved, retained_marker_sha256=digest(Path(directory) / 'retained.json'),
        retained_prefix_remains_incomplete=True, full_training_sha256=training['sha256'])
    require(saved['finite_complete_state'] == record['complete_finite_state'], 'retained state summary differs')
    return result


def audit_untrained(training, initial_path, evaluation_path, replay_path, checkpoint, schema, encoder):
    initial, evaluation = matches.read_run(initial_path), matches.read_run(evaluation_path)
    header = initial['start']
    require(header['protocol'] == retain.VECTOR_PROTOCOL and header['mode'] == 'evaluate_sample'
        and header['num_envs'] == 6 and header['steps'] == initial['accounting']['actions'] == 6
        and initial['accounting']['updates'] == 0 and initial['accounting']['budget_complete']
        and header['seed'] == training['start']['seed'] == header['config']['seed']
        and header['starting_environment_step'] == header['starting_learner_step'] == 0
        and header['restored_checkpoint'] is None and 'exploration' not in header,
        'untrained control must be a separately saved same-seed fresh model')
    for key in SHARED:
        require(header[key] == training['start'][key], 'untrained control identity differs: ' + key)
    require(header['environment_seeds'] == [header['seed'] + stream * 1000003 for stream in range(6)]
        and header['unix_time'] > training['end']['unix_time'], 'changed untrained initial environments or ordering')
    event = initial['checkpoint']
    require(event is not None and event['run_step'] == 6 and event['learner_step'] == 0, 'missing zero-update control save')
    # The frozen protocol is identical, but restoration is bound to this fresh
    # control's actual event, never to the trained checkpoint or a synthetic end.
    saved = check_saved_stage(initial, event, evaluation, checkpoint, schema, encoder)
    for name in retain.PARTS:
        with safe_open(Path(checkpoint) / f'{name}.safetensors', framework='numpy') as state:
            require(all(np.all(state.get_tensor(key) == 0) for key in state.keys()
                if key.startswith(('adam_m.', 'adam_v.'))), 'untrained optimizer moments are nonzero')
    result = score_replayed(evaluation, replay_path)
    result.update(checkpoint=saved, initial={key: initial[key] for key in ('path', 'sha256', 'accounting', 'end')},
                  all_untrained_optimizer_moments_zero=True)
    return result


def summarize(stages, control):
    require([row['checkpoint']['checkpoint_stage_actions'] for row in stages] == list(STAGES),
        'both declared dose checkpoints are required in order')
    require(stages[0]['checkpoint']['checkpoint_stage_updates'] < stages[1]['checkpoint']['checkpoint_stage_updates'],
        'no additional learner updates between stages')
    require(stages[0]['full_training_sha256'] == stages[1]['full_training_sha256'], 'stages came from different training histories')
    passed = (stages[1]['score']['task_gate_passed'] and not control['score']['task_gate_passed']
        and control['score']['mean_completed_return'] is not None
        and stages[1]['score']['mean_completed_return'] > control['score']['mean_completed_return'])
    return dict(data_complete=True, final_paired_gate_passed=passed,
        final_minus_midpoint_mean=(None if any(row['score']['mean_completed_return'] is None for row in stages)
            else stages[1]['score']['mean_completed_return'] - stages[0]['score']['mean_completed_return']),
        independent_training_roots=1, reliability_assessed=False, five_game_goal_complete=False,
        caveat='One continuous training history; midpoint and final are not independent seeds.')


def verify_declaration(declaration):
    require(declaration['protocol'] == 'kindle-atari-continuous-dose-v1'
        and declaration['training_actions'] == STAGES[-1]
        and [row['actions'] for row in declaration['stages']] == list(STAGES), 'wrong declared dose budget or stages')
    expected = declaration['training_header']
    require(expected['steps'] == STAGES[-1], 'a 200k pilot cannot satisfy the 400k dose study')
    retain.check_training_header(expected, STAGES[-1])
    require(declaration['criteria'] == tasks.TASK_CRITERIA[expected['environment']], 'changed competence gate')
    schema, source = Path(declaration['schema']), Path(declaration['source'])
    reference = read(schema / 'metadata.json')
    require(expected['config'] == dict(reference['config'], seed=expected['seed']), 'changed qualified learning recipe')
    require(all(expected['config'].get(key) == value for key, value in RECIPE.items())
        and expected['config']['loss_scales']['reconstruction'] == 0
        and expected['config']['loss_scales']['future_prediction'] == 0.25, 'changed fixed dose recipe')
    for name in ('dreamerv3_revision', 'meganeura_revision', 'blade_revision', 'future_head_revision', 'perception'):
        require(expected['model_provenance'][name] == reference[name], 'changed qualified provenance: ' + name)
    required = [Path(__file__), Path(retain.__file__), Path(matches.__file__), Path(tasks.__file__),
        source / 'atari_vector.py', source / 'atari.py', Path(declaration['native_extension']),
        Path(declaration['encoder']), *(schema / name for name in retain.FILES)]
    require(all(str(path.resolve()) in declaration['pins'] for path in required), 'missing declared source/state pins')
    verify_pins(declaration['pins'])
    require(expected['native_extension_sha256'] == digest(declaration['native_extension'])
        and expected['runner_sha256'] == digest(source / 'atari_vector.py')
        and expected['wrapper_sha256'] == digest(source / 'atari.py'), 'declared runtime differs')


def audit_dose(declaration):
    verify_declaration(declaration)
    training = read_training(declaration['training'], declaration['training_header'], STAGES[-1], declaration['source'])
    stages = [audit_stage(training, stage['retained'], stage['evaluation'], stage['replay'], action=stage['actions'],
        schema=declaration['schema'], encoder=declaration['encoder']) for stage in declaration['stages']]
    control = declaration['untrained']
    untrained = audit_untrained(training, control['initial'], control['evaluation'], control['replay'],
                               control['checkpoint'], declaration['schema'], declaration['encoder'])
    verify_pins(declaration['pins'])
    return dict(protocol='kindle-atari-continuous-dose-audit-v1', **summarize(stages, untrained),
        training={key: training[key] for key in ('path', 'sha256', 'accounting', 'end')},
        stages=stages, untrained=untrained, gpu_runtime_evidence_checked=False,
        command_lifecycle_checked=False, midpoint_observer_checked=False, new_gpu_work_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), 'output must be fresh')
    result = audit_dose(read(args.declaration))
    retain.write(args.output, result)
    print(json.dumps(dict(output=str(args.output), data_complete=result['data_complete'],
        final_paired_gate_passed=result['final_paired_gate_passed'], reliability_assessed=False)), flush=True)


if __name__ == '__main__':
    main()
