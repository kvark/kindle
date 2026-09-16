"""Retain a settled Atari checkpoint without pausing or restarting its learner."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from kindle._vector_audit import VECTOR_PROTOCOL, audit
from kindle._exploration import EXPLORATION_KIND, EXPLORATION_PROTOCOL

FILES = ('metadata.json', 'world.safetensors', 'behavior.safetensors', 'slow_value.safetensors')
PARTS = ('world', 'behavior', 'slow_value')
COUNTS = dict(world=164, behavior=66, slow_value=11)
MUTABLE_METADATA = {'config', 'learner_step', 'environment_step', 'return_low', 'return_high', 'tensor_sha256'}
RUNTIME_HEADER = {'unix_time', 'agent_construction_seconds'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checkpoint_event(log, action):
    require(type(action) is int and action > 0 and action % 6 == 0, 'invalid checkpoint action')
    hasher, size = hashlib.sha256(), 0
    with Path(log).open('rb') as stream:
        header = None
        for line in stream:
            require(line.endswith(b'\n'), 'unfinished source row before checkpoint')
            row = json.loads(line)
            if header is None:
                require(row['event'] == 'run_start', 'missing source header')
                header = row
            require(row.get('run_step', 0) <= action, 'declared checkpoint event is missing')
            hasher.update(line)
            size += len(line)
            if row['event'] == 'checkpoint' and row['run_step'] == action:
                return dict(header=header, event=row, bytes=size, sha256=hasher.hexdigest())
    raise ValueError('declared checkpoint event is not available')


def prefix_digest(path, size):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while size:
            block = stream.read(min(size, 1024 * 1024))
            require(block, 'source prefix shortened')
            hasher.update(block)
            size -= len(block)
    return hasher.hexdigest()


def verify_files(directory, identity):
    directory = Path(directory)
    require(set(identity['tensor_sha256']) == set(PARTS), 'incomplete tensor-file identity')
    expected = dict(zip(FILES, [identity['metadata_sha256'],
        *(identity['tensor_sha256'][name] for name in PARTS)]))
    require(all((directory / name).is_file() and not (directory / name).is_symlink()
        and digest(directory / name) == value for name, value in expected.items()), 'checkpoint file identity differs')


def verify_state(directory, identity, header, event, schema, encoder):
    directory, schema = Path(directory), Path(schema)
    verify_files(directory, identity)
    metadata, reference = read(directory / 'metadata.json'), read(schema / 'metadata.json')
    require(metadata['format'] == reference['format'] == 3
        and metadata['architecture'] == 'dreamerv3-visual-features', 'unsupported checkpoint format')
    require({k: v for k, v in metadata.items() if k not in MUTABLE_METADATA}
        == {k: v for k, v in reference.items() if k not in MUTABLE_METADATA}, 'checkpoint architecture or frontend changed')
    require(metadata['environment_step'] == event['run_step']
        and metadata['learner_step'] == event['learner_step']
        and metadata['collection_streams'] == header['num_envs']
        and metadata['config'] == header['config'] and metadata['tensor_sha256'] == identity['tensor_sha256'],
        'checkpoint counters or recipe changed')
    require(metadata['perception'] == header['model_provenance']['perception']
        and metadata['perception']['kind'] == 'levjepa'
        and digest(encoder) == metadata['perception']['checkpoint_sha256'], 'actual encoder identity changed')
    for key in ('dreamerv3_revision', 'meganeura_revision', 'blade_revision', 'future_head_revision'):
        require(metadata[key] == header['model_provenance'][key], 'checkpoint provenance changed: ' + key)
    require(all(math.isfinite(metadata[key]) for key in ('return_low', 'return_high'))
        and metadata['return_low'] <= metadata['return_high'], 'invalid return normalizer')
    counts, parameter_count, moment_count = {}, 0, 0
    for name in PARTS:
        require(digest(schema / f'{name}.safetensors') == reference['tensor_sha256'][name], 'reference tensor file changed')
        with safe_open(directory / f'{name}.safetensors', framework='numpy') as actual, \
                safe_open(schema / f'{name}.safetensors', framework='numpy') as expected:
            current, template = actual.metadata(), expected.metadata()
            require(current['meganeura_checkpoint_format'] == template['meganeura_checkpoint_format'] == '3',
                'native checkpoint format changed')
            layout = json.loads(current['meganeura_logical_layout'])
            require(layout == json.loads(template['meganeura_logical_layout']), 'logical tensor layout changed')
            require(int(current['adam_step']) == (0 if name == 'slow_value' else event['learner_step']),
                'optimizer counter differs')
            parameters, adam = set(layout['parameters']), set(layout['adam_parameters'])
            require(adam <= parameters, 'optimizer layout contains unknown parameters')
            complete = parameters | {f'{kind}.{key}' for kind in ('adam_m', 'adam_v') for key in adam}
            require(set(actual.keys()) == set(expected.keys()) == complete
                and len(complete) == COUNTS[name], 'incomplete tensor set')
            for key in actual.keys():
                value, shape_reference = actual.get_tensor(key), expected.get_tensor(key)
                require(value.shape == shape_reference.shape and value.dtype == shape_reference.dtype
                    and np.isfinite(value).all(), 'tensor shape, dtype or finite-value check failed')
                if key.startswith('adam_v.'):
                    require((value >= 0).all(), 'negative optimizer second moment')
            counts[name] = len(complete)
            parameter_count += len(parameters)
            moment_count += 2 * len(adam)
    require(parameter_count == 95 and moment_count == 146, 'incomplete parameters or optimizer moments')
    return dict(tensors=counts, parameters=parameter_count, optimizer_moments=moment_count,
                environment_step=metadata['environment_step'], learner_step=metadata['learner_step'])


def checked_prefix(path):
    try:
        audit(path)
    except ValueError as error:
        require(str(error) == 'missing run_end', 'prefix audit failed: ' + str(error))
    else:
        raise ValueError('retained checkpoint prefix unexpectedly completed training')


def copy_exclusive(source, destination, limit=None):
    with Path(source).open('rb') as src, Path(destination).open('xb') as dst:
        while limit is None or limit:
            block = src.read(1024 * 1024 if limit is None else min(limit, 1024 * 1024))
            if not block:
                require(limit in (None, 0), 'copy source shortened')
                break
            dst.write(block)
            if limit is not None:
                limit -= len(block)


def check_training_header(header, action):
    require(header['environment'] in ('ALE/Qbert-v5', 'ALE/Freeway-v5')
        and header['mode'] == 'train' and header['num_envs'] == 6
        and header['restored_checkpoint'] is None
        and header['starting_environment_step'] == header['starting_learner_step'] == 0
        and header['steps'] >= action, 'not fresh vector Qbert/Freeway training')
    if header['environment'] == 'ALE/Qbert-v5':
        require(header['protocol'] == VECTOR_PROTOCOL
            and not {'exploration', 'exploration_sha256'}.intersection(header), 'Qbert must remain unassisted')
    else:
        require(header['protocol'] == EXPLORATION_PROTOCOL and header.get('exploration') == dict(
            kind=EXPLORATION_KIND, probability=0.5, hold_actions=64, seed=header['seed']),
            'Freeway must retain its declared probability .5/hold64 exploration')


def retain(log, checkpoint, destination, *, action, expected_header, schema, encoder):
    log, checkpoint, destination, schema, encoder = map(Path, (log, checkpoint, destination, schema, encoder))
    require(not destination.exists() and not destination.is_symlink() and destination.parent.is_dir(),
        'archive destination must be fresh')
    require(not destination.resolve().is_relative_to(checkpoint.resolve())
        and not checkpoint.resolve().is_relative_to(destination.resolve()), 'archive overlaps source checkpoint')
    prefix = checkpoint_event(log, action)
    header, event = prefix['header'], prefix['event']
    require({k: v for k, v in header.items() if k not in RUNTIME_HEADER}
        == {k: v for k, v in expected_header.items() if k not in RUNTIME_HEADER}, 'declared source header changed')
    check_training_header(header, action)
    identity = event['identity']
    source_identity = log.stat()
    state = verify_state(checkpoint, identity, header, event, schema, encoder)
    reference_pins = {str(path): digest(path) for path in [schema / name for name in FILES] + [encoder]}
    destination.mkdir()
    saved = destination / 'checkpoint'
    saved.mkdir()
    archive = destination / 'training-prefix.jsonl'
    copy_exclusive(log, archive, prefix['bytes'])
    require(digest(archive) == prefix['sha256'], 'source prefix changed during copy')
    checked_prefix(archive)
    for name in FILES:
        copy_exclusive(checkpoint / name, saved / name)
    require(verify_state(saved, identity, header, event, schema, encoder) == state, 'archived state differs')
    verify_files(checkpoint, identity)
    require((log.stat().st_dev, log.stat().st_ino) == (source_identity.st_dev, source_identity.st_ino)
        and prefix_digest(log, prefix['bytes']) == prefix['sha256'], 'original source prefix changed')
    require(all(digest(path) == value for path, value in reference_pins.items()), 'reference changed during retention')
    pins = dict(reference_pins)
    pins.update({str(path): digest(path) for path in [Path(__file__), archive, *(saved / name for name in FILES)]})
    result = dict(protocol='kindle-atari-settled-checkpoint-retention-v1', source_log=str(log),
        source_checkpoint=str(checkpoint), source_header=header, source_checkpoint_event=event,
        source_prefix_bytes=prefix['bytes'], source_prefix_sha256=prefix['sha256'],
        archived_checkpoint=str(saved), complete_finite_state=state, identity=identity, pins=pins,
        prefix_expected_error='missing run_end', complete_training=False, checkpoint_restored=False,
        learner_called=False, source_modified=False, atomic_recovery_claimed=False,
        caveat='Completion marker requires every check. A failure leaves an unqualified partial archive; no automatic retry.')
    write(destination / 'retained.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('log', 'checkpoint', 'destination', 'header', 'schema', 'encoder'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--action', required=True, type=int)
    args = parser.parse_args(argv)
    result = retain(args.log, args.checkpoint, args.destination, action=args.action,
                    expected_header=read(args.header), schema=args.schema, encoder=args.encoder)
    print(json.dumps(dict(archive=str(args.destination), state=result['complete_finite_state'],
                          learner_called=False, complete_training=False)), flush=True)


if __name__ == '__main__':
    main()
