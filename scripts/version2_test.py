"""Shard official V2 test inference and merge only complete, compatible results.

This entry point imports the original V2 runtime without changing the files
hashed into training checkpoints. Each shard uses the full audited manifest.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medmeasure.data import sha256, write_json
from scripts.summarize_baseline import summarize


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    commands = command.add_subparsers(dest='command', required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--data-root', type=Path, required=True)
    evaluate.add_argument('--manifests', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--checkpoint', type=Path)
    evaluate.add_argument('--config', type=Path, default=ROOT / 'configs/version2-kits23.json')
    evaluate.add_argument('--shard-index', type=int, required=True)
    evaluate.add_argument('--shard-count', type=int, default=4)
    evaluate.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    evaluate.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    evaluate.add_argument('--local-files-only', action='store_true')
    merge = commands.add_parser('merge')
    merge.add_argument('--manifests', type=Path, required=True)
    merge.add_argument('--run', type=Path, required=True, help='Directory containing shard-0, shard-1, ...')
    return command


def source_hashes():
    """Match every runtime file recorded by the original V2 checkpoint."""
    names = ('scripts/version2.py', 'scripts/version1.py', 'scripts/summarize_baseline.py',
             'medmeasure/version2.py', 'medmeasure/version1.py', 'medmeasure/version1_data.py',
             'configs/version2-prompts.json')
    return {name: sha256(ROOT / name) for name in names}


def progress_rows(rows):
    """Report completed generation every 25 rows without changing inference."""
    for count, row in enumerate(rows, 1):
        yield row
        if count % 25 == 0 or count == len(rows):
            print(f'Completed {count}/{len(rows)} rows', flush=True)


def checked_predictions(path, expected_rows):
    """Check identities in order and recompute scores from the saved answers."""
    from medmeasure.version2 import score_answer

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [record['sample_id'] for record in records] != [row['sample_id'] for row in expected_rows]:
        raise ValueError(f'Missing, duplicated, reordered, or unexpected predictions in {path}')
    rescored = []
    for row, record in zip(expected_rows, records):
        expected = score_answer(row, record['answer'])
        if record != expected:
            raise ValueError(f'Saved metrics or references differ for {row["sample_id"]}')
        rescored.append(expected)
    return rescored


def evaluate(args):
    import torch
    from medmeasure.data import read_plan, rows
    from medmeasure.version1_data import read_manifest, validate_manifests
    from medmeasure.version2 import add_vision_adapters
    from scripts.version1 import load_model
    from scripts.version2 import evaluate as generate

    pin = json.loads((ROOT / 'configs/kits23-pilot.json').read_text())
    manifests = {name: read_manifest(args.manifests / f'{name}.jsonl')
                 for name in ('train', 'validation', 'test')}
    plan = args.data_root / 'metadata/benchmark_plan_biometry_v1.4.0.json.gz'
    validate_manifests(manifests, plan, pin)
    if manifests['test'] != rows(read_plan(plan), 'test'):
        raise ValueError('Test manifest must match the full ordered official test selection')
    if not 1 <= args.shard_count <= len(manifests['test']):
        raise ValueError('Shard count must be between one and the test row count')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Shard index is outside the requested shard count')

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    config = json.loads(args.config.read_text())
    hashes = source_hashes()
    manifest_hashes = {name: sha256(args.manifests / f'{name}.jsonl') for name in manifests}
    if checkpoint:
        contract = checkpoint['contract']
        expected = {'format_version': 1, 'pin': pin, 'config': config,
                    'source_sha256': hashes, 'manifest_sha256': manifest_hashes,
                    'dtype': args.dtype, 'torch': str(torch.__version__)}
        for key, value in expected.items():
            if contract[key] != value:
                raise ValueError(f'Test runtime differs from training checkpoint in {key}')
        if contract['train_limit']:
            raise ValueError('Use a full-training checkpoint for official test evaluation')

    metadata = {'format_version': 1, 'split': 'test', 'pin': pin, 'config': config,
                'source_sha256': hashes, 'manifest_sha256': manifest_hashes,
                'evaluation_script_sha256': sha256(Path(__file__)),
                'torch': str(torch.__version__), 'dtype': args.dtype, 'device': args.device,
                'checkpoint_sha256': sha256(args.checkpoint) if checkpoint else None,
                'objective': checkpoint['contract']['objective'] if checkpoint else 'frozen',
                'epoch': checkpoint['epoch'] if checkpoint else None,
                'shard_count': args.shard_count}
    selected = manifests['test'][args.shard_index::args.shard_count]
    receipt = {**metadata, 'shard_index': args.shard_index,
               'sample_ids': [row['sample_id'] for row in selected]}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / 'run.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError('Existing shard output belongs to a different evaluation')
    if (args.output / 'summary.json').exists():
        records = checked_predictions(args.output / 'predictions.jsonl', selected)
        if json.loads((args.output / 'summary.json').read_text()) != summarize(records):
            raise ValueError('Existing shard summary differs from its predictions')
        print('Complete shard already verified; skipping.', flush=True)
        return

    write_json(receipt_path, receipt)
    torch.manual_seed(config['seed'])
    processor, features, decoder = load_model(config, pin, args)
    backbone = features.backbone
    del features, decoder
    if checkpoint:
        add_vision_adapters(backbone, config['first_adapter_block'], config['adapter_rank'])
        adapters = {name: param for name, param in backbone.named_parameters() if param.requires_grad}
        if adapters.keys() != checkpoint['adapters'].keys():
            raise ValueError('Checkpoint adapter names differ from model')
        with torch.no_grad():
            for name, param in adapters.items():
                param.copy_(checkpoint['adapters'][name])
    temporary = args.output / 'predictions.partial.jsonl'
    print(f'Shard {args.shard_index + 1}/{args.shard_count}: {len(selected)} rows', flush=True)
    # Atomic promotion leaves only completed predictions eligible for aggregation.
    summary = generate(backbone, processor, progress_rows(selected), args.data_root / 'Datasets/KiTS23',
                       config, args.device, temporary)
    checked_predictions(temporary, selected)
    temporary.replace(args.output / 'predictions.jsonl')
    write_json(args.output / 'summary.json', summary)
    print(summary, flush=True)


def merge(args):
    from medmeasure.version1_data import read_manifest

    rows = read_manifest(args.manifests / 'test.jsonl')
    first = json.loads((args.run / 'shard-0/run.json').read_text())
    if first['manifest_sha256']['test'] != sha256(args.manifests / 'test.jsonl'):
        raise ValueError('Test manifest differs from the inference manifest')
    common = {key: value for key, value in first.items() if key not in ('shard_index', 'sample_ids')}
    predictions = {}
    for index in range(first['shard_count']):
        directory = args.run / f'shard-{index}'
        receipt = json.loads((directory / 'run.json').read_text())
        if {key: value for key, value in receipt.items()
                if key not in ('shard_index', 'sample_ids')} != common:
            raise ValueError('Shard model, runtime, configuration, or manifests differ')
        selected = rows[index::first['shard_count']]
        if receipt['shard_index'] != index or receipt['sample_ids'] != [r['sample_id'] for r in selected]:
            raise ValueError('Shard membership differs from the expected partition')
        records = checked_predictions(directory / 'predictions.jsonl', selected)
        if json.loads((directory / 'summary.json').read_text()) != summarize(records):
            raise ValueError('Shard completion summary differs from recomputed scores')
        for record in records:
            if record['sample_id'] in predictions:
                raise ValueError('Duplicate sample across shards')
            predictions[record['sample_id']] = record
    if set(predictions) != {row['sample_id'] for row in rows}:
        raise ValueError('Incomplete official test coverage')
    ordered = [predictions[row['sample_id']] for row in rows]
    temporary = args.run / 'predictions.partial.jsonl'
    temporary.write_text(''.join(json.dumps(record, allow_nan=False) + '\n' for record in ordered))
    temporary.replace(args.run / 'predictions.jsonl')
    summary = {'complete_official_test': True, 'groups': {'all': summarize(ordered)}}
    diagnostic = read_manifest(args.manifests / 'test_legacy_disjoint.jsonl')
    official = {row['sample_id']: row for row in rows}
    if any(official.get(row['sample_id']) != row for row in diagnostic):
        raise ValueError('Diagnostic includes rows outside the official test selection')
    summary['groups']['legacy_disjoint'] = summarize(
        [predictions[row['sample_id']] for row in diagnostic])
    write_json(args.run / 'run.json', common)
    write_json(args.run / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'evaluate':
        evaluate(args)
    else:
        merge(args)
