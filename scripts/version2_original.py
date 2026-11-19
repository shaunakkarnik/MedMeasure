"""Run pinned benchmark tasks with an explicit shared image/question order."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medmeasure.data import sha256, write_json
from scripts.colab_baseline import sample_log
from scripts.summarize_baseline import checked_samples, key, read_rows, summarize


def script(name, *args):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *map(str, args)], check=True)


def reference_samples(root):
    result = {}
    for path in sorted(root.glob('chunks/shard-*/results/*samples*.jsonl')):
        for record in read_rows(path):
            if record['doc_id'] in result:
                raise ValueError('Duplicate reference baseline sample')
            result[record['doc_id']] = record
    if not result:
        raise ValueError('Reference baseline sample logs are missing')
    return result


def check_protocol(samples, selected, reference):
    """Require identical task prompts, labels and row identities to the old run."""
    checked_samples(samples, selected)
    for record in samples:
        original = reference[record['doc_id']]
        if (key(record['doc']) != key(original['doc']) or
                record['input'] != original['input'] or record['target'] != original['target']):
            raise ValueError(f'Original benchmark prompt/reference mismatch at {record["doc_id"]}')


def evaluate(args):
    records = read_rows(args.audit / 'test.jsonl')
    if not 0 <= args.shard_index < args.shard_count <= len(records):
        raise ValueError('Invalid shard index/count')
    selected = records[:10] if args.smoke else records[args.shard_index::args.shard_count]
    reference = reference_samples(args.reference_results)
    checked_samples(list(reference.values()), records)

    provenance_path = args.model / 'medmeasure-provenance.json'
    provenance = json.loads(provenance_path.read_text())
    if provenance.get('kind') == 'vision_lora_merged':
        for name, expected in provenance['files_sha256'].items():
            if sha256(args.model / name) != expected:
                raise ValueError(f'Exported model file changed: {name}')
    elif provenance.get('kind', 'released') != 'released':
        raise ValueError('Unknown model provenance')
    receipt = {'model_provenance': provenance, 'model_path': str(args.model.resolve()),
               'model_provenance_sha256': sha256(provenance_path),
               'manifest_sha256': sha256(args.audit / 'test.jsonl'),
               'shard_count': args.shard_count, 'shard_index': args.shard_index,
               'smoke': args.smoke, 'sample_ids': [row['sample_id'] for row in selected],
               'runner_sha256': sha256(Path(__file__)),
               'content_order': args.content_order,
               'execution_script_sha256': sha256(ROOT / 'scripts/execute_baseline.py'),
               'input_adapter_sha256': (sha256(ROOT / 'scripts/image_first_benchmark.py')
                                         if args.content_order == 'image_first' else None)}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / 'run.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError('Existing shard has different model or sample provenance')
    write_json(receipt_path, receipt)
    marker = args.output / 'completed.json'
    if marker.exists():
        completed = json.loads(marker.read_text())
        log = sample_log(args.output / completed['attempt'])
        if sha256(log) != completed['samples_sha256']:
            raise ValueError('Completed sample log changed')
        check_protocol(read_rows(log), selected, reference)
        print('Compatible complete shard verified.', flush=True)
        return
    number = 1
    while (args.output / f'attempt-{number:03d}').exists():
        number += 1
    attempt = args.output / f'attempt-{number:03d}'
    indices = args.output / 'indices.json'
    write_json(indices, [row['loader_index'] for row in selected])
    script('prepare_baseline.py', '--upstream', args.upstream, '--data-dir', args.data_root,
           '--model-path', args.model, '--output', attempt, '--sample-indices', indices)
    # Retain explicit derivation in addition to the base revision expected by preflight.
    manifest_path = attempt / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['model_provenance'] = provenance
    manifest['content_order'] = args.content_order
    write_json(manifest_path, manifest)
    script('preflight_baseline.py', '--run', attempt)
    script('execute_baseline.py', '--run', attempt)
    log = sample_log(attempt)
    samples = read_rows(log)
    check_protocol(samples, selected, reference)
    write_json(attempt / 'summary.json', summarize(samples))
    write_json(marker, {'attempt': attempt.name, 'samples_sha256': sha256(log),
                        'protocol_check': 'Exact prompts and targets match saved original baseline'})
    print(json.dumps(summarize(samples), indent=2), flush=True)


def merge(args):
    records = read_rows(args.audit / 'test.jsonl')
    reference = reference_samples(args.reference_results)
    checked_samples(list(reference.values()), records)
    first = json.loads((args.output / 'shard-0/run.json').read_text())
    if first['smoke'] or first['manifest_sha256'] != sha256(args.audit / 'test.jsonl'):
        raise ValueError('Expected full test shards with the current manifest')
    common = {key: value for key, value in first.items() if key not in ('shard_index', 'sample_ids')}
    logs = []
    samples = []
    for index in range(first['shard_count']):
        directory = args.output / f'shard-{index}'
        receipt = json.loads((directory / 'run.json').read_text())
        if {key: value for key, value in receipt.items()
                if key not in ('shard_index', 'sample_ids')} != common:
            raise ValueError('Shard model or evaluation provenance differs')
        selected = records[index::first['shard_count']]
        if receipt['shard_index'] != index or receipt['sample_ids'] != [r['sample_id'] for r in selected]:
            raise ValueError('Incorrect shard membership')
        completion = json.loads((directory / 'completed.json').read_text())
        log = sample_log(directory / completion['attempt'])
        if sha256(log) != completion['samples_sha256']:
            raise ValueError('Completed sample log changed')
        shard_samples = read_rows(log)
        check_protocol(shard_samples, selected, reference)
        logs.append(log)
        samples.extend(shard_samples)
    check_protocol(samples, records, reference)
    by_id = {sample['doc_id']: sample for sample in samples}
    ordered = [by_id[row['loader_index']] for row in records]
    diagnostic = read_rows(args.audit / 'test_legacy_disjoint.jsonl')
    summary = {'complete_official_test': True, 'protocol': 'pinned upstream benchmark',
               'content_order': first.get('content_order', 'text_first'),
               'groups': {'all': summarize(ordered), 'legacy_disjoint': summarize(
                   [by_id[row['loader_index']] for row in diagnostic])}}
    temporary = args.output / 'samples.partial.jsonl'
    temporary.write_text(''.join(json.dumps(row) + '\n' for row in ordered))
    temporary.replace(args.output / 'samples.jsonl')
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'provenance.json', {**common, 'sample_logs_sha256':
               {str(path.relative_to(args.output)): sha256(path) for path in logs}})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('evaluate', 'merge'):
        command = commands.add_parser(name)
        command.add_argument('--audit', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--reference-results', type=Path,
                             default=ROOT / 'results/kits23-medvision-v0-v140-rik')
        if name == 'evaluate':
            command.add_argument('--model', type=Path, required=True)
            command.add_argument('--upstream', type=Path, required=True)
            command.add_argument('--data-root', type=Path, required=True)
            command.add_argument('--shard-index', type=int, default=0)
            command.add_argument('--shard-count', type=int, default=4)
            command.add_argument('--smoke', action='store_true')
            command.add_argument('--content-order', choices=('text_first', 'image_first'),
                                 default='text_first')
    args = parser.parse_args()
    if args.command == 'evaluate':
        evaluate(args)
    else:
        merge(args)
