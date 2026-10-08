"""Run restartable Colab evaluation chunks using the unchanged upstream evaluator.

Completed chunks are reused only after strict identity/coverage checks. Interrupted
chunks remain as evidence and are retried in a fresh attempt directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medmeasure.data import sha256, write_json
from scripts.summarize_baseline import checked_samples, read_rows


def run_script(name, *args):
    subprocess.run([sys.executable, str(ROOT/'scripts'/name), *map(str, args)], check=True)


def sample_log(attempt):
    paths = list((attempt/'results').rglob('*samples*.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'Expected one task sample log in {attempt}, found {len(paths)}')
    return paths[0]


def completed_log(chunk, indices, expected):
    marker = chunk/'completed.json'
    if not marker.exists():
        return None
    saved = json.loads(marker.read_text())
    attempt = chunk/saved['attempt']
    manifest = json.loads((attempt/'manifest.json').read_text())
    if manifest['sample_indices'] != indices:
        raise ValueError('Completed chunk indices differ from this evaluation plan')
    path = sample_log(attempt)
    if sha256(path) != saved['samples_sha256']:
        raise ValueError('Completed sample log changed')
    checked_samples(read_rows(path), expected)
    return path


def evaluate_chunk(args, name, indices, records):
    chunk = args.output/'chunks'/name
    chunk.mkdir(parents=True, exist_ok=True)
    by_index = {r['loader_index']: r for r in records}
    selected = [by_index[i] for i in indices]
    existing = completed_log(chunk, indices, selected)
    if existing:
        print(f'Reusing {name}: {len(indices)} verified rows', flush=True)
        return existing
    number = 1
    while (chunk/f'attempt-{number:03d}').exists():
        number += 1
    attempt = chunk/f'attempt-{number:03d}'
    index_file = chunk/'indices.json'
    write_json(index_file, indices)
    run_script('prepare_baseline.py', '--upstream', args.upstream, '--data-dir', args.data_dir,
               '--model-path', args.model_path, '--output', attempt, '--sample-indices', index_file)
    # Always use the isolated environment's Python, including upstream subprocesses.
    run_script('preflight_baseline.py', '--run', attempt)
    run_script('execute_baseline.py', '--run', attempt)
    path = sample_log(attempt)
    checked_samples(read_rows(path), selected)
    run_script('summarize_baseline.py', '--samples', path, '--selected-rows',
               attempt/'selected_rows.json', '--output', attempt/'summary.json')
    write_json(chunk/'completed.json', {'attempt':attempt.name, 'samples_sha256':sha256(path)})
    print(f'Completed {name}: {len(indices)} rows', flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['smoke', 'run', 'score'])
    for name in ['upstream', 'data-dir', 'model-path', 'audit', 'shards', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-chunks', type=int, default=0, help='0 runs all pending chunks')
    args = p.parse_args()
    if args.max_chunks < 0:
        p.error('--max-chunks cannot be negative')
    records = read_rows(args.audit/'test.jsonl')
    plan = json.loads((args.shards/'plan.json').read_text())
    if plan['manifest_sha256'] != sha256(args.audit/'test.jsonl'):
        raise ValueError('Shard plan does not match the evaluation manifest')
    shard_paths = sorted(args.shards.glob('shard-*.json'))
    flat = [i for path in shard_paths for i in json.loads(path.read_text())]
    if len(shard_paths) != plan['shards'] or len(flat) != len(set(flat)) or set(flat) != {r['loader_index'] for r in records}:
        raise ValueError('Shards must cover every test row exactly once')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'medmeasure_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
                  'pilot':json.loads((ROOT/'configs/kits23-pilot.json').read_text()),
                  'test_manifest_sha256':sha256(args.audit/'test.jsonl'), 'shard_plan':plan}
    saved = args.output/'evaluation.json'
    if saved.exists() and json.loads(saved.read_text()) != provenance:
        raise ValueError('Run provenance changed; use a new RUN_NAME or restore the original code/configuration')
    if not saved.exists():
        write_json(saved, provenance)
    if args.action == 'smoke':
        evaluate_chunk(args, 'smoke', [r['loader_index'] for r in records[:10]], records)
        return
    if args.action == 'run':
        smoke_indices = [r['loader_index'] for r in records[:10]]
        if not completed_log(args.output/'chunks/smoke', smoke_indices, records[:10]):
            raise ValueError('Complete the ten-row smoke run before running evaluation chunks')
    logs = []
    newly_run = 0
    for path in shard_paths:
        indices = json.loads(path.read_text())
        selected = [r for r in records if r['loader_index'] in indices]
        chunk = args.output/'chunks'/path.stem
        existing = completed_log(chunk, indices, selected)
        if existing:
            logs.append(existing)
        elif args.action == 'run':
            if args.max_chunks and newly_run >= args.max_chunks:
                break
            logs.append(evaluate_chunk(args, path.stem, indices, records))
            newly_run += 1
        else:
            raise ValueError(f'{path.stem} is incomplete; finish all chunks before scoring')
    print(f'{len(logs)}/{len(shard_paths)} chunks verified', flush=True)
    if args.action != 'score':
        return
    for name in ['test', 'test_legacy_disjoint']:
        output = args.output/f'{name}-summary.json'
        if output.exists():
            raise FileExistsError(f'{output} already exists; retain it or remove it explicitly before rescoring')
        command = ['--selected-rows', args.audit/f'{name}.jsonl', '--output', output]
        if name != 'test':
            command += ['--subset-of', args.audit/'test.jsonl']
        for path in logs:
            command += ['--samples', path]
        run_script('summarize_baseline.py', *command)


if __name__ == '__main__':
    main()
