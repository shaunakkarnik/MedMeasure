"""Stage, validate and package KiTS23 locally for Colab inference (CPU only).

Requires the separate requirements-local-data.txt environment. Writes only to the
specified data/audit/archive paths. Does not download model weights or use a GPU.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medmeasure.data import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/medvision')
    p.add_argument('--output', type=Path, default=ROOT/'runs/local-data-preparation')
    p.add_argument('--archive', type=Path, help='Default: data/exports/kits23-<revision>-portable-v2.tar')
    p.add_argument('--workers', type=int, default=2, help='CPU worker cap; lower if memory is constrained')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    config = json.loads((ROOT/'configs/kits23-pilot.json').read_text())
    archive = args.archive or ROOT/f"data/exports/kits23-{config['dataset_revision']}-portable-v2.tar"
    data, output = args.data_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
               PIP_CONSTRAINT=str(ROOT/'requirements-local-data.txt'))
    timings = {}

    def run(name, *arguments):
        started = time.monotonic()
        print(f'\nStarting {name}', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts'/name), *map(str, arguments)],
                       env=env, cwd=ROOT, check=True)
        timings[name] = round(time.monotonic()-started, 2)
        print(f'{name}: {timings[name]:.1f} seconds', flush=True)

    run('stage_data.py', '--data-dir', data, '--with-images', '--local-cpu', '--workers', args.workers)
    audit = output/'audit'
    with tempfile.TemporaryDirectory() as directory:
        fresh = Path(directory)/'audit'
        run('prepare_data.py', '--plan', data/'metadata/benchmark_plan_biometry_v1.4.0.json.gz',
            '--legacy-plan', data/'metadata/benchmark_plan_biometry_v1.0.0.json.gz',
            '--legacy-plan', data/'metadata/benchmark_plan_biometry_v1.1.1.json.gz', '--output', fresh)
        if audit.exists():
            for path in fresh.iterdir():
                if not (audit/path.name).is_file() or (audit/path.name).read_bytes() != path.read_bytes():
                    raise ValueError('Existing audit differs; preserve it and use a fresh --output directory')
        else:
            import shutil
            shutil.copytree(fresh, audit)
    geometry = output/'geometry-smoke'
    if not geometry.exists():
        run('check_geometry.py', '--manifest', audit/'train.jsonl',
            '--dataset-dir', data/'Datasets/KiTS23', '--output', geometry, '--limit', 20)
    checked = json.loads((geometry/'summary.json').read_text())
    if (checked['failed'] or checked['checked'] != 20
            or checked['manifest_sha256'] != sha256(audit/'train.jsonl')):
        raise ValueError('Local geometry check failed/incomplete; inspect it before packaging')
    (output/'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], env=env, text=True))
    write_json(data/'local-preparation.json', {
        'dataset_revision':config['dataset_revision'], 'workers':args.workers,
        'phase_seconds':timings, 'geometry_smoke':checked,
        'test_manifest_sha256':sha256(audit/'test.jsonl'),
        'package_versions':(output/'pip-freeze.txt').read_text().splitlines(),
        'medmeasure_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
        'stage_script_sha256':sha256(ROOT/'scripts/stage_data.py'),
        'visual_review_required':True,
    })
    run('cache_colab_data.py', 'backup', '--portable', '--data-dir', data,
        '--archive', archive, '--config', ROOT/'configs/kits23-pilot.json')
    print(f'\nReview all overlays in {geometry}. Upload {archive} and {archive}.json '
          'to MyDrive/MedMeasure/cache before starting a Colab GPU runtime.', flush=True)


if __name__ == '__main__':
    main()
