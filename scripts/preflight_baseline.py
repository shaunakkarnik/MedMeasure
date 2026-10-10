"""Fail before inference if the staged run lacks pinned data or usable local weights."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medmeasure.data import read_plan, rows, sha256, write_json
from medmeasure.baseline import validate_loaded_row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    run = json.loads((args.run/'manifest.json').read_text())
    os.environ.update(run['environment'])
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root/'configs/kits23-pilot.json').read_text())
    if run['annotation_version'] != '1.4.0' or run['task'] != lock['task']:
        raise ValueError('Baseline preflight currently supports only the KiTS23 v1.4.0 pilot')
    dataset_dir = Path(run['data_dir'])/'Datasets/KiTS23'
    plan = dataset_dir/'benchmark_plan_biometry_v1.4.0.json.gz'
    if sha256(plan) != lock['sha256'][plan.name]:
        raise ValueError('Staged annotation plan differs from the pinned plan')
    import medvision_bm
    benchmark_root = Path(medvision_bm.__file__).parent
    for relative, expected_hash in lock['benchmark_files_sha256'].items():
        if sha256(benchmark_root/relative) != expected_hash:
            raise ValueError(f'Installed evaluator differs from pinned source: {relative}')
    import lmms_eval
    active_lmms = Path(lmms_eval.__file__).parent
    prefix = 'medvision_lmms_eval/lmms_eval/'
    for relative, expected_hash in lock['benchmark_files_sha256'].items():
        if relative.startswith(prefix) and sha256(active_lmms/relative[len(prefix):]) != expected_hash:
            raise ValueError('Active lmms_eval differs from the pinned vendored evaluator')
    planner = importlib.import_module('medvision_ds.utils.benchmark_planner')
    if sha256(planner.__file__) != lock['planner_source_sha256']:
        raise ValueError('Installed medvision_ds geometry source is not the pinned revision')
    model = Path(run['model_path'])
    if not (model/'config.json').is_file() or not list(model.glob('*.safetensors')):
        raise ValueError('Expected local checkpoint config and safetensors weights')
    # This sidecar is written during staging after a revision-pinned HF download.
    provenance = json.loads((model/'medmeasure-provenance.json').read_text())
    if provenance['revision'] != run['model_revision_declared']:
        raise ValueError('Checkpoint staging revision differs from run revision')
    records = rows(read_plan(plan), 'test')
    indices = run['sample_indices'] if run['sample_indices'] is not None else list(range(min(run['requested_limit'], len(records))))
    for index in indices:
        row = records[index]
        for field in ['image_file', 'mask_file', 'landmark_file']:
            if not (dataset_dir/row[field]).is_file():
                raise FileNotFoundError(dataset_dir/row[field])
    # Check HF loader ordering and float16 reference conversion against our plan adapter.
    # Offline flags are set by run.sh: missing caches fail rather than updating inputs.
    from datasets import load_dataset
    loader_path = run.get('dataset_loader_path')
    if loader_path:
        if sha256(loader_path) != lock['loader_source_sha256'] or sha256(loader_path) != run['dataset_loader_sha256']:
            raise ValueError('Local dataset loader differs from the frozen run')
    options = {} if loader_path else {'revision':run['dataset_revision']}
    loaded = load_dataset(loader_path or 'YongchengYAO/MedVision',
                          name='KiTS23_TumorLesionSize_Task01_Axial_Test',
                          trust_remote_code=True, split='test', **options)
    if len(loaded) != len(records):
        raise ValueError('HF loader count differs from prepared manifest')
    for index in indices:
        validate_loaded_row(loaded[index], records[index])
    write_json(args.run/'selected_rows.json', [records[i] for i in indices])
    (args.run/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    # Invoked only by the explicit GPU run, never by preparation.
    (args.run/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    write_json(args.run/'preflight.json', {'status':'passed', 'selected_rows':len(indices),
               'planner_sha256':sha256(planner.__file__), 'plan_sha256':sha256(plan)})


if __name__ == '__main__':
    main()
