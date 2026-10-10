"""Rebuild portable KiTS23 test metadata at its destination, without network/staging.

Install medvision_ds from the restored src directory with --no-deps first. This
uses the unchanged pinned local HF loader, verifies every test identity/reference,
and refuses any attempt to fetch data or code rather than repeating preprocessing.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medmeasure.data import read_plan, rows, sha256, write_json
from medmeasure.baseline import validate_loaded_row
from scripts.cache_colab_data import check_staging


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    args = p.parse_args()
    data = args.data_dir.resolve()
    config = json.loads((ROOT/'configs/kits23-pilot.json').read_text())
    check_staging(data, config)
    loader = data/'MedVision.py'
    if sha256(loader) != config['loader_source_sha256']:
        raise ValueError('Local loader differs from the pinned dataset revision')
    tracker = json.loads((data/'.downloaded_datasets.json').read_text())
    if not tracker.get('dataset_KiTS23'):
        raise ValueError('Processed dataset completion marker is missing')
    if any(tracker.get(name) != '1.4.0' for name in ['medvision_ds', 'medvision_ds_installed']):
        raise ValueError('Bundle lacks pinned package completion markers; rebuild it locally')
    os.environ.update(MedVision_DATA_DIR=str(data), MedVision_PLANNER_VERSION='1.4.0',
                      MedVision_ACK_RELEASE='1.4.0', MedVision_FORCE_DOWNLOAD_DATA='False',
                      MedVision_FORCE_INSTALL_CODE='False', MedVision_DOWNLOAD_QC_FIGURES='False',
                      MedVision_DISABLE_SAMPLE_FILTERING='false',
                      HF_HOME=str(data/'.cache/huggingface'),
                      HF_DATASETS_CACHE=str(data/'.cache/huggingface/datasets'),
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    planner = importlib.import_module('medvision_ds.utils.benchmark_planner')
    if sha256(planner.__file__) != config['planner_source_sha256']:
        raise ValueError('Install the restored pinned medvision_ds source before rebuilding metadata')
    package_root = Path(planner.__file__).parents[1]
    for relative, expected in config['preprocessing_files_sha256'].items():
        if sha256(package_root/relative) != expected:
            raise ValueError(f'Installed preprocessing source mismatch: {relative}')

    def no_download(*args, **kwargs):
        raise RuntimeError('Inference setup attempted a download. Restore a complete portable bundle; do not rerun preprocessing on GPU.')

    import huggingface_hub
    huggingface_hub.snapshot_download = no_download
    huggingface_hub.hf_hub_download = no_download
    import requests
    requests.sessions.Session.request = no_download
    from datasets import load_dataset
    started = time.monotonic()
    loaded = load_dataset(str(loader), name='KiTS23_TumorLesionSize_Task01_Axial_Test',
                          trust_remote_code=True, split='test')
    plan = data/'Datasets/KiTS23/benchmark_plan_biometry_v1.4.0.json.gz'
    if sha256(plan) != config['sha256'][plan.name]:
        raise ValueError('Restored test plan checksum mismatch')
    records = rows(read_plan(plan), 'test')
    if len(loaded) != len(records):
        raise ValueError('Rebuilt loader count differs from pinned test manifest')
    for index, expected in enumerate(records):
        actual = loaded[index]
        validate_loaded_row(actual, expected)
        for field in ['image_file', 'mask_file', 'landmark_file']:
            path = data/'Datasets/KiTS23'/expected[field]
            if not path.is_file():
                raise FileNotFoundError(path)
            if field in actual and Path(actual[field]).resolve() != path.resolve():
                raise ValueError(f'Loader retained an incorrect absolute {field} path')
    write_json(data/'inference-data-ready.json', {
        'test_rows':len(records), 'data_dir':str(data), 'loader_sha256':sha256(loader),
        'dataset_revision':config['dataset_revision'], 'plan_sha256':sha256(plan),
        'metadata_rebuild_seconds':round(time.monotonic()-started, 2),
        'network_downloads_allowed':False,
    })
    print(f'Checked {len(records)} test rows and destination paths. No scans downloaded or reoriented.', flush=True)


if __name__ == '__main__':
    main()
