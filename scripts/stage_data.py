"""Explicit network staging for KiTS23; never runs GPU inference.

Default: download pinned annotation archive and source only. --with-images invokes
MedVision's CPU download/preprocessing loader and may transfer the full CT dataset.
Run on an approved transfer/CPU node, not a cluster login node without permission.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medmeasure.data import DATASET_REVISION, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--with-images', action='store_true')
    p.add_argument('--workers', type=int, default=1, help='Upstream CPU preprocessing worker limit')
    p.add_argument('--local-cpu', action='store_true', help='Install pinned dataset source without unrelated dataset dependencies; use requirements-local-data.txt')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    started = time.monotonic()
    root = args.data_dir.resolve()
    os.environ.update(MedVision_DATA_DIR=str(root), MedVision_PLANNER_VERSION='1.4.0',
                      MedVision_ACK_RELEASE='1.4.0', MedVision_DISABLE_SAMPLE_FILTERING='false',
                      HF_HOME=str(root/'.cache/huggingface'),
                      HF_DATASETS_CACHE=str(root/'.cache/huggingface/datasets'))
    if os.environ.get('HF_HUB_OFFLINE') == '1':
        p.error('Staging needs network access; unset HF_HUB_OFFLINE/HF_DATASETS_OFFLINE')
    import huggingface_hub as hub
    # The upstream loader makes nested snapshot calls without a revision. Constrain
    # those calls in this process too, so pinning the top-level script is sufficient.
    original_snapshot = hub.snapshot_download
    def pinned_snapshot(*positional, **kwargs):
        repo = kwargs.get('repo_id', positional[0] if positional else None)
        if repo == 'YongchengYAO/MedVision':
            kwargs['revision'] = DATASET_REVISION
        return original_snapshot(*positional, **kwargs)
    hub.snapshot_download = pinned_snapshot
    pinned_snapshot(repo_id='YongchengYAO/MedVision', repo_type='dataset',
                    allow_patterns=['src/*', 'MedVision.py', 'Datasets/KiTS23.zip'], local_dir=root)
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/kits23-pilot.json').read_text())
    if sha256(root/'MedVision.py') != lock['loader_source_sha256']:
        raise ValueError('Dataset loader source checksum mismatch')
    for relative, expected in lock['preprocessing_files_sha256'].items():
        if sha256(root/'src/medvision_ds'/relative) != expected:
            raise ValueError(f'Preprocessing source checksum mismatch: {relative}')
    archive = root/'Datasets/KiTS23.zip'
    if sha256(archive) != lock['annotation_archive_sha256']:
        raise ValueError('Annotation archive checksum mismatch')
    # Metadata-only preparation extracts just the plan files, leaving the full loader
    # responsible for landmarks, images, masks and its completion marker.
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            basename = Path(name).name
            if basename in lock['sha256']:
                target = root/'metadata'/basename
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(zipped.read(name))
                if sha256(target) != lock['sha256'][basename]:
                    raise ValueError(f'Plan checksum mismatch: {basename}')
    if args.with_images:
        if args.local_cpu:
            # All KiTS23 imports are in requirements-local-data.txt. The dataset
            # package also declares dependencies for unrelated datasets; avoid
            # downloading those and changing the CPU environment mid-process.
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(root/'src')], check=True)
            importlib.invalidate_caches()
            import medvision_ds
            if medvision_ds.__version__ != '1.4.0':
                raise ValueError('Installed dataset source has an unexpected release version')
            tracker_path = root/'.downloaded_datasets.json'
            tracker = json.loads(tracker_path.read_text()) if tracker_path.exists() else {}
            tracker.update(medvision_ds='1.4.0', medvision_ds_installed='1.4.0')
            write_json(tracker_path, tracker)
        os.environ['MedVision_FORCE_INSTALL_CODE'] = 'False' if args.local_cpu else 'True'
        from datasets import load_dataset
        for split in ['train', 'test']:
            split_started = time.monotonic()
            load_dataset('YongchengYAO/MedVision', revision=DATASET_REVISION,
                         name=f'KiTS23_TumorLesionSize_Task01_Axial_{split.title()}',
                         split=split, trust_remote_code=True, num_proc=args.workers)
            print(f'{split} loader staging: {time.monotonic()-split_started:.1f} seconds', flush=True)
            # The first split installed the pinned package. Avoid resolving and
            # reinstalling its dependencies a second time for the test cache.
            os.environ['MedVision_FORCE_INSTALL_CODE'] = 'False'
        import medvision_ds.utils.benchmark_planner as planner
        if sha256(planner.__file__) != lock['planner_source_sha256']:
            raise ValueError('Installed planner source differs from pinned source')
    write_json(root/'staging.json', {'revision':DATASET_REVISION,
               'with_images':args.with_images, 'annotation_archive_sha256':lock['annotation_archive_sha256'],
               'workers':args.workers, 'elapsed_seconds':round(time.monotonic()-started, 2)})
    print('Staging complete. No model inference was run.')


if __name__ == '__main__':
    main()
