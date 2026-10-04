"""Explicit network staging for KiTS23; never runs GPU inference.

Default: download pinned annotation archive and source only. --with-images invokes
MedVision's CPU download/preprocessing loader and may transfer the full CT dataset.
Run on an approved transfer/CPU node, not a cluster login node without permission.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medmeasure.data import DATASET_REVISION, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--with-images', action='store_true')
    args = p.parse_args()
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
                    allow_patterns=['src/*', 'Datasets/KiTS23.zip'], local_dir=root)
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/kits23-pilot.json').read_text())
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
        os.environ['MedVision_FORCE_INSTALL_CODE'] = 'True'
        from datasets import load_dataset
        for split in ['train', 'test']:
            load_dataset('YongchengYAO/MedVision', revision=DATASET_REVISION,
                         name=f'KiTS23_TumorLesionSize_Task01_Axial_{split.title()}',
                         split=split, trust_remote_code=True)
        import medvision_ds.utils.benchmark_planner as planner
        if sha256(planner.__file__) != lock['planner_source_sha256']:
            raise ValueError('Installed planner source differs from pinned source')
    write_json(root/'staging.json', {'revision':DATASET_REVISION,
               'with_images':args.with_images, 'annotation_archive_sha256':lock['annotation_archive_sha256']})
    print('Staging complete. No model inference was run.')


if __name__ == '__main__':
    main()
