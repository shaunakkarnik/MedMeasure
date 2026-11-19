"""Restore source/markers from the existing archive and rebuild benchmark metadata."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medmeasure.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    data = work / 'data/medvision'
    pin = json.loads((ROOT / 'configs/kits23-pilot.json').read_text())
    # Recover only missing source and staging records. Scans are already restored.
    archive = work / f'kits23-{pin["dataset_revision"]}-v140.tar'
    required = ('src', 'staging.json', '.downloaded_datasets.json')
    if any(not (data / name).exists() for name in required):
        sidecar = json.loads(Path(str(archive) + '.json').read_text())
        if archive.stat().st_size != sidecar['archive_bytes'] or sha256(archive) != sidecar['archive_sha256']:
            raise ValueError('Archive checksum differs from its sidecar')
        with tarfile.open(archive) as tar:
            for member in tar:
                relative = Path(member.name)
                if relative.parts and relative.parts[0] in required:
                    destination = data / relative
                    if not destination.exists():
                        tar.extract(member, data, filter='data')
    from huggingface_hub import snapshot_download

    # The old backup lacks the root loader. Fetch pinned source files only.
    snapshot_download(pin['dataset_repo'], repo_type='dataset', revision=pin['dataset_revision'],
                      local_dir=data, allow_patterns=['MedVision.py', 'src/*'], max_workers=1)
    if sha256(data / 'MedVision.py') != pin['loader_source_sha256']:
        raise ValueError('Dataset loader differs from the pinned source')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(data / 'src')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_inference_data.py'),
                    '--data-dir', str(data)], check=True)

    # A separate view of the immutable cache keeps staging records out of HF snapshots.
    os.environ['HF_HOME'] = str(work / 'hf-cache')
    base = Path(snapshot_download(pin['model_repo'], revision=pin['model_revision'],
                                  cache_dir=work / 'hf-cache/hub', local_files_only=True))
    frozen = work / 'models/original-frozen'
    frozen.mkdir(parents=True, exist_ok=True)
    for source in base.iterdir():
        if source.is_file() and not source.name.startswith('.') and source.name != 'medmeasure-provenance.json':
            target = frozen / source.name
            if target.exists():
                if target.resolve() != source.resolve():
                    raise ValueError(f'Frozen model view points to a different file: {target}')
            else:
                target.symlink_to(source.resolve())
    write_json(frozen / 'medmeasure-provenance.json',
               {'kind': 'released', 'repo': pin['model_repo'], 'revision': pin['model_revision']})
    print('Original benchmark data and frozen checkpoint ready.', flush=True)


if __name__ == '__main__':
    main()
