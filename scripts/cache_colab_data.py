"""Back up staged Colab data to a checksummed tar archive and restore it locally.

Run only after staging finishes, without another process writing the data directory.
The archive includes processed scans, upstream completion markers, pinned source,
metadata and HF loader caches. Model weights are stored separately and excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


EXCLUDED_NAMES = {'token', 'stored_tokens', '.DS_Store'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_staging(data, config):
    staged = json.loads((data/'staging.json').read_text())
    if (staged.get('revision') != config['dataset_revision']
            or staged.get('with_images') is not True
            or staged.get('annotation_archive_sha256') != config['annotation_archive_sha256']):
        raise ValueError('Complete revision-pinned stage_data.py --with-images before backing up')
    for relative, expected in config.get('preprocessing_files_sha256', {}).items():
        if digest(data/'src/medvision_ds'/relative) != expected:
            raise ValueError(f'Preprocessing source checksum mismatch: {relative}')
    for name, expected in config['sha256'].items():
        if name.endswith(('v1.0.0.json.gz', 'v1.1.1.json.gz', 'v1.4.0.json.gz')):
            if digest(data/'metadata'/name) != expected:
                raise ValueError(f'Metadata checksum mismatch: {name}')
    for name in ['Images', 'Masks']:
        if not any((data/'Datasets/KiTS23'/name).glob('*.nii.gz')):
            raise ValueError(f'No processed {name} found')


def included(path, data, portable=False):
    parts = path.relative_to(data).parts
    if any(part in EXCLUDED_NAMES or part.endswith('.lock') for part in parts):
        return False
    if not portable or not parts:
        return True
    # Arrow rows contain absolute paths. Ship only data/source, then rebuild the
    # small metadata cache at the destination using the unchanged local loader.
    if any(part in {'__pycache__', 'build'} or part.endswith('.egg-info') for part in parts):
        return False
    return (parts[0] in {'metadata', 'src', 'MedVision.py', 'staging.json',
                         '.downloaded_datasets.json', 'local-preparation.json'}
            or parts == ('Datasets',)
            or parts[:2] == ('Datasets', 'KiTS23'))


def inspect_data(data, config, portable=False):
    check_staging(data, config)
    if portable:
        if digest(data/'MedVision.py') != config['loader_source_sha256']:
            raise ValueError('Portable bundle requires the pinned local MedVision.py loader')
        tracker = json.loads((data/'.downloaded_datasets.json').read_text())
        if not tracker.get('dataset_KiTS23'):
            raise ValueError('Upstream processed-dataset completion marker is missing')
    files = [p for p in data.rglob('*') if p.is_file() and included(p, data, portable)]
    size = sum(p.stat().st_size for p in files)
    print(f'Archive input: {len(files):,} files, {size/2**30:.2f} GiB before tar overhead.', flush=True)
    print('Check that Google Drive has sufficient free quota before backup. '
          'Already-compressed scans use an uncompressed tar to avoid expensive recompression.', flush=True)
    return size


def cache_metadata(archive, config, data):
    data, archive = data.resolve(), archive.resolve()
    sidecar = archive.with_name(archive.name+'.json')
    if not archive.is_file() or not sidecar.is_file():
        raise ValueError('Cache is incomplete: both the archive and its completed JSON sidecar are required')
    metadata = json.loads(sidecar.read_text())
    if (metadata['format'] not in {1, 2} or metadata['dataset_revision'] != config['dataset_revision']
            or metadata['annotation_archive_sha256'] != config['annotation_archive_sha256']):
        raise ValueError('Persistent cache does not match the pinned dataset configuration')
    # HF Arrow caches can contain absolute paths. Restore to the same Colab path.
    if metadata['format'] == 1 and metadata['data_dir'] != str(data):
        raise ValueError(f'Restore to the original data directory: {metadata["data_dir"]}')
    if archive.stat().st_size != metadata['archive_bytes'] or digest(archive) != metadata['archive_sha256']:
        raise ValueError('Persistent archive checksum mismatch; do not reuse this cache')
    return metadata


class HashingWriter:
    """Compute the archive checksum while writing, avoiding a second Drive read."""
    def __init__(self, stream):
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.size = 0

    def write(self, block):
        written = self.stream.write(block)
        self.hasher.update(block[:written])
        self.size += written
        return written


def backup(data, archive, config, portable=False):
    data, archive = data.resolve(), archive.resolve()
    source_bytes = inspect_data(data, config, portable)
    if archive.exists() or archive.with_name(archive.name+'.json').exists():
        existing = cache_metadata(archive, config, data)
        if existing['format'] != (2 if portable else 1):
            raise ValueError('Archive format differs; use a distinct name for a portable bundle')
        print('Existing complete archive verified; leaving it unchanged.', flush=True)
        return
    partial = archive.with_name(archive.name+'.partial')
    if partial.exists():
        raise FileExistsError(f'Preserve or remove interrupted backup {partial} before retrying')
    archive.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    def select(member):
        nonlocal count
        relative = Path(member.name)
        if not included(data/relative, data, portable):
            return None
        if member.issym() and not (data/relative).resolve().is_relative_to(data):
            raise ValueError(f'Symlink outside the data directory: {member.name}')
        if portable and member.issym() and Path(member.linkname).is_absolute():
            raise ValueError(f'Portable bundle cannot contain absolute symlinks: {member.name}')
        count += 1
        if count % 1000 == 0:
            print(f'Archived {count:,} entries...', flush=True)
        return member

    # Write directly to Drive: no second full-size archive on runtime-local disk.
    with partial.open('xb') as stream:
        writer = HashingWriter(stream)
        with tarfile.open(fileobj=writer, mode='w|', dereference=False) as tar:
            tar.add(data, arcname='.', filter=select)
    partial.rename(archive)
    metadata = {'format':2 if portable else 1, 'dataset_revision':config['dataset_revision'],
                'annotation_archive_sha256':config['annotation_archive_sha256'],
                'data_dir':str(data), 'source_bytes':source_bytes,
                'archive_bytes':writer.size, 'archive_sha256':writer.hasher.hexdigest(),
                'note':('Portable processed data; rebuild HF metadata caches at destination.' if portable
                        else 'Staged dataset/cache backup, not evidence of real-mask geometry parity.')}
    if portable:
        metadata['loader_source_sha256'] = config['loader_source_sha256']
    archive.with_name(archive.name+'.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Backup complete: {archive} ({writer.size/2**30:.2f} GiB). '
          'Keep its JSON sidecar with it.', flush=True)


def restore(data, archive, config):
    data, archive = data.resolve(), archive.resolve()
    if data.exists() and any(data.iterdir()):
        raise ValueError('Restore requires an empty data directory; existing runtime data is never overwritten')
    metadata = cache_metadata(archive, config, data)
    if not hasattr(tarfile, 'data_filter'):
        raise RuntimeError('Safe extraction requires Python 3.11.8+ (the notebook uses isolated Python 3.11)')
    data.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(data.parent).free < metadata['source_bytes']:
        raise ValueError('Not enough runtime-local disk space for the restored dataset')
    data.mkdir(parents=True, exist_ok=True)
    print('Archive checksum verified; extracting to runtime-local storage...', flush=True)
    with tarfile.open(archive, 'r:') as tar:
        tar.extractall(data, filter='data')
    check_staging(data, config)
    if metadata['format'] == 2:
        if (metadata['loader_source_sha256'] != config['loader_source_sha256']
                or digest(data/'MedVision.py') != config['loader_source_sha256']):
            raise ValueError('Restored portable loader does not match the pinned source')
    print(f'Restored {metadata["source_bytes"]/2**30:.2f} GiB to {data}. '
          'Run normal staging/preflight to verify the environment and reuse these files.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inspect', 'backup', 'restore'])
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True, help='Pinned configs/kits23-pilot.json')
    p.add_argument('--portable', action='store_true', help='Export format 2 without machine-specific HF caches')
    args = p.parse_args()
    data, archive = args.data_dir.resolve(), args.archive.resolve()
    if archive.is_relative_to(data):
        p.error('Archive must be outside the runtime data directory')
    config = json.loads(args.config.read_text())
    if args.action == 'inspect':
        inspect_data(data, config, args.portable)
    elif args.action == 'backup':
        backup(data, archive, config, args.portable)
    else:
        restore(data, archive, config)


if __name__ == '__main__':
    main()
