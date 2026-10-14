"""Resumable, lossless frozen features and native training inputs on disk."""
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
import transformers

from medmeasure.data import sha256, write_json


def contract(config, pin, manifests, data_root, dtype):
    """Bind features to their inputs, extraction code and numeric precision."""
    root = Path(__file__).resolve().parents[1]
    selected = {name: manifests[name] for name in ('train', 'validation')}
    files = sorted({row[key] for records in selected.values() for row in records
                    for key in ('image_file', 'mask_file')})
    dataset = data_root / 'Datasets/KiTS23'
    print(f'Hashing {len(files)} source image/mask files for feature-cache identity.', flush=True)
    return {'format_version': 1, 'pin': pin, 'dtype': dtype,
            'feature_config': {key: config.get(key) for key in
                               ('vision_layers', 'min_pixels', 'max_pixels', 'question')},
            'rows': selected,
            'input_sha256': {name: sha256(dataset / name) for name in files},
            'source_sha256': {name: sha256(root / name) for name in
                              ('scripts/version1.py', 'medmeasure/version1.py', 'medmeasure/version1_data.py',
                               'medmeasure/version1_cache.py', 'requirements-v1.txt')},
            'torch': str(torch.__version__), 'transformers': transformers.__version__}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class FeatureCache:
    def __init__(self, directory, expected, *, create=False):
        self.directory = Path(directory)
        self.expected = expected
        self.identity = identity(expected)
        self.rows = expected['rows']['train'] + expected['rows']['validation']
        manifest = self.directory / 'contract.json'
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                if any(self.directory.iterdir()):
                    raise ValueError('Feature cache directory is nonempty without a contract')
                write_json(manifest, expected)
        if not manifest.exists() or json.loads(manifest.read_text()) != expected:
            raise ValueError('Feature cache differs from current inputs, pins, preprocessing or precision')
        complete = self.directory / 'complete.json'
        self.complete = json.loads(complete.read_text()) if complete.exists() else None
        if self.complete:
            ids = {row['sample_id'] for row in self.rows}
            if (self.complete['contract_sha256'] != self.identity
                    or set(self.complete['file_sha256']) != ids):
                raise ValueError('Feature cache completion record differs from its contract')
        elif not create:
            raise ValueError('Feature cache is incomplete; resume cache-features first')

    def path(self, row):
        return self.directory / (identity(row['sample_id']) + '.pt')

    def read(self, row):
        path = self.path(row)
        raw = path.read_bytes()
        if self.complete and hashlib.sha256(raw).hexdigest() != self.complete['file_sha256'][row['sample_id']]:
            raise ValueError(f"Feature cache checksum mismatch: {row['sample_id']}")
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if value['contract_sha256'] != self.identity or value['row'] != row:
            raise ValueError(f"Feature cache row identity mismatch: {row['sample_id']}")
        maps, query, mask = value['maps'], value['query'], value['mask']
        if (len(maps) != len(self.expected['feature_config']['vision_layers'])
                or query.ndim != 2 or query.shape[0] != 1
                or any(x.ndim != 4 or x.shape[0] != 1 for x in maps)
                or mask.dtype != torch.bool or mask.ndim != 2):
            raise ValueError('Invalid cached feature/target shapes')
        image = Image.fromarray(value['image'].numpy())
        return maps, query, mask.float()[None, None], image, value['spacing'].numpy(), mask.numpy()

    def write(self, row, item):
        maps, query, _, image, spacing, mask = item
        # Clone tensors to avoid serializing a larger backing storage or GPU state.
        value = {'contract_sha256': self.identity, 'row': row,
                 'maps': [x.detach().cpu().contiguous().clone() for x in maps],
                 'query': query.detach().cpu().contiguous().clone(),
                 'image': torch.from_numpy(np.asarray(image).copy()),
                 'spacing': torch.from_numpy(np.asarray(spacing).copy()),
                 'mask': torch.from_numpy(mask.copy())}
        if not all(torch.isfinite(x).all() for x in [*value['maps'], value['query']]):
            raise ValueError('Nonfinite frozen features; cache entry was not saved')
        needed = sum(x.numel() * x.element_size() for x in
                     [*value['maps'], value['query'], value['image'], value['spacing'], value['mask']])
        if shutil.disk_usage(self.directory).free < needed + 1024 * 1024:
            raise ValueError('Insufficient disk space for next feature-cache entry')
        path = self.path(row)
        temporary = path.with_suffix('.pt.tmp')
        torch.save(value, temporary)
        temporary.replace(path)


def build_cache(args, config, pin, manifests, load_model, sample):
    expected = contract(config, pin, manifests, args.data_root, args.dtype)
    cache = FeatureCache(args.output, expected, create=True)
    hashes = {}
    dimensions = None
    processor = features = decoder = None
    written = 0
    for index, row in enumerate(cache.rows):
        if not cache.path(row).exists():
            if args.max_rows and written >= args.max_rows:
                print(f'Cache incomplete: {index}/{len(cache.rows)} rows available; rerun to continue.', flush=True)
                return
            if features is None:
                processor, features, decoder = load_model(config, pin, args)
                del decoder
            cache.write(row, sample(row, args.data_root / 'Datasets/KiTS23',
                                    processor, features, config, args.device))
            written += 1
        # Validate saved entries on reuse; a corrupt entry is never silently reused.
        maps, query, *_ = cache.read(row)
        current = {'vision_dim': maps[0].shape[1], 'language_dim': query.shape[1]}
        if dimensions is not None and dimensions != current:
            raise ValueError('Feature dimensions vary between cache entries')
        dimensions = current
        hashes[row['sample_id']] = sha256(cache.path(row))
        if index == 0:
            estimate = cache.path(row).stat().st_size * len(cache.rows) / 2**30
            print(f'Estimated complete cache size from first row: {estimate:.1f} GiB.', flush=True)
        if index % 50 == 0:
            print(f'Cached/verified {index + 1}/{len(cache.rows)} rows.', flush=True)
    total = sum(cache.path(row).stat().st_size for row in cache.rows)
    temporary = cache.directory / 'complete.json.tmp'
    write_json(temporary, {'contract_sha256': cache.identity, 'dimensions': dimensions,
                           'file_sha256': hashes, 'bytes': total})
    temporary.replace(cache.directory / 'complete.json')
    print(f'Complete: {len(cache.rows)} rows, {total / 2**30:.1f} GiB.', flush=True)
