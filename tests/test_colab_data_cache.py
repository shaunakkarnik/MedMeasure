"""Small synthetic archive checks; no real-data downloads or Drive access."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from scripts.cache_colab_data import backup, cache_metadata, restore


class ProcessedDataCache(unittest.TestCase):
    def staged_data(self, root):
        data = root/'runtime-data'
        dataset = data/'Datasets/KiTS23'
        for folder in ['Images', 'Masks']:
            (dataset/folder).mkdir(parents=True)
            (dataset/folder/'case_001.nii.gz').write_bytes(b'synthetic scan')
        (dataset/'complete.marker').write_text('staged')
        (data/'metadata').mkdir()
        plan = data/'metadata/benchmark_plan_biometry_v1.4.0.json.gz'
        plan.write_bytes(b'pinned metadata')
        config = {'dataset_revision':'a'*40, 'annotation_archive_sha256':'b'*64,
                  'sha256':{plan.name:hashlib.sha256(plan.read_bytes()).hexdigest()}}
        (data/'staging.json').write_text(json.dumps({'revision':config['dataset_revision'],
            'with_images':True, 'annotation_archive_sha256':config['annotation_archive_sha256']}))
        cache = data/'.cache/huggingface'
        cache.mkdir(parents=True)
        (cache/'token').write_text('must not be archived')
        (cache/'loader.arrow').write_bytes(b'synthetic cache')
        (cache/'alias.arrow').symlink_to('loader.arrow')
        return data, config

    def test_round_trip_preserves_cache_markers_and_excludes_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config = self.staged_data(root)
            archive = root/'drive/cache.tar'
            backup(data, archive, config)
            with tarfile.open(archive) as tar:
                self.assertNotIn('./.cache/huggingface/token', tar.getnames())
            with self.assertRaisesRegex(ValueError, 'empty data directory'):
                restore(data, archive, config)
            shutil.rmtree(data)
            restore(data, archive, config)
            self.assertEqual((data/'.cache/huggingface/alias.arrow').read_bytes(), b'synthetic cache')
            self.assertEqual((data/'Datasets/KiTS23/complete.marker').read_text(), 'staged')
            self.assertFalse((data/'.cache/huggingface/token').exists())

    def test_corrupt_wrong_revision_and_incomplete_archives_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config = self.staged_data(root)
            archive = root/'drive/cache.tar'
            backup(data, archive, config)
            changed = dict(config, dataset_revision='c'*40)
            with self.assertRaisesRegex(ValueError, 'pinned dataset'):
                cache_metadata(archive, changed, data)
            with archive.open('r+b') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                cache_metadata(archive, config, data)
            archive.with_name(archive.name+'.json').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                cache_metadata(archive, config, data)

    def test_unfinished_staging_and_external_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config = self.staged_data(root)
            archive = root/'drive/cache.tar'
            staged = data/'staging.json'
            original = staged.read_text()
            staged.write_text(json.dumps({'with_images':False}))
            with self.assertRaisesRegex(ValueError, 'with-images'):
                backup(data, archive, config)
            staged.write_text(original)
            (root/'outside.txt').write_text('outside data root')
            (data/'external-link').symlink_to(root/'outside.txt')
            with self.assertRaisesRegex(ValueError, 'outside'):
                backup(data, archive, config)
            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_name(archive.name+'.json').exists())

    def test_portable_bundle_relocates_without_absolute_arrow_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config = self.staged_data(root)
            loader = data/'MedVision.py'
            loader.write_text('# pinned local loader\n')
            config['loader_source_sha256'] = hashlib.sha256(loader.read_bytes()).hexdigest()
            (data/'.downloaded_datasets.json').write_text(json.dumps({'dataset_KiTS23':'1.4.0'}))
            (data/'src').mkdir()
            (data/'src/package.py').write_text('# source\n')
            archive = root/'drive/portable.tar'
            backup(data, archive, config, portable=True)
            with tarfile.open(archive) as tar:
                self.assertFalse(any('/.cache/' in name for name in tar.getnames()))
            destination = root/'different-runtime-root'
            restore(destination, archive, config)
            self.assertTrue((destination/'MedVision.py').is_file())
            self.assertTrue((destination/'src/package.py').is_file())
            self.assertFalse((destination/'.cache').exists())
            self.assertTrue((destination/'Datasets/KiTS23/Images/case_001.nii.gz').is_file())

    def test_legacy_cache_still_refuses_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config = self.staged_data(root)
            archive = root/'drive/cache.tar'
            backup(data, archive, config)
            with self.assertRaisesRegex(ValueError, 'original data directory'):
                restore(root/'another-root', archive, config)
