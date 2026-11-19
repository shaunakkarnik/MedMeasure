"""Offline regression checks for official-test shard coverage and retries."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    import medmeasure.version1_data
    from medmeasure.version2 import score_answer
    from scripts import version2_test as runner
except ModuleNotFoundError as error:
    raise unittest.SkipTest('Requires the V1/V2 Python environment') from error

from medmeasure.data import write_json
from scripts.summarize_baseline import summarize


class OfficialTestShardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifests = self.root / 'audit'
        self.manifests.mkdir()
        self.rows = [{'sample_id': f'sample-{index}', 'case_id': f'case-{index}',
                      'major_mm': 12., 'minor_mm': 5.} for index in range(5)]
        for name in ('train', 'validation', 'test', 'test_legacy_disjoint'):
            selected = self.rows[:2] if name == 'test_legacy_disjoint' else self.rows
            (self.manifests / f'{name}.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in selected))
        self.run = self.root / 'run'

    def make_shards(self):
        for index in range(2):
            selected = self.rows[index::2]
            directory = self.run / f'shard-{index}'
            directory.mkdir(parents=True)
            write_json(directory / 'run.json', {
                'shard_count': 2, 'shard_index': index,
                'sample_ids': [row['sample_id'] for row in selected],
                'checkpoint_sha256': 'same-checkpoint',
                'manifest_sha256': {'test': runner.sha256(self.manifests / 'test.jsonl')}})
            records = [score_answer(row, '<answer>(12, 5)</answer>') for row in selected]
            (directory / 'predictions.jsonl').write_text(
                ''.join(json.dumps(record) + '\n' for record in records))
            write_json(directory / 'summary.json', summarize(records))

    def test_merge_orders_all_rows_and_scores_diagnostic(self):
        self.make_shards()
        runner.merge(SimpleNamespace(run=self.run, manifests=self.manifests))
        records = [json.loads(line) for line in (self.run / 'predictions.jsonl').read_text().splitlines()]
        self.assertEqual([r['sample_id'] for r in records], [r['sample_id'] for r in self.rows])
        summary = json.loads((self.run / 'summary.json').read_text())
        self.assertTrue(summary['complete_official_test'])
        self.assertEqual(summary['groups']['all']['total'], 5)
        self.assertEqual(summary['groups']['legacy_disjoint']['total'], 2)

    def test_missing_and_mismatched_shards_are_rejected(self):
        self.make_shards()
        path = self.run / 'shard-1/run.json'
        receipt = json.loads(path.read_text())
        receipt['checkpoint_sha256'] = 'different'
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, 'Shard model'):
            runner.merge(SimpleNamespace(run=self.run, manifests=self.manifests))
        receipt['checkpoint_sha256'] = 'same-checkpoint'
        write_json(path, receipt)
        (self.run / 'shard-1/summary.json').unlink()
        with self.assertRaises(FileNotFoundError):
            runner.merge(SimpleNamespace(run=self.run, manifests=self.manifests))
        self.assertFalse((self.run / 'summary.json').exists())

    def test_duplicate_rows_and_tampered_scores_are_rejected(self):
        self.make_shards()
        path = self.run / 'shard-0/predictions.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[1] = records[0]
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        with self.assertRaisesRegex(ValueError, 'Missing, duplicated'):
            runner.checked_predictions(path, self.rows[::2])
        records = [score_answer(row, '<answer>12, 5</answer>') for row in self.rows[::2]]
        records[0]['avgMAE']['MAE'] = 99.
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        with self.assertRaisesRegex(ValueError, 'Saved metrics'):
            runner.checked_predictions(path, self.rows[::2])

    def test_checkpoint_adapters_and_training_contract_are_enforced(self):
        from medmeasure.version2 import add_vision_adapters

        config = json.loads((runner.ROOT / 'configs/version2-kits23.json').read_text())
        config.update(first_adapter_block=0, adapter_rank=2)
        config_path = self.root / 'config.json'
        write_json(config_path, config)

        def backbone():
            model = torch.nn.Module()
            model.visual = torch.nn.Module()
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.attn.qkv = torch.nn.Linear(2, 6)
            model.visual.blocks = torch.nn.ModuleList([block])
            return model.requires_grad_(False)

        original = backbone()
        add_vision_adapters(original, 0, 2)
        adapters = {name: torch.full_like(param, .25)
                    for name, param in original.named_parameters() if param.requires_grad}
        contract = {'format_version': 1, 'pin': json.loads(
            (runner.ROOT / 'configs/kits23-pilot.json').read_text()), 'config': config,
            'source_sha256': runner.source_hashes(), 'dtype': 'float32',
            'torch': str(torch.__version__), 'objective': 'text', 'train_limit': 0,
            'manifest_sha256': {name: runner.sha256(self.manifests / f'{name}.jsonl')
                                for name in ('train', 'validation', 'test')}}
        checkpoint = self.root / 'best.pt'
        state = {'contract': contract, 'epoch': 3, 'adapters': adapters}
        torch.save(state, checkpoint)
        args = SimpleNamespace(data_root=self.root / 'data', manifests=self.manifests,
                               output=self.run / 'shard-0', checkpoint=checkpoint,
                               config=config_path, shard_index=0, shard_count=2,
                               device='cpu', dtype='float32', local_files_only=True)

        def generate(model, processor, rows, dataset_dir, config, device, path):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    torch.testing.assert_close(param, adapters[name])
            records = [score_answer(row, '<answer>12, 5</answer>') for row in rows]
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            return summarize(records)

        with patch('medmeasure.version1_data.validate_manifests'), \
                patch('medmeasure.data.read_plan'), \
                patch('medmeasure.data.rows', return_value=self.rows), \
                patch('scripts.version1.load_model',
                      return_value=(None, SimpleNamespace(backbone=backbone()), None)), \
                patch('scripts.version2.evaluate', side_effect=generate):
            runner.evaluate(args)
            contract['source_sha256']['medmeasure/version2.py'] = 'changed'
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, 'source_sha256'):
                runner.evaluate(args)
        receipt = json.loads((args.output / 'run.json').read_text())
        self.assertEqual(receipt['epoch'], 3)
        self.assertEqual(receipt['objective'], 'text')

    def test_evaluate_retries_completed_shard_without_loading_model(self):
        args = SimpleNamespace(data_root=self.root / 'data', manifests=self.manifests,
                               output=self.run / 'shard-0', checkpoint=None,
                               config=runner.ROOT / 'configs/version2-kits23.json',
                               shard_index=0, shard_count=2, device='cpu', dtype='float32',
                               local_files_only=True)

        def generate(backbone, processor, rows, dataset_dir, config, device, path):
            records = [score_answer(row, '<answer>12, 5</answer>') for row in rows]
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            return summarize(records)

        with patch('medmeasure.version1_data.validate_manifests'), \
                patch('medmeasure.data.read_plan'), \
                patch('medmeasure.data.rows', return_value=self.rows), \
                patch('scripts.version1.load_model',
                      return_value=(None, SimpleNamespace(backbone=None), None)) as load, \
                patch('scripts.version2.evaluate', side_effect=generate):
            runner.evaluate(args)
            runner.evaluate(args)
            self.assertEqual(load.call_count, 1)
        self.assertEqual(len((args.output / 'predictions.jsonl').read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main()
