"""Regression checks for LoRA export and original-benchmark protocol matching."""
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import load_file, save_file
except ModuleNotFoundError as error:
    raise unittest.SkipTest('Requires Torch and safetensors') from error

from scripts.export_version2 import export, merged_weight
from scripts.version2_original import check_protocol, reference_samples, merge
from scripts.version2_test import source_hashes
from medmeasure.data import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


class OriginalBenchmarkTests(unittest.TestCase):
    def test_merge_matches_float32_low_rank_math(self):
        torch.manual_seed(12)
        weight = torch.randn(6, 4)
        down = torch.randn(2, 4)
        up = torch.randn(6, 2)
        x = torch.randn(3, 4)
        merged = merged_weight(weight, down, up, torch.float32)
        torch.testing.assert_close(x @ merged.T, x @ weight.T + (x @ down.T) @ up.T)
        with self.assertRaisesRegex(ValueError, 'shape'):
            merged_weight(weight[:3], down, up, torch.float32)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            merged_weight(weight * float('nan'), down, up, torch.float32)

    def test_export_preserves_assets_and_unadapted_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, output = root / 'base', root / 'export'
            base.mkdir()
            config = {'vision_config': {'depth': 1}, 'torch_dtype': 'bfloat16'}
            write_json(base / 'config.json', config)
            write_json(base / 'preprocessor_config.json', {'min_pixels': 3136, 'max_pixels': 12845056})
            write_json(base / 'tokenizer_config.json', {'test': 'unchanged'})
            name = 'visual.blocks.0.attn.qkv.weight'
            weight = torch.randn(6, 4)
            bias = torch.randn(6)
            language = torch.randn(5, 4)
            save_file({name: weight, 'visual.blocks.0.attn.qkv.bias': bias},
                      str(base / 'model-00001-of-00002.safetensors'), metadata={'format': 'pt'})
            save_file({'lm_head.weight': language}, str(base / 'model-00002-of-00002.safetensors'))
            mapping = {name: 'model-00001-of-00002.safetensors',
                       'visual.blocks.0.attn.qkv.bias': 'model-00001-of-00002.safetensors',
                       'lm_head.weight': 'model-00002-of-00002.safetensors'}
            write_json(base / 'model.safetensors.index.json',
                       {'metadata': {'total_size': (weight.numel() + bias.numel() + language.numel()) * 4},
                        'weight_map': mapping})
            down, up = torch.randn(2, 4), torch.randn(6, 2)
            contract = {'pin': json.loads((ROOT / 'configs/kits23-pilot.json').read_text()),
                        'source_sha256': source_hashes(), 'train_limit': 0,
                        'objective': 'joint', 'dtype': 'bfloat16',
                        'config': {'first_adapter_block': 0, 'adapter_rank': 2}}
            checkpoint = root / 'best.pt'
            torch.save({'contract': contract, 'epoch': 3,
                        'adapters': {'visual.blocks.0.attn.qkv.down.weight': down,
                                     'visual.blocks.0.attn.qkv.up.weight': up}}, checkpoint)
            export(checkpoint, output, base)
            tensors = load_file(str(output / mapping[name]))
            torch.testing.assert_close(tensors[name], merged_weight(weight, down, up, torch.bfloat16))
            torch.testing.assert_close(tensors['visual.blocks.0.attn.qkv.bias'], bias)
            self.assertEqual(sha256(output / 'model-00002-of-00002.safetensors'),
                             sha256(base / 'model-00002-of-00002.safetensors'))
            for asset in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
                self.assertEqual(sha256(output / asset), sha256(base / asset))
            receipt = json.loads((output / 'medmeasure-provenance.json').read_text())
            self.assertEqual(receipt['epoch'], 3)
            self.assertEqual(receipt['objective'], 'joint')
            export(checkpoint, output, base)
            (output / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                export(checkpoint, output, base)

    def test_protocol_checks_prompts_targets_and_coverage(self):
        rows = [{'image_file': 'case.nii.gz', 'slice_dim': 2, 'slice_idx': 3, 'label': 2}]
        sample = {'doc_id': 0, 'doc': rows[0], 'input': 'original prompt', 'target': [12, 5]}
        check_protocol([sample], rows, {0: sample})
        for field, value in [('input', 'different prompt'), ('target', [11, 5])]:
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                check_protocol([{**sample, field: value}], rows, {0: sample})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            check_protocol([sample, sample], rows, {0: sample})

    def test_merge_preserves_failed_upstream_samples_and_checks_membership(self):
        import math
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, reference, output = root / 'audit', root / 'reference', root / 'run'
            audit.mkdir()
            rows = [{'sample_id': str(index), 'loader_index': index,
                     'image_file': f'case{index}.nii.gz', 'slice_dim': 2,
                     'slice_idx': 1, 'label': 2} for index in range(2)]
            (audit / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            (audit / 'test_legacy_disjoint.jsonl').write_text(json.dumps(rows[0]) + '\n')
            samples = [{'doc_id': index, 'doc': row, 'input': 'same prompt', 'target': [12, 5],
                        'avgMAE': {'MAE': 0., 'success': True},
                        'avgMRE': {'MRE': 0., 'success': True}} for index, row in enumerate(rows)]
            directory = reference / 'chunks/shard-000/results'
            directory.mkdir(parents=True)
            (directory / '0_samples_test.jsonl').write_text(
                ''.join(json.dumps(sample) + '\n' for sample in samples))
            samples[1]['avgMAE'] = {'MAE': float('nan'), 'success': False}
            samples[1]['avgMRE'] = {'MRE': float('nan'), 'success': False}
            for index, sample in enumerate(samples):
                shard = output / f'shard-{index}'
                attempt = shard / 'attempt-001'
                (attempt / 'results').mkdir(parents=True)
                log = attempt / 'results/0_samples_test.jsonl'
                log.write_text(json.dumps(sample) + '\n')
                write_json(shard / 'completed.json',
                           {'attempt': 'attempt-001', 'samples_sha256': sha256(log)})
                write_json(shard / 'run.json',
                           {'shard_count': 2, 'shard_index': index, 'smoke': False,
                            'manifest_sha256': sha256(audit / 'test.jsonl'),
                            'sample_ids': [str(index)]})
            args = SimpleNamespace(audit=audit, output=output, reference_results=reference)
            merge(args)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['groups']['all']['failures'], 1)
            self.assertEqual(summary['groups']['all']['below_10_percent_fraction_all_rows'], .5)
            self.assertEqual(summary['groups']['legacy_disjoint']['total'], 1)
            merged = [json.loads(line) for line in (output / 'samples.jsonl').read_text().splitlines()]
            self.assertTrue(math.isnan(merged[1]['avgMAE']['MAE']))
            receipt = json.loads((output / 'shard-1/run.json').read_text())
            receipt['sample_ids'] = ['0']
            write_json(output / 'shard-1/run.json', receipt)
            with self.assertRaisesRegex(ValueError, 'membership'):
                merge(args)

    def test_final_comparison_uses_common_success_denominator(self):
        from scripts.compare_final_results import summarize
        rows = [
            {'methods': {'frozen': {'success': True, 'mae_mm': 2., 'mre': .2},
                         'version1': {'success': True, 'mae_mm': 1., 'mre': .05}}},
            {'methods': {'frozen': {'success': True, 'mae_mm': 100., 'mre': 10.},
                         'version1': {'success': False, 'mae_mm': None, 'mre': None}}}]
        result = summarize(rows)
        self.assertEqual(result['all_methods_success_count'], 1)
        self.assertEqual(result['methods']['frozen']['mae_mm_common_successes'], 2.)
        self.assertEqual(result['methods']['version1']['within_10_percent_all_rows'], .5)

    def test_saved_original_baseline_has_full_coverage(self):
        root = ROOT / 'results/kits23-medvision-v0-v140-rik'
        saved = reference_samples(root)
        rows = [json.loads(line) for line in (root / 'test.jsonl').read_text().splitlines()]
        self.assertEqual(len(saved), 3179)
        check_protocol(list(saved.values()), rows, saved)


if __name__ == '__main__':
    unittest.main()
