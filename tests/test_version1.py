"""Offline checks with real Qwen2.5-VL modules at tiny random dimensions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

try:
    import numpy as np
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
except ModuleNotFoundError as error:
    raise unittest.SkipTest('Version 1 tests require the separate requirements-v1.txt environment') from error

from medmeasure.version1 import (FrozenMedVisionFeatures, TargetMaskDecoder,
                                restore_spatial_tokens, segmentation_loss)
from medmeasure.version1_data import (measure_prediction, score_prediction,
                                     summarize_predictions, validate_manifests)


def tiny_backbone():
    config = Qwen2_5_VLConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling={'type': 'mrope', 'mrope_section': [1, 1, 2]},
        image_token_id=60, video_token_id=61, vision_start_token_id=62,
        vision_end_token_id=63,
        vision_config={'depth': 2, 'hidden_size': 32, 'intermediate_size': 64,
                       'out_hidden_size': 32, 'num_heads': 4, 'patch_size': 2,
                       'temporal_patch_size': 2, 'spatial_merge_size': 2,
                       'window_size': 8, 'fullatt_block_indexes': [1]})
    config._attn_implementation = 'sdpa'
    return Qwen2_5_VLForConditionalGeneration(config)


def tiny_inputs():
    # Non-square grid spanning multiple attention windows, plus right padding.
    return {'input_ids': torch.tensor([[5, 62, *([60] * 12), 63, 6, 7, 0, 0]]),
            'attention_mask': torch.tensor([[1] * 17 + [0, 0]]),
            'pixel_values': torch.randn(48, 3 * 2 * 2 * 2),
            'image_grid_thw': torch.tensor([[1, 6, 8]])}


class Version1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_restore_both_window_and_merge_packing(self):
        height, width, merge = 6, 8, 2
        raster = torch.arange(height * width).reshape(height, width, 1)
        packed = raster.reshape(height // merge, merge, width // merge, merge, 1).permute(0, 2, 1, 3, 4)
        backbone = tiny_backbone()
        grid = torch.tensor([[1, height, width]])
        window, _ = backbone.visual.get_window_index(grid)
        self.assertFalse(torch.equal(window, torch.arange(len(window))))
        tokens = packed.reshape(-1, merge * merge, 1)[window].reshape(-1, 1)
        restored = restore_spatial_tokens(tokens, grid, window, merge)
        self.assertTrue(torch.equal(restored[0, 0], raster[:, :, 0]))
        with self.assertRaises(ValueError):
            restore_spatial_tokens(tokens, torch.tensor([[2, height, width]]), window, merge)

    def test_backbone_parity_prompt_conditioning_and_decoder_gradients(self):
        torch.manual_seed(2)
        backbone = tiny_backbone().eval()
        extractor = FrozenMedVisionFeatures(backbone, [0, 1])
        inputs = tiny_inputs()
        maps, query = extractor(inputs)
        # Compare against the original HF multimodal forward, including its RoPE.
        with torch.no_grad():
            original = backbone(**inputs, output_hidden_states=True, use_cache=False)
        torch.testing.assert_close(query, original.hidden_states[-1][:, 16])
        self.assertEqual([tuple(x.shape) for x in maps], [(1, 32, 6, 8)] * 2)
        changed = {k: v.clone() for k, v in inputs.items()}
        changed['input_ids'][0, 16] = 8
        _, other = extractor(changed)
        self.assertFalse(torch.allclose(query, other))
        extractor.train()
        self.assertFalse(backbone.training)
        decoder = TargetMaskDecoder(32, 32, 2, dim=16, heads=4)
        logits = decoder(maps, query, (24, 32))
        target = torch.zeros_like(logits)
        target[:, :, 7:17, 9:21] = 1
        segmentation_loss(logits, target).backward()
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()))
        self.assertEqual(len(backbone.visual.blocks[0]._forward_hooks), 0)
        with self.assertRaises(ValueError):
            extractor({**inputs, 'labels': inputs['input_ids']})

    def test_small_frozen_feature_overfit(self):
        torch.manual_seed(3)
        decoder = TargetMaskDecoder(8, 12, 2, dim=16, heads=4)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 10), indexing='ij')
        target = ((x ** 2 + y ** 2) < .4).float()[None, None]
        maps = [torch.randn(1, 8, 8, 10), torch.randn(1, 8, 8, 10)]
        maps[0][:, 0] = target[:, 0]
        query = torch.randn(1, 12)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=.01)
        initial = segmentation_loss(decoder(maps, query, target.shape[-2:]), target).item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = segmentation_loss(decoder(maps, query, target.shape[-2:]), target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = decoder(maps, query, target.shape[-2:])
            final = segmentation_loss(logits, target).item()
            prediction = logits.sigmoid() >= .5
        self.assertLess(final, initial * .1)
        self.assertTrue(torch.equal(prediction, target.bool()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.pt'
            torch.save({'decoder': decoder.state_dict(), 'torch': str(torch.__version__)}, path)
            restored = TargetMaskDecoder(8, 12, 2, dim=16, heads=4)
            restored.load_state_dict(torch.load(path, weights_only=True)['decoder'])
            torch.testing.assert_close(logits, restored(maps, query, target.shape[-2:]))


class Version1MeasurementTests(unittest.TestCase):
    def test_failures_and_coverage_are_retained(self):
        rr, cc = np.indices((64, 80))
        target = ((rr - 30) / 12) ** 2 + ((cc - 40) / 20) ** 2 <= 1
        measurement = measure_prediction(target, [1.5, .7])
        self.assertEqual(measurement['status'], 'success')
        row = {'sample_id': 'one', 'case_id': 'case', **measurement['measurement']}
        success = score_prediction(row, measurement, target, target)
        empty = np.zeros_like(target)
        failed_row = {**row, 'sample_id': 'two'}
        failed = score_prediction(failed_row, measure_prediction(empty, [1.5, .7]), empty, target)
        self.assertEqual(failed['status'], 'empty_mask')
        tiny = empty.copy()
        tiny[1, 1] = True
        self.assertEqual(measure_prediction(tiny, [1.5, .7])['status'], 'failed_fit')
        report = summarize_predictions([success, failed], [row, failed_row])
        self.assertEqual(report['failures'], 1)
        self.assertEqual(report['below_10_percent_fraction_all_rows'], .5)
        self.assertEqual(success['planner_mae_mm'], 0)
        with self.assertRaises(ValueError):
            summarize_predictions([success], [row, failed_row])
        with self.assertRaises(ValueError):
            summarize_predictions([success, success], [row])

    def test_pinned_split_checks(self):
        root = Path(__file__).resolve().parents[1]
        plan = root / 'data/medvision/metadata/benchmark_plan_biometry_v1.4.0.json.gz'
        if not plan.exists():
            self.skipTest('Optional local pinned annotation plan is unavailable')
        from medmeasure.data import read_plan, rows
        task = read_plan(plan)
        training, test = rows(task, 'train'), rows(task, 'test')
        pin = json.loads((root / 'configs/kits23-pilot.json').read_text())
        validate_manifests({'train': training[:1], 'test': test[:1]}, plan, pin)
        with self.assertRaises(ValueError):
            validate_manifests({'train': test[:1]}, plan, pin)
        with self.assertRaises(ValueError):
            validate_manifests({'train': training[:1], 'validation': training[:1]}, plan, pin)
        with self.assertRaises(ValueError):
            validate_manifests({'train': [{**training[0], 'slice_idx': -1}]}, plan, pin)


class Version1WorkflowTests(unittest.TestCase):
    def test_training_resume_and_evaluation_with_tiny_real_backbone(self):
        import nibabel as nib
        from scripts.version1 import train, evaluate

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.question = messages[0]['content'][1]['text']
                return self.question

            def __call__(self, **kwargs):
                return tiny_inputs()

        torch.manual_seed(4)
        torch.set_num_threads(1)
        config = {'vision_layers': [0, 1], 'decoder_dim': 16, 'attention_heads': 4,
                  'question': 'Segment the kidney tumor.', 'threshold': .5,
                  'learning_rate': .001, 'seed': 4}
        pin = {'model_revision': 'tiny-random-test-only'}
        def models(*args):
            return (Processor(), FrozenMedVisionFeatures(tiny_backbone(), [0, 1]),
                    TargetMaskDecoder(32, 32, 2, 16, 4))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'Datasets/KiTS23'
            (dataset / 'Images').mkdir(parents=True)
            (dataset / 'Masks').mkdir()
            rr, cc = np.indices((24, 32))
            mask = ((rr - 12) ** 2 / 25 + (cc - 16) ** 2 / 49 <= 1)
            manifests = {}
            for name in ('train', 'validation', 'test'):
                image = nib.Nifti1Image(np.zeros((24, 32, 1), np.float32), np.eye(4))
                nib.save(image, dataset / 'Images' / f'{name}.nii.gz')
                nib.save(nib.Nifti1Image((mask[:, :, None] * 2).astype(np.uint8), np.eye(4)),
                         dataset / 'Masks' / f'{name}.nii.gz')
                measurement = measure_prediction(mask, [1, 1])['measurement']
                manifests[name] = [{'sample_id': name, 'case_id': name, 'slice_dim': 2, 'slice_idx': 0,
                    'label': 2, 'image_file': f'Images/{name}.nii.gz', 'mask_file': f'Masks/{name}.nii.gz',
                    'image_size_3d': [24, 32, 1], 'affine': np.eye(4).tolist(),
                    'orientation': ['R', 'A', 'S'], 'pixel_size': [1, 1], **measurement}]
            (root / 'test_legacy_disjoint.jsonl').write_text(json.dumps(manifests['test'][0]) + '\n')
            args = SimpleNamespace(device='cpu', dtype='float32', data_root=root, output=root / 'train-run',
                                   epochs=1, overfit_rows=0, overlays=1, resume=None)
            cases = {name: [name] for name in manifests}
            metadata = {'manifest_sha256': {'train': 'test-fixture'}, 'overfit_rows': 0}
            with patch('scripts.version1.load_model', side_effect=models):
                train(args, config, pin, manifests, cases, metadata, None)
                state = torch.load(args.output / 'last.pt', weights_only=True)
                self.assertEqual(state['epoch'], 1)
                self.assertTrue((args.output / 'best.pt').is_file())
                args.resume = args.output / 'last.pt'
                args.output = root / 'resumed-run'
                args.epochs = 2
                train(args, config, pin, manifests, cases, metadata, state)
                state = torch.load(args.output / 'last.pt', weights_only=True)
                self.assertEqual(state['epoch'], 2)
                self.assertTrue((args.output / 'best.pt').is_file())
                args.output = root / 'overfit-run'
                args.resume = None
                args.overfit_rows = 1
                train(args, config, pin, manifests, cases, {**metadata, 'overfit_rows': 1}, None)
                overfit = json.loads((args.output / 'epochs.jsonl').read_text().splitlines()[-1])
                self.assertIsNone(overfit['validation_loss'])
                self.assertIsNotNone(overfit['overfit_loss'])
                args.output = root / 'evaluation'
                args.limit = 0
                args.manifests = root
                evaluate(args, config, pin, manifests, metadata, state)
            report = json.loads((args.output / 'summary.json').read_text())
            self.assertTrue(report['complete_official_test'])
            self.assertEqual(report['groups']['all']['total'], 1)
            self.assertEqual(report['groups']['legacy_disjoint']['total'], 1)
            self.assertTrue((args.output / 'masks/test_0.png').is_file())


if __name__ == '__main__':
    unittest.main()
