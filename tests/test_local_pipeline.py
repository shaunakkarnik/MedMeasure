"""CPU contracts for the local-data evaluator handoff; no GPU or network."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import execute_baseline

ROOT = Path(__file__).resolve().parents[1]


class LocalDataEvaluator(unittest.TestCase):
    def test_local_task_uses_restored_loader_and_remote_task_retains_revision(self):
        for local in [False, True]:
            with self.subTest(local=local), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = {'environment':{}, 'dataset_revision':'a'*40,
                            'model_path':'/models/pinned', 'requested_limit':10,
                            'sample_indices':[2, 7],
                            'dataset_loader_path':'/content/medmeasure-data/MedVision.py' if local else None}
                (root/'manifest.json').write_text(json.dumps(manifest))
                package = ModuleType('medvision_bm')
                package.__file__ = str(root/'upstream/__init__.py')
                prompts = ModuleType('medvision_bm.rft.verl.rft_prompts')
                prompts.SYSTEM_PROMPT = 'unchanged system prompt'
                modules = {'medvision_bm':package,
                           'medvision_bm.rft':ModuleType('medvision_bm.rft'),
                           'medvision_bm.rft.verl':ModuleType('medvision_bm.rft.verl'),
                           'medvision_bm.rft.verl.rft_prompts':prompts}
                helper = SimpleNamespace(resolve_qwen25vl_hf_overrides=lambda _: {})
                with patch.dict(sys.modules, modules), \
                     patch.object(execute_baseline.importlib, 'import_module', return_value=helper), \
                     patch.object(execute_baseline.subprocess, 'run') as execute, \
                     patch.object(sys, 'argv', ['execute_baseline.py','--run',str(root)]):
                    execute_baseline.main()
                task = json.loads((root/'task_definition/pilot.yaml').read_text())
                if local:
                    self.assertEqual(task['dataset_path'], manifest['dataset_loader_path'])
                    self.assertNotIn('revision', task['dataset_kwargs'])
                else:
                    self.assertEqual(task['dataset_path'], 'YongchengYAO/MedVision')
                    self.assertEqual(task['dataset_kwargs']['revision'], 'a'*40)
                command = execute.call_args.args[0]
                model_args = command[command.index('--model_args')+1]
                self.assertIn('reshape_image_hw=512x512', model_args)
                self.assertIn('max_new_tokens=4096', model_args)
                self.assertEqual(command[command.index('--sample_indices')+1], '[2, 7]')

    def test_inference_notebook_requires_bundle_and_has_no_scan_staging(self):
        notebook = json.loads((ROOT/'notebooks/medvision_v0_kits23_local_data.ipynb').read_text())
        code = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type']=='code')
        self.assertNotIn("script('stage_data.py'", code)
        self.assertNotIn('--with-images', code)
        self.assertIn("script('prepare_inference_data.py'", code)
        self.assertIn("cache_info.get('format') != 2", code)
        for index,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                compile(cell['source'], f'local notebook cell {index}', 'exec')

    def test_cli_bootstrap_has_only_explicit_setup_sections(self):
        source = (ROOT/'scripts/colab_inference_setup.py').read_text()
        self.assertNotIn("'## Download and preprocess scans; stage checkpoint'", source)
        self.assertNotIn("script('colab_baseline.py', 'run'", source)
        self.assertIn("'## Restore processed scans and rebuild destination metadata'", source)
