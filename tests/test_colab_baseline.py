"""CPU checks for chunk recovery and complete-run diagnostic selection."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medmeasure.data import write_json
from scripts.colab_baseline import completed_log, evaluate_chunk


class ColabRecovery(unittest.TestCase):
    def test_retry_uses_fresh_attempt_and_reuses_only_verified_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'loader_index':0, 'image_file':'Images/a.nii.gz', 'slice_dim':2,
                   'slice_idx':1, 'label':2, 'major_mm':5}
            sample = {'doc':row, 'avgMAE':{'success':True, 'MAE':1},
                      'avgMRE':{'success':True, 'MRE':.05}}
            args = SimpleNamespace(output=root, upstream=root, data_dir=root, model_path=root)
            chunk = root/'chunks/shard-000'
            (chunk/'attempt-001').mkdir(parents=True)  # Simulate an interrupted attempt.

            def fake_script(name, *arguments):
                if name == 'prepare_baseline.py':
                    attempt = Path(arguments[arguments.index('--output')+1])
                    attempt.mkdir()
                    write_json(attempt/'manifest.json', {'sample_indices':[0]})
                    write_json(attempt/'selected_rows.json', [row])
                if name == 'execute_baseline.py':
                    attempt = Path(arguments[1])
                    (attempt/'results').mkdir()
                    (attempt/'results/task_samples.jsonl').write_text(json.dumps(sample)+'\n')

            with patch('scripts.colab_baseline.run_script', side_effect=fake_script) as runner:
                log = evaluate_chunk(args, 'shard-000', [0], [row])
                self.assertIn('attempt-002', str(log))
                self.assertEqual(evaluate_chunk(args, 'shard-000', [0], [row]), log)
                self.assertEqual(runner.call_count, 4)
            log.write_text(log.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                completed_log(chunk, [0], [row])

    def test_diagnostic_requires_complete_full_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'image_file':f'Images/{i}.nii.gz', 'slice_dim':2, 'slice_idx':0,
                     'label':2, 'major_mm':5} for i in range(2)]
            samples = [{'doc':row, 'avgMAE':{'success':True, 'MAE':i+1},
                        'avgMRE':{'success':True, 'MRE':.05}} for i,row in enumerate(rows)]
            for name,items in [('full',rows), ('diagnostic',rows[1:]), ('samples',samples)]:
                (root/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
            command = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/summarize_baseline.py'),
                       '--samples', str(root/'samples.jsonl'), '--selected-rows', str(root/'diagnostic.jsonl'),
                       '--subset-of', str(root/'full.jsonl'), '--output', str(root/'summary.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root/'summary.json').read_text())['groups']['all']
            self.assertEqual(report['total'], 1)
            self.assertEqual(report['mae_mm_success_only'], 2)
            (root/'samples.jsonl').write_text(json.dumps(samples[1])+'\n')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('refusing partial score', result.stderr)
