import copy
import unittest
import numpy as np
from medmeasure.baseline import validate_loaded_row
from scripts.plan_evaluation import plan_shards
from scripts.summarize_baseline import summarize


class BaselineContracts(unittest.TestCase):
    def setUp(self):
        self.expected = {'image_file':'Images/case_001.nii.gz', 'mask_file':'Masks/case_001.nii.gz',
                         'slice_idx':12, 'slice_dim':2, 'label':2, 'major_mm':13.147,
                         'minor_mm':7.312, 'pixel_size':[.7423,.7423]}
        self.actual = dict(self.expected)
        self.actual['pixel_size'] = [float(np.float16(x)) for x in self.expected['pixel_size']]
        self.actual['biometric_profile'] = {
            'metric_value_major_axis':[float(np.float16(13.147))],
            'metric_value_minor_axis':[float(np.float16(7.312))], 'metric_unit':['mm']}

    def test_hf_columnar_profile(self):
        validate_loaded_row(self.actual,self.expected)

    def test_wrong_target_rejected(self):
        for change in ['extra_target','wrong_measurement','wrong_spacing','wrong_mask']:
            with self.subTest(change=change):
                actual=copy.deepcopy(self.actual)
                if change=='extra_target': actual['biometric_profile']['metric_value_major_axis'].append(2)
                if change=='wrong_measurement': actual['biometric_profile']['metric_value_major_axis'][0]=8
                if change=='wrong_spacing': actual['pixel_size'][0]=1
                if change=='wrong_mask': actual['mask_file']='other.nii.gz'
                with self.assertRaises(ValueError): validate_loaded_row(actual,self.expected)

    def test_noncontiguous_shards_preserve_exact_selection(self):
        records=[{'loader_index':i,'sample_id':str(i),'official_split':'test'} for i in [2,7,11,900,1000]]
        self.assertEqual(plan_shards(records,2),[[2,7],[11,900],[1000]])
        records[0]['official_split']='train'
        with self.assertRaises(ValueError): plan_shards(records,2)

    def test_scoring_failures_and_strict_threshold(self):
        samples=[{'avgMAE':{'success':True,'MAE':1},'avgMRE':{'success':True,'MRE':.1}},
                 {'avgMAE':{'success':True,'MAE':2},'avgMRE':{'success':True,'MRE':.05}},
                 {'avgMAE':{'success':False,'MAE':None},'avgMRE':{'success':False,'MRE':None}},
                 {'avgMAE':{'success':True,'MAE':float('nan')},'avgMRE':{'success':True,'MRE':float('nan')}}]
        report=summarize(samples)
        self.assertEqual(report['failures'],2)
        self.assertEqual(report['below_10_percent_fraction_all_rows'],.25)
        self.assertEqual(report['below_10_percent_fraction_success_only'],.5)
        self.assertEqual(report['mae_mm_success_only'],1.5)


class ScoreFiles(unittest.TestCase):
    def test_disjoint_shard_merge_and_overlap_rejection(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            selected=[{'image_file':f'Images/{i}.nii.gz','slice_dim':2,'slice_idx':0,'label':2,'major_mm':5} for i in range(2)]
            (root/'test.jsonl').write_text('\n'.join(json.dumps(r) for r in selected))
            paths=[]
            for i,row in enumerate(selected):
                path=root/f'{i}.jsonl';paths.append(path)
                path.write_text(json.dumps({'doc':row,'avgMAE':{'success':True,'MAE':1},'avgMRE':{'success':True,'MRE':.05}}))
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/summarize_baseline.py'),
                     '--selected-rows',str(root/'test.jsonl'),'--output',str(root/'summary.json')]
            for path in paths: command+=['--samples',str(path)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/'summary.json').read_text())['groups']['all']['total'],2)
            result=subprocess.run(command+['--samples',str(paths[0])],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('duplicate, missing or unexpected',result.stderr)
