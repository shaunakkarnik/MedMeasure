"""Execute the pinned upstream evaluator on one locally generated task (GPU required).

Called only by an explicitly launched run.sh. The direct lmms_eval CLI accepts
literal index lists; the higher-level upstream launcher accepts ranges only.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = json.loads((args.run/'manifest.json').read_text())
    os.environ.update(run['environment'])
    import medvision_bm
    helper = importlib.import_module('medvision_bm.benchmark.eval__medvision-model-rft')
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT
    base = Path(medvision_bm.__file__).parent/'medvision_lmms_eval/lmms_eval/tasks/KiTS23/KiTS23_TumorLesionSize_Task01_Axial-CoT.yaml'
    task_dir = args.run/'task_definition'
    task_dir.mkdir(exist_ok=True)
    # JSON is valid YAML, and quoted absolute paths avoid YAML escaping ambiguities.
    (task_dir/'pilot.yaml').write_text(json.dumps({
        'include': str(base), 'task': 'MedMeasure_KiTS23_TL_Axial',
        'dataset_kwargs': {'trust_remote_code': True, 'revision': run['dataset_revision']},
    }, indent=2))
    overrides = helper.resolve_qwen25vl_hf_overrides(run['model_path'])
    model_args = ','.join([
        f"model_hf={run['model_path']}", 'tensor_parallel_size=1', 'max_num_seqs=1',
        'gpu_memory_utilization=0.9', 'max_new_tokens=4096', 'dtype=auto',
        'reshape_image_hw=512x512',
        'hf_overrides='+json.dumps(overrides,separators=(',',':')),
        'system_prompt='+json.dumps([SYSTEM_PROMPT],separators=(',',':')),
    ])
    command = [sys.executable,'-m','lmms_eval','--model','vllm_qwen25vl',
               '--model_args',model_args,'--include_path',str(task_dir),
               '--tasks','MedMeasure_KiTS23_TL_Axial','--batch_size','1',
               '--limit',str(run['requested_limit']),'--log_samples',
               '--output_path',str(args.run/'results')]
    if run['sample_indices'] is not None:
        command += ['--sample_indices',json.dumps(run['sample_indices'])]
    (args.run/'executed-command.json').write_text(json.dumps(command,indent=2)+'\n')
    subprocess.run(command,check=True)


if __name__ == '__main__':
    main()
