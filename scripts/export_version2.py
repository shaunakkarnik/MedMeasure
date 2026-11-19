"""Export V2 vision LoRA checkpoints as standard sharded Hugging Face weights.

Only adapted QKV weight tensors change. Processor, tokenizer and model config
assets come from the pinned released checkpoint for upstream evaluation.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medmeasure.data import sha256, write_json


def merged_weight(weight, down, up, dtype):
    """Fold the rank-scaled update (alpha/rank = 1) into a vision QKV weight."""
    import torch

    if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
        raise ValueError('LoRA projection dimensions are incompatible')
    if tuple(weight.shape) != (up.shape[0], down.shape[1]):
        raise ValueError('LoRA update shape differs from the base weight')
    if not all(torch.isfinite(tensor).all() for tensor in (weight, down, up)):
        raise ValueError('Export contains nonfinite weights')
    # Reproduce the base-weight cast used during training before adding LoRA.
    return (weight.to(dtype).float() + up.float() @ down.float()).to(dtype)


def export(checkpoint_path, output, base=None):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from scripts.version2_test import source_hashes

    pin = json.loads((ROOT / 'configs/kits23-pilot.json').read_text())
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    contract = state['contract']
    if contract['pin'] != pin or contract['source_sha256'] != source_hashes():
        raise ValueError('Training checkpoint pins or runtime sources differ')
    if contract['train_limit'] or contract['objective'] not in ('text', 'joint'):
        raise ValueError('Export requires a full-training text or joint checkpoint')
    if contract['dtype'] != 'bfloat16':
        raise ValueError('This benchmark export expects the trained bfloat16 configuration')
    config = contract['config']
    dtype = torch.bfloat16
    if base is None:
        base = Path(snapshot_download(pin['model_repo'], revision=pin['model_revision'],
                                      local_files_only=True))
    else:
        base = Path(base)
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    # Each adapted block contributes precisely the down and up weights.
    pairs = {}
    for name, tensor in state['adapters'].items():
        match = re.fullmatch(r'visual\.blocks\.(\d+)\.attn\.qkv\.(down|up)\.weight', name)
        if not match:
            raise ValueError(f'Unexpected adapter parameter: {name}')
        block, part = match.groups()
        pairs.setdefault(int(block), {})[part] = tensor
    depth = json.loads((base / 'config.json').read_text())['vision_config']['depth']
    if set(pairs) != set(range(config['first_adapter_block'], depth)):
        raise ValueError('Adapter blocks differ from the training configuration')
    updates = {}
    for block, pair in pairs.items():
        if set(pair) != {'up', 'down'} or pair['down'].shape[0] != config['adapter_rank']:
            raise ValueError('Incomplete adapter pair or incorrect rank')
        name = f'visual.blocks.{block}.attn.qkv.weight'
        if name not in index['weight_map']:
            raise ValueError(f'Base checkpoint is missing {name}')
        updates[name] = pair

    receipt = {'kind': 'vision_lora_merged', 'repo': pin['model_repo'],
               'revision': pin['model_revision'], 'revision_role': 'released base checkpoint',
               'training_checkpoint_sha256': sha256(checkpoint_path),
               'objective': contract['objective'], 'epoch': state['epoch'],
               'training_contract': contract, 'export_script_sha256': sha256(Path(__file__)),
               'merge': 'Cast base QKV to training bfloat16, add float32 up@down, round to bfloat16.',
               'rounding_note': 'A merged bfloat16 matrix can differ slightly from separate LoRA operations.'}
    marker = output / 'medmeasure-provenance.json'
    if marker.exists():
        existing = json.loads(marker.read_text())
        if any(existing.get(key) != value for key, value in receipt.items()):
            raise ValueError('Existing export belongs to a different checkpoint or exporter')
        for name, expected in existing['files_sha256'].items():
            if sha256(output / name) != expected:
                raise ValueError(f'Existing export file changed: {name}')
        print(f'Existing export verified: {output}', flush=True)
        return
    plan = output / '.medmeasure-export-plan.json'
    if output.exists():
        if not plan.exists() or json.loads(plan.read_text()) != receipt:
            raise FileExistsError(f'Export directory belongs to another operation: {output}')
        print(f'Repeating incomplete export: {output}', flush=True)
    else:
        output.mkdir(parents=True)
        write_json(plan, receipt)

    # Keep the original processor defaults. V2 training's image budget is absent.
    for source in base.iterdir():
        if source.is_file() and source.suffix in ('.json', '.txt', '.jinja') and not source.name.startswith('.'):
            if source.name != 'medmeasure-provenance.json':
                shutil.copy2(source, output / source.name)
    total_size = index.get('metadata', {}).get('total_size', 0)
    for shard in sorted(set(index['weight_map'].values())):
        source = base / shard
        keys = [name for name in updates if index['weight_map'][name] == shard]
        if not keys:
            shutil.copy2(source, output / shard)
        else:
            with safe_open(source, framework='pt', device='cpu') as handle:
                tensors = {name: handle.get_tensor(name) for name in handle.keys()}
                metadata = handle.metadata()
            for name in keys:
                old = tensors[name]
                pair = updates[name]
                tensors[name] = merged_weight(old, pair['down'], pair['up'], dtype)
                total_size += tensors[name].numel() * tensors[name].element_size() - old.numel() * old.element_size()
            save_file(tensors, str(output / shard), metadata=metadata)
            del tensors
        print(f'Exported {shard}', flush=True)
    index.setdefault('metadata', {})['total_size'] = total_size
    write_json(output / 'model.safetensors.index.json', index)
    receipt['files_sha256'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path != plan}
    write_json(marker, receipt)
    plan.unlink(missing_ok=True)
    print(f'Export complete: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
