"""Train and evaluate matched text-only and text-plus-mask KiTS23 models."""
import argparse
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    for name in ('data-root', 'manifests', 'output'):
        common.add_argument(f'--{name}', type=Path, required=True)
    common.add_argument('--local-files-only', action='store_true')
    common.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    common.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    common.add_argument('--limit', type=int, default=0, help='Rows per split for smoke runs')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', parents=[common])
    train.add_argument('--objective', choices=('text', 'joint'), required=True)
    train.add_argument('--epochs', type=int, default=3, help='Total epochs, including resumed epochs')
    train.add_argument('--resume', type=Path)
    evaluate = commands.add_parser('evaluate', parents=[common])
    evaluate.add_argument('--checkpoint', type=Path, help='Omit for the frozen MedVision control')
    evaluate.add_argument('--split', choices=('validation', 'test'), default='test')
    for command in (train, evaluate):
        command.add_argument('--config', type=Path, default=ROOT / 'configs/version2-kits23.json')
    args = parser.parse_args()
    if args.limit < 0 or (args.command == 'train' and args.epochs < 1):
        parser.error('Limit must be nonnegative and epochs must be positive')
    return args


def evaluate(backbone, processor, rows, dataset_dir, config, device, path):
    import torch
    from medmeasure.version1_data import load_image
    from medmeasure.version2 import score_answer, sizing_inputs
    from scripts.summarize_baseline import summarize

    results = []
    with torch.inference_mode(), path.open('w') as stream:
        for row in rows:
            image, spacing, _ = load_image(row, dataset_dir)
            prompt = sizing_inputs(processor, image, spacing, device)
            ids = backbone.generate(**prompt, do_sample=False, use_cache=True,
                                    max_new_tokens=config['max_new_tokens'])
            answer = processor.tokenizer.decode(
                ids[0, prompt['input_ids'].shape[1]:], skip_special_tokens=True)
            result = score_answer(row, answer)
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            results.append(result)
    return summarize(results)


def save_checkpoint(path, checkpoint):
    import torch

    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(args, backbone, processor, manifests, dataset_dir, contract, checkpoint):
    import torch
    from medmeasure.data import write_json
    from medmeasure.version1_data import load_image, load_target
    from medmeasure.version2 import sizing_inputs, training_losses
    from scripts.version1 import load_decoder

    config = contract['config']
    adapters = {name: param for name, param in backbone.named_parameters() if param.requires_grad}
    groups = [{'params': list(adapters.values()), 'lr': config['adapter_learning_rate']}]
    decoder = None
    if args.objective == 'joint':
        decoder = load_decoder(backbone.config.vision_config.hidden_size,
                               backbone.config.hidden_size, config, args.device)
        groups.append({'params': list(decoder.parameters()), 'lr': config['decoder_learning_rate']})
    optimizer = torch.optim.AdamW(groups)
    start, best = 0, None
    if checkpoint:
        start, best = checkpoint['epoch'], checkpoint['best_score']
        if decoder is not None:
            decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Carry the earlier best checkpoint into the new run directory.
        import shutil
        shutil.copyfile(args.resume.with_name('best.pt'), args.output / 'best.pt')

    for epoch in range(start + 1, args.epochs + 1):
        rows = list(manifests['train'])
        random.Random(config['seed'] + epoch).shuffle(rows)
        totals = [0., 0.]
        for step, row in enumerate(rows, 1):
            image, spacing, volume = load_image(row, dataset_dir)
            prompt = sizing_inputs(processor, image, spacing, args.device)
            target = None
            if decoder is not None:
                target = torch.from_numpy(load_target(row, dataset_dir, volume))[None, None]
            optimizer.zero_grad(set_to_none=True)
            answer, mask = training_losses(backbone, decoder, config['vision_layers'],
                                           prompt, processor.tokenizer, row, target)
            loss = answer if mask is None else answer + config['mask_weight'] * mask
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for group in groups for param in group['params']], 1., error_if_nonfinite=True)
            optimizer.step()
            totals[0] += answer.item()
            if mask is not None:
                totals[1] += mask.item()
            if step == 1 or step % 100 == 0:
                print(f'Epoch {epoch}, row {step}/{len(rows)}, mean loss {totals[0]/step:.4f}'
                      + (f' + mask {totals[1]/step:.4f}' if mask is not None else ''), flush=True)

        validation = evaluate(backbone, processor, manifests['validation'], dataset_dir,
                              config, args.device, args.output / f'validation-{epoch}.jsonl')
        # Maximize all-row within-10% accuracy, then coverage, then minimize MRE.
        mre = validation['mre_fraction_success_only']
        score = (validation['below_10_percent_fraction_all_rows'], validation['success_fraction'],
                 -mre if mre is not None else -float('inf'))
        improved = best is None or score > best
        if improved:
            best = score
        state = {'contract': contract, 'epoch': epoch, 'best_score': best,
                 'adapters': {name: param.detach().cpu() for name, param in adapters.items()},
                 'decoder': decoder.state_dict() if decoder is not None else None,
                 'optimizer': optimizer.state_dict()}
        if improved:
            save_checkpoint(args.output / 'best.pt', state)
        write_json(args.output / f'epoch-{epoch}.json',
                   {'epoch': epoch, 'text_loss': totals[0] / len(rows),
                    'mask_loss': totals[1] / len(rows) if decoder is not None else None,
                    'validation': validation, 'new_best': improved})
        save_checkpoint(args.output / 'last.pt', state)
        print(f'Epoch {epoch} validation: {validation}', flush=True)


def main():
    args = parse_args()
    import torch
    from medmeasure.data import read_plan, rows, sha256, write_json
    from medmeasure.version1_data import read_manifest, validate_manifests
    from medmeasure.version2 import add_vision_adapters
    from scripts.version1 import load_model

    path = args.resume if args.command == 'train' else args.checkpoint
    checkpoint = torch.load(path, map_location='cpu', weights_only=True) if path else None
    config = (checkpoint['contract']['config'] if checkpoint and args.command == 'evaluate'
              else json.loads(args.config.read_text()))
    pin = json.loads((ROOT / 'configs/kits23-pilot.json').read_text())
    plan = args.data_root / 'metadata/benchmark_plan_biometry_v1.4.0.json.gz'
    files = {name: args.manifests / f'{name}.jsonl' for name in ('train', 'validation', 'test')}
    manifests = {name: read_manifest(file) for name, file in files.items()}
    validate_manifests(manifests, plan, pin)
    if manifests['test'] != rows(read_plan(plan), 'test'):
        raise ValueError('test.jsonl must be the full ordered official test split')
    contract = {'format_version': 1, 'config': config, 'pin': pin,
                'manifest_sha256': {name: sha256(file) for name, file in files.items()},
                'source_sha256': {name: sha256(ROOT / name) for name in (
                    'scripts/version2.py', 'scripts/version1.py', 'scripts/summarize_baseline.py',
                    'medmeasure/version2.py', 'medmeasure/version1.py', 'medmeasure/version1_data.py',
                    'configs/version2-prompts.json')},
                'torch': str(torch.__version__), 'dtype': args.dtype}
    if args.command == 'train':
        contract.update(objective=args.objective, train_limit=args.limit)
        if checkpoint and (contract != checkpoint['contract'] or args.epochs <= checkpoint['epoch']):
            raise ValueError('Resume requires the same training contract and more epochs')
    elif checkpoint:
        saved = checkpoint['contract']
        for name in ('format_version', 'pin', 'manifest_sha256', 'source_sha256'):
            if saved[name] != contract[name]:
                raise ValueError(f'Evaluation differs from checkpoint in {name}')
        if saved['train_limit'] and args.split == 'test':
            raise ValueError('Smoke checkpoints may only be evaluated on validation')
    if args.limit:
        manifests = {name: records[:args.limit] for name, records in manifests.items()}

    torch.manual_seed(config['seed'])
    processor, frozen_features, initial_decoder = load_model(config, pin, args)
    backbone = frozen_features.backbone
    del frozen_features, initial_decoder
    if args.command == 'train' or checkpoint:
        add_vision_adapters(backbone, config['first_adapter_block'], config['adapter_rank'])
    if checkpoint:
        adapters = {name: param for name, param in backbone.named_parameters() if param.requires_grad}
        if adapters.keys() != checkpoint['adapters'].keys():
            raise ValueError('Checkpoint adapter names differ from model')
        with torch.no_grad():
            for name, param in adapters.items():
                param.copy_(checkpoint['adapters'][name])

    args.output.mkdir(parents=True)
    write_json(args.output / 'run.json',
               {**contract, 'command': args.command, 'row_limit': args.limit,
                'device': args.device, 'checkpoint_sha256': sha256(path) if path else None,
                **({'split': args.split, 'training': checkpoint['contract'] if checkpoint else None}
                   if args.command == 'evaluate' else {})})
    dataset_dir = args.data_root / 'Datasets/KiTS23'
    if args.command == 'train':
        train(args, backbone, processor, manifests, dataset_dir, contract, checkpoint)
    else:
        summary = evaluate(backbone, processor, manifests[args.split], dataset_dir, config,
                           args.device, args.output / 'predictions.jsonl')
        write_json(args.output / 'summary.json', summary)
        print(summary)


if __name__ == '__main__':
    main()
