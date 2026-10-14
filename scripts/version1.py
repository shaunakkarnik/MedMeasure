"""Train/evaluate the frozen-backbone Version 1 KiTS23 specialist."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate', 'cache-features'):
        command = sub.add_parser(name)
        command.add_argument('--data-root', type=Path, required=True,
                             help='Prepared data root containing metadata/ and Datasets/KiTS23/')
        command.add_argument('--manifests', type=Path, required=True, help='Audited split directory')
        command.add_argument('--output', type=Path, required=True, help='Run directory, or resumable feature-cache directory')
        command.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
        command.add_argument('--dtype', default='bfloat16', choices=('bfloat16', 'float32'))
        command.add_argument('--local-files-only', action='store_true')
        command.add_argument('--overlays', type=int, default=20)
        if name in ('train', 'cache-features'):
            command.add_argument('--config', type=Path, default=ROOT / 'configs/version1-kits23.json')
        if name == 'train':
            command.add_argument('--epochs', type=int, default=10)
            command.add_argument('--overfit-rows', type=int, default=0,
                                 help='Cache this many training rows and repeat; diagnostic only')
            command.add_argument('--resume', type=Path, help='Continue from last.pt in a new output directory')
            command.add_argument('--feature-cache', type=Path,
                                 help='Complete frozen train/validation cache; skips backbone loading')
        elif name == 'cache-features':
            command.add_argument('--max-rows', type=int, default=0,
                                 help='Extract at most this many new rows; zero means all remaining rows')
        else:
            command.add_argument('--checkpoint', type=Path, required=True)
            command.add_argument('--limit', type=int, default=0, help='Nonzero makes a smoke run, not a full score')
    return p


def load_backbone_config(pin, local_files_only):
    from transformers import Qwen2_5_VLConfig

    config = Qwen2_5_VLConfig.from_pretrained(
        pin['model_repo'], revision=pin['model_revision'], local_files_only=local_files_only)
    text_config = getattr(config, 'text_config', None)
    if isinstance(text_config, dict):
        # The released 4.56.1 config duplicates text settings here. In 4.51.3,
        # Qwen uses top-level settings, but generation setup mistakes this dict
        # for a config object. Preserve those settings and remove only the copy.
        for name in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                     'num_attention_heads', 'num_key_value_heads', 'hidden_act',
                     'rms_norm_eps', 'rope_theta', 'rope_scaling', 'max_position_embeddings'):
            if name in text_config and text_config[name] != getattr(config, name):
                raise ValueError(f'Nested text_config differs from backbone config: {name}')
        del config.text_config
    return config


def load_model(config, pin, args):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from medmeasure.version1 import FrozenMedVisionFeatures

    if transformers.__version__ != '4.51.3':
        raise ValueError('Version 1 adapter requires transformers==4.51.3')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu --dtype float32 for small checks')
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(
        pin['model_repo'], revision=pin['model_revision'], local_files_only=args.local_files_only,
        min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pin['model_repo'], revision=pin['model_revision'], torch_dtype=dtype,
        config=load_backbone_config(pin, args.local_files_only),
        attn_implementation='sdpa', local_files_only=args.local_files_only).to(args.device)
    features = FrozenMedVisionFeatures(backbone, config['vision_layers'])
    decoder = load_decoder(backbone.config.vision_config.hidden_size,
                           backbone.config.hidden_size, config, args.device)
    return processor, features, decoder


def load_decoder(vision_dim, language_dim, config, device):
    from medmeasure.version1 import TargetMaskDecoder
    return TargetMaskDecoder(vision_dim, language_dim, len(config['vision_layers']),
                             config['decoder_dim'], config['attention_heads']).to(device)


def sample(row, dataset_dir, processor, features, config, device):
    from medmeasure.version1_data import load_image, load_target, processor_inputs
    import torch
    image, spacing, volume = load_image(row, dataset_dir)
    maps, query = features(processor_inputs(processor, image, config['question'], device))
    target = load_target(row, dataset_dir, volume)
    tensor = torch.from_numpy(target.copy()).float()[None, None].to(device)
    return maps, query, tensor, image, spacing, target


def train(args, config, pin, manifests, case_splits, metadata, checkpoint):
    import torch
    from medmeasure.data import write_json
    from medmeasure.version1 import segmentation_loss
    from medmeasure.version1_data import save_overlay

    disk_cache = None
    if getattr(args, 'feature_cache', None):
        from medmeasure.version1_cache import FeatureCache, contract
        disk_cache = FeatureCache(args.feature_cache,
            contract(config, pin, manifests, args.data_root, args.dtype))
        dimensions = disk_cache.complete['dimensions']
        decoder = load_decoder(dimensions['vision_dim'], dimensions['language_dim'], config, args.device)
        metadata['feature_cache'] = {'contract_sha256': disk_cache.identity,
                                    'directory': str(args.feature_cache.resolve()),
                                    'bytes': disk_cache.complete['bytes']}
    else:
        processor, features, decoder = load_model(config, pin, args)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config['learning_rate'])
    start, best = 0, float('inf')
    if checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best = checkpoint['epoch'], checkpoint['best_selection_loss']
        random.setstate(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng_state']])
        prior_best_path = args.resume.parent / 'best.pt'
        prior_best = torch.load(prior_best_path, map_location='cpu', weights_only=True)
        if (prior_best['config'] != config or prior_best['pin'] != pin
                or prior_best['best_selection_loss'] != best
                or prior_best['metadata']['manifest_sha256'] != metadata['manifest_sha256']):
            raise ValueError('Resume requires the matching best.pt beside the input checkpoint')
        del prior_best
    if start >= args.epochs:
        raise ValueError('--epochs must exceed the resumed checkpoint epoch')
    dataset_dir = args.data_root / 'Datasets/KiTS23'
    train_rows = manifests['train'][:args.overfit_rows] if args.overfit_rows else manifests['train']
    cache = {}
    if args.overfit_rows and disk_cache is None:
        for row in train_rows:
            item = sample(row, dataset_dir, processor, features, config, args.device)
            cache[row['sample_id']] = ([x.cpu() for x in item[0]], item[1].cpu(),
                                       item[2].cpu(), *item[3:])
        print(f'Cached {len(cache)} frozen image/question feature sets for the overfit diagnostic.', flush=True)
    def get(row):
        if disk_cache is not None:
            return disk_cache.read(row)
        if row['sample_id'] in cache:
            return cache[row['sample_id']]
        return sample(row, dataset_dir, processor, features, config, args.device)

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'run.json', metadata)
    if checkpoint:
        shutil.copy2(prior_best_path, args.output / 'best.pt')
    for epoch in range(start, args.epochs):
        decoder.train()
        order = list(train_rows)
        random.shuffle(order)
        training_loss, training_dice = 0., 0.
        for index, row in enumerate(order):
            maps, query, target, image, spacing, mask = get(row)
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(maps, query, mask.shape)
            loss = segmentation_loss(logits, target)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {row['sample_id']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
            optimizer.step()
            training_loss += loss.item()
            prediction = logits.detach().sigmoid()[0, 0].cpu().numpy() >= config['threshold']
            training_dice += 2 * (prediction & mask).sum() / max(1, prediction.sum() + mask.sum())
            if index % 50 == 0:
                print(f'Epoch {epoch+1}, row {index+1}/{len(order)}, loss {loss.item():.4f}', flush=True)
            if index < args.overlays and epoch == args.epochs - 1:
                save_overlay(args.output / f'train_{index:03d}.png', image, prediction, mask)
        decoder.eval()
        selection_loss = 0.
        selection_rows = train_rows if args.overfit_rows else manifests['validation']
        with torch.no_grad():
            for row in selection_rows:
                maps, query, target, _, _, mask = get(row)
                selection_loss += segmentation_loss(decoder(maps, query, mask.shape), target).item()
        selection_loss /= len(selection_rows)
        if not torch.isfinite(torch.tensor(selection_loss)):
            raise ValueError('Nonfinite checkpoint-selection loss')
        improved = selection_loss < best
        best = min(best, selection_loss)
        state = {'format_version': 1, 'config': config, 'pin': pin, 'metadata': metadata,
                 'case_splits': case_splits, 'decoder': decoder.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                 'best_selection_loss': best, 'random_state': random.getstate(),
                 'torch_rng_state': torch.get_rng_state(),
                 'cuda_rng_state': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}
        temporary = args.output / 'last.pt.tmp'
        torch.save(state, temporary)
        temporary.replace(args.output / 'last.pt')
        if improved:
            torch.save(state, args.output / 'best.pt')
        report = {'epoch': epoch + 1, 'train_loss': training_loss / len(order),
                  'train_dice_online': float(training_dice / len(order)),
                  'validation_loss': None if args.overfit_rows else selection_loss,
                  'overfit_loss': selection_loss if args.overfit_rows else None}
        with open(args.output / 'epochs.jsonl', 'a') as stream:
            stream.write(json.dumps(report, allow_nan=False) + '\n')
        print(report, flush=True)


def evaluate(args, config, pin, manifests, metadata, checkpoint):
    import torch
    import numpy as np
    from PIL import Image
    from medmeasure.data import write_json, write_rows
    from medmeasure.version1_data import (load_image, load_target, processor_inputs,
        measure_prediction, score_prediction, summarize_predictions, save_overlay)

    test_rows = manifests['test']
    seen = set(checkpoint['case_splits']['train']) | set(checkpoint['case_splits']['validation'])
    if seen & {r['case_id'] for r in test_rows}:
        raise ValueError('Evaluation cases were used for decoder training or validation')
    disjoint = None
    if not args.limit:
        disjoint = read_rows(args.manifests / 'test_legacy_disjoint.jsonl')
        official = {r['sample_id']: r for r in test_rows}
        if any(official.get(r['sample_id']) != r for r in disjoint):
            raise ValueError('Diagnostic rows must be a subset of the complete official test')
    processor, features, decoder = load_model(config, pin, args)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    selected = test_rows[:args.limit] if args.limit else test_rows
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'masks').mkdir()
    write_json(args.output / 'run.json', metadata)
    results = []
    with torch.no_grad(), open(args.output / 'predictions.jsonl', 'w') as stream:
        for index, row in enumerate(selected):
            image, spacing, volume = load_image(row, args.data_root / 'Datasets/KiTS23')
            maps, query = features(processor_inputs(processor, image, config['question'], args.device))
            logits = decoder(maps, query, (image.height, image.width))
            finite_logits = bool(torch.isfinite(logits).all())
            predicted = (logits.sigmoid()[0, 0].cpu().numpy() >= config['threshold'] if finite_logits
                         else np.zeros((image.height, image.width), dtype=bool))
            # Only after prediction do we load the target mask for scoring.
            target = load_target(row, args.data_root / 'Datasets/KiTS23', volume)
            measurement = measure_prediction(predicted, spacing)
            if not finite_logits:
                measurement['status'] = 'nonfinite_logits'
            result = score_prediction(row, measurement, predicted, target)
            filename = f"{row['case_id']}_{row['slice_idx']}.png"
            Image.fromarray(predicted.astype(np.uint8) * 255).save(args.output / 'masks' / filename)
            result['mask_file'] = 'masks/' + filename
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            if index < args.overlays:
                save_overlay(args.output / filename, image, predicted, target)
            if index % 25 == 0:
                print(f'Evaluated {index+1}/{len(selected)}', flush=True)
    report = {'complete_official_test': len(selected) == len(test_rows),
              'groups': {'all': summarize_predictions(results, selected)},
              'definition': 'Per-row mean relative error over major/minor ellipse axes; strict <0.1.',
              'reference_precision': 'HF float16 labels for matched baseline scoring; full planner errors also saved.',
              'checkpoint_exposure_note': 'Released backbone may have seen 104 official test cases; see baseline audit.'}
    if not args.limit:
        ids = {r['sample_id'] for r in disjoint}
        report['groups']['legacy_disjoint'] = summarize_predictions(
            [r for r in results if r['sample_id'] in ids], disjoint)
    write_rows(args.output / 'selected_rows.jsonl', selected)
    write_json(args.output / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


def read_rows(path):
    from medmeasure.version1_data import read_manifest
    return read_manifest(path)


def main():
    import torch
    from medmeasure.data import sha256
    from medmeasure.version1_data import validate_manifests
    p = parser()
    args = p.parse_args()
    if args.overlays < 0 or (args.command == 'train' and (args.epochs < 1 or args.overfit_rows < 0)):
        p.error('Epochs must be positive and row/overlay limits nonnegative')
    if args.command == 'evaluate' and args.limit < 0:
        p.error('Limit must be nonnegative')
    if args.command == 'cache-features' and args.max_rows < 0:
        p.error('Cache row limit must be nonnegative')
    pin = json.loads((ROOT / 'configs/kits23-pilot.json').read_text())
    path = (args.resume if args.command == 'train' else
            args.checkpoint if args.command == 'evaluate' else None)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True) if path else None
    if checkpoint and (checkpoint['format_version'] != 1 or checkpoint['pin'] != pin):
        raise ValueError('Checkpoint format or dataset/model pins differ')
    config = (checkpoint['config'] if args.command == 'evaluate' else json.loads(args.config.read_text()))
    if checkpoint and checkpoint['config'] != config:
        raise ValueError('Resume configuration differs from checkpoint')
    if not 0 < config['threshold'] < 1:
        raise ValueError('Mask threshold must be between zero and one')
    manifests = {name: read_rows(args.manifests / f'{name}.jsonl')
                 for name in ('train', 'validation', 'test')}
    case_splits = validate_manifests(manifests,
        args.data_root / 'metadata/benchmark_plan_biometry_v1.4.0.json.gz', pin)
    # Evaluation must cover the whole official test selection unless explicitly a smoke run.
    from medmeasure.data import read_plan, rows
    official_test = rows(read_plan(args.data_root / 'metadata/benchmark_plan_biometry_v1.4.0.json.gz'), 'test')
    if manifests['test'] != official_test:
        raise ValueError('test.jsonl must contain the complete ordered official selection')
    manifest_hashes = {name: sha256(args.manifests / f'{name}.jsonl') for name in manifests}
    if args.command == 'cache-features':
        from medmeasure.version1_cache import build_cache
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        build_cache(args, config, pin, manifests, load_model, sample)
        return
    if args.command == 'train' and checkpoint:
        if checkpoint['metadata']['manifest_sha256'] != manifest_hashes:
            raise ValueError('Resume manifests differ from checkpoint')
        if checkpoint['metadata']['overfit_rows'] != args.overfit_rows:
            raise ValueError('Resume overfit row limit differs from checkpoint')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    overfit_rows = args.overfit_rows if args.command == 'train' else checkpoint['metadata']['overfit_rows']
    metadata = {'config': config, 'pin': pin, 'manifest_sha256': manifest_hashes,
                'source_sha256': {name: sha256(ROOT / name) for name in
                    ('scripts/version1.py', 'medmeasure/version1.py', 'medmeasure/version1_data.py',
                     'medmeasure/version1_cache.py',
                     'medmeasure/geometry.py', 'requirements-v1.txt')},
                'case_splits': case_splits, 'torch': str(torch.__version__),
                'device': args.device, 'dtype': args.dtype,
                'preprocessing': 'Native axial array, HU [-160,240], RGB; whole-image processor resize; no overlays/crops.',
                'target_state': 'Final nonpadding prompt token after assistant generation prefix; no answer tokens.',
                'overfit_rows': overfit_rows,
                'checkpoint_selection': 'training loss (overfit diagnostic)' if overfit_rows
                    else 'held-out validation loss'}
    if path:
        metadata['checkpoint_sha256'] = sha256(path)
    if args.command == 'train':
        train(args, config, pin, manifests, case_splits, metadata, checkpoint)
    else:
        evaluate(args, config, pin, manifests, metadata, checkpoint)


if __name__ == '__main__':
    main()
