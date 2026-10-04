"""Build case-separated manifests and audit annotation-version split changes (CPU only)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medmeasure.data import ANNOTATION_VERSION, DATASET_REVISION, TASK, case_ids, read_plan, rows, sha256, write_json, write_rows


def summarize(records):
    sizes = sorted(row['major_mm'] for row in records)
    bins = Counter('<10 mm' if x < 10 else '10–<20 mm' if x < 20 else '>=20 mm' for x in sizes)
    return {
        'rows': len(records), 'cases_with_rows': len({r['case_id'] for r in records}),
        'major_mm_min_median_max': [sizes[0], sizes[len(sizes)//2], sizes[-1]] if sizes else [],
        'slice_size_bins_not_clinical_categories': dict(bins),
        'cases_with_a_slice_below_10mm': len({r['case_id'] for r in records if r['major_mm'] < 10}),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True, help='KiTS23 benchmark_plan_biometry_v1.4.0.json.gz')
    p.add_argument('--legacy-plan', type=Path, action='append', required=True, help='Repeat for v1.0.0 and v1.1.1 plans')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=.2)
    args = p.parse_args()
    if not 0 < args.validation_fraction < 1:
        p.error('validation fraction must be between zero and one')
    if args.plan.name != 'benchmark_plan_biometry_v1.4.0.json.gz':
        p.error('This adapter supports the pinned v1.4.0 plan only')
    lock = json.loads((Path(__file__).resolve().parents[1] / 'configs/kits23-pilot.json').read_text())
    for path in [args.plan, *args.legacy_plan]:
        if sha256(path) != lock['sha256'].get(path.name):
            raise ValueError(f'Plan does not match pinned metadata: {path}')
    task = read_plan(args.plan)
    train_ids, test_ids = case_ids(task, 'train'), case_ids(task, 'test')
    if train_ids & test_ids:
        raise ValueError('Official train/test case overlap')
    legacy_train = set()
    comparisons = {}
    for path in args.legacy_plan:
        old = read_plan(path)
        old_train = case_ids(old, 'train')
        legacy_train |= old_train
        comparisons[path.name] = {
            'sha256': sha256(path),
            'new_test_in_legacy_train': sorted(test_ids & old_train),
            'new_train_in_legacy_test': sorted(train_ids & case_ids(old, 'test')),
        }
    # Hash ordering makes the partition independent of plan ordering and Python RNG versions.
    ordered = sorted(train_ids, key=lambda c: hashlib.sha256(f'{args.seed}:{c}'.encode()).hexdigest())
    nval = max(1, min(len(ordered)-1, round(len(ordered)*args.validation_fraction)))
    val_ids = set(ordered[:nval])
    train, test = rows(task, 'train'), rows(task, 'test')
    groups = {
        'train': [r for r in train if r['case_id'] not in val_ids],
        'validation': [r for r in train if r['case_id'] in val_ids],
        'test': test,
        'test_legacy_disjoint': [r for r in test if r['case_id'] not in legacy_train],
    }
    all_ids = [r['sample_id'] for r in train + test]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Duplicate slice-target IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in groups.items():
        write_rows(args.output / f'{name}.jsonl', records)
    write_json(args.output / 'test_indices.json', [r['loader_index'] for r in test])
    write_json(args.output / 'test_legacy_disjoint_indices.json', [r['loader_index'] for r in groups['test_legacy_disjoint']])
    write_json(args.output / 'case_splits.json', {
        'train': sorted(train_ids-val_ids), 'validation': sorted(val_ids), 'test': sorted(test_ids),
    })
    report = {
        'annotation_version': ANNOTATION_VERSION, 'dataset_revision_expected': DATASET_REVISION,
        'task': TASK, 'plan_sha256': sha256(args.plan), 'seed': args.seed,
        'validation_fraction': args.validation_fraction,
        'official_case_counts': {'train': len(train_ids), 'test': len(test_ids)},
        'groups': {k: summarize(v) for k,v in groups.items()}, 'legacy_split_comparisons': comparisons,
        'limitations': [
            'Metadata only: images, masks and checkpoint training exposure have not been verified.',
            'Legacy disjoint means absent from supplied legacy training plans, not proven unseen by the model.',
            'Small axial cross-sections do not necessarily mean small 3D tumors.',
            'case_ID is the source case identity; external patient linkage remains to be checked.',
        ],
    }
    write_json(args.output / 'audit.json', report)
    print({name: summarize(records) for name,records in groups.items()})


if __name__ == '__main__':
    main()
