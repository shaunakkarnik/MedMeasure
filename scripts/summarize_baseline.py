"""Summarize upstream per-sample T/L metrics with explicit failure denominators."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medmeasure.data import write_json


def summarize(items):
    valid = [s for s in items if s['avgMAE']['success'] and s['avgMRE']['success']
             and math.isfinite(s['avgMAE']['MAE']) and math.isfinite(s['avgMRE']['MRE'])]
    below = sum(s['avgMRE']['MRE'] < .1 for s in valid)
    return {'total':len(items), 'finite_successes':len(valid), 'failures':len(items)-len(valid),
            'mae_mm_success_only':statistics.mean(s['avgMAE']['MAE'] for s in valid) if valid else None,
            'mre_fraction_success_only':statistics.mean(s['avgMRE']['MRE'] for s in valid) if valid else None,
            'success_fraction':len(valid)/len(items) if items else None,
            'below_10_percent_fraction_all_rows':below/len(items) if items else None,
            'below_10_percent_fraction_success_only':below/len(valid) if valid else None}


def key(row):
    return (Path(row['image_file']).name, row['slice_dim'], row['slice_idx'], row['label'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=Path, required=True, action='append', help='Repeat for disjoint shard samples JSONL files')
    p.add_argument('--selected-rows', type=Path, required=True, help='preflight selected_rows.json or the full test JSONL manifest')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    text = args.selected_rows.read_text()
    selected = ([json.loads(line) for line in text.splitlines() if line.strip()]
                if args.selected_rows.suffix == '.jsonl' else json.loads(text))
    expected = {key(r):r for r in selected}
    if not selected or len(expected) != len(selected):
        raise ValueError('Selected-row manifest must be nonempty and contain unique identities')
    samples = [json.loads(line) for path in args.samples for line in path.read_text().splitlines() if line.strip()]
    actual = [key(s['doc']) for s in samples]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('Sample log has duplicate, missing or unexpected rows; refusing partial score')
    groups = {'all': samples}
    for name, low, high in [('<10mm',0,10),('10-to-20mm',10,20),('>=20mm',20,float('inf'))]:
        groups[name] = [s for s in samples if low <= expected[key(s['doc'])]['major_mm'] < high]
    report = {'groups':{k:summarize(v) for k,v in groups.items()},
              'definition':'Per-row MRE averages relative error over major/minor axes; threshold is strictly <0.1.',
              'note':'Uses upstream parsed metrics. Nonfinite scores count as failures here; keep upstream aggregate outputs too.'}
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, report)


if __name__ == '__main__':
    main()
