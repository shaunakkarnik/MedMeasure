"""Pair official baseline outputs with Version 1 rows without hiding failures."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_baseline import checked_samples, key, read_rows


def mean(values):
    return statistics.mean(values) if values else None


def summarize(records):
    paired = [r for r in records if r['baseline_success'] and r['version1_status'] == 'success']
    case_errors = {}
    for row in paired:
        case_errors.setdefault(row['case_id'], []).append(row)
    return {
        'total': len(records), 'cases': len({r['case_id'] for r in records}),
        'baseline_successes': sum(r['baseline_success'] for r in records),
        'version1_successes': sum(r['version1_status'] == 'success' for r in records),
        'baseline_below10_all_rows': mean([r['baseline_below10'] for r in records]),
        'version1_below10_all_rows': mean([r['version1_below10'] for r in records]),
        'below10_contingency': {
            'both': sum(r['baseline_below10'] and r['version1_below10'] for r in records),
            'baseline_only': sum(r['baseline_below10'] and not r['version1_below10'] for r in records),
            'version1_only': sum(not r['baseline_below10'] and r['version1_below10'] for r in records),
            'neither': sum(not r['baseline_below10'] and not r['version1_below10'] for r in records)},
        'both_successful': {
            'total': len(paired),
            'baseline_mae_mm': mean([r['baseline_mae_mm'] for r in paired]),
            'version1_mae_mm': mean([r['version1_mae_mm'] for r in paired]),
            'baseline_mre': mean([r['baseline_mre'] for r in paired]),
            'version1_mre': mean([r['version1_mre'] for r in paired]),
            'version1_lower_mre': sum(r['version1_mre'] < r['baseline_mre'] for r in paired),
            'version1_higher_mre': sum(r['version1_mre'] > r['baseline_mre'] for r in paired),
            'equal_mre': sum(r['version1_mre'] == r['baseline_mre'] for r in paired),
            'case_macro_baseline_mre': mean([mean([r['baseline_mre'] for r in rows]) for rows in case_errors.values()]),
            'case_macro_version1_mre': mean([mean([r['version1_mre'] for r in rows]) for rows in case_errors.values()]),
            'cases_with_paired_success': len(case_errors)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version1-run', type=Path, required=True)
    parser.add_argument('--baseline-ref', default='1866d3c3c1b1ad61d021e8f1e117004dad98e752')
    parser.add_argument('--baseline-path', default='results/kits23-medvision-v0-v140-rik')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def baseline_file(name):
        return subprocess.check_output(['git', 'show', f'{args.baseline_ref}:{args.baseline_path}/{name}'],
                                       cwd=root, text=True)

    selected = read_rows(args.version1_run / 'selected_rows.jsonl')
    manifest_digest = hashlib.sha256((args.version1_run / 'selected_rows.jsonl').read_bytes()).hexdigest()
    provenance = json.loads(baseline_file('evaluation.json'))
    if manifest_digest != provenance['test_manifest_sha256']:
        raise ValueError('Version 1 selected rows differ from the baseline official manifest')
    predictions = read_rows(args.version1_run / 'predictions.jsonl')
    by_id = {r['sample_id']: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['sample_id'] for r in selected}:
        raise ValueError('Version 1 predictions have duplicate/missing/extra sample identities')
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', args.baseline_ref,
                                    '--', args.baseline_path], cwd=root, text=True).splitlines()
    logs = [p for p in paths if '_samples_' in p and p.endswith('.jsonl') and '/results/' in p]
    if not logs:
        raise ValueError('No baseline sample logs found')
    samples = []
    for path in logs:
        samples.extend(json.loads(line) for line in baseline_file(path[len(args.baseline_path) + 1:]).splitlines() if line.strip())
    checked_samples(samples, selected)
    by_key = {key(s['doc']): s for s in samples}
    diagnostic = {r['sample_id'] for r in [json.loads(line) for line in
                  baseline_file('test_legacy_disjoint.jsonl').splitlines() if line.strip()]}
    records = []
    for row in selected:
        current, previous = by_id[row['sample_id']], by_key[key(row)]
        profile = previous['doc']['biometric_profile']
        reference = {'major_mm': profile['metric_value_major_axis'][0],
                     'minor_mm': profile['metric_value_minor_axis'][0]}
        if current['reference'] != reference or current['case_id'] != row['case_id']:
            raise ValueError(f"Reference/identity mismatch: {row['sample_id']}")
        valid = (previous['avgMAE']['success'] and previous['avgMRE']['success']
                 and math.isfinite(previous['avgMAE']['MAE']) and math.isfinite(previous['avgMRE']['MRE']))
        if current['status'] == 'success' and not all(
                isinstance(current[k], (int, float)) and math.isfinite(current[k]) for k in ('mae_mm', 'mre')):
            raise ValueError(f"Nonfinite successful Version 1 measurement: {row['sample_id']}")
        response = previous['filtered_resps'][0]
        answers = re.findall(r'<answer>(.*?)</answer>', response, flags=re.DOTALL)
        measurement = current['measurement'] or {}
        records.append({
            'sample_id': row['sample_id'], 'case_id': row['case_id'], 'slice_idx': row['slice_idx'],
            'legacy_disjoint': row['sample_id'] in diagnostic,
            'reference_major_mm': reference['major_mm'], 'reference_minor_mm': reference['minor_mm'],
            'size_bin': '<10mm' if row['major_mm'] < 10 else '10-to-20mm' if row['major_mm'] < 20 else '>=20mm',
            'baseline_answer_text': answers[-1].strip() if answers else response,
            'baseline_success': bool(valid),
            'baseline_mae_mm': previous['avgMAE']['MAE'] if valid else None,
            'baseline_mre': previous['avgMRE']['MRE'] if valid else None,
            'version1_status': current['status'],
            'version1_major_mm': measurement.get('major_mm'), 'version1_minor_mm': measurement.get('minor_mm'),
            'version1_mae_mm': current['mae_mm'], 'version1_mre': current['mre'],
            'version1_dice': current['dice'],
            'baseline_below10': bool(valid and previous['avgMRE']['MRE'] < .1),
            'version1_below10': bool(current['status'] == 'success' and current['mre'] < .1),
            'mask_file': str(args.version1_run.resolve() / current['mask_file']),
            'overlay_file': str(args.version1_run.resolve() / Path(current['mask_file']).name)
                if (args.version1_run / Path(current['mask_file']).name).exists() else None,
            'baseline_response': response})
    report = {'manifest_sha256': manifest_digest, 'baseline_ref': args.baseline_ref,
              'baseline_shard_logs': len(logs), 'groups': {'all': summarize(records),
                'legacy_disjoint': summarize([r for r in records if r['legacy_disjoint']])},
              'notes': ['MAE/MRE are upstream baseline scores, not re-parsed answers.',
                        'Answer tags are extracted for display only.',
                        'Both-successful metrics use identical rows; excluded failures remain in all-row accuracy.',
                        'Case macro means weight included cases equally; no statistical uncertainty estimated.']}
    for size in ('<10mm', '10-to-20mm', '>=20mm'):
        report['groups'][size] = summarize([r for r in records if r['size_bin'] == size])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (args.output / 'paired.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    columns = [name for name in records[0] if name != 'baseline_response']
    with (args.output / 'paired.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report['groups']['all'], indent=2))


if __name__ == '__main__':
    main()
