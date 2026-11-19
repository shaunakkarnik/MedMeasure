"""Compare frozen MedVision, V1, and image-first V2 on shared test rows."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medmeasure.data import sha256, write_json
from scripts.summarize_baseline import read_rows, checked_samples
from scripts.version2_original import reference_samples, check_protocol


def text_metrics(sample):
    valid = (sample['avgMAE']['success'] and sample['avgMRE']['success']
             and math.isfinite(sample['avgMAE']['MAE']) and math.isfinite(sample['avgMRE']['MRE']))
    return {'success': bool(valid),
            'mae_mm': float(sample['avgMAE']['MAE']) if valid else None,
            'mre': float(sample['avgMRE']['MRE']) if valid else None}


def summarize(rows):
    """Keep failures in threshold accuracy and share the paired error denominator."""
    common = [row for row in rows if all(value['success'] for value in row['methods'].values())]
    result = {'total': len(rows), 'all_methods_success_count': len(common), 'methods': {}}
    for method in rows[0]['methods']:
        values = [row['methods'][method] for row in rows]
        success = [value for value in values if value['success']]
        paired = [row['methods'][method] for row in common]
        result['methods'][method] = {
            'successes': len(success), 'failures': len(rows) - len(success),
            'within_10_percent_all_rows': sum(value['mre'] < .1 for value in success) / len(rows),
            'mae_mm_success_only': statistics.mean(value['mae_mm'] for value in success) if success else None,
            'mre_success_only': statistics.mean(value['mre'] for value in success) if success else None,
            'mae_mm_common_successes': statistics.mean(value['mae_mm'] for value in paired) if paired else None,
            'mre_common_successes': statistics.mean(value['mre'] for value in paired) if paired else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-run', type=Path, required=True)
    parser.add_argument('--joint-run', type=Path, required=True)
    parser.add_argument('--frozen-run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'results/kits23-medvision-v0-v140-rik')
    parser.add_argument('--version1-run', type=Path, default=ROOT / 'results/v1-eval-6252813')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.baseline / 'test.jsonl')
    baseline = reference_samples(args.baseline)
    checked_samples(list(baseline.values()), rows)
    v1_path = args.version1_run / 'predictions.jsonl'
    v1_records = read_rows(v1_path)
    v1 = {record['sample_id']: record for record in v1_records}
    if len(v1) != len(v1_records) or set(v1) != {row['sample_id'] for row in rows}:
        raise ValueError('V1 sample coverage differs from the official test selection')
    methods = {}
    protocols = []
    directories = [('frozen', args.frozen_run), ('text', args.text_run), ('joint', args.joint_run)]
    manifest_hash = sha256(args.baseline / 'test.jsonl')
    for method, directory in directories:
        receipt = json.loads((directory / 'provenance.json').read_text())
        if receipt['smoke'] or receipt['manifest_sha256'] != manifest_hash:
            raise ValueError('Expected complete aligned test predictions')
        provenance = receipt['model_provenance']
        if method == 'frozen':
            if provenance.get('kind', 'released') != 'released':
                raise ValueError('Frozen run must use the released checkpoint')
        elif provenance['objective'] != method:
            raise ValueError('Text and joint result directories are mislabeled')
        protocol = {name: receipt.get(name) for name in (
            'content_order', 'execution_script_sha256', 'input_adapter_sha256')}
        if protocol['content_order'] != 'image_first' or not all(protocol.values()):
            raise ValueError('Expected image-first results with recorded execution and adapter hashes')
        protocols.append(protocol)
        for relative, expected in receipt['sample_logs_sha256'].items():
            if sha256(directory / relative) != expected:
                raise ValueError('Source sample log changed since aggregation')
        samples = read_rows(directory / 'samples.jsonl')
        check_protocol(samples, rows, baseline)
        methods[method] = {sample['doc_id']: sample for sample in samples}
    if any(protocol != protocols[0] for protocol in protocols):
        raise ValueError('Text models must share one evaluation/content-order protocol')
    diagnostic = {row['sample_id'] for row in read_rows(args.baseline / 'test_legacy_disjoint.jsonl')}
    paired = []
    for row in rows:
        original = baseline[row['loader_index']]
        current = v1[row['sample_id']]
        profile = original['doc']['biometric_profile']
        reference = {axis: profile[field][0] for axis, field in (
            ('major_mm', 'metric_value_major_axis'), ('minor_mm', 'metric_value_minor_axis'))}
        if current['case_id'] != row['case_id'] or current['reference'] != reference:
            raise ValueError('V1 identities or measurement references differ')
        values = {method: text_metrics(samples[row['loader_index']]) for method, samples in methods.items()}
        valid = current['status'] == 'success'
        if valid:
            errors = [abs(current['measurement'][axis] - reference[axis]) for axis in reference]
            mae = statistics.mean(errors)
            mre = statistics.mean(error / reference[axis] for error, axis in zip(errors, reference))
            if not (math.isclose(mae, current['mae_mm'], rel_tol=1e-6, abs_tol=1e-6)
                    and math.isclose(mre, current['mre'], rel_tol=1e-6, abs_tol=1e-6)):
                raise ValueError('V1 saved errors do not reproduce from its measurements')
        values['version1'] = {'success': valid, 'mae_mm': current['mae_mm'] if valid else None,
                              'mre': current['mre'] if valid else None}
        paired.append({'sample_id': row['sample_id'], 'case_id': row['case_id'],
                       'legacy_disjoint': row['sample_id'] in diagnostic, 'methods': values})
    report = {'groups': {'all': summarize(paired), 'legacy_disjoint': summarize(
        [row for row in paired if row['legacy_disjoint']])},
        'text_evaluation_protocol': protocols[0],
        'notes': ['All-row threshold accuracy includes every failure.',
                  'Common-success MAE/MRE use the same rows for all four methods.',
                  'V1 uses mask geometry; text methods share the pinned evaluator and recorded content order.',
                  'V1 uses its trained 448-pixel processor budget; text models use benchmark preprocessing.'],
        'source_sha256': {'version1_predictions': sha256(v1_path),
                          'text_samples': sha256(args.text_run / 'samples.jsonl'),
                          'joint_samples': sha256(args.joint_run / 'samples.jsonl'),
                          'frozen_samples': sha256(args.frozen_run / 'samples.jsonl'),
                          'test_manifest': manifest_hash}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'summary.json', report)
    (args.output / 'paired.jsonl').write_text(
        ''.join(json.dumps(record, allow_nan=False) + '\n' for record in paired))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
