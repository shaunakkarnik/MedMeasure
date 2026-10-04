"""Split a frozen test manifest into disjoint CPU-prepared index lists for later GPU jobs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medmeasure.data import sha256, write_json


def plan_shards(records, shard_size):
    if not records or shard_size < 1:
        raise ValueError('Need a nonempty manifest and positive shard size')
    indices = [r['loader_index'] for r in records]
    if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Expected unique, nonnegative integer loader indices')
    identities = [r['sample_id'] for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate sample identities')
    if any(r['official_split'] != 'test' for r in records):
        raise ValueError('Evaluation shards must come from the official test partition')
    return [indices[start:start+shard_size] for start in range(0,len(indices),shard_size)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-size', type=int, default=100, help='Rows per job; tune after a timed GPU smoke run')
    args = p.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    shards = plan_shards(records, args.shard_size)
    args.output.mkdir(parents=True,exist_ok=False)
    for number,indices in enumerate(shards):
        write_json(args.output/f'shard-{number:03d}.json',indices)
    write_json(args.output/'plan.json',{
        'manifest_sha256':sha256(args.manifest),'total_rows':len(records),
        'shards':len(shards),'max_rows_per_shard':args.shard_size,
        'case_count':len({r['case_id'] for r in records}),
        'size_bins':dict(Counter('<10mm' if r['major_mm']<10 else '10-to-20mm' if r['major_mm']<20 else '>=20mm' for r in records)),
        'note':'All manifest rows included exactly once; job boundaries are not statistical partitions.'})
    print(f'Prepared {len(shards)} shards covering {len(records)} rows; no jobs submitted.')


if __name__=='__main__':
    main()
