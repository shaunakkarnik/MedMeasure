"""Download a revision-pinned baseline checkpoint; does not load it or use a GPU."""
import argparse
import json
from pathlib import Path


def main():
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/kits23-pilot.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision', default=lock['model_revision'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        p.error('Use a full lowercase HF model commit SHA')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use an empty model directory to avoid mixing checkpoint revisions')
    from huggingface_hub import snapshot_download
    snapshot_download('YongchengYAO/MedVision-V0-7B', revision=args.revision, local_dir=args.output)
    (args.output/'medmeasure-provenance.json').write_text(json.dumps({
        'repo':'YongchengYAO/MedVision-V0-7B','revision':args.revision},indent=2)+'\n')


if __name__ == '__main__':
    main()
