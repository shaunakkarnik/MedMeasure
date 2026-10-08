"""Download a revision-pinned baseline checkpoint; does not load it or use a GPU."""
import argparse
import json
from pathlib import Path


def main():
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/kits23-pilot.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision', default=lock['model_revision'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Resume only a directory marked with this exact repo/revision')
    args = p.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        p.error('Use a full lowercase HF model commit SHA')
    provenance = {'repo':'YongchengYAO/MedVision-V0-7B', 'revision':args.revision}
    pending = args.output/'medmeasure-download.json'
    complete = args.output/'medmeasure-provenance.json'
    if args.output.exists() and any(args.output.iterdir()):
        marker = complete if complete.exists() else pending
        if not args.resume or not marker.exists() or json.loads(marker.read_text()) != provenance:
            p.error('Use an empty directory, or --resume with matching pinned download provenance')
    args.output.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(provenance, indent=2)+'\n')
    from huggingface_hub import snapshot_download
    snapshot_download('YongchengYAO/MedVision-V0-7B', revision=args.revision, local_dir=args.output)
    complete.write_text(json.dumps(provenance, indent=2)+'\n')
    pending.unlink()


if __name__ == '__main__':
    main()
