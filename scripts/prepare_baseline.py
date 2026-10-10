"""Prepare one MedVision baseline run without downloading data or running inference."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


REFERENCE_COMMIT = "45fa8e52684a65da6e1c0ef6ef422d5d4c3e94bd"


def main():
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/kits23-pilot.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--task", default="KiTS23_TumorLesionSize_Task01_Axial-CoT", help="Exact upstream axial T/L CoT task name")
    parser.add_argument("--annotation-version", choices=["1.0.0", "1.1.1", "1.4.0"], default="1.4.0")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Local HF checkpoint snapshot pinned to a revision")
    parser.add_argument("--output", type=Path, required=True, help="New run directory; existing directories are refused")
    parser.add_argument("--limit", type=int, default=10, help="Smoke-test limit; use 1000 for the upstream evaluation cap")
    parser.add_argument("--sample-indices", type=Path, help="Explicit loader indices from the data audit; overrides limit")
    parser.add_argument("--model-revision", default=lock["model_revision"], help="40-character HF checkpoint commit for the local snapshot")
    args = parser.parse_args()
    if len(args.model_revision) != 40 or any(c not in '0123456789abcdef' for c in args.model_revision):
        parser.error("--model-revision must be a full lowercase commit SHA")
    indices = None
    if args.sample_indices:
        indices = json.loads(args.sample_indices.read_text())
        if not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
            parser.error("Sample indices must be nonempty, unique nonnegative integers")
    if args.annotation_version != '1.4.0' or args.task != 'KiTS23_TumorLesionSize_Task01_Axial-CoT':
        parser.error('This executable pilot supports KiTS23 axial T/L v1.4.0 only')
    if ',' in str(args.model_path):
        parser.error('Model paths cannot contain commas (upstream model-argument format)')
    if args.limit < 1:
        parser.error("--limit must be positive")
    loader = args.data_dir.resolve()/'MedVision.py'
    loader_hash = None
    if loader.exists():
        loader_hash = hashlib.sha256(loader.read_bytes()).hexdigest()
        if loader_hash != lock['loader_source_sha256']:
            parser.error('Local dataset loader differs from the pinned revision')

    upstream = args.upstream.resolve()
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT:
        parser.error(f"Expected inspected upstream commit {REFERENCE_COMMIT}, got {commit}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        parser.error("Use an unchanged upstream checkout")

    inventory = upstream / "dataset-info" / f"all_tasks__ds_v{args.annotation_version}" / "tasks_MedVision-TL-CoT__Axial__Test.json"
    counts = json.loads(inventory.read_text())
    if args.task not in counts:
        parser.error(f"Task is absent from {inventory}")
    if indices is not None and max(indices) >= counts[args.task]:
        parser.error("Sample index exceeds the pinned test inventory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_file = output / "tasks.json"
    task_file.write_text(json.dumps({args.task: counts[args.task]}, indent=2) + "\n")
    command = ["python", str(Path(__file__).resolve().with_name("execute_baseline.py")), "--run", str(output)]
    environment = {
        "MedVision_PLANNER_VERSION": args.annotation_version,
        "MedVision_DATA_DIR": str(args.data_dir.resolve()),
        "HF_HOME": str(args.data_dir.resolve()/'.cache/huggingface'),
        "HF_DATASETS_CACHE": str(args.data_dir.resolve()/'.cache/huggingface/datasets'),
        "MedVision_ACK_RELEASE": "1.4.0",
        "MedVision_DISABLE_SAMPLE_FILTERING": "false",
        "MedVision_FORCE_INSTALL_CODE": "False",
        "MedVision_FORCE_DOWNLOAD_DATA": "False",
        "MedVision_DOWNLOAD_QC_FIGURES": "False",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    manifest = {
        "status": "prepared_not_run",
        "dataset_revision": "f4040ed7d2d2b45e09c1a996ad2969f2018051d3",
        "dataset_loader_path": str(loader) if loader_hash else None,
        "dataset_loader_sha256": loader_hash,
        "model_revision_declared": args.model_revision,
        "model_path": str(args.model_path.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "sample_indices": indices,
        "sample_indices_sha256": hashlib.sha256(args.sample_indices.read_bytes()).hexdigest() if args.sample_indices else None,
        "upstream_commit": commit,
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "task": args.task,
        "annotation_version": args.annotation_version,
        "upstream_inventory_test_rows": counts[args.task],
        "requested_limit": args.limit,
        "environment": environment,
        "command": command,
        "required_before_reporting": [
            "Record checkpoint revision and dataset repository/code revisions.",
            "Verify installed evaluator matches the pinned checkout; save pip freeze and GPU details.",
            "Audit actual loaded row IDs, case IDs, labels and counts against the inventory.",
            "Run upstream parsing and scoring; audit failure denominators and ambiguous-case filtering.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in environment.items())
    (output / "run.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + exports + "\n" + shlex.join(["python", str(Path(__file__).resolve().with_name("preflight_baseline.py")), "--run", str(output)]) + "\n" + shlex.join(command) + "\n")
    print(f"Prepared {output}; no data downloaded and no inference run.")


if __name__ == "__main__":
    main()
