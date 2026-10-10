"""CLI bootstrap, executed ON Colab with `colab exec -f` (not locally).

Mount Drive and upload /content/medmeasure.bundle and /content/medmeasure-config.json
first. The Git bundle transfers committed code without remote GitHub credentials.
Reuses the pinned inference notebook's setup cells; never runs its scan-staging,
backup, smoke, full-evaluation or publishing cells automatically.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

config = json.loads(Path('/content/medmeasure-config.json').read_text())
CODE_COMMIT = config['code_commit']
RUN_NAME = config['run_name']
SHARD_SIZE = config.get('shard_size', 100)
MAX_CHUNKS = config.get('max_chunks', 1)
if not re.fullmatch(r'[0-9a-f]{40}', CODE_COMMIT):
    raise ValueError('code_commit must be the full committed pipeline SHA')
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', RUN_NAME):
    raise ValueError('Use a simple run_name')
if SHARD_SIZE < 1 or MAX_CHUNKS < 0:
    raise ValueError('Invalid chunk configuration')
REPO = Path('/content/MedMeasure')
if not REPO.exists():
    subprocess.run(['git', 'clone', '/content/medmeasure.bundle', str(REPO)], check=True)
if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'], text=True).strip():
    raise ValueError('Preserve local checkout changes before using this bootstrap')
subprocess.run(['git','-C',str(REPO),'checkout','--detach',CODE_COMMIT], check=True)
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip() == CODE_COMMIT
lock = json.loads((REPO/'configs/kits23-pilot.json').read_text())
PERSIST = Path('/content/drive/MyDrive/MedMeasure')/RUN_NAME
if not Path('/content/drive/MyDrive').is_dir():
    raise ValueError('Run colab drivemount before inference setup')
PERSIST.mkdir(parents=True, exist_ok=True)
AUDIT, SHARDS, OUTPUT = PERSIST/'audit', PERSIST/'shards', PERSIST/'evaluation'
DATA = Path('/content/medmeasure-data')
MODEL = Path('/content/MedVision-V0-7B')
UPSTREAM = Path('/content/MedVision')
VENV = Path('/content/medmeasure-venv')
PYTHON = VENV/'bin/python'
DATA_ARCHIVE = Path('/content/drive/MyDrive/MedMeasure/cache')/f"kits23-{lock['dataset_revision']}-portable-v2.tar"
if not DATA_ARCHIVE.is_file() or not DATA_ARCHIVE.with_name(DATA_ARCHIVE.name+'.json').is_file():
    raise FileNotFoundError('Upload the completed portable-v2 archive and sidecar to Drive before allocating GPU compute')
cache_info = json.loads(DATA_ARCHIVE.with_name(DATA_ARCHIVE.name+'.json').read_text())
if cache_info.get('format') != 2 or cache_info.get('dataset_revision') != lock['dataset_revision']:
    raise ValueError('A matching portable-v2 bundle is required')
CACHE_HELPER = REPO/'scripts/cache_colab_data.py'
notebook = json.loads((REPO/'notebooks/medvision_v0_kits23_local_data.ipynb').read_text())


def section_code(title):
    """Read the explicit setup section from this checkout's pinned notebook."""
    collecting = False
    collected = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            if collecting:
                break
            collecting = cell['source'].splitlines()[0] == title
        elif collecting:
            collected.append(cell['source'])
    if not collected:
        raise ValueError(f'Missing notebook setup section: {title}')
    return '\n'.join(collected)


def git(*args, cwd=None, authenticated=False):
    if authenticated:
        raise ValueError('CLI bootstrap uses a Git bundle; use local Git to publish downloaded results')
    return subprocess.check_output(['git', *map(str,args)], cwd=cwd, text=True).strip()


def secret(name):
    # Public model/data downloads need no token. Optional credentials can already
    # be in the remote environment; never include tokens in the JSON config.
    return os.environ.get(name)


# Globals persist in the CLI's kernel for the subsequent smoke/run/score steps.
exec(section_code('## Stream command output'), globals())
import shutil
import tempfile
from IPython.display import display, Image
for section in [
    '## Install the pinned evaluation environment',
    '## Restore processed scans and rebuild destination metadata',
    '## Stage pinned metadata and prepare the frozen evaluation',
    '## Stage the model',
]:
    print(section, flush=True)
    exec(section_code(section), globals())
# Only run the geometry check here, not the separate manual-review/smoke cell.
geometry_section = section_code('## Geometry check and visual review')
geometry_check = geometry_section.split('GEOMETRY_REVIEWED = False')[0]
exec(geometry_check, globals())
COMMON = ['--upstream', UPSTREAM, '--data-dir', DATA, '--model-path', MODEL,
          '--audit', AUDIT, '--shards', SHARDS, '--output', OUTPUT]
print('Setup complete. Inspect geometry overlays before the explicit smoke step. '
      'No GPU inference was run by setup.', flush=True)
