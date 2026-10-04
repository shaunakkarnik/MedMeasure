#!/usr/bin/env bash
# Run inside a GPU allocation after activating the prepared Python environment.
# This script does not request or submit an allocation.
set -euo pipefail
: "${MEDMEASURE_RUN_DIR:?Set MEDMEASURE_RUN_DIR to the generated absolute run directory}"
if [[ -d "${MEDMEASURE_RUN_DIR}/results" ]]; then
  echo 'Results already exist. Prepare a new run directory to avoid mixing runs.' >&2
  exit 1
fi
bash "${MEDMEASURE_RUN_DIR}/run.sh"
