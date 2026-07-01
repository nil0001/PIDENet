#!/usr/bin/env bash
# Ubuntu quick-run script.
# Assumes: python3 -m venv .venv && pip install -r requirements.txt
# Edit LINEMOD_PATH below, then chmod +x run_ubuntu.sh && ./run_ubuntu.sh

set -euo pipefail

LINEMOD_PATH="${LINEMOD_PATH:-$HOME/datasets/LINEMOD}"
OBJECTS="${OBJECTS:-1 5}"
OUT_DIR="${OUT_DIR:-outputs}"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "Running phase-1 annotation on objects ${OBJECTS} from ${LINEMOD_PATH} ..."
python run_phase1.py --linemod "${LINEMOD_PATH}" --objects ${OBJECTS} --out "${OUT_DIR}"

echo
echo "Generating debug pipeline figures ..."
python make_debug_figure.py --linemod "${LINEMOD_PATH}" --objects ${OBJECTS} --out "${OUT_DIR}"

echo
echo "Done. Outputs in $(pwd)/${OUT_DIR}"
