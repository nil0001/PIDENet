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

echo "=== Phase 1: object-frame candidates ==="
python run_phase1.py --linemod "${LINEMOD_PATH}" --objects ${OBJECTS} --out "${OUT_DIR}"

echo
echo "=== (optional) Debug pipeline figures ==="
python make_debug_figure.py --linemod "${LINEMOD_PATH}" --objects ${OBJECTS} --out "${OUT_DIR}"

echo
echo "=== Phase 2: per-frame camera-coordinate labels ==="
python run_phase2.py --linemod "${LINEMOD_PATH}" --phase1 "${OUT_DIR}" --out "${OUT_DIR}" --objects ${OBJECTS}

echo
echo "Done. Outputs in $(pwd)/${OUT_DIR}"
