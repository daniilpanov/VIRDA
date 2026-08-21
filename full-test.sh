#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${1:-}" ]]; then
    echo "Usage: $0 <DATASET>" >&2
    exit 1
fi

DATASET="$1"
echo "Dataset: $DATASET"

if [[ -z "${NONIFTI:-}" ]]; then
  if [[ -z "${NIFTI_PATH:-}" ]]; then
    NIFTI_PATH="test-data/${DATASET}/head.nii.gz"
  fi

  echo "NIfTI:   $NIFTI_PATH"

  if [[ -z "${NORUN:-}" ]]; then
    PYTHONPATH=src uv run python -m virda --auto_detect_fiducials true --nifti-path "$NIFTI_PATH" --project-dir "research/out/$DATASET"
  fi

  if [[ -z "${NOCHECK:-}" ]]; then
    if [[ -z "${NOMESH:-}" ]]; then
      uv run --active python research/scripts/mesh_over_mri_viewer.py \
          --nifti "$NIFTI_PATH" \
          --mesh "research/out/$DATASET/mesh/final_mesh.ply" \
          --fiducials "research/out/$DATASET/fiducials/fiducials.json"
    else
      uv run --active python research/scripts/mesh_over_mri_viewer.py --nifti "$NIFTI_PATH"
    fi
  fi

  exit 0
fi

if [[ -z "${NOCHECK:-}" ]]; then
  if [[ -z "${NOMESH:-}" ]]; then
    uv run --active python research/scripts/mesh_over_mri_viewer.py \
        --mesh "research/out/$DATASET/mesh/final_mesh.ply" \
        --fiducials "research/out/$DATASET/fiducials/fiducials.json"
  else
    echo "Error: nothing to display"
    exit 1
  fi
fi
