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
    if [[ "$DATASET" == CTRL_* ]]; then
      NIFTI_PATH="research/test-data/${DATASET}/${DATASET}_Raw_Data/${DATASET}_T1_3D_PROSET_Sag.nii.gz"
    else
      NIFTI_PATH="$(find "research/test-data/$DATASET" -type f \( -name '*.nii' -o -name '*.nii.gz' \) ! -name '*_mask*' -print -quit)"
    fi
  fi

  if [[ -z "$NIFTI_PATH" ]]; then
      echo "Error: no NIfTI file found in research/test-data/$DATASET" >&2
      exit 1
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
