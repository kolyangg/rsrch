#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <snapshot_dir> [new_env_name]"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH."
  exit 1
fi

SNAPSHOT_DIR="$1"
if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
  echo "ERROR: snapshot_dir does not exist: ${SNAPSHOT_DIR}"
  exit 1
fi

EXPLICIT_FILE="${SNAPSHOT_DIR}/conda_explicit.txt"
NOBUILDS_FILE="${SNAPSHOT_DIR}/environment_nobuilds.yml"
PIP_FILE="${SNAPSHOT_DIR}/pip_freeze.txt"

if [[ -n "${2:-}" ]]; then
  TARGET_ENV="$2"
else
  TARGET_ENV=""
  if [[ -f "${SNAPSHOT_DIR}/snapshot_meta.txt" ]]; then
    TARGET_ENV="$(grep -E '^env_name=' "${SNAPSHOT_DIR}/snapshot_meta.txt" | head -n1 | cut -d'=' -f2- || true)"
  fi
  if [[ -z "${TARGET_ENV}" && -f "${NOBUILDS_FILE}" ]]; then
    TARGET_ENV="$(grep -E '^name:' "${NOBUILDS_FILE}" | head -n1 | cut -d':' -f2- | xargs || true)"
  fi
  if [[ -z "${TARGET_ENV}" ]]; then
    TARGET_ENV="env_restored"
  fi
fi

echo "Restoring environment:"
echo "  snapshot: ${SNAPSHOT_DIR}"
echo "  target_env: ${TARGET_ENV}"

if conda env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV}"; then
  echo "ERROR: conda env '${TARGET_ENV}' already exists. Choose another name."
  exit 1
fi

set +e
if [[ -f "${EXPLICIT_FILE}" ]]; then
  echo "[1/3] Trying exact conda restore from ${EXPLICIT_FILE}"
  conda create -y -n "${TARGET_ENV}" --file "${EXPLICIT_FILE}"
  STATUS=$?
else
  STATUS=1
fi
set -e

if [[ ${STATUS} -ne 0 ]]; then
  if [[ ! -f "${NOBUILDS_FILE}" ]]; then
    echo "ERROR: exact conda restore failed and ${NOBUILDS_FILE} is missing."
    exit 1
  fi
  echo "[1/3] Exact restore failed (likely OS/arch/build mismatch)."
  echo "[2/3] Falling back to portable conda spec ${NOBUILDS_FILE}"
  conda env create -n "${TARGET_ENV}" -f "${NOBUILDS_FILE}"
fi

eval "$(conda shell.bash hook)"
conda activate "${TARGET_ENV}"

if [[ -f "${PIP_FILE}" ]]; then
  echo "[3/3] Installing pip packages from ${PIP_FILE}"
  python -m pip install --upgrade pip
  python -m pip install -r "${PIP_FILE}"
else
  echo "[3/3] pip_freeze.txt not found; skipping pip install."
fi

echo "Done. Restored env: ${TARGET_ENV}"
echo "Verify:"
echo "  conda activate ${TARGET_ENV}"
echo "  python --version"
echo "  python -m pip freeze | head"
