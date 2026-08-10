#!/usr/bin/env bash
# 10 Aug 2026 - E13C-CFG-02/PERF-03: The Serv entry point activates the pinned
# environment, loads machine-local paths, then delegates to the same audited
# one-GPU launcher used for every family member.
set -euo pipefail

: "${PROJECT_ROOT:?Absolute Serv path to this diffusion_template checkout}"
: "${RUN_NAME:?Unique run name}"
: "${CONFIG_NAME:?E13, BC_E13 or CL14 config name}"
: "${CONDA_ENV_PATH:?Absolute path to the existing photomaker_NS environment}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  echo "Conda is unavailable on Serv" >&2
  exit 70
fi
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

cd "${PROJECT_ROOT}"
test -f .env || { echo "Missing ${PROJECT_ROOT}/.env" >&2; exit 71; }
exec bash launchers/active/run_e13_family_24k_1gpu.sh
