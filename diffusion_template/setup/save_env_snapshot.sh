#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is not available in PATH."
  exit 1
fi

if ! command -v pip >/dev/null 2>&1; then
  echo "ERROR: pip is not available in PATH."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH."
  exit 1
fi

OUT_DIR="${1:-$(pwd)/setup/env_snapshot}"
mkdir -p "${OUT_DIR}"

ENV_NAME="${2:-${CONDA_DEFAULT_ENV:-}}"
if [[ -z "${ENV_NAME}" ]]; then
  ENV_NAME="$(conda info --json | python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("active_prefix_name",""))')"
fi
if [[ -z "${ENV_NAME}" ]]; then
  echo "ERROR: could not detect active conda env name. Pass it as 2nd argument."
  exit 1
fi

echo "Saving environment snapshot..."
echo "  env_name: ${ENV_NAME}"
echo "  output:   ${OUT_DIR}"

{
  echo "env_name=${ENV_NAME}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "platform=$(uname -a)"
  echo "python_executable=$(command -v python)"
  echo "python_version=$(python --version 2>&1)"
  echo "pip_executable=$(command -v pip)"
  echo "conda_executable=$(command -v conda)"
  echo "conda_default_env=${CONDA_DEFAULT_ENV:-}"
} > "${OUT_DIR}/snapshot_meta.txt"

conda info --json > "${OUT_DIR}/conda_info.json"
conda list > "${OUT_DIR}/conda_list.txt"
conda list --explicit > "${OUT_DIR}/conda_explicit.txt"
conda env export -n "${ENV_NAME}" > "${OUT_DIR}/environment_full.yml"
conda env export -n "${ENV_NAME}" --no-builds > "${OUT_DIR}/environment_nobuilds.yml"

python -m pip freeze > "${OUT_DIR}/pip_freeze.txt"
python -m pip list --format=json > "${OUT_DIR}/pip_list.json"

cat > "${OUT_DIR}/README.txt" <<'EOF'
Snapshot contents:
- conda_explicit.txt        # exact conda package URLs/builds (best for same OS/arch)
- environment_nobuilds.yml  # portable conda spec
- pip_freeze.txt            # pip packages pinned exactly

Recommended restore command:
  ./setup/recreate_env_from_snapshot.sh <snapshot_dir> <new_env_name>
EOF

echo "Done. Snapshot saved to: ${OUT_DIR}"
