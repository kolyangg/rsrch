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
  CLIP_SOURCE="${CLIP_SOURCE:-git+https://github.com/openai/CLIP.git}"
  TMP_PIP_FILE="$(mktemp)"
  FAILED_REQS_FILE="${SNAPSHOT_DIR}/pip_failed_requirements.txt"
  : > "${FAILED_REQS_FILE}"

  # Normalize freeze file for cross-machine restore:
  # - replace clip==1.0 with installable OpenAI CLIP source
  # - drop comments/empty lines
  # - map local file refs ("pkg @ file://...") to bare package name
  # - deduplicate while preserving first occurrence
  awk -v clip_src="${CLIP_SOURCE}" '
    {
      gsub(/\r$/, "", $0)
      if ($0 ~ /^[[:space:]]*$/) next
      if ($0 ~ /^[[:space:]]*#/) next
      if ($0 == "clip==1.0") $0 = clip_src
      if ($0 ~ /@ file:\/\//) {
        n = split($0, parts, " @ ")
        if (n >= 1 && parts[1] != "") {
          $0 = parts[1]
        } else {
          next
        }
      }
      if (!seen[$0]++) print $0
    }
  ' "${PIP_FILE}" > "${TMP_PIP_FILE}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "[3/3] Installing uv for faster pip restore"
    python -m pip install --upgrade uv
  fi

  PY_BIN="$(command -v python)"
  echo "[3/3] Installing with uv (bulk pass)"
  set +e
  uv pip install --python "${PY_BIN}" -r "${TMP_PIP_FILE}"
  UV_STATUS=$?
  set -e

  if [[ ${UV_STATUS} -ne 0 ]]; then
    echo "[3/3] Bulk uv install failed. Retrying package-by-package to maximize coverage."
    while IFS= read -r req; do
      [[ -z "${req}" ]] && continue
      set +e
      uv pip install --python "${PY_BIN}" "${req}"
      ONE_STATUS=$?
      set -e
      if [[ ${ONE_STATUS} -ne 0 ]]; then
        echo "${req}" >> "${FAILED_REQS_FILE}"
      fi
    done < "${TMP_PIP_FILE}"
  fi

  rm -f "${TMP_PIP_FILE}"

  if [[ -s "${FAILED_REQS_FILE}" ]]; then
    echo "WARNING: Some pip requirements could not be installed."
    echo "See: ${FAILED_REQS_FILE}"
  else
    rm -f "${FAILED_REQS_FILE}"
  fi
else
  echo "[3/3] pip_freeze.txt not found; skipping pip install."
fi

# Optional but recommended for insightface/onnx/other compiled wheels:
# create an isolated GCC runtime overlay and export it for the shell.
SKIP_RUNTIME_OVERLAY="${SKIP_RUNTIME_OVERLAY:-0}"
RUNTIME_EXPORT_FILE="${SNAPSHOT_DIR}/activate_runtime_${TARGET_ENV}.sh"
if [[ "${SKIP_RUNTIME_OVERLAY}" != "1" ]]; then
  echo "[4/4] Preparing C++ runtime overlay (libstdc++)"
  RUNTIME_PREFIX="${RUNTIME_PREFIX:-${SNAPSHOT_DIR}/_gcc_runtime}"
  RUNTIME_CHANNEL="${RUNTIME_CHANNEL:-conda-forge}"
  RUNTIME_LIBSTDCPP_VERSION="${RUNTIME_LIBSTDCPP_VERSION:-13}"
  RUNTIME_LIBGCC_VERSION="${RUNTIME_LIBGCC_VERSION:-13}"

  if [[ ! -f "${RUNTIME_PREFIX}/lib/libstdc++.so.6" ]]; then
    echo "[4/4] Creating runtime overlay at ${RUNTIME_PREFIX}"
    conda create -y -p "${RUNTIME_PREFIX}" \
      -c "${RUNTIME_CHANNEL}" \
      --override-channels \
      "libstdcxx-ng=${RUNTIME_LIBSTDCPP_VERSION}" \
      "libgcc-ng=${RUNTIME_LIBGCC_VERSION}"
  else
    echo "[4/4] Reusing existing runtime overlay at ${RUNTIME_PREFIX}"
  fi

  cat > "${RUNTIME_EXPORT_FILE}" <<EOF
#!/usr/bin/env bash
# Auto-generated by recreate_env_from_snapshot.sh for env: ${TARGET_ENV}
export LD_LIBRARY_PATH="${RUNTIME_PREFIX}/lib:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${RUNTIME_PREFIX}/lib/libstdc++.so.6:\${LD_PRELOAD:-}"
EOF
  chmod +x "${RUNTIME_EXPORT_FILE}"

  if [[ -f "${RUNTIME_PREFIX}/lib/libstdc++.so.6" ]]; then
    if strings "${RUNTIME_PREFIX}/lib/libstdc++.so.6" | grep -q "GLIBCXX_3.4.32"; then
      echo "[4/4] Runtime overlay OK (GLIBCXX_3.4.32 found)."
    else
      echo "WARNING: ${RUNTIME_PREFIX}/lib/libstdc++.so.6 does not expose GLIBCXX_3.4.32"
    fi
  fi
else
  echo "[4/4] Skipping runtime overlay (SKIP_RUNTIME_OVERLAY=1)."
fi

echo "Done. Restored env: ${TARGET_ENV}"
echo "Verify:"
echo "  conda activate ${TARGET_ENV}"
if [[ "${SKIP_RUNTIME_OVERLAY}" != "1" ]]; then
  echo "  source ${RUNTIME_EXPORT_FILE}"
fi
echo "  python --version"
echo "  python -m pip freeze | head"
if [[ "${SKIP_RUNTIME_OVERLAY}" != "1" ]]; then
  echo "  # then run your accelerate launch command in this same shell"
fi
