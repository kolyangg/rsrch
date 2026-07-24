#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-error}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export DIFFUSERS_VERBOSITY="${DIFFUSERS_VERBOSITY:-error}"
export COMET_DISABLE_AUTO_LOGGING="${COMET_DISABLE_AUTO_LOGGING:-1}"
export COMET_LOGGING_CONSOLE="${COMET_LOGGING_CONSOLE:-ERROR}"

WRITER="${WRITER:-cometml}"
RUN_NAME="${RUN_NAME:-rhca_1e-4_ml_step2_allst_trref_diff_replay}"
PM_PATH="${PM_PATH:-}"
LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-}"

# Current InsightFace wheels require GLIBCXX_3.4.32, while the historical
# photomaker_NS environment may resolve an older conda libstdc++. Prefer the
# existing repository runtime overlay, then known conda locations. This changes
# only process linkage; the historical model/runtime code remains untouched.
LIBSTDCXX_CANDIDATES=(
  "${LIBSTDCXX_PATH}"
  "${ROOT_DIR}/setup/env_snapshot_photomaker_NS/_gcc_runtime/lib/libstdc++.so.6"
  "${CONDA_PREFIX:-}/lib/libstdc++.so.6"
  "${HOME}/miniconda3/envs/photomaker/lib/libstdc++.so.6"
  "${HOME}/anaconda3/envs/photomaker/lib/libstdc++.so.6"
  "${HOME}/miniconda3/lib/libstdc++.so.6"
  "${HOME}/anaconda3/lib/libstdc++.so.6"
)
for candidate in \
  "${HOME}"/miniconda3/pkgs/libstdcxx-*/lib/libstdc++.so.6 \
  "${HOME}"/anaconda3/pkgs/libstdcxx-*/lib/libstdc++.so.6; do
  LIBSTDCXX_CANDIDATES+=("${candidate}")
done

COMPATIBLE_LIBSTDCXX=""
for candidate in "${LIBSTDCXX_CANDIDATES[@]}"; do
  if [[ -f "${candidate}" ]] && grep -aFq "GLIBCXX_3.4.32" "${candidate}"; then
    COMPATIBLE_LIBSTDCXX="${candidate}"
    break
  fi
done

if [[ -z "${COMPATIBLE_LIBSTDCXX}" ]]; then
  echo "No libstdc++.so.6 exposing GLIBCXX_3.4.32 was found." >&2
  echo "Install a current runtime in photomaker_NS, for example:" >&2
  echo "  conda install -n photomaker_NS -c conda-forge 'libstdcxx-ng>=13'" >&2
  echo "Then rerun, or pass LIBSTDCXX_PATH=/absolute/path/libstdc++.so.6." >&2
  exit 3
fi

export LD_LIBRARY_PATH="$(dirname "${COMPATIBLE_LIBSTDCXX}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="${COMPATIBLE_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"
echo "Using compatible C++ runtime: ${COMPATIBLE_LIBSTDCXX}"

if [[ "${WRITER}" == "cometml" && -z "${COMET_API_KEY:-}" ]]; then
  echo "COMET_API_KEY must be exported when WRITER=cometml." >&2
  echo "For an offline smoke test, run with WRITER=console." >&2
  exit 2
fi

MODEL_OVERRIDES=()
if [[ -n "${PM_PATH}" ]]; then
  MODEL_OVERRIDES+=("model.photomaker_path=${PM_PATH}")
fi

accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  --config-name=one_id_rhca_apr2026_replay \
  "writer=${WRITER}" \
  "writer.run_name=${RUN_NAME}" \
  "${MODEL_OVERRIDES[@]}" \
  "$@"
