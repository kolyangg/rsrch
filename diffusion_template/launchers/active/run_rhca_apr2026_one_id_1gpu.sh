#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# Credentials and machine-local paths live here and are never committed.
# ENV_FILE can point to a different file when needed on another server.
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  echo "Loaded environment from ${ENV_FILE}"
fi

EXPECTED_CODE_COMMIT="aede146e2e2a2dae1cb3d14a0ea5daed25ae9604"
# Dataset/dataloader registries are intentionally excluded because this branch
# adds isolated entries for new datasets; the historical model/runtime remains
# hash-locked below.
HISTORICAL_RUNTIME_FILES=(
  "train.py"
  "src/configs/one_id_09Feb_testing.yaml"
  "src/configs/model/photomaker_branched_lora2.yaml"
  "src/configs/pipeline/pm_br_09Feb_testing.yaml"
  "src/configs/trainer/photomaker_lora.yaml"
  "src/datasets/cosmic.py"
  "src/datasets/manual_val.py"
  "src/loss/diffusion_loss.py"
  "src/model/photomaker_branched/attn_processor_cleanest.py"
  "src/model/photomaker_branched/branched_runtime.py"
  "src/model/photomaker_branched/lora2.py"
  "src/model/photomaker_branched/lora2_helpers.py"
  "src/pipelines/br_pipeline_helpers.py"
  "src/pipelines/photomaker_branched_clean.py"
  "src/trainer/base_trainer.py"
  "src/trainer/sdxl_trainers.py"
)

GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${GIT_ROOT}" ]]; then
  echo "Historical replay must run from its Git checkout." >&2
  exit 4
fi
for relative_path in "${HISTORICAL_RUNTIME_FILES[@]}"; do
  repo_path="diffusion_template/${relative_path}"
  expected_blob="$(git rev-parse "${EXPECTED_CODE_COMMIT}:${repo_path}" 2>/dev/null || true)"
  actual_blob="$(git hash-object "${ROOT_DIR}/${relative_path}" 2>/dev/null || true)"
  if [[ -z "${expected_blob}" || "${actual_blob}" != "${expected_blob}" ]]; then
    echo "Historical code-integrity check failed: ${repo_path}" >&2
    echo "Expected the Apr 3 launch-time implementation from ${EXPECTED_CODE_COMMIT}." >&2
    exit 4
  fi
done
echo "Historical runtime verified: ${EXPECTED_CODE_COMMIT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-error}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export DIFFUSERS_VERBOSITY="${DIFFUSERS_VERBOSITY:-error}"
export COMET_DISABLE_AUTO_LOGGING="${COMET_DISABLE_AUTO_LOGGING:-1}"
export COMET_LOGGING_CONSOLE="${COMET_LOGGING_CONSOLE:-ERROR}"

WRITER="${WRITER:-cometml}"
RUN_NAME="${RUN_NAME:-rhca_1e-4_ml_step2_allst_trref_diff_replay}"
COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"
CONFIG_NAME="${CONFIG_NAME:-one_id_rhca_apr2026_replay}"
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
  echo "Set COMET_API_KEY in ${ENV_FILE} when WRITER=cometml." >&2
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
  "--config-name=${CONFIG_NAME}" \
  "writer=${WRITER}" \
  "writer.run_name=${RUN_NAME}" \
  "writer.project_name=${COMET_PROJECT}" \
  "${MODEL_OVERRIDES[@]}" \
  "$@"
