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

face_quality_enabled="${FACE_QUALITY_ENABLED:-true}"
for override in "$@"; do
  if [[ "${override}" == "trainer.face_quality.enabled=false" ]]; then
    face_quality_enabled=false
  fi
done
if [[ "${face_quality_enabled}" == "true" ]]; then
  scorer_python="${FACE_QUALITY_SCORER_PYTHON:-}"
  neb_metric_python="$(dirname "${ROOT_DIR}")/metric_envs/pyiqa-0.1.15/bin/python"
  serv_owner_root=""
  if [[ "${CONDA_PREFIX:-}" == */conda_env/* ]]; then
    serv_owner_root="${CONDA_PREFIX%%/conda_env/*}"
  fi
  pyiqa_overlay="${PYIQA_OVERLAY:-}"
  if [[ -z "${pyiqa_overlay}" && -n "${serv_owner_root}" ]]; then
    pyiqa_overlay="${serv_owner_root}/python_overlays/pyiqa-0.1.15"
  fi
  if [[ -n "${pyiqa_overlay}" && -d "${pyiqa_overlay}/pyiqa" ]]; then
    export PYTHONPATH="${pyiqa_overlay}${PYTHONPATH:+:${PYTHONPATH}}"
    if [[ -n "${serv_owner_root}" ]]; then
      export TORCH_HOME="${FACE_QUALITY_TORCH_HOME:-${serv_owner_root}/metric_cache/torch}"
      mkdir -p "${TORCH_HOME}"
    fi
  fi
  if [[ -z "${scorer_python}" ]]; then
    if python -c 'import importlib.metadata; assert importlib.metadata.version("pyiqa") == "0.1.15"' \
        >/dev/null 2>&1; then
      scorer_python="$(command -v python)"
    elif [[ -x "${neb_metric_python}" ]] \
        && "${neb_metric_python}" -c 'import importlib.metadata; assert importlib.metadata.version("pyiqa") == "0.1.15"' \
          >/dev/null 2>&1; then
      scorer_python="${neb_metric_python}"
    fi
  fi
  if [[ -z "${scorer_python}" ]] \
      || ! "${scorer_python}" -c 'import importlib.metadata; assert importlib.metadata.version("pyiqa") == "0.1.15"' \
        >/dev/null 2>&1; then
    echo "PyIQA 0.1.15 is required by default validation." >&2
    echo "Set FACE_QUALITY_SCORER_PYTHON or PYIQA_OVERLAY, or explicitly pass trainer.face_quality.enabled=false." >&2
    exit 3
  fi
  export FACE_QUALITY_SCORER_PYTHON="${scorer_python}"
  echo "Face-quality scorer verified: ${FACE_QUALITY_SCORER_PYTHON}"
fi

EXPECTED_CODE_COMMIT="aede146e2e2a2dae1cb3d14a0ea5daed25ae9604"
# Dataset/dataloader registries are intentionally excluded because this branch
# adds isolated entries for new datasets. Unmodified architecture files remain
# locked to the historical commit; the audited runtime patch is hash-locked
# separately below.
HISTORICAL_RUNTIME_FILES=(
  "src/configs/pipeline/pm_br_09Feb_testing.yaml"
  "src/datasets/manual_val.py"
  "src/loss/diffusion_loss.py"
  "src/pipelines/br_pipeline_helpers.py"
  "src/pipelines/photomaker_branched_clean.py"
  "src/trainer/sdxl_trainers.py"
)
declare -A AUDITED_RUNTIME_SHA256=(
  # 28 Jul 2026 - Adds an opt-in rank-serialized model-cache warmup for fresh
  # distributed MLS containers; model construction itself is unchanged.
  ["train.py"]="0219250219046fa98e8a92d95d986c49ea8580006edee6f88402e9f059d1b46a"
  # 27 Jul 2026 - Defaults validation to one image per item and installs the
  # exact 2k/full-96 face-quality contract without changing model routing.
  ["src/configs/one_id_09Feb_testing.yaml"]="5fa548c2da9458d113ee6b8ae4bf1b687d6c224c26ee78b60cdeadeb33e60aae"
  ["src/configs/trainer/photomaker_lora.yaml"]="395b7f77881eee609dd6a00264c7c26c09ef7e9deb4e8bc21a806e0d5fadb3b8"
  ["src/logger/cometml.py"]="1987d8ff26f2bb43ffa9ea63b31a0ddcc36b19cc0d87023a47a04c833b53637b"
  ["src/metrics/face_quality_validation.py"]="141b21dd9f95be547cf2df4d3d2572d85dae6154dd3ad40867a7cd8135cb8605"
  ["tools/inference/calculate_face_quality_metrics.py"]="8225a0f009c5c5f588afef63ddcd6db3248e4b442940ce8e5bb65f5e32e78c3a"
  ["src/configs/model/photomaker_branched_lora2.yaml"]="5c894c48a646ad4b7548ca71f0f29809a27c2cd1683a87081e73039772c1e6c5"
  ["src/datasets/cosmic.py"]="660d069a9f77ac1b7e0cb06fce245a342428159d9f05c49db32140bbd1a2467e"
  # 26 Jul 2026 - Restores the existing pose-adapt config as an opt-in runtime
  # control; the historical zero ratio remains the default.
  ["src/model/photomaker_branched/attn_processor_cleanest.py"]="e1c9f2bcbf5ebbc5e80ebb2b82fdd4471d0c0b4ea28ab01cb2fb680da969149a"
  ["src/model/photomaker_branched/branched_runtime.py"]="0589ffc9a4a6628db3a8238994855ef8e4c47457f636130e9884d86fffc21ffc"
  # 26 Jul 2026 - Adds opt-in batched frozen conditioning for unique-reference
  # datasets; historical one-ID configs retain the per-sample/cache path.
  ["src/model/photomaker_branched/lora2.py"]="f67b5153600ea3cbc0defd2511cc38b15665f95c4c0992c40a128116287abd96"
  ["src/model/photomaker_branched/lora2_helpers.py"]="404316d06ad5b253bdfc28269aa0c46ce162b90899898e6645db2991798efdd9"
  # 25 Jul 2026 - Adds an opt-in validation-only branch; default training
  # behavior and the historical model path remain unchanged.
  # 26 Jul 2026 - Adds an opt-in validation-only pose-adapt ratio; null keeps
  # training and validation on the historical shared setting.
  # 26 Jul 2026 - Adds an opt-in multi-checkpoint validation schedule; the
  # historical single-checkpoint path remains the default.
  ["src/trainer/base_trainer.py"]="fdf91ecff26272313a3ecbc4f2190d4e3beece571b579c83e71eec4ba639155d"
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
for relative_path in "${!AUDITED_RUNTIME_SHA256[@]}"; do
  actual_sha256="$(sha256sum "${ROOT_DIR}/${relative_path}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${AUDITED_RUNTIME_SHA256[${relative_path}]}" ]]; then
    echo "Audited runtime-patch integrity check failed: ${relative_path}" >&2
    exit 4
  fi
done
echo "Historical architecture and audited runtime patch verified"

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
TRAIN_EPOCH_LEN="${TRAIN_EPOCH_LEN:-2000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"  # total endpoint; 2 × 2,000 = 4,000 by default
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-1}"
if ! [[ "${ACCELERATE_NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACCELERATE_NUM_PROCESSES must be a positive integer." >&2
  exit 2
fi
if ! [[ "${TRAIN_EPOCH_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_EPOCH_LEN must be a positive integer." >&2
  exit 2
fi

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
  --num_processes="${ACCELERATE_NUM_PROCESSES}" \
  train.py \
  "--config-name=${CONFIG_NAME}" \
  "writer=${WRITER}" \
  "writer.run_name=${RUN_NAME}" \
  "++writer.project_name=${COMET_PROJECT}" \
  "trainer.epoch_len=${TRAIN_EPOCH_LEN}" \
  "trainer.n_epochs=${TRAIN_EPOCHS}" \
  "${MODEL_OVERRIDES[@]}" \
  "$@"
