#!/usr/bin/env bash
# 10 Aug 2026 - E13C-CFG-02/PERF-03/04: One fail-closed launcher for the three
# clean recipes; it rejects Hydra overrides, verifies GPU runtime/config parity,
# trains first, and performs face-quality scoring only after training succeeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${RUN_NAME:?Set a unique RUN_NAME}"
: "${CONFIG_NAME:?Set CONFIG_NAME to E13, BC_E13 or CL14}"
: "${COMET_API_KEY:?Set COMET_API_KEY in diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA 0.1.15 Python interpreter}"

if [[ "$#" -ne 0 ]]; then
  echo "The clean E13-family launcher rejects ad-hoc Hydra overrides." >&2
  exit 2
fi
if [[ -n "${CUDA_LAUNCH_BLOCKING:-}" && "${CUDA_LAUNCH_BLOCKING}" != "0" ]]; then
  echo "CUDA_LAUNCH_BLOCKING must be unset or 0 for the audited speed profile." >&2
  exit 3
fi

# 10 Aug 2026 - E13C-CFG-02: Dataset manifests are experimental inputs, so
# every leaf pins one before GPU/model startup just as the source commit is
# pinned. BigCelebs additionally has a known sealed release hash.
verify_manifest_sha256() {
  local manifest="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${manifest}" | cut -d' ' -f1)"
  [[ "${actual}" == "${expected}" ]] || {
    echo "Manifest hash mismatch for ${manifest}: ${actual}" >&2
    exit 4
  }
}

case "${CONFIG_NAME}" in
  E13_large_ds_joint_shadow_sa128_24k)
    : "${LARGE_DATASET_MANIFEST:?Set LARGE_DATASET_MANIFEST}"
    : "${LARGE_DATASET_IMAGES:?Set LARGE_DATASET_IMAGES}"
    : "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256:?Pin the Large Dataset manifest SHA-256}"
    test -s "${LARGE_DATASET_MANIFEST}"
    test -d "${LARGE_DATASET_IMAGES}"
    [[ "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256}" == \
      "0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50" ]] || {
      echo "LARGE_DATASET_EXPECTED_MANIFEST_SHA256 is not the sealed E13 hash" >&2
      exit 4
    }
    verify_manifest_sha256 \
      "${LARGE_DATASET_MANIFEST}" "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256}"
    EXPERIMENT_COMMENT="Clean E13 replay: sealed rank-128 hard BA plus effective generic/default adapters on Large Dataset."
    ;;
  BC_E13_big_celebs_joint_shadow_sa128_24k)
    : "${BIG_CELEBS_MANIFEST:?Set BIG_CELEBS_MANIFEST}"
    : "${BIG_CELEBS_IMAGES:?Set BIG_CELEBS_IMAGES}"
    : "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256:?Pin the BigCelebs manifest SHA-256}"
    test -s "${BIG_CELEBS_MANIFEST}"
    test -d "${BIG_CELEBS_IMAGES}"
    [[ "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}" == \
      "f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa" ]] || {
      echo "BIG_CELEBS_EXPECTED_MANIFEST_SHA256 is not the sealed v2 hash" >&2
      exit 4
    }
    verify_manifest_sha256 \
      "${BIG_CELEBS_MANIFEST}" "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}"
    EXPERIMENT_COMMENT="BC_E13: exact E13 architecture transferred only to sealed BigCelebs."
    ;;
  CL14_cosmic_joint_shadow_sa128_softmask_24k)
    : "${COSMIC_LARGE_MANIFEST:?Set COSMIC_LARGE_MANIFEST}"
    : "${COSMIC_LARGE_ROOT:?Set COSMIC_LARGE_ROOT}"
    : "${COSMIC_LARGE_EXPECTED_MANIFEST_SHA256:?Pin the exact CL14 Cosmic manifest SHA-256}"
    test -s "${COSMIC_LARGE_MANIFEST}"
    test -d "${COSMIC_LARGE_ROOT}"
    verify_manifest_sha256 \
      "${COSMIC_LARGE_MANIFEST}" "${COSMIC_LARGE_EXPECTED_MANIFEST_SHA256}"
    EXPERIMENT_COMMENT="CL14 replay: exact E13 architecture, corrected Cosmic scale/pose policy, and training-only two-cell target-mask feather."
    ;;
  *) echo "Unsupported CONFIG_NAME=${CONFIG_NAME}" >&2; exit 2 ;;
esac

test ! -e "saved/${RUN_NAME}" || {
  echo "Refusing to overwrite saved/${RUN_NAME}" >&2; exit 5;
}
python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py

# 10 Aug 2026 - E13C-DATA-01/02/03/04: Decode and policy preflights run before
# the expensive model is loaded. Their JSON becomes part of the local run
# record and makes the selected manifest/pairing/reference policy auditable.
PREFLIGHT_DIR="preflight_records/${RUN_NAME}"
test ! -e "${PREFLIGHT_DIR}" || {
  echo "Refusing to overwrite ${PREFLIGHT_DIR}" >&2; exit 5;
}
mkdir -p "${PREFLIGHT_DIR}"
case "${CONFIG_NAME}" in
  E13_large_ds_joint_shadow_sa128_24k)
    python tools/datasets/preflight_large_dataset.py \
      --manifest "${LARGE_DATASET_MANIFEST}" \
      --images-root "${LARGE_DATASET_IMAGES}" \
      --sample-count 64 \
      --output "${PREFLIGHT_DIR}/large_dataset.json"
    ;;
  BC_E13_big_celebs_joint_shadow_sa128_24k)
    python tools/datasets/preflight_big_celebs.py \
      --manifest "${BIG_CELEBS_MANIFEST}" \
      --images-root "${BIG_CELEBS_IMAGES}" \
      --expected-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}" \
      --min-face-res 192 \
      --sample-count 64 \
      --output "${PREFLIGHT_DIR}/big_celebs.json"
    ;;
  CL14_cosmic_joint_shadow_sa128_softmask_24k)
    python tools/datasets/preflight_cosmic_cl.py \
      --config-name "${CONFIG_NAME}" \
      --sample-count 64 \
      --prompt-sample-count 2000 \
      --seed 0 \
      --output "${PREFLIGHT_DIR}/cosmic_cl14.json"
    ;;
esac

python - <<'PY'
import onnxruntime as ort
if ort.__version__ != "1.20.1":
    raise RuntimeError(f"Expected onnxruntime-gpu 1.20.1, got {ort.__version__}")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError(f"ORT CUDA provider unavailable: {ort.get_available_providers()}")
print("ONNX Runtime preflight OK", ort.__version__, ort.get_available_providers())
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FACEANALYSIS_CPU=0
export HYDRA_FULL_ERROR=1
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error
export COMET_DISABLE_AUTO_LOGGING=1
export COMET_LOGGING_CONSOLE=ERROR
export ACCELERATE_NUM_PROCESSES=1

MODEL_OVERRIDES=()
if [[ -n "${PM_PATH:-}" ]]; then
  test -s "${PM_PATH}"
  MODEL_OVERRIDES+=("model.photomaker_path=${PM_PATH}")
fi

accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  "--config-name=${CONFIG_NAME}" \
  writer=cometml \
  "writer.run_name=${RUN_NAME}" \
  writer.project_name=aug-large-ds \
  writer.require_online_registration=true \
  "writer.experiment_comment=${EXPERIMENT_COMMENT}" \
  "${MODEL_OVERRIDES[@]}"

test -s "saved/${RUN_NAME}/comet_experiment.json"
"${FACE_QUALITY_SCORER_PYTHON}" \
  tools/comet/finalize_deferred_face_quality.py \
  --run-dir "saved/${RUN_NAME}" \
  --expected-project aug-large-ds \
  --expected-steps 0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000 \
  --images-per-step 96 \
  --partition manual_val \
  --scorer-python "${FACE_QUALITY_SCORER_PYTHON}" \
  --device cuda \
  --batch-size 8 \
  --write \
  --upload-per-image-asset \
  --nonfatal
