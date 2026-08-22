#!/usr/bin/env bash
# 21 Aug 2026 - One fail-closed launcher covers the ten clean recipes,
# including the isolated CL23/CL27/CL39 extension. It rejects ad-hoc overrides
# and performs deferred face-quality scoring only after training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
# 12 Aug 2026 - Dataset builders execute from nested tool directories; expose
# this checkout explicitly so their `src.*` imports cannot resolve elsewhere.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${RUN_NAME:?Set a unique RUN_NAME}"
: "${CONFIG_NAME:?Set CONFIG_NAME to a supported E13-family leaf}"
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

verify_subject_v2() {
  : "${SUBJECT_V2_ID_EMBEDS:?Set the sealed subject-v2 identity embeddings}"
  test -s "${SUBJECT_V2_ID_EMBEDS}"
  verify_manifest_sha256 \
    "${SUBJECT_V2_ID_EMBEDS}" \
    "e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465"
}

verify_corrected_r2_cosmic() {
  : "${COSMIC_LARGE_MANIFEST:?Set COSMIC_LARGE_MANIFEST}"
  : "${COSMIC_LARGE_ROOT:?Set COSMIC_LARGE_ROOT}"
  : "${COSMIC_LARGE_EXPECTED_MANIFEST_SHA256:?Pin the Cosmic manifest SHA-256}"
  test -s "${COSMIC_LARGE_MANIFEST}"
  test -d "${COSMIC_LARGE_ROOT}"
  [[ "${COSMIC_LARGE_EXPECTED_MANIFEST_SHA256}" == \
    "8ba369ef2fdc0496a0d3d55afb5c7923c1aa299343a676ac6bc0d94f3a3a0196" ]] || {
    echo "COSMIC_LARGE_EXPECTED_MANIFEST_SHA256 is not the corrected-r2 hash" >&2
    exit 4
  }
  verify_manifest_sha256 \
    "${COSMIC_LARGE_MANIFEST}" "${COSMIC_LARGE_EXPECTED_MANIFEST_SHA256}"
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
  # 13 Aug 2026 - CL14_CA uses corrected-r2 Cosmic/subject-v2 inputs while
  # retaining the shared fail-closed 24k training and validation launch path.
  CL14_CA_cosmic_residual_identity_ca_24k)
    verify_corrected_r2_cosmic
    verify_subject_v2
    EXPERIMENT_COMMENT="CL14_CA clean replay: CL14 plus bounded rank-64 target-Q/PhotoMaker-ID-KV residual CA in up_blocks.0/1."
    ;;
  CL18_cosmic_crossview_spatial_consistency_24k)
    verify_corrected_r2_cosmic
    verify_subject_v2
    EXPERIMENT_COMMENT="CL18 corrected-r2 replay: CL14 plus training-only alternate-view spatial prediction consistency."
    ;;
  CL19_cosmic_true_soft_fullquery_router_24k)
    verify_corrected_r2_cosmic
    verify_subject_v2
    EXPERIMENT_COMMENT="CL19 corrected-r2 replay: CL14 full-query messages with one two-cell cosine target router."
    ;;
  CL20_cosmic_bigcelebs_hardcase_curriculum_24k)
    verify_corrected_r2_cosmic
    verify_subject_v2
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
    EXPERIMENT_COMMENT="CL20 corrected-r2 replay: exact CL14 model with the sealed Cosmic/BigCelebs hard-case curriculum."
    ;;
  CL23_cosmic_temporal_frequency_router_24k)
    verify_corrected_r2_cosmic
    verify_subject_v2
    EXPERIMENT_COMMENT="CL23 clean replay: CL19 plus fixed denoising-progress low/high frequency routing on the reference-minus-native message."
    ;;
  CL27_cosmic_frequency_surface_energy_24k)
    verify_corrected_r2_cosmic
    verify_subject_v2
    EXPERIMENT_COMMENT="CL27 clean replay: exact CL23 inference plus deterministic training-only frequency-surface energy supervision."
    ;;
  CL39_cosmic_null_key_confidence_router_24k)
    verify_corrected_r2_cosmic
    verify_subject_v2
    EXPERIMENT_COMMENT="CL39 clean replay: CL27 plus parameter-free entropy abstention to native target self-attention in up_blocks.0/1."
    ;;
  *) echo "Unsupported CONFIG_NAME=${CONFIG_NAME}" >&2; exit 2 ;;
esac

test ! -e "saved/${RUN_NAME}" || {
  echo "Refusing to overwrite saved/${RUN_NAME}" >&2; exit 5;
}
python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py
case "${CONFIG_NAME}" in
  CL14_CA_cosmic_residual_identity_ca_24k)
    python tools/validate_cl14_ca_config.py
    ;;
  CL18_*|CL19_*|CL20_*)
    python tools/validate_cl18_cl20_config.py --config-name "${CONFIG_NAME}"
    ;;
  CL23_*|CL27_*|CL39_*)
    python tools/validate_cl23_cl27_config.py --config-name "${CONFIG_NAME}"
    ;;
esac

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
  CL14_cosmic_joint_shadow_sa128_softmask_24k|\
  CL14_CA_cosmic_residual_identity_ca_24k|\
  CL18_cosmic_crossview_spatial_consistency_24k|\
  CL19_cosmic_true_soft_fullquery_router_24k|\
  CL23_cosmic_temporal_frequency_router_24k|\
  CL27_cosmic_frequency_surface_energy_24k|\
  CL39_cosmic_null_key_confidence_router_24k)
    python tools/datasets/preflight_cosmic_cl.py \
      --config-name "${CONFIG_NAME}" \
      --sample-count 64 \
      --prompt-sample-count 2000 \
      --seed 0 \
      --output "${PREFLIGHT_DIR}/cosmic.json"
    ;;
  CL20_cosmic_bigcelebs_hardcase_curriculum_24k)
    # 12 Aug 2026 - Generate the schedule from sealed inputs on the server and
    # reject it unless it reproduces the corrected-r2 bytes exactly.
    export CL20_SCHEDULE="${PREFLIGHT_DIR}/train_48k_bs2.jsonl"
    export CL20_SCHEDULE_SUMMARY="${PREFLIGHT_DIR}/train_48k_bs2.summary.json"
    python tools/datasets/build_cl20_hardcase_schedule.py \
      --cosmic-manifest "${COSMIC_LARGE_MANIFEST}" \
      --cosmic-root "${COSMIC_LARGE_ROOT}" \
      --big-manifest "${BIG_CELEBS_MANIFEST}" \
      --big-images-root "${BIG_CELEBS_IMAGES}" \
      --output "${CL20_SCHEDULE}" \
      --summary-output "${CL20_SCHEDULE_SUMMARY}"
    export CL20_SCHEDULE_SHA256="$(sha256sum "${CL20_SCHEDULE}" | cut -d' ' -f1)"
    [[ "${CL20_SCHEDULE_SHA256}" == \
      "783eb1729871e4ac423c770042315572ee7ea24171797402fc4a565999dd5289" ]] || {
      echo "CL20 schedule does not match the corrected-r2 seal" >&2
      exit 4
    }
    python tools/datasets/preflight_cl20_curriculum.py \
      --config-name "${CONFIG_NAME}" \
      --output "${PREFLIGHT_DIR}/cl20_curriculum.json"
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

# 12 Aug 2026 - Register and persist the immutable Comet key during startup;
# waiting until a 24k run exits would leave an untraceable live experiment.
set +e
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
  "${MODEL_OVERRIDES[@]}" &
TRAIN_PID=$!
set -e

COMET_RECORD="saved/${RUN_NAME}/comet_experiment.json"
COMET_READY=0
for _ in $(seq 1 300); do
  if [[ -s "${COMET_RECORD}" ]] && python - "${COMET_RECORD}" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
key = (record.get("comet") or {}).get("experiment_key")
raise SystemExit(0 if isinstance(key, str) and len(key) == 32 else 1)
PY
  then
    COMET_READY=1
    echo "COMET_STARTUP_VERIFIED ${COMET_RECORD}"
    break
  fi
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    wait "${TRAIN_PID}"
    exit $?
  fi
  sleep 2
done
if [[ "${COMET_READY}" -ne 1 ]]; then
  echo "Comet immutable key was not registered within 10 minutes." >&2
  kill "${TRAIN_PID}" 2>/dev/null || true
  wait "${TRAIN_PID}" || true
  exit 78
fi

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Training failed with status ${TRAIN_STATUS}; deferred face quality skipped." >&2
  exit "${TRAIN_STATUS}"
fi

test -s "${COMET_RECORD}"
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
