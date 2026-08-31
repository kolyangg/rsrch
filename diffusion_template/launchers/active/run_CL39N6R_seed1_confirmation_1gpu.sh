#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${CL39_CHECKPOINT_16K:?Set the immutable CL39 epoch-8 checkpoint}"
: "${CL39_CHECKPOINT_16K_SHA256:?Set its SHA-256}"
: "${CL39N6R_CONFIRM_ROOT:?Set a fresh absent confirmation root}"
: "${SUBJECT_V2_ID_EMBEDS:?Set sealed subject-v2 embeddings}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA-capable Python}"
test ! -e "${CL39N6R_CONFIRM_ROOT}"
mkdir -p "${CL39N6R_CONFIRM_ROOT}/saved" "${CL39N6R_CONFIRM_ROOT}/logs"
printf '%s  %s\n' "${CL39_CHECKPOINT_16K_SHA256}" "${CL39_CHECKPOINT_16K}" | sha256sum -c -
MAP_PATH="${ROOT_DIR}/src/configs/maps/CL39N6R_up1_low_pruned_v1.json"
MAP_SHA=858c4663083ccffbd461e94215d4e9951f2765b59b4f49ce454de92c5910904f
printf '%s  %s\n' "${MAP_SHA}" "${MAP_PATH}" | sha256sum -c -

BBOX_BASE="${CL39N6R_CONFIRM_ROOT}/pm96_bboxes_seed1.json"
cp -p "${ROOT_DIR}/../dataset_full/val_dataset/pm96_bboxes_new.json" "${BBOX_BASE}"
BBOX_AUTO="${CL39N6R_CONFIRM_ROOT}/pm96_bboxes_seed1_auto.json"
PM_PREVIEW="${CL39N6R_CONFIRM_ROOT}/pm_preview"
export CL39N6R_CONFIRM_SAVE_DIR="${CL39N6R_CONFIRM_ROOT}/saved"
export CL39N6R_SEED1_BBOX_PATH="${BBOX_BASE}"
export CL39N6R_PM_PREVIEW_DIR="${PM_PREVIEW}"
export CL39N6R_CONFIRM_RUN_NAME=CL39N6R_seed1_all_on_confirmation

accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py --config-name=CL39N6R_up1_low_prune_seed1_confirmation writer=console \
  2>&1 | tee "${CL39N6R_CONFIRM_ROOT}/logs/all_on.log"
ALL_ON="${CL39N6R_CONFIRM_ROOT}/saved/${CL39N6R_CONFIRM_RUN_NAME}/val_images/manual_val"
test "$(find "${ALL_ON}" -type f -name '*.png' | wc -l)" -eq 96
test "$(find "${PM_PREVIEW}" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 96
test -s "${BBOX_AUTO}"
python tools/analysis/verify_seed1_dynamic_bbox.py \
  --bbox-json "${BBOX_AUTO}" --images-root "${ALL_ON}" \
  --output "${CL39N6R_CONFIRM_ROOT}/bbox_gate.json"

export CL39N6R_SEED1_BBOX_PATH="${BBOX_AUTO}"
export CL39N6R_PM_PREVIEW_DIR=""
export CL39N6R_CONFIRM_RUN_NAME=CL39N6R_seed1_up1_low_off_confirmation
accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py --config-name=CL39N6R_up1_low_prune_seed1_confirmation writer=console \
  automatic_bboxes=false \
  "+validation_args.cl39_group_band_map_path=${MAP_PATH}" \
  "+validation_args.cl39_group_band_map_sha256=${MAP_SHA}" \
  2>&1 | tee "${CL39N6R_CONFIRM_ROOT}/logs/up1_low_off.log"
PRUNED="${CL39N6R_CONFIRM_ROOT}/saved/${CL39N6R_CONFIRM_RUN_NAME}/val_images/manual_val"
test "$(find "${PRUNED}" -type f -name '*.png' | wc -l)" -eq 96
grep -Fq "CL39N6R_CONFIRMATION_ROUTE_ACTIVE map_sha256=${MAP_SHA}" \
  "${CL39N6R_CONFIRM_ROOT}/logs/up1_low_off.log"

"${FACE_QUALITY_SCORER_PYTHON}" tools/analysis/confirm_cl39n6r_seed1.py \
  --all-on-root "${ALL_ON}" --pruned-root "${PRUNED}" \
  --pm-root "${PM_PREVIEW}" --bbox-json "${BBOX_AUTO}" \
  --bbox-gate "${CL39N6R_CONFIRM_ROOT}/bbox_gate.json" \
  --route-log "${CL39N6R_CONFIRM_ROOT}/logs/up1_low_off.log" \
  --subject-v2-embeds "${SUBJECT_V2_ID_EMBEDS}" \
  --references "${ROOT_DIR}/../dataset_full/val_dataset/references" \
  --prompts "${ROOT_DIR}/../dataset_full/val_dataset/prompts_10.txt" \
  --classes "${ROOT_DIR}/../dataset_full/val_dataset/classes_ref.json" \
  --output "${CL39N6R_CONFIRM_ROOT}/score" --device cuda
echo "CL39N6R_CONFIRMATION_READY_FOR_VISUAL_REVIEW ${CL39N6R_CONFIRM_ROOT}/score/confirmation.json"
