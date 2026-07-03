#!/usr/bin/env bash
set -uo pipefail

# Branched-attention debug A/B matrix on a saved checkpoint.
# See debug_planning_03Jul/ba_debug_plan_v1.md (test matrix) and
# debug_planning_03Jul/ba_debug_runbook_v1.md (how to run / what to compare).
#
# Usage (from diffusion_template/, with the training conda env active):
#   bash serv_new_runs/run_ba_debug_matrix.sh            # full matrix
#   bash serv_new_runs/run_ba_debug_matrix.sh T0 T1_gs1  # selected tests only
#
# Env overrides:
#   CKPT=saved/<run>/weights-epochN.pth   checkpoint to probe (default: 2k vast run)
#   PYTHON_BIN=python                     python executable
#   VAL_DS=../dataset_full/val_dataset    dataset location
#   EXTRA="k=v k=v"                       extra hydra overrides appended to every run
#                                         (e.g. EXTRA="enable_vae_tiling=true dataset.limit=2" for small GPUs)

CKPT="${CKPT:-saved/03Jul_start_ba_cosm_new1_vast/weights-epoch1.pth}"
CFG="inference/ba_2k_debug"
OUT="${OUT:-outputs/ba_debug}"
PY="${PYTHON_BIN:-python}"
VAL_DS="${VAL_DS:-../dataset_full/val_dataset}"

mkdir -p "${OUT}"

run() {
    local id="$1"; shift
    echo ""
    echo "================ [${id}] extra overrides: $* ${EXTRA:-} ================"
    # ${EXTRA} is intentionally unquoted: it carries whitespace-separated overrides
    if ${PY} infer.py --config-name="${CFG}" \
        saved_checkpoint="${CKPT}" \
        output_dir="${OUT}/${id}" \
        "$@" ${EXTRA:-} 2>&1 | tee "${OUT}/${id}.log"; then
        echo "[${id}] OK"
    else
        echo "[${id}] FAILED (see ${OUT}/${id}.log)"
    fi
}

# Which tests to run: all by default, or the ones passed as CLI args.
ALL_TESTS=(T0 T0b T1_gs1 T1_gs2 T1_gs3 T2_sdxl T3_refcrop T4_noca T4b_nosa T5_pmonly T6_top50 T7_uncondfix)
TESTS=("${@:-${ALL_TESTS[@]}}")

for test_id in "${TESTS[@]}"; do
case "${test_id}" in

# T0: baseline = reproduce validation behavior at the checkpoint
#     (RealVis trunk, gs=5). Also populates the shared auto-bbox store so all
#     later RealVis tests use IDENTICAL gen masks.
T0) run T0 ;;

# T0b: untrained processors (identity clones) - should look like step-0 val.
T0b) run T0b saved_checkpoint=null ;;

# T1: guidance sweep. If face artifacts collapse at low gs -> CFG-amplification
#     through the trained branch layers is confirmed (cause 1).
T1_gs1) run T1_gs1 validation_args.guidance_scale=1 ;;
T1_gs2) run T1_gs2 validation_args.guidance_scale=2 ;;
T1_gs3) run T1_gs3 validation_args.guidance_scale=3 ;;

# T2: run BA inside its TRAINING trunk (SDXL-base) - isolates base mismatch
#     (cause 4). Own bbox store: the PhotoMaker pass differs on this base.
T2_sdxl) run T2_sdxl \
    model.pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    pipeline.pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    bbox_mask_gen_path="${OUT}/bbox_gen_auto_sdxl.json" ;;

# T3: training-style tight face crops as refs (cause 2: ref-domain gap).
#     Requires scripts/crop_refs_to_face.py output. Reuses T0 gen bboxes.
T3_refcrop) run T3_refcrop \
    dataset.images_dir="${VAL_DS}/references_two_cropped" \
    dataset.bbox_mask_ref="${VAL_DS}/ref_bboxes_two_cropped.json" ;;

# T4: disable branched cross-attention (largest trained deltas live there) /
#     branched self-attention, one at a time (cause 3 localization).
T4_noca) run T4_noca disable_branched_ca=true ;;
T4b_nosa) run T4b_nosa disable_branched_sa=true ;;

# T5: plain PhotoMaker sanity - trained ckpt loaded but BA off. lora_adapter is
#     frozen-at-init so this must be clean; if not, something else leaks.
T5_pmonly) run T5_pmonly \
    validation_args.use_branched_attention=false \
    validation_args.use_bbox_mask_gen=false \
    automatic_bboxes=false ;;

# T6: patch only the earliest 50% of self-attention sites (depth localization).
T6_top50) run T6_top50 model.ba_patch_top_k=0.5 ba_patch_top_k=0.5 ;;
T6_top25) run T6_top25 model.ba_patch_top_k=0.25 ba_patch_top_k=0.25 ;;

# T7: F1 fix - plain negative prompt for the uncond face branch under CFG.
T7_uncondfix) run T7_uncondfix ba_uncond_face_fix=true ;;

*) echo "Unknown test id: ${test_id}" ;;
esac
done

echo ""
echo "Done. Outputs in ${OUT}/<TEST_ID>/, logs in ${OUT}/<TEST_ID>.log"
echo "Post-hoc identity report: ${PY} scripts/idsim_report.py --refs-dir ${VAL_DS}/references_two ${OUT}/T*"
