#!/usr/bin/env bash
set -euo pipefail

# Quick 2k-step rerun variants for the BA validation-artifact debugging.
# See debug_planning_03Jul/ba_debug_plan_v1.md (Section 4) and
# debug_planning_03Jul/ba_debug_runbook_v1.md.
#
# Usage: bash serv_new_runs/start_ba_cosm_new1_vast_Rx.sh <R1|R2|R3|R1R3> [extra hydra overrides...]
#
# All variants also include (vs the original cosm_new1_vast run):
#   +model.ba_uncond_face_fix=true                     (F1: sane uncond face prompt under CFG)
#   pretrained_model_for_validation_name_or_path=null  (validate on the training base;
#                                                       safe now thanks to the
#                                                       ensure_branched_after_eval fix)
#
# R1   train_branched_ca_lora=false      freeze cross-attn branches (largest drift)
# R2   loss_kind=blended_masked          smooth face loss instead of hard alternation
# R3   grad clip + weight decay + lower lr
# R1R3 R1 and R3 combined (recommended first rerun)
#
# NOTE: hydra rejects duplicate overrides, so the keys that vary per run
# (train_branched_ca_lora, lr_for_lora, loss_kind, ...) are set ONLY via
# VARIANT_OVERRIDES below; everything else matches start_ba_cosm_new1_vast.sh.

VARIANT="${1:-R1R3}"
shift || true

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

case "${VARIANT}" in
R1)
    VARIANT_OVERRIDES=(
        train_branched_ca_lora=false
        lr_for_lora=1e-4
        loss_kind=masked_alternating
    )
    ;;
R2)
    VARIANT_OVERRIDES=(
        train_branched_ca_lora=true
        lr_for_lora=1e-4
        loss_kind=blended_masked
        lambda_face=0.15
    )
    ;;
R3)
    VARIANT_OVERRIDES=(
        train_branched_ca_lora=true
        lr_for_lora=3e-5
        loss_kind=masked_alternating
        trainer.max_grad_norm=1.0
        optimizer.weight_decay=1e-2
    )
    ;;
R1R3)
    VARIANT_OVERRIDES=(
        train_branched_ca_lora=false
        lr_for_lora=3e-5
        loss_kind=masked_alternating
        trainer.max_grad_norm=1.0
        optimizer.weight_decay=1e-2
    )
    ;;
*)
    log "Unknown variant: ${VARIANT} (expected R1|R2|R3|R1R3)"
    exit 1
    ;;
esac

log "Variant ${VARIANT}: ${VARIANT_OVERRIDES[*]}"

if ACCELERATE_LOG_LEVEL=error \
    TRANSFORMERS_VERBOSITY=error \
    DIFFUSERS_VERBOSITY=error \
    PYTHONWARNINGS="ignore::FutureWarning" \
    COMET_DISABLE_AUTO_LOGGING=1 \
    COMET_LOGGING_CONSOLE=ERROR \
    CUDA_VISIBLE_DEVICES=0 \
    COMET_API_KEY="${COMET_API_KEY:?export COMET_API_KEY first}" \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 train.py \
        --config-name=one_id_09Feb_testing \
        datasets=all_datasets \
        train_dataset_name=cosmic_large_vast \
        datasets.train.cosmic_large_vast.num_refs=1 \
        val_datasets_names='[manual_val_two]' \
        trainer.epoch_len=2000 \
        dataloaders.train.batch_size=2 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=2000 \
        model.weight_dtype=bf16 \
        pipeline.variant=null \
        dataloaders.manual_val_two.batch_size=4 \
        datasets.val.manual_val_two.limit=24 \
        val_debug=false \
        branched_attn_weight_mode=noise_and_ref \
        branched_attn_new_weight_kind=lora \
        automatic_bboxes=true \
        automatic_bboxes_every_val=false \
        force_log_first_auto_bbox=true \
        ba_patch_top_k=1.0 \
        ba_train_top_k=1.0 \
        non_ba_train=false \
        train_ba_only=true \
        trainer.masked_loss_step=2 \
        train_ba_all_steps=true \
        metrics=all_metrics \
        +model.ba_uncond_face_fix=true \
        pretrained_model_for_validation_name_or_path=null \
        writer=cometml writer.run_name="cosm_new1_vast_${VARIANT}" \
        "${VARIANT_OVERRIDES[@]}" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
