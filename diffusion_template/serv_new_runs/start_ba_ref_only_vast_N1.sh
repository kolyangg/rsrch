#!/usr/bin/env bash
set -euo pipefail

# N1: "branched attention trained as intended" run.
# Plan/rationale: debug_planning_03Jul/ba_training_fix_plan_v2.md
#
# vs cosm_new1_vast (the failing run):
#   branched_attn_weight_mode=ref_only     only reference-side weights train
#                                          (SA ref branch + face-branch K/V + CA ref half);
#                                          the gen/noise pathway stays at base weights,
#                                          so whole-image drift is structurally impossible
#   +model.ba_uncond_face_fix=true         F1: plain negative prompt for the uncond face branch
#   +model.ba_face_prompt_mode=full_boosted B1: full fused prompt with boosted ID tokens
#                                          (known-good, no zero-token attention sinks)
#   loss_kind=blended_masked lambda_face=0.2  smooth face weighting (no hard alternation)
#   lr 5e-5 + grad clip 1.0 + wd 1e-2      drift hygiene
#   pretrained_model_for_validation_name_or_path=null  validate on the training base
#                                          (safe after the ensure_branched_after_eval fix)
#   val dataset = manual_val_two (default)  SAME dataset as start_ba_cosm_new1_vast.sh
#                                          (../dataset_full/val_dataset/references_two)
#   train ref-crop jitter                  margin U(0.2,0.6) + sharpness jitter p=0.5
#                                          (runtime augmentation on the same cosmic_large_vast set)
#   ba_norm/* Comet curves                 drift canary (sa/ca x ref/noise), every 50 steps
#
# Self-contained: COMET_API_KEY and PM_PATH are baked in below (same values as
# start_ba_cosm_new1_vast.sh / serv_new_runs/.env), so this runs with a single command
# and needs no exported variables. Override by exporting COMET_API_KEY / PM_PATH first.

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

# Baked-in secrets/paths so the run needs no exported variables (single command).
# Both fall back to these defaults but can be overridden by exporting them first.
PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
COMET_API_KEY="${COMET_API_KEY:-wSzl6h2PsRcopvISb2TJvtkzH}"
export PM_PATH COMET_API_KEY

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if ACCELERATE_LOG_LEVEL=error \
    TRANSFORMERS_VERBOSITY=error \
    DIFFUSERS_VERBOSITY=error \
    PYTHONWARNINGS="ignore::FutureWarning" \
    COMET_DISABLE_AUTO_LOGGING=1 \
    COMET_LOGGING_CONSOLE=ERROR \
    CUDA_VISIBLE_DEVICES=0 \
    COMET_API_KEY="${COMET_API_KEY}" \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 train.py \
        --config-name=one_id_09Feb_testing \
        datasets=all_datasets \
        train_dataset_name=cosmic_large_vast \
        datasets.train.cosmic_large_vast.num_refs=1 \
        +datasets.train.cosmic_large_vast.ref_crop_margin_min=0.2 \
        +datasets.train.cosmic_large_vast.ref_crop_margin_max=0.6 \
        +datasets.train.cosmic_large_vast.ref_downscale_jitter=0.5 \
        val_datasets_names='[manual_val_two]' \
        trainer.epoch_len=2000 \
        dataloaders.train.batch_size=2 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        +model.ba_uncond_face_fix=true \
        +model.ba_face_prompt_mode=full_boosted \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=2000 \
        model.weight_dtype=bf16 \
        pipeline.variant=null \
        dataloaders.manual_val_two.batch_size=4 \
        datasets.val.manual_val_two.limit=24 \
        val_debug=false \
        branched_attn_weight_mode=ref_only \
        branched_attn_new_weight_kind=lora \
        lr_for_lora=5e-5 \
        trainer.max_grad_norm=1.0 \
        optimizer.weight_decay=1e-2 \
        loss_kind=blended_masked \
        lambda_face=0.2 \
        automatic_bboxes=true \
        automatic_bboxes_every_val=false \
        force_log_first_auto_bbox=true \
        train_branched_ca_lora=true \
        ba_patch_top_k=1.0 \
        ba_train_top_k=1.0 \
        non_ba_train=false \
        train_ba_only=true \
        trainer.masked_loss_step=2 \
        train_ba_all_steps=true \
        pretrained_model_for_validation_name_or_path=null \
        metrics=all_metrics \
        writer=cometml writer.run_name="ba_refonly_N1" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
