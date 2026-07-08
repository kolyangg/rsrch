#!/usr/bin/env bash
set -euo pipefail

# ============================================================================================
# N20 = N17 recipe with LOWER ID-loss pressure, 16000 steps, checkpoint every 1000.
#
# Why this after N17:
#   Full-val over N17 intermediate checkpoints shows the same frozen-CA combo peaks around 12k
#   by 96-image mean (12k=0.3500, final 26k=0.3482) but several visual failures worsen late:
#   over-canonical face placement, long-neck/pasted-face artifacts, and prop/occlusion collisions.
#   The likely culprit is too much cumulative ID pressure inside the fixed face box.
#
# Change vs N17:
#   +model.id_loss_weight: 0.1 -> 0.075
#
# Keep:
#   train_branched_ca_lora=false (freeze CA), blended λ0.15, noise_and_ref, ba_noise_lr_scale=0.1,
#   lr 1e-4, wd 1e-3, grad clip 1.0, warmup 200, id_only, uncond_face_fix, RealVis val, bs=1.
#
# Length:
#   16000 steps = trainer.n_epochs=16 x epoch_len=1000.
#   Validate/save every 1000 so the best checkpoint can be selected by full-val + visual review.
# ============================================================================================

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

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
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    COMET_API_KEY="${COMET_API_KEY}" \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 train.py \
        --config-name=one_id_09Feb_testing \
        datasets=all_datasets \
        train_dataset_name=cosmic_large_vast \
        datasets.train.cosmic_large_vast.num_refs=1 \
        val_datasets_names='[manual_val_two]' \
        trainer.epoch_len=1000 \
        trainer.n_epochs=16 \
        dataloaders.train.batch_size=2 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        +model.ba_uncond_face_fix=true \
        +model.ba_face_prompt_mode=id_only \
        +model.use_id_loss=true \
        +model.id_loss_weight=0.075 \
        +model.id_loss_max_timestep=500 \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=200 \
        model.weight_dtype=bf16 \
        pipeline.variant=null \
        dataloaders.manual_val_two.batch_size=8 \
        datasets.val.manual_val_two.limit=96 \
        val_debug=false \
        branched_attn_weight_mode=noise_and_ref \
        branched_attn_new_weight_kind=lora \
        lr_for_lora=1e-4 \
        +ba_noise_lr_scale=0.1 \
        trainer.max_grad_norm=1.0 \
        optimizer.weight_decay=1e-3 \
        loss_kind=blended_masked \
        lambda_face=0.15 \
        automatic_bboxes=true \
        automatic_bboxes_every_val=false \
        force_log_first_auto_bbox=true \
        train_branched_ca_lora=false \
        ba_patch_top_k=1.0 \
        ba_train_top_k=1.0 \
        non_ba_train=false \
        train_ba_only=true \
        trainer.masked_loss_step=2 \
        train_ba_all_steps=true \
        pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0 \
        metrics=all_metrics \
        writer=cometml writer.run_name="ba_combo_id075_16k_N20" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
