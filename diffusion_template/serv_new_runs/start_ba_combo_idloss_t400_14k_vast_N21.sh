#!/usr/bin/env bash
set -euo pipefail

# ============================================================================================
# N21 = N17/N14 core recipe, but reduce cumulative ID-loss pressure by gating timesteps.
#
# Why after N20:
#   N20 lowered ID-loss weight from 0.1 -> 0.075 and reached only 0.3238 at 10k full-val,
#   below N17@10k (0.3431), N17@12k (0.3500), and N14@6k (0.3324). It did reduce some late
#   N17 collapses, but the identity signal became too weak on Jisoo/Keanu/Lex/Eddie.
#
# Change vs N17:
#   keep +model.id_loss_weight=0.1, but gate ID loss more tightly:
#   +model.id_loss_max_timestep: 500 -> 400
#
# Rationale:
#   Preserve per-step ID-loss strength when x0 is cleanest, while reducing total gated steps
#   and avoiding the broad under-strength seen in N20.
#
# Keep:
#   train_branched_ca_lora=false (freeze CA), blended λ0.15, noise_and_ref, ba_noise_lr_scale=0.1,
#   lr 1e-4, wd 1e-3, grad clip 1.0, warmup 200, id_only, uncond_face_fix, RealVis val, bs=1.
#
# Length:
#   14000 steps = trainer.n_epochs=14 x epoch_len=1000, checkpoint every 1000.
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
        trainer.n_epochs=14 \
        dataloaders.train.batch_size=1 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        +model.ba_uncond_face_fix=true \
        +model.ba_face_prompt_mode=id_only \
        +model.use_id_loss=true \
        +model.id_loss_weight=0.1 \
        +model.id_loss_max_timestep=400 \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=200 \
        model.weight_dtype=bf16 \
        pipeline.variant=null \
        dataloaders.manual_val_two.batch_size=4 \
        datasets.val.manual_val_two.limit=24 \
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
        writer=cometml writer.run_name="ba_combo_idloss_t400_14k_N21" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
