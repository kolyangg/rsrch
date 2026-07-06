#!/usr/bin/env bash
set -euo pipefail

# ============================================================================================
# N15 = SA-only (freeze branched cross-attn), 6000 steps, bs=2. Ablation vs N14: does the ID loss
# add on top of SA-only? Same as the N11 winner but 2x longer. See
# debug_04Jul/8Jul_results_N10_N11_N13_and_next.md §5.
# ============================================================================================

# N11: FREEZE THE CROSS-ATTENTION BRANCH (train self-attention only). Substantial pathway ablation.
# Analysis & design: debug_04Jul/7Jul_experiments_analysis.md §5 (2nd matrix).
#
# Change vs the N6 blended anchor: train_branched_ca_lora=true -> FALSE. The branched CROSS-attn
# processors (attn2 ref_to_* / noise_to_*) stay at base weights; only the branched SELF-attn
# processors (attn1, the spatial face injection) train. Motivation: §4.2 identified the CA noise
# pathway (attn2 noise_to_v, the whole-image text channel) as the drift/melt-prone one. This tests
# whether the SA branch alone yields cleaner identity (less prop-melt on Skiing/Kickboxing) — a
# direct test of the "CA is the damage pathway" hypothesis (N5 froze noise on BOTH attn1+attn2 and
# didn't help; N11 instead freezes CA on BOTH ref+noise, the complementary cut).
#
# Everything else = N6: blended_masked λ0.15, noise_and_ref, ba_noise_lr_scale=0.1, clean ref
# (no jitter), lr 1e-4, wd 1e-3, grad-clip 1.0, warmup 200, uncond_face_fix, id_only, RealVis val,
# epoch_len=1000, n_epochs=3 -> 3000 steps, val at 1000/2000/3000, then exits cleanly.
#
# Self-contained: COMET_API_KEY and PM_PATH baked in (override by exporting them first).

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
    CUDA_VISIBLE_DEVICES=0 \
    COMET_API_KEY="${COMET_API_KEY}" \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 train.py \
        --config-name=one_id_09Feb_testing \
        datasets=all_datasets \
        train_dataset_name=cosmic_large_vast \
        datasets.train.cosmic_large_vast.num_refs=1 \
        val_datasets_names='[manual_val_two]' \
        trainer.epoch_len=1000 \
        trainer.n_epochs=6 \
        dataloaders.train.batch_size=2 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        +model.ba_uncond_face_fix=true \
        +model.ba_face_prompt_mode=id_only \
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
        writer=cometml writer.run_name="ba_saonly6k_N15" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
