#!/usr/bin/env bash
set -euo pipefail

# N22: mechanism probe after the 9 Jul BA audit.
# Tests a more PhotoMaker-like recipe without fine-tuning ID-loss weights:
#   - original-style masked_alternating loss (50% face-mask steps)
#   - train_ba_all_steps=false to match the inference BA schedule
#   - frozen branched CA runtime (train_branched_ca_lora=false)
#   - opt-in SA runtime knobs: pose_adapt_ratio=0.25 and ca_mixing_for_face=true
#   - id_embeds face strategy with trainable SA id_to_hidden projection
#
# Run length: 10k steps for comparison with N17/N20 at 10k. If memory is tight,
# rerun with `dataloaders.train.batch_size=1` appended to the command.

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
COMET_API_KEY="${COMET_API_KEY:-}"
export PM_PATH COMET_API_KEY

if [[ -z "${COMET_API_KEY}" ]]; then
    echo "COMET_API_KEY is not set; export it before running N22." >&2
    exit 2
fi

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
        --config-name=one_id_ba_runtime_idemb_alt_N22 \
        datasets=all_datasets \
        train_dataset_name=cosmic_large_vast \
        datasets.train.cosmic_large_vast.num_refs=1 \
        val_datasets_names='[manual_val_two]' \
        trainer.epoch_len=1000 \
        trainer.n_epochs=10 \
        dataloaders.train.batch_size=2 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        model.weight_dtype=bf16 \
        model.use_id_embeds=true \
        model.ba_enable_runtime_sa_knobs=true \
        model.ba_train_sa_id_embed_proj=true \
        model.ba_ca_train_mode=ref_only \
        model.use_id_loss=false \
        model.id_loss_weight=0.0 \
        +model.ba_uncond_face_fix=true \
        +model.ba_face_prompt_mode=id_only \
        pipeline.face_embed_strategy=id_embeds \
        pipeline.use_id_embeds=true \
        pipeline.ba_enable_runtime_sa_knobs=true \
        pipeline.pose_adapt_ratio=0.25 \
        pipeline.ca_mixing_for_face=true \
        validation_args.face_embed_strategy=id_embeds \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=200 \
        pipeline.variant=null \
        dataloaders.manual_val_two.batch_size=4 \
        datasets.val.manual_val_two.limit=24 \
        val_debug=false \
        branched_attn_weight_mode=noise_and_ref \
        branched_attn_new_weight_kind=lora \
        lr_for_lora=1e-4 \
        ba_noise_lr_scale=0.1 \
        ba_ca_lr_scale=0.1 \
        trainer.max_grad_norm=1.0 \
        optimizer.weight_decay=1e-3 \
        loss_kind=masked_alternating \
        trainer.masked_loss_step=2 \
        train_branched_ca_lora=false \
        ba_patch_top_k=1.0 \
        ba_train_top_k=1.0 \
        non_ba_train=false \
        train_ba_only=true \
        train_ba_all_steps=false \
        automatic_bboxes=true \
        automatic_bboxes_every_val=false \
        force_log_first_auto_bbox=true \
        pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0 \
        metrics=all_metrics \
        writer=cometml writer.run_name="ba_runtime_idemb_alt_N22" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
