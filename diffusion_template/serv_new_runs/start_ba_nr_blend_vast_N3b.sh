#!/usr/bin/env bash
set -euo pipefail

# N3b: noise_and_ref + drift hygiene + BLENDED loss (blended_masked, lambda_face=0.15).
# Analysis & rationale: debug_04Jul/04Jul_findings.md (§4.2 drift engine, §5 proposal, §6 A/B).
#
# IDENTICAL to start_ba_nr_alt_vast_N3a.sh except the loss:
#   loss_kind=blended_masked lambda_face=0.15
# Rationale (§4.2): masked_alternating makes every 2nd step's loss ONLY the face bbox crop; in
# those steps the noise CA group (which renders the whole gen image, face included — CA has no
# face/bg branching for the gen half) trains with zero background anchor. That is the diagnosed
# drift engine of the initial run (attn2 noise_to_v = largest delta; near-doubling per 2k steps).
# blended_masked keeps a (1-lambda) full-image anchor in EVERY step. lambda 0.15 sits between the
# initial run's config (0.1, unused by the alternating loss) and refonly1's 0.2.
#
# Run order: N3a (original loss + fixes) first, then this on top — the pair isolates the loss
# function's contribution with everything else held fixed.
#
# Canary gate: after 2 epochs (4k steps) check Comet ba_norm/ca_noise — flat/sublinear = good;
# then train to 20k steps (10 epochs) for the equal-step comparison vs initial@20k and
# refonly@e10(=20k).
#
# Self-contained: COMET_API_KEY and PM_PATH baked in (same as start_ba_cosm_new1_vast.sh);
# override by exporting them first.

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

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
        +model.ba_face_prompt_mode=id_only \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=2000 \
        model.weight_dtype=bf16 \
        pipeline.variant=null \
        dataloaders.manual_val_two.batch_size=4 \
        datasets.val.manual_val_two.limit=24 \
        val_debug=false \
        branched_attn_weight_mode=noise_and_ref \
        branched_attn_new_weight_kind=lora \
        lr_for_lora=5e-5 \
        +ba_noise_lr_scale=0.25 \
        trainer.max_grad_norm=1.0 \
        optimizer.weight_decay=1e-2 \
        loss_kind=blended_masked \
        lambda_face=0.15 \
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
        pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0 \
        metrics=all_metrics \
        writer=cometml writer.run_name="ba_nr_blend_N3b" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
