#!/usr/bin/env bash
set -euo pipefail

# N6: LOSS-FUNCTION ABLATION. Identical to N4 except the loss is blended_masked (lambda_face=0.15)
# instead of masked_alternating. Analysis: debug_04Jul/04Jul_findings.md §9 +
# debug_04Jul/overnight_N4_N6_plan.md.
#
# Purpose: test whether the every-step full-image anchor of blended_masked reduces the noise-CA
# melt vs alternating. §4.2: masked_alternating makes every 2nd step's loss ONLY the face crop, so
# the noise-CA group trains with no background anchor on those steps — the drift/melt driver.
# blended_masked keeps a (1-lambda) full-image term in EVERY step, which should damp that drift.
# This also runs the alt-vs-blend comparison the user originally wanted, now on the improved N4
# recipe (clean ref + damped noise + fast warmup).
#
# Read of the N4-vs-N6 pair (same noise damper 0.1, only loss differs):
#   - N6 less melt / higher id-sim than N4 -> blended is the better loss; adopt it.
#   - N6 ~ N4 -> loss shape is not the lever here; the damper + objective dominate.
#
# vs N4 (start_ba_nr_alt_vast_N4.sh):
#   loss_kind masked_alternating -> blended_masked
#   +lambda_face=0.15                       (between initial 0.1 and refonly1 0.2)
#   trainer.epoch_len 500 -> 1000           coarser val to save overnight time
#   trainer.n_epochs 6 -> 3                 3000 steps total
#   writer.run_name -> ba_nr_blend_N6
# Everything else identical to N4: noise_and_ref, ba_noise_lr_scale=0.1, clean ref (no jitter),
# lr 1e-4, wd 1e-3, grad-clip 1.0, warmup 200, uncond_face_fix, id_only, RealVis val. Exits at 3000.
#
# NB masked_loss_step=2 is still passed but is INERT for blended_masked (that loss ignores the
# alternation flag and always blends full+face) — kept only so the override list matches N4/N5.
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
        trainer.n_epochs=3 \
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
        train_branched_ca_lora=true \
        ba_patch_top_k=1.0 \
        ba_train_top_k=1.0 \
        non_ba_train=false \
        train_ba_only=true \
        trainer.masked_loss_step=2 \
        train_ba_all_steps=true \
        pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0 \
        metrics=all_metrics \
        writer=cometml writer.run_name="ba_nr_blend_N6" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
