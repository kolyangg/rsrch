#!/usr/bin/env bash
set -euo pipefail

# N5: NOISE-PATHWAY ABLATION. Identical to N4 except the noise pathway is FROZEN
# (ba_noise_lr_scale=0.0). Analysis: debug_04Jul/04Jul_findings.md §9 + debug_04Jul/overnight_N4_N6_plan.md.
#
# Purpose: isolate whether the noise cross-attn pathway is the face-DAMAGE vector. §4.2 says the
# noise-CA group renders the whole gen image (face included) and is what warps face color / melts
# props (the orange/melt cast growing across N3a steps). N5 freezes it (noise clones stay at base;
# only the ref_to_* pathway trains) with EVERYTHING ELSE equal to N4.
#
# Read of the N4-vs-N5 pair:
#   - N5 clean (no melt) AND id-sim >= N4  -> the noise pathway is the damage vector; the fix is to
#     make noise trainable-but-not-melting (e.g. an identity loss), not the current MSE.
#   - N5 clean BUT face<->body inconsistency (the "keanu smear") returns -> confirms BOTH pathways
#     are needed (constraint #2) yet noise melts -> problem is well-posed: need noise to train
#     without melting.
#   - N5 also crashes below step-0 -> the MSE OBJECTIVE (not the noise pathway) degrades identity;
#     escalate to an identity loss regardless of pathway.
#
# NOTE: with noise frozen, the forward is numerically ref_only-equivalent, but this is a CLEAN,
# matched-recipe diagnostic (vs the old refonly1 run, which differed in loss/jitter/LR/boost). It
# is a diagnostic, not a production candidate — constraint #2 (train both) still stands.
#
# vs N4 (start_ba_nr_alt_vast_N4.sh):
#   +ba_noise_lr_scale 0.1 -> 0.0          freeze the noise pathway (noise group LR = 1e-4*0 = 0)
#   trainer.epoch_len 500 -> 1000          coarser val (1000/2000/3000) to save overnight time;
#   trainer.n_epochs 6 -> 3                N4's fine 500-cadence already maps the early crash shape
#   writer.run_name -> ba_nr_alt_N5
# Everything else identical to N4: alt loss, noise_and_ref, clean ref (no jitter), lr 1e-4, wd 1e-3,
# grad-clip 1.0, warmup 200, uncond_face_fix, id_only, RealVis val. 3000 steps, then exits.
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
        +ba_noise_lr_scale=0.0 \
        trainer.max_grad_norm=1.0 \
        optimizer.weight_decay=1e-3 \
        loss_kind=masked_alternating \
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
        writer=cometml writer.run_name="ba_nr_alt_N5" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
