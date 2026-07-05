#!/usr/bin/env bash
set -euo pipefail

# N4: diagnostic run after N3a showed training DEGRADES identity below the untrained (step-0)
# baseline. Full analysis: debug_04Jul/04Jul_findings.md §9.
#
# N3a result (mean id-sim vs ref, 24 panels): step0 0.412 -> 2k 0.188 -> 8k 0.211 (flat).
# The initial cosm_new1 run has the same shape (step0 0.402, plateau ~0.32 through 28k) and never
# beats step0. So "train longer" cannot help. N4 tests whether the identity crash (which happens
# INSIDE steps 0-2000, currently invisible at 2000-step val cadence) has an early sweet spot, and
# whether hard-damping the noise/drift pathway lets the ref pathway climb without the orange/melt
# damage. Keeps the ORIGINAL masked_alternating loss (user preference) + noise_and_ref + RealVis.
#
# vs N3a (start_ba_nr_alt_vast_N3a.sh):
#   ref-crop jitter REMOVED                 the jitter fed the face branch blurry/variable ref
#                                           crops -> weaker identity signal; prime suspect for
#                                           N3a < initial (§9). Back to clean fixed ref crop.
#   lr_for_lora 5e-5 -> 1e-4                un-stall: initial recovered at 1e-4, N3a's 5e-5 didn't
#   +ba_noise_lr_scale 0.25 -> 0.1          hard-damp the noise pathway (the drift/melt vector);
#                                           noise group trains at 1e-4*0.1 = 1e-5. Both pathways
#                                           stay trainable (constraint #2) but noise barely moves.
#   optimizer.weight_decay 1e-2 -> 1e-3     lighter pull toward base (=good step0), less added noise
#   lr_scheduler.warmup_steps 2000 -> 200   reach target LR fast so the 0-2000 trajectory is real,
#                                           not a slow ramp (needed to read the fine early val)
#   trainer.epoch_len 2000 -> 500           THE KEY DIAGNOSTIC: val+checkpoint every 500 steps, so
#                                           the crash/peak inside 0-2000 is finally visible
#                                           (val at 500/1000/1500/2000/...). save_period=1 keeps
#                                           a weights checkpoint at each.
#   writer.run_name -> ba_nr_alt_N4
#
# Everything else identical to N3a: noise_and_ref, masked_alternating, ba_uncond_face_fix=true,
# ba_face_prompt_mode=id_only, grad-clip 1.0, RealVis val, batch_size=2, rank=32.
#
# RUN LENGTH: trainer.n_epochs=6 x epoch_len=500 = 3000 steps, then the run EXITS cleanly (return 0)
# so an overnight master script can start the next experiment. Val+checkpoint at 500/1000/.../3000.
#
# DECISION RULE (see §9.1): if the best (likely early) checkpoint id-sim > 0.40 -> real win, extend.
# If it never beats 0.40 -> MSE training can't improve identity; escalate to an identity loss
# (ArcFace/InsightFace cosine on the gen face crop) — a code change, needs approval first.
#
# Self-contained: COMET_API_KEY and PM_PATH baked in (override by exporting them first).

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"   # 0: async CUDA (faster) — N3a used 1

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
        trainer.epoch_len=500 \
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
        writer=cometml writer.run_name="ba_nr_alt_N4" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
