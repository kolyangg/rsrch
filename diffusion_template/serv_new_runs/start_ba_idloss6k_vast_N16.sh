#!/usr/bin/env bash
set -euo pipefail

# ============================================================================================
# N16 = ID loss only (CA branch still trained), 6000 steps, bs=1. Ablation vs N14: does freeze-CA
# add on top of the ID loss? Same as the N13 winner but 2x longer. See
# debug_04Jul/8Jul_results_N10_N11_N13_and_next.md §5.
# ============================================================================================

# N13: IDENTITY LOSS (the flagship — the OBJECTIVE change). Runs FIRST in the batch.
# Analysis & design: debug_04Jul/7Jul_experiments_analysis.md §7. Implementation: src/loss/id_loss.py.
#
# The diagnosed ceiling (04Jul §9, 7Jul §2): plain denoising MSE rewards reconstructing the training
# image, NOT identity, so trained checkpoints recover toward but never beat the untrained step-0
# baseline (~0.40). N13 adds an auxiliary IDENTITY loss that directly rewards the generated face
# matching the reference identity — the only lever likely to push above 0.40.
#
# How it works (fully differentiable, gradients reach the BA weights): each low-noise step, the
# predicted x0 is decoded, the face is cropped at the gen bbox, embedded with a frozen FaceNet
# (facenet-pytorch InceptionResnetV1 / VGGFace2), and compared by cosine distance to the FaceNet
# embedding of the ground-truth face. Verified end-to-end on a local smoke run (id_loss finite ~0.3,
# gradients flow, no OOM at bs=1 on 16 GB).
#
# vs the N6 blended anchor, ADD only (easy on/off — all under +model.*):
#   +model.use_id_loss=true              enable the ID loss (default false = zero overhead)
#   +model.id_loss_weight=0.1            weight on the ID term. Smoke showed weight 0.5 makes the ID
#                                        term ~4x the MSE (too dominant); 0.1 keeps it a meaningful
#                                        but non-dominant nudge. TUNE THIS if id-sim barely moves
#                                        (raise) or the base image degrades (lower).
#   +model.id_loss_max_timestep=500      only apply when the sampled t <= 500 (x0 is meaningful);
#                                        the whole batch shares one t/step, so high-noise steps skip
#                                        the VAE decode entirely.
#   writer.run_name -> ba_idloss6k_N16
# Everything else = N6: blended_masked λ0.15, noise_and_ref, ba_noise_lr_scale=0.1, clean ref
# (no jitter), lr 1e-4, wd 1e-3, grad-clip 1.0, warmup 200, uncond_face_fix, id_only, RealVis val,
# epoch_len=1000, n_epochs=3 -> 3000 steps, val at 1000/2000/3000, then exits cleanly.
# NB the ID loss adds a VAE decode on gated steps -> somewhat slower than N10-12; watch train/id_loss
# in Comet (should trend DOWN).
#
# MEMORY: batch_size is 1 here (N10-N12 use 2). Branched training at bs=2 already sits at ~47/47.6 GB
# on this card, so the ID-loss VAE decode has no headroom and OOMs intermittently on low-noise steps.
# bs=1 halves the base activation memory, leaving plenty of room. The decode itself also now runs
# under VAE tiling+slicing (lora2._compute_id_loss) to keep its peak small. If you still see
# [OOM_SKIP] lines, add grad accumulation instead of raising bs, or lower id_loss_max_timestep.
#
# Self-contained: COMET_API_KEY and PM_PATH baked in (override by exporting them first).

# DEPENDENCY: needs `facenet-pytorch` (imported lazily only when use_id_loss=true).
#   INSTALL WITH:  pip install --no-deps facenet-pytorch
#   The --no-deps is REQUIRED: facenet-pytorch's metadata pins torch<2.3, so a plain install
#   uninstalls a newer torch (it clobbered the cu130 nightly once). --no-deps installs only the
#   package; its runtime deps (numpy/pillow/requests/tqdm) are already present. Only the standard
#   InceptionResnetV1 layers are used, so any modern torch works. FaceNet weights (~107 MB) download
#   on first use (needs network once). If absent, N13 fails fast at startup and the master continues
#   to N10-N12 (which do not need it).
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
        dataloaders.train.batch_size=1 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        +model.ba_uncond_face_fix=true \
        +model.ba_face_prompt_mode=id_only \
        +model.use_id_loss=true \
        +model.id_loss_weight=0.1 \
        +model.id_loss_max_timestep=500 \
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
        writer=cometml writer.run_name="ba_idloss6k_N16" \
        "$@"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
