#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-/workspace/.cache/insightface}"
export FACEANALYSIS_CPU="${FACEANALYSIS_CPU:-1}"

mkdir -p "${INSIGHTFACE_HOME}/models"

python - <<'PY2'
import os
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    root=os.environ["INSIGHTFACE_HOME"],
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))
print(f"InsightFace cache OK: {os.environ['INSIGHTFACE_HOME']}")
PY2

ACCELERATE_LOG_LEVEL=error \
TRANSFORMERS_VERBOSITY=error \
DIFFUSERS_VERBOSITY=error \
PYTHONWARNINGS="ignore::FutureWarning" \
COMET_DISABLE_AUTO_LOGGING=1 \
COMET_LOGGING_CONSOLE=ERROR \
CUDA_VISIBLE_DEVICES=0,1 \
    COMET_API_KEY=wSzl6h2PsRcopvISb2TJvtkzH \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml --main_process_port=29511 --num_processes=2 train.py \
        --config-name=one_id_09Feb_testing \
        datasets=all_datasets \
        train_dataset_name=cosmic_large_vast \
        val_datasets_names='[manual_val]' \
        trainer.epoch_len=2000 \
        dataloaders.train.batch_size=4 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=2000 \
        model.weight_dtype=bf16 \
        pipeline.variant=null \
        dataloaders.manual_val.batch_size=6 \
        datasets.val.manual_val.limit=96 \
        val_debug=false \
        branched_attn_weight_mode=noise_and_ref \
        branched_attn_new_weight_kind=lora \
        lr_for_lora=1e-4 \
        automatic_bboxes=true \
        automatic_bboxes_every_val=false \
        force_log_first_auto_bbox=true \
        train_branched_ca_lora=true \
        ba_patch_top_k=1.0 \
        ba_train_top_k=1.0 \
        non_ba_train=false \
        train_ba_only=true \
        loss_kind=masked_alternating \
        trainer.masked_loss_step=2 \
        train_ba_all_steps=true \
        train_on_separate_image=true \
        train_dataset_const_ref=false \
        train_dataset_crop_ref=false \
        train_dataset_ref_similar=true \
        train_dataset_crop_nonface_min=0.2 \
        train_dataset_crop_nonface_max=0.4 \
        train_dataset_upscale_to_1024=false \
        metrics=all_metrics \
        val_datasets_names='[manual_val]' \
        writer=cometml writer.run_name="cometL_256_new_vast_2gpu"