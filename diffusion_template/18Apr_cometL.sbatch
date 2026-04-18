#!/bin/bash
#SBATCH --nodes=1                    # Количество используемых узлов
#SBATCH --job-name=br_face            # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/pm_branched-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/pm_branched-%j.log   # Файл для вывода результатов
#SBATCH --gpus=1                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=8            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=60:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_e]"   # Выбор, какие узлы использовать. 
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023

source deactivate
echo "old env deactivated"

conda activate photomaker_NS
echo "photomaker_NS env activated"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 
echo "lib_path exported"
export HYDRA_FULL_ERROR=1


# Sanity checks: these must point inside your conda env
which python
which accelerate
python - <<'PY'
import sys, hydra, omegaconf, accelerate
print("PY:", sys.executable)
print("hydra-core:", hydra.version)
print("omegaconf:", omegaconf.version)
print("accelerate:", accelerate.version)
print("sys.path has .local? ->", any(".local" in p for p in sys.path))
PY

# HTTPS_PROXY=https://373:egXSTyRwQSOy@xy.2a2i.org:4443Denial=420 \

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Git provenance (printed into SLURM log)
echo "=== GIT INFO ==="
echo "PWD: $(pwd)"
GIT_TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$GIT_TOPLEVEL" ]; then
    echo "ERROR: not inside a git repository"
    exit 1
fi
echo "REPO: $GIT_TOPLEVEL"
echo "BRANCH: $(git rev-parse --abbrev-ref HEAD)"
echo "COMMIT: $(git rev-parse HEAD)"
echo "SHORT:  $(git rev-parse --short HEAD)"
echo "STATUS:"
git status --short
echo "================"

if [ "$(git rev-parse --abbrev-ref HEAD)" != "main_clean" ]; then
    echo "ERROR: expected branch main_clean"
    exit 1
fi


# Branched version
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
    train_dataset_name=cosmic_large_clust \
    val_datasets_names='[manual_val]' \
    trainer.epoch_len=2000 \
    dataloaders.train.batch_size=8 \
    dataloaders.train.num_workers=12 \
    model.rank=32 \
    model.photomaker_path="${PM_PATH}" \
    validation_args.num_images_per_prompt=1 \
    lr_scheduler.warmup_steps=2000 \
    model.weight_dtype=bf16 \
    pipeline.variant=null \
    dataloaders.manual_val.batch_size=12 \
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
    writer=cometml writer.run_name="cometL_256_new_clust"; then