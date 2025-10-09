#!/bin/bash
#SBATCH --nodes=1                    # Количество используемых узлов
#SBATCH --job-name=aboba             # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/aboba-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/aboba-%j.log   # Файл для вывода результатов
#SBATCH --gpus=2                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=8            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=00:30:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать. 
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023

# source deactivate
source deactivate
echo "old env deactivated"

conda activate photomaker_NS
echo "photomaker_NS env activated"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 
echo "lib_path exported"
export HYDRA_FULL_ERROR=1

# conda info
# conda list
# pip show hydra-core


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

CUDA_VISIBLE_DEVICES=0 \
    WANDB_API_KEY=XXX \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml train.py \
    --config-name=photomaker_train_lora \
    trainer.epoch_len=5000 \
    dataloaders.train.batch_size=12 \
    dataloaders.train.num_workers=12 \
    model.rank=16 \
    validation_args.num_images_per_prompt=3 \
    lr_scheduler.warmup_steps=2000 \
    writer=console writer.run_name=photomaker_bf16 model.weight_dtype=bf16