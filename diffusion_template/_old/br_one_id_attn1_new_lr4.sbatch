#!/bin/bash
#SBATCH --nodes=1                    # Количество используемых узлов
#SBATCH --job-name=br_1r_sp             # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/pm_branched-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/pm_branched-%j.log   # Файл для вывода результатов
#SBATCH --gpus=2                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=8            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=40:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
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


# Branched version
CUDA_VISIBLE_DEVICES=0,1 \
    COMET_API_KEY=wSzl6h2PsRcopvISb2TJvtkzH \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=2 train.py \
    --config-name=one_id_br_attn1_lr4 \
    trainer.epoch_len=500 \
    dataloaders.train.batch_size=4 \
    dataloaders.train.num_workers=12 \
    model.rank=16 \
    validation_args.num_images_per_prompt=1 \
    lr_scheduler.warmup_steps=20 \
    writer=cometml writer.run_name=photomaker_bf16 model.weight_dtype=bf16 \
    dataloaders.manual_val.batch_size=4

