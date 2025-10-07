## Photomaker

```bash

conda create --name photomaker_NS python=3.10
conda activate photomaker_NS
pip install -U pip

cd diffusion_template
pip install -r hpc_requirements.txt

python3 scripts/create_manual_val_id_embeds.py   --images-dir ../dataset_full/val_dataset/references   --output ../dataset_full/val_dataset/id_embeds_manual_val.pth

sbatch script_small.sbatch
```
