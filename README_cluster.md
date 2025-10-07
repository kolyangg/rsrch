## Photomaker

```bash
cd PhotoMaker
conda create --name photomaker_NS python=3.10
conda activate photomaker_NS
pip install -U pip
pip install -r pm_requirements.txt

python3 scripts/create_manual_val_id_embeds.py   --images-dir ../dataset_full/val_dataset/references   --output ../dataset_full/val_dataset/id_embeds_manual_val.pth

CUDA_VISIBLE_DEVICES=0     WANDB_API_KEY=XXX     accelerate launch --config_file=src/configs/ddp/accelerate.yaml train.py     --config-name=photomaker_train_lora     trainer.epoch_len=5     dataloaders.train.batch_size=4     dataloaders.train.num_workers=12     model.rank=16     validation_args.num_images_per_prompt=3     lr_scheduler.warmup_steps=2000     writer=wandb writer.run_name=photomaker_bf16 model.weight_dtype=bf16
```
