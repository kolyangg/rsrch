import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
import os
import datetime


warnings.filterwarnings("ignore", category=UserWarning)

def _format_numel(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def _print_trainable_summary(model, optimizer=None, max_examples: int = 6):
    """
    Print a concise summary of what is trainable/frozen.
    Uses model.requires_grad flags, and (optionally) optimizer param groups.
    """
    # --- module-level summary (major components) ---
    major = ("unet", "vae", "text_encoder", "text_encoder_2", "id_encoder")
    print("[Trainable Summary] major modules (trainable/total params):")
    for attr in major:
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        params = list(mod.parameters())
        trainable = [p for p in params if p.requires_grad]
        t_numel = sum(int(p.numel()) for p in trainable)
        a_numel = sum(int(p.numel()) for p in params)
        dt = getattr(mod, "dtype", None)
        dt_s = str(dt).replace("torch.", "") if dt is not None else "?"
        print(f"  - {attr}: {_format_numel(t_numel)}/{_format_numel(a_numel)}  dtype={dt_s}")

    # --- name-based categories for trainables ---
    cats = {
        "unet_lora": {"tensors": 0, "numel": 0, "examples": []},
        "unet_processors": {"tensors": 0, "numel": 0, "examples": []},
        "unet_other": {"tensors": 0, "numel": 0, "examples": []},
        "non_unet": {"tensors": 0, "numel": 0, "examples": []},
    }
    total_tensors = 0
    total_numel = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total_tensors += 1
        n = int(p.numel())
        total_numel += n
        if name.startswith("unet."):
            if "lora_A" in name or "lora_B" in name:
                key = "unet_lora"
            elif ".attn1.processor." in name or ".attn2.processor." in name:
                key = "unet_processors"
            else:
                key = "unet_other"
        else:
            key = "non_unet"
        cats[key]["tensors"] += 1
        cats[key]["numel"] += n
        if len(cats[key]["examples"]) < max_examples:
            cats[key]["examples"].append(name)

    print(f"[Trainable Summary] total trainable: {total_tensors} tensors / {_format_numel(total_numel)} params")
    for key, info in cats.items():
        if info["tensors"] == 0:
            continue
        ex = ", ".join(info["examples"])
        print(f"  - {key}: {info['tensors']} tensors / {_format_numel(info['numel'])} params  e.g. {ex}")

    # --- optimizer param groups (what is actually being optimized) ---
    if optimizer is not None:
        try:
            name_by_id = {id(p): n for n, p in model.named_parameters()}
            print("[Optimizer Groups] (name → #tensors / #params; examples):")
            for g in optimizer.param_groups:
                gname = g.get("name", "<unnamed>")
                ps = list(g.get("params", []))
                g_numel = sum(int(p.numel()) for p in ps)
                # map params to names if possible
                ex_names = []
                for p in ps:
                    n = name_by_id.get(id(p))
                    if n is not None:
                        ex_names.append(n)
                    if len(ex_names) >= max_examples:
                        break
                ex = ", ".join(ex_names) if ex_names else "-"
                print(f"  - {gname}: {len(ps)} / {_format_numel(g_numel)}  e.g. {ex}")
        except Exception:
            pass


@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_train_lora")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    # Let Accelerate own distributed init; keep long timeout for validation
    ddp_timeout = int(getattr(config, "ddp_timeout_seconds", 3600))
    pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=ddp_timeout))
    # Disable Accelerate's dataloader RNG synchronization to avoid extra
    # broadcast collectives at iterator start (can desync ranks after validation
    # on some cluster setups).
    # Also disable DDP buffer broadcasts: validation runs on rank0 only and may
    # leave rank-local non-critical buffers out of sync at train restart.
    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs, ddp_kwargs], rng_types=[])

    project_config = OmegaConf.to_container(config)
    logger = None
    writer = None
    
    if accelerator.is_main_process:
        logger = setup_saving_and_logging(config)
        # Allow resuming the same CometML experiment by passing experiment_key (run_id)
        comet_run_id = getattr(config, "cometml_id", None)
        writer = instantiate(config.writer, logger, project_config, run_id=comet_run_id)

    device = accelerator.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    ### 28 Nov: train only BA layers ###
    # Optional flag: when true, restrict training to branched attention processors only.
    train_ba_only = bool(getattr(config, "train_ba_only", False))
    # Optional flag: when true, enable clean separation of BA-specific parameters.
    ### 29 Nov - Clean separataion of BA-specific parameters ###
    ba_weights_split = bool(getattr(config, "ba_weights_split", False))
    # Optional flag: select v2 (trainable) vs legacy branched attention processors.
    use_attn_v2 = bool(getattr(config, "use_attn_v2", False)) # use attn_v1 by default (no Linear layers)
    ### 29 Nov - Clean separataion of BA-specific parameters ###
    ba_kwargs = {}
    model_target = str(getattr(getattr(config, "model", {}), "_target_", ""))
    if (
        "src.model.photomaker_branched.lora2.PhotomakerBranchedLora" in model_target
        or "src.model.photomaker_branched.lora3.PhotomakerBranchedLora" in model_target
    ):
        ba_kwargs["train_ba_only"] = train_ba_only
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        ba_kwargs["ba_weights_split"] = ba_weights_split
        ba_kwargs["use_attn_v2"] = use_attn_v2
        ### 29 Nov - Clean separataion of BA-specific parameters ###
    ### 28 Nov: train only BA layers ###

    # build model architecture, then print to console
    model = instantiate(config.model, device=device, **ba_kwargs)
    if accelerator.is_main_process:
        base_name = getattr(config.model, "pretrained_model_name_or_path", None)
        print(f"[Base Model Switch] Training base: '{base_name}'")

    ### 25 Nov: AB testing to disable BranchedCrossAttnProcessor
    # Optional flags to disable branched self- and cross-attention while keeping
    # the rest of the two-branch logic intact. Controlled via top-level config:
    #   disable_branched_sa: False by default
    #   disable_branched_ca: False by default
    disable_sa = bool(getattr(config, "disable_branched_sa", False))
    disable_ca = bool(getattr(config, "disable_branched_ca", False))
    setattr(model, "disable_branched_sa", disable_sa)
    setattr(model, "disable_branched_ca", disable_ca)
    ### 25 Nov: AB testing to disable BranchedCrossAttnProcessor

    model.prepare_for_training()

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    # build optimizer, learning rate scheduler
    trainable_params = model.get_trainable_params(config)
    optimizer = instantiate(config.optimizer, params=trainable_params)
    
    if accelerator.is_main_process:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"Param group <{group['name']}>:")
            logger.info(f"  learning rate: {group['lr']}")
            logger.info(f"  weight decay:  {group['weight_decay']}")
            logger.info(f"  betas:  {group['betas']}")
            logger.info(f"  eps:  {group['eps']}")

            # list the names or number of params
            logger.info(f"  num params:    {len(group['params'])}")

        # Print what is actually trainable/frozen for this run.
        _print_trainable_summary(model, optimizer=optimizer)

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer) 

    # Quick check: confirm optimizer includes branched-attn processor params
    if accelerator.is_main_process:
        try:
            # Map model params by id for matching against optimizer groups
            name_by_id = {id(p): n for n, p in model.named_parameters()}
            opt_ids = set()
            for g in optimizer.param_groups:
                for p in g.get("params", []):
                    opt_ids.add(id(p))

            proc_names = [
                n for n, p in model.named_parameters()
                if (".attn1.processor." in n or ".attn2.processor." in n)
            ]
            proc_in_opt = [n for n in proc_names if id(dict(model.named_parameters())[n]) in opt_ids]
            msg = (
                f"[Check] Processor params in optimizer: {len(proc_in_opt)}/{len(proc_names)}"
            )
            # Show a few examples for sanity
            if proc_in_opt:
                preview = ", ".join(proc_in_opt[:3])
                msg += f"  first: {preview}"
            print(msg)
        except Exception:
            pass

    
    train_dataloader = dataloaders["train"]
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )
    dataloaders["train"] = train_dataloader

    pipeline = None
    if accelerator.is_main_process:
        # Allow using a different pretrained base for validation-only pipeline
        val_pretrained = getattr(config, "pretrained_model_for_validation_name_or_path", None)
        prev_base = None
        if val_pretrained:
            # Temporarily override only for pipeline instantiation
            prev_base = getattr(config.pipeline, "pretrained_model_name_or_path", None)
            config.pipeline.pretrained_model_name_or_path = val_pretrained
        pipeline = instantiate(
            config.pipeline,
            model=model,
            accelerator=accelerator,
        )
        # Mirror the same branched-attn flags on the validation pipeline.
        ### 25 Nov: AB testing to disable BranchedCrossAttnProcessor
        setattr(pipeline, "disable_branched_sa", disable_sa)
        setattr(pipeline, "disable_branched_ca", disable_ca)
        ### 25 Nov: AB testing to disable BranchedCrossAttnProcessor
        if val_pretrained:
            # Restore original config value immediately after
            config.pipeline.pretrained_model_name_or_path = prev_base
        
    # Optionally resume training from a checkpoint if requested at top-level config
    resume_from = None
    if bool(getattr(config, "continue_run", False)):
        resume_from = getattr(config, "saved_checkpoint", None)

    trainer = instantiate(
        config.trainer,
        model=model,
        pipe=pipeline,
        accelerator=accelerator,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        global_config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        resume_from=resume_from,
        _recursive_=False
    )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    main()
