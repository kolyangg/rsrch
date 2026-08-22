"""Validated entry point for allowlisted clean_full training configurations."""

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
            if ".attn1.processor." in name or ".attn2.processor." in name:
                key = "unet_processors"
            elif "lora_A" in name or "lora_B" in name:
                key = "unet_lora"
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


def _assert_expected_trainable_contract(model, optimizer, config):
    """Fail closed on an explicitly declared non-standard ownership profile."""
    contract = getattr(config, "expected_trainable_contract", None)
    if contract is None or not bool(getattr(contract, "enabled", False)):
        return

    named_parameters = dict(model.named_parameters())
    trainable = {
        name: parameter
        for name, parameter in named_parameters.items()
        if parameter.requires_grad
    }
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    trainable_ids = {id(parameter) for parameter in trainable.values()}
    unknown_optimizer_ids = optimizer_ids - {
        id(parameter) for parameter in named_parameters.values()
    }

    actual = {
        "total_tensors": len(trainable),
        "total_parameters": sum(int(parameter.numel()) for parameter in trainable.values()),
        "optimizer_tensors": len(optimizer_ids),
        "optimizer_parameters": sum(
            int(parameter.numel())
            for parameter in optimizer_parameters
            if id(parameter) in optimizer_ids
        ),
    }
    for key, value in actual.items():
        expected = int(getattr(contract, key))
        if value != expected:
            raise RuntimeError(
                f"Expected trainable contract mismatch for {key}: "
                f"expected={expected}, actual={value}"
            )
    if len(optimizer_parameters) != len(optimizer_ids):
        raise RuntimeError("Expected trainable contract found duplicate optimizer parameters")
    if unknown_optimizer_ids or optimizer_ids != trainable_ids:
        raise RuntimeError(
            "Expected trainable contract optimizer membership mismatch: "
            f"missing={len(trainable_ids - optimizer_ids)}, "
            f"unexpected={len(optimizer_ids - trainable_ids)}, "
            f"unknown={len(unknown_optimizer_ids)}"
        )

    claimed_names = set()
    for category_name, category in contract.categories.items():
        substring = str(category.name_substring)
        matches = {
            name: parameter
            for name, parameter in trainable.items()
            if substring in name
        }
        overlap = claimed_names & set(matches)
        if overlap:
            raise RuntimeError(
                f"Expected trainable categories overlap in {category_name}: "
                f"{sorted(overlap)[:3]}"
            )
        claimed_names.update(matches)
        expected_tensors = int(category.tensors)
        expected_parameters = int(category.parameters)
        actual_parameters = sum(
            int(parameter.numel()) for parameter in matches.values()
        )
        if len(matches) != expected_tensors or actual_parameters != expected_parameters:
            raise RuntimeError(
                f"Expected trainable category mismatch for {category_name}: "
                f"expected={expected_tensors}/{expected_parameters}, "
                f"actual={len(matches)}/{actual_parameters}"
            )
    if claimed_names != set(trainable):
        raise RuntimeError(
            "Expected trainable categories do not partition ownership: "
            f"unclaimed={sorted(set(trainable) - claimed_names)[:6]}"
        )

    # 4 Aug 2026 - AICODE-NOTE: The historical E0 arm deliberately preserves
    # r4's broad fail-open ownership. This independent exact gate prevents a
    # future bug fix or adapter change from silently turning it into another run.
    print(
        "[Expected Trainable Contract] exact match: "
        f"{actual['total_tensors']} tensors / "
        f"{actual['total_parameters']} parameters"
    )


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
    ddp_find_unused = bool(getattr(config, "ddp_find_unused_parameters", False))
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=ddp_find_unused,
    )
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
    ba_train_top_k = float(getattr(config, "ba_train_top_k", 1.0))
    ba_patch_top_k = float(getattr(config, "ba_patch_top_k", 1.0))
    non_ba_train = bool(getattr(config, "non_ba_train", False))
    train_ba_all_steps = bool(getattr(config, "train_ba_all_steps", False))
    # Optional flag: when true, enable clean separation of BA-specific parameters.
    ### 29 Nov - Clean separataion of BA-specific parameters ###
    ba_weights_split = bool(getattr(config, "ba_weights_split", False))
    # Optional flag: select v2 (trainable) vs legacy branched attention processors.
    use_attn_v2 = bool(getattr(config, "use_attn_v2", False)) # use attn_v1 by default (no Linear layers)
    ### 29 Nov - Clean separataion of BA-specific parameters ###
    ba_kwargs = {}
    model_target = str(getattr(getattr(config, "model", {}), "_target_", ""))
    if (
        "src.model.photomaker_branched.clean_full_model.PhotomakerBranchedLora"
        in model_target
        or "src.model.photomaker_branched.lora3.PhotomakerBranchedLora" in model_target
    ):
        ba_kwargs["train_ba_only"] = train_ba_only
        ba_kwargs["ba_train_top_k"] = ba_train_top_k
        ba_kwargs["ba_patch_top_k"] = ba_patch_top_k
        ba_kwargs["non_ba_train"] = non_ba_train
        ba_kwargs["train_ba_all_steps"] = train_ba_all_steps
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        ba_kwargs["ba_weights_split"] = ba_weights_split
        ba_kwargs["use_attn_v2"] = use_attn_v2
        ### 29 Nov - Clean separataion of BA-specific parameters ###
    ### 28 Nov: train only BA layers ###

    # 28 Jul 2026 - AICODE-NOTE: Fresh MLS containers can deadlock when two
    # ranks populate the same model cache concurrently. This opt-in gate keeps
    # model construction identical but lets rank 0 populate the cache first.
    serialize_model_init = bool(
        getattr(config, "serialize_distributed_model_init", False)
    )
    if (
        serialize_model_init
        and accelerator.num_processes > 1
        and not accelerator.is_main_process
    ):
        print(
            f"[Distributed Init] rank={accelerator.process_index} "
            "waiting for rank 0 model-cache warmup"
        )
        accelerator.wait_for_everyone()

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
    strict_face_routing = bool(getattr(config, "strict_face_routing", False))
    setattr(model, "strict_face_routing", strict_face_routing)
    ### 25 Nov: AB testing to disable BranchedCrossAttnProcessor

    model.prepare_for_training()
    if serialize_model_init and accelerator.num_processes > 1:
        if accelerator.is_main_process:
            print("[Distributed Init] rank=0 model ready; releasing other ranks")
            accelerator.wait_for_everyone()
        accelerator.wait_for_everyone()
        print(
            f"[Distributed Init] rank={accelerator.process_index} "
            "all model replicas ready"
        )

    # get function handles of loss and metrics
    loss_kind = str(getattr(config, "loss_kind", "masked_alternating")).lower()
    if loss_kind != "masked_alternating":
        raise ValueError(
            "clean_full supports loss_kind=masked_alternating only, "
            f"got {loss_kind!r}"
        )

    loss_cfg = OmegaConf.create(OmegaConf.to_container(config.loss_function, resolve=False))
    loss_cfg["_target_"] = "src.loss.diffusion_loss.MaskedDiffusionLoss"
    if "lambda_face" in loss_cfg:
        del loss_cfg["lambda_face"]
    loss_function = instantiate(loss_cfg).to(device)

    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    # build optimizer, learning rate scheduler
    trainable_params = model.get_trainable_params(config)
    optimizer = instantiate(config.optimizer, params=trainable_params)

    _assert_expected_trainable_contract(model, optimizer, config)

    # 1 Aug 2026 - AICODE-NOTE: Inclusion-only processor counts missed 140.3M
    # unintended adapter parameters. Strict runs compare the optimizer against
    # the complete BA allowlist on every rank before Accelerate wraps the model.
    if bool(getattr(model, "strict_trainable_contract", False)):
        contract = model.assert_trainable_contract(optimizer=optimizer)
        print(
            "[BA Trainable Contract] exact match: "
            f"{contract['tensor_count']} tensors / "
            f"{contract['parameter_count']} parameters"
        )
    
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

    # Legacy diagnostic retained for historical log comparability. Strict runs
    # have already passed the exact inclusion-and-exclusion contract above.
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
        try:
            pipeline_model = accelerator.unwrap_model(model)
        except Exception:
            pipeline_model = model
        for attribute in (
            "ba_architecture_version",
            "branched_trainable_dtype",
            "ba_ref_kv_rank",
            "ba_output_rank",
            "ba_branch_q_rank",
            "ba_face_fusion_mode",
            "ba_face_branch_scale",
            "ba_gate_init",
            "ba_gate_max",
            "ba_gate_timestep",
            "ba_gate_face_area",
            "ba_mix_init",
            "ba_mix_floor",
            "ba_mix_max",
            "ba_mix_timestep",
            "ba_mix_face_area",
            "ba_reference_rms_match",
            "ba_reference_rms_clip_min",
            "ba_reference_rms_clip_max",
            "ba_mix_override",
            "ba_telemetry_enabled",
            "ba_telemetry_interval",
            "ba_require_denoise_progress",
            "ba_self_attention_groups",
            "ba_reference_loss_mode",
            "ba_enforce_reference_only_hard_route",
            "ba_hard_v1_true_reference_key_mask",
            "ba_hard_v1_branch_output_rank",
            "ba_hard_v1_reference_roi_warp",
            "ba_hardcase_mode",
            "ba_hardcase_groups",
            "ba_hardcase_fallback_mode",
            "ba_hardcase_rank",
            "ba_hardcase_gate_max",
            "ba_hardcase_roi_size",
            "ba_hardcase_face_threshold_px",
            "ba_hardcase_transition_cells",
            "ba_hardcase_ownership_hidden_dim",
            "ba_hardcase_visible_face_floor",
            "ba_hardcase_top_native_floor",
            "ba_hardcase_frequency_low_early",
            "ba_hardcase_frequency_low_late",
            "ba_hardcase_frequency_high_early",
            "ba_hardcase_frequency_high_late",
            "ba_frequency_surface_loss_enabled",
            "ba_frequency_surface_loss_groups",
            "ba_frequency_surface_top_weight",
            "ba_frequency_surface_top_low_band_factor",
            "ba_frequency_surface_visible_floor_weight",
            "ba_frequency_surface_visible_floor_ratio",
            "ba_frequency_learnable_schedule_enabled",
            "ba_frequency_learnable_low_early",
            "ba_frequency_low_late_center",
            "ba_frequency_low_late_half_range",
            "ba_frequency_high_early_center",
            "ba_frequency_high_early_half_range",
            "ba_frequency_high_late_center",
            "ba_frequency_high_late_half_range",
            "ba_frequency_schedule_anchor_weight",
            "ba_frequency_lowband_contrastive_enabled",
            "ba_frequency_lowband_contrastive_groups",
            "ba_frequency_lowband_contrastive_probability",
            "ba_frequency_lowband_contrastive_weight",
            "ba_frequency_lowband_contrastive_temperature",
            "ba_frequency_lowband_contrastive_ramp_start_step",
            "ba_frequency_lowband_contrastive_ramp_end_step",
            "ba_frequency_lowband_contrastive_detach_target_query",
            "ba_frequency_lowband_contrastive_negative_mode",
            "ba_frequency_positive_sameid_enabled",
            "ba_frequency_positive_sameid_groups",
            "ba_attention_ownership_loss_enabled",
            "ba_attention_ownership_groups",
            "ba_frequency_surface_region_mode",
            "ba_frequency_surface_contact_width",
            "ba_frequency_surface_top_interior_factor",
            "ba_frequency_surface_contact_factor",
            "ba_frequency_shared_schedule_enabled",
            "ba_frequency_shared_low_late_center",
            "ba_frequency_shared_low_late_half_range",
            "ba_frequency_shared_high_early_center",
            "ba_frequency_shared_high_early_half_range",
            "ba_frequency_shared_high_late_center",
            "ba_frequency_shared_high_late_half_range",
            "ba_roi_teacher_distill_enabled",
            "ba_roi_teacher_distill_groups",
            "ba_hardcase_roi_gate_init",
            "ba_hardcase_roi_gate_min",
            "ba_hardcase_roi_progress_min",
            "ba_hardcase_roi_rms_cap",
            "ba_visibility_ownership_v2_enabled",
            "ba_visibility_ownership_v2_groups",
            "ba_visibility_ownership_v2_dilate_cells",
            "ba_visibility_ownership_v2_min_top_area",
            "ba_visibility_ownership_v2_delta_only",
            "ba_null_key_router_enabled",
            "ba_null_key_router_groups",
            "ba_null_key_entropy_threshold",
            "ba_null_key_temperature",
            "ba_null_key_max_abstention",
            "ba_null_key_min_reference_fraction",
            "ba_landmark_canonical_kv_enabled",
            "ba_landmark_canonical_kv_groups",
            "ba_landmark_canonical_kv_mix",
            "ba_landmark_canonical_kv_min_confidence",
            "ba_component_token_memory_enabled",
            "ba_component_token_memory_groups",
            "ba_component_token_memory_scale",
            "ba_component_token_memory_sigma_cells",
            "ba_component_token_memory_min_confidence",
            "ba_identity_motion_projector_enabled",
            "ba_identity_motion_projector_groups",
            "ba_identity_motion_projector_rank",
            "ba_identity_motion_projector_gate_max",
            "ba_identity_motion_projector_ramp_start_step",
            "ba_identity_motion_projector_ramp_end_step",
            "ba_id_adaptive_modulation_enabled",
            "ba_id_adaptive_modulation_groups",
            "ba_id_adaptive_modulation_embedding_dim",
            "ba_id_adaptive_modulation_bottleneck",
            "ba_id_adaptive_modulation_scale_max",
            "ba_id_adaptive_modulation_ramp_start_step",
            "ba_id_adaptive_modulation_ramp_end_step",
            "ba_semantic_window_gate_enabled",
            "ba_semantic_window_gate_groups",
            "ba_semantic_window_gate_progress_start",
            "ba_semantic_window_gate_progress_end",
            "ba_semantic_window_gate_progress_temperature",
            "ba_semantic_window_gate_agreement_threshold",
            "ba_semantic_window_gate_agreement_temperature",
            "ba_semantic_window_gate_min_scale",
            "ba_semantic_window_gate_max_scale",
        ):
            # 17 Aug 2026 - AICODE-NOTE: Validation must use the composed experiment flags,
            # even when Accelerate's wrapper does not expose a newly added
            # model attribute. This keeps the reused processor map auditable.
            if hasattr(config.model, attribute):
                setattr(pipeline, attribute, getattr(config.model, attribute))
            elif hasattr(pipeline_model, attribute):
                setattr(pipeline, attribute, getattr(pipeline_model, attribute))
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
