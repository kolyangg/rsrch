from abc import abstractmethod
from pathlib import Path

import gc
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.face_quality_validation import FaceQualityValidationSession
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from hydra.utils import instantiate

import os
import time


def _resolve_validation_processor_base_mode(config) -> str:
    explicit = getattr(config, "validation_processor_base_mode", None)
    if explicit is None:
        return (
            "legacy_full_copy"
            if bool(getattr(config, "update_proc_weights_val", False))
            else "no_processor_update"
        )
    mode = str(explicit).lower()
    allowed = {"legacy_full_copy", "validation_native", "no_processor_update"}
    if mode not in allowed:
        raise ValueError(
            f"Unknown validation_processor_base_mode={mode!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return mode


def _copy_full_processor_state(train_unet, val_unet, *, strict: bool) -> int:
    train_processors = getattr(train_unet, "attn_processors", {})
    val_processors = getattr(val_unet, "attn_processors", {})
    copied = 0
    for name, train_processor in train_processors.items():
        if not hasattr(train_processor, "state_dict"):
            continue
        processor_state = train_processor.state_dict()
        if not processor_state:
            continue
        val_processor = val_processors.get(name)
        if val_processor is None or not hasattr(val_processor, "load_state_dict"):
            if strict:
                raise RuntimeError(
                    f"Validation U-Net is missing stateful processor {name!r}"
                )
            continue
        try:
            val_processor.load_state_dict(processor_state, strict=strict)
        except Exception:
            if strict:
                raise
            continue
        copied += 1
    if strict and copied == 0:
        raise RuntimeError("Strict legacy processor copy found no stateful processors")
    return copied


def _snapshot_adapter_parameters(unet, adapter_marker: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in unet.named_parameters()
        if adapter_marker in name and parameter.requires_grad
    }


def _restore_adapter_parameters(
    unet,
    snapshot: dict[str, torch.Tensor],
    *,
    adapter_marker: str,
) -> int:
    named_parameters = dict(unet.named_parameters())
    current_names = {
        name
        for name, parameter in named_parameters.items()
        if adapter_marker in name and parameter.requires_grad
    }
    if current_names != set(snapshot):
        raise RuntimeError(
            "Validation shadow-adapter parameter map changed: "
            f"missing={sorted(set(snapshot) - current_names)[:3]}, "
            f"unexpected={sorted(current_names - set(snapshot))[:3]}"
        )
    with torch.no_grad():
        for name, value in snapshot.items():
            parameter = named_parameters[name]
            parameter.copy_(value.to(parameter.device, parameter.dtype))
    return len(snapshot)



class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(
        self,
        model,
        pipe,
        accelerator,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        global_config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms,
        # trainer args
        device_tensors,
        max_grad_norm,
        cfg_step,
        log_step,
        n_epochs,
        epoch_len,
        resume_from,
        from_pretrained,
        save_period,
        save_dir,
        seed,
        post_backward_parameter_touch=True,
        grad_norm_log_only=False,
        active_grad_norm_mode="every_step",
        validation_pose_adapt_ratio=None,
        validation_interval_steps=2000,
        face_quality=None,
        log_per_image_id_sim_table=True,
        skip_initial_validation=False,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = global_config
        self.device = device

        self.logger = logger
        self.log_step = log_step
        self.validation_debug_timing = bool(getattr(self.config, "validation_debug_timing", False))

        self.model = model
        self.pipe = pipe
        self.accelerator = accelerator
        self.criterion = criterion

        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms
        self.writer = writer

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        self._train_dataset_for_resume_validation = getattr(
            self.train_dataloader,
            "dataset",
            None,
        )
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = n_epochs

        self.save_period = save_period  # checkpoint each save_period epochs
        self.max_grad_norm = max_grad_norm
        self.device_tensors = device_tensors
        self.cfg_step = cfg_step
        self.post_backward_parameter_touch = bool(post_backward_parameter_touch)
        self.grad_norm_log_only = bool(grad_norm_log_only)
        self.active_grad_norm_mode = str(active_grad_norm_mode).lower()
        if self.active_grad_norm_mode not in {"every_step", "requested_only", "off"}:
            raise ValueError(
                "active_grad_norm_mode must be every_step, requested_only, or off"
            )
        self.validation_pose_adapt_ratio = validation_pose_adapt_ratio
        self.validation_interval_steps = (
            None
            if validation_interval_steps is None
            else int(validation_interval_steps)
        )
        self.face_quality_config = face_quality
        self.log_per_image_id_sim_table = bool(log_per_image_id_sim_table)
        self.skip_initial_validation = bool(skip_initial_validation)
        if (
            not bool(getattr(self.config, "validation_only", False))
            and self.validation_interval_steps is not None
            and self.validation_interval_steps > 0
            and self.validation_interval_steps % self.epoch_len != 0
        ):
            raise ValueError(
                "trainer.validation_interval_steps must be an exact multiple "
                f"of trainer.epoch_len ({self.epoch_len}); got "
                f"{self.validation_interval_steps}"
            )
        

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker()
        self.evaluation_metrics = MetricTracker()

        # define checkpoint dir and init everything if required
        self.checkpoint_dir = (
            ROOT_PATH / save_dir / self.config.writer.run_name
        )

        if resume_from is not None:
            resume_path = self.checkpoint_dir / resume_from
            self._resume_checkpoint(resume_path)

        if from_pretrained is not None:
            self._from_pretrained(from_pretrained)

        # Keep epoch counters identical across ranks. If one rank accidentally
        # resumes from a different epoch, rank-conditional first-epoch logic
        # (initial validation, barriers) can desynchronize collectives.
        self._sync_start_epoch()

        # 1 Aug 2026 - Scheduled datasets encode exact global sample order.
        # AICODE-NOTE: Validate after checkpoint loading sets start_epoch but
        # before the lazy inf_loop consumes its first batch.
        validate_resume_position = getattr(
            self._train_dataset_for_resume_validation,
            "validate_resume_position",
            None,
        )
        if validate_resume_position is not None:
            completed_steps = (self.start_epoch - 1) * self.epoch_len
            validate_resume_position(completed_steps)


    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        # 25 Jul 2026 - Evaluation jobs must be able to reuse the trainer's
        # exact batched validation/Comet path without taking an optimizer step.
        # AICODE-NOTE: This opt-in branch is intentionally before the training
        # interrupt handler so validation-only interruption never saves a new
        # checkpoint that could be mistaken for a trained endpoint.
        if bool(getattr(self.config, "validation_only", False)):
            self._validate_only()
            return

        try:
            self._train_process()
        except KeyboardInterrupt as e:
            if self.accelerator.is_main_process:
                self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch)
            raise e

    def _validate_only(self):
        """Run the configured single- or multi-checkpoint validation schedule."""
        schedule = self._validation_only_schedule()
        is_multistep = getattr(self.config, "validation_epochs", None) is not None
        for validation_epoch, checkpoint_path in schedule:
            # 26 Jul 2026 - AICODE-NOTE: Epoch zero intentionally evaluates the
            # seeded model initialization. Later entries load source checkpoints
            # into that same model/writer so one Comet run records the trajectory.
            # The historical single-checkpoint path was already loaded by
            # __init__. Only the opt-in schedule changes weights between gates.
            if checkpoint_path is not None and is_multistep:
                self._from_pretrained(checkpoint_path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                validation_step = validation_epoch * self.epoch_len
                checkpoint_label = (
                    str(checkpoint_path)
                    if checkpoint_path is not None
                    else "seeded_initial_state"
                )
                self.logger.info(
                    "Validation-only run: checkpoint=%s epoch=%s step=%s",
                    checkpoint_label,
                    validation_epoch,
                    validation_step,
                )
                for part, dataloader in self.evaluation_dataloaders.items():
                    val_logs = self._evaluation_epoch(
                        validation_epoch,
                        part,
                        dataloader,
                    )
                    for name, value in val_logs.items():
                        self.logger.info("    %s/%s: %s", part, name, value)
            self.accelerator.wait_for_everyone()

    def _validation_only_schedule(self):
        """Resolve a backward-compatible validation-only checkpoint schedule."""
        configured_epochs = getattr(self.config, "validation_epochs", None)
        if configured_epochs is None:
            checkpoint_path = getattr(
                self.config.trainer,
                "from_pretrained",
                None,
            )
            if checkpoint_path in (None, ""):
                raise ValueError(
                    "validation_only=true requires trainer.from_pretrained to "
                    "point to a checkpoint"
                )
            validation_epoch = int(
                getattr(self.config, "validation_epoch", 0)
            )
            if validation_epoch < 0:
                raise ValueError("validation_epoch must be non-negative")
            return [(validation_epoch, checkpoint_path)]

        epochs = [int(epoch) for epoch in configured_epochs]
        configured_paths = getattr(
            self.config,
            "validation_checkpoint_paths",
            None,
        )
        if configured_paths is None:
            raise ValueError(
                "validation_epochs requires validation_checkpoint_paths"
            )
        paths = [
            None if path in (None, "") else str(path)
            for path in configured_paths
        ]
        if len(epochs) != len(paths):
            raise ValueError(
                "validation_epochs and validation_checkpoint_paths must have "
                "the same length"
            )
        if not epochs or epochs != sorted(set(epochs)):
            raise ValueError(
                "validation_epochs must be a non-empty, strictly increasing "
                "sequence"
            )
        for epoch, path in zip(epochs, paths):
            if epoch < 0:
                raise ValueError("validation epochs must be non-negative")
            if epoch == 0 and path is not None:
                raise ValueError(
                    "validation epoch zero must use the seeded initial state"
                )
            if epoch > 0 and path is None:
                raise ValueError(
                    f"validation epoch {epoch} requires a checkpoint path"
                )
            if path is not None:
                checkpoint = Path(path)
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise ValueError(
                        f"validation checkpoint is missing: {checkpoint}"
                    )
        return list(zip(epochs, paths))

    def _sync_start_epoch(self):
        """Force a single start_epoch value across all distributed ranks."""
        if int(getattr(self.accelerator, "num_processes", 1)) <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "[DDP Sync] Distributed is not initialized while num_processes > 1; "
                "cannot synchronize start_epoch safely."
            )

        local_start = int(self.start_epoch)
        rank = int(getattr(self.accelerator, "process_index", torch.distributed.get_rank()))
        world = int(getattr(self.accelerator, "num_processes", torch.distributed.get_world_size()))

        # First, audit local values to expose unexpected per-rank resume behavior.
        local_tensor = torch.tensor([local_start], device=self.device, dtype=torch.long)
        gathered_local = [torch.zeros_like(local_tensor) for _ in range(world)]
        torch.distributed.all_gather(gathered_local, local_tensor)
        all_local = [int(t.item()) for t in gathered_local]

        # Then force rank0 value on all ranks via broadcast.
        epoch_tensor = torch.tensor(
            [local_start if rank == 0 else 0],
            device=self.device,
            dtype=torch.long,
        )
        torch.distributed.broadcast(epoch_tensor, src=0)
        self.start_epoch = int(epoch_tensor.item())

        if len(set(all_local)) > 1 and self.accelerator.is_main_process:
            msg = (
                f"[DDP Sync] start_epoch mismatch across ranks: {sorted(set(all_local))}. "
                f"Using rank0 value {self.start_epoch} on all ranks."
            )
            if self.logger is not None:
                self.logger.warning(msg)
            else:
                print(msg)

        # Per-rank startup trace to verify all workers converge to one epoch.
        rank_msg = (
            f"[DDP Sync] rank={rank}/{world} start_epoch local={local_start} "
            f"all_local={all_local} synced={self.start_epoch}"
        )
        if self.accelerator.is_main_process and self.logger is not None:
            self.logger.info(rank_msg)
        else:
            print(rank_msg)

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch and evaluating it on non-train partitions
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # 28 Jul 2026 - AICODE-NOTE: Keep every rank behind rank 0 while
            # it logs and writes checkpoints. Letting another rank start its
            # next backward while rank 0 serializes state can enqueue different
            # NCCL collectives and corrupt the checkpoint on watchdog abort.
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                # save logged information into logs dict
                logs = {"epoch": epoch}
                logs.update(result)

                # print logged information to the screen
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)
                weights_only_period = int(getattr(self.config, "weights_only_save_period", 0) or 0)
                if weights_only_period > 0 and epoch % weights_only_period == 0:
                    self._save_weights_only_checkpoint(epoch)
            self.accelerator.wait_for_everyone()


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        logs = {}
        self.is_train = True
        pid = os.getpid()
        self.train_metrics.reset()

        if self.accelerator.is_main_process:
            self.writer.set_step((epoch - 1) * self.epoch_len)
            self.writer.add_scalar("general/epoch", epoch)

        # 28 Jul 2026 - AICODE-NOTE: A no-update recovery may reuse a completed
        # step-0 validation and enter training directly. This explicit opt-in
        # avoids duplicating Comet assets while preserving the default protocol.
        if epoch == 1 and not self.skip_initial_validation:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                for part, dataloader in self.evaluation_dataloaders.items():
                    val_logs = self._evaluation_epoch(epoch - 1, part, dataloader)
                    logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})
                self.is_train = True

            self.accelerator.wait_for_everyone()

            ### Modified for attention processors training ###
            # Ensure branched processors are re-installed on all ranks only when
            # validation used the training base model itself.
            val_pretrained = getattr(self.config, "pretrained_model_for_validation_name_or_path", None)
            needs_reinstall = not bool(val_pretrained)
            if needs_reinstall:
                try:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                except Exception:
                    unwrapped = self.model
                if hasattr(unwrapped, "ensure_branched_after_eval"):
                    unwrapped.ensure_branched_after_eval()
                self.accelerator.wait_for_everyone()
            ### Modified for attention processors training ###

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc=f"train_{pid}", total=self.epoch_len)
        ):

            batch["batch_idx"] = batch_idx
            batch["global_step"] = (epoch - 1) * self.epoch_len + batch_idx
            batch = self.process_batch(
                batch,
                train_metrics=self.train_metrics,
            )

            ### Modified to fix accelerate error after adding training of attn processors ###
            # --- DDP safety: make branched-attn params "participate" every step (adds zero to loss) ---
            # try:
            #     unwrapped = self.accelerator.unwrap_model(self.model)
            #     if hasattr(unwrapped, "unet") and hasattr(unwrapped.unet, "attn_processors"):
            #         extra = None
            #         for proc in unwrapped.unet.attn_processors.values():
            #             for p in getattr(proc, "parameters", lambda: [])():
            #                 if p.requires_grad:
            #                     # Touch a single element per param to keep it cheap yet connected
            #                     term = (p.reshape(-1)[:1].sum().to(torch.float32) * 0.0)
            #                     extra = term if extra is None else (extra + term)
            #         if extra is not None:
            #             batch["loss"] = batch["loss"] + extra.to(batch["loss"].dtype)
            # except Exception:
            #     pass
            
            if self.post_backward_parameter_touch:
                # Legacy diagnostic only. This executes after backward and
                # therefore cannot make parameters participate in the current
                # DDP reduction.
                try:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                except Exception:
                    unwrapped = self.model
                extra = None
                for pname, p in unwrapped.named_parameters():
                    if p.requires_grad and (
                        ".attn1.processor." in pname
                        or ".attn2.processor." in pname
                    ):
                        term = p.reshape(-1)[:1].sum().to(torch.float32) * 0.0
                        extra = term if extra is None else (extra + term)
                if extra is not None:
                    batch["loss"] = batch["loss"] + extra.to(batch["loss"].dtype)
            
            
            # --- end DDP safety ---
            ### Modified to fix accelerate error after adding training of attn processors ###

            should_log = batch_idx % self.log_step == 0
            if not self.grad_norm_log_only or should_log:
                grad_norms = self._get_grad_norms()
                for part_name, part_norm in grad_norms.items():
                    self.train_metrics.update(f"grad_norm/{part_name}", part_norm)
            
            # log current results
            if should_log:
                if self.accelerator.is_main_process:
                    self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Reduced Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )

                    lrs = self._get_lrs()
                    for part_name, part_lr in lrs.items():
                        self.writer.add_scalar(f"lrs/{part_name}", part_lr)

                    self._log_scalars(self.train_metrics, "train")
                    self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                # last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        # logs.update(last_train_metrics)

        # Run val/test at an exact optimizer-step cadence. Step zero remains a
        # separate initial validation above.
        validation_step = epoch * self.epoch_len
        should_validate = self._should_run_periodic_validation(validation_step)
        self.accelerator.wait_for_everyone()
        if should_validate and self.accelerator.is_main_process:
            for part, dataloader in self.evaluation_dataloaders.items():
                val_logs = self._evaluation_epoch(epoch, part, dataloader)
                logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})
                
        self.accelerator.wait_for_everyone()
        
        ### Modified for attention processors training ###
        # Ensure branched processors are re-installed only when validation used
        # the training base model itself.
        if should_validate:
            val_pretrained = getattr(self.config, "pretrained_model_for_validation_name_or_path", None)
            needs_reinstall = not bool(val_pretrained)
            if needs_reinstall:
                try:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                except Exception:
                    unwrapped = self.model
                if hasattr(unwrapped, "ensure_branched_after_eval"):
                    unwrapped.ensure_branched_after_eval()
        self.accelerator.wait_for_everyone()
        ### Modified for attention processors training ###

        return logs

    def _should_run_periodic_validation(self, step):
        """Return whether this optimizer step is a configured validation gate."""
        interval = self.validation_interval_steps
        if interval is None:
            return True
        if interval <= 0:
            return False
        return int(step) % interval == 0

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.evaluation_metrics.reset()
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
        except Exception:
            unwrapped = self.model
        if hasattr(unwrapped, "clear_runtime_caches"):
            unwrapped.clear_runtime_caches()
        torch.cuda.empty_cache()

        for metric in self.metrics:
            metric.to_cuda()

        validation_step = epoch * self.epoch_len
        self.writer.set_step(validation_step, part)
        self._validation_per_image_id_rows = []
        self._validation_per_image_id_next_index = 0
        face_quality_enabled = bool(
            self.face_quality_config is not None
            and self.face_quality_config.get("enabled", True)
        )
        face_quality_session = None
        if face_quality_enabled:
            face_quality_session = FaceQualityValidationSession(
                config=self.face_quality_config,
                checkpoint_dir=self.checkpoint_dir,
                writer=self.writer,
                logger=self.logger,
                part=part,
                step=validation_step,
                partition_count=len(self.evaluation_dataloaders),
            )
        prev_time = time.time()
        with torch.no_grad():
            # Optionally swap to an alternate base model for validation only
            val_pretrained = getattr(self.config, "pretrained_model_for_validation_name_or_path", None)
            _orig_pipe = getattr(self, "pipe", None)
            _val_model = None
            state = None
            default_snapshot = None
            train_unet = None
            val_unet = None
            _created_val = False
            _offloaded_train_model = False
            if val_pretrained:
                # In multi-GPU, do NOT offload the DDP-wrapped training model on a single rank.
                # This avoids desynchronizing DDP parameter buckets across ranks.
                num_procs = int(getattr(self.accelerator, "num_processes", 1))
                should_offload_train = (num_procs == 1)
                try:
                    if should_offload_train:
                        # Offload training model to CPU to free VRAM (safe on single GPU)
                        self.accelerator.unwrap_model(self.model).to("cpu")
                        _offloaded_train_model = True
                        if self.accelerator.is_main_process:
                            try:
                                print(f"[Base Model Switch] Offloaded training model to CPU (num_procs={num_procs})")
                            except Exception:
                                pass
                    else:
                        if self.accelerator.is_main_process:
                            try:
                                print(f"[Base Model Switch] Skipping offload on multi-GPU (num_procs={num_procs})")
                            except Exception:
                                pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception:
                    _offloaded_train_model = False

                # Instantiate a temporary validation model on the alternate base
                prev_model_base = getattr(self.config.model, "pretrained_model_name_or_path", None)
                try:
                    self.config.model.pretrained_model_name_or_path = val_pretrained
                    validation_ba_kwargs = {}
                    validation_model_target = str(
                        getattr(self.config.model, "_target_", "")
                    )
                    if (
                        "src.model.photomaker_branched.lora2.PhotomakerBranchedLora"
                        in validation_model_target
                        or "src.model.photomaker_branched.lora3.PhotomakerBranchedLora"
                        in validation_model_target
                    ):
                        # 2 Aug 2026 - AICODE-NOTE: These top-level routing
                        # controls are constructor invariants. Passing them only
                        # after construction can reject inference-active v2 and
                        # can install a validation architecture unlike training.
                        for attribute, default in (
                            ("train_ba_only", False),
                            ("ba_train_top_k", 1.0),
                            ("ba_patch_top_k", 1.0),
                            ("non_ba_train", False),
                            ("train_ba_all_steps", False),
                            ("ba_weights_split", False),
                            ("use_attn_v2", False),
                        ):
                            validation_ba_kwargs[attribute] = getattr(
                                self.config, attribute, default
                            )
                    _val_model = instantiate(
                        self.config.model,
                        device=self.device,
                        **validation_ba_kwargs,
                    )
                    # 25 Jul 2026 - The alternate validation model must receive
                    # the architecture toggles before processor installation.
                    # AICODE-NOTE: Setting these after prepare_for_training()
                    # leaves the already-installed branched processor map intact.
                    setattr(
                        _val_model,
                        "disable_branched_sa",
                        bool(getattr(self.config, "disable_branched_sa", False)),
                    )
                    setattr(
                        _val_model,
                        "disable_branched_ca",
                        bool(getattr(self.config, "disable_branched_ca", False)),
                    )
                    setattr(_val_model, "strict_face_routing", bool(getattr(self.config, "strict_face_routing", False)))
                    # Keep the explicit attributes aligned for historical model
                    # classes that accept but do not retain every constructor
                    # argument. New branched models already received these
                    # values through validation_ba_kwargs above.
                    for attribute, default in (
                        ("train_ba_only", False),
                        ("ba_train_top_k", 1.0),
                        ("ba_patch_top_k", 1.0),
                        ("non_ba_train", False),
                        ("train_ba_all_steps", False),
                        ("ba_weights_split", False),
                        ("use_attn_v2", False),
                    ):
                        setattr(
                            _val_model,
                            attribute,
                            getattr(self.config, attribute, default),
                        )
                    # Ensure adapters are initialized before loading LoRA weights
                    if hasattr(_val_model, "prepare_for_training"):
                        _val_model.prepare_for_training()
                    try:
                        state = self.accelerator.unwrap_model(self.model).get_state_dict()
                    except Exception:
                        state = self.model.get_state_dict()
                    processor_base_mode = _resolve_validation_processor_base_mode(
                        self.config
                    )
                    print(
                        "[Validation Processor Base] "
                        f"mode={processor_base_mode}"
                    )
                    if processor_base_mode == "no_processor_update" and isinstance(state, dict):
                        if int(state.get("schema_version", 1)) == 2:
                            raise RuntimeError(
                                "no_processor_update is incompatible with schema-v2 "
                                "trainable-only checkpoints; use validation_native or "
                                "legacy_full_copy"
                            )
                        state = dict(state)
                        state.pop("attn_processors", None)
                    shadow_default = bool(
                        getattr(
                            self.config,
                            "validation_shadow_photomaker_default",
                            False,
                        )
                    )
                    default_snapshot = (
                        _snapshot_adapter_parameters(_val_model.unet, ".default.")
                        if shadow_default
                        else None
                    )
                    if shadow_default and not default_snapshot:
                        raise RuntimeError(
                            "Requested PhotoMaker-default shadow validation but "
                            "the alternate validation model has no default adapter"
                        )
                    if hasattr(_val_model, "load_state_dict_"):
                        _val_model.load_state_dict_(state)
                    if default_snapshot is not None:
                        restored = _restore_adapter_parameters(
                            _val_model.unet,
                            default_snapshot,
                            adapter_marker=".default.",
                        )
                        print(
                            "[Validation Adapter Policy] restored pretrained "
                            f"PhotoMaker default tensors={restored}"
                        )

                    # The historical path copies base buffers as well as learned
                    # deltas. validation_native deliberately keeps the alternate
                    # validation U-Net's own processor bases.
                    if processor_base_mode == "legacy_full_copy":
                        try:
                            train_unet = self.accelerator.unwrap_model(self.model).unet
                        except Exception:
                            train_unet = getattr(self.model, "unet", None)
                        val_unet = getattr(_val_model, "unet", None)
                        if train_unet is not None and val_unet is not None:
                            copied = _copy_full_processor_state(
                                train_unet,
                                val_unet,
                                strict=bool(
                                    getattr(
                                        self.config,
                                        "strict_validation_processor_copy",
                                        False,
                                    )
                                ),
                            )
                            print(
                                "[Validation Processor Base] legacy full-copy "
                                f"stateful_processors={copied}"
                            )
                    # Move the temporary validation model to the active device (GPU)
                    try:
                        _val_model.to(self.device)
                    except Exception:
                        pass
                    _created_val = True
                finally:
                    # Restore original model base in config
                    self.config.model.pretrained_model_name_or_path = prev_model_base

                # Build a temporary pipeline bound to the validation model
                prev_pipe_base = getattr(self.config.pipeline, "pretrained_model_name_or_path", None)
                try:
                    self.config.pipeline.pretrained_model_name_or_path = val_pretrained
                    self.pipe = instantiate(
                        self.config.pipeline,
                        model=_val_model,
                        accelerator=self.accelerator,
                    )
                    setattr(
                        self.pipe,
                        "disable_branched_sa",
                        bool(getattr(self.config, "disable_branched_sa", False)),
                    )
                    setattr(
                        self.pipe,
                        "disable_branched_ca",
                        bool(getattr(self.config, "disable_branched_ca", False)),
                    )
                    # 2 Aug 2026 - Alternate-base validation must refresh the
                    # exact versioned processor runtime, not infer v3 through a
                    # residual-v2 superclass or lose its bounded-mix settings.
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
                        "ba_hard_v1_lora_rank",
                        "ba_identity_ca_v2_enabled",
                        "ba_identity_ca_v2_groups",
                        "ba_identity_ca_v2_rank",
                        # 05 Aug 2026 - AICODE-NOTE: The validation U-Net may
                        # already contain v3 processors copied from training,
                        # so its pipeline must receive the identical selector
                        # and gate contract before the first denoising patch.
                        "ba_residual_identity_ca_v3_enabled",
                        "ba_residual_identity_ca_v3_groups",
                        "ba_residual_identity_ca_v3_rank",
                        "ba_residual_identity_ca_v3_gate_init",
                        "ba_residual_identity_ca_v3_gate_max",
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
                        # 14 Aug 2026 - AICODE-NOTE: alternate-base validation
                        # reuses the model's already-installed CL27-CL29
                        # processors, so the pipeline must advertise the same
                        # extension map before its first denoising patch.
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
                        # 17 Aug 2026 - AICODE-NOTE: CL30-CL37 reuse these
                        # extension-bearing processors on the alternate
                        # validation base; the temporary pipeline must expose
                        # the same installed map before its first denoise.
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
                        "ba_reference_face_ownership_enabled",
                        "ba_reference_face_ownership_groups",
                        "ba_reference_face_ownership_probability",
                        "ba_reference_face_ownership_seed",
                        "ba_reference_face_ownership_ramp_start_step",
                        "ba_reference_face_ownership_ramp_end_step",
                        "ba_reference_face_ownership_max_strength",
                        "ba_band_reliability_gate_enabled",
                        "ba_band_reliability_gate_groups",
                        "ba_band_reliability_gate_feature_version",
                        "ba_band_reliability_gate_hidden_dim",
                        "ba_band_reliability_gate_max_delta",
                        "ba_band_reliability_gate_init_seed",
                        "ba_band_rms_cap_enabled",
                        "ba_band_rms_cap_groups",
                        "ba_band_rms_cap_low_ratio",
                        "ba_band_rms_cap_high_ratio",
                        "ba_band_rms_cap_epsilon",
                        "ba_valid_key_attention_enabled",
                        "ba_valid_key_attention_groups",
                        "ba_valid_key_attention_warmup_steps",
                        "ba_valid_key_attention_logit_floor",
                        "ba_valid_key_attention_entropy_support",
                        "ba_valid_key_attention_entropy_normalization",
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
                        if hasattr(_val_model, attribute):
                            setattr(
                                self.pipe,
                                attribute,
                                getattr(_val_model, attribute),
                            )
                    # Ensure pipeline modules are on GPU
                    try:
                        self.pipe.to(self.device)
                    except Exception:
                        pass
                finally:
                    self.config.pipeline.pretrained_model_name_or_path = prev_pipe_base

                # Announce successful switch to validation base model
                if _created_val and self.accelerator.is_main_process:
                    try:
                        print(f"[Base Model Switch] Validation start: swapping base '{prev_model_base}' -> '{val_pretrained}'")
                    except Exception:
                        pass
            total_images = len(dataloader.dataset) if hasattr(dataloader, "dataset") else len(dataloader)
            validation_pose_adapt_ratio = self.validation_pose_adapt_ratio
            validation_pose_restore = None
            if validation_pose_adapt_ratio is not None:
                validation_pose_adapt_ratio = float(validation_pose_adapt_ratio)
                if not 0.0 <= validation_pose_adapt_ratio <= 1.0:
                    raise ValueError(
                        "validation_pose_adapt_ratio must be in [0, 1], "
                        f"got {validation_pose_adapt_ratio}"
                    )
                if not hasattr(self, "pipe"):
                    raise RuntimeError(
                        "validation_pose_adapt_ratio requires an active pipeline"
                    )
                # 26 Jul 2026 - AICODE-NOTE: Keep the training branch ratio
                # unchanged while allowing a fixed inference intervention.
                # The original value is restored before training resumes.
                validation_pose_restore = float(
                    getattr(self.pipe, "pose_adapt_ratio", 0.0)
                )
                self.pipe.pose_adapt_ratio = validation_pose_adapt_ratio
                print(
                    "VALIDATION_POSE_ADAPT_RUNTIME "
                    f"ratio={validation_pose_adapt_ratio:.4f} "
                    f"training_ratio={validation_pose_restore:.4f}"
                )
            if hasattr(self, 'pipe'):
                for attr in ('_call_debug_counter', '_current_debug_idx', '_current_debug_total'):
                    if hasattr(self.pipe, attr):
                        setattr(self.pipe, attr, 0)
            self._val_generation_counter = 0
            if self.accelerator.is_main_process:
                val_dir = Path("hm_debug") / "val_generation"
                val_dir.mkdir(parents=True, exist_ok=True)
                self._val_generation_dir = val_dir
            else:
                self._val_generation_dir = None
            print(f"[DebugImage] total validation images: {total_images}")  # always show total
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                print(f"[DebugImage] validation image {batch_idx:02d}/{total_images:02d}")  # always show current id
                batch["debug_idx"] = batch_idx  # --- MODIFIED For training integration ---
                batch["debug_total"] = total_images  # --- MODIFIED For training integration ---
                fetch_done = time.time()
                fetch_time = fetch_done - prev_time
                process_start = time.time()
                batch = self.process_evaluation_batch(
                    batch,
                    eval_metrics=self.evaluation_metrics,
                )
                if face_quality_session is not None:
                    face_quality_session.add_batch(batch, batch_idx)
                process_time = time.time() - process_start
                prev_time = time.time()

                # Save final generated images in a single, stable sequence
                if (
                    self.accelerator.is_main_process
                    and getattr(self, "_val_generation_dir", None) is not None
                ):
                    images = batch.get("generated")
                    if images is not None:
                        # flatten possible nested lists
                        if isinstance(images, list):
                            flat = []
                            for item in images:
                                if isinstance(item, list):
                                    flat.extend(item)
                                else:
                                    flat.append(item)
                            images = flat
                        else:
                            images = [images]
                        for img in images:
                            idx = getattr(self, "_val_generation_counter", 0)
                            filename = f"{idx:02d}.png"
                            save_path = self._val_generation_dir / filename
                            if hasattr(img, "save"):
                                img.save(save_path)
                            self._val_generation_counter = idx + 1

                if self.validation_debug_timing and self.accelerator.is_main_process:
                    msg = (
                        f"[VAL TIMING] part={part} idx={batch_idx} "
                        f"fetch={fetch_time:.3f}s process={process_time:.3f}s"
                    )
                    if self.logger is not None:
                        self.logger.info(msg)
                    else:
                        print(msg)
                self._log_batch(
                    batch_idx, batch, part
                ) 
            self._log_per_image_id_sim_table(
                part=part,
                step=validation_step,
                expected_rows=(
                    len(dataloader.dataset)
                    if hasattr(dataloader, "dataset")
                    else None
                ),
            )
            self._log_scalars(self.evaluation_metrics, part)

        if validation_pose_restore is not None and hasattr(self, "pipe"):
            self.pipe.pose_adapt_ratio = validation_pose_restore

        # Restore the original pipeline object after alternate-base validation.
        # Moving the training model back to GPU is deferred until after optional
        # GPU face-quality scoring.
        if val_pretrained:
            # Announce restoration back to training base model
            if _created_val and self.accelerator.is_main_process:
                try:
                    print(f"[Base Model Switch] Validation end: restoring base '{val_pretrained}' -> '{prev_model_base}'")
                except Exception:
                    pass
            if _orig_pipe is not None:
                self.pipe = _orig_pipe
            # Ensure temporary validation model is dereferenced before IQA.
            _val_model = None
            state = None
            default_snapshot = None
            train_unet = None
            val_unet = None
            batch = None
            gc.collect()
            torch.cuda.empty_cache()

        for metric in self.metrics:
            metric.to_cpu()

        face_quality_result = {}
        face_quality_offloaded_pipe = False
        face_quality_offloaded_model = False
        try:
            if face_quality_session is not None:
                num_procs = int(getattr(self.accelerator, "num_processes", 1))
                face_quality_device = face_quality_session.resolve_device(num_procs)
                if (
                    face_quality_session.execution_mode != "deferred"
                    and face_quality_device.startswith("cuda")
                ):
                    if _offloaded_train_model:
                        pass
                    elif hasattr(self, "pipe") and hasattr(self.pipe, "to"):
                        self.pipe.to("cpu")
                        face_quality_offloaded_pipe = True
                    else:
                        self.accelerator.unwrap_model(self.model).to("cpu")
                        face_quality_offloaded_model = True
                    # 27 Jul 2026 - AICODE-NOTE: PyIQA peaks near 25 GB for
                    # this metric set. Single-GPU validation must release the
                    # generation model before the scorer subprocess starts.
                    gc.collect()
                    torch.cuda.empty_cache()
                face_quality_result = face_quality_session.finalize(
                    num_processes=num_procs
                )
        finally:
            if face_quality_offloaded_pipe:
                self.pipe.to(self.device)
            elif face_quality_offloaded_model or _offloaded_train_model:
                self.accelerator.unwrap_model(self.model).to(self.device)
            gc.collect()
            torch.cuda.empty_cache()

        result = self.evaluation_metrics.result()
        result.update(
            {
                f"face_quality/{name}": value
                for name, value in face_quality_result.items()
            }
        )
        return result

    def _log_per_image_id_sim_table(self, *, part, step, expected_rows):
        """Persist and publish one exact-row ID-sim table per validation gate."""
        if not self.log_per_image_id_sim_table:
            return
        rows = list(getattr(self, "_validation_per_image_id_rows", ()))
        if not rows:
            raise RuntimeError(
                "Per-image ID table is enabled but validation produced no id_sim rows"
            )
        rows.sort(key=lambda row: int(row["image_index"]))
        indices = [int(row["image_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise RuntimeError("Per-image ID table contains duplicate image indices")
        if expected_rows is not None:
            expected_rows = int(expected_rows)
            if len(rows) != expected_rows:
                raise RuntimeError(
                    "Per-image ID table row mismatch: "
                    f"expected={expected_rows}, actual={len(rows)}"
                )
            expected_indices = list(range(expected_rows))
            if indices != expected_indices:
                raise RuntimeError(
                    "Per-image ID table index mismatch: "
                    f"expected=0..{expected_rows - 1}, actual={indices[:10]}"
                )
        for row in rows:
            row["validation_step"] = int(step)
            row["partition"] = str(part)

        columns = [
            "validation_step",
            "partition",
            "image_index",
            "output_key",
            "identity",
            "prompt",
            "seed",
            "generated_image_count",
            "id_sim",
        ]
        diagnostic_columns = [
            "id_sim_legacy_best",
            "id_sim_mask_iou",
            "id_sim_face_count",
            "id_sim_no_face",
            "id_sim_unowned",
            "id_sim_ambiguous",
        ]
        columns.extend(
            name for name in diagnostic_columns if any(name in row for row in rows)
        )
        table = pd.DataFrame(rows, columns=columns)
        filename = f"id_sim__{part}__step_{int(step):06d}.csv"
        table_dir = self.checkpoint_dir / "validation_tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_dir / filename, index=False)

        exact_table_logger = getattr(self.writer, "add_table_file", None)
        if exact_table_logger is not None:
            exact_table_logger(filename, table)
        else:
            fallback = getattr(self.writer, "add_table", None)
            if fallback is not None:
                fallback(filename.removesuffix(".csv"), table)
        if self.logger is not None:
            self.logger.info(
                "Per-image ID table: %s rows=%s step=%s",
                table_dir / filename,
                len(table),
                step,
            )


    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.max_grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            

    @torch.no_grad()
    def _get_grad_norms(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            grad_norms (dict): the calculated norms.
        """
        # # Helper function to compute norm 
        # def compute_params_grad_norm(parameters):
        #     return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type).item()

        # Helper: robust to params with no gradients yet
        def compute_params_grad_norm(parameters):
            grads = [p.grad for p in parameters if p.grad is not None]
            if not grads:
                return 0.0
            pieces = [torch.norm(g.detach(), norm_type) for g in grads]
            return torch.norm(torch.stack(pieces), norm_type).item()


        grad_norms = {}
        for group in self.optimizer.param_groups:
            grad_norms[group["name"]] = compute_params_grad_norm(group["params"])

        # # Compute total norm
        # total_norm = torch.norm(torch.tensor(list(grad_norms.values())), norm_type).item()
        # grad_norms["total_norm"] = total_norm
        # self.optimizer.zero_grad()

        total_norm = torch.norm(torch.tensor(list(grad_norms.values()), dtype=torch.float32), norm_type).item()
        grad_norms["total_norm"] = total_norm


        return grad_norms

    @torch.no_grad()
    def _get_lrs(self):
        """
        Returns lrs for logging.

        Returns:
            lres (dict): last lrs
        """
        lrs = {}
        for last_lr, group in zip(self.lr_scheduler.get_last_lr(), self.optimizer.param_groups):
            lrs[group["name"]] = last_lr

        return lrs

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker, part):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{part}/{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.accelerator.unwrap_model(self.model)).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.accelerator.unwrap_model(self.model).get_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config,
        }

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if self.accelerator.is_main_process:
            self.logger.info(f"Saving checkpoint: {filename} ...")

        temporary = f"{filename}.tmp"
        torch.save(state, temporary)
        os.replace(temporary, filename)

    def _save_weights_only_checkpoint(self, epoch):
        state = self.accelerator.unwrap_model(self.model).get_state_dict()
        filename = str(self.checkpoint_dir / f"weights-epoch{epoch}.pth")
        if self.accelerator.is_main_process:
            self.logger.info(f"Saving weights-only checkpoint: {filename} ...")
        temporary = f"{filename}.tmp"
        torch.save(state, temporary)
        os.replace(temporary, filename)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        if self.accelerator.is_main_process:
            self.logger.info(f"Loading checkpoint: {resume_path} ...")
        # Full training checkpoints include optimizer state and an OmegaConf
        # DictConfig, so PyTorch 2.6+'s weights-only default cannot load them.
        # Resume paths are trusted, locally produced experiment artifacts.
        checkpoint = torch.load(
            resume_path,
            map_location=self.device,
            weights_only=False,
        )
        self.start_epoch = checkpoint["epoch"] + 1

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            if self.accelerator.is_main_process:
                self.logger.warning(
                    "Warning: Architecture configuration given in the config file is different from that "
                    "of the checkpoint. This may yield an exception when state_dict is loaded."
                )
        self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            if self.accelerator.is_main_process:
                self.logger.warning(
                    "Warning: Optimizer or lr_scheduler given in the config file is different "
                    "from that of the checkpoint. Optimizer and scheduler parameters "
                    "are not resumed."
                )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
            )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            if self.accelerator.is_main_process:
                self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(
            pretrained_path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint.get("state_dict") is not None:
            self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint["state_dict"])
        else:
            self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint)
           
