from abc import abstractmethod
from pathlib import Path

import gc
import torch
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.face_quality_validation import FaceQualityValidationSession
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from hydra.utils import instantiate

import os
import time



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
        validation_pose_adapt_ratio=None,
        validation_interval_steps=2000,
        face_quality=None,
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
        self.validation_pose_adapt_ratio = validation_pose_adapt_ratio
        self.validation_interval_steps = (
            None
            if validation_interval_steps is None
            else int(validation_interval_steps)
        )
        self.face_quality_config = face_quality
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
                    _val_model = instantiate(self.config.model, device=self.device)
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
                    # Ensure adapters are initialized before loading LoRA weights
                    if hasattr(_val_model, "prepare_for_training"):
                        _val_model.prepare_for_training()
                    try:
                        state = self.accelerator.unwrap_model(self.model).get_state_dict()
                    except Exception:
                        state = self.model.get_state_dict()
                    if not bool(getattr(self.config, "update_proc_weights_val", False)) and isinstance(state, dict):
                        state = dict(state)
                        state.pop("attn_processors", None)
                    if hasattr(_val_model, "load_state_dict_"):
                        _val_model.load_state_dict_(state)

                    # Optionally copy branched-attention processor weights into the
                    # validation UNet so their effect is visible in validation.
                    if bool(getattr(self.config, "update_proc_weights_val", False)):
                        try:
                            train_unet = self.accelerator.unwrap_model(self.model).unet
                        except Exception:
                            train_unet = getattr(self.model, "unet", None)
                        val_unet = getattr(_val_model, "unet", None)
                        if train_unet is not None and val_unet is not None:
                            t_procs = getattr(train_unet, "attn_processors", {})
                            v_procs = getattr(val_unet, "attn_processors", {})
                            for name, t_proc in t_procs.items():
                                v_proc = v_procs.get(name)
                                if v_proc is None:
                                    continue
                                try:
                                    v_proc.load_state_dict(t_proc.state_dict(), strict=False)
                                except Exception:
                                    continue
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

        for metric in self.metrics:
            metric.to_cpu()

        face_quality_result = {}
        face_quality_offloaded_pipe = False
        face_quality_offloaded_model = False
        try:
            if face_quality_session is not None:
                num_procs = int(getattr(self.accelerator, "num_processes", 1))
                face_quality_device = face_quality_session.resolve_device(num_procs)
                if face_quality_device.startswith("cuda"):
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
           
