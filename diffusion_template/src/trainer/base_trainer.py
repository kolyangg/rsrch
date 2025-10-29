from abc import abstractmethod
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

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
        seed
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


    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            if self.accelerator.is_main_process:
                self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch and evaluating it on non-train partitions
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            if self.accelerator.is_main_process:
                # save logged information into logs dict
                logs = {"epoch": epoch}
                logs.update(result)

                # print logged information to the screen
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)


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

        # Synchronize ranks around the initial validation to avoid NCCL timeouts
        if epoch == 1:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                for part, dataloader in self.evaluation_dataloaders.items():
                    val_logs = self._evaluation_epoch(epoch - 1, part, dataloader)
                    logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})
                self.is_train = True
            self.accelerator.wait_for_everyone()
        
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc=f"train_{pid}", total=self.epoch_len)
        ):  

            batch["batch_idx"] = batch_idx
            batch = self.process_batch(
                batch,
                train_metrics=self.train_metrics,
            )

            grad_norms = self._get_grad_norms()
            for part_name, part_norm in grad_norms.items():
                self.train_metrics.update(f"grad_norm/{part_name}", part_norm)
            
            # log current results
            if batch_idx % self.log_step == 0:
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

        # Run val/test
        # Synchronize ranks around per-epoch validation to avoid NCCL timeouts
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            for part, dataloader in self.evaluation_dataloaders.items():
                val_logs = self._evaluation_epoch(epoch, part, dataloader)
                logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})
        self.accelerator.wait_for_everyone()

        return logs

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
        torch.cuda.empty_cache()

        for metric in self.metrics:
            metric.to_cuda()

        self.writer.set_step(epoch * self.epoch_len, part)
        prev_time = time.time()
        with torch.no_grad():
            total_images = len(dataloader.dataset) if hasattr(dataloader, "dataset") else len(dataloader)
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

                        # infer.py-style names: f"{prompt[:10]}_{ref_stem}_{i:02d}.png"
                        prompts = batch.get("prompt")
                        if isinstance(prompts, str):
                            prompts = [prompts]
                        elif isinstance(prompts, list):
                            flat_prompts = []
                            for it in prompts:
                                if isinstance(it, list):
                                    flat_prompts.extend(it)
                                else:
                                    flat_prompts.append(it)
                            prompts = flat_prompts
                        else:
                            prompts = []

                        ids = batch.get("id")
                        if isinstance(ids, str):
                            ids = [ids]
                        elif isinstance(ids, list):
                            flat_ids = []
                            for it in ids:
                                if isinstance(it, list):
                                    flat_ids.extend(it)
                                else:
                                    flat_ids.append(it)
                            ids = flat_ids
                        else:
                            ids = []

                        try:
                            npp = int(self.config.validation_args.get("num_images_per_prompt", 1))
                        except Exception:
                            npp = 1

                        names = []
                        if prompts and ids and len(prompts) == len(ids) and len(images) == len(prompts) * max(1, npp):
                            for p, ident in zip(prompts, ids):
                                stem = ident if ident is not None else "id"
                                base = f"{p[:10]}_{stem}"
                                if npp > 1:
                                    for i in range(npp):
                                        names.append(f"{base}_{i:02d}.png")
                                else:
                                    names.append(f"{base}.png")
                        else:
                            # Fallback to sequential naming
                            names = [f"{getattr(self, '_val_generation_counter', 0) + i:02d}.png" for i in range(len(images))]

                        for img, name in zip(images, names):
                            save_path = self._val_generation_dir / name
                            if hasattr(img, "save"):
                                img.save(save_path)
                            self._val_generation_counter = getattr(self, "_val_generation_counter", 0) + 1

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

        for metric in self.metrics:
            metric.to_cpu()
            
        torch.cuda.empty_cache()
        return self.evaluation_metrics.result()


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
        # Helper function to compute norm 
        def compute_params_grad_norm(parameters):
            return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type).item()

        grad_norms = {}
        for group in self.optimizer.param_groups:
            grad_norms[group["name"]] = compute_params_grad_norm(group["params"])

        # Compute total norm
        total_norm = torch.norm(torch.tensor(list(grad_norms.values())), norm_type).item()
        grad_norms["total_norm"] = total_norm
        self.optimizer.zero_grad()

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

        torch.save(state, filename)

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
        # PyTorch 2.6 defaults weights_only=True which breaks loading our pickled
        # DictConfig inside the checkpoint. Explicitly disable weights_only.
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
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
        # Allow full pickled payload for backward compatibility with saved checkpoints
        checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)

        if checkpoint.get("state_dict") is not None:
            self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint["state_dict"])
        else:
            self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint)
           
