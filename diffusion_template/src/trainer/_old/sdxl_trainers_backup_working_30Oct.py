import time
import os
from pathlib import Path  # --- MODIFIED For training integration ---
import torch
from omegaconf import OmegaConf  # --- MODIFIED For training integration ---

DEBUG_LOG_DEBUG_IMAGES = os.environ.get("PM_DEBUG_IMAGES", "1") not in {"0", "false", "False", ""}

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class SDXLTrainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, train_metrics: MetricTracker):
        """
        Run batch through the model, compute loss,
        and do training step.

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            train_metrics (MetricTracker): MetricTracker object that computes
                and aggregates training losses.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        self.optimizer.zero_grad()
            
        do_cfg =  (batch["batch_idx"] % self.cfg_step == 0)
        output = self.model(**batch, do_cfg=do_cfg)
        batch.update(output)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        
        if self.is_train:
            assert torch.isfinite(batch["loss"]) # sum of all losses is always called loss
            self.accelerator.backward(batch["loss"]) 
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
            train_metrics.update(loss_name, batch[loss_name].item())

        return batch

    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        seed = batch.get("seed", self.config.validation_args.get("seed", 0))
        generator = torch.Generator(device='cpu').manual_seed(seed)
        validation_kwargs = OmegaConf.to_container(self.config.validation_args, resolve=True)
        if not isinstance(validation_kwargs, dict):
            validation_kwargs = dict(validation_kwargs)
        debug_base = validation_kwargs.get("debug_dir", "hm_debug")
        debug_idx = batch.get("debug_idx", 0)
        debug_total = batch.get("debug_total")
        validation_kwargs["debug_dir"] = str(Path(debug_base) / f"{int(debug_idx):02d}")
        validation_kwargs["debug_idx"] = int(debug_idx)
        if debug_total is not None:
            validation_kwargs["debug_total"] = int(debug_total)
        if DEBUG_LOG_DEBUG_IMAGES:
            print(f"[DebugImage] validation batch idx={debug_idx} → debug_dir={validation_kwargs['debug_dir']}")
        generated_images = self.pipe(
            prompt=batch['prompt'],
            generator=generator,
            **validation_kwargs
        ).images

        batch['generated'] = generated_images

        for metric in self.metrics:
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)
                
        return batch
        
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            # --- MODIFIED For training integration ---
            prompts = batch.get('prompt')
            if isinstance(prompts, str):
                prompts = [prompts]
            elif isinstance(prompts, list):
                flat_prompts = []
                for item in prompts:
                    if isinstance(item, list):
                        flat_prompts.extend(item)
                    else:
                        flat_prompts.append(item)
                prompts = flat_prompts 
            else: 
                prompts = []

            generated = batch.get('generated')
            if not generated:
                return
            images = []
            if isinstance(generated, list):
                for item in generated:
                    if isinstance(item, list):
                        images.extend(item)
                    else:
                        images.append(item)
            else:
                images = [generated]
            if not images:
                return 
            # --- MODIFIED For training integration ---
            
            # --- MODIFIED For training integration ---
            num_per_prompt = self.config.validation_args.get("num_images_per_prompt", 1)  
            labels = []  
            if prompts and len(prompts) * num_per_prompt == len(images): 
                for p_idx, prompt in enumerate(prompts):
                    for img_idx in range(num_per_prompt):
                        labels.append(f"{prompt}_b{batch_idx:03d}_p{p_idx:02d}_img{img_idx}")
            elif prompts and len(prompts) == len(images):
                labels = [f"{p}_b{batch_idx:03d}" for p in prompts]
            else:
                labels = [f"{mode}_{batch_idx}_img{i}" for i in range(len(images))] 

            sanitized = [label.replace(" ", "_")[:80] for label in labels]  
            save_root = Path(self.checkpoint_dir) / "val_images" / mode / f"step_{getattr(self.writer, 'step', 0)}_batch_{batch_idx}"
            save_root.mkdir(parents=True, exist_ok=True)

            for img, name in zip(images, sanitized):
                image_name = f"{mode}_images/{name}"
                self.writer.add_image(image_name, img)
                if hasattr(img, "save"):
                    img.save(save_root / f"{name}.png")
            # --- MODIFIED For training integration ---


class PhotomakerLoraTrainer(SDXLTrainer):
    def __init__(self, masked_loss_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_loss_step = masked_loss_step
        
    def process_batch(self, batch, train_metrics: MetricTracker):
        self.optimizer.zero_grad()
            
        do_cfg = (batch["batch_idx"] % self.cfg_step == 0)
        output = self.model(**batch, do_cfg=do_cfg)
        batch.update(output)

        batch["is_masked_loss"] = (batch["batch_idx"] % self.masked_loss_step == 0)
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        
        if self.is_train:
            assert torch.isfinite(batch["loss"]) # sum of all losses is always called loss
            self.accelerator.backward(batch["loss"]) 
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
            train_metrics.update(loss_name, batch[loss_name].item())

        return batch
        
    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        prompts = batch["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_size = len(prompts)

        def get_value(key, default=None):
            if key not in batch:
                return default
            value = batch[key]
            if isinstance(value, list) and batch_size > 1 and len(value) == batch_size:
                return value
            return value

        ref_images_list = get_value("ref_images")
        if batch_size == 1 and not isinstance(ref_images_list, list):
            ref_images_list = [ref_images_list]
        if batch_size > 1 and not isinstance(ref_images_list, list):
            ref_images_list = [ref_images_list] * batch_size

        ids_list = get_value("id", [None] * batch_size)
        if not isinstance(ids_list, list):
            ids_list = [ids_list] * batch_size

        seeds_value = batch.get("seed", None)
        if isinstance(seeds_value, list) and len(seeds_value) == batch_size:
            seeds_list = seeds_value
        else:
            default_seed = self.config.validation_args.get("seed", 0) if seeds_value is None else seeds_value
            seeds_list = [default_seed] * batch_size

        generated_collection = []
        total_pipe_time = 0.0
        total_metric_time = 0.0
        total_steps = 0
        step_max = 0.0

        for idx in range(batch_size):
            sample = {}
            for key, value in batch.items():
                if isinstance(value, list) and batch_size > 1 and len(value) == batch_size:
                    sample[key] = value[idx]
                else:
                    sample[key] = value

            sample_prompt = prompts[idx]
            sample_ref_images = ref_images_list[idx]
            sample_id = ids_list[idx]
            sample_seed = seeds_list[idx]

            generator = torch.Generator(device='cpu').manual_seed(sample_seed)
            callback = None
            step_durations = []
            pipe_start = time.time()

            if self.validation_debug_timing:
                last_time = pipe_start

                def _callback(pipe, step, timestep, callback_kwargs):
                    nonlocal last_time
                    now = time.time()
                    step_duration = now - last_time
                    step_durations.append(step_duration)
                    last_time = now
                    return callback_kwargs

                callback = _callback

            generated_images = self.pipe(
                prompt=sample_prompt,
                generator=generator,
                input_id_images=sample_ref_images,
                # Optional fixed bbox masks per sample
                face_bbox_ref=sample.get("face_bbox_ref"),
                face_bbox_gen=sample.get("face_bbox_gen"),
                callback_on_step_end=callback,
                **self.config.validation_args
            ).images
            pipe_time = time.time() - pipe_start
            total_pipe_time += pipe_time

            sample["prompt"] = sample_prompt
            sample["ref_images"] = sample_ref_images
            sample["generated"] = generated_images
            sample["id"] = sample_id
            sample["seed"] = sample_seed

            metric_time = 0.0
            for metric in self.metrics:
                metric_start = time.time()
                metric_result = metric(**sample)
                metric_time += time.time() - metric_start
                for k, v in metric_result.items():
                    eval_metrics.update(k, v)

            total_metric_time += metric_time
            generated_collection.append(generated_images)

            if self.validation_debug_timing and step_durations:
                total_steps += len(step_durations)
                step_max = max(step_max, max(step_durations))

        batch["generated"] = generated_collection if batch_size > 1 else generated_collection[0]

        if self.validation_debug_timing and self.accelerator.is_main_process:
            if total_steps > 0:
                step_mean = total_pipe_time / total_steps
                step_stats = f" step_mean={step_mean:.3f}s step_max={step_max:.3f}s steps={total_steps}"
            else:
                step_stats = ""
            msg = (
                f"[VAL TIMING] pipeline={total_pipe_time:.3f}s "
                f"metrics={total_metric_time:.3f}s{step_stats}"
            )
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)

        return batch