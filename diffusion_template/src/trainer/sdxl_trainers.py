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
            # One-time check: print grads for a few processor params
            if not hasattr(self, "_printed_proc_grad_check"):
                try:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                except Exception:
                    unwrapped = self.model
                to_check = []
                for name, p in unwrapped.named_parameters():
                    if (
                        "unet.down_blocks" in name
                        and ".attn1.processor.id_to_hidden.weight" in name
                    ):
                        to_check.append((name, p))
                    if len(to_check) >= 3:
                        break
                if to_check and self.accelerator.is_main_process:
                    lines = []
                    for n, p in to_check:
                        has_grad = (p.grad is not None)
                        lines.append(f"{n}: grad={'OK' if has_grad else 'None'}")
                    msg = "[Check] Processor id_to_hidden grads after first backward:\n  " + "\n  ".join(lines)
                    if getattr(self, "logger", None) is not None:
                        self.logger.info(msg)
                    else:
                        print(msg)
                self._printed_proc_grad_check = True
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
        val_debug = bool(validation_kwargs.get("val_debug", getattr(self.config, "val_debug", True)))
        validation_kwargs["val_debug"] = val_debug
        debug_base = validation_kwargs.get("debug_dir", "hm_debug")
        debug_idx = batch.get("debug_idx", 0)
        debug_total = batch.get("debug_total")
        validation_kwargs["debug_dir"] = str(Path(debug_base) / f"{int(debug_idx):02d}")
        validation_kwargs["debug_idx"] = int(debug_idx)
        if debug_total is not None:
            validation_kwargs["debug_total"] = int(debug_total)
        if DEBUG_LOG_DEBUG_IMAGES and val_debug:
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

            # ### To align validation with infer.py generation ###
            ### Make validation filenames match bbox JSON keys: f"{prompt[:10]}_{id}.png" ###
            ids = batch.get('id')
            if isinstance(ids, str):
                ids = [ids]
            elif isinstance(ids, list):
                flat_ids = []
                for item in ids:
                    if isinstance(item, list):
                        flat_ids.extend(item)
                    else:
                        flat_ids.append(item)
                ids = flat_ids

            labels = []
            if prompts and ids and len(prompts) == len(ids) and (len(prompts) * num_per_prompt == len(images) or len(prompts) == len(images)):
                for p_idx, (p_text, p_id) in enumerate(zip(prompts, ids)):
                    base = f"{p_text[:10]}_{p_id}"
                    if len(prompts) * num_per_prompt == len(images) and num_per_prompt > 1:
                        for _ in range(num_per_prompt):
                            labels.append(base)
                    else:
                        labels.append(base)
            else:
                # Fallback to previous prompt-based naming if alignment is unclear
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

            for i, (img, name) in enumerate(zip(images, sanitized)):
                # ### To align validation with infer.py generation ###
                # Log and save using the exact bbox-JSON-like filename
                self.writer.add_image(f"{name}.png", img)
                if hasattr(img, "save"):
                    img.save(save_root / f"{name}.png")
            ### Make validation filenames match bbox JSON keys: f"{prompt[:10]}_{id}.png" ###
            # --- MODIFIED For training integration ---


class PhotomakerLoraTrainer(SDXLTrainer):
    def __init__(self, masked_loss_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_loss_step = masked_loss_step
        
    def process_batch(self, batch, train_metrics: MetricTracker):
        ### 25 APR - ADD GRAD ACCUM ###
        accum_steps = int(getattr(self, "grad_accum_steps", 1))
        is_accum_start = batch["batch_idx"] % accum_steps == 0
        is_accum_end = (batch["batch_idx"] + 1) % accum_steps == 0
        if self.is_train and is_accum_start:
            self.optimizer.zero_grad()
        ### 25 APR - ADD GRAD ACCUM ###
            
        do_cfg = (batch["batch_idx"] % self.cfg_step == 0)
        oom_flag = torch.zeros(1, device=self.device)
        local_oom = False
        output = None
        try:
            output = self.model(**batch, do_cfg=do_cfg)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            local_oom = True
            oom_flag.fill_(1)
            self.optimizer.zero_grad(set_to_none=True)
            self._cleanup_cuda_state()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(oom_flag, op=torch.distributed.ReduceOp.MAX)

        if bool(oom_flag.item()):
            if output is not None:
                del output
            self.optimizer.zero_grad(set_to_none=True)
            self._cleanup_cuda_state()
            rank = int(getattr(self.accelerator, "process_index", 0))
            msg = (
                f"[OOM_SKIP] time={time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"rank={rank} batch_idx={batch.get('batch_idx')} "
                f"local_oom={local_oom}"
            )
            if self.logger is not None:
                self.logger.warning(msg)
            else:
                print(msg, flush=True)
            batch["skip_batch"] = True
            batch["loss"] = torch.zeros((), device=self.device)
            return batch

        batch.update(output)
        ba_telemetry = output.get("ba_telemetry")
        if ba_telemetry:
            batch.update(ba_telemetry)

        batch["is_masked_loss"] = (
            self.masked_loss_step > 0
            and batch["batch_idx"] % self.masked_loss_step == 0
        )
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        
        if self.is_train:
            assert torch.isfinite(batch["loss"]) # sum of all losses is always called loss
            ### 25 APR - ADD GRAD ACCUM ###
            self.accelerator.backward(batch["loss"] / accum_steps)
            if is_accum_end:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            ### 25 APR - ADD GRAD ACCUM ###

        # 13 Aug 2026 - CL14_CA-PERF-03: stack diagnostics for one device sync;
        # a one-GPU Serv run needs no collective at all. Scalar means are exact.
        loss_names = tuple(self.config.writer.loss_names)
        local_scalars = torch.stack([
            batch[name].detach().reshape(()) for name in loss_names
        ]).float()
        if self.accelerator.num_processes == 1:
            gathered = local_scalars.unsqueeze(0)
        else:
            gathered = self.accelerator.gather(local_scalars).reshape(
                -1, len(loss_names)
            )
        means = gathered.mean(dim=0)
        for index, (loss_name, mean_value) in enumerate(
            zip(loss_names, means.cpu().tolist())
        ):
            batch[loss_name] = means[index]
            train_metrics.update(loss_name, mean_value)

        return batch
        
    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        prompts = batch["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_size = len(prompts)
        val_debug = bool(self.config.validation_args.get("val_debug", getattr(self.config, "val_debug", True)))

        def get_value(key, default=None):
            value = batch.get(key, default)
            if (
                isinstance(value, list)
                and batch_size > 1
                and len(value) == batch_size
            ):
                return value
            return value

        ref_images_list = get_value("ref_images")
        if batch_size == 1 and not isinstance(ref_images_list, list):
            ref_images_list = [ref_images_list]
        if batch_size > 1 and not isinstance(ref_images_list, list):
            ref_images_list = [ref_images_list] * batch_size

        def get_sample_refs(index):
            return ref_images_list if batch_size == 1 else ref_images_list[index]

        ids_list = get_value("id", [None] * batch_size)
        if not isinstance(ids_list, list):
            ids_list = [ids_list] * batch_size

        seeds_value = batch.get("seed")
        if isinstance(seeds_value, list) and len(seeds_value) == batch_size:
            seeds_list = seeds_value
        else:
            default_seed = (
                self.config.validation_args.get("seed", 0)
                if seeds_value is None
                else seeds_value
            )
            seeds_list = [default_seed] * batch_size

        def normalize_bbox_list(value):
            if (
                batch_size > 1
                and isinstance(value, list)
                and len(value) == batch_size
                and all(
                    item is None or isinstance(item, (list, tuple))
                    for item in value
                )
            ):
                return value
            return [value] * batch_size

        face_bbox_ref_list = normalize_bbox_list(
            get_value("face_bbox_ref")
        )
        face_bbox_gen_list = normalize_bbox_list(
            get_value("face_bbox_gen")
        )
        if any(box is None for box in face_bbox_gen_list):
            raise RuntimeError(
                "The fixed validation protocol is missing a generation bbox"
            )

        batch_debug_idx = int(batch.get("debug_idx", 0))
        batch_debug_total = int(batch.get("debug_total", 0))
        sample_debug_indices = [
            batch_debug_idx * batch_size + index
            if batch_size > 1
            else batch_debug_idx
            for index in range(batch_size)
        ]
        debug_dir = (
            self.config.validation_args.get("debug_dir") if val_debug else None
        )


        val_refs = []
        refs_iterable = [get_sample_refs(idx) for idx in range(batch_size)]
        for refs in refs_iterable:
            refs_list = list(refs) if isinstance(refs, (list, tuple)) else [refs]
            if len(refs_list) == 0:
                raise RuntimeError("Validation sample has empty reference image list.")
            val_refs.append(refs_list)

        val_kwargs = dict(self.config.validation_args)
        val_kwargs["debug_idx"] = int(batch_debug_idx)
        val_kwargs["debug_total"] = int(batch_debug_total)
        val_kwargs["val_debug"] = val_debug

        callback = None
        step_durations = []
        total_pipe_time = 0.0
        total_metric_time = 0.0
        total_steps = 0
        step_max = 0.0
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

        generators = [
            torch.Generator(device=self.device).manual_seed(int(sample_seed))
            for sample_seed in seeds_list
        ]

        generated_flat = self.pipe(
            prompt=prompts if batch_size > 1 else prompts[0],
            generator=generators if batch_size > 1 else generators[0],
            input_id_images=val_refs if batch_size > 1 else val_refs[0],
            face_bbox_ref=face_bbox_ref_list if batch_size > 1 else face_bbox_ref_list[0],
            face_bbox_gen=face_bbox_gen_list if batch_size > 1 else face_bbox_gen_list[0],
            callback_on_step_end=callback,
            **val_kwargs,
        ).images
        if not isinstance(generated_flat, list):
            generated_flat = [generated_flat]

        total_pipe_time += time.time() - pipe_start
        if self.validation_debug_timing and step_durations:
            total_steps += len(step_durations)
            step_max = max(step_max, max(step_durations))

        num_per_prompt = int(self.config.validation_args.get("num_images_per_prompt", 1))
        if num_per_prompt <= 0:
            num_per_prompt = 1
        expected_total = batch_size * num_per_prompt
        if len(generated_flat) != expected_total:
            if batch_size == 1:
                num_per_prompt = len(generated_flat)
            else:
                err = (
                    f"Validation generation returned {len(generated_flat)} images for "
                    f"batch_size={batch_size}, num_images_per_prompt={num_per_prompt}."
                )
                if getattr(self, "logger", None) is not None:
                    self.logger.error(err)
                else:
                    print(err)
                raise RuntimeError(err)

        generated_collection = []
        for idx in range(batch_size):
            start = idx * num_per_prompt
            end = start + num_per_prompt
            sample_images = generated_flat[start:end]
            generated_collection.append(sample_images)

            # Save final BA images into per-sample hm_debug/<idx>/ folders.
            try:
                if debug_dir and sample_images:
                    out_dir = Path(str(debug_dir)) / f"{int(sample_debug_indices[idx]):02d}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    if len(sample_images) == 1:
                        sample_images[0].save(out_dir / "generated_ba.png")
                    else:
                        for j, img in enumerate(sample_images):
                            img.save(out_dir / f"generated_ba_{j:02d}.png")
            except Exception:
                pass

        for idx in range(batch_size):
            sample = {}
            for key, value in batch.items():
                if isinstance(value, list) and batch_size > 1 and len(value) == batch_size:
                    sample[key] = value[idx]
                else:
                    sample[key] = value

            sample["prompt"] = prompts[idx]
            sample["ref_images"] = get_sample_refs(idx)
            sample["generated"] = generated_collection[idx]
            sample["id"] = ids_list[idx]
            sample["seed"] = seeds_list[idx]
            # 12 Aug 2026 - Subject-v2 scores the face owned by the exact box
            # passed to BA, not an unrelated detection elsewhere in the image.
            sample["face_bbox_gen"] = face_bbox_gen_list[idx]
            sample["face_bbox_ref"] = face_bbox_ref_list[idx]

            metric_time = 0.0
            for metric in self.metrics:
                metric_start = time.time()
                metric_result = metric(**sample)
                metric_time += time.time() - metric_start
                for k, v in metric_result.items():
                    eval_metrics.update(k, v)
            total_metric_time += metric_time

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

        #### 08 MAR - FIX BATCHED VALIDATION ####
        return batch
