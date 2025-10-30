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

        # Lazily load name-keyed bbox map once, matching infer.py behavior
        if not hasattr(self, "_gen_bbox_by_name"):
            gen_bbox = None
            # Try to read from the active validation dataset object
            try:
                for _name, _loader in getattr(self, "evaluation_dataloaders", {}).items():
                    ds = getattr(_loader, "dataset", None)
                    # Prefer a raw JSON dict if present (ManualPhotoMakerValDataset stores one)
                    if ds is not None and hasattr(ds, "_bbox_gen_json") and getattr(ds, "_bbox_gen_json") is not None:
                        gen_bbox = getattr(ds, "_bbox_gen_json")
                        break
            except Exception:
                gen_bbox = None

            # Fallback to path in config if available
            if gen_bbox is None:
                try:
                    val_names = list(getattr(self.config, "val_datasets_names", []))
                    if val_names:
                        ds_name = val_names[0]
                        ds_cfg = self.config.datasets.val.get(ds_name)
                        bbox_path = getattr(ds_cfg, "bbox_mask_gen", None) if ds_cfg is not None else None
                        if bbox_path:
                            import json as _json
                            with open(str(bbox_path), "r", encoding="utf-8") as _fh:
                                gen_bbox = _json.load(_fh)
                except Exception:
                    gen_bbox = None

            self._gen_bbox_by_name = gen_bbox if isinstance(gen_bbox, dict) else None

        # If generation bbox masks are required, enforce presence of the map
        use_gen_mask = bool(self.config.validation_args.get("use_bbox_mask_gen", False))
        if use_gen_mask and self._gen_bbox_by_name is None:
            err = (
                "use_bbox_mask_gen=True but bbox mask map not loaded. "
                "Ensure validation dataset provides bbox_mask_gen or config.datasets.val[...] has bbox_mask_gen set."
            )
            if getattr(self, "logger", None) is not None:
                self.logger.error(err)
            else:
                print(err)
            raise RuntimeError(err)

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

            # ### To align validation with infer.py generation ###
            # Use a device-matched generator (GPU when available)
            generator = torch.Generator(device=self.device).manual_seed(int(sample_seed))
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

            # Match infer.py: if a filename-keyed bbox map is provided, override face_bbox_gen by exact output name
            face_bbox_ref = sample.get("face_bbox_ref")
            face_bbox_gen = sample.get("face_bbox_gen")
            if isinstance(sample_prompt, str) and sample_id is not None and self._gen_bbox_by_name is not None:
                base = f"{sample_prompt[:10]}_{sample_id}"
                key = f"{base}.png"
                entry = self._gen_bbox_by_name.get(key)
                if use_gen_mask:
                    if entry is None:
                        err = f"No bbox entry in bbox_mask_gen for expected output name '{key}'"
                        if getattr(self, "logger", None) is not None:
                            self.logger.error(err)
                        else:
                            print(err)
                        raise RuntimeError(err)
                    fb = entry.get("face_crop_new") or entry.get("face_crop_old") if isinstance(entry, dict) else None
                    if fb is None:
                        err = f"BBox record for '{key}' missing face_crop_new/old"
                        if getattr(self, "logger", None) is not None:
                            self.logger.error(err)
                        else:
                            print(err)
                        raise RuntimeError(err)
                    face_bbox_gen = fb
                else:
                    # Optional override (no strict requirement)
                    if isinstance(entry, dict):
                        fb = entry.get("face_crop_new") or entry.get("face_crop_old")
                        if fb is not None:
                            face_bbox_gen = fb

            # ### To align validation with infer.py generation ###
            # Compute and pass 512-D id_embeds from the first ref image via FaceAnalysis2
            id_embeds_vec = None
            try:
                # Lazily prepare FaceAnalysis once
                if not hasattr(self, "_val_face_analyzer"):
                    from src.model.photomaker_branched.insightface_package import FaceAnalysis2, analyze_faces
                    _fa = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
                    try:
                        _fa.prepare(ctx_id=0, det_size=(640, 640))
                    except Exception:
                        # Best-effort fallback
                        _fa.prepare(ctx_id=-1, det_size=(640, 640))
                    self._val_face_analyzer = _fa
                # Extract from the first reference image if available
                first_ref = sample_ref_images[0] if isinstance(sample_ref_images, (list, tuple)) and sample_ref_images else sample_ref_images
                if first_ref is not None:
                    import numpy as _np
                    from src.model.photomaker_branched.insightface_package import analyze_faces
                    _np_img = _np.array(first_ref.convert("RGB"))[:, :, ::-1]
                    _faces = analyze_faces(self._val_face_analyzer, _np_img)
                    if _faces:
                        id_embeds_vec = torch.from_numpy(_faces[0]["embedding"]).float()
            except Exception:
                id_embeds_vec = None

            generated_images = self.pipe(
                prompt=sample_prompt,
                generator=generator,
                input_id_images=sample_ref_images,
                id_embeds=id_embeds_vec,
                # Optional fixed bbox masks per sample (after possible override)
                face_bbox_ref=face_bbox_ref,
                face_bbox_gen=face_bbox_gen,
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
