import time
import os
from pathlib import Path  # --- MODIFIED For training integration ---
import torch
from omegaconf import OmegaConf  # --- MODIFIED For training integration ---

DEBUG_LOG_DEBUG_IMAGES = os.environ.get("PM_DEBUG_IMAGES", "1") not in {"0", "false", "False", ""}

from src.metrics.tracker import MetricTracker
from src.model.photomaker_branched.lora2_helpers import InvalidBranchedSampleError
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

            generated_masks = batch.get("generated_masks")
            mask_images = []
            if generated_masks is not None:
                if isinstance(generated_masks, list):
                    for item in generated_masks:
                        if isinstance(item, list):
                            mask_images.extend(item)
                        else:
                            mask_images.append(item)
                else:
                    mask_images = [generated_masks]
            if mask_images and len(mask_images) < len(images):
                mask_images.extend([None] * (len(images) - len(mask_images)))
            
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
                if i < len(mask_images) and mask_images[i] is not None:
                    self.writer.add_image(f"{name}_mask.png", mask_images[i])
            ### Make validation filenames match bbox JSON keys: f"{prompt[:10]}_{id}.png" ###
            # --- MODIFIED For training integration ---


class PhotomakerLoraTrainer(SDXLTrainer):
    def __init__(self, masked_loss_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_loss_step = masked_loss_step

    def _update_ba_weight_norms(self, train_metrics):
        """Drift canary: L2 norm of branched-processor lora_B weights per group.

        lora_B starts at zero, so these norms are a clean monotone signal of how
        far each branch group has moved. The cosm_new1 failure showed doubling
        per 2k steps (worst in ca_noise); healthy runs should grow sublinearly.
        Groups log 0 when a branch has no LoRA params (e.g. ref_only mode -> *_noise).
        """
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
        except Exception:
            unwrapped = self.model
        sums = {"sa_ref": 0.0, "sa_noise": 0.0, "ca_ref": 0.0, "ca_noise": 0.0}
        with torch.no_grad():
            for name, p in unwrapped.named_parameters():
                if ".processor." not in name or "lora_B" not in name:
                    continue
                if ".attn1.processor." in name:
                    kind = "sa"
                elif ".attn2.processor." in name:
                    kind = "ca"
                else:
                    continue
                if ".ref_to_" in name:
                    branch = "ref"
                elif ".noise_to_" in name:
                    branch = "noise"
                else:
                    continue
                sums[f"{kind}_{branch}"] += float(p.detach().float().pow(2).sum().item())
        for group, sq_sum in sums.items():
            train_metrics.update(f"ba_norm/{group}", sq_sum ** 0.5)
        
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
        invalid_flag = torch.zeros(1, device=self.device)
        local_oom = False
        invalid_reason = None
        output = None
        try:
            output = self.model(**batch, do_cfg=do_cfg)
        except InvalidBranchedSampleError as exc:
            invalid_flag.fill_(1)
            invalid_reason = exc.reason
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            local_oom = True
            oom_flag.fill_(1)
            self.optimizer.zero_grad(set_to_none=True)
            self._cleanup_cuda_state()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(oom_flag, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(invalid_flag, op=torch.distributed.ReduceOp.MAX)

        if bool(invalid_flag.item()):
            if output is not None:
                del output
            reason = invalid_reason or "rejected_on_other_rank"
            train_metrics.update(f"invalid_sample/{reason}", 1.0)
            rank = int(getattr(self.accelerator, "process_index", 0))
            msg = (
                f"[INVALID_SAMPLE_SKIP] time={time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"rank={rank} batch_idx={batch.get('batch_idx')} reason={reason}"
            )
            if self.logger is not None:
                self.logger.warning(msg)
            else:
                print(msg, flush=True)
            batch["skip_batch"] = True
            batch["loss"] = torch.zeros((), device=self.device)
            return batch

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

        batch["is_masked_loss"] = (
            self.masked_loss_step > 0
            and batch["batch_idx"] % self.masked_loss_step == 0
        )
        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if "id_loss" in output:
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
            except Exception:
                unwrapped = self.model
            id_loss = output["id_loss"]
            id_loss_weight = float(getattr(unwrapped, "id_loss_weight", 0.0))
            if not torch.isfinite(id_loss):
                raise FloatingPointError(f"Non-finite identity loss: {id_loss}")
            batch["loss"] = batch["loss"] + id_loss_weight * id_loss
            gathered_id = self.accelerator.gather(id_loss.detach()).mean()
            gathered_applied = self.accelerator.gather(
                output["id_loss_applied"].detach()
            ).mean()
            train_metrics.update("id_loss", gathered_id.item())
            train_metrics.update("id_loss_applied", gathered_applied.item())
            train_metrics.update("id_loss_weighted", (id_loss_weight * gathered_id).item())
        
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

            # Branched-attention drift canary, logged on the same cadence as
            # the scalar logs (base_trainer flushes train_metrics every log_step).
            if batch["batch_idx"] % self.log_step == 0:
                self._update_ba_weight_norms(train_metrics)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
            train_metrics.update(loss_name, batch[loss_name].item())

        return batch
        
    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        if bool(getattr(self.config, "ppr_diagnostic_matrix", False)):
            from src.trainer.ppr_diagnostic import run_ppr_diagnostic_batch

            return run_ppr_diagnostic_batch(self, batch, eval_metrics)

        prompts = batch["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_size = len(prompts)
        val_debug = bool(self.config.validation_args.get("val_debug", getattr(self.config, "val_debug", True)))

        # Optional: generate gen-bboxes on-the-fly via an extra PhotoMaker-only pass.
        # Only makes sense when branched attention is expected to run.
        automatic_bboxes = bool(getattr(self.config, "automatic_bboxes", False))
        automatic_bboxes_every_val = bool(getattr(self.config, "automatic_bboxes_every_val", True))
        force_log_first_auto_bbox = bool(getattr(self.config, "force_log_first_auto_bbox", True))
        use_gen_mask = bool(self.config.validation_args.get("use_bbox_mask_gen", False))
        use_branched_attention = bool(self.config.validation_args.get("use_branched_attention", False))
        mask_expansion_ratio = float(
            self.config.validation_args.get(
                "mask_expansion_ratio",
                getattr(self.config, "mask_expansion_ratio", 1.0),
            )
        )
        try:
            sm = int(self.config.validation_args.get("photomaker_start_step", 0))
            bs = int(self.config.validation_args.get("branched_attn_start_step", 0))
            nsteps = int(self.config.validation_args.get("num_inference_steps", 50))
        except Exception:
            sm, bs, nsteps = 0, 1, 50

        # Branched attention is expected to run whenever it is enabled and starts within the denoising horizon.
        # (bs==sm is a valid "start from step 0" configuration.)
        branched_expected = bool(use_branched_attention) and (bs < nsteps)
        auto_bbox_enabled = bool(automatic_bboxes and use_gen_mask and branched_expected)
        force_cached_auto_bbox_log = bool(
            auto_bbox_enabled
            and (not automatic_bboxes_every_val)
            and force_log_first_auto_bbox
            and int(getattr(self.writer, "step", 0)) == 0
        )
        if auto_bbox_enabled and not hasattr(self, "_printed_auto_bbox"):
            print("[AutoBboxGen] enabled: will run PhotoMaker-only pass to detect gen bboxes")
            self._printed_auto_bbox = True

        # Lazily load name-keyed bbox map once, matching infer.py behavior
        if not hasattr(self, "_gen_bbox_by_name"):
            gen_bbox = None
            manual_gen_bbox = None

            if auto_bbox_enabled:
                # Prefer placing auto JSON next to the configured bbox_mask_gen path.
                bbox_path = None
                images_dir = None
                try:
                    for _name, _loader in getattr(self, "evaluation_dataloaders", {}).items():
                        ds = getattr(_loader, "dataset", None)
                        if ds is None:
                            continue
                        images_dir = getattr(ds, "images_dir", None) or images_dir
                except Exception:
                    images_dir = None

                try:
                    val_names = list(getattr(self.config, "val_datasets_names", []))
                    if val_names:
                        ds_name = val_names[0]
                        ds_cfg = self.config.datasets.val.get(ds_name)
                        bbox_path = getattr(ds_cfg, "bbox_mask_gen", None) if ds_cfg is not None else None
                except Exception:
                    bbox_path = None

                if bbox_path:
                    p = Path(str(bbox_path))
                    auto_path = p.with_name(p.stem + "_auto.json")
                    # Load the manual bbox map too (to support per-entry force_manual flags).
                    try:
                        import json as _json
                        with open(str(p), "r", encoding="utf-8") as _fh:
                            manual_gen_bbox = _json.load(_fh)
                    except Exception:
                        manual_gen_bbox = None
                elif images_dir:
                    auto_path = Path(str(images_dir)).resolve().parent / "bbox_mask_gen_auto.json"
                else:
                    auto_path = Path("bbox_mask_gen_auto.json")

                from src.utils.auto_bbox_gen import AutoGenBboxStore
                face_detector = getattr(self.config, "face_detector", "mtcnn")
                face_model = getattr(self.config, "face_model", "yolov8n-face.pt")
                self._auto_bbox_store = AutoGenBboxStore(
                    auto_path,
                    face_detector=face_detector,
                    face_model=face_model,
                )
                gen_bbox = self._auto_bbox_store.data
            else:
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
            self._manual_gen_bbox_by_name = manual_gen_bbox if isinstance(manual_gen_bbox, dict) else None

        # If generation bbox masks are required, enforce presence of the map
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

        #### 08 MAR - FIX BATCHED VALIDATION ####
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

        def get_sample_refs(idx):
            if batch_size == 1:
                return ref_images_list
            return ref_images_list[idx]

        ids_list = get_value("id", [None] * batch_size)
        if not isinstance(ids_list, list):
            ids_list = [ids_list] * batch_size

        seeds_value = batch.get("seed", None)
        if isinstance(seeds_value, list) and len(seeds_value) == batch_size:
            seeds_list = seeds_value
        else:
            default_seed = self.config.validation_args.get("seed", 0) if seeds_value is None else seeds_value
            seeds_list = [default_seed] * batch_size

        def normalize_bbox_list(raw_bbox):
            if (
                batch_size > 1
                and isinstance(raw_bbox, list)
                and len(raw_bbox) == batch_size
                and all((x is None) or isinstance(x, (list, tuple)) for x in raw_bbox)
            ):
                return raw_bbox
            return [raw_bbox] * batch_size

        face_bbox_ref_list = normalize_bbox_list(get_value("face_bbox_ref", None))
        face_bbox_gen_list = normalize_bbox_list(get_value("face_bbox_gen", None))

        batch_debug_idx = batch.get("debug_idx", 0)
        batch_debug_total = batch.get("debug_total", 0)
        try:
            batch_debug_idx = int(batch_debug_idx)
        except Exception:
            batch_debug_idx = 0
        try:
            batch_debug_total = int(batch_debug_total)
        except Exception:
            batch_debug_total = 0

        sample_debug_indices = [
            batch_debug_idx * batch_size + idx if batch_size > 1 else batch_debug_idx
            for idx in range(batch_size)
        ]
        debug_dir = self.config.validation_args.get("debug_dir", None) if val_debug else None
        sample_mask_images = [None] * batch_size

        # Match infer.py: if a filename-keyed bbox map is provided, override face_bbox_gen by exact output name
        keys = [None] * batch_size
        entries = [None] * batch_size
        if self._gen_bbox_by_name is not None:
            for idx in range(batch_size):
                sample_prompt = prompts[idx]
                sample_id = ids_list[idx]
                if isinstance(sample_prompt, str) and sample_id is not None:
                    base = f"{sample_prompt[:10]}_{sample_id}"
                    key = f"{base}.png"
                    keys[idx] = key
                    entries[idx] = self._gen_bbox_by_name.get(key)

        if use_gen_mask:
            pending_pm = []
            for idx in range(batch_size):
                key = keys[idx]
                if key is None:
                    err = "use_bbox_mask_gen=True requires string prompt and id to build bbox key."
                    if getattr(self, "logger", None) is not None:
                        self.logger.error(err)
                    else:
                        print(err)
                    raise RuntimeError(err)

                entry = entries[idx]
                manual_entry = None
                try:
                    if auto_bbox_enabled and getattr(self, "_manual_gen_bbox_by_name", None) is not None:
                        manual_entry = self._manual_gen_bbox_by_name.get(key)
                except Exception:
                    manual_entry = None

                force_manual = bool(isinstance(manual_entry, dict) and manual_entry.get("force_manual", False))
                if force_manual:
                    entry = manual_entry

                should_recompute_entry = bool(automatic_bboxes_every_val)
                should_log_cached_entry = bool(
                    force_cached_auto_bbox_log and (not force_manual) and entry is not None
                )
                if (
                    (not force_manual)
                    and auto_bbox_enabled
                    and hasattr(self, "_auto_bbox_store")
                    and (entry is None or should_recompute_entry or should_log_cached_entry)
                ):
                    pending_pm.append(idx)

                entries[idx] = entry

            if pending_pm:
                from src.pipelines.br_pipeline_helpers import (
                    annotate_original_and_expanded_bbox,
                    expand_bbox_xyxy,
                )

                pm_prompts = [prompts[idx] for idx in pending_pm]
                pm_refs = []
                for idx in pending_pm:
                    refs = get_sample_refs(idx)
                    refs_list = list(refs) if isinstance(refs, (list, tuple)) else [refs]
                    if len(refs_list) == 0:
                        raise RuntimeError(f"Missing reference image for validation sample index {idx}")
                    pm_refs.append(refs_list)

                pm_face_bbox_ref = [face_bbox_ref_list[idx] for idx in pending_pm]
                pm_gens = [
                    torch.Generator(device=self.device).manual_seed(int(seeds_list[idx]))
                    for idx in pending_pm
                ]

                pm_kwargs = dict(self.config.validation_args)
                pm_kwargs["use_branched_attention"] = False
                pm_kwargs["use_bbox_mask_gen"] = False
                pm_kwargs["debug_dir"] = None
                pm_kwargs["debug_idx"] = int(batch_debug_idx)
                pm_kwargs["debug_total"] = int(batch_debug_total)
                pm_kwargs["val_debug"] = val_debug

                pm_images = self.pipe(
                    prompt=pm_prompts if len(pm_prompts) > 1 else pm_prompts[0],
                    generator=pm_gens if len(pm_gens) > 1 else pm_gens[0],
                    input_id_images=pm_refs if len(pm_refs) > 1 else pm_refs[0],
                    face_bbox_ref=pm_face_bbox_ref if len(pm_face_bbox_ref) > 1 else pm_face_bbox_ref[0],
                    face_bbox_gen=None,
                    **pm_kwargs,
                ).images
                if not isinstance(pm_images, list):
                    pm_images = [pm_images]

                for local_i, idx in enumerate(pending_pm):
                    key = keys[idx]
                    pm_img = pm_images[local_i]
                    overlay_path = None
                    if debug_dir:
                        overlay_path = Path(str(debug_dir)) / f"{int(sample_debug_indices[idx]):02d}" / "auto_bbox_overlay.png"
                    should_recompute_entry = bool(automatic_bboxes_every_val)
                    entry = self._auto_bbox_store.ensure(
                        key,
                        photomaker_image=pm_img,
                        meta={
                            "debug_idx": int(sample_debug_indices[idx]),
                            "prompt": str(prompts[idx]),
                            "id": str(ids_list[idx]),
                            "seed": int(seeds_list[idx]),
                        },
                        overlay_path=None,
                        force_overlay=False,
                        force_recompute=should_recompute_entry,
                    )
                    self._gen_bbox_by_name[key] = entry
                    entries[idx] = entry

                    try:
                        line_w = int(getattr(self._auto_bbox_store, "line_width", 4))
                        face_box_orig = (
                            entry.get("face_crop_new") or entry.get("face_crop_old")
                            if isinstance(entry, dict)
                            else None
                        )
                        if face_box_orig is not None:
                            face_box_expanded = expand_bbox_xyxy(
                                face_box_orig,
                                expansion_ratio=mask_expansion_ratio,
                                width=pm_img.width,
                                height=pm_img.height,
                            )
                            overlay_img = annotate_original_and_expanded_bbox(
                                pm_img,
                                original_bbox=face_box_orig,
                                expanded_bbox=face_box_expanded,
                                line_width=line_w,
                            )
                            sample_mask_images[idx] = overlay_img
                            if overlay_path is not None:
                                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                                overlay_img.save(overlay_path)
                    except Exception:
                        sample_mask_images[idx] = None

            for idx in range(batch_size):
                key = keys[idx]
                entry = entries[idx]
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
                face_bbox_gen_list[idx] = fb
        else:
            for idx in range(batch_size):
                entry = entries[idx]
                if isinstance(entry, dict):
                    fb = entry.get("face_crop_new") or entry.get("face_crop_old")
                    if fb is not None:
                        face_bbox_gen_list[idx] = fb

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
        generated_masks_collection = []
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

            if sample_mask_images[idx] is None:
                generated_masks_collection.append([None] * len(sample_images))
            else:
                generated_masks_collection.append([sample_mask_images[idx].copy() for _ in range(len(sample_images))])

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

            metric_time = 0.0
            for metric in self.metrics:
                metric_start = time.time()
                metric_result = metric(**sample)
                metric_time += time.time() - metric_start
                for k, v in metric_result.items():
                    eval_metrics.update(k, v)
            total_metric_time += metric_time

        batch["generated"] = generated_collection if batch_size > 1 else generated_collection[0]
        batch["generated_masks"] = generated_masks_collection if batch_size > 1 else generated_masks_collection[0]

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

    def finalize_ppr_diagnostic_matrix(self):
        if not bool(getattr(self.config, "ppr_diagnostic_matrix", False)):
            return
        from src.trainer.ppr_diagnostic import finalize_ppr_diagnostic_matrix

        finalize_ppr_diagnostic_matrix(self)
