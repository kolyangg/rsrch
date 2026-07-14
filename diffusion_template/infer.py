import os
from pathlib import Path

import argparse
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from src.metrics.tracker import MetricTracker


class _NoAccelerator:
    def unwrap_model(self, model, keep_fp32_wrapper=False):
        return model


def _to_plain(d):
    obj = OmegaConf.to_container(d, resolve=True)
    return dict(obj) if obj is not None else {}


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _apply_saved_ba_architecture(cfg, checkpoint) -> None:
    """Restore architecture switches that are not encoded by tensor shapes alone."""
    if not checkpoint:
        return
    saved_config_path = Path(str(checkpoint)).expanduser().resolve().parent / "config.yaml"
    if not saved_config_path.is_file():
        return
    saved = OmegaConf.load(saved_config_path)
    model_keys = (
        "ba_sa_mode",
        "ba_face_kv_mode",
        "ba_face_roi_size",
        "ba_ca_mode",
        "ba_ca_train_mode",
        "ba_identity_token_count",
        "ba_pm_preservation_mode",
        "ba_hard_mask_resize",
        "disable_reference_spatial_branch",
        "branched_attn_weight_mode",
        "branched_attn_new_weight_kind",
        "train_branched_ca_lora",
    )
    for key in model_keys:
        if "model" in saved and key in saved.model:
            cfg.model[key] = saved.model[key]
    if "pipeline" in saved and "face_embed_strategy" in saved.pipeline:
        strategy = saved.pipeline.face_embed_strategy
        cfg.pipeline.face_embed_strategy = strategy
        cfg.model.face_embed_strategy = strategy
        cfg.validation_args.face_embed_strategy = strategy
    for key in ("disable_branched_sa", "disable_branched_ca"):
        if key in saved:
            cfg[key] = saved[key]


def _iter_named_images(images, prompt: str, ref_stem: str):
    base = f"{prompt[:10]}_{ref_stem}"
    if isinstance(images, list):
        if len(images) == 1:
            yield f"{base}.png", images[0]
        else:
            for i, img in enumerate(images):
                yield f"{base}_{i:02d}.png", img
    else:
        yield f"{base}.png", images


def _save_images(images, out_dir: Path, prompt: str, ref_stem: str):
    for name, img in _iter_named_images(images, prompt, ref_stem):
        img.save(out_dir / name)


from pathlib import Path as _Path


_ABS_CFG_DIR = str((_Path(__file__).parent / "src" / "configs").resolve())


def main():
    parser = argparse.ArgumentParser(description="Single-GPU inference")
    parser.add_argument(
        "--config-name",
        type=str,
        default="inference/photomaker_origv2_infer",
        help="Config path relative to src/configs (e.g. inference/photomaker_origv2_infer)",
    )
    args, overrides = parser.parse_known_args()

    cfg_path = _Path(_ABS_CFG_DIR) / f"{args.config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.set_struct(cfg, False)
    cli_cfg = OmegaConf.from_dotlist(overrides) if overrides else None
    if cli_cfg is not None:
        cfg = OmegaConf.merge(cfg, cli_cfg)
    _apply_saved_ba_architecture(cfg, getattr(cfg, "saved_checkpoint", None))
    if cli_cfg is not None:
        cfg = OmegaConf.merge(cfg, cli_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match torch dtype choice with PhotoMaker (bf16 if supported, else fp16) (Alligned with PhotoMaker)
    try:
        _pipe_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        _model_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if hasattr(cfg, "pipeline"):
            cfg.pipeline.torch_dtype = _pipe_dtype
        if hasattr(cfg, "model"):
            cfg.model.weight_dtype = _model_dtype
    except Exception:
        pass

    # Validate essential nodes (avoid OmegaConf getattr side-effects)
    top_keys = set(cfg.keys()) if hasattr(cfg, 'keys') else set()
    if ("model" not in top_keys) or ("pipeline" not in top_keys):
        raise KeyError(
            f"Config must define 'model' and 'pipeline' blocks. Top-level keys: {sorted(top_keys)}"
        )

    writer = None
    if "writer" in top_keys:
        project_config = OmegaConf.to_container(cfg, resolve=True)
        comet_run_id = getattr(cfg, "cometml_id", None)
        writer = instantiate(cfg.writer, None, project_config, run_id=comet_run_id)
        if hasattr(writer, "set_step"):
            writer.set_step(0, mode="infer")

    metrics = []
    infer_metrics = MetricTracker()
    if "inference_metrics" in top_keys and "metrics" in top_keys:
        for metric_name in cfg.inference_metrics:
            metric_config = cfg.metrics[metric_name]
            metrics.append(instantiate(metric_config, name=metric_name, device=device))

    # Instantiate model (PhotoMaker v2 + LoRA adapters)
    model = instantiate(cfg.model, device=device)
    for attr in ("disable_branched_sa", "disable_branched_ca", "strict_face_routing", "ba_uncond_face_fix"):
        if attr in top_keys:
            setattr(model, attr, bool(getattr(cfg, attr)))
    if "ba_face_prompt_mode" in top_keys:
        setattr(model, "ba_face_prompt_mode", str(getattr(cfg, "ba_face_prompt_mode")).lower())
    # Ensure LoRA adapter slot "lora_adapter" exists before loading checkpoints
    if hasattr(model, "prepare_for_training"):
        try:
            model.prepare_for_training()
        except Exception:
            # Some models may not require this; continue if it fails harmlessly
            pass
    # Move full module tree to target device for single-GPU inference
    model = model.to(device)

    # Optional: load saved LoRA checkpoint
    ckpt = getattr(cfg, "saved_checkpoint", None)
    if ckpt and str(ckpt).lower() not in {"na", "none", "null", ""}:
        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        sd = state.get("state_dict", state)
        model.load_state_dict_(sd)

    # Build pipeline via existing factory (no accelerate)
    accel = _NoAccelerator()
    pipeline = instantiate(cfg.pipeline, model=model, accelerator=accel, _recursive_=False)
    pipeline.to(device)
    for attr in ("disable_branched_sa", "disable_branched_ca", "strict_face_routing", "ba_patch_top_k", "ba_uncond_face_fix"):
        if attr in top_keys:
            setattr(pipeline, attr, getattr(cfg, attr))
    if "ba_face_prompt_mode" in top_keys:
        setattr(pipeline, "ba_face_prompt_mode", str(getattr(cfg, "ba_face_prompt_mode")).lower())
    # Optional VRAM relief for small GPUs (e.g. 16GB laptop cards)
    if bool(getattr(cfg, "enable_vae_tiling", False)) and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    # Ensure custom components attached to the pipeline (e.g., id_encoder) are on device
    if hasattr(pipeline, "id_encoder"):
        try:
            target_dtype = pipeline.unet.dtype if hasattr(pipeline, "unet") else None
            if target_dtype is not None:
                pipeline.id_encoder.to(device=device, dtype=target_dtype)
            else:
                pipeline.id_encoder.to(device=device)
        except Exception:
            # Best-effort move; continue if component lacks .to()
            pass

    # Dataset (manual_val-like)
    dataset = instantiate(cfg.dataset)

    out_dir = Path(getattr(cfg, "output_dir", "outputs/infer"))
    _ensure_dir(out_dir)

    val_args = _to_plain(cfg.validation_args)
    batch_size = int(getattr(cfg, "batch_size", 1) or 1)
    total = len(dataset)
    use_gen_mask = bool(val_args.get("use_bbox_mask_gen", False))
    automatic_bboxes = bool(getattr(cfg, "automatic_bboxes", False))
    automatic_bboxes_every_val = bool(getattr(cfg, "automatic_bboxes_every_val", False))

    # Optional: load generation bbox maps keyed by final filename.
    gen_bbox_by_name = None
    manual_gen_bbox_by_name = None
    auto_bbox_store = None
    bbox_gen_path = getattr(cfg, "bbox_mask_gen_path", None)
    if not bbox_gen_path:
        bbox_gen_path = getattr(cfg.dataset, "bbox_mask_gen", None)
    bbox_manual_path = getattr(cfg, "bbox_mask_gen_fallback_path", None)
    if not bbox_manual_path:
        bbox_manual_path = getattr(cfg.dataset, "bbox_mask_gen", None)
    try:
        import json as _json

        if bbox_manual_path and str(bbox_manual_path).strip():
            with open(str(bbox_manual_path), "r", encoding="utf-8") as _fh:
                manual_gen_bbox_by_name = _json.load(_fh)

        if automatic_bboxes and use_gen_mask:
            from src.utils.auto_bbox_gen import AutoGenBboxStore

            if bbox_gen_path and str(bbox_gen_path).strip():
                auto_bbox_path = Path(str(bbox_gen_path))
            elif bbox_manual_path and str(bbox_manual_path).strip():
                manual_path = Path(str(bbox_manual_path))
                auto_bbox_path = manual_path.with_name(manual_path.stem + "_auto.json")
            else:
                auto_bbox_path = Path("bbox_mask_gen_auto.json")
            auto_bbox_store = AutoGenBboxStore(
                auto_bbox_path,
                face_detector=getattr(cfg, "face_detector", "mtcnn"),
                face_model=getattr(cfg, "face_model", "yolov8n-face.pt"),
            )
            gen_bbox_by_name = auto_bbox_store.data
        elif bbox_gen_path and str(bbox_gen_path).strip():
            with open(str(bbox_gen_path), "r", encoding="utf-8") as _fh:
                gen_bbox_by_name = _json.load(_fh)
    except Exception:
        gen_bbox_by_name = None
        manual_gen_bbox_by_name = None
        auto_bbox_store = None

    with tqdm(total=total, desc="Inference", dynamic_ncols=True) as pbar:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_samples = [dataset[idx] for idx in range(start, end)]
            prompts = [sample["prompt"] for sample in batch_samples]
            refs_batch = [sample["ref_images"] for sample in batch_samples]
            seeds = [int(sample.get("seed", 0)) for sample in batch_samples]
            face_bbox_ref_batch = [sample.get("face_bbox_ref") for sample in batch_samples]
            face_bbox_gen_batch = [sample.get("face_bbox_gen") for sample in batch_samples]
            ref_stems = []
            keys = []

            call_args = dict(val_args)
            dbg_base = call_args.get("debug_dir", "hm_debug") or "hm_debug"
            call_args["debug_dir"] = str(Path(dbg_base) / f"{start:02d}")
            call_args["debug_idx"] = start
            call_args["debug_total"] = total

            generators = [
                torch.Generator(device=device.type).manual_seed(seed)
                for seed in seeds
            ]

            pending_pm = []
            for rel_idx, sample in enumerate(batch_samples):
                idx = start + rel_idx
                ref_path = sample.get("image_path")
                ref_stem = Path(ref_path).stem if ref_path is not None else sample.get("id", f"idx{idx:04d}")
                ref_stems.append(ref_stem)
                key = f"{prompts[rel_idx][:10]}_{ref_stem}.png"
                keys.append(key)
                manual_entry = None
                if isinstance(manual_gen_bbox_by_name, dict):
                    manual_entry = manual_gen_bbox_by_name.get(key)
                force_manual = bool(isinstance(manual_entry, dict) and manual_entry.get("force_manual", False))
                entry = manual_entry if force_manual else None
                if entry is None and isinstance(gen_bbox_by_name, dict):
                    entry = gen_bbox_by_name.get(key)
                face_bbox = None
                if isinstance(entry, dict):
                    face_bbox = entry.get("face_crop_new") or entry.get("face_crop_old")
                    if face_bbox is not None:
                        face_bbox_gen_batch[rel_idx] = face_bbox
                if (
                    auto_bbox_store is not None
                    and use_gen_mask
                    and (not force_manual)
                    and (entry is None or face_bbox is None or automatic_bboxes_every_val)
                ):
                    pending_pm.append(rel_idx)

            if pending_pm:
                pm_kwargs = dict(call_args)
                pm_kwargs["use_branched_attention"] = False
                pm_kwargs["use_bbox_mask_gen"] = False
                pm_kwargs["debug_dir"] = None
                pm_images = pipeline(
                    prompt=[prompts[i] for i in pending_pm] if len(pending_pm) > 1 else prompts[pending_pm[0]],
                    input_id_images=[refs_batch[i] for i in pending_pm] if len(pending_pm) > 1 else refs_batch[pending_pm[0]],
                    generator=[generators[i] for i in pending_pm] if len(pending_pm) > 1 else generators[pending_pm[0]],
                    face_bbox_ref=[face_bbox_ref_batch[i] for i in pending_pm] if len(pending_pm) > 1 else face_bbox_ref_batch[pending_pm[0]],
                    face_bbox_gen=None,
                    **pm_kwargs,
                ).images
                if not isinstance(pm_images, list):
                    pm_images = [pm_images]
                pm_num_per_prompt = int(pm_kwargs.get("num_images_per_prompt", 1) or 1)
                for local_idx, rel_idx in enumerate(pending_pm):
                    entry = auto_bbox_store.ensure(
                        keys[rel_idx],
                        photomaker_image=pm_images[local_idx * pm_num_per_prompt],
                        meta={
                            "prompt": str(prompts[rel_idx]),
                            "id": str(batch_samples[rel_idx].get("id")),
                            "seed": int(seeds[rel_idx]),
                        },
                        force_recompute=automatic_bboxes_every_val,
                    )
                    gen_bbox_by_name[keys[rel_idx]] = entry
                    face_bbox_gen_batch[rel_idx] = entry.get("face_crop_new") or entry.get("face_crop_old")

            if use_gen_mask:
                for rel_idx, face_bbox in enumerate(face_bbox_gen_batch):
                    if face_bbox is None:
                        raise RuntimeError(
                            f"No bbox entry in bbox_mask_gen for expected output name '{keys[rel_idx]}'"
                        )

            # The PhotoMaker preview pass above (when it runs to auto-detect the gen-bbox)
            # consumes these generators, so re-seed them here to guarantee the branched pass
            # starts from the SAME initial latents the gen-bbox was detected on. Otherwise the
            # bbox is for a different image than the branched pass generates -> misaligned
            # face / ghost. (The training-time validation path already uses separate
            # freshly-seeded generators for each pass; this matches that behaviour.)
            for _g, _s in zip(generators, seeds):
                _g.manual_seed(int(_s))

            images_flat = pipeline(
                prompt=prompts if len(prompts) > 1 else prompts[0],
                input_id_images=refs_batch if len(refs_batch) > 1 else refs_batch[0],
                generator=generators if len(generators) > 1 else generators[0],
                face_bbox_ref=face_bbox_ref_batch if len(face_bbox_ref_batch) > 1 else face_bbox_ref_batch[0],
                face_bbox_gen=face_bbox_gen_batch if len(face_bbox_gen_batch) > 1 else face_bbox_gen_batch[0],
                **call_args,
            ).images

            if not isinstance(images_flat, list):
                images_flat = [images_flat]

            num_per_prompt = int(call_args.get("num_images_per_prompt", 1) or 1)
            expected_total = len(batch_samples) * num_per_prompt
            if len(images_flat) != expected_total:
                if len(batch_samples) == 1:
                    num_per_prompt = len(images_flat)
                else:
                    raise RuntimeError(
                        f"Inference generation returned {len(images_flat)} images for "
                        f"batch_size={len(batch_samples)}, num_images_per_prompt={num_per_prompt}."
                    )

            for rel_idx, sample in enumerate(batch_samples):
                idx = start + rel_idx
                img_start = rel_idx * num_per_prompt
                img_end = img_start + num_per_prompt
                sample_images = images_flat[img_start:img_end]
                prompt = prompts[rel_idx]
                ref_stem = ref_stems[rel_idx]
                _save_images(sample_images, out_dir, prompt, ref_stem)
                if writer is not None:
                    for name, img in _iter_named_images(sample_images, prompt, ref_stem):
                        writer.add_image(name, img)
                if metrics:
                    metric_sample = {
                        "prompt": prompt,
                        "generated": sample_images,
                        "id": sample.get("id"),
                    }
                    for metric in metrics:
                        metric_result = metric(**metric_sample)
                        for k, v in metric_result.items():
                            infer_metrics.update(k, v)
                pbar.update(1)

    if writer is not None:
        for metric_name in infer_metrics.keys():
            writer.add_scalar(f"infer/{metric_name}", infer_metrics.avg(metric_name))
    if infer_metrics.keys():
        print({k: infer_metrics.avg(k) for k in infer_metrics.keys()})



if __name__ == "__main__":
    main()
