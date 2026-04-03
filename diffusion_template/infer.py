import os
from pathlib import Path

import argparse
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import numpy as np  # Alligned with PhotoMaker
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
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
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
    for attr in ("disable_branched_sa", "disable_branched_ca", "strict_face_routing"):
        if attr in top_keys:
            setattr(model, attr, bool(getattr(cfg, attr)))
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
    for attr in ("disable_branched_sa", "disable_branched_ca", "strict_face_routing", "ba_patch_top_k"):
        if attr in top_keys:
            setattr(pipeline, attr, getattr(cfg, attr))
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

    # Optional: load generation bbox map keyed by final filename
    gen_bbox_by_name = None
    try:
        bbox_gen_path = getattr(cfg, "bbox_mask_gen_path", None)
        if not bbox_gen_path:
            bbox_gen_path = getattr(cfg.dataset, "bbox_mask_gen", None)
        if bbox_gen_path and str(bbox_gen_path).strip():
            import json as _json
            with open(str(bbox_gen_path), "r", encoding="utf-8") as _fh:
                gen_bbox_by_name = _json.load(_fh)
    except Exception:
        gen_bbox_by_name = None

    out_dir = Path(getattr(cfg, "output_dir", "outputs/infer"))
    _ensure_dir(out_dir)

    val_args = _to_plain(cfg.validation_args)
    batch_size = int(getattr(cfg, "batch_size", 1) or 1)
    total = len(dataset)


    # Prepare face analyzer once (Alligned with PhotoMaker)
    try:
        from src.model.photomaker_branched.insightface_package import FaceAnalysis2, analyze_faces  # Alligned with PhotoMaker
        _face_an = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
        _face_an.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        _face_an = None

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

            # If a filename-keyed bbox map is provided, override face_bbox_gen by exact output name.
            for rel_idx, sample in enumerate(batch_samples):
                idx = start + rel_idx
                ref_path = sample.get("image_path")
                ref_stem = Path(ref_path).stem if ref_path is not None else sample.get("id", f"idx{idx:04d}")
                ref_stems.append(ref_stem)
                if gen_bbox_by_name is not None:
                    base = f"{prompts[rel_idx][:10]}_{ref_stem}"
                    key = f"{base}.png"
                    entry = gen_bbox_by_name.get(key)
                    if isinstance(entry, dict):
                        fb = entry.get("face_crop_new")
                        if fb is None:
                            fb = entry.get("face_crop_old")
                        if fb is not None:
                            face_bbox_gen_batch[rel_idx] = fb
                    if face_bbox_gen_batch[rel_idx] is None:
                        raise RuntimeError(
                            f"No bbox entry in bbox_mask_gen for expected output name '{key}'"
                        )

            generators = [
                torch.Generator(device=device.type).manual_seed(seed)
                for seed in seeds
            ]

            id_embeds_batch = []
            has_any_id_embed = False
            for ref_images in refs_batch:
                id_embeds_vec = None
                try:
                    if _face_an is not None and isinstance(ref_images, (list, tuple)) and len(ref_images) > 0:
                        _pil = ref_images[0]
                        _np = np.array(_pil.convert("RGB"))[:, :, ::-1]
                        _faces = analyze_faces(_face_an, _np)
                        if _faces:
                            id_embeds_vec = torch.from_numpy(_faces[0]["embedding"]).float()
                except Exception:
                    id_embeds_vec = None
                if id_embeds_vec is None:
                    id_embeds_vec = torch.zeros(512, dtype=torch.float32)
                else:
                    has_any_id_embed = True
                id_embeds_batch.append(id_embeds_vec)

            id_embeds_arg = None
            if has_any_id_embed and id_embeds_batch:
                id_embeds_arg = torch.stack(id_embeds_batch, dim=0)

            call_args = dict(val_args)
            dbg_base = call_args.get("debug_dir", "hm_debug") or "hm_debug"
            call_args["debug_dir"] = str(Path(dbg_base) / f"{start:02d}")
            call_args["debug_idx"] = start
            call_args["debug_total"] = total

            images_flat = pipeline(
                prompt=prompts if len(prompts) > 1 else prompts[0],
                input_id_images=refs_batch if len(refs_batch) > 1 else refs_batch[0],
                generator=generators if len(generators) > 1 else generators[0],
                face_bbox_ref=face_bbox_ref_batch if len(face_bbox_ref_batch) > 1 else face_bbox_ref_batch[0],
                face_bbox_gen=face_bbox_gen_batch if len(face_bbox_gen_batch) > 1 else face_bbox_gen_batch[0],
                id_embeds=id_embeds_arg,
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
