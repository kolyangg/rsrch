import os
from types import SimpleNamespace
from pathlib import Path

import argparse
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import numpy as np  # Alligned with PhotoMaker


class _NoAccelerator:
    def unwrap_model(self, model, keep_fp32_wrapper=False):
        return model


def _to_plain(d):
    obj = OmegaConf.to_container(d, resolve=True)
    return dict(obj) if obj is not None else {}


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_images(images, out_dir: Path, prompt: str, ref_stem: str):
    base = f"{prompt[:10]}_{ref_stem}"
    if isinstance(images, list):
        for i, img in enumerate(images):
            name = f"{base}_{i:02d}.png" if len(images) > 1 else f"{base}.png"
            img.save(out_dir / name)
    else:
        (images).save(out_dir / f"{base}.png")


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
    args = parser.parse_args()

    cfg_path = _Path(_ABS_CFG_DIR) / f"{args.config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.set_struct(cfg, False)
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

    # Instantiate model (PhotoMaker v2 + LoRA adapters)
    model = instantiate(cfg.model, device=device)
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
            for idx in range(start, end):
                sample = dataset[idx]
                prompt = sample["prompt"]
                ref_images = sample["ref_images"]
                seed = sample.get("seed", 0)
                face_bbox_ref = sample.get("face_bbox_ref")
                face_bbox_gen = sample.get("face_bbox_gen")

                # If a filename-keyed bbox map is provided, override face_bbox_gen by exact output name (Alligned with diffusion_template)
                if gen_bbox_by_name is not None:
                    ref_path = sample.get("image_path")
                    ref_stem = Path(ref_path).stem if ref_path is not None else sample.get("id", f"idx{idx:04d}")
                    base = f"{prompt[:10]}_{ref_stem}"
                    key = f"{base}.png"
                    entry = gen_bbox_by_name.get(key)
                    if entry is None:
                        raise RuntimeError(f"No bbox entry in bbox_mask_gen for expected output name '{key}'")
                    fb = entry.get("face_crop_new") if isinstance(entry, dict) else None
                    if fb is None and isinstance(entry, dict):
                        fb = entry.get("face_crop_old")
                    if fb is None:
                        raise RuntimeError(f"BBox record for '{key}' missing face_crop_new/old")
                    face_bbox_gen = fb

                # Use generator on pipeline device for parity with PhotoMaker (Alligned with PhotoMaker)
                gen = torch.Generator(device=device.type).manual_seed(int(seed))

                # Precompute 512-D id embedding via FaceAnalysis (Alligned with PhotoMaker)
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

                # per-sample debug directory and indices (to satisfy pipeline debug hooks)
                call_args = dict(val_args)
                dbg_base = call_args.get("debug_dir", "hm_debug") or "hm_debug"
                call_args["debug_dir"] = str(Path(dbg_base) / f"{idx:02d}")
                call_args["debug_idx"] = idx
                call_args["debug_total"] = total

                images = pipeline(
                    prompt=prompt,
                    input_id_images=ref_images,
                    generator=gen,
                    face_bbox_ref=face_bbox_ref,
                    face_bbox_gen=face_bbox_gen,
                    id_embeds=id_embeds_vec,  # Alligned with PhotoMaker
                    **call_args,
                ).images

                ref_path = sample.get("image_path")
                ref_stem = Path(ref_path).stem if ref_path is not None else sample.get("id", f"idx{idx:04d}")
                _save_images(images, out_dir, prompt, ref_stem)
                pbar.update(1)



if __name__ == "__main__":
    main()
