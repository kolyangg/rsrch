#!/usr/bin/env bash
set -euo pipefail

# Download exactly the weights the BA debug matrix needs into the HF cache.
#
# The SDXL model class (src/model/sdxl/original.py) loads the full-precision
# component safetensors (no variant), so we fetch those and skip fp16 variants
# and the big single-file checkpoints. Sizes: ~13.9GB per base repo, ~0.9GB
# PhotoMaker-V2. InsightFace (buffalo_l, ~0.3GB) and YOLO face weights need no
# manual download (auto-fetched / already in repo).
#
# Usage:
#   bash scripts/download_ba_debug_weights.sh            # RealVis + PhotoMaker-V2 (~15GB)
#   bash scripts/download_ba_debug_weights.sh --with-sdxl # + SDXL-base for test T2 (~29GB total)

HF_CLI="${HF_CLI:-huggingface-cli}"

DIFFUSERS_INCLUDE=(
    "model_index.json"
    "scheduler/*"
    "tokenizer/*"
    "tokenizer_2/*"
    "text_encoder/config.json"  "text_encoder/model.safetensors"
    "text_encoder_2/config.json" "text_encoder_2/model.safetensors"
    "unet/config.json" "unet/diffusion_pytorch_model.safetensors"
    "vae/config.json"  "vae/diffusion_pytorch_model.safetensors"
)

echo ">>> PhotoMaker-V2 checkpoint (~0.9GB)"
"${HF_CLI}" download TencentARC/PhotoMaker-V2 photomaker-v2.bin

echo ">>> RealVisXL_V4.0 full-precision components (~13.9GB)"
"${HF_CLI}" download SG161222/RealVisXL_V4.0 --include "${DIFFUSERS_INCLUDE[@]}"

if [[ "${1:-}" == "--with-sdxl" ]]; then
    echo ">>> stable-diffusion-xl-base-1.0 full-precision components (~13.9GB)"
    "${HF_CLI}" download stabilityai/stable-diffusion-xl-base-1.0 --include "${DIFFUSERS_INCLUDE[@]}"
fi

echo ">>> Done. Cache usage:"
du -sh ~/.cache/huggingface/hub/models--SG161222--RealVisXL_V4.0 2>/dev/null || true
du -sh ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0 2>/dev/null || true
du -sh ~/.cache/huggingface/hub/models--TencentARC--PhotoMaker-V2 2>/dev/null || true
