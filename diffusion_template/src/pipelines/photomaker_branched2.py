"""
### Modified to make attn_processor trainable in branched version ###
Branched pipeline copy that imports from branched_new2 to use trainable
attention processors.
### Modified to make attn_processor trainable in branched version ###
"""

# photomaker/pipeline.py (branched2)

#####
# Modified from https://github.com/huggingface/diffusers/blob/v0.29.1/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
# PhotoMaker v2 @ TencentARC and MCG-NKU 
# Author: Zhen Li
#####

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

# --- Import from branched_new2 for trainable processors ---
### Modified to make attn_processor trainable in branched version ###
from src.model.photomaker_branched.branched_new2 import (
    two_branch_predict,
    prepare_reference_latents,
    encode_face_prompt,
    patch_unet_attention_processors,
    restore_original_processors,
    save_debug_images,
)
### Modified to make attn_processor trainable in branched version ###

from src.model.photomaker_branched.branch_helpers import (
    aggregate_heatmaps_to_mask,
    prepare_mask4,
    save_branch_previews,
    debug_reference_latents_once,
    save_debug_ref_latents,
    save_debug_ref_mask_overlay,
    collect_attention_hooks,
)

from src.model.photomaker_branched.branch_helpers import log_debug_image

from src.model.photomaker_branched.mask_utils import compute_binary_face_mask, simple_threshold_mask
from src.model.photomaker_branched.mask_utils import MASK_LAYERS_CONFIG

from src.model.photomaker_branched.add_masking import DynamicMaskGenerator, get_default_mask_config

import os
import numpy as np
from PIL import Image
import torch.nn.functional as F

import PIL
import torch
from transformers import CLIPImageProcessor

from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.callbacks import (
    MultiPipelineCallbacks,
    PipelineCallback,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    _get_model_file,
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    scale_lora_layers,
    unscale_lora_layers,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from src.model.photomaker_branched.model import PhotoMakerIDEncoder  # PhotoMaker v1
from src.model.photomaker_branched.model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken  # PhotoMaker v2
from src.model.photomaker_branched.insightface_package import FaceAnalysis2, analyze_faces

# Copied utility functions and the PhotomakerBranchedPipeline class body from
# the original photomaker_branched.py without further changes, only imports above
# are redirected to branched_new2.

# To avoid large duplication in this patch, we re-export the class from the
# original module after adjusting imports; if needed, copy full class body here.

from .photomaker_branched import PhotomakerBranchedPipeline as _OrigPhotomakerBranchedPipeline


class PhotomakerBranchedPipeline(_OrigPhotomakerBranchedPipeline):
    pass

