import gc
import os

import cv2
import insightface
import numpy as np
import torch
import torch.nn as nn
from .utils import img2tensor, tensor2img
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from pulid.encoders_transformer import IDFormer
from pulid.utils import is_torch2_available, sample_dpmpp_2m, sample_dpmpp_sde
from .branched_core_NS import PipeProxy, letterbox_and_encode_reference, build_face_masks, two_branch_cfg_step
import torch.nn.functional as F

# Reuse PuLID ID processor without touching it
from pulid.attention_processor import IDAttnProcessor, AttnProcessor

class PuLIDPipeline:
    def __init__(self, sdxl_repo='Lykon/dreamshaper-xl-lightning', sampler='dpmpp_sde', *args, **kwargs):
        super().__init__()
        self.device = 'cuda'

        # load base model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_repo, torch_dtype=torch.float16, variant="fp16").to(
            self.device
        )
        self.pipe.watermark = None
        self.hack_unet_attn_layers(self.pipe.unet)

        # scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        # ID adapters
        self.id_adapter = IDFormer().to(self.device)

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        self.app = FaceAnalysis(
            name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        self.load_pretrain()

        # other configs
        self.debug_img_list = []

        # karras schedule related code, borrow from lllyasviel/Omost
        linear_start = 0.00085
        linear_end = 0.012
        timesteps = 1000
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)

        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_data = 1.0

        if sampler == 'dpmpp_sde':
            self.sampler = sample_dpmpp_sde
        elif sampler == 'dpmpp_2m':
            self.sampler = sample_dpmpp_2m
        else:
            raise NotImplementedError(f'sampler {sampler} not implemented')
        # ==== Branched Attention (config) BEGIN ========================================
        self.use_branched_attention = False
        self._branched_start_step = 1
        self.pose_adapt_ratio = 0.25
        self.ca_mixing_for_face = True
        self._face_mask_img = None
        self._face_mask_ref_img = None
        self._reference_latents = None
        self._debug_dir = None
        # ==== Branched Attention (config) END ==========================================

        # ─── Branched toggle (minimal) ───────────────────────────────────────
        self.use_branched_attention = False
        self._orig_attn1 = {}
        self._face_mask_img = None     # H×W numpy/tensor (0/1)
        self._face_mask_ref_img = None # H×W numpy/tensor (0/1)
        self._step_counter = 0  # monotonic per-run counter for branched gating 


    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def get_sigmas_karras(self, n, rho=7.0):
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def hack_unet_attn_layers(self, unet):
        id_adapter_attn_procs = {}
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is not None:
                id_adapter_attn_procs[name] = IDAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(id_adapter_attn_procs)
        self.id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())

    def load_pretrain(self):
        hf_hub_download('guozinan/PuLID', 'pulid_v1.1.safetensors', local_dir='models')
        ckpt_path = 'models/pulid_v1.1.safetensors'
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    def get_id_embedding(self, image_list):
        """
        Args:
            image in image_list: numpy rgb image, range [0, 255]
        """
        id_cond_list = []
        id_vit_hidden_list = []
        for ii, image in enumerate(image_list):
            self.face_helper.clean_all()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # get antelopev2 embedding
            face_info = self.app.get(image_bgr)
            if len(face_info) > 0:
                face_info = sorted(
                    face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
                )[
                    -1
                ]  # only use the maximum face
                id_ante_embedding = face_info['embedding']
                self.debug_img_list.append(
                    image[
                        int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                        int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                    ]
                )
            else:
                id_ante_embedding = None

            # using facexlib to detect and align face
            self.face_helper.read_image(image_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) == 0:
                raise RuntimeError('facexlib align face fail')
            align_face = self.face_helper.cropped_faces[0]
            # incase insightface didn't detect face
            if id_ante_embedding is None:
                print('fail to detect face using insightface, extract embedding on align face')
                id_ante_embedding = self.handler_ante.get_feat(align_face)

            id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
            if id_ante_embedding.ndim == 1:
                id_ante_embedding = id_ante_embedding.unsqueeze(0)

            # parsing
            input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
            input = input.to(self.device)
            parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[
                0
            ]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)
            # only keep the face features
            face_features_image = torch.where(bg, white_image, self.to_gray(input))
            self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

            # transform img before sending to eva-clip-vit
            face_features_image = resize(
                face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC
            )
            face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
            id_cond_vit, id_vit_hidden = self.clip_vision_model(
                face_features_image, return_all_features=False, return_hidden=True, shuffle=False
            )
            id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
            id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

            id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

            id_cond_list.append(id_cond)
            id_vit_hidden_list.append(id_vit_hidden)

        id_uncond = torch.zeros_like(id_cond_list[0])
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden_list[0])):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden_list[0][layer_idx]))

        id_cond = torch.stack(id_cond_list, dim=1)
        id_vit_hidden = id_vit_hidden_list[0]
        for i in range(1, len(image_list)):
            for j, x in enumerate(id_vit_hidden_list[i]):
                id_vit_hidden[j] = torch.cat([id_vit_hidden[j], x], dim=1)
        id_embedding = self.id_adapter(id_cond, id_vit_hidden)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)

        # return id_embedding
        return uncond_id_embedding, id_embedding

    # ───────────────────────────── Branched (minimal) ─────────────────────────
    def _make_center_mask(self, B, H, W, device, dtype):
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        cy, cx = H / 2.0, W / 2.0
        ry, rx = H * 0.35, W * 0.35
        m = (((yy - cy) ** 2) / (ry ** 2) + ((xx - cx) ** 2) / (rx ** 2)) <= 1.0
        m = m.to(dtype=dtype)[None, None].expand(B, 1, H, W)
        return m

    def _mask_from_image(self, B, H, W, device, dtype, ref=False):
        src = self._face_mask_ref_img if ref else self._face_mask_img
        if src is None:
            return None
        if isinstance(src, np.ndarray):
            t = torch.from_numpy(src.astype(np.float32))  # H_img×W_img
        else:
            t = src.float()
        t = t[None, None]  # 1×1×H×W
        t = t.to(device=device, dtype=dtype)
        if (t.shape[-2], t.shape[-1]) != (H, W):
            t = F.interpolate(t, size=(H, W), mode='nearest')
        return t.expand(B, 1, H, W)

    def _patch_self_attn(self, mask4: torch.Tensor):
        if not self._orig_attn1:
            self._orig_attn1 = {}
        procs = {}
        patched = 0
        total_attn1 = 0
        for name, proc in self.pipe.unet.attn_processors.items():
            if name.endswith('attn1.processor'):
                total_attn1 += 1
                if name not in self._orig_attn1:
                    self._orig_attn1[name] = proc
                # infer hidden size by block
                if 'mid_block' in name:
                    hidden_size = self.pipe.unet.config.block_out_channels[-1]
                elif name.startswith('up_blocks'):
                    bid = int(name[len('up_blocks.'):].split('.')[0])
                    hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[bid]
                elif name.startswith('down_blocks'):
                    bid = int(name[len('down_blocks.'):].split('.')[0])
                    hidden_size = self.pipe.unet.config.block_out_channels[bid]
                else:
                    hidden_size = self.pipe.unet.config.block_out_channels[0]
                mproc = MaskedSelfAttnProcessor(hidden_size=hidden_size, strength=0.6)
                mproc = mproc.to(self.device, dtype=self.pipe.unet.dtype)
                mproc.set_mask(mask4)
                procs[name] = mproc
                patched += 1
            else:
                procs[name] = proc
        self.pipe.unet.set_attn_processor(procs)
        print(f"[BrNS] patch_self_attn: patched={patched} / total_attn1={total_attn1}")

    def _restore_self_attn(self):
        if not self._orig_attn1:
            return
        procs = dict(self.pipe.unet.attn_processors)
        for name, orig in self._orig_attn1.items():
            if name in procs:
                procs[name] = orig
        self.pipe.unet.set_attn_processor(procs)
        self._orig_attn1.clear()

    # ───────────────────────────── Sampling entry ─────────────────────────────
    def __call__(self, x, sigma, **extra_args):
        x_ddim_space = x / (sigma[:, None, None, None] ** 2 + self.sigma_data**2) ** 0.5

        t = self.timestep(sigma)
        # use a monotonic per-run counter for start-step gating
        step_idx = getattr(self, "_step_counter", 0)
        self._step_counter = step_idx + 1

        cfg_scale = extra_args['cfg_scale']

        # Optionally compute branched variant via two-branch predict (PhotoMaker logic)
        use_br = getattr(self, 'use_branched_attention', False)
        start_step = int(getattr(self, '_branched_start_step', 1))
        if use_br and step_idx >= start_step:
            B, _, H, W = x_ddim_space.shape
            # # Build masks (gen + ref); fallback to center masks if missing
            # m_gen = self._mask_from_image(B, H, W, x_ddim_space.device, x_ddim_space.dtype, ref=False)
            # m_ref = self._mask_from_image(B, H, W, x_ddim_space.device, x_ddim_space.dtype, ref=True)
            # # Fallback and diagnostics
            # if m_gen is None and m_ref is None:
            #     print("[BrNS] No manual masks found; using center fallback.")
            #     m_gen = self._make_center_mask(B, H, W, x_ddim_space.device, x_ddim_space.dtype)
            #     m_ref = m_gen
            # else:
            #     if m_gen is None:
            #         print("[BrNS] Gen mask missing; using ref mask only.")
            #         m_gen = m_ref
            #     if m_ref is None:
            #         print("[BrNS] Ref mask missing; using gen mask only.")
            #         m_ref = m_gen
            #     # Intersect to restrict to face area and avoid large blobs
            #     m_gen = (m_gen > 0.5).to(dtype=x_ddim_space.dtype)
            #     m_ref = (m_ref > 0.5).to(dtype=x_ddim_space.dtype)
            #     with torch.no_grad():
            #         cov = float((m_gen * m_ref).mean().item())
            #         print(f"[BrNS] Using face intersection mask; coverage={cov:.4f}")
            # Build/resize masks on latent grid (with sensible fallback)
            m_gen, m_ref = build_face_masks(self, B, H, W, x_ddim_space.device, x_ddim_space.dtype)

            # Ensure reference latents are available
            reference_latents = getattr(self, '_reference_latents', None)
            if reference_latents is None:
                raise RuntimeError("Branched mode requires reference latents; pass reference_pil to inference().")

            # Build a proxy that exposes do_cfg + tuning knobs without mutating base pipeline
            proxy = PipeProxy(
                base=self.pipe,
                do_cfg=bool(cfg_scale > 1.0),
                pose_adapt_ratio=float(getattr(self, 'pose_adapt_ratio', 0.25)),
                ca_mixing_for_face=bool(getattr(self, 'ca_mixing_for_face', True)),
            )

            # Prepare positive/negative args for two-branch prediction and run CFG step
            pos_args = extra_args['positive']
            neg_args = extra_args['negative']
            noise_pred = two_branch_cfg_step(
                proxy,
                x_ddim_space,
                t,
                pos_args,
                neg_args,
                m_gen,
                m_ref,
                reference_latents,
                cfg_scale,
                step_idx,
                getattr(self, '_debug_dir', None),
            )
        else:
            # Baseline (no branched processors)
            eps_pos_base = self.pipe.unet(x_ddim_space, t, return_dict=False, **extra_args['positive'])[0]
            eps_neg_base = self.pipe.unet(x_ddim_space, t, return_dict=False, **extra_args['negative'])[0]
            noise_pred = eps_neg_base + cfg_scale * (eps_pos_base - eps_neg_base)

        return x - noise_pred * sigma[:, None, None, None]

    def inference(
        self,
        prompt,
        size,
        prompt_n='',
        id_embedding=None,
        uncond_id_embedding=None,
        id_scale=1.0,
        guidance_scale=1.2,
        steps=4,
        seed=-1,
        use_branched_attention=False,  # NEW: toggle minimal branched
        branched_attn_start_step=1,    # NEW: start step for branched gating
        pose_adapt_ratio: float = 0.25,
        ca_mixing_for_face: bool = True,
        import_mask: str | None = None,
        import_mask_ref: str | None = None,
        import_mask_folder: str | None = None,
        use_mask_folder: bool = False,
        reference_pil=None,
        debug_dir: str | None = None,
    ):

        # sigmas
        sigmas = self.get_sigmas_karras(steps).to(self.device)

        # latents
        noise = torch.randn((size[0], 4, size[1] // 8, size[2] // 8), device="cpu", generator=torch.manual_seed(seed))
        noise = noise.to(dtype=self.pipe.unet.dtype, device=self.device)
        latents = noise * sigmas[0].to(noise)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=prompt_n,
        )

        add_time_ids = list((size[1], size[2]) + (0, 0) + (size[1], size[2]))
        add_time_ids = torch.tensor([add_time_ids], dtype=self.pipe.unet.dtype, device=self.device)
        add_neg_time_ids = add_time_ids.clone()

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                cross_attention_kwargs={'id_embedding': id_embedding, 'id_scale': id_scale},
            ),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids},
                cross_attention_kwargs={'id_embedding': uncond_id_embedding, 'id_scale': id_scale},
            ),
            seed=seed,
        )

        # Load manual masks if provided
        def _load_mask(path: str | None):
            if not path:
                return None
            try:
                from PIL import Image as _PILImage
                m = _PILImage.open(path).convert('L')
                arr = (np.array(m) > 127).astype(np.float32)
                try:
                    abspath = os.path.abspath(path)
                    print(f"[BrNS] Loaded mask: {abspath}  coverage={arr.mean():.4f}  shape={arr.shape}")
                except Exception:
                    pass
                return arr
            except Exception as e:
                print(f"[BrNS] Warning: failed to load mask '{path}': {e}")
                return None

        if use_branched_attention:
            # Prefer explicit import paths; the CLI precomputes them if use_mask_folder is True
            self._face_mask_img = _load_mask(import_mask)
            self._face_mask_ref_img = _load_mask(import_mask_ref)
            # # Prepare reference latents for branched two-branch predict
            # if reference_pil is not None:
            #     from PIL import Image as _PILImage
            #     # Preserve aspect ratio by letterboxing to (width,height) = (size[2], size[1])
            #     tgt_h, tgt_w = int(size[1]), int(size[2])
            #     # PIL size returns (W,H)
            #     orig_w, orig_h = reference_pil.size
            #     scale = min(tgt_w / float(orig_w), tgt_h / float(orig_h))
            #     rw = max(1, int(round(orig_w * scale)))
            #     rh = max(1, int(round(orig_h * scale)))
            #     pl = (tgt_w - rw) // 2
            #     pr = tgt_w - rw - pl
            #     pt = (tgt_h - rh) // 2
            #     pb = tgt_h - rh - pt
            #     resized = reference_pil.resize((rw, rh), resample=_PILImage.LANCZOS)
            #     canvas = _PILImage.new('RGB', (tgt_w, tgt_h), color=(0, 0, 0))
            #     canvas.paste(resized, (pl, pt))

            #     # Store padding and sizes for debug consumers
            #     self._ref_pad = (pl, pr, pt, pb)
            #     self._ref_scaled_size = (rh, rw)
            #     self._ref_orig_size = (orig_h, orig_w)
            #     # Mirror on inner pipe for PhotoMaker helpers
            #     setattr(self.pipe, "_ref_pad", self._ref_pad)
            #     setattr(self.pipe, "_ref_scaled_size", self._ref_scaled_size)
            #     setattr(self.pipe, "_ref_orig_size", self._ref_orig_size)

            #     # If a reference mask was provided at original resolution, store it 1:1
            #     if self._face_mask_ref_img is not None:
            #         # Save on the inner pipe so helpers use high‑res path without any stretch
            #         import numpy as _np
            #         hi = (self._face_mask_ref_img > 0.5).astype(_np.uint8)
            #         setattr(self.pipe, "_face_mask_highres_ref", hi)
            #         setattr(self.pipe, "_face_mask_scaled_size_ref", (rh, rw))
            #         setattr(self.pipe, "_face_mask_pad_ref", (pl, pr, pt, pb))

            #     # Preprocess image to pixel values in [-1,1]
            #     pixel_values = self.pipe.image_processor.preprocess(
            #         canvas, height=tgt_h, width=tgt_w
            #     )
            #     pixel_values = pixel_values.to(device=self.device, dtype=self.pipe.vae.dtype)
            #     with torch.no_grad():
            #         dist = self.pipe.vae.encode(pixel_values).latent_dist
            #         latents_ref = dist.mean
            #     latents_ref = latents_ref * self.pipe.vae.config.scaling_factor
            #     # Save for __call__ and debug
            #     self._reference_latents = latents_ref
            #     # Also attach to the inner diffusers pipeline so PM helpers can find it
            #     setattr(self.pipe, "_reference_latents", latents_ref)
            
            # Prepare reference latents via AR-preserving letterbox + VAE encode
            if reference_pil is not None:
                letterbox_and_encode_reference(
                    self,
                    reference_pil,
                    size_hw=(size[1], size[2]),
                    mask_ref_img=self._face_mask_ref_img,
                )
            
            # Debug output directory for branched helpers
            if debug_dir is not None:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                except Exception:
                    pass
                setattr(self, '_debug_dir', debug_dir)
                try:
                    setattr(self.pipe, '_debug_dir', debug_dir)
                except Exception:
                    pass
            # Persist loaded masks for visual verification
            dbg = debug_dir or getattr(self, '_debug_dir', None)
            if dbg is not None:
                from PIL import Image as _PILImage
                if self._face_mask_img is not None:
                    _PILImage.fromarray((self._face_mask_img*255).astype(np.uint8)).save(os.path.join(dbg, 'mask_gen_loaded.png'))
                if self._face_mask_ref_img is not None:
                    _PILImage.fromarray((self._face_mask_ref_img*255).astype(np.uint8)).save(os.path.join(dbg, 'mask_ref_loaded.png'))

            # Save once-off debug images for reference latents and mask overlay
            dbg = debug_dir or getattr(self, '_debug_dir', None)
            if dbg is not None and getattr(self, "_reference_latents", None) is not None:
                # Build a latent-resolution ref mask tensor if possible
                B, C, H, W = self._reference_latents.shape
                mask4_ref = self._mask_from_image(1, H, W, self.device, self.pipe.unet.dtype, ref=True)
                from photomaker.branch_helpers import (
                    save_debug_ref_latents as _pm_save_ref_latents,
                    save_debug_ref_mask_overlay as _pm_save_ref_mask_overlay,
                )
                # Helpers expect attributes on the inner pipeline
                setattr(self.pipe, "_reference_latents", self._reference_latents)
                _pm_save_ref_latents(self.pipe, dbg)
                _pm_save_ref_mask_overlay(self.pipe, mask4_ref, dbg)

        # NEW: Toggle branched
        self.use_branched_attention = bool(use_branched_attention)
        if self.use_branched_attention:
            print(f"[BrNS] Branched self-attention ON (strength=0.9, start_step={branched_attn_start_step})")
        # store start step for __call__ gating
        self._branched_start_step = int(branched_attn_start_step)
        # store tuning knobs (used by self-attn patch)
        self.pose_adapt_ratio = float(pose_adapt_ratio)
        self.ca_mixing_for_face = bool(ca_mixing_for_face)
        

        # reset per-run step counter just before the sampler loop
        self._step_counter = 0

        latents = self.sampler(self, latents, sigmas, extra_args=sampler_kwargs, disable=False)
        latents = latents.to(dtype=self.pipe.vae.dtype, device=self.device) / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            images = self.pipe.vae.decode(latents).sample
        images = images.detach()
        images = self.pipe.image_processor.postprocess(images, output_type='pil')

        # Save debug reference latents + mask overlay once per run (final)
        dbg = getattr(self, '_debug_dir', None)
        if bool(use_branched_attention) and dbg is not None and getattr(self, '_reference_latents', None) is not None:
            # Ensure inner pipe carries the attribute
            setattr(self.pipe, "_reference_latents", self._reference_latents)
            # Build mask4_ref on latent grid
            B, C, H, W = self._reference_latents.shape
            mask4_ref = self._mask_from_image(1, H, W, self.device, self.pipe.unet.dtype, ref=True)
            from photomaker.branch_helpers import (
                save_debug_ref_latents as _pm_save_ref_latents,
                save_debug_ref_mask_overlay as _pm_save_ref_mask_overlay,
            )
            _pm_save_ref_latents(self.pipe, dbg)
            _pm_save_ref_mask_overlay(self.pipe, mask4_ref, dbg)

        # Reset flag to avoid accidental reuse
        self.use_branched_attention = False
        return images
