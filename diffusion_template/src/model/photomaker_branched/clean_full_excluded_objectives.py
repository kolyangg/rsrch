"""Quarantined objective helpers retained only for checkpoint compatibility.

All allowlisted clean_full configs keep these objectives disabled.  Keeping the
methods on a mixin preserves fail-closed legacy branches without obscuring the
shared model forward path.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ExcludedObjectiveCompatibilityMixin:
    """Inactive CL19-teacher and predicted-x0 identity objective helpers."""
    def _swap_unet_processors(self, processors) -> None:
        """Install a processor map without allowing Diffusers to consume it."""
        self.unet.set_attn_processor(dict(processors))

    def _frozen_cl19_prediction(
        self,
        *,
        noisy_latents,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
        mask4,
        mask4_ref,
        reference_latents,
        face_prompt_embeds,
        class_tokens_mask,
        id_features,
    ):
        current = dict(self.unet.attn_processors)
        if self._ba_frozen_teacher_unet is None:
            frozen = {name: copy.deepcopy(proc) for name, proc in current.items()}
            for processor in frozen.values():
                if isinstance(processor, torch.nn.Module):
                    processor.requires_grad_(False)
                    processor.eval()
            # Training-only snapshot; deliberately outside the module tree so
            # the anchor cannot enter optimizer or checkpoint ownership.
            self._ba_frozen_teacher_unet = frozen
        previous_suppression = bool(getattr(self, "_ba_suppress_telemetry", False))
        self._ba_suppress_telemetry = True
        try:
            self._swap_unet_processors(self._ba_frozen_teacher_unet)
            with torch.no_grad():
                return run_branched_forward_pass(
                    self,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    prompt_embeds=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    mask4=mask4,
                    mask4_ref=mask4_ref,
                    reference_latents=reference_latents,
                    face_prompt_embeds=face_prompt_embeds,
                    class_tokens_mask=class_tokens_mask,
                    id_features=id_features,
                    reference_noise=getattr(self, "_ref_noise", None),
                ).detach()
        finally:
            self._swap_unet_processors(current)
            self._ba_suppress_telemetry = previous_suppression

    def _native_photomaker_prediction(
        self,
        *,
        noisy_latents,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
    ):
        original = getattr(self, "_original_attn_processors", None)
        if not original:
            raise RuntimeError("Native PhotoMaker teacher processors were not retained")
        current = dict(self.unet.attn_processors)
        try:
            self._swap_unet_processors(original)
            with torch.no_grad():
                return self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0].detach()
        finally:
            self._swap_unet_processors(current)

    def _pm_boundary_distillation(
        self,
        *,
        student,
        noisy_latents,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
        face_mask,
    ):
        zero = student.float().new_tensor(0.0)
        top = getattr(self, "_ba_ownership_target_mask", None)
        if (
            not self.training
            or not self.ba_pm_boundary_distill_enabled
            or top is None
            or float(top.detach().sum()) <= 0.0
            or torch.rand((), device=student.device).item()
            >= self.ba_pm_boundary_distill_probability
        ):
            return zero, zero, zero, zero, zero
        top = F.interpolate(top.float(), size=student.shape[-2:], mode="nearest")
        face = F.interpolate(face_mask.float(), size=student.shape[-2:], mode="nearest")
        top = top.clamp(0.0, 1.0) * (F.max_pool2d(face, 3, 1, 1) > 0).float()
        width = self.ba_pm_boundary_distill_width
        kernel = 2 * width + 1
        dilated = F.max_pool2d(top, kernel, 1, width)
        eroded = 1.0 - F.max_pool2d(1.0 - top, kernel, 1, width)
        boundary = (dilated - eroded).clamp(0.0, 1.0) * face
        teacher = self._native_photomaker_prediction(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        )
        charbonnier = ((student.float() - teacher.float()).square() + 1.0e-6).sqrt()

        def masked_mean(mask):
            return (charbonnier * mask).sum() / (
                mask.sum() * charbonnier.shape[1]
            ).clamp_min(1.0)

        boundary_loss = masked_mean(boundary)
        top_loss = masked_mean(top)
        weighted = (
            self.ba_pm_boundary_distill_weight * boundary_loss
            + self.ba_pm_boundary_distill_top_weight * top_loss
        )
        divergence = ((student.float() - teacher.float()).square().mean()).sqrt()
        return weighted, boundary_loss, top_loss, boundary.mean(), divergence

    @staticmethod
    def _reference_prediction_delta_ratio(
        correct: torch.Tensor,
        wrong: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = target_mask.detach().float()
        if mask.shape[-2:] != correct.shape[-2:]:
            mask = F.interpolate(mask, size=correct.shape[-2:], mode="nearest")
        if mask.shape[0] != correct.shape[0]:
            if correct.shape[0] % mask.shape[0] != 0:
                raise RuntimeError("BA prediction-delta mask batch mismatch")
            mask = mask.repeat(correct.shape[0] // mask.shape[0], 1, 1, 1)
        denom = (
            mask.sum(dim=(1, 2, 3)) * correct.shape[1]
        ).clamp_min(1.0)
        correct_energy = (
            correct.detach().float().square() * mask
        ).sum(dim=(1, 2, 3)) / denom
        delta_energy = (
            (correct.detach().float() - wrong.detach().float()).square() * mask
        ).sum(dim=(1, 2, 3)) / denom
        return (
            delta_energy.clamp_min(0.0).sqrt()
            / correct_energy.clamp_min(1.0e-12).sqrt()
        ).mean()

    def _identity_aux_weight(self, global_step: int) -> float:
        if global_step < self.identity_aux_ramp_start_step:
            return 0.0
        progress = min(
            1.0,
            (global_step - self.identity_aux_ramp_start_step)
            / (
                self.identity_aux_ramp_end_step
                - self.identity_aux_ramp_start_step
            ),
        )
        return self.identity_aux_max_weight * progress

    def _face_crop_for_identity_proxy(
        self,
        image: torch.Tensor,
        bbox: Sequence[float],
    ) -> torch.Tensor:
        height, width = image.shape[-2:]
        x0, y0, x1, y1 = [float(value) for value in bbox]
        pad_x = (x1 - x0) * self.identity_aux_crop_padding
        pad_y = (y1 - y0) * self.identity_aux_crop_padding
        x0 = max(0, min(width - 1, int(np.floor(x0 - pad_x))))
        y0 = max(0, min(height - 1, int(np.floor(y0 - pad_y))))
        x1 = max(x0 + 1, min(width, int(np.ceil(x1 + pad_x))))
        y1 = max(y0 + 1, min(height, int(np.ceil(y1 + pad_y))))
        crop = image[..., y0:y1, x0:x1]
        return F.interpolate(
            crop,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _predicted_x0_photomaker_clip_auxiliary(
        self,
        *,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        pixel_values: torch.Tensor,
        face_bbox: Sequence[Sequence[float]],
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = noise_pred.float().new_tensor(0.0)
        weight = self._identity_aux_weight(global_step)
        if (
            not self.identity_aux_enabled
            or weight <= 0.0
            or global_step % self.identity_aux_cadence != 0
        ):
            return zero, zero, zero
        eligible = torch.nonzero(
            timesteps <= self.identity_aux_max_timestep,
            as_tuple=False,
        ).flatten()
        if eligible.numel() == 0:
            return zero, zero, zero
        index = int(eligible[0].item())
        timestep = int(timesteps[index].item())
        alpha = self.noise_scheduler.alphas_cumprod[timestep].to(
            device=noise_pred.device,
            dtype=torch.float32,
        )
        predicted_x0 = (
            noisy_latents[index : index + 1].float()
            - (1.0 - alpha).sqrt() * noise_pred[index : index + 1].float()
        ) / alpha.sqrt().clamp_min(1.0e-6)
        decoded = self.vae.decode(
            (
                predicted_x0
                / float(self.vae.config.scaling_factor)
            ).to(dtype=self.vae.dtype),
            return_dict=False,
        )[0]
        predicted_face = self._face_crop_for_identity_proxy(
            decoded,
            face_bbox[index],
        )
        target_face = self._face_crop_for_identity_proxy(
            pixel_values[index : index + 1],
            face_bbox[index],
        )

        mean = torch.tensor(
            self.id_image_processor.image_mean,
            device=predicted_face.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.id_image_processor.image_std,
            device=predicted_face.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        def normalize(image: torch.Tensor) -> torch.Tensor:
            image = (image.float().clamp(-1.0, 1.0) + 1.0) * 0.5
            return ((image - mean) / std).to(self.id_encoder.dtype)

        predicted_embedding = self.id_encoder.vision_model(
            normalize(predicted_face)
        )[1].float()
        with torch.no_grad():
            target_embedding = self.id_encoder.vision_model(
                normalize(target_face)
            )[1].float()
        identity_loss = 1.0 - F.cosine_similarity(
            predicted_embedding,
            target_embedding,
            dim=-1,
        ).mean()
        return (
            identity_loss,
            zero.new_tensor(weight),
            zero.new_tensor(1.0),
        )

    def _predicted_clean_latents(
        self,
        *,
        noisy_latents: torch.Tensor,
        model_prediction: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        alpha = self.noise_scheduler.alphas_cumprod[timestep].to(
            device=model_prediction.device,
            dtype=torch.float32,
        )
        beta = 1.0 - alpha
        prediction_type = str(
            getattr(self.noise_scheduler.config, "prediction_type", "epsilon")
        ).lower()
        noisy = noisy_latents.float()
        prediction = model_prediction.float()
        if prediction_type == "epsilon":
            return (noisy - beta.sqrt() * prediction) / alpha.sqrt().clamp_min(
                1.0e-6
            )
        if prediction_type == "v_prediction":
            return alpha.sqrt() * noisy - beta.sqrt() * prediction
        if prediction_type == "sample":
            return prediction
        raise RuntimeError(
            f"Unsupported scheduler prediction_type for identity loss: {prediction_type}"
        )

    def _arcface_roi_crop(
        self,
        image: torch.Tensor,
        bbox: Sequence[float],
    ) -> torch.Tensor:
        from torchvision.ops import roi_align

        if image.shape[0] != 1:
            raise ValueError("ArcFace ROI helper expects a single image")
        height, width = image.shape[-2:]
        x0, y0, x1, y1 = [float(value) for value in bbox]
        pad_x = (x1 - x0) * self.identity_aux_crop_padding
        pad_y = (y1 - y0) * self.identity_aux_crop_padding
        x0 = max(0.0, min(float(width - 1), x0 - pad_x))
        y0 = max(0.0, min(float(height - 1), y0 - pad_y))
        x1 = max(x0 + 1.0, min(float(width), x1 + pad_x))
        y1 = max(y0 + 1.0, min(float(height), y1 + pad_y))
        boxes = image.new_tensor([[0.0, x0, y0, x1, y1]], dtype=torch.float32)
        return roi_align(
            image.float(),
            boxes,
            output_size=(112, 112),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )

    @staticmethod
    def _pil_to_normalized_rgb(
        image: Image.Image,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        # InsightFace's buffalo_l ArcFace path uses (RGB - 127.5) / 127.5.
        return (tensor.to(device=device, dtype=torch.float32) - 127.5) / 127.5

    def _predicted_x0_arcface_auxiliary(
        self,
        *,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        pixel_values: torch.Tensor,
        face_bbox: Sequence[Sequence[float]],
        ref_images: Sequence[Sequence[Image.Image]],
        face_bbox_ref: Sequence[Sequence[float]] | None,
        identity_face_bboxes_ref: Sequence[Sequence[Sequence[float]]] | None,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        zero = noise_pred.float().new_tensor(0.0)
        telemetry = {
            "identity_aux_cosine": zero,
            "identity_aux_timestep": zero,
            "identity_aux_pred_norm": zero,
            "identity_aux_target_norm": zero,
        }
        weight = self._identity_aux_weight(global_step)
        if (
            not self.identity_aux_enabled
            or weight <= 0.0
            or global_step % self.identity_aux_cadence != 0
        ):
            return zero, zero, zero, telemetry
        eligible = torch.nonzero(
            timesteps <= self.identity_aux_max_timestep,
            as_tuple=False,
        ).flatten()
        if eligible.numel() == 0:
            return zero, zero, zero, telemetry
        if self.identity_aux_recognizer is None:
            raise RuntimeError("ArcFace auxiliary recognizer was not initialized")
        if face_bbox_ref is None:
            raise RuntimeError("ArcFace identity auxiliary requires reference boxes")

        index = int(eligible[0].item())
        timestep = int(timesteps[index].item())
        predicted_x0 = self._predicted_clean_latents(
            noisy_latents=noisy_latents[index : index + 1],
            model_prediction=noise_pred[index : index + 1],
            timestep=timestep,
        )
        decoded = self.vae.decode(
            (
                predicted_x0 / float(self.vae.config.scaling_factor)
            ).to(dtype=self.vae.dtype),
            return_dict=False,
        )[0]
        predicted_face = self._arcface_roi_crop(decoded, face_bbox[index])

        refs = ref_images[index]
        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        if not refs:
            raise RuntimeError("ArcFace identity auxiliary received no reference image")
        reference_boxes = (
            identity_face_bboxes_ref[index]
            if identity_face_bboxes_ref is not None
            else [face_bbox_ref[index]]
        )
        if len(reference_boxes) != len(refs):
            raise RuntimeError(
                "ArcFace centroid requires one bbox per distinct reference: "
                f"refs={len(refs)}, boxes={len(reference_boxes)}"
            )
        reference_faces = []
        for reference_image, reference_box in zip(refs, reference_boxes):
            reference_rgb = self._pil_to_normalized_rgb(
                reference_image,
                device=decoded.device,
            )
            reference_faces.append(
                self._arcface_roi_crop(reference_rgb, reference_box)
            )
        target_face = self._arcface_roi_crop(
            pixel_values[index : index + 1].float(),
            face_bbox[index],
        )

        predicted_raw = self.identity_aux_recognizer(
            predicted_face.float().clamp(-1.0, 1.0)
        )
        with torch.no_grad():
            target_raw = self.identity_aux_recognizer(
                torch.cat(
                    [target_face.float().clamp(-1.0, 1.0)]
                    + [face.float().clamp(-1.0, 1.0) for face in reference_faces],
                    dim=0,
                )
            )
            target_embedding = F.normalize(target_raw.float(), dim=-1).mean(
                dim=0,
                keepdim=True,
            )
            target_embedding = F.normalize(target_embedding, dim=-1)
        predicted_embedding = F.normalize(predicted_raw.float(), dim=-1)
        cosine = F.cosine_similarity(
            predicted_embedding,
            target_embedding,
            dim=-1,
        ).mean()
        if self.identity_aux_mode == "quadratic_hinge":
            identity_loss = F.relu(
                cosine.new_tensor(self.identity_aux_hinge_margin) - cosine
            ).square()
        else:
            identity_loss = 1.0 - cosine
        telemetry = {
            "identity_aux_cosine": cosine.detach(),
            "identity_aux_timestep": zero.new_tensor(float(timestep)),
            "identity_aux_pred_norm": predicted_raw.detach().float().norm(dim=-1).mean(),
            "identity_aux_target_norm": target_raw.detach().float().norm(dim=-1).mean(),
        }
        return (
            identity_loss,
            zero.new_tensor(weight),
            zero.new_tensor(1.0),
            telemetry,
        )

    def _predicted_x0_identity_auxiliary(
        self,
        *,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        pixel_values: torch.Tensor,
        face_bbox: Sequence[Sequence[float]],
        ref_images: Sequence[Sequence[Image.Image]],
        face_bbox_ref: Sequence[Sequence[float]] | None,
        identity_face_bboxes_ref: Sequence[Sequence[Sequence[float]]] | None,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        zero = noise_pred.float().new_tensor(0.0)
        if self.identity_aux_backend == "arcface_torch_v2":
            return self._predicted_x0_arcface_auxiliary(
                noisy_latents=noisy_latents,
                noise_pred=noise_pred,
                timesteps=timesteps,
                pixel_values=pixel_values,
                face_bbox=face_bbox,
                ref_images=ref_images,
                face_bbox_ref=face_bbox_ref,
                identity_face_bboxes_ref=identity_face_bboxes_ref,
                global_step=global_step,
            )
        loss, weight, applied = self._predicted_x0_photomaker_clip_auxiliary(
            noisy_latents=noisy_latents,
            noise_pred=noise_pred,
            timesteps=timesteps,
            pixel_values=pixel_values,
            face_bbox=face_bbox,
            global_step=global_step,
        )
        return loss, weight, applied, {
            "identity_aux_cosine": zero,
            "identity_aux_timestep": zero,
            "identity_aux_pred_norm": zero,
            "identity_aux_target_norm": zero,
        }

    @staticmethod
    def _dino_input(face: torch.Tensor) -> torch.Tensor:
        face = F.interpolate(
            (face.float().clamp(-1.0, 1.0) + 1.0) * 0.5,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = face.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = face.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (face - mean) / std

    def _predicted_x0_patch_identity_auxiliary(
        self,
        *,
        noisy_latents,
        noise_pred,
        timesteps,
        face_bbox,
        ref_images,
        face_bbox_ref,
        identity_face_bboxes_ref,
        global_step: int,
        visible_gate_mass: torch.Tensor,
    ):
        zero = noise_pred.float().new_tensor(0.0)
        metrics = {
            "loss_ba_patch_identity": zero,
            "ba/patch_identity_similarity": zero,
            "ba/patch_identity_gate_mass": visible_gate_mass.detach(),
            "ba/patch_identity_applied_fraction": zero,
        }
        if (
            not self.ba_patch_identity_enabled
            or global_step % self.ba_patch_identity_cadence != 0
            or float(visible_gate_mass.detach())
            < self.ba_patch_identity_min_gate_mass
        ):
            return zero, metrics
        eligible = torch.nonzero(
            timesteps <= self.ba_patch_identity_max_timestep, as_tuple=False
        ).flatten()
        if eligible.numel() == 0:
            return zero, metrics
        if self.ba_patch_identity_encoder is None or face_bbox_ref is None:
            raise RuntimeError("DINO patch identity backend or reference boxes missing")
        index = int(eligible[0].item())
        timestep = int(timesteps[index].item())
        predicted_x0 = self._predicted_clean_latents(
            noisy_latents=noisy_latents[index:index + 1],
            model_prediction=noise_pred[index:index + 1],
            timestep=timestep,
        )
        decoded = self.vae.decode(
            (predicted_x0 / float(self.vae.config.scaling_factor)).to(self.vae.dtype),
            return_dict=False,
        )[0]
        predicted_face = self._arcface_roi_crop(decoded, face_bbox[index])
        refs = ref_images[index]
        refs = refs if isinstance(refs, (list, tuple)) else [refs]
        boxes = (
            identity_face_bboxes_ref[index]
            if identity_face_bboxes_ref is not None
            else [face_bbox_ref[index]]
        )
        if len(refs) != len(boxes) or len(refs) < 2:
            raise RuntimeError("DINO patch identity requires distinct same-ID references")
        ref_faces = [
            self._arcface_roi_crop(
                self._pil_to_normalized_rgb(image, device=decoded.device), box
            )
            for image, box in zip(refs, boxes)
        ]
        predicted_tokens = self.ba_patch_identity_encoder.forward_features(
            self._dino_input(predicted_face)
        )["x_norm_patchtokens"]
        with torch.no_grad():
            reference_tokens = self.ba_patch_identity_encoder.forward_features(
                self._dino_input(torch.cat(ref_faces))
            )["x_norm_patchtokens"].flatten(0, 1)
            reference_tokens = F.normalize(reference_tokens.float(), dim=-1)
        predicted_tokens = F.normalize(predicted_tokens.float().squeeze(0), dim=-1)
        similarity = (predicted_tokens @ reference_tokens.transpose(0, 1)).max(1).values.mean()
        ramp = max(
            0.0,
            min(
                1.0,
                (global_step - self.ba_patch_identity_ramp_start_step)
                / float(
                    self.ba_patch_identity_ramp_end_step
                    - self.ba_patch_identity_ramp_start_step
                ),
            ),
        )
        weighted = ramp * self.ba_patch_identity_weight * (1.0 - similarity)
        metrics.update(
            {
                "loss_ba_patch_identity": (1.0 - similarity).detach(),
                "ba/patch_identity_similarity": similarity.detach(),
                "ba/patch_identity_applied_fraction": zero.new_tensor(1.0),
            }
        )
        return weighted, metrics
