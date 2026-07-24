"""Training-only NN3a_new1 variants.

The production model is imported unchanged.  This subclass only controls
which already-installed branched-attention parameters enter the optimizer.
Consequently all scope variants have the exact same step-zero forward graph.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

from src.model.photomaker_branched.lora2 import (
    PhotomakerBranchedLora as ProductionPhotomakerBranchedLora,
)


class NN3aTrainingLabModel(ProductionPhotomakerBranchedLora):
    """NN3a_new1 with experiment-local trainable-scope toggles."""

    _VALID_SCOPES = {"all", "up", "up0", "up1"}

    def __init__(
        self,
        *args,
        lab_train_scope: str = "all",
        lab_optimizer_recipe: str = "production",
        lab_ref_kv_lr_scale: float = 1.0,
        lab_ref_v_lr_scale: float = 1.0,
        lab_ref_q_lr_scale: float = 1.0,
        lab_noise_lr_scale: float = 0.25,
        lab_up0_lr_scale: float = 1.0,
        lab_pm_teacher_weight: float = 0.0,
        **kwargs,
    ):
        scope = str(lab_train_scope).lower()
        if scope not in self._VALID_SCOPES:
            raise ValueError(
                f"lab_train_scope must be one of {sorted(self._VALID_SCOPES)}, "
                f"got {scope!r}"
            )

        # train.py only injects these legacy controls for its production target
        # string.  Keep their NN3a_new1 values explicit for this local subclass.
        kwargs.setdefault("train_ba_only", True)
        kwargs.setdefault("train_ba_all_steps", True)
        kwargs.setdefault("ba_train_top_k", 1.0)
        kwargs.setdefault("ba_patch_top_k", 1.0)
        kwargs.setdefault("non_ba_train", False)
        kwargs.setdefault("ba_weights_split", False)
        kwargs.setdefault("use_attn_v2", False)

        self.lab_train_scope = scope
        self.lab_optimizer_recipe = str(lab_optimizer_recipe).lower()
        if self.lab_optimizer_recipe not in {
            "production",
            "projection_split",
            "reference_value_only",
        }:
            raise ValueError(
                "lab_optimizer_recipe must be 'production', 'projection_split', "
                "or 'reference_value_only'"
            )
        self.lab_ref_kv_lr_scale = float(lab_ref_kv_lr_scale)
        self.lab_ref_v_lr_scale = float(lab_ref_v_lr_scale)
        self.lab_ref_q_lr_scale = float(lab_ref_q_lr_scale)
        self.lab_noise_lr_scale = float(lab_noise_lr_scale)
        self.lab_up0_lr_scale = float(lab_up0_lr_scale)
        self.lab_pm_teacher_weight = float(lab_pm_teacher_weight)
        if self.lab_pm_teacher_weight < 0.0:
            raise ValueError("lab_pm_teacher_weight must be non-negative")
        super().__init__(*args, **kwargs)
        if self.lab_pm_teacher_weight > 0.0:
            # Reuse the production trainer's generic scalar auxiliary-loss
            # plumbing without changing production code. The local forward
            # below emits this slot as the PM-teacher preservation MSE.
            self.ba_null_residual_loss_weight = self.lab_pm_teacher_weight
            print(
                "[NN3a PM teacher] "
                f"full_prediction_mse_weight={self.lab_pm_teacher_weight:g}"
            )

    def forward(self, *args, **kwargs):
        if not self.training or self.lab_pm_teacher_weight <= 0.0:
            return super().forward(*args, **kwargs)

        # Generate a matched PhotoMaker teacher prediction with the exact same
        # target latent, noise, timestep, prompt, and reference augmentation.
        # Restoring the RNG state before the differentiable BA pass means this
        # consumes no additional randomness from the experiment trajectory.
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(self.device)
            if torch.cuda.is_available()
            else None
        )
        controls = {
            name: getattr(self, name)
            for name in (
                "train_ba_all_steps",
                "branched_attn_start_step",
                "use_id_loss",
                "ba_counterfactual_enabled",
                "ba_correctness_guards",
                "ba_pm_id_attenuation_probability",
            )
        }
        original_processors = dict(self.unet.attn_processors)
        teacher_processors = {
            name: (
                AttnProcessor2_0()
                if bool(getattr(processor, "_is_branched_processor", False))
                else processor
            )
            for name, processor in original_processors.items()
        }
        try:
            self.train_ba_all_steps = False
            self.branched_attn_start_step = int(self.num_inference_steps) + 1
            self.use_id_loss = False
            self.ba_counterfactual_enabled = False
            self.ba_correctness_guards = False
            self.ba_pm_id_attenuation_probability = 0.0
            # A one-row PhotoMaker UNet pass cannot go through the installed
            # branched processors because they require [target, reference]
            # doubled batches. Temporarily restore vanilla attention, while
            # retaining the same UNet/PhotoMaker weights and conditioning.
            self.unet.set_attn_processor(dict(teacher_processors))
            with torch.no_grad():
                teacher = super().forward(*args, **kwargs)["model_pred"].detach()
        finally:
            self.unet.set_attn_processor(dict(original_processors))
            for name, value in controls.items():
                setattr(self, name, value)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)

        output = super().forward(*args, **kwargs)
        if "ba_null_residual_loss" in output:
            raise RuntimeError(
                "PM-teacher slot collides with a production null-residual loss"
            )
        output["ba_null_residual_loss"] = F.mse_loss(
            output["model_pred"].float(),
            teacher.float(),
        )
        return output

    def _scope_accepts(self, name: str) -> bool:
        if self.lab_train_scope == "all":
            return True
        if self.lab_train_scope == "up":
            return name.startswith("up_blocks.")
        if self.lab_train_scope == "up0":
            return name.startswith("up_blocks.0.")
        if self.lab_train_scope == "up1":
            return name.startswith("up_blocks.1.")
        raise AssertionError(self.lab_train_scope)

    def _apply_scope_freeze(self) -> tuple[list[str], list[str]]:
        kept_names = []
        removed_names = []
        if self.lab_train_scope == "all":
            return kept_names, removed_names
        for name, parameter in self.unet.named_parameters():
            if not parameter.requires_grad:
                continue
            if self._scope_accepts(name):
                kept_names.append(name)
            else:
                parameter.requires_grad_(False)
                removed_names.append(name)
        return kept_names, removed_names

    def prepare_for_training(self):
        super().prepare_for_training()
        kept_names, removed_names = self._apply_scope_freeze()
        if self.lab_train_scope != "all":
            if not kept_names:
                raise RuntimeError(
                    f"NN3a lab scope {self.lab_train_scope!r} selected no parameters"
                )
            print(
                "[NN3a lab prepare] "
                f"scope={self.lab_train_scope} kept_tensors={len(kept_names)} "
                f"frozen_tensors={len(removed_names)}"
            )
        if self.lab_optimizer_recipe == "reference_value_only":
            value_names = []
            recipe_removed_names = []
            for name, parameter in self.unet.named_parameters():
                if not parameter.requires_grad:
                    continue
                if ".processor.ref_to_v." in name:
                    value_names.append(name)
                else:
                    parameter.requires_grad_(False)
                    recipe_removed_names.append(name)
            if not value_names:
                raise RuntimeError(
                    "Reference-value-only recipe selected no ref_to_v parameters"
                )
            print(
                "[NN3a lab prepare] recipe=reference_value_only "
                f"kept_tensors={len(value_names)} "
                f"frozen_tensors={len(recipe_removed_names)}"
            )

    def get_trainable_params(self, config):
        if self.lab_optimizer_recipe == "projection_split":
            return self._projection_split_groups(config)
        if self.lab_optimizer_recipe == "reference_value_only":
            return self._reference_value_only_groups(config)

        groups = super().get_trainable_params(config)
        if self.lab_train_scope == "all":
            print("[NN3a lab scope] all production trainable BA parameters retained")
            return groups

        name_by_id = {id(parameter): name for name, parameter in self.unet.named_parameters()}
        kept_groups = []
        kept_names = []

        for group in groups:
            kept = []
            for parameter in group["params"]:
                name = name_by_id.get(id(parameter), "")
                if self._scope_accepts(name):
                    kept.append(parameter)
                    kept_names.append(name)
            if kept:
                filtered = dict(group)
                filtered["params"] = kept
                filtered["name"] = f"{group.get('name', 'params')}__scope_{self.lab_train_scope}"
                kept_groups.append(filtered)

        if not kept_groups or not kept_names:
            raise RuntimeError(
                f"NN3a lab scope {self.lab_train_scope!r} selected no parameters"
            )

        kept_blocks = Counter(".".join(name.split(".")[:2]) for name in kept_names)
        print(
            "[NN3a lab scope] "
            f"scope={self.lab_train_scope} kept_tensors={len(kept_names)} "
            f"blocks={dict(sorted(kept_blocks.items()))}"
        )
        return kept_groups

    def _projection_split_groups(self, config):
        named = [
            (name, parameter)
            for name, parameter in self.unet.named_parameters()
            if parameter.requires_grad
        ]
        grouped = {}
        for name, parameter in named:
            if ".processor.ref_to_q." in name:
                family = "ref_q"
                scale = self.lab_ref_q_lr_scale
            elif (
                ".processor.ref_to_k." in name
                or ".processor.ref_to_v." in name
            ):
                family = "ref_kv"
                scale = self.lab_ref_kv_lr_scale
            elif ".processor.noise_to_" in name:
                family = "noise_qkv"
                scale = self.lab_noise_lr_scale
            else:
                raise RuntimeError(
                    "Projection-split optimizer found an unclassified "
                    f"trainable parameter: {name}"
                )

            block = "up0" if name.startswith("up_blocks.0.") else "other"
            if block == "up0":
                scale *= self.lab_up0_lr_scale
            group_name = f"{family}__{block}"
            grouped.setdefault((group_name, scale), []).append(parameter)

        if not grouped:
            raise RuntimeError("Projection-split optimizer selected no parameters")
        result = [
            {
                "params": parameters,
                "lr": float(config.lr_for_lora) * scale,
                "name": f"lab_{group_name}",
            }
            for (group_name, scale), parameters in sorted(grouped.items())
        ]
        print(
            "[NN3a lab optimizer] recipe=projection_split "
            + " ".join(
                f"{group['name']}:{len(group['params'])}@{group['lr']:.3g}"
                for group in result
            )
        )
        return result

    def _reference_value_only_groups(self, config):
        grouped = {}
        for name, parameter in self.unet.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".processor.ref_to_v." not in name:
                raise RuntimeError(
                    "Reference-value-only optimizer found an unexpected "
                    f"trainable parameter: {name}"
                )
            block = "up0" if name.startswith("up_blocks.0.") else "other"
            scale = self.lab_ref_v_lr_scale
            if block == "up0":
                scale *= self.lab_up0_lr_scale
            grouped.setdefault((block, scale), []).append(parameter)

        if not grouped:
            raise RuntimeError(
                "Reference-value-only optimizer selected no parameters"
            )
        result = [
            {
                "params": parameters,
                "lr": float(config.lr_for_lora) * scale,
                "name": f"lab_ref_v__{block}",
            }
            for (block, scale), parameters in sorted(grouped.items())
        ]
        print(
            "[NN3a lab optimizer] recipe=reference_value_only "
            + " ".join(
                f"{group['name']}:{len(group['params'])}@{group['lr']:.3g}"
                for group in result
            )
        )
        return result
