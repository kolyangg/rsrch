"""Training preparation, ownership, and checkpoint contract for clean_full."""

from __future__ import annotations

import copy
import hashlib
import time

import torch
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)

from .clean_full_model_helpers import (
    assert_branched_trainable_contract,
    branched_trainable_role_groups,
    install_branched_processors_for_training,
)


class CleanFullModelContractMixin:
    """Own the audited optimizer and schema-v2 checkpoint boundary."""
    def prepare_for_training(self):
        prepare_started = time.perf_counter()
        if (
            self.ba_architecture_version in {
                "residual_sa_v2",
                "anchored_mix_sa_v3",
                "query_adaptive_hard_sa_v4",
            }
            and self.ba_install_on_device
        ):
            # 2 Aug 2026 - AICODE-NOTE: Effective PEFT K/V materialization is
            # prohibitively slow in BF16 on CPU. Opt-in v2 runs stage only the
            # U-Net on its assigned GPU before dtype conversion and processor
            # installation; historical hard-replacement behavior is unchanged.
            self.unet.to(self.device)
            print(
                "[BA Init Timing] staged_unet_on_device "
                f"seconds={time.perf_counter() - prepare_started:.3f} "
                f"device={self.device}"
            )
        super().prepare_for_training()
        print(
            "[BA Init Timing] base_prepare_complete "
            f"seconds={time.perf_counter() - prepare_started:.3f}"
        )
        self.unet.requires_grad_(False)
        self.id_encoder.to(dtype=self.weight_dtype)
        self.id_encoder.requires_grad_(False)
        if self.identity_aux_recognizer is not None:
            self.identity_aux_recognizer.to(device=self.device, dtype=torch.float32)
            self.identity_aux_recognizer.requires_grad_(False)
            self.identity_aux_recognizer.eval()
        if self.ba_patch_identity_enabled:
            # 17 Aug 2026 - Reuse the exact frozen DINOv2 backend used by the
            # project metric; only decoded input pixels retain gradients.
            self.ba_patch_identity_encoder = torch.hub.load(
                "facebookresearch/dinov2", self.ba_patch_identity_backend
            ).to(device=self.device, dtype=torch.float32)
            self.ba_patch_identity_encoder.requires_grad_(False)
            self.ba_patch_identity_encoder.eval()

        adapter_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights=self.init_lora_weights,
            target_modules=self.lora_modules,
        )
        self.unet.add_adapter(adapter_lora_config, adapter_name="lora_adapter")
        self.unet.set_adapter(["lora_adapter", "default"])
        print(
            "[BA Init Timing] adapters_ready "
            f"seconds={time.perf_counter() - prepare_started:.3f}"
        )


        ##### BRANCHED ATTENTION - NEW BLOCK 1 #####
        """NEW BLOCK 1: pre-install branched attention processors before optimizer creation and mark their params trainable."""
        install_branched_processors_for_training(self)
        print(
            "[BA Init Timing] processors_ready "
            f"seconds={time.perf_counter() - prepare_started:.3f}"
        )

        ##### BRANCHED ATTENTION - NEW BLOCK 1 #####


    def load_photomaker_state_dict_(self, state_dict):
        # load lora
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="default")
        
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            assert not unexpected_keys, unexpected_keys

        # load id_encoder
        self.id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)

    def get_trainable_params(self, config):

        ##### BRANCHED ATTENTION - NEW BLOCK 2 #####
        """NEW BLOCK 2: optional custom optimizer grouping for branched processor parameters and BA-related LoRA params."""
        # ### TRAIN_BA_ONLY - CHECK ###
        # if getattr(self, "train_ba_only", False):
        #     # Train branched attention processors + LoRA weights on attention projections.
        #     proc_params = []
        #     lora_params = []
        #     for name, p in self.unet.named_parameters():
        #         if not p.requires_grad:
        #             continue
        #         if ".attn1.processor." in name or ".attn2.processor." in name:
        #             proc_params.append(p)
        #         elif "lora_A" in name or "lora_B" in name:
        #             lora_params.append(p)

        #     param_groups = []
        #     if proc_params:
        #         param_groups.append(
        #             {"params": proc_params, "lr": config.lr_for_lora, "name": "branched_processors"}
        #         )
        #     if lora_params:
        #         param_groups.append(
        #             {"params": lora_params, "lr": config.lr_for_lora, "name": "branched_lora"}
        #         )
        #     return param_groups
        # ### TRAIN_BA_ONLY - CHECK ###
        ##### BRANCHED ATTENTION - NEW BLOCK 2 #####

        if (
            self.ba_architecture_version == "hard_replace_v1"
            and self.train_ba_only
            and (
                self.generic_adapter_train_scope != "none"
                or self.photomaker_default_train_scope != "none"
            )
        ):
            # 5 Aug 2026 - AICODE-NOTE: E13+ jointly trains three explicitly
            # owned paths. Separate groups make the persistent PhotoMaker
            # default LR independently controllable without changing ownership.
            role_parameters = {
                "ba": [],
                "generic_adapter": [],
                "photomaker_default": [],
            }
            for name, parameter in self.unet.named_parameters():
                if not parameter.requires_grad:
                    continue
                if ".lora_adapter." in name:
                    role = "generic_adapter"
                elif ".default." in name:
                    role = "photomaker_default"
                elif (
                    ".attn1.processor." in name
                    or ".attn2.processor." in name
                    or name == "ba_frequency_shared_schedule_raw"
                ):
                    role = "ba"
                else:
                    raise RuntimeError(
                        f"Joint hard-v1 optimizer cannot classify {name!r}"
                    )
                role_parameters[role].append(parameter)
            lr_by_role = {
                "ba": float(getattr(config, "ba_lr", config.lr_for_lora)),
                "generic_adapter": float(
                    getattr(config, "generic_adapter_lr", config.lr_for_lora)
                ),
                "photomaker_default": float(
                    getattr(config, "photomaker_default_lr", config.lr_for_lora)
                ),
            }
            groups = [
                {
                    "params": role_parameters[role],
                    "lr": lr_by_role[role],
                    "name": role,
                }
                for role in ("ba", "generic_adapter", "photomaker_default")
                if role_parameters[role]
            ]
            if not groups:
                raise RuntimeError("Joint hard-v1 optimizer resolved no parameters")
            return groups

        if self.ba_architecture_version in {
            "residual_sa_v2",
            "anchored_mix_sa_v3",
            "query_adaptive_hard_sa_v4",
        } and self.train_ba_only:
            role_groups = branched_trainable_role_groups(self)
            lr_by_role = {
                "ref_kv": float(getattr(config, "ba_ref_kv_lr", config.lr_for_lora)),
                "ref_output": float(
                    getattr(config, "ba_ref_output_lr", config.lr_for_lora)
                ),
                "gate": float(getattr(config, "ba_gate_lr", config.lr_for_lora)),
                "mix": float(getattr(config, "ba_mix_lr", config.lr_for_lora)),
                "ref_query": float(
                    getattr(config, "ba_ref_query_lr", config.lr_for_lora)
                ),
            }
            unknown_roles = set(role_groups) - set(lr_by_role)
            if unknown_roles:
                raise RuntimeError(
                    f"Unknown {self.ba_architecture_version} optimizer roles: "
                    f"{unknown_roles}"
                )
            if self.ba_architecture_version == "residual_sa_v2":
                role_order = ("ref_kv", "ref_output", "gate")
            elif self.ba_architecture_version == "anchored_mix_sa_v3":
                role_order = ("ref_kv", "ref_output", "mix")
            else:
                role_order = ("ref_query", "ref_kv", "ref_output")
            return [
                {
                    "params": role_groups[role],
                    "lr": lr_by_role[role],
                    "name": f"ba_{role}",
                }
                for role in role_order
                if role_groups.get(role)
            ]

        # Default behavior: train all UNet parameters with requires_grad=True (LoRA + processors).
        lora_params = filter(lambda p: p.requires_grad, self.unet.parameters())
        trainable_params = [
            {"params": lora_params, "lr": config.lr_for_lora, "name": "lora_params"},
        ]
        return trainable_params

    def assert_trainable_contract(self, optimizer=None) -> dict:
        if not self.strict_trainable_contract:
            return {}
        return assert_branched_trainable_contract(self, optimizer=optimizer)

    def _branched_architecture_manifest(self) -> dict:
        named_parameters = dict(self.unet.named_parameters())
        trainable_names = tuple(
            sorted(
                name for name, parameter in named_parameters.items()
                if parameter.requires_grad
            )
        )
        processor_names = list(getattr(self, "_ba_patched_processor_names", ()))
        semantic_names = list(getattr(self, "_ba_semantic_processor_names", ()))
        semantic_names_sha256 = hashlib.sha256(
            "\n".join(semantic_names).encode("utf-8")
        ).hexdigest()
        hard_v1_extended = bool(
            self.ba_architecture_version == "hard_replace_v1"
            and (
                self.ba_hard_v1_true_reference_key_mask
                or self.ba_hard_v1_branch_output_rank is not None
                or self.ba_hard_v1_reference_roi_warp
                or self.ba_hard_v1_lora_rank is not None
                or self.ba_identity_ca_v2_enabled
                or self.ba_residual_identity_ca_v3_enabled
                or self.ba_hardcase_mode != "off"
                or self.ba_crossview_consistency_enabled
            )
        )
        processor_code_version = {
            "hard_replace_v1": (
                4
                if self.ba_residual_identity_ca_v3_enabled
                else (
                    3
                    if self.ba_identity_ca_v2_enabled
                    else (2 if hard_v1_extended else 1)
                )
            ),
            "residual_sa_v2": 2,
            "anchored_mix_sa_v3": 3,
            "query_adaptive_hard_sa_v4": 4,
        }[self.ba_architecture_version]
        manifest = {
            "format": "photomaker_branched_trainable_unet_v2",
            "ba_architecture_version": self.ba_architecture_version,
            "processor_code_version": processor_code_version,
            "branched_attn_lora_rank": int(self.branched_attn_lora_rank),
            "branched_attn_weight_mode": self.branched_attn_weight_mode,
            "branched_attn_new_weight_kind": self.branched_attn_new_weight_kind,
            "train_ba_only": bool(self.train_ba_only),
            "train_branched_ca_lora": bool(self.train_branched_ca_lora),
            "ba_patch_top_k": float(self.ba_patch_top_k),
            "ba_train_top_k": float(self.ba_train_top_k),
            "non_ba_train": bool(self.non_ba_train),
            "disable_branched_sa": bool(getattr(self, "disable_branched_sa", False)),
            "disable_branched_ca": bool(getattr(self, "disable_branched_ca", False)),
            "branched_trainable_dtype": self.branched_trainable_dtype,
            "ba_ref_kv_rank": int(self.ba_ref_kv_rank),
            "ba_output_rank": int(self.ba_output_rank),
            "ba_branch_q_rank": int(self.ba_branch_q_rank),
            "ba_face_fusion_mode": self.ba_face_fusion_mode,
            "ba_face_branch_scale": float(self.ba_face_branch_scale),
            "ba_gate_init": float(self.ba_gate_init),
            "ba_gate_max": float(self.ba_gate_max),
            "ba_gate_timestep": bool(self.ba_gate_timestep),
            "ba_gate_face_area": bool(self.ba_gate_face_area),
            "ba_require_denoise_progress": bool(self.ba_require_denoise_progress),
            "ba_training_timestep_policy": self.ba_training_timestep_policy,
            "ba_spatial_reference_shuffle_probability": float(
                self.ba_spatial_reference_shuffle_probability
            ),
            "ba_install_on_device": bool(self.ba_install_on_device),
            "ba_self_attention_groups": (
                None
                if self.ba_self_attention_groups is None
                else list(self.ba_self_attention_groups)
            ),
            "semantic_processor_names": semantic_names,
            "semantic_processor_names_sha256": semantic_names_sha256,
            "merge_kind": (
                "reference_residual"
                if self.ba_architecture_version == "residual_sa_v2"
                else (
                    "anchored_reference_interpolation"
                    if self.ba_architecture_version == "anchored_mix_sa_v3"
                    else (
                        "query_adaptive_hard_reference_replacement"
                        if self.ba_architecture_version
                        == "query_adaptive_hard_sa_v4"
                        else "hard_face_replacement"
                    )
                )
            ),
            "target_query_source": (
                "frozen_target"
                if self.ba_architecture_version in {
                    "residual_sa_v2",
                    "anchored_mix_sa_v3",
                }
                else (
                    "branch_adapted_target"
                    if self.ba_architecture_version
                    == "query_adaptive_hard_sa_v4"
                    else "configured_noise_projection"
                )
            ),
            "reference_key_mask": (
                self.ba_hard_v1_true_reference_key_mask
                if self.ba_architecture_version == "hard_replace_v1"
                else self.ba_architecture_version in {
                    "residual_sa_v2",
                    "anchored_mix_sa_v3",
                    "query_adaptive_hard_sa_v4",
                }
            ),
            "strict_face_routing": bool(getattr(self, "strict_face_routing", False)),
            "pose_adapt_ratio": float(self.pose_adapt_ratio),
            "ca_mixing_for_face": bool(self.ca_mixing_for_face),
            "photomaker_start_step": int(self.photomaker_start_step),
            "branched_attn_start_step": int(self.branched_attn_start_step),
            "num_inference_steps": int(self.num_inference_steps),
            "patched_processor_names": processor_names,
            "trainable_processor_names": list(
                getattr(self, "_ba_trainable_processor_names", ())
            ),
            "trainable_names": list(trainable_names),
            "trainable_shapes": {
                name: list(named_parameters[name].shape) for name in trainable_names
            },
            "trainable_dtypes": {
                name: str(named_parameters[name].dtype).replace("torch.", "")
                for name in trainable_names
            },
        }
        if (
            self.generic_adapter_train_scope != "none"
            or self.photomaker_default_train_scope != "none"
        ):
            # 4 Aug 2026 - AICODE-NOTE: Only new explicit adapter-scope runs
            # extend the manifest. Defaults-off manifests stay byte-compatible
            # with existing E0-E6 schema-v2 checkpoints.
            manifest["generic_adapter_train_scope"] = (
                self.generic_adapter_train_scope
            )
            manifest["photomaker_default_train_scope"] = (
                self.photomaker_default_train_scope
            )
        if hard_v1_extended:
            hard_v1_extensions = {
                "true_reference_key_mask": bool(
                    self.ba_hard_v1_true_reference_key_mask
                ),
                "branch_output_rank": self.ba_hard_v1_branch_output_rank,
                "reference_roi_warp": bool(
                    self.ba_hard_v1_reference_roi_warp
                ),
                "face_fusion_mode": "hard_reference_replace",
            }
            if self.ba_hard_v1_lora_rank is not None:
                hard_v1_extensions["lora_rank"] = int(
                    self.ba_hard_v1_lora_rank
                )
            if self.ba_identity_ca_v2_enabled:
                identity_names = list(
                    getattr(self, "_ba_identity_ca_processor_names", ())
                )
                hard_v1_extensions["identity_ca_v2"] = {
                    "enabled": True,
                    "groups": list(self.ba_identity_ca_v2_groups or ()),
                    "rank": int(self.ba_identity_ca_v2_rank),
                    "processor_names": identity_names,
                    "routing": "target_q_active_photomaker_id_kv",
                    "merge": "native_outside_face_id_only_inside_face",
                    "legacy_branched_ca": False,
                }
            if self.ba_residual_identity_ca_v3_enabled:
                identity_names = list(
                    getattr(self, "_ba_identity_ca_processor_names", ())
                )
                hard_v1_extensions["residual_identity_ca_v3"] = {
                    "enabled": True,
                    "groups": list(
                        self.ba_residual_identity_ca_v3_groups or ()
                    ),
                    "rank": int(self.ba_residual_identity_ca_v3_rank),
                    "gate_init": float(
                        self.ba_residual_identity_ca_v3_gate_init
                    ),
                    "gate_max": float(
                        self.ba_residual_identity_ca_v3_gate_max
                    ),
                    "processor_names": identity_names,
                    "routing": "target_q_active_photomaker_id_kv",
                    "merge": "native_plus_face_mask_times_bounded_gate_times_rms_id_delta",
                    "delta_output_zero_init": True,
                    "legacy_branched_ca": False,
                }
            if self.ba_hardcase_mode != "off":
                hardcase_route = {
                    "mode": self.ba_hardcase_mode,
                    "groups": list(self.ba_hardcase_groups or ()),
                    "rank": int(self.ba_hardcase_rank),
                    "gate_max": float(self.ba_hardcase_gate_max),
                    "roi_size": int(self.ba_hardcase_roi_size),
                    "face_threshold_px": int(
                        self.ba_hardcase_face_threshold_px
                    ),
                    "transition_cells": int(
                        self.ba_hardcase_transition_cells
                    ),
                    "ownership_hidden_dim": int(
                        self.ba_hardcase_ownership_hidden_dim
                    ),
                    "visible_face_floor": float(
                        self.ba_hardcase_visible_face_floor
                    ),
                    "semantic_loss_weight": float(
                        self.ba_semantic_ownership_loss_weight
                    ),
                }
                if self.ba_hardcase_fallback_mode != "off":
                    hardcase_route["fallback_mode"] = self.ba_hardcase_fallback_mode
                if self.ba_hardcase_mode == "visibility_order":
                    hardcase_route["top_native_floor"] = float(
                        self.ba_hardcase_top_native_floor
                    )
                if self.ba_hardcase_mode == "temporal_frequency":
                    hardcase_route["frequency_scales"] = [
                        float(self.ba_hardcase_frequency_low_early),
                        float(self.ba_hardcase_frequency_low_late),
                        float(self.ba_hardcase_frequency_high_early),
                        float(self.ba_hardcase_frequency_high_late),
                    ]
                    if self.ba_frequency_surface_loss_enabled:
                        hardcase_route["frequency_surface_loss"] = {
                            "groups": list(self.ba_frequency_surface_loss_groups),
                            "top_weight": self.ba_frequency_surface_top_weight,
                            "top_low_band_factor": (
                                self.ba_frequency_surface_top_low_band_factor
                            ),
                            "visible_floor_weight": (
                                self.ba_frequency_surface_visible_floor_weight
                            ),
                            "visible_floor_ratio": (
                                self.ba_frequency_surface_visible_floor_ratio
                            ),
                        }
                    if self.ba_frequency_learnable_schedule_enabled:
                        hardcase_route["learnable_frequency_schedule"] = {
                            "low_early_fixed": self.ba_hardcase_frequency_low_early,
                            "low_late": [
                                self.ba_frequency_low_late_center,
                                self.ba_frequency_low_late_half_range,
                            ],
                            "high_early": [
                                self.ba_frequency_high_early_center,
                                self.ba_frequency_high_early_half_range,
                            ],
                            "high_late": [
                                self.ba_frequency_high_late_center,
                                self.ba_frequency_high_late_half_range,
                            ],
                            "anchor_weight": self.ba_frequency_schedule_anchor_weight,
                        }
                    if self.ba_frequency_lowband_contrastive_enabled:
                        hardcase_route["lowband_contrastive"] = {
                            "groups": list(
                                self.ba_frequency_lowband_contrastive_groups
                            ),
                            "probability": (
                                self.ba_frequency_lowband_contrastive_probability
                            ),
                            "weight": self.ba_frequency_lowband_contrastive_weight,
                            "temperature": (
                                self.ba_frequency_lowband_contrastive_temperature
                            ),
                            "ramp_steps": [
                                self.ba_frequency_lowband_contrastive_ramp_start_step,
                                self.ba_frequency_lowband_contrastive_ramp_end_step,
                            ],
                            "negative": "in_batch_different_identity",
                            "target_query_detached": True,
                        }
                    cl38_cl44 = {
                        "visibility_ownership_v2": self.ba_visibility_ownership_v2_enabled,
                        "null_key_router": self.ba_null_key_router_enabled,
                        "identity_motion_projector": self.ba_identity_motion_projector_enabled,
                        "landmark_canonical_kv": self.ba_landmark_canonical_kv_enabled,
                        "component_token_memory": self.ba_component_token_memory_enabled,
                        "id_adaptive_modulation": self.ba_id_adaptive_modulation_enabled,
                        "semantic_window_gate": self.ba_semantic_window_gate_enabled,
                    }
                    active_cl38_cl44 = [
                        name for name, enabled in cl38_cl44.items() if enabled
                    ]
                    if active_cl38_cl44:
                        name = active_cl38_cl44[0]
                        hardcase_route["cl38_cl44_extension"] = {
                            "name": name,
                            "groups": list(getattr(self, f"ba_{name}_groups")),
                        }
                if self.ba_hardcase_mode == "anchored_roi":
                    hardcase_route["roi_gate"] = [
                        float(self.ba_hardcase_roi_gate_min),
                        float(self.ba_hardcase_roi_gate_init),
                        float(self.ba_hardcase_gate_max),
                    ]
                    hardcase_route["roi_progress_min"] = float(
                        self.ba_hardcase_roi_progress_min
                    )
                    hardcase_route["roi_rms_cap"] = float(
                        self.ba_hardcase_roi_rms_cap
                    )
                hard_v1_extensions["hardcase_route"] = hardcase_route
            if self.ba_crossview_consistency_enabled:
                hard_v1_extensions["crossview_consistency"] = {
                    "probability": float(
                        self.ba_crossview_consistency_probability
                    ),
                    "weight": float(self.ba_crossview_consistency_weight),
                    "teacher_stop_gradient": True,
                }
            manifest["hard_v1_extensions"] = hard_v1_extensions
        if self.identity_aux_enabled:
            if self.identity_aux_backend == "photomaker_clip_v1":
                # Preserve E16's schema-v2 manifest byte-for-byte.
                manifest["identity_auxiliary"] = {
                    "kind": "predicted_x0_photomaker_clip_cosine",
                    "cadence": self.identity_aux_cadence,
                    "max_timestep": self.identity_aux_max_timestep,
                    "ramp_steps": [
                        self.identity_aux_ramp_start_step,
                        self.identity_aux_ramp_end_step,
                    ],
                    "max_weight": self.identity_aux_max_weight,
                    "crop_padding": self.identity_aux_crop_padding,
                }
            else:
                manifest["identity_auxiliary"] = {
                    "kind": "predicted_x0_arcface_buffalo_l_cosine_v2",
                    "backend": self.identity_aux_backend,
                    "model_sha256": self.identity_aux_model_sha256,
                    "preprocessing": "rgb_minus_127.5_over_127.5_roi112",
                    "target": "normalized_target_plus_distinct_ref_centroid",
                    "cadence": self.identity_aux_cadence,
                    "max_timestep": self.identity_aux_max_timestep,
                    "ramp_steps": [
                        self.identity_aux_ramp_start_step,
                        self.identity_aux_ramp_end_step,
                    ],
                    "max_weight": self.identity_aux_max_weight,
                    "crop_padding": self.identity_aux_crop_padding,
                    "dynamic_weight": self.identity_aux_dynamic_weight,
                    "grad_target_ratio": self.identity_aux_grad_target_ratio,
                    "grad_norm_interval": self.identity_aux_grad_norm_interval,
                }
        if self.ba_pm_boundary_distill_enabled:
            manifest["pm_boundary_distillation"] = {
                "probability": self.ba_pm_boundary_distill_probability,
                "boundary_weight": self.ba_pm_boundary_distill_weight,
                "top_weight": self.ba_pm_boundary_distill_top_weight,
                "width": self.ba_pm_boundary_distill_width,
                "teacher": "frozen_step0_native_photomaker",
            }
        if self.ba_low_noise_id_reward_enabled:
            manifest["low_noise_id_reward"] = {
                "last_ddim_steps": self.ba_low_noise_id_reward_last_steps,
                "trajectory_anchor_weight": self.ba_low_noise_id_reward_kl_weight,
                "source": "frozen_loaded_cl19",
            }
        if self.ba_architecture_version == "anchored_mix_sa_v3":
            manifest.update(
                {
                    "routing": "target_q_reference_kv_true_key_mask",
                    "merge_equation": (
                        "native_plus_target_mask_times_alpha_times_"
                        "reference_minus_native"
                    ),
                    "reference_output_base": "frozen_native_to_out",
                    "ba_mix_init": float(self.ba_mix_init),
                    "ba_mix_floor": float(self.ba_mix_floor),
                    "ba_mix_max": float(self.ba_mix_max),
                    "ba_mix_timestep": bool(self.ba_mix_timestep),
                    "ba_mix_face_area": bool(self.ba_mix_face_area),
                    "ba_reference_rms_match": bool(
                        self.ba_reference_rms_match
                    ),
                    "ba_reference_rms_clip": [
                        float(self.ba_reference_rms_clip_min),
                        float(self.ba_reference_rms_clip_max),
                    ],
                    "ba_telemetry_enabled": bool(self.ba_telemetry_enabled),
                    "ba_telemetry_interval": int(self.ba_telemetry_interval),
                    "ba_reference_loss_mode": self.ba_reference_loss_mode,
                }
            )
        elif self.ba_architecture_version == "query_adaptive_hard_sa_v4":
            manifest.update(
                {
                    "routing": "branch_target_q_reference_kv_true_key_mask",
                    "merge_equation": (
                        "native_outside_target_mask_reference_inside_target_mask"
                    ),
                    "face_fusion_mode": "hard_reference_replace",
                    "face_branch_scale": 1.0,
                    "reference_output_base": "frozen_native_to_out",
                    "ba_reference_loss_mode": self.ba_reference_loss_mode,
                    "ba_telemetry_enabled": bool(self.ba_telemetry_enabled),
                    "ba_telemetry_interval": int(self.ba_telemetry_interval),
                }
            )
        return manifest

    def _get_trainable_state_dict_v2(self) -> dict:
        self.assert_trainable_contract()
        named_parameters = dict(self.unet.named_parameters())
        trainable_names = tuple(
            sorted(
                name for name, parameter in named_parameters.items()
                if parameter.requires_grad
            )
        )
        if not trainable_names:
            raise RuntimeError("Refusing to save an empty trainable U-Net state")
        return {
            "schema_version": 2,
            "state_format": "trainable_unet_v2",
            "architecture": self._branched_architecture_manifest(),
            "trainable_unet": {
                name: named_parameters[name].detach().cpu().clone()
                for name in trainable_names
            },
        }

    def get_state_dict(self):
        if self.branched_state_dict_mode == "trainable_v2":
            # 1 Aug 2026 - Save the exact requires-grad allowlist so checkpoint
            # fidelity cannot depend on a hand-maintained adapter-name subset.
            return self._get_trainable_state_dict_v2()

        lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="lora_adapter"))
        state = {
            'lora_weights': lora_weights,
        }
        if hasattr(self.unet, "attn_processors"):
            proc_sd = {}
            patched_proc_names = set(getattr(self, "_ba_patched_processor_names", ()))
            for name, proc in self.unet.attn_processors.items():
                if patched_proc_names and name not in patched_proc_names:
                    continue
                if not isinstance(proc, torch.nn.Module):
                    continue
                trainable = tuple(n for n, p in proc.named_parameters() if p.requires_grad)
                if not trainable:
                    continue
                full_sd = proc.state_dict()
                sd = {
                    k: v for k, v in full_sd.items()
                    if any(k == n or k.startswith(n + ".") for n in trainable)
                }
                if sd:
                    proc_sd[name] = sd
            if proc_sd:
                state["attn_processors"] = proc_sd
        return state

    def _load_trainable_state_dict_v2(self, state_dict: dict) -> None:
        if state_dict.get("state_format") != "trainable_unet_v2":
            raise RuntimeError(
                f"Unknown schema-v2 state format: {state_dict.get('state_format')!r}"
            )
        saved_manifest = state_dict.get("architecture")
        current_manifest = self._branched_architecture_manifest()
        manifests_match = saved_manifest == current_manifest
        if (
            not manifests_match
            and self.ba_allow_objective_only_checkpoint_init
            and isinstance(saved_manifest, dict)
        ):
            # 13 Aug 2026 - CL25 is a weights-only continuation from CL19.
            # Permit only training-objective metadata to differ; routing,
            # tensor names/shapes, validation semantics, and ownership remain
            # protected by the complete residual manifest comparison below.
            saved_comparable = dict(saved_manifest)
            current_comparable = dict(current_manifest)
            for key in ("identity_auxiliary", "low_noise_id_reward"):
                saved_comparable.pop(key, None)
                current_comparable.pop(key, None)
            manifests_match = saved_comparable == current_comparable
        if not manifests_match:
            raise RuntimeError(
                "Branched checkpoint architecture mismatch: "
                f"saved={saved_manifest!r}, current={current_manifest!r}"
            )

        received = state_dict.get("trainable_unet")
        if not isinstance(received, dict):
            raise RuntimeError("Schema-v2 checkpoint is missing trainable_unet")
        named_parameters = dict(self.unet.named_parameters())
        expected = set(current_manifest["trainable_names"])
        received_names = set(received)
        if expected != received_names:
            raise RuntimeError(
                "Schema-v2 trainable state mismatch: "
                f"missing={sorted(expected - received_names)}, "
                f"unexpected={sorted(received_names - expected)}"
            )

        with torch.no_grad():
            for name in sorted(expected):
                value = received[name]
                parameter = named_parameters[name]
                if not torch.is_tensor(value):
                    raise TypeError(f"Checkpoint value for {name} is not a tensor")
                if tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {name}: "
                        f"saved={tuple(value.shape)}, current={tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        if self.ba_low_noise_id_reward_enabled:
            frozen = {
                name: copy.deepcopy(processor)
                for name, processor in self.unet.attn_processors.items()
            }
            for processor in frozen.values():
                if isinstance(processor, torch.nn.Module):
                    processor.requires_grad_(False)
                    processor.eval()
            self._ba_frozen_teacher_unet = frozen
        self.assert_trainable_contract()

    def load_state_dict_(self, state_dict):
        if int(state_dict.get("schema_version", 1)) == 2:
            self._load_trainable_state_dict_v2(state_dict)
            return

        # Historical schema-v1 checkpoints retain their original loader even
        # when the current run writes schema v2.
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="lora_adapter")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            # In newer peft versions this is an empty list when there are no unexpected keys
            assert not unexpected_keys, unexpected_keys
        for name, sd in state_dict.get("attn_processors", {}).items():
            proc = self.unet.attn_processors.get(name)
            if proc is not None and hasattr(proc, "load_state_dict"):
                proc.load_state_dict(sd, strict=False)
