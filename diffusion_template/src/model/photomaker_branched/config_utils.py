def branched_model_runtime_kwargs(config) -> dict:
    """Keep training and alternate-base validation model construction identical."""
    model_target = str(getattr(getattr(config, "model", {}), "_target_", ""))
    if not any(
        target in model_target
        for target in (
            "src.model.photomaker_branched.lora2.PhotomakerBranchedLora",
            "src.model.photomaker_branched.lora3.PhotomakerBranchedLora",
        )
    ):
        return {}
    return {
        "train_ba_only": bool(getattr(config, "train_ba_only", False)),
        "ba_train_top_k": float(getattr(config, "ba_train_top_k", 1.0)),
        "ba_patch_top_k": float(getattr(config, "ba_patch_top_k", 1.0)),
        "non_ba_train": bool(getattr(config, "non_ba_train", False)),
        "train_ba_all_steps": bool(getattr(config, "train_ba_all_steps", False)),
        "ba_weights_split": bool(getattr(config, "ba_weights_split", False)),
        "use_attn_v2": bool(getattr(config, "use_attn_v2", False)),
    }
