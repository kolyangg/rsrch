from types import SimpleNamespace

import pytest

from src.model.photomaker_branched.cl39x_contract import GROUPS, configure_cl39x


def parent():
    return SimpleNamespace(
        ba_hardcase_mode="temporal_frequency",
        ba_frequency_surface_loss_enabled=True,
        ba_null_key_router_enabled=True,
        ba_null_key_router_groups=GROUPS,
        pose_adapt_ratio=0.0,
        ca_mixing_for_face=False,
    )


def test_exactly_one_arm_and_exact_cl39_parent_are_enforced():
    model = parent()
    configure_cl39x(model, {
        "ba_valid_kv_enabled": True,
        "ba_valid_kv_groups": GROUPS,
    })
    assert model._cl39x_manifest["active_arm"] == "valid_kv"
    with pytest.raises(ValueError, match="Exactly one"):
        configure_cl39x(parent(), {
            "ba_valid_kv_enabled": True, "ba_valid_kv_groups": GROUPS,
            "ba_cycle_confidence_enabled": True,
            "ba_cycle_confidence_groups": GROUPS,
        })
    invalid = parent()
    invalid.pose_adapt_ratio = 0.5
    with pytest.raises(ValueError, match="exact CL39"):
        configure_cl39x(invalid, {
            "ba_valid_kv_enabled": True, "ba_valid_kv_groups": GROUPS,
        })
