#!/usr/bin/env python3
"""Run dependency-light tensor checks for the CL39-X01..X08 mechanisms."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _namespace(name: str, path: Path) -> None:
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_tests(relative_path: str) -> dict:
    return runpy.run_path(str(ROOT / relative_path))


def _check_contract() -> None:
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

    model = parent()
    configure_cl39x(model, {"ba_valid_kv_enabled": True, "ba_valid_kv_groups": GROUPS})
    assert model._cl39x_manifest["active_arm"] == "valid_kv"
    try:
        configure_cl39x(parent(), {
            "ba_valid_kv_enabled": True,
            "ba_valid_kv_groups": GROUPS,
            "ba_cycle_confidence_enabled": True,
            "ba_cycle_confidence_groups": GROUPS,
        })
    except ValueError as error:
        assert "Exactly one" in str(error)
    else:
        raise AssertionError("Multiple CL39-X arms were accepted")


def main() -> None:
    # Avoid importing the full PhotoMaker package: these checks exercise only
    # the isolated, dependency-light CL39-X modules and exact frame transform.
    _namespace("src", ROOT / "src")
    _namespace("src.model", ROOT / "src/model")
    _namespace("src.model.photomaker_branched", ROOT / "src/model/photomaker_branched")
    _namespace("src.datasets", ROOT / "src/datasets")

    _check_contract()
    for relative_path in (
        "tests/photomaker_branched/test_cl39x_extensions.py",
        "tests/photomaker_branched/test_counterfactual_reference.py",
        "tests/photomaker_branched/test_ownership_transforms.py",
    ):
        namespace = _load_tests(relative_path)
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                function()
    with tempfile.TemporaryDirectory(prefix="cl39x_ownership_") as directory:
        namespace = _load_tests("tests/photomaker_branched/test_ownership_maps.py")
        namespace["test_ownership_round_trip_and_soft_routing"](Path(directory))
        namespace["test_automask_policy_builds_normalized_subject_owned_maps"]()
    print("CL39X_CPU_PREFLIGHT_OK")


if __name__ == "__main__":
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("PyTorch is required for the CL39-X CPU preflight")
    main()
