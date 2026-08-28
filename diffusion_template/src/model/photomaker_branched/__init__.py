"""Public PhotoMaker-BA exports, loaded only when requested."""

from importlib import import_module

__all__ = [
    "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken",
    "FacePerceiverResampler",
    "PhotomakerBranchedLora",
]

_EXPORTS = {
    "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken": (
        ".model_v2_NS", "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken"
    ),
    "FacePerceiverResampler": (".resampler", "FacePerceiverResampler"),
    "PhotomakerBranchedLora": (".lora2", "PhotomakerBranchedLora"),
}


def __getattr__(name):
    # 24 Aug 2026 - Ownership-cache imports run in data workers; keep them from
    # eagerly importing the model while preserving the package's public API.
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
