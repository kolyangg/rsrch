#!/usr/bin/env python3
"""Verify exact ONNX/PyTorch parity and input gradients for E22 ArcFace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnxruntime as ort
import torch

from src.model.photomaker_branched.arcface_identity_aux import FrozenOnnxArcFace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-absolute-error", type=float, default=5.0e-4)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    args = parser.parse_args()

    torch.manual_seed(20260806)
    sample = torch.rand(1, 3, 112, 112, dtype=torch.float32) * 2.0 - 1.0
    executor = FrozenOnnxArcFace(
        args.model,
        expected_sha256=args.expected_sha256,
    ).eval()
    session = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
    )
    expected = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: sample.numpy()},
    )[0]
    with torch.no_grad():
        actual = executor(sample).cpu().numpy()
    max_error = float(np.max(np.abs(expected - actual)))
    cosine = float(
        np.dot(expected.ravel(), actual.ravel())
        / (np.linalg.norm(expected) * np.linalg.norm(actual))
    )
    if max_error > args.max_absolute_error or cosine < args.min_cosine:
        raise RuntimeError(
            f"ArcFace parity failed: max_error={max_error}, cosine={cosine}"
        )

    differentiable = sample.clone().requires_grad_(True)
    loss = executor(differentiable).float().square().mean()
    loss.backward()
    gradient = differentiable.grad
    if (
        gradient is None
        or not torch.isfinite(gradient).all()
        or float(gradient.norm().item()) <= 0.0
    ):
        raise RuntimeError("ArcFace PyTorch executor has invalid input gradients")

    print(
        json.dumps(
            {
                "status": "ok",
                "model": str(args.model.resolve()),
                "sha256": executor.model_sha256,
                "max_absolute_error": max_error,
                "cosine": cosine,
                "input_gradient_norm": float(gradient.norm().item()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
