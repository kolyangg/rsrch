"""Per-validation face-quality staging, scoring, and compact logging."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_MODELS = (
    "topiq_nr-face",
    "topiq_nr",
    "musiq",
    "maniqa-pipal",
)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).tolist()
    except ImportError:
        pass
    return [value]


def _expand(values: list[Any], count: int) -> list[Any]:
    if not values:
        return [None] * count
    if len(values) == count:
        return values
    if count % len(values) == 0:
        repeats = count // len(values)
        return [value for value in values for _ in range(repeats)]
    return [None] * count


def _safe_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value[:96] or "validation_image"


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class FaceQualityValidationSession:
    """Collect one validation partition and score it with the canonical tool."""

    def __init__(
        self,
        *,
        config: Any,
        checkpoint_dir: Path,
        writer: Any,
        logger: Any,
        part: str,
        step: int,
        partition_count: int,
    ) -> None:
        self.config = config
        self.writer = writer
        self.logger = logger
        self.part = str(part)
        self.step = int(step)
        self.partition_count = int(partition_count)
        self.project_root = Path(__file__).resolve().parents[2]
        self.output_dir = (
            Path(checkpoint_dir)
            / "face_quality"
            / self.part
            / f"step_{self.step:08d}"
        )
        self.input_dir = self.output_dir / "inputs"
        if self.input_dir.exists():
            shutil.rmtree(self.input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    @property
    def requested_device(self) -> str:
        return str(_config_get(self.config, "device", "auto")).lower()

    def resolve_device(self, num_processes: int) -> str:
        requested = self.requested_device
        if requested == "auto":
            return "cuda" if int(num_processes) == 1 else "cpu"
        if requested.startswith("cuda") and int(num_processes) > 1:
            raise ValueError(
                "trainer.face_quality.device=cuda is unsafe during rank-0-only "
                "DDP validation; use device=auto or cpu"
            )
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError(f"Unsupported face-quality device: {requested}")
        return requested

    def add_batch(self, batch: dict[str, Any], batch_idx: int) -> None:
        images = _flatten(batch.get("generated"))
        if not images:
            raise ValueError("Face-quality validation batch has no generated images")
        prompts = _expand(_flatten(batch.get("prompt")), len(images))
        identities = _expand(_flatten(batch.get("id")), len(images))
        seeds = _expand(_flatten(batch.get("seed")), len(images))

        for image_in_batch, image in enumerate(images):
            if not hasattr(image, "save"):
                raise TypeError(
                    "Face-quality validation requires generated PIL-like images"
                )
            sample_index = len(self.records)
            prompt = prompts[image_in_batch]
            identity = identities[image_in_batch]
            seed = seeds[image_in_batch]
            label = _safe_label(
                f"{identity or 'unknown'}__{str(prompt or '')[:48]}"
            )
            file_name = f"{sample_index:04d}__{label}.png"
            local_path = self.input_dir / file_name
            image.save(local_path)
            stable_key = (
                f"{self.part}|{self.step}|{sample_index}|"
                f"{identity}|{prompt}|{seed}"
            )
            self.records.append(
                {
                    "asset_id": hashlib.sha256(
                        stable_key.encode("utf-8")
                    ).hexdigest(),
                    "file_name": file_name,
                    "local_path": str(local_path.resolve()),
                    "file_size": local_path.stat().st_size,
                    "sha256": _sha256(local_path),
                    "sample_index": sample_index,
                    "validation_batch_idx": int(batch_idx),
                    "image_in_batch": int(image_in_batch),
                    "prompt": None if prompt is None else str(prompt),
                    "identity": None if identity is None else str(identity),
                    "seed": _json_scalar(seed),
                }
            )

    def finalize(self, *, num_processes: int) -> dict[str, float]:
        expected_images = _config_get(self.config, "expected_images", None)
        if expected_images not in (None, ""):
            expected_images = int(expected_images)
            if len(self.records) != expected_images:
                raise ValueError(
                    f"Face-quality validation collected {len(self.records)} "
                    f"images for {self.part}; expected {expected_images}"
                )
        if not self.records:
            raise ValueError("Face-quality validation collected no images")

        manifest_path = self.output_dir / "input_manifest.json"
        manifest = {
            "schema_version": 1,
            "kind": "training_validation_images",
            "experiment_key": getattr(self.writer, "run_id", None),
            "project_name": None,
            "steps": {str(self.step): self.records},
        }
        _atomic_json(manifest_path, manifest)

        results_json = self.output_dir / "face_quality_metrics.json"
        results_csv = self.output_dir / "face_quality_per_image.csv"
        scorer_script = Path(
            str(
                _config_get(
                    self.config,
                    "scorer_script",
                    "tools/inference/calculate_face_quality_metrics.py",
                )
            )
        )
        if not scorer_script.is_absolute():
            scorer_script = self.project_root / scorer_script
        scorer_python_value = str(
            _config_get(self.config, "scorer_python", "python")
        )
        scorer_python = (
            Path(scorer_python_value).absolute()
            if os.path.sep in scorer_python_value
            else Path(shutil.which(scorer_python_value) or scorer_python_value)
        )
        if not scorer_python.is_file():
            raise FileNotFoundError(
                "Face-quality scorer interpreter is unavailable: "
                f"{scorer_python_value}. Set FACE_QUALITY_SCORER_PYTHON or "
                "trainer.face_quality.scorer_python."
            )
        if not scorer_script.is_file():
            raise FileNotFoundError(scorer_script)

        metrics = [
            str(value)
            for value in _config_get(self.config, "models", DEFAULT_MODELS)
        ]
        device = self.resolve_device(num_processes)
        command = [
            str(scorer_python),
            str(scorer_script),
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(results_json),
            "--output-csv",
            str(results_csv),
            "--metrics",
            ",".join(metrics),
            "--device",
            device,
            "--batch-size",
            str(int(_config_get(self.config, "batch_size", 8))),
            "--crop-padding",
            str(float(_config_get(self.config, "crop_padding", 0.25))),
            "--crop-size",
            str(int(_config_get(self.config, "crop_size", 512))),
        ]
        # 27 Jul 2026 - AICODE-NOTE: The training pipeline calls the same
        # standalone scorer used by historical backfills. This keeps crop,
        # detector, model, and aggregate definitions exactly comparable.
        subprocess.run(command, cwd=self.project_root, check=True)

        result = json.loads(results_json.read_text(encoding="utf-8"))
        if result.get("metric_backend", {}).get("metrics") != metrics:
            raise ValueError("Face-quality scorer metric list drifted")
        if result.get("metric_backend", {}).get("pyiqa_version") != "0.1.15":
            raise ValueError("Face-quality scorer must use PyIQA 0.1.15")
        step_result = result.get("steps", {}).get(str(self.step))
        if not step_result or step_result.get("image_count") != len(self.records):
            raise ValueError("Face-quality scorer result is incomplete")
        metric_results = step_result["metrics"]
        image_count = float(step_result["image_count"])
        compact = {
            "face_detection_rate": float(step_result["face_detection_rate"]),
            "topiq_face_mean": float(metric_results["topiq_nr_face"]["mean"]),
            "topiq_face_p10": float(metric_results["topiq_nr_face"]["p10"]),
            "topiq_face_coverage": (
                float(metric_results["topiq_nr_face"]["count"]) / image_count
            ),
            "topiq_mean": float(metric_results["topiq_nr"]["mean"]),
            "musiq_mean": float(metric_results["musiq"]["mean"]),
            "maniqa_mean": float(metric_results["maniqa_pipal"]["mean"]),
        }

        namespace = str(_config_get(self.config, "namespace", "face_quality"))
        include_part = _config_get(
            self.config, "include_partition_in_metric_name", "auto"
        )
        if str(include_part).lower() == "auto":
            include_part = self.partition_count > 1
        prefix = f"{namespace}/{self.part}" if bool(include_part) else namespace
        for name, value in compact.items():
            self.writer.add_scalar(f"{prefix}/{name}", value)

        if bool(_config_get(self.config, "log_per_image_asset", True)):
            csv_sha256 = _sha256(results_csv)
            row_count = 0
            with results_csv.open(encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
            if row_count != len(self.records):
                raise ValueError(
                    f"Face-quality per-image CSV has {row_count} rows; "
                    f"expected {len(self.records)}"
                )
            asset_name = (
                f"face_quality_details__{self.part}__"
                f"step_{self.step:08d}.csv"
            )
            metadata = {
                "schema_version": 1,
                "kind": "face_quality_per_image_metrics",
                "namespace": "face_quality_details",
                "hidden_in_report_by_default": True,
                "validation_partition": self.part,
                "step": self.step,
                "row_count": row_count,
                "sha256": csv_sha256,
            }
            add_asset = getattr(self.writer, "add_asset", None)
            if callable(add_asset):
                add_asset(
                    asset_name,
                    results_csv,
                    metadata=metadata,
                    overwrite=True,
                )

        if not bool(_config_get(self.config, "keep_inputs", False)):
            shutil.rmtree(self.input_dir)
        if self.logger is not None:
            self.logger.info(
                "Face-quality validation complete: part=%s step=%s images=%s "
                "device=%s",
                self.part,
                self.step,
                len(self.records),
                device,
            )
        return compact
