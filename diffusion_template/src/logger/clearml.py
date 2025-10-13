from datetime import datetime
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd


class ClearMLWriter:
    """
    ClearML-based experiment tracker that mirrors the WandB writer API.

    The trainer code assumes writers expose the same add_* methods; this
    implementation routes them to ClearML's Task logger.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        tags=None,
        output_uri=None,
        **kwargs,
    ):
        self.logger = logger
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()
        self.run_id = run_id
        self._task = None
        self._clearml_logger = None
        self._resource_monitor = None
        self._task_name = run_name or "training"

        try:
            from clearml import Task
        except ImportError:
            if self.logger is not None:
                self.logger.warning("For use clearml install it via \n\t pip install clearml")
            return

        resource_monitor_cls = None
        try:
            from clearml.utilities.resource_monitor import ResourceMonitor

            resource_monitor_cls = ResourceMonitor
        except ImportError:
            resource_monitor_cls = None

        init_kwargs = {}
        collected_tags = list(tags or [])
        if entity:
            collected_tags.append(str(entity))
        if collected_tags:
            init_kwargs["tags"] = collected_tags
        if output_uri is not None:
            init_kwargs["output_uri"] = output_uri

        if mode == "offline":
            os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")

        if run_id:
            init_kwargs["reuse_last_task_id"] = run_id

        try:
            self._task = Task.init(project_name=project_name, task_name=self._task_name, **init_kwargs)
        except TypeError:
            init_kwargs.pop("reuse_last_task_id", None)
            self._task = Task.init(project_name=project_name, task_name=self._task_name, **init_kwargs)

        self.run_id = self._task.id
        self._clearml_logger = self._task.get_logger()

        if resource_monitor_cls is not None:
            try:
                self._resource_monitor = resource_monitor_cls(task=self._task)
            except Exception as error:
                if self.logger is not None:
                    self.logger.warning(f"Failed to start ClearML resource monitor: {error}")

        if project_config is not None:
            try:
                self._task.connect(project_config)
            except Exception as error:
                if self.logger is not None:
                    self.logger.warning(f"Failed to sync config with ClearML: {error}")

    def set_step(self, step, mode="train"):
        """
        Update current step and partition mode.
        """
        self.mode = mode
        previous_step = self.step
        self.step = int(step)
        if self.step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            seconds = duration.total_seconds() or 1e-8
            self.add_scalar("general/steps_per_sec", (self.step - previous_step) / seconds)
            self.timer = datetime.now()

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar value to ClearML.
        """
        if self._clearml_logger is None:
            return
        title, series = self._scalar_title_series(scalar_name)
        value = self._to_number(scalar)
        iteration = int(self.step)
        try:
            self._clearml_logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=iteration,
                report_freq=1,
            )
        except TypeError:
            try:
                self._clearml_logger.report_scalar(title, series, value, iteration)
            except Exception as error:
                self._log_warning("report_scalar", error)
        else:
            if self.logger is not None and not getattr(self, "_scalar_debug_logged", False):
                self.logger.debug(f"[ClearML] Logged scalar {title}/{series}={value} @ {iteration}")
                self._scalar_debug_logged = True
            try:
                self._clearml_logger.flush()
            except Exception:
                pass
        try:
            name = f"{title}/{series}".strip("/")
            self._clearml_logger.report_single_value(name or title, value)
        except Exception:
            pass

    def add_scalars(self, scalars):
        """
        Log multiple scalars at once.
        """
        for scalar_name, scalar_value in scalars.items():
            self.add_scalar(scalar_name, scalar_value)

    def add_image(self, image_name, image):
        """
        Log a single image.
        """
        if self._clearml_logger is None:
            return
        title, series = self._image_title_series(image_name)
        prepared = self._prepare_image(image)
        if prepared is None:
            return
        iteration = int(self.step)
        try:
            self._clearml_logger.report_image(
                title=title,
                series=series,
                iteration=iteration,
                image=prepared,
            )
        except TypeError:
            try:
                self._clearml_logger.report_image(title, series, iteration, prepared)
            except Exception as error:
                self._log_warning("report_image", error)
        else:
            if self.logger is not None and not getattr(self, "_image_debug_logged", False):
                self.logger.debug(f"[ClearML] Logged image {title}/{series} @ {iteration}")
                self._image_debug_logged = True
            try:
                self._clearml_logger.flush()
            except Exception:
                pass

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log audio data if supported. Falls back to uploading the raw array.
        """
        if self._clearml_logger is None or self._task is None or audio is None:
            return
        title, series = self._scalar_title_series(audio_name)
        audio_np = self._to_numpy(audio)

        report_audio = getattr(self._clearml_logger, "report_audio", None)
        if callable(report_audio):
            try:
                report_audio(
                    title=title,
                    series=series,
                    iteration=int(self.step),
                    audio=audio_np,
                    sampling_rate=sample_rate,
                )
                return
            except TypeError:
                try:
                    report_audio(title, series, int(self.step), audio_np, sample_rate)
                    return
                except Exception as error:
                    self._log_warning("report_audio", error)
        try:
            artifact_name = f"{title}/{series}".strip("/")
            metadata = {"sample_rate": sample_rate} if sample_rate is not None else None
            self._task.upload_artifact(artifact_name or "audio", artifact_object=audio_np, metadata=metadata)
        except Exception as error:
            self._log_warning("upload_artifact", error)

    def add_text(self, text_name, text):
        """
        Log text output.
        """
        if self._clearml_logger is None:
            return
        title, series = self._scalar_title_series(text_name)
        report_text = getattr(self._clearml_logger, "report_text", None)
        if not callable(report_text):
            return
        try:
            report_text(
                title=title,
                series=series,
                iteration=int(self.step),
                text=str(text),
            )
        except TypeError:
            try:
                report_text(title, series, str(text))
            except TypeError:
                try:
                    report_text(f"{title}/{series}".strip("/"), str(text))
                except Exception as error:
                    self._log_warning("report_text", error)
        except Exception as error:
            self._log_warning("report_text", error)

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram data.
        """
        if self._clearml_logger is None or values_for_hist is None:
            return
        title, series = self._scalar_title_series(hist_name)
        data = self._to_numpy(values_for_hist)

        n_bins = None
        if isinstance(bins, int):
            n_bins = bins
        elif isinstance(bins, str):
            n_bins = 512

        report_histogram = getattr(self._clearml_logger, "report_histogram", None)
        if not callable(report_histogram):
            return

        try:
            if n_bins is not None:
                report_histogram(
                    title=title,
                    series=series,
                    iteration=int(self.step),
                    values=data,
                    n_bins=n_bins,
                )
            else:
                report_histogram(title=title, series=series, iteration=int(self.step), values=data)
        except TypeError:
            try:
                if n_bins is not None:
                    report_histogram(title, series, int(self.step), data, n_bins=n_bins)
                else:
                    report_histogram(title, series, int(self.step), data)
            except Exception as error:
                self._log_warning("report_histogram", error)
        except Exception as error:
            self._log_warning("report_histogram", error)

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log tabular data.
        """
        if self._clearml_logger is None or table is None:
            return
        title, series = self._scalar_title_series(table_name)
        report_table = getattr(self._clearml_logger, "report_table", None)
        if callable(report_table):
            try:
                report_table(
                    title=title,
                    series=series,
                    iteration=int(self.step),
                    table_plot=table,
                )
                return
            except TypeError:
                try:
                    report_table(title, series, int(self.step), table)
                    return
                except Exception as error:
                    self._log_warning("report_table", error)
        if self._task is not None:
            try:
                self._task.upload_artifact(f"{title}/{series}".strip("/"), artifact_object=table.to_dict())
            except Exception as error:
                self._log_warning("upload_artifact", error)

    def add_images(self, images_name, images):
        """
        Log multiple images.
        """
        if not images:
            return
        for idx, image in enumerate(images):
            name = f"{images_name}/{idx}"
            self.add_image(name, image)

    def _scalar_title_series(self, name: str) -> Tuple[str, str]:
        """
        Map scalar names to ClearML (title, series) following reference examples.
        """
        mode = self.mode or "general"
        if not name:
            return (mode, "value")
        parts = [p for p in str(name).split("/") if p]
        if len(parts) >= 2:
            return (parts[0], "/".join(parts[1:]))
        return (mode, parts[0])

    def _image_title_series(self, name: str) -> Tuple[str, str]:
        """
        Map image names to ClearML (title, series) following reference examples.
        """
        if not name:
            return ("images", self.mode or "general")
        parts = [p for p in str(name).split("/") if p]
        if len(parts) >= 2:
            return (parts[0], "/".join(parts[1:]))
        return ("images", parts[0])

    @staticmethod
    def _to_number(value: Any) -> float:
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                pass
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        if isinstance(value, (list, tuple)):
            return np.array(value)
        return np.asarray(value)

    def _prepare_image(self, image: Any) -> np.ndarray:
        if image is None:
            return None
        array = self._to_numpy(image)
        if array is None:
            return None
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        array = np.ascontiguousarray(array)
        if array.dtype != np.uint8:
            max_val = float(np.max(array)) if array.size else 1.0
            min_val = float(np.min(array)) if array.size else 0.0
            if max_val <= 1.0 and min_val >= 0.0:
                array = np.clip(array, 0.0, 1.0)
                array = (array * 255.0).round().astype(np.uint8)
            else:
                array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        return array

    def _log_warning(self, method: str, error: Exception):
        if self.logger is not None:
            self.logger.warning(f"ClearML logger {method} failed: {error}")
