from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


class CometMLWriter:
    """
    CometML-based experiment tracker.

    Matches the interface expected by the training pipeline so the writer can
    be swapped in via Hydra configs (similar to the WandB writer).
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        tags: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        self.logger = logger
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()
        self.run_id = run_id
        self._experiment = None

        try:
            from comet_ml import Experiment, OfflineExperiment, ExistingExperiment  # type: ignore
        except ImportError:
            if self.logger is not None:
                self.logger.warning("For use comet_ml install it via \n\t pip install comet-ml")
            return

        # Select the appropriate Comet class/kwargs
        if run_id is not None and mode != "offline":
            # Properly resume logging to an existing online experiment
            ExperimentClass = ExistingExperiment
            experiment_kwargs = {"previous_experiment": run_id}
        else:
            ExperimentClass = OfflineExperiment if mode == "offline" else Experiment
            experiment_kwargs = {
                "project_name": project_name,
                "workspace": workspace,
                "disabled": mode == "disabled",
            }

        try:
            self._experiment = ExperimentClass(**experiment_kwargs)
        except TypeError:
            # Some older versions of comet_ml don't accept the disabled flag
            experiment_kwargs.pop("disabled", None)
            self._experiment = ExperimentClass(**experiment_kwargs)

        if self._experiment is None:
            return

        if run_name is not None:
            try:
                self._experiment.set_name(str(run_name))
            except Exception as error:
                if self.logger is not None:
                    self.logger.warning(f"Failed to set CometML run name: {error}")

        if tags:
            try:
                self._experiment.add_tags(list(tags))
            except Exception as error:
                if self.logger is not None:
                    self.logger.warning(f"Failed to add CometML tags: {error}")

        if project_config:
            try:
                # Log nested parameters (flattened)
                self._experiment.log_parameters(self._flatten_config(project_config))
            except Exception as error:
                if self.logger is not None:
                    self.logger.warning(f"Failed to log CometML parameters: {error}")

        # Set/derive run id
        if run_id is not None:
            self.run_id = run_id
        else:
            try:
                self.run_id = self._experiment.get_key()
            except Exception:
                pass

    def set_step(self, step, mode="train"):
        """
        Update current step and mode, logging step time for monitoring.
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
        Log a scalar metric to CometML.
        """
        if self._experiment is None:
            return
        value = self._to_number(scalar)
        try:
            self._experiment.log_metric(
                name=scalar_name,
                value=value,
                step=int(self.step),
            )
        except Exception as error:
            self._log_warning("log_metric", error)

    def add_scalars(self, scalars):
        """
        Log multiple scalar metrics in a single call.
        """
        if self._experiment is None:
            return
        for name, value in scalars.items():
            self.add_scalar(name, value)

    def add_image(self, image_name, image):
        """
        Log an image to CometML.
        """
        if self._experiment is None:
            return
        prepared = self._prepare_image(image)
        if prepared is None:
            return
        try:
            self._experiment.log_image(
                prepared,
                name=image_name,
                step=int(self.step),
            )
        except Exception as error:
            self._log_warning("log_image", error)

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log audio to CometML (expects numpy array or path).
        """
        if self._experiment is None or audio is None:
            return
        audio_np = self._to_numpy(audio)
        try:
            self._experiment.log_audio(
                audio_np,
                sample_rate=sample_rate,
                name=audio_name,
                step=int(self.step),
            )
        except Exception as error:
            self._log_warning("log_audio", error)

    def add_text(self, text_name, text):
        """
        Log a text snippet to CometML.
        """
        if self._experiment is None:
            return
        try:
            self._experiment.log_text(text_name, str(text), step=int(self.step))
        except Exception as error:
            self._log_warning("log_text", error)

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram data to CometML.
        """
        if self._experiment is None or values_for_hist is None:
            return
        values = self._to_numpy(values_for_hist)
        try:
            self._experiment.log_histogram_3d(
                values=values,
                name=hist_name,
                step=int(self.step),
                bins=bins,
            )
        except Exception as error:
            self._log_warning("log_histogram_3d", error)

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log tabular data to CometML.
        """
        if self._experiment is None or table is None:
            return
        try:
            self._experiment.log_table(
                filename=f"{self.mode}_{table_name}.csv",
                tabular_data=table,
                step=int(self.step),
            )
        except Exception as error:
            self._log_warning("log_table", error)

    def add_images(self, images_name, images):
        """
        Log multiple images at once.
        """
        if not images:
            return
        for idx, image in enumerate(images):
            name = f"{images_name}/{idx}"
            self.add_image(name, image)

    @staticmethod
    def _flatten_config(config: Any, parent_key: str = "", sep: str = "."):
        """
        Flatten nested configs into a dict compatible with Comet log_parameters.
        """
        items = {}
        if isinstance(config, dict):
            iterable = config.items()
        elif hasattr(config, "items"):
            iterable = config.items()
        else:
            return {parent_key: config} if parent_key else {}

        for key, value in iterable:
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            if isinstance(value, dict) or hasattr(value, "items"):
                items.update(CometMLWriter._flatten_config(value, new_key, sep=sep))
            else:
                items[new_key] = value
        return items

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
        if isinstance(value, (str, Path)):
            return np.array(value)
        return np.asarray(value)

    @staticmethod
    def _prepare_image(image: Any):
        if image is None:
            return None
        if hasattr(image, "save"):
            # PIL image
            return image
        array = CometMLWriter._to_numpy(image)
        if array is None:
            return None
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        array = np.ascontiguousarray(array)
        return array

    def _log_warning(self, method: str, error: Exception):
        if self.logger is not None:
            self.logger.warning(f"CometML logger {method} failed: {error}")
