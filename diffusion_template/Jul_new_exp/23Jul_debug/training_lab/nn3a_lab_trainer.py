"""Experiment-local validation logging helpers.

The production trainer is unchanged.  This subclass only prefixes validation
image IDs so every mode/checkpoint can share one Comet experiment without
ambiguous or colliding image names.
"""

from __future__ import annotations

import copy

from src.trainer.sdxl_trainers import PhotomakerLoraTrainer


class NN3aLabTrainer(PhotomakerLoraTrainer):
    """Local staged-training and collision-proof validation behavior."""

    def process_batch(self, batch, train_metrics):
        stage_step = getattr(self.config, "lab_staged_up0_start_step", None)
        if self.is_train and stage_step is not None:
            logical_step = (
                (int(self._last_epoch) - 1) * int(self.epoch_len)
                + int(batch["batch_idx"])
            )
            stage_step = int(stage_step)
            up0_groups = [
                group
                for group in self.optimizer.param_groups
                if "__up0" in str(group.get("name", ""))
            ]
            if not up0_groups:
                raise RuntimeError(
                    "Staged up0 training requested but no __up0 optimizer groups exist"
                )
            if logical_step < stage_step:
                for group in up0_groups:
                    group["lr"] = 0.0
            elif not getattr(self, "_lab_logged_up0_unfreeze", False):
                print(
                    "[NN3a staged optimizer] "
                    f"step={logical_step} enabled_up0_groups="
                    + ",".join(str(group.get("name")) for group in up0_groups)
                )
                self._lab_logged_up0_unfreeze = True
        return super().process_batch(batch, train_metrics)

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            return super()._log_batch(batch_idx, batch, mode)

        tagged = copy.copy(batch)
        stream = str(getattr(self.config, "lab_validation_stream", mode))
        step = int(getattr(self.writer, "step", 0))
        prefix = f"{stream}__step{step:04d}__"
        ids = batch.get("id")
        if isinstance(ids, str):
            tagged["id"] = prefix + ids
        elif isinstance(ids, list):
            tagged["id"] = [
                [prefix + str(value) for value in item]
                if isinstance(item, list)
                else prefix + str(item)
                for item in ids
            ]
        return super()._log_batch(batch_idx, tagged, mode)
