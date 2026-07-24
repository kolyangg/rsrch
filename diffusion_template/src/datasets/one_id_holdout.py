from __future__ import annotations

from collections.abc import Sequence

from src.datasets.cosmic import OneIDTrain


class OneIDHoldoutTrain(OneIDTrain):
    """One-ID training pool with explicit filenames removed before sampling."""

    def __init__(
        self,
        excluded_filenames: Sequence[str],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        excluded = {str(name) for name in excluded_filenames}
        if not excluded:
            raise ValueError("excluded_filenames must contain at least one filename")
        if len(self.ids) != len(self._index):
            raise RuntimeError(
                "OneIDHoldoutTrain requires aligned ids/index entries; "
                "do not apply dataset limit or index shuffling before exclusion"
            )

        missing = excluded.difference(self.ids)
        if missing:
            raise ValueError(
                f"Excluded one-ID filenames are absent from the training manifest: "
                f"{sorted(missing)}"
            )

        # 24 Jul 2026 - Remove held-out validation images from both diffusion
        # targets and the same-ID reference candidate pool.
        # AICODE-NOTE: OneIDTrain samples references from self.ids, so ids and
        # _index must be filtered together to prevent validation-reference leakage.
        retained = [
            (image_id, metadata)
            for image_id, metadata in zip(self.ids, self._index)
            if image_id not in excluded
        ]
        if not retained:
            raise ValueError("OneIDHoldoutTrain exclusion removed every training image")

        self.ids = [image_id for image_id, _ in retained]
        self._index = [metadata for _, metadata in retained]
        self.excluded_filenames = tuple(sorted(excluded))
