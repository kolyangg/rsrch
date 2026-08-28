"""Analysis-only indexed view of the sealed manual validation panel."""

from collections.abc import Sequence

from src.datasets.manual_val import ManualPhotoMakerValDataset


class IndexedManualPhotoMakerValDataset(ManualPhotoMakerValDataset):
    """Expose selected original indices without renumbering bbox lookup."""

    def __init__(self, *, indices: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        selected = [int(index) for index in indices]
        if not selected:
            raise ValueError("indices must contain at least one validation item")
        if len(set(selected)) != len(selected):
            raise ValueError(f"indices must be unique: {selected}")
        invalid = [index for index in selected if not 0 <= index < len(self.samples)]
        if invalid:
            raise IndexError(
                f"manual-val indices outside [0, {len(self.samples) - 1}]: {invalid}"
            )
        self.indices = selected

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        return super().__getitem__(self.indices[position])
