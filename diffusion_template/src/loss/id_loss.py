"""Optional differentiable reference-identity loss for NN1e/NN1f."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """Cosine distance between generated and trusted-reference face embeddings."""

    def __init__(self, face_size: int = 160, device=None):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:  # pragma: no cover - depends on the run environment
            raise ImportError(
                "NN1 identity loss requires facenet-pytorch. Install it without dependency "
                "downgrades using: pip install --no-deps facenet-pytorch"
            ) from exc

        self.face_size = int(face_size)
        self.net = InceptionResnetV1(pretrained="vggface2").eval()
        self.net.requires_grad_(False)
        if device is not None:
            self.net.to(device)

    @staticmethod
    def _standardize(images: torch.Tensor) -> torch.Tensor:
        images = images.clamp(-1.0, 1.0)
        return (((images + 1.0) * 0.5) * 255.0 - 127.5) / 128.0

    def _crop_resize(self, image: torch.Tensor, bbox) -> torch.Tensor | None:
        height, width = image.shape[-2:]
        x0, y0, x1, y1 = (int(round(float(value))) for value in list(bbox)[:4])
        x0, x1 = max(0, min(width - 1, x0)), max(0, min(width, x1))
        y0, y1 = max(0, min(height - 1, y0)), max(0, min(height, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return F.interpolate(
            image[:, y0:y1, x0:x1].unsqueeze(0),
            size=(self.face_size, self.face_size),
            mode="bilinear",
            align_corners=False,
        )

    def _embed(self, faces: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        dtype = next(self.net.parameters()).dtype
        embeddings = self.net(self._standardize(faces).to(dtype=dtype))
        return F.normalize(embeddings, dim=-1)

    @staticmethod
    def _reference_tensor(image, *, device: torch.device) -> torch.Tensor:
        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("Reference image list is empty")
            image = image[0]
        if torch.is_tensor(image):
            tensor = image.detach().float()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError(f"Unsupported reference tensor shape: {tuple(tensor.shape)}")
            if tensor.max() > 1.5:
                tensor = tensor / 127.5 - 1.0
            elif tensor.min() >= 0:
                tensor = tensor * 2.0 - 1.0
            return tensor.to(device=device)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).to(device=device)

    def forward(
        self,
        generated_images: torch.Tensor,
        target_images: torch.Tensor,
        target_bboxes,
        *,
        reference_images=None,
        reference_bboxes=None,
    ) -> torch.Tensor:
        batch_size = generated_images.shape[0]
        target_bboxes = list(target_bboxes)
        if len(target_bboxes) == 1 and batch_size > 1:
            target_bboxes *= batch_size
        if len(target_bboxes) != batch_size:
            raise ValueError(
                f"Identity loss expected {batch_size} target boxes, got {len(target_bboxes)}"
            )

        if reference_images is not None:
            if reference_bboxes is None:
                raise ValueError("reference_bboxes are required with reference_images")
            reference_images = list(reference_images)
            reference_bboxes = list(reference_bboxes)
            if len(reference_images) != batch_size or len(reference_bboxes) != batch_size:
                raise ValueError("Identity loss reference metadata does not match the batch")

        generated_faces, identity_faces = [], []
        for index in range(batch_size):
            generated_face = self._crop_resize(
                generated_images[index],
                target_bboxes[index],
            )
            if reference_images is None:
                identity_face = self._crop_resize(
                    target_images[index],
                    target_bboxes[index],
                )
            else:
                reference = self._reference_tensor(
                    reference_images[index],
                    device=generated_images.device,
                )
                identity_face = self._crop_resize(reference, reference_bboxes[index])
            if generated_face is None or identity_face is None:
                raise ValueError("Identity loss received an invalid face crop after strict validation")
            generated_faces.append(generated_face)
            identity_faces.append(identity_face)

        generated_embeddings = self._embed(torch.cat(generated_faces, dim=0))
        with torch.no_grad():
            identity_embeddings = self._embed(torch.cat(identity_faces, dim=0))
        return (1.0 - (generated_embeddings * identity_embeddings).sum(dim=-1)).mean()
