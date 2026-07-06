"""Identity loss for branched-attention training.

Compares a face-recognition embedding of the GENERATED face against the
GROUND-TRUTH face (from the training image), both cropped at the gen face bbox,
via cosine distance. Fully differentiable (VAE decode + FaceNet), so gradients
flow back into the branched-attention weights and reward the generated face
matching the target identity — the thing plain denoising MSE does not reward.

Off by default. Enable with `+model.use_id_loss=true` (see lora2.py). The FaceNet
recognizer (facenet-pytorch InceptionResnetV1, VGGFace2) is frozen and never
added to the optimizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    def __init__(self, face_size: int = 160, device=None):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "id_loss needs facenet-pytorch. Install it WITHOUT its deps so it does NOT "
                "downgrade torch/torchvision (its metadata pins torch<2.3, which will uninstall a "
                "newer torch):\n    pip install --no-deps facenet-pytorch\n"
                "Only InceptionResnetV1 (standard layers) is used, so any modern torch works."
            ) from e

        self.face_size = int(face_size)
        net = InceptionResnetV1(pretrained="vggface2").eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        if device is not None:
            self.net.to(device)

    @staticmethod
    def _standardize(img: torch.Tensor) -> torch.Tensor:
        """img in [-1, 1] (B,3,H,W) -> FaceNet fixed_image_standardization."""
        img = img.clamp(-1.0, 1.0)
        img01 = (img + 1.0) * 0.5                       # [0, 1]
        return (img01 * 255.0 - 127.5) / 128.0

    def _crop_resize(self, img: torch.Tensor, bbox, H: int, W: int):
        """img (3,H,W) in [-1,1]; bbox [x0,y0,x1,y1] in pixel coords.
        Returns (1,3,face,face) or None if the box is degenerate."""
        x0, y0, x1, y1 = (int(round(float(v))) for v in list(bbox)[:4])
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        crop = img[:, y0:y1, x0:x1].unsqueeze(0)        # (1,3,h,w)
        return F.interpolate(
            crop, size=(self.face_size, self.face_size),
            mode="bilinear", align_corners=False,
        )

    def _embed(self, faces: torch.Tensor) -> torch.Tensor:
        net_dtype = next(self.net.parameters()).dtype
        x = self._standardize(faces).to(net_dtype)
        emb = self.net(x)
        return F.normalize(emb, dim=-1)

    def forward(self, gen_images: torch.Tensor, gt_images: torch.Tensor, face_bboxes):
        """gen_images/gt_images: (B,3,H,W) in [-1,1] (gen carries grad, gt does not).
        face_bboxes: per-sample [x0,y0,x1,y1] in pixel coords.
        Returns a scalar cosine-distance loss (0 when no valid face crop)."""
        B, _, H, W = gen_images.shape
        if not isinstance(face_bboxes, (list, tuple)):
            face_bboxes = [face_bboxes]
        if len(face_bboxes) == 1 and B > 1:
            face_bboxes = list(face_bboxes) * B

        gen_faces, gt_faces = [], []
        for i in range(B):
            bbox = face_bboxes[i] if i < len(face_bboxes) else face_bboxes[-1]
            gc = self._crop_resize(gen_images[i], bbox, H, W)
            tc = self._crop_resize(gt_images[i], bbox, H, W)
            if gc is None or tc is None:
                continue
            gen_faces.append(gc)
            gt_faces.append(tc)

        if not gen_faces:
            return gen_images.new_zeros(())

        gen_emb = self._embed(torch.cat(gen_faces, dim=0))
        with torch.no_grad():
            gt_emb = self._embed(torch.cat(gt_faces, dim=0))
        cos = (gen_emb * gt_emb).sum(dim=-1)
        return (1.0 - cos).mean()
