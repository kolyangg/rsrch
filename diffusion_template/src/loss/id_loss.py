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
import numpy as np


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
        # This module is attached lazily after DDP construction. Keep the
        # frozen recognizer in evaluation mode even after parent train() calls.
        self.net.eval()
        net_dtype = next(self.net.parameters()).dtype
        x = self._standardize(faces).to(net_dtype)
        emb = self.net(x)
        return F.normalize(emb, dim=-1)

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
        gen_images: torch.Tensor,
        gt_images: torch.Tensor,
        face_bboxes,
        *,
        reference_images=None,
        reference_bboxes=None,
    ):
        """gen_images/gt_images: (B,3,H,W) in [-1,1] (gen carries grad, gt does not).
        face_bboxes: per-sample [x0,y0,x1,y1] in pixel coords.
        Returns a scalar cosine-distance loss (0 when no valid face crop)."""
        B, _, H, W = gen_images.shape
        if not isinstance(face_bboxes, (list, tuple)):
            face_bboxes = [face_bboxes]
        if len(face_bboxes) == 1 and B > 1:
            face_bboxes = list(face_bboxes) * B

        if reference_images is not None:
            if reference_bboxes is None:
                raise ValueError("reference_bboxes are required with reference_images")
            reference_images = list(reference_images)
            reference_bboxes = list(reference_bboxes)

        gen_faces, gt_faces = [], []
        for i in range(B):
            bbox = face_bboxes[i] if i < len(face_bboxes) else face_bboxes[-1]
            gc = self._crop_resize(gen_images[i], bbox, H, W)
            if reference_images is None:
                tc = self._crop_resize(gt_images[i], bbox, H, W)
            else:
                ref = self._reference_tensor(reference_images[i], device=gen_images.device)
                ref_h, ref_w = ref.shape[-2:]
                ref_bbox = reference_bboxes[i]
                tc = self._crop_resize(ref, ref_bbox, ref_h, ref_w)
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


class CausalIdentityLoss(IdentityLoss):
    """Correct/null/wrong identity-direction loss on aligned decoded faces."""

    def __init__(self, face_size: int = 160, alignment_size: int = 112, device=None):
        super().__init__(face_size=face_size, device=device)
        self.alignment_size = int(alignment_size)

    @staticmethod
    def _warp_affine(
        image: torch.Tensor,
        matrix,
        *,
        output_size: int,
    ) -> torch.Tensor:
        """Differentiably warp a source image with a source->destination affine matrix."""
        _, source_h, source_w = image.shape
        matrix3 = torch.eye(3, device=image.device, dtype=torch.float32)
        matrix3[:2] = torch.as_tensor(matrix, device=image.device, dtype=torch.float32)
        inverse = torch.linalg.inv(matrix3)
        ys, xs = torch.meshgrid(
            torch.arange(output_size, device=image.device, dtype=torch.float32),
            torch.arange(output_size, device=image.device, dtype=torch.float32),
            indexing="ij",
        )
        destination = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
        source = destination @ inverse.T
        grid = torch.stack(
            [
                2.0 * source[..., 0] / max(source_w - 1, 1) - 1.0,
                2.0 * source[..., 1] / max(source_h - 1, 1) - 1.0,
            ],
            dim=-1,
        )
        return F.grid_sample(
            image.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _aligned_face(self, image: torch.Tensor, landmarks, bbox) -> torch.Tensor | None:
        if landmarks is not None:
            from insightface.utils import face_align

            kps = np.asarray(landmarks, dtype=np.float32)
            if kps.shape == (5, 2) and np.isfinite(kps).all():
                matrix, _ = face_align.estimate_norm(kps, image_size=self.alignment_size)
                aligned = self._warp_affine(
                    image,
                    matrix,
                    output_size=self.alignment_size,
                )
                return F.interpolate(
                    aligned,
                    size=(self.face_size, self.face_size),
                    mode="bilinear",
                    align_corners=False,
                )
        return self._crop_resize(
            image,
            bbox,
            int(image.shape[-2]),
            int(image.shape[-1]),
        )

    @staticmethod
    def _normalize_items(items, batch_size):
        if items is None:
            return [None] * batch_size
        items = list(items)
        if len(items) == 1 and batch_size > 1:
            items = items * batch_size
        if len(items) != batch_size:
            raise ValueError(f"Expected {batch_size} metadata items, got {len(items)}")
        return items

    def _generated_faces(self, images, landmarks, bboxes):
        batch_size = images.shape[0]
        landmarks = self._normalize_items(landmarks, batch_size)
        bboxes = self._normalize_items(bboxes, batch_size)
        faces = [
            self._aligned_face(images[i], landmarks[i], bboxes[i])
            for i in range(batch_size)
        ]
        if any(face is None for face in faces):
            raise ValueError("Causal identity loss received an invalid generated face crop")
        return torch.cat(faces, dim=0)

    def _reference_faces(self, images, landmarks, bboxes, *, device):
        images = list(images)
        batch_size = len(images)
        landmarks = self._normalize_items(landmarks, batch_size)
        bboxes = self._normalize_items(bboxes, batch_size)
        faces = []
        for image, kps, bbox in zip(images, landmarks, bboxes):
            tensor = self._reference_tensor(image, device=device)
            face = self._aligned_face(tensor, kps, bbox)
            if face is None:
                raise ValueError("Causal identity loss received an invalid reference face crop")
            faces.append(face)
        return torch.cat(faces, dim=0)

    @torch.no_grad()
    def prepare_reference_embeddings(
        self,
        images,
        landmarks,
        bboxes,
        *,
        device,
        global_negatives: bool,
    ):
        local = self._embed(
            self._reference_faces(images, landmarks, bboxes, device=device)
        )
        candidates = local
        if (
            global_negatives
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            gathered = [
                torch.empty_like(local)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered, local)
            candidates = torch.cat(gathered, dim=0)
        return local, candidates

    def forward(
        self,
        correct_images: torch.Tensor,
        null_images: torch.Tensor,
        wrong_images: torch.Tensor,
        *,
        target_landmarks,
        target_bboxes,
        reference_images,
        reference_landmarks,
        reference_bboxes,
        wrong_indices: torch.Tensor,
        global_negatives: bool,
        margin: float,
        direct_weight: float,
        wrong_weight: float,
        cross_weight: float,
        preservation_weight: float,
        structure_weight: float,
        prepared_reference_embeddings=None,
    ):
        correct_faces = self._generated_faces(
            correct_images, target_landmarks, target_bboxes
        )
        null_faces = self._generated_faces(
            null_images, target_landmarks, target_bboxes
        )
        wrong_faces = self._generated_faces(
            wrong_images, target_landmarks, target_bboxes
        )

        correct_embeddings = self._embed(correct_faces)
        wrong_embeddings = self._embed(wrong_faces)
        with torch.no_grad():
            null_embeddings = self._embed(null_faces)
            if prepared_reference_embeddings is None:
                reference_embeddings, candidates = self.prepare_reference_embeddings(
                    reference_images,
                    reference_landmarks,
                    reference_bboxes,
                    device=correct_images.device,
                    global_negatives=global_negatives,
                )
            else:
                reference_embeddings, candidates = prepared_reference_embeddings
            wrong_reference_embeddings = candidates.index_select(
                0, wrong_indices.to(device=candidates.device, dtype=torch.long)
            )

        sim_correct_correct = (correct_embeddings * reference_embeddings).sum(dim=-1)
        sim_null_correct = (null_embeddings * reference_embeddings).sum(dim=-1)
        sim_wrong_wrong = (wrong_embeddings * wrong_reference_embeddings).sum(dim=-1)
        sim_null_wrong = (null_embeddings * wrong_reference_embeddings).sum(dim=-1)
        sim_correct_wrong = (correct_embeddings * wrong_reference_embeddings).sum(dim=-1)
        sim_wrong_correct = (wrong_embeddings * reference_embeddings).sum(dim=-1)

        correct_gain = sim_correct_correct - sim_null_correct
        wrong_gain = sim_wrong_wrong - sim_null_wrong
        correct_rank = F.relu(float(margin) - correct_gain).mean()
        wrong_rank = F.relu(float(margin) - wrong_gain).mean()
        correct_cross_rank = F.relu(
            float(margin) + sim_correct_wrong - sim_correct_correct
        ).mean()
        wrong_cross_rank = F.relu(
            float(margin) + sim_wrong_correct - sim_wrong_wrong
        ).mean()
        cross_rank = 0.5 * (correct_cross_rank + wrong_cross_rank)
        direct = (1.0 - sim_correct_correct).mean()

        pooled_correct = F.adaptive_avg_pool2d(correct_faces, output_size=(16, 16))
        pooled_null = F.adaptive_avg_pool2d(null_faces.detach(), output_size=(16, 16))
        chroma_correct = pooled_correct - pooled_correct.mean(dim=1, keepdim=True)
        chroma_null = pooled_null - pooled_null.mean(dim=1, keepdim=True)
        preservation = F.l1_loss(chroma_correct, chroma_null)
        gray_correct = pooled_correct.mean(dim=1, keepdim=True)
        gray_null = pooled_null.mean(dim=1, keepdim=True)
        structure = 0.5 * (
            F.l1_loss(
                gray_correct[..., 1:, :] - gray_correct[..., :-1, :],
                gray_null[..., 1:, :] - gray_null[..., :-1, :],
            )
            + F.l1_loss(
                gray_correct[..., :, 1:] - gray_correct[..., :, :-1],
                gray_null[..., :, 1:] - gray_null[..., :, :-1],
            )
        )

        loss = (
            correct_rank
            + float(wrong_weight) * wrong_rank
            + float(cross_weight) * cross_rank
            + float(direct_weight) * direct
            + float(preservation_weight) * preservation
            + float(structure_weight) * structure
        )
        return {
            "loss": loss,
            "correct_gain": correct_gain.mean(),
            "wrong_gain": wrong_gain.mean(),
            "correct_similarity": sim_correct_correct.mean(),
            "wrong_similarity": sim_wrong_wrong.mean(),
            "preservation": preservation,
            "structure": structure,
        }
