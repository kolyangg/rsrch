"""Deterministic AutoMask-OS class mapping and subject selection."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .ownership_maps import OwnershipMaps


POLICY_VERSION = "automask_os_v1"
BISEnet_CLASSES = {
    "visible_face": (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13),
    "hair_head": (17,),
    "accessory": (6, 18),
}


def image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def select_subject(detections, reference_embedding, expected_location=None,
                   score_threshold=0.35, margin_threshold=0.05):
    def location_score(bbox):
        if expected_location is None:
            return 1.0
        ax0, ay0, ax1, ay1 = (float(value) for value in bbox)
        bx0, by0, bx1, by1 = (float(value) for value in expected_location)
        intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
            0.0, min(ay1, by1) - max(ay0, by0)
        )
        union = max(1.0e-6, (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - intersection)
        return intersection / union

    scored = []
    reference = F.normalize(torch.as_tensor(reference_embedding).float(), dim=0)
    for detection in detections:
        embedding = F.normalize(torch.as_tensor(detection["embedding"]).float(), dim=0)
        identity = float(torch.dot(reference, embedding))
        location = float(detection.get("location_score", location_score(detection["bbox"])))
        detector = float(detection.get("det_score", 0.0))
        scored.append((0.70 * identity + 0.20 * location + 0.10 * detector, detection))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < score_threshold:
        raise RuntimeError("AutoMask-OS found no eligible subject")
    margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0.0)
    if len(scored) > 1 and margin < margin_threshold:
        raise RuntimeError("AutoMask-OS subject selection is ambiguous")
    return scored[0][1], scored[0][0], margin


def probabilities_from_parser(parser_probabilities: torch.Tensor, class_ids: dict,
                              face_support: torch.Tensor) -> torch.Tensor:
    """Map pinned parser classes to the six fixed ownership classes."""
    def combine(name):
        ids = class_ids.get(name, ())
        return parser_probabilities[list(ids)].sum(0) if ids else parser_probabilities.new_zeros(parser_probabilities.shape[1:])
    visible = combine("visible_face")
    hair = combine("hair_head")
    accessory = combine("accessory")
    parser_entropy = -(parser_probabilities * parser_probabilities.clamp_min(1e-8).log()).sum(0)
    uncertain = (parser_entropy / np.log(max(2, parser_probabilities.shape[0]))).clamp(0, 1) * face_support
    known = (visible + hair + accessory).clamp(0, 1)
    occluder = ((1.0 - known) * face_support * (1.0 - uncertain)).clamp(0, 1)
    background = (1.0 - face_support).clamp(0, 1)
    result = torch.stack((visible, hair, accessory, occluder, uncertain, background)).clamp_min(0)
    return result / result.sum(0, keepdim=True).clamp_min(1e-8)


class AutoMaskOS:
    """Thin adapter around pinned detector/parser callables used by the CLI."""
    def __init__(self, detector, parser, class_ids, *, policy_version=POLICY_VERSION,
                 score_threshold=0.35, margin_threshold=0.05):
        self.detector, self.parser, self.class_ids = detector, parser, class_ids
        self.policy_version = str(policy_version)
        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)

    def build(self, image, *, reference_embedding, expected_location=None):
        selected, score, margin = select_subject(
            self.detector(image), reference_embedding, expected_location,
            score_threshold=self.score_threshold,
            margin_threshold=self.margin_threshold,
        )
        parser_probabilities, face_support = self.parser(image, selected["bbox"])
        probabilities = probabilities_from_parser(parser_probabilities, self.class_ids, face_support)
        confidence = 1.0 - probabilities[4]
        return OwnershipMaps(
            probabilities=probabilities, confidence=confidence,
            selected_bbox=tuple(selected["bbox"]), subject_score=score,
            subject_margin=margin, policy_version=self.policy_version,
            source_hash=image_sha256(image),
        ).validate()


class PinnedAutoMaskBuilder:
    """Pinned InsightFace/BiSeNet builder shared by offline and two-pass validation."""

    def __init__(self, device="cuda:0", *, policy_version=POLICY_VERSION,
                 score_threshold=0.35, margin_threshold=0.05):
        from facexlib.parsing import init_parsing_model
        from src.model.photomaker_branched.insightface_package import (
            analyze_faces, create_face_analyzer,
        )

        self.device = torch.device(device)
        self.parser_model = init_parsing_model(
            model_name="bisenet", device=str(self.device)
        ).eval()
        self.analyzer = create_face_analyzer(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": self.device.index or 0}, {}],
            allowed_modules=["detection", "recognition"],
            ctx_id=self.device.index or 0, det_size=(640, 640),
            fallback_ctx_id=-1, quiet=True,
        )

        def detector(image):
            values = analyze_faces(
                self.analyzer,
                np.asarray(image.convert("RGB"))[:, :, ::-1],
            )
            return [dict(
                embedding=value["embedding"], bbox=value["bbox"],
                det_score=value.get("det_score", 0.0),
            ) for value in values]

        def parser(image, bbox):
            rgb = torch.from_numpy(
                np.asarray(image.convert("RGB")).copy()
            ).permute(2, 0, 1)[None].float() / 255.0
            height, width = rgb.shape[-2:]
            network = F.interpolate(
                rgb, (512, 512), mode="bilinear", align_corners=False
            ).to(self.device)
            network = network - network.new_tensor(
                [0.485, 0.456, 0.406]
            )[None, :, None, None]
            network = network / network.new_tensor(
                [0.229, 0.224, 0.225]
            )[None, :, None, None]
            with torch.no_grad():
                output = self.parser_model(network)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                probabilities = torch.softmax(logits.float(), dim=1)
                probabilities = F.interpolate(
                    probabilities, (height, width), mode="bilinear", align_corners=False
                )[0].cpu()
            x0, y0, x1, y1 = (int(round(float(value))) for value in bbox)
            expand_x, expand_y = int(0.15 * (x1-x0)), int(0.15 * (y1-y0))
            support = torch.zeros(height, width)
            support[max(0, y0-expand_y):min(height, y1+expand_y),
                    max(0, x0-expand_x):min(width, x1+expand_x)] = 1.0
            return probabilities, support

        self.automask = AutoMaskOS(
            detector, parser, BISEnet_CLASSES, policy_version=policy_version,
            score_threshold=score_threshold, margin_threshold=margin_threshold,
        )
        self.detector = detector

    def build(self, image: Image.Image, reference: Image.Image, *, expected_location=None):
        reference_faces = self.detector(reference)
        if not reference_faces:
            raise RuntimeError("AutoMask-OS reference has no detected face")
        return self.automask.build(
            image, reference_embedding=reference_faces[0]["embedding"],
            expected_location=expected_location,
        )
