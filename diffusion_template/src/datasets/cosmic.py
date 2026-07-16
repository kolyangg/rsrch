from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import get_bigger_crop, get_crop_values

logger = logging.getLogger(__name__)

PROMPT_CLASS_RE = re.compile(r"\b(woman|man|girl|boy|child|person)\b", re.IGNORECASE)


class CosmicDoubledTrain(BaseDataset):
    def __init__(
        self,
        cosmic_json_pth=None,
        cosmic_texts_json_pth=None,
        cosmic_large_json_pth=None,
        cosmic_large_texts_json_pth=None,
        images_path=None,
        num_refs=1,
        train_on_separate_image=False,
        same_id_ref_map_json_pth=None,
        *args,
        **kwargs,
    ):
        self.images_path = images_path
        self.train_on_separate_image = bool(train_on_separate_image)
        self.same_id_ref_map = {}
        if same_id_ref_map_json_pth is not None:
            with open(same_id_ref_map_json_pth) as f:
                self.same_id_ref_map = json.load(f)

        # Init cosmic
        with open(cosmic_json_pth) as f:
            cosmic_json = json.load(f)

        with open(cosmic_texts_json_pth) as f:
            cosmic_texts_json = json.load(f)

        self.num_refs = num_refs

        index = []
        self.ids = []
        self.meta_by_path = {}
        for k, v in tqdm(cosmic_json.items()):
            if k not in cosmic_texts_json:
                continue
            v.update(cosmic_texts_json[k])
            index.append(v)
            self.ids.append(k)
            self.meta_by_path[k] = v

        # Init cosmic large
        with open(cosmic_large_json_pth) as f:
            cosmic_large_json = json.load(f)

        with open(cosmic_large_texts_json_pth) as f:
            cosmic_large_texts_json = json.load(f)

        for k, v in tqdm(cosmic_large_json.items()):
            if not min(v["face_crop_new"]) >= 0 or not max(v["face_crop_new"]) <= 1024:
                continue
            if k not in cosmic_large_texts_json:
                continue
            v.update(cosmic_large_texts_json[k])
            index.append(v)
            path = k.replace("LAION-5B", "LAION-5B-Filtered-Large")
            self.ids.append(path)
            self.meta_by_path[path] = v

        # self.flip = RandomHorizontalFlip(p=0.5)

        super().__init__(index, *args, **kwargs)

    def _load_train_image(self, path, img_data):
        img = Image.open(f"{self.images_path}/{path}").convert("RGB")
        if img.size != (1024, 1024):
            body_crop = img_data["body_crop"]
            img_arr = np.array(img)[body_crop[1]:body_crop[3], body_crop[0]:body_crop[2]]
            assert img_arr.shape[0] == 1024, img_arr
            assert img_arr.shape[1] == 1024, img_arr
            img = Image.fromarray(img_arr)
        return img

    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = self.ids[ind]

        instance_data = {}

        img = self._load_train_image(path, img_data)

        # instance_data["pixel_values"] = img
        # bbox = img_data["face_crop_new"]
        # instance_data["face_bbox"] = deepcopy(bbox)

        # ref_img = deepcopy(img)
        # ref_images = [self.flip(get_bigger_crop(ref_img, crop=deepcopy(bbox)))]

        ### FIX 01 FEB ###
        bbox = deepcopy(img_data["face_crop_new"])
        if random.random() < 0.5:
            w, _ = img.size
            img = ImageOps.mirror(img)
            x0, y0, x1, y1 = bbox
            bbox = [w - x1, y0, w - x0, y1]

        instance_data["pixel_values"] = img
        instance_data["face_bbox"] = bbox
        if self.train_on_separate_image:
            ref_candidates = [p for p in self.same_id_ref_map.get(path, []) if p != path]
            if not ref_candidates:
                raise ValueError(
                    "train_on_separate_image=True for CosmicDoubledTrain requires "
                    f"same_id_ref_map_json_pth with non-empty candidates for '{path}'"
                )
            ref_path = random.choice(ref_candidates)
            ref_data = self.meta_by_path[ref_path]
            ref_img = self._load_train_image(ref_path, ref_data)
            ref_images = [ref_img]
            instance_data["face_bbox_ref"] = deepcopy(ref_data["face_crop_new"])
        else:
            instance_data["face_bbox_ref"] = deepcopy(bbox)
            ref_images = [deepcopy(img)]
        ### FIX 01 FEB ###
    
    
        instance_data["ref_images"] = ref_images

        prompt = ", ".join(
            [img_data["facial_caption"], img_data["pose_caption"], img_data["background_caption"]]
        )
        instance_data["prompts"] = prompt
        instance_data["prompt"] = prompt  # ensure eval has the key it expects

        if "orig_size" in img_data:
            orig_size = img_data["orig_size"]
            instance_data["original_sizes"] = (orig_size[1], orig_size[0])
            instance_data["crop_top_lefts"] = get_crop_values(img_data)
        else:
            orig_size = (1024, 1024)
            img_data["orig_size"] = (1024, 1024)
            instance_data["original_sizes"] = (orig_size[1], orig_size[0])
            instance_data["crop_top_lefts"] = (0, 0)

        instance_data = self.preprocess_data(instance_data)

        assert min(instance_data["face_bbox"]) >= 0
        assert max(instance_data["face_bbox"]) <= 1024

        return instance_data

class OneIDTrain(BaseDataset):
    """
    Small single-ID training dataset that reuses the CosmicDoubledTrain
    item structure but reads from a simpler JSON:
      { "<filename>": { "body_crop": [...], "face_crop": [...],
                        "orig_image_size": [H, W], "text": "..." }, ... }
    """

    def __init__(
        self,
        cosmic_json_pth=None,
        images_path=None,
        num_refs=1,
        train_on_separate_image=False,
        *args,
        **kwargs,
    ):
        self.images_path = images_path
        self.num_refs = num_refs
        self.train_on_separate_image = bool(train_on_separate_image)

        with open(cosmic_json_pth) as f:
            data_json = json.load(f)

        index = []
        self.ids = []
        for k, v in data_json.items():
            index.append(v)
            self.ids.append(k)

        # self.flip = RandomHorizontalFlip(p=0.5)

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = self.ids[ind]

        instance_data = {}

        img = Image.open(f"{self.images_path}/{path}").convert("RGB")
        # instance_data["pixel_values"] = img

        # bbox = img_data["face_crop"]
        # instance_data["face_bbox"] = deepcopy(bbox)

        # ref_img = deepcopy(img)
        # ref_images = [self.flip(get_bigger_crop(ref_img, crop=deepcopy(bbox)))]
        
        ### 01 FEB ###
        bbox = deepcopy(img_data["face_crop"])
        if random.random() < 0.5:
            w, _ = img.size
            img = ImageOps.mirror(img)
            x0, y0, x1, y1 = bbox
            bbox = [w - x1, y0, w - x0, y1]

        instance_data["pixel_values"] = img
        instance_data["face_bbox"] = bbox
        if self.train_on_separate_image and len(self.ids) > 1:
            ref_ind = random.choice([i for i in range(len(self.ids)) if i != ind])
            ref_path = self.ids[ref_ind]
            ref_data = self._index[ref_ind]
            ref_img = Image.open(f"{self.images_path}/{ref_path}").convert("RGB")
            ref_images = [ref_img]
            instance_data["face_bbox_ref"] = deepcopy(ref_data["face_crop"])
        else:
            instance_data["face_bbox_ref"] = deepcopy(bbox)
            ref_images = [deepcopy(img)]
        ### 01 FEB ###
        
        instance_data["ref_images"] = ref_images

        text = img_data.get("text", "")
        prompt = text if isinstance(text, str) and text else "img person"
        instance_data["prompts"] = prompt
        instance_data["prompt"] = prompt

        orig_size = img_data.get("orig_image_size")
        if orig_size is not None and len(orig_size) == 2:
            instance_data["original_sizes"] = (orig_size[1], orig_size[0])
        else:
            h, w = img.size[1], img.size[0]
            instance_data["original_sizes"] = (h, w)
        instance_data["crop_top_lefts"] = (0, 0)

        instance_data = self.preprocess_data(instance_data)

        assert min(instance_data["face_bbox"]) >= 0
        assert max(instance_data["face_bbox"]) <= 1024

        return instance_data


class LargeDatasetTrain(BaseDataset):
    """
    Multi-ID training dataset for large_dataset_adj.

    Expected JSON format:
      {
        "<identity>": {
          "<image_id>": {
            "body_crop": [x0, x1, y0, y1],
            "new_face_crop": [x0, y0, x1, y1],
            "orig_image_size": [H, W],
            "text": "..."
          },
          ...
        },
        ...
      }

    Expected image layout under images_path:
      <images_path>/<identity>/<image_id>.jpg
    """

    def __init__(
        self,
        data_json_pth=None,
        images_path=None,
        num_refs=1,
        train_on_separate_image=False,
        same_id_ref_map_json_pth=None,
        *args,
        **kwargs,
    ):
        self.images_path = images_path
        self.num_refs = num_refs
        self.train_on_separate_image = bool(train_on_separate_image)

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = []
        self.meta_by_path = {}
        self.identity_by_path = {}
        self.same_id_ref_map = {}

        index = []
        for identity, image_records in tqdm(data_json.items()):
            if not isinstance(image_records, dict):
                continue

            identity_paths = []
            for image_id, image_data in image_records.items():
                path = f"{identity}/{image_id}.jpg"
                index.append(image_data)
                self.ids.append(path)
                self.meta_by_path[path] = image_data
                self.identity_by_path[path] = identity
                identity_paths.append(path)

            if identity_paths:
                self.same_id_ref_map[identity] = identity_paths

        if same_id_ref_map_json_pth is not None:
            with open(same_id_ref_map_json_pth) as f:
                loaded_map = json.load(f)
            normalized_map = {}
            for key, values in loaded_map.items():
                if not isinstance(values, list):
                    continue
                normalized_map[key] = [str(v) for v in values]
            self.same_id_ref_map = normalized_map

        super().__init__(index, *args, **kwargs)

    def _load_train_image(self, path, img_data):
        img = Image.open(f"{self.images_path}/{path}").convert("RGB")
        if img.size != (1024, 1024):
            body_crop = img_data["body_crop"]
            img_arr = np.array(img)[body_crop[2]:body_crop[3], body_crop[0]:body_crop[1]]
            assert img_arr.shape[0] == 1024, img_arr.shape
            assert img_arr.shape[1] == 1024, img_arr.shape
            img = Image.fromarray(img_arr)
        return img

    def _get_same_id_ref_candidates(self, path):
        identity = self.identity_by_path.get(path)
        if identity is not None and identity in self.same_id_ref_map:
            return [p for p in self.same_id_ref_map[identity] if p != path]
        return [p for p in self.same_id_ref_map.get(path, []) if p != path]

    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = self.ids[ind]

        instance_data = {}

        img = self._load_train_image(path, img_data)

        bbox = deepcopy(img_data["new_face_crop"])
        if random.random() < 0.5:
            w, _ = img.size
            img = ImageOps.mirror(img)
            x0, y0, x1, y1 = bbox
            bbox = [w - x1, y0, w - x0, y1]

        instance_data["pixel_values"] = img
        instance_data["face_bbox"] = bbox
        if self.train_on_separate_image:
            ref_candidates = self._get_same_id_ref_candidates(path)
            if not ref_candidates:
                raise ValueError(
                    "train_on_separate_image=True for LargeDatasetTrain requires "
                    f"at least two images for identity '{self.identity_by_path.get(path, path)}'"
                )
            ref_path = random.choice(ref_candidates)
            ref_data = self.meta_by_path[ref_path]
            ref_img = self._load_train_image(ref_path, ref_data)
            ref_images = [ref_img]
            instance_data["face_bbox_ref"] = deepcopy(ref_data["new_face_crop"])
        else:
            instance_data["face_bbox_ref"] = deepcopy(bbox)
            ref_images = [deepcopy(img)]

        instance_data["ref_images"] = ref_images

        text = img_data.get("text", "")
        prompt = text if isinstance(text, str) and text else "img person"
        instance_data["prompts"] = prompt
        instance_data["prompt"] = prompt

        instance_data["original_sizes"] = (1024, 1024)
        instance_data["crop_top_lefts"] = (0, 0)

        instance_data = self.preprocess_data(instance_data)

        assert min(instance_data["face_bbox"]) >= 0
        assert max(instance_data["face_bbox"]) <= 1024

        return instance_data


class CosmicLargeTrain_old(BaseDataset):
    def __init__(
        self,
        data_json_pth=None,
        images_path=None,
        target_images_path=None,
        num_refs=1,
        train_on_separate_image=False,
        same_id_ref_map_json_pth=None,
        path_prefix_to_strip=None,
        require_nested_identity_subdir=True,
        upscale_to_1024=True,
        const_ref=True,
        crop_ref=False,
        ref_similar=False,
        origtarget_genref=True,
        crop_nonface_min=0.2,
        crop_nonface_max=0.4,
        *args,
        **kwargs,
    ):
        self.images_path = images_path
        self.target_images_path = target_images_path
        self.num_refs = num_refs
        self.train_on_separate_image = bool(train_on_separate_image)
        self.path_prefix_to_strip = path_prefix_to_strip.strip("/") if path_prefix_to_strip else None
        self.require_nested_identity_subdir = bool(require_nested_identity_subdir)
        self.upscale_to_1024 = bool(upscale_to_1024)
        self.const_ref = bool(const_ref)
        self.crop_ref = bool(crop_ref)
        self.ref_similar = bool(ref_similar)
        self.origtarget_genref = bool(origtarget_genref)
        self.crop_nonface_min = float(crop_nonface_min)
        self.crop_nonface_max = float(crop_nonface_max)
        if self.crop_nonface_min < 0 or self.crop_nonface_max < self.crop_nonface_min:
            raise ValueError(
                "CosmicLargeTrain requires 0 <= crop_nonface_min <= crop_nonface_max; "
                f"got {self.crop_nonface_min} and {self.crop_nonface_max}"
            )
        self.train_image_size = 1024 if (self.origtarget_genref or self.upscale_to_1024) else 256
        images_root = Path(self.images_path) if self.images_path is not None else None
        self.dataset_root = images_root.parents[1] if images_root is not None and len(images_root.parents) >= 2 else None
        self.target_images_root = Path(self.target_images_path) if self.target_images_path is not None else None

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = []
        self.meta_by_path = {}
        self.identity_by_path = {}
        self.parent_image_by_path = {}
        self.same_id_ref_map = {}
        self.face_bbox_by_path = {}

        index = []
        for path, image_data in tqdm(data_json.items()):
            if not isinstance(image_data, dict):
                continue

            paths = [str(path)]
            if self.require_nested_identity_subdir and not self.origtarget_genref:
                face_paths = image_data.get("face_paths")
                if isinstance(face_paths, list):
                    paths = [str(face_path) for face_path in face_paths]

            for sample_path in paths:
                rel_path = self._get_relative_path(sample_path)
                if self.require_nested_identity_subdir and not self.origtarget_genref and len(Path(rel_path).parts) != 3:
                    continue
                bbox = self._resolve_sample_bbox(sample_path, rel_path, image_data)
                if bbox is None:
                    continue
                identity = str(path) if self.origtarget_genref else str(Path(rel_path).parent)

                index.append(image_data)
                self.ids.append(sample_path)
                self.meta_by_path[sample_path] = image_data
                self.identity_by_path[sample_path] = identity
                self.parent_image_by_path[sample_path] = str(path)
                self.face_bbox_by_path[sample_path] = bbox
                self.same_id_ref_map.setdefault(identity, []).append(sample_path)

        if same_id_ref_map_json_pth is not None:
            with open(same_id_ref_map_json_pth) as f:
                loaded_map = json.load(f)
            normalized_map = {}
            for key, values in loaded_map.items():
                if not isinstance(values, list):
                    continue
                normalized_map[key] = [str(v) for v in values]
            self.same_id_ref_map = normalized_map

        super().__init__(index, *args, **kwargs)

    def _get_relative_path(self, path):
        path = str(path)
        if self.images_path:
            images_path = str(self.images_path).rstrip("/")
            prefix = f"{images_path}/"
            if path.startswith(prefix):
                return path[len(prefix):]

        path = path.lstrip("/")
        if self.path_prefix_to_strip:
            prefix = f"{self.path_prefix_to_strip}/"
            if path.startswith(prefix):
                return path[len(prefix):]
        if self.images_path:
            root_name = Path(self.images_path).name
            marker = f"/{root_name}/"
            marked_path = f"/{path}"
            if marker in marked_path:
                return marked_path.split(marker, 1)[1]
        return path

    def _load_train_image(self, path, img_data):
        rel_path = self._get_relative_path(path)
        img = Image.open(f"{self.images_path}/{rel_path}").convert("RGB")
        target_size = (self.train_image_size, self.train_image_size)
        if img.size != target_size:
            body_crop = img_data.get("body_crop")
            if body_crop is not None and len(body_crop) == 4:
                x0, y0, x1, y1 = body_crop
                if 0 <= x0 < x1 <= img.size[0] and 0 <= y0 < y1 <= img.size[1]:
                    img = Image.fromarray(np.array(img)[y0:y1, x0:x1])
            if img.size != target_size:
                img = img.resize(target_size, Image.BICUBIC)
        return img

    def _get_parent_image_full_path(self, path):
        path = Path(str(path))
        if path.is_absolute():
            return path
        if self.target_images_root is not None:
            return self.target_images_root / self._get_relative_path(path)
        if self.dataset_root is not None:
            return self.dataset_root / str(path).lstrip("/")
        return path

    @staticmethod
    def _is_valid_bbox(bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        x0, y0, x1, y1 = bbox
        return x1 > x0 and y1 > y0 and min(bbox) >= 0

    def _resolve_sample_bbox(self, sample_path, rel_path, img_data):
        if self.require_nested_identity_subdir:
            face_bboxes = img_data.get("face_bboxes") or {}
            candidates = [
                str(sample_path),
                str(sample_path).lstrip("/"),
                rel_path,
                rel_path.lstrip("/"),
            ]
            for candidate in candidates:
                bbox = face_bboxes.get(candidate)
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    raw_bbox = [float(v) for v in bbox]
                    clipped_bbox = self._clip_bbox_to_image(raw_bbox, (256, 256))
                    if clipped_bbox is not None:
                        if clipped_bbox != raw_bbox:
                            logger.warning(
                                "Clipped cosmic_large face bbox for %s: raw=%s clipped=%s",
                                candidate,
                                raw_bbox,
                                clipped_bbox,
                            )
                        return clipped_bbox

        bbox = img_data.get("face_crop_new")
        if self._is_valid_bbox(bbox):
            return [float(v) for v in bbox]
        return None

    @staticmethod
    def _scale_bbox_to_size(bbox, src_size, dst_size):
        if bbox is None:
            return None
        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        scale_x = dst_w / max(src_w, 1)
        scale_y = dst_h / max(src_h, 1)
        x0, y0, x1, y1 = [float(v) for v in bbox]
        scaled_bbox = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
        return [
            max(0.0, min(float(dst_w), scaled_bbox[0])),
            max(0.0, min(float(dst_h), scaled_bbox[1])),
            max(0.0, min(float(dst_w), scaled_bbox[2])),
            max(0.0, min(float(dst_h), scaled_bbox[3])),
        ]

    @staticmethod
    def _crop_square_with_bbox(img, bbox, crop_side):
        img_w, img_h = img.size
        crop_side = int(max(1, round(min(float(crop_side), float(img_w), float(img_h)))))
        x0, y0, x1, y1 = [float(v) for v in bbox]
        face_cx = 0.5 * (x0 + x1)
        face_cy = 0.5 * (y0 + y1)

        crop_x0 = int(round(face_cx - 0.5 * crop_side))
        crop_y0 = int(round(face_cy - 0.5 * crop_side))
        crop_x0 = min(max(crop_x0, 0), max(img_w - crop_side, 0))
        crop_y0 = min(max(crop_y0, 0), max(img_h - crop_side, 0))
        crop_x1 = crop_x0 + crop_side
        crop_y1 = crop_y0 + crop_side

        cropped_img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        cropped_bbox = [
            x0 - crop_x0,
            y0 - crop_y0,
            x1 - crop_x0,
            y1 - crop_y0,
        ]
        return cropped_img, cropped_bbox

    def _prepare_constant_ref_crop(self, img, bbox):
        target_size = (self.train_image_size, self.train_image_size)
        if (
            not self.crop_ref
            or self.train_image_size != 256
            or (img.size[0] <= 256 and img.size[1] <= 256)
        ):
            if img.size != target_size:
                resized_img = img.resize(target_size, Image.BICUBIC)
                bbox = self._scale_bbox_to_size(bbox, img.size, resized_img.size)
                return resized_img, bbox
            return img, bbox

        face_w = max(float(bbox[2] - bbox[0]), 1.0)
        face_h = max(float(bbox[3] - bbox[1]), 1.0)
        context_ratio_w = random.uniform(self.crop_nonface_min, self.crop_nonface_max)
        context_ratio_h = random.uniform(self.crop_nonface_min, self.crop_nonface_max)
        desired_w = face_w * (1.0 + context_ratio_w)
        desired_h = face_h * (1.0 + context_ratio_h)
        crop_side = max(float(self.train_image_size), desired_w, desired_h)

        cropped_img, cropped_bbox = self._crop_square_with_bbox(img, bbox, crop_side)
        if cropped_img.size != target_size:
            resized_img = cropped_img.resize(target_size, Image.BICUBIC)
            cropped_bbox = self._scale_bbox_to_size(cropped_bbox, cropped_img.size, resized_img.size)
            return resized_img, cropped_bbox
        return cropped_img, cropped_bbox

    @staticmethod
    def _clip_bbox_to_image(bbox, img_size):
        if bbox is None:
            return None
        img_w, img_h = img_size
        x0, y0, x1, y1 = [float(v) for v in bbox]
        clipped_bbox = [
            max(0.0, min(float(img_w), x0)),
            max(0.0, min(float(img_h), y0)),
            max(0.0, min(float(img_w), x1)),
            max(0.0, min(float(img_h), y1)),
        ]
        if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
            return None
        return clipped_bbox

    @staticmethod
    def _get_bigger_crop_with_bbox(img, face_bbox, scale=0.2):
        crop = [int(round(v)) for v in deepcopy(face_bbox)]
        if crop[3] - crop[1] < crop[2] - crop[0]:
            diff = crop[2] - crop[0] - (crop[3] - crop[1])
            if diff % 2 != 0:
                crop[0] -= 1
                diff += 1
            crop[3] += diff // 2
            crop[1] -= diff // 2
        elif crop[2] - crop[0] < crop[3] - crop[1]:
            diff = crop[3] - crop[1] - (crop[2] - crop[0])
            if diff % 2 != 0:
                crop[1] -= 1
                diff += 1
            crop[2] += diff // 2
            crop[0] -= diff // 2

        assert crop[3] - crop[1] == crop[2] - crop[0], crop

        to_add = int((crop[3] - crop[1]) * scale)
        img_w, img_h = img.size
        crop = [
            max(0, crop[0] - to_add),
            max(0, crop[1] - to_add),
            min(img_w, crop[2] + to_add),
            min(img_h, crop[3] + to_add),
        ]
        cropped_img = img.crop((crop[0], crop[1], crop[2], crop[3]))
        cropped_bbox = [
            face_bbox[0] - crop[0],
            face_bbox[1] - crop[1],
            face_bbox[2] - crop[0],
            face_bbox[3] - crop[1],
        ]
        return cropped_img, CosmicLargeTrain_old._clip_bbox_to_image(cropped_bbox, cropped_img.size)

    def _load_origref_target_image_and_bbox(self, path, img_data):
        full_path = self._get_parent_image_full_path(path)
        raw_img = Image.open(full_path).convert("RGB")
        bbox = deepcopy(img_data.get("face_crop_new"))
        if not self._is_valid_bbox(bbox):
            raise ValueError(f"Missing valid top-level face_crop_new for cosmic_large target: {path}")

        img = raw_img
        body_crop = img_data.get("body_crop")
        if body_crop is not None and len(body_crop) == 4:
            x0, y0, x1, y1 = body_crop
            if 0 <= x0 < x1 <= raw_img.size[0] and 0 <= y0 < y1 <= raw_img.size[1]:
                img = raw_img.crop((x0, y0, x1, y1))
                bbox = [bbox[0] - x0, bbox[1] - y0, bbox[2] - x0, bbox[3] - y0]
                bbox = self._clip_bbox_to_image(bbox, img.size)
                if bbox is None:
                    raise ValueError(f"Invalid cropped face bbox for cosmic_large target: {path}")

        target_size = (self.train_image_size, self.train_image_size)
        if img.size != target_size:
            bbox = self._scale_bbox_to_size(bbox, img.size, target_size)
            img = img.resize(target_size, Image.BICUBIC)

        return img, bbox

    def _load_train_image_and_bbox(self, path, img_data):
        if self.origtarget_genref:
            return self._load_origref_target_image_and_bbox(path, img_data)

        rel_path = self._get_relative_path(path)
        raw_img = Image.open(f"{self.images_path}/{rel_path}").convert("RGB")
        raw_size = raw_img.size
        target_size = (self.train_image_size, self.train_image_size)
        bbox = deepcopy(self.face_bbox_by_path.get(path))
        if bbox is None:
            bbox = self._resolve_sample_bbox(path, rel_path, img_data)
            if bbox is None:
                raise ValueError(f"Missing valid face bbox for cosmic_large sample: {path}")

        img = raw_img
        if self.require_nested_identity_subdir:
            if img.size != target_size:
                img = img.resize(target_size, Image.BICUBIC)
            bbox = self._scale_bbox_to_size(bbox, raw_size, img.size)
        else:
            if img.size != target_size:
                body_crop = img_data.get("body_crop")
                if body_crop is not None and len(body_crop) == 4:
                    x0, y0, x1, y1 = body_crop
                    if 0 <= x0 < x1 <= img.size[0] and 0 <= y0 < y1 <= img.size[1]:
                        img = Image.fromarray(np.array(img)[y0:y1, x0:x1])

                if img.size != target_size:
                    img = img.resize(target_size, Image.BICUBIC)

        return img, bbox

    def _load_constant_ref_image_and_bbox(self, path, img_data):
        full_path = self._get_parent_image_full_path(path)
        raw_img = Image.open(full_path).convert("RGB")
        bbox = deepcopy(img_data.get("face_crop_new"))
        if not self._is_valid_bbox(bbox):
            raise ValueError(f"Missing valid top-level face_crop_new for cosmic_large ref: {path}")

        img = raw_img
        body_crop = img_data.get("body_crop")
        if body_crop is not None and len(body_crop) == 4:
            x0, y0, x1, y1 = body_crop
            if 0 <= x0 < x1 <= raw_img.size[0] and 0 <= y0 < y1 <= raw_img.size[1]:
                img = Image.fromarray(np.array(raw_img)[y0:y1, x0:x1])

        return self._prepare_constant_ref_crop(img, bbox)

    def _load_similar_ref_images_and_bbox(self, img_data, target_path=None):
        face_paths = [str(p) for p in img_data.get("face_paths") or []]
        if self.train_on_separate_image and target_path is not None:
            ref_candidates = [p for p in face_paths if p != str(target_path)]
            if not ref_candidates:
                raise ValueError(
                    "train_on_separate_image=True with ref_similar=True for CosmicLargeTrain "
                    f"requires at least two face_paths for target '{target_path}'"
                )
        else:
            ref_candidates = face_paths

        if not ref_candidates:
            raise ValueError("CosmicLargeTrain ref_similar=True requires non-empty face_paths")

        replace = len(ref_candidates) < self.num_refs
        chosen_paths = np.random.choice(ref_candidates, size=self.num_refs, replace=replace).tolist()

        ref_images = []
        ref_bboxes = []
        for face_path in chosen_paths:
            rel_path = self._get_relative_path(face_path)
            ref_img = Image.open(f"{self.images_path}/{rel_path}").convert("RGB")
            face_bbox = self._resolve_sample_bbox(face_path, rel_path, img_data)
            if face_bbox is None:
                raise ValueError(f"Missing valid face bbox for ref_similar sample: {face_path}")
            ref_face, ref_bbox = self._get_bigger_crop_with_bbox(ref_img, face_bbox, scale=0.2)
            if ref_bbox is None:
                raise ValueError(f"Invalid ref_similar crop bbox for sample: {face_path}")
            if random.random() < 0.5:
                w, _ = ref_face.size
                ref_face = ImageOps.mirror(ref_face)
                x0, y0, x1, y1 = ref_bbox
                ref_bbox = [w - x1, y0, w - x0, y1]
            ref_images.append(ref_face)
            ref_bboxes.append(ref_bbox)

        return ref_images, deepcopy(ref_bboxes[0])

    def _get_same_id_ref_candidates(self, path):
        identity = self.identity_by_path.get(path)
        if identity is not None and identity in self.same_id_ref_map:
            return [p for p in self.same_id_ref_map[identity] if p != path]
        return [p for p in self.same_id_ref_map.get(path, []) if p != path]

    def _build_prompt(self, img_data):
        if self.origtarget_genref:
            facial_caption = img_data.get("facial_caption", "")
            if isinstance(facial_caption, str):
                facial_caption = facial_caption.strip()
                if facial_caption and re.search(r"\bimg\b", facial_caption, re.IGNORECASE):
                    return facial_caption
            return "A person img"

        text = img_data.get("text", "")
        if isinstance(text, str) and text:
            return text

        facial_caption = img_data.get("facial_caption", "")
        prompt_class = "person"
        if isinstance(facial_caption, str) and facial_caption:
            match = PROMPT_CLASS_RE.search(facial_caption)
            if match is not None:
                prompt_class = match.group(1).lower()

        if self.require_nested_identity_subdir:
            return f"A {prompt_class} img"

        prompt_parts = [
            facial_caption,
            img_data.get("pose_caption"),
            img_data.get("background_caption"),
        ]
        prompt = ", ".join(part for part in prompt_parts if isinstance(part, str) and part)
        return prompt or f"A {prompt_class} img"

    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = self.ids[ind]

        if (not self.origtarget_genref) and (self.const_ref or self.ref_similar) and self.train_on_separate_image:
            identity = self.identity_by_path.get(path)
            target_candidates = self.same_id_ref_map.get(identity)
            if not target_candidates:
                raise ValueError(
                    "train_on_separate_image=True for CosmicLargeTrain requires "
                    f"at least one target candidate for identity '{identity}'"
                )
            path = random.choice(target_candidates)
            img_data = self.meta_by_path[path]

        instance_data = {}

        img, bbox = self._load_train_image_and_bbox(path, img_data)

        if random.random() < 0.5:
            w, _ = img.size
            img = ImageOps.mirror(img)
            x0, y0, x1, y1 = bbox
            bbox = [w - x1, y0, w - x0, y1]

        instance_data["pixel_values"] = img
        instance_data["face_bbox"] = bbox
        if self.origtarget_genref:
            ref_images, ref_bbox = self._load_similar_ref_images_and_bbox(img_data, target_path=None)
            instance_data["face_bbox_ref"] = deepcopy(ref_bbox)
        elif self.ref_similar:
            ref_images, ref_bbox = self._load_similar_ref_images_and_bbox(img_data, target_path=path)
            instance_data["face_bbox_ref"] = deepcopy(ref_bbox)
        elif self.const_ref:
            ref_path = self.parent_image_by_path.get(path)
            if ref_path is None:
                raise ValueError(f"Missing parent image path for cosmic_large sample: {path}")
            ref_img, ref_bbox = self._load_constant_ref_image_and_bbox(ref_path, img_data)
            ref_images = [ref_img]
            instance_data["face_bbox_ref"] = deepcopy(ref_bbox)
        elif self.train_on_separate_image:
            ref_candidates = self._get_same_id_ref_candidates(path)
            if not ref_candidates:
                raise ValueError(
                    "train_on_separate_image=True for CosmicLargeTrain requires "
                    f"at least two images for identity '{self.identity_by_path.get(path, path)}'"
                )
            ref_path = random.choice(ref_candidates)
            ref_data = self.meta_by_path[ref_path]
            ref_img, ref_bbox = self._load_train_image_and_bbox(ref_path, ref_data)
            ref_images = [ref_img]
            instance_data["face_bbox_ref"] = deepcopy(ref_bbox)
        else:
            instance_data["face_bbox_ref"] = deepcopy(bbox)
            ref_images = [deepcopy(img)]

        instance_data["ref_images"] = ref_images

        prompt = self._build_prompt(img_data)
        instance_data["prompts"] = prompt
        instance_data["prompt"] = prompt

        instance_data["original_sizes"] = (self.train_image_size, self.train_image_size)
        instance_data["crop_top_lefts"] = (0, 0)

        instance_data = self.preprocess_data(instance_data)

        assert min(instance_data["face_bbox"]) >= 0
        assert max(instance_data["face_bbox"]) <= self.train_image_size
        assert min(instance_data["face_bbox_ref"]) >= 0
        assert max(instance_data["face_bbox_ref"]) <= self.train_image_size

        return instance_data


### MODIFIED ###
import torch
### MODIFIED ###


### MODIFIED ###
class CosmicLargeTrain(BaseDataset):
    def __init__(
        self,
        data_json_pth=None,
        data_json_path=None,
        images_path=None,
        target_images_path=None,
        num_refs=1,
        # min_face_res=64,
        min_face_res=192,
        target_crop_256=False,
        embeds_path=None,
        use_embeds=False,
        only_complex_background=False,
        path_prefix_to_strip=None,
        require_nested_identity_subdir=True,
        upscale_to_1024=False,
        const_ref=False,
        crop_ref=False,
        ref_similar=True,
        origtarget_genref=True,
        crop_nonface_min=0.2,
        crop_nonface_max=0.4,
        ref_crop_margin_min=0.2,
        ref_crop_margin_max=None,
        ref_downscale_jitter=0.0,
        train_on_separate_image=False,
        same_id_ref_map_json_pth=None,
        *args,
        **kwargs,
    ):
        del (
            require_nested_identity_subdir,
            upscale_to_1024,
            const_ref,
            crop_ref,
            ref_similar,
            origtarget_genref,
            train_on_separate_image,
            same_id_ref_map_json_pth,
        )

        data_json_path = data_json_path or data_json_pth
        if data_json_path is None:
            raise ValueError("CosmicLargeTrain requires data_json_pth or data_json_path")

        self.num_refs = num_refs
        self.images_path = images_path
        self.target_images_path = target_images_path
        self.path_prefix_to_strip = path_prefix_to_strip.strip("/") if path_prefix_to_strip else None
        self.use_embeds = use_embeds
        self.embeds = torch.load(embeds_path, weights_only=False) if embeds_path is not None else None
        self.target_crop_256 = bool(target_crop_256)
        self.train_image_size = 256 if self.target_crop_256 else 1024
        self.crop_nonface_min = float(crop_nonface_min)
        self.crop_nonface_max = float(crop_nonface_max)
        if (
            self.crop_nonface_min < 0
            or self.crop_nonface_max < self.crop_nonface_min
            or self.crop_nonface_max >= 1
        ):
            raise ValueError(
                "CosmicLargeTrain requires 0 <= crop_nonface_min <= crop_nonface_max < 1; "
                f"got {self.crop_nonface_min} and {self.crop_nonface_max}"
            )

        # Reference-crop augmentation (defaults keep the legacy fixed +20% margin crop).
        self.ref_crop_margin_min = float(ref_crop_margin_min)
        self.ref_crop_margin_max = float(
            ref_crop_margin_max if ref_crop_margin_max is not None else ref_crop_margin_min
        )
        if self.ref_crop_margin_max < self.ref_crop_margin_min or self.ref_crop_margin_min < 0:
            raise ValueError(
                "CosmicLargeTrain requires 0 <= ref_crop_margin_min <= ref_crop_margin_max; "
                f"got {self.ref_crop_margin_min} and {self.ref_crop_margin_max}"
            )
        self.ref_downscale_jitter = float(ref_downscale_jitter)
        if not 0.0 <= self.ref_downscale_jitter <= 1.0:
            raise ValueError(
                f"ref_downscale_jitter must be in [0, 1], got {self.ref_downscale_jitter}"
            )

        images_root = Path(self.images_path) if self.images_path is not None else None
        self.dataset_root = images_root.parents[1] if images_root is not None and len(images_root.parents) >= 2 else images_root
        self.target_images_root = Path(self.target_images_path) if self.target_images_path is not None else None

        with open(data_json_path) as f:
            data = json.load(f)

        index = []
        for img_path in list(data.keys()):
            img_data = data[img_path]
            bbox = img_data["face_crop_new"]
            if (min(bbox[2] - bbox[0], bbox[3] - bbox[1])) < min_face_res:
                continue
            if only_complex_background and (
                img_data.get("has_simple_back", False) or img_data.get("is_simp", False)
            ):
                continue
            if len(img_data.get("face_paths") or []) < 1:
                continue
            if self.embeds is not None and img_path not in self.embeds:
                continue

            img_data["image_path"] = img_path
            index.append(img_data)

        if self.target_crop_256:
            kwargs = dict(kwargs)
            kwargs["instance_transforms"] = self._remove_pixel_resize_from_instance_transforms(
                kwargs.get("instance_transforms")
            )

        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _remove_resize_from_transform(transform):
        if transform is None or not hasattr(transform, "transforms"):
            return transform

        kept_transforms = []
        removed_resize = False
        for item in transform.transforms:
            if item.__class__.__name__ == "Resize":
                removed_resize = True
                continue
            kept_transforms.append(item)

        if not removed_resize:
            return transform
        logger.info("CosmicLargeTrain target_crop_256=True: removed pixel_values Resize transform")
        return transform.__class__(kept_transforms)

    @classmethod
    def _remove_pixel_resize_from_instance_transforms(cls, instance_transforms):
        if instance_transforms is None:
            return None
        instance_transforms = dict(instance_transforms)
        if "pixel_values" in instance_transforms:
            instance_transforms["pixel_values"] = cls._remove_resize_from_transform(
                instance_transforms["pixel_values"]
            )
        return instance_transforms

    def _get_relative_path(self, path):
        path = str(path)
        if self.images_path:
            images_path = str(self.images_path).rstrip("/")
            prefix = f"{images_path}/"
            if path.startswith(prefix):
                return path[len(prefix):]

        path = path.lstrip("/")
        if self.path_prefix_to_strip:
            prefix = f"{self.path_prefix_to_strip}/"
            if path.startswith(prefix):
                return path[len(prefix):]
        if self.images_path:
            root_name = Path(self.images_path).name
            marker = f"/{root_name}/"
            marked_path = f"/{path}"
            if marker in marked_path:
                return marked_path.split(marker, 1)[1]
        return path

    @staticmethod
    def _first_existing_path(candidates):
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate
        return next(candidate for candidate in candidates if candidate is not None)

    def _target_full_path(self, path):
        path_obj = Path(str(path))
        if path_obj.is_absolute():
            return path_obj

        candidates = []
        if self.target_images_root is not None:
            candidates.append(self.target_images_root / self._get_relative_path(path))
        if self.dataset_root is not None:
            candidates.append(self.dataset_root / str(path).lstrip("/"))
        if self.images_path is not None:
            candidates.append(Path(self.images_path) / self._get_relative_path(path))
        return self._first_existing_path(candidates)

    def _face_full_path(self, path):
        path_obj = Path(str(path))
        if path_obj.is_absolute():
            return path_obj

        candidates = []
        if self.images_path is not None:
            candidates.append(Path(self.images_path) / self._get_relative_path(path))
            candidates.append(Path(self.images_path) / str(path).lstrip("/"))
        if self.dataset_root is not None:
            candidates.append(self.dataset_root / str(path).lstrip("/"))
        return self._first_existing_path(candidates)

    def _mask_full_path(self, path):
        path_obj = Path(str(path))
        if path_obj.is_absolute():
            return path_obj

        candidates = []
        if self.dataset_root is not None:
            candidates.append(self.dataset_root / str(path).lstrip("/"))
        if self.images_path is not None:
            candidates.append(Path(self.images_path) / self._get_relative_path(path))
            candidates.append(Path(self.images_path) / str(path).lstrip("/"))
        return self._first_existing_path(candidates)

    def _load_train_image(self, path, img_data):
        img = Image.open(self._target_full_path(path)).convert("RGB")
        if img.size != (1024, 1024):
            body_crop = img_data["body_crop"]
            img_arr = np.array(img)[body_crop[1]:body_crop[3], body_crop[0]:body_crop[2]]
            assert img_arr.shape[0] == 1024, img_arr
            assert img_arr.shape[1] == 1024, img_arr
            img = Image.fromarray(img_arr)
        assert img.size == (1024, 1024)
        return img

    def _load_body_mask(self, img_data):
        if "body_mask_path" not in img_data:
            return None
        body_mask_pth = self._mask_full_path(img_data["body_mask_path"])
        if not body_mask_pth.exists():
            return None
        body_mask = Image.open(body_mask_pth).convert("1").resize((32, 32))
        body_mask = torch.from_numpy(np.array(body_mask)).bool()
        assert body_mask.long().sum() > 0, body_mask_pth
        return body_mask

    @staticmethod
    def _clip_bbox_to_image(bbox, img_size):
        img_w, img_h = img_size
        x0, y0, x1, y1 = [float(v) for v in bbox]
        clipped_bbox = [
            max(0.0, min(float(img_w), x0)),
            max(0.0, min(float(img_h), y0)),
            max(0.0, min(float(img_w), x1)),
            max(0.0, min(float(img_h), y1)),
        ]
        if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
            return None
        return clipped_bbox

    @staticmethod
    def _scale_bbox_to_size(bbox, src_size, dst_size):
        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        scale_x = dst_w / max(float(src_w), 1.0)
        scale_y = dst_h / max(float(src_h), 1.0)
        return [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y,
        ]

    @staticmethod
    def _clamp_int(value, min_value, max_value):
        return int(max(min_value, min(max_value, value)))

    def _crop_target_256_around_bbox(self, img, face_bbox):
        target_side = self.train_image_size
        face_bbox = self._clip_bbox_to_image(face_bbox, img.size)
        if face_bbox is None:
            raise ValueError("Invalid target face bbox before 256 crop")

        x0, y0, x1, y1 = face_bbox
        face_w = max(x1 - x0, 1.0)
        face_h = max(y1 - y0, 1.0)
        context_ratio_w = random.uniform(self.crop_nonface_min, self.crop_nonface_max)
        context_ratio_h = random.uniform(self.crop_nonface_min, self.crop_nonface_max)
        resize_scale = min(
            1.0,
            target_side / (face_w * (1.0 + context_ratio_w)),
            target_side / (face_h * (1.0 + context_ratio_h)),
        )
        if resize_scale < 1.0:
            resized_size = (
                max(1, int(round(img.size[0] * resize_scale))),
                max(1, int(round(img.size[1] * resize_scale))),
            )
            img = img.resize(resized_size, Image.BICUBIC)
            face_bbox = [coord * resize_scale for coord in face_bbox]

        img_w, img_h = img.size
        if img_w < target_side or img_h < target_side:
            canvas_size = (max(target_side, img_w), max(target_side, img_h))
            pad_left = (canvas_size[0] - img_w) // 2
            pad_top = (canvas_size[1] - img_h) // 2
            canvas = Image.new("RGB", canvas_size)
            canvas.paste(img, (pad_left, pad_top))
            img = canvas
            face_bbox = [
                face_bbox[0] + pad_left,
                face_bbox[1] + pad_top,
                face_bbox[2] + pad_left,
                face_bbox[3] + pad_top,
            ]
            img_w, img_h = img.size

        x0, y0, x1, y1 = face_bbox
        left_min = max(0, int(np.ceil(x1 - target_side)))
        left_max = min(img_w - target_side, int(np.floor(x0)))
        top_min = max(0, int(np.ceil(y1 - target_side)))
        top_max = min(img_h - target_side, int(np.floor(y0)))

        if left_min <= left_max:
            crop_left = random.randint(left_min, left_max)
        else:
            crop_left = self._clamp_int(
                round(0.5 * (x0 + x1) - 0.5 * target_side),
                0,
                img_w - target_side,
            )
        if top_min <= top_max:
            crop_top = random.randint(top_min, top_max)
        else:
            crop_top = self._clamp_int(
                round(0.5 * (y0 + y1) - 0.5 * target_side),
                0,
                img_h - target_side,
            )

        cropped_img = img.crop((crop_left, crop_top, crop_left + target_side, crop_top + target_side))
        cropped_bbox = [
            x0 - crop_left,
            y0 - crop_top,
            x1 - crop_left,
            y1 - crop_top,
        ]
        cropped_bbox = self._clip_bbox_to_image(cropped_bbox, cropped_img.size)
        if cropped_bbox is None:
            raise ValueError("Invalid target face bbox after 256 crop")

        return cropped_img, cropped_bbox

    @staticmethod
    def _get_bigger_crop_with_bbox(img, face_bbox, scale=0.2):
        crop = [int(round(v)) for v in deepcopy(face_bbox)]
        if crop[3] - crop[1] < crop[2] - crop[0]:
            diff = crop[2] - crop[0] - (crop[3] - crop[1])
            if diff % 2 != 0:
                crop[0] -= 1
                diff += 1
            crop[3] += diff // 2
            crop[1] -= diff // 2
        elif crop[2] - crop[0] < crop[3] - crop[1]:
            diff = crop[3] - crop[1] - (crop[2] - crop[0])
            if diff % 2 != 0:
                crop[1] -= 1
                diff += 1
            crop[2] += diff // 2
            crop[0] -= diff // 2

        assert crop[3] - crop[1] == crop[2] - crop[0], crop

        to_add = int((crop[3] - crop[1]) * scale)
        img_w, img_h = img.size
        crop = [
            max(0, crop[0] - to_add),
            max(0, crop[1] - to_add),
            min(img_w, crop[2] + to_add),
            min(img_h, crop[3] + to_add),
        ]
        cropped_img = img.crop((crop[0], crop[1], crop[2], crop[3]))
        cropped_bbox = [
            face_bbox[0] - crop[0],
            face_bbox[1] - crop[1],
            face_bbox[2] - crop[0],
            face_bbox[3] - crop[1],
        ]
        return cropped_img, CosmicLargeTrain._clip_bbox_to_image(cropped_bbox, cropped_img.size)

    def _get_face_bbox(self, img_data, face_path):
        face_bboxes = img_data["face_bboxes"]
        candidates = [
            str(face_path),
            str(face_path).lstrip("/"),
            self._get_relative_path(face_path),
            self._get_relative_path(face_path).lstrip("/"),
        ]
        for candidate in candidates:
            if candidate in face_bboxes:
                return face_bboxes[candidate]
        raise KeyError(f"Missing face bbox for reference image: {face_path}")

    def get_ref_image(self, img_data):
        face_path = np.random.choice(img_data["face_paths"])
        if self.use_embeds:
            return self.embeds[str(face_path)], self._get_face_bbox(img_data, face_path)

        ref_img = Image.open(self._face_full_path(face_path)).convert("RGB")
        face_bbox = self._get_face_bbox(img_data, face_path)
        ref_margin = random.uniform(self.ref_crop_margin_min, self.ref_crop_margin_max)
        ref_face, ref_bbox = self._get_bigger_crop_with_bbox(ref_img, face_bbox, scale=ref_margin)
        if ref_bbox is None:
            raise ValueError(f"Invalid reference face bbox after crop: {face_path}")
        # Sharpness jitter: occasional downscale + re-upscale so the branches see a
        # range of ref sharpness instead of a single crop/blur style. Bboxes are
        # unaffected (final size unchanged).
        if self.ref_downscale_jitter > 0 and random.random() < self.ref_downscale_jitter:
            w0, h0 = ref_face.size
            factor = random.uniform(0.5, 0.9)
            small = ref_face.resize(
                (max(8, int(w0 * factor)), max(8, int(h0 * factor))), Image.BILINEAR
            )
            ref_face = small.resize((w0, h0), Image.BILINEAR)
        if random.random() < 0.5:
            w, _ = ref_face.size
            ref_face = ImageOps.mirror(ref_face)
            x0, y0, x1, y1 = ref_bbox
            ref_bbox = [w - x1, y0, w - x0, y1]
        return ref_face, ref_bbox

    def get_face_mask_from_bbox(self, bbox):
        scale = max(float(self.train_image_size) / 32.0, 1.0)
        scaled_box = [
            int(bbox[0] // scale),
            int(bbox[1] // scale),
            int(bbox[2] // scale),
            int(bbox[3] // scale),
        ]

        hor = scaled_box[3] - scaled_box[1]
        add = int(hor * 0.25)
        scaled_box[1] = scaled_box[1] - add
        scaled_box[3] = scaled_box[3] + add

        vert = scaled_box[2] - scaled_box[0]
        add = int(vert * 0.3)
        scaled_box[0] = scaled_box[0] - add
        scaled_box[2] = scaled_box[2] + add

        scaled_box = np.clip(scaled_box, 0, 32)

        mask = torch.zeros(32, 32, dtype=torch.bool)
        mask[scaled_box[1]:scaled_box[3], scaled_box[0]:scaled_box[2]] = True
        return mask

    @staticmethod
    def _build_prompt(img_data):
        return ", ".join(
            [
                img_data["facial_caption"],
                img_data["pose_caption"],
                img_data["background_caption"],
            ]
        )

    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = img_data["image_path"]

        instance_data = {}

        img = self._load_train_image(path, img_data)
        bbox = deepcopy(img_data["face_crop_new"])
        if self.target_crop_256:
            img, bbox = self._crop_target_256_around_bbox(img, bbox)
            body_mask = None
        else:
            body_mask = self._load_body_mask(img_data)

        if body_mask is not None:
            instance_data["body_mask"] = body_mask
        instance_data["pixel_values"] = deepcopy(img)
        instance_data["facial_caption"] = img_data["facial_caption"]
        instance_data["pose_caption"] = img_data["pose_caption"]
        instance_data["background_caption"] = img_data["background_caption"]

        prompt = self._build_prompt(img_data)
        instance_data["prompts"] = prompt
        instance_data["prompt"] = prompt
        instance_data["identity_id"] = str(path)

        instance_data["bbox"] = bbox
        instance_data["face_bbox"] = deepcopy(bbox)

        ref_image, ref_bbox = self.get_ref_image(img_data)
        instance_data["ref_images"] = [ref_image]
        instance_data["face_bbox_ref"] = deepcopy(ref_bbox)


        if self.target_crop_256:
            instance_data["original_sizes"] = (self.train_image_size, self.train_image_size)
            instance_data["crop_top_lefts"] = (0, 0)
        elif "orig_size" in img_data:
            orig_size = img_data["orig_size"]
            instance_data["original_sizes"] = (orig_size[1], orig_size[0])
            instance_data["crop_top_lefts"] = get_crop_values(img_data)
        else:
            instance_data["original_sizes"] = (1024, 1024)
            instance_data["crop_top_lefts"] = (0, 0)

        face_mask = self.get_face_mask_from_bbox(instance_data["bbox"])
        instance_data["face_mask"] = face_mask
        assert face_mask.any(), path

        instance_data["target_sizes"] = (self.train_image_size, self.train_image_size)

        instance_data = self.preprocess_data(instance_data)

        assert min(instance_data["bbox"]) >= 0
        assert max(instance_data["bbox"]) <= self.train_image_size
        assert min(instance_data["face_bbox"]) >= 0
        assert max(instance_data["face_bbox"]) <= self.train_image_size
        assert min(instance_data["face_bbox_ref"]) >= 0

        return instance_data
### MODIFIED ###
