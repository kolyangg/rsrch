from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import get_bigger_crop, get_crop_values


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


class CosmicLargeTrain(BaseDataset):
    def __init__(
        self,
        data_json_pth=None,
        images_path=None,
        num_refs=1,
        train_on_separate_image=False,
        same_id_ref_map_json_pth=None,
        path_prefix_to_strip=None,
        require_nested_identity_subdir=True,
        *args,
        **kwargs,
    ):
        self.images_path = images_path
        self.num_refs = num_refs
        self.train_on_separate_image = bool(train_on_separate_image)
        self.path_prefix_to_strip = path_prefix_to_strip.strip("/") if path_prefix_to_strip else None
        self.require_nested_identity_subdir = bool(require_nested_identity_subdir)

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = []
        self.meta_by_path = {}
        self.identity_by_path = {}
        self.same_id_ref_map = {}

        index = []
        for path, image_data in tqdm(data_json.items()):
            if not isinstance(image_data, dict):
                continue

            bbox = image_data.get("face_crop_new")
            if bbox is None or min(bbox) < 0 or max(bbox) > 1024:
                continue

            paths = [str(path)]
            if self.require_nested_identity_subdir:
                face_paths = image_data.get("face_paths")
                if isinstance(face_paths, list):
                    paths = [str(face_path) for face_path in face_paths]

            for sample_path in paths:
                rel_path = self._get_relative_path(sample_path)
                if self.require_nested_identity_subdir and len(Path(rel_path).parts) != 3:
                    continue
                identity = str(Path(rel_path).parent)

                index.append(image_data)
                self.ids.append(sample_path)
                self.meta_by_path[sample_path] = image_data
                self.identity_by_path[sample_path] = identity
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
        if img.size != (1024, 1024):
            body_crop = img_data.get("body_crop")
            if body_crop is not None and len(body_crop) == 4:
                x0, y0, x1, y1 = body_crop
                if 0 <= x0 < x1 <= img.size[0] and 0 <= y0 < y1 <= img.size[1]:
                    img = Image.fromarray(np.array(img)[y0:y1, x0:x1])
            if img.size != (1024, 1024):
                img = img.resize((1024, 1024), Image.BICUBIC)
        return img

    def _get_same_id_ref_candidates(self, path):
        identity = self.identity_by_path.get(path)
        if identity is not None and identity in self.same_id_ref_map:
            return [p for p in self.same_id_ref_map[identity] if p != path]
        return [p for p in self.same_id_ref_map.get(path, []) if p != path]

    @staticmethod
    def _build_prompt(img_data):
        text = img_data.get("text", "")
        if isinstance(text, str) and text:
            return text

        prompt_parts = [
            img_data.get("facial_caption"),
            img_data.get("pose_caption"),
            img_data.get("background_caption"),
        ]
        prompt = ", ".join(part for part in prompt_parts if isinstance(part, str) and part)
        return prompt or "img person"

    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = self.ids[ind]

        instance_data = {}

        img = self._load_train_image(path, img_data)

        bbox = deepcopy(img_data["face_crop_new"])
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
                    "train_on_separate_image=True for CosmicLargeTrain requires "
                    f"at least two images for identity '{self.identity_by_path.get(path, path)}'"
                )
            ref_path = random.choice(ref_candidates)
            ref_data = self.meta_by_path[ref_path]
            ref_img = self._load_train_image(ref_path, ref_data)
            ref_images = [ref_img]
            instance_data["face_bbox_ref"] = deepcopy(ref_data["face_crop_new"])
        else:
            instance_data["face_bbox_ref"] = deepcopy(bbox)
            ref_images = [deepcopy(img)]

        instance_data["ref_images"] = ref_images

        prompt = self._build_prompt(img_data)
        instance_data["prompts"] = prompt
        instance_data["prompt"] = prompt

        instance_data["original_sizes"] = (1024, 1024)
        instance_data["crop_top_lefts"] = (0, 0)

        instance_data = self.preprocess_data(instance_data)

        assert min(instance_data["face_bbox"]) >= 0
        assert max(instance_data["face_bbox"]) <= 1024

        return instance_data
