from copy import deepcopy
import json
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
        *args,
        **kwargs,
    ):
        self.images_path = images_path

        # Init cosmic
        with open(cosmic_json_pth) as f:
            cosmic_json = json.load(f)

        with open(cosmic_texts_json_pth) as f:
            cosmic_texts_json = json.load(f)

        self.num_refs = num_refs

        index = []
        self.ids = []
        for k, v in tqdm(cosmic_json.items()):
            if k not in cosmic_texts_json:
                continue
            v.update(cosmic_texts_json[k])
            index.append(v)
            self.ids.append(k)

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
            self.ids.append(k.replace("LAION-5B", "LAION-5B-Filtered-Large"))

        # self.flip = RandomHorizontalFlip(p=0.5)

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = self.ids[ind]

        instance_data = {}

        img = Image.open(f"{self.images_path}/{path}").convert("RGB")
        if img.size != (1024, 1024):
            body_crop = img_data["body_crop"]
            img_arr = np.array(img)[body_crop[1]:body_crop[3], body_crop[0]:body_crop[2]]
            assert img_arr.shape[0] == 1024, img_arr
            assert img_arr.shape[1] == 1024, img_arr
            img = Image.fromarray(img_arr)

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

        # ref_images = [get_bigger_crop(deepcopy(img), crop=deepcopy(bbox))]
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
        *args,
        **kwargs,
    ):
        self.images_path = images_path
        self.num_refs = num_refs

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

        # ref_images = [get_bigger_crop(deepcopy(img), crop=deepcopy(bbox))]
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
