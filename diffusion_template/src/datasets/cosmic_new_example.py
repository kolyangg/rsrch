from src.datasets.base_dataset import BaseDataset
import torch
import json
from PIL import Image
import numpy as np
from copy import deepcopy
from src.datasets.data_utils import get_bigger_crop, get_crop_values
from torchvision.transforms import RandomHorizontalFlip


class CosmicBaseTrainDouble:
    def __init__(
        self, 
        data_json_path,
        data_large_json_path,
        embeds_path,
        embeds_path_large,
        images_path=None,
        num_refs=1, 
        *args, 
        **kwargs):
        self.dataset = CosmicBaseTrain(
            data_json_path,
            images_path=images_path,
            num_refs=num_refs,
            embeds_path=embeds_path,
            *args,
            **kwargs
        )
        self.dataset_large = CosmicBaseTrain(
            data_large_json_path,
            images_path=images_path,
            num_refs=num_refs,
            embeds_path=embeds_path_large,
            *args,
            **kwargs
        )
        print("Data len", len(self.dataset) + len(self.dataset_large))
    
    def __len__(self):
        return len(self.dataset) + len(self.dataset_large)

    def __getitem__(self, ind):
        if ind < len(self.dataset):
            return self.dataset[ind]
        return self.dataset_large[ind - len(self.dataset)]


class CosmicBaseTrain(BaseDataset):
    def __init__(
        self, 
        data_json_path,
        images_path=None,
        num_refs=1, 
        min_face_res=64,
        embeds_path=None,
        use_embeds=False,
        only_complex_background=False,
        *args, 
        **kwargs):
        
        self.num_refs = num_refs
        self.images_path = images_path
        self.use_embeds = use_embeds
        self.embeds = torch.load(embeds_path, weights_only=False)
        index = []
        self.ids = []
        
        with open(data_json_path) as f:
            data = json.load(f)
        
        for k in list(data.keys()):
            bbox = data[k]["face_crop_new"]
            if (min(bbox[2] - bbox[0], bbox[3] - bbox[1])) < min_face_res:
                data.pop(k)
            elif only_complex_background and data[k].get("has_simple_back", False):
                # print("Popping", k)
                data.pop(k)
        
        for img_path, img_data in data.items():
            if len(img_data["face_paths"]) >= num_refs and img_path in self.embeds:
                img_data["image_path"] = img_path
                index.append(img_data)
        
        self.flip = RandomHorizontalFlip(p=0.5)
        super().__init__(index, *args, **kwargs)
    
    def __getitem__(self, ind):
        img_data = self._index[ind]
        path = img_data["image_path"]
        
        instance_data = {}
        ref_images = []

        img = Image.open(f"{self.images_path}/{path}").convert("RGB")
        if img.size != (1024, 1024):
            body_crop = img_data["body_crop"]
            img_arr = np.array(img)[body_crop[1]:body_crop[3], body_crop[0]:body_crop[2]]
            assert img_arr.shape[0] == 1024, img_arr
            assert img_arr.shape[1] == 1024, img_arr
            img = Image.fromarray(img_arr)
        assert img.size == (1024, 1024)

        body_mask_pth = f"{self.images_path}/{img_data['body_mask_path']}"
        body_mask = Image.open(body_mask_pth).convert("1").resize((32, 32))
        body_mask = torch.from_numpy(np.array(body_mask)).bool()
        assert body_mask.long().sum() > 0, body_mask_pth
        
        instance_data["body_mask"] = body_mask

        instance_data["pixel_values"] =  deepcopy(img) 
        instance_data["facial_caption"] = img_data["facial_caption"]
        instance_data["pose_caption"] = img_data["pose_caption"]
        instance_data["background_caption"] = img_data["background_caption"]
        prompt = ", ".join([img_data["facial_caption"], img_data["pose_caption"], img_data["background_caption"]])
        instance_data["prompts"] = prompt
        
        instance_data["bbox"] = deepcopy(img_data['face_crop_new'])



        instance_data['ref_images'] = self.get_ref_images(img_data, instance_data)
        
        orig_size = img_data.get("orig_size", (1024, 1024))
        instance_data["original_sizes"] = (orig_size[1], orig_size[0])
        instance_data["crop_top_lefts"] = get_crop_values(img_data)
    
        face_mask = self.get_face_mask_from_bbox(instance_data["bbox"])
        instance_data["face_mask"] = face_mask
        assert face_mask.any(), path
        
        instance_data["target_sizes"] = (1024, 1024)
    
        instance_data = self.preprocess_data(instance_data)

        assert min(instance_data["bbox"]) >= 0
        assert max(instance_data["bbox"]) <= 1024

        return instance_data


    def get_ref_images(self, img_data, instance_data):
        ref_images = []
        ref_images_paths = np.random.choice(img_data["face_paths"], size=self.num_refs, replace=False)
        for face_path in ref_images_paths:
            if self.use_embeds:
                ref_embed = self.embeds[str(face_path)]
                ref_images.append(ref_embed)
            else:
                full_face_path = f"{self.images_path}/{face_path}"
                ref_img = Image.open(full_face_path)
                face_bbox = img_data["face_bboxes"][face_path]
                ref_face = self.flip(get_bigger_crop(ref_img, face_crop=face_bbox))
                ref_images.append(ref_face)
        return ref_images


    def get_face_mask_from_bbox(self, bbox):
        scaled_box = [int(bbox[0] // 32), int(bbox[1] // 32), int(bbox[2] // 32), int(bbox[3] // 32)]
        
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


### ARGS TO RUN
# args = {"data_large_json_path": "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/cosmic_data/gathered_data_cosmic_large_filtered2.json",
#         "data_json_path": "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/cosmic_data/gathered_data_cosmic_filtered2.json",
#         "images_path": "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/cosmic_data",
#         "num_refs": 1,
#         "instance_transforms": transforms,
#         "min_face_res": 96,
#         "embeds_path": "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/cosmic_data/cosmic_embeds.pt",
#         "embeds_path_large": "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/cosmic_data/cosmic_large_embeds.pt",
#         "use_embeds": False,
#         "only_complex_background": False}
# dataset = CosmicBaseTrainDouble(**args)
