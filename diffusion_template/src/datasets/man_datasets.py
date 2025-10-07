from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import get_bigger_crop
from copy import deepcopy
import json


class ManValDataset(BaseDataset):
    def __init__(self, data_json_pth=None, images_path = None, *args, **kwargs):

        self.images_path = images_path

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = {
            'nm0000609': [1, 2, 3, 5, 7], 
            'nm0000334': [0, 1, 2, 3, 5], 
            'nm0186505': [0, 1, 2, 3, 4, 7], 
            'nm0003683': [0, 3, 4, 5], 
            'nm0005452': [0, 1, 4, 5, 8], 
            'dai_yun_fan': [0, 1, 2, 3, 5, 6], 
            'jang_ki_yong': [0, 1, 2, 3, 4],  
            'nm0000563': [0, 1, 2, 4, 5, 6],
            'nm0511088': [0, 2, 3, 4, 5, 6, 9]
        }
        prompts = [
            {
                "facial_caption": "man img, mohawk purple hairstyle, screaming, looks up",
                "pose_caption": "full body view, playing red electro guitar, singing, wears black leather jacket",
                "background_caption": "on rock concert scene, blue spotlights, burning torches behind",
            },
            {
                "facial_caption": "man img with beard and sideburns, brown cowboy hat, green eyes, laughing",
                "pose_caption": "riding a horse, full body view, wears white suit over a black shirt, shows thumb up",
                "background_caption": "on a black horse with brown saddle, surounded by blue flowers, old windmill behind",
            },
            {
                "facial_caption": "man img, bald, surprised, green headband, freckles",
                "pose_caption": "surfing, full body view, spread arms to the sides, wears blue shorts and pink t-shirt",
                "background_caption": "on the wave, sunset, island with palm trees behind",
            },
            {
                "facial_caption": "man img, undercut silver hair, smiling, thin wire-frame glasses, mustache",
                "pose_caption": "full body view, conducting an orchestra, wearing a black tailcoat with crimson bow tie, left arm raised",
                "background_caption": "in a grand concert hall with gilded balconies, velvet curtains, and crystal chandeliers, dim stage lighting",
            },
            {
                "facial_caption": "man img, goatee, wry smile, freckled cheeks, black afro hairstyle",
                "pose_caption": "full body view, skateboarding down a ramp, wearing ripped denim jeans and white sneakers, mid-air trick",
                "background_caption": "in 1990s-style skate park with graffiti-covered concrete bowls and palm trees, midday haze, skyscrapers behind",
            }
        ]

        index = self.create_index(prompts, data_json)
        super().__init__(index, *args, **kwargs)

    def create_index(self, prompts, data_json):
        index = []
        for id in self.ids:
            references = []
            all_id_imges = data_json[id]
            img_nums = sorted(list(all_id_imges.keys()))
            for img_ind in self.ids[id]:
                ref_img_num = img_nums[img_ind]

                ref_img = self.load_object(f"{self.images_path}/{id}/{ref_img_num}.jpg")
                ref_img_data = deepcopy(data_json[id][ref_img_num])
                ref_crop = ref_img_data['new_face_crop']
                ref_face_img = get_bigger_crop(deepcopy(ref_img), crop=deepcopy(ref_crop))
                references.append(ref_face_img)

            for prompt_num, prompt in enumerate(prompts):
                item = {}
                item["pixel_values"] = ref_img
                item["ref_images"] = deepcopy(references)
                item["prompt"] = self.get_prompt(prompt)

                item["original_sizes"] = (1024, 1024)
                item["crop_top_lefts"] = (0, 0)
                item["id"] = id
                index.append(item)
        return index

    def __getitem__(self, ind):
        img_dict = self._index[ind]
        instance_data = self.preprocess_data(img_dict)
        return instance_data

    def get_prompt(self, prompt):
        return ", ".join([prompt["facial_caption"], prompt["pose_caption"], prompt["background_caption"]])


class ManValFaceDataset(ManValDataset):
    def get_prompt(self, prompt):
        return prompt["facial_caption"]