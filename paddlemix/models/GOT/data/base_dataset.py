# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import copy
# import io
# import json
import logging

# from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing import Dict

import paddle
import paddlenlp
from paddle.io import Dataset
from PIL import ImageFile  # , Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
# from ..utils.constants import *


class BaseDataset(Dataset):
    def __init__(self, datasets: str, tokenizer: paddlenlp.transformers.PretrainedTokenizer, multimodal_cfg: dict):
        super(BaseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

        logging.warning(f"Using {multimodal_cfg['image_token_len']} tokens for representing image")

    def image_processor(self, image):
        # processor = self.multimodal_cfg['image_processor']  # the first processor, usually is the clip pretrained model (vit)
        processor_high = self.multimodal_cfg[
            "image_processor_high"
        ]  # the second processor, usually is the designed image encoder (sam/swin/cnn)
        image_high = image.copy()

        #  Vary old codes

        # # TODO the 'keep', 'padding' only used for the first processor
        # if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
        #     max_hw, min_hw = max(image.size), min(image.size)
        #     aspect_ratio = max_hw / min_hw
        #     max_len, min_len = 448, 224
        #     shortest_edge = int(min(max_len / aspect_ratio, min_len))
        #     image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
        # elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':
        #     def expand2square(pil_img, background_color):
        #         width, height = pil_img.size
        #         if width == height:
        #             return pil_img
        #         elif width > height:
        #             result = Image.new(pil_img.mode, (width, width), background_color)
        #             result.paste(pil_img) # for simpler box processing
        #             return result
        #         else:
        #             result = Image.new(pil_img.mode, (height, height), background_color)
        #             result.paste(pil_img) # for simpler box processing
        #             return result
        #     image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        #     image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": 224})['pixel_values'][0]
        # else:
        #     image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        image_high = processor_high(image_high)

        return image_high

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, paddle.Tensor]:
        pass
