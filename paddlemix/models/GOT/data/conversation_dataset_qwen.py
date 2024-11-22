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

import copy

# import io
import json
import logging

# import os
import random

# from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing import Dict

import paddle
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from megfile import smart_glob
from natsort import natsorted

# from ..utils.constants import CONVERSATION_DATA
from ..utils.conversation import (  # conv_templates,; default_conversation,
    SeparatorStyle,
    conv_mpt,
)
from .base_dataset import BaseDataset

IGNORE_INDEX = -100
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"

DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"


class ConversationDataset(BaseDataset):
    """Conversation format dataset stage2 fine-tuning."""

    def __init__(self, meta_path, tokenizer, multimodal_cfg):
        super(ConversationDataset, self).__init__(meta_path, tokenizer, multimodal_cfg)
        # v0 version format conversation
        # default_conversation = conv_templates["mpt"]
        logging.warning("Formatting inputs into conversation type: mpt-fixed")
        logging.warning("Loading data...")

        list_data_dict = []
        list_image_path = []

        # add your data  [data1, data2, data3, .....]
        # got_data_dict = {
        #     "pdf-ocr": ["data1"],
        #     #'scene-ocr': ["data3", "data4"]
        #     # ......
        # }
        # for name_all in datasets.split("+"):
        #    for name in got_data_dict[name_all]:
        ds_collections = json.loads(open(meta_path).read())
        for ds_idx, ds_name in enumerate(ds_collections.keys()):
            if 1:
                # dataset = CONVERSATION_DATA[ds_name]
                dataset = ds_collections[ds_name]

                data_path = dataset["annotations"]
                if data_path.endswith(".json"):
                    data = json.load(open(data_path, "r"))
                elif data_path.endswith(".jsonl"):
                    with open(data_path, "r") as f:
                        data = f.readlines()
                        for ii in range(len(data)):
                            data[ii] = json.loads(data[ii])
                else:
                    raise ValueError(f"Unknown file extension: {data_path}")

                list_data_dict.extend(data)

                image_path = dataset["images"]  # image_root

                list_image_path.extend([image_path] * len(data))

                logging.warning(f"Data from {data_path} provide {len(data)} conversations.")

        assert len(list_data_dict) == len(list_image_path)
        logging.warning(f"{len(list_data_dict)} conversations in total.")
        a_new_list = list(zip(list_data_dict, list_image_path))
        random.shuffle(a_new_list)
        list_data_dict_new, list_image_path_new = zip(*a_new_list)
        self.list_data_dict = list_data_dict_new
        self.list_image_path = list_image_path_new

        self.im_patch_token = 151859

        self.im_start_token = 151857

        self.im_end_token = 151858

    def multimodal_processor(self, sources, flag_num_patches):
        for source in sources:
            if self.multimodal_cfg["sep_image_conv_front"]:
                assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
                source[0]["value"] = source[0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                source[0]["value"] = DEFAULT_IMAGE_TOKEN + conv_mpt.sep + conv_mpt.roles[0] + ": " + source[0]["value"]

            for sentence in source:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg["image_token_len"] * flag_num_patches
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                # sentence["value"] = str(sentence["value"]).replace('\qquad', '\quad')
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources

    def _tokenize_fn(self, strings):
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pd",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.not_equal(paddle.to_tensor(self.tokenizer.pad_token_id)).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def _mask_targets(self, target, tokenized_lens, speakers):
        # cur_idx = 0
        cur_idx = tokenized_lens[0]
        tokenized_lens = tokenized_lens[1:]
        target[:cur_idx] = IGNORE_INDEX
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker.lower() == "human":
                target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len

    def token_processor(self, sources, image_name):
        conv = conv_mpt.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pd",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        targets = input_ids.clone()
        assert conv.sep_style == SeparatorStyle.MPT

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.not_equal(paddle.to_tensor(self.tokenizer.pad_token_id)).sum())

            rounds = conversation.split(conv.sep)
            re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids) + len(self.tokenizer(conv.sep).input_ids)
                # round_len = len(tokenizer_image_token(rou, self.tokenizer)) + len(tokenizer_image_token(conv.sep, self.tokenizer))
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                instruction_len = len(self.tokenizer(parts[0]).input_ids)
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")
                    print(image_name)

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def __getitem__(self, i) -> Dict[str, paddle.Tensor]:
        # data = self.list_data_dict[i]
        data = copy.deepcopy(self.list_data_dict[i])

        if isinstance(data, dict):
            image_list = []
            image_high_list = []
            flag_num_patches = 1
            if "image" in data:
                image_path = self.list_image_path[i]
                image_file = data["image"]

                # multi-crop or multi page, only support .png files
                if (
                    0
                ):  # ('.jpg' not in image_file and '.png' not in image_file and '.jpeg' not in image_file) and ('.jpg' not in image_path and '.png' not in image_path and '.jpeg' not in image_path):
                    if image_file[0] == "/":
                        patch_dir = image_path[:-1] + image_file
                        patches = smart_glob(patch_dir + "*.png")
                    else:
                        patch_dir = image_path + image_file
                        patches = smart_glob(patch_dir + "*.png")

                    # print(patches)
                    if not patches:
                        print(f"cannot glob the dir {patch_dir}.")
                        return self.__getitem__(0)

                    # sort multi images by name
                    patches = natsorted(patches)
                    flag_num_patches = len(patches)

                    for patch in patches:
                        try:
                            image = Image.open(patch).convert("RGB")
                        except:
                            print(f"cannot identify image file {patch}.")
                            return self.__getitem__(0)

                        try:
                            img = self.image_processor(image)
                            image_list.append(img)
                            image_high_list.append(img)
                        except:
                            print(
                                f"image {image_path + image_file + patch} are broken or grayscale! we thus select 0-th sample instead!"
                            )
                            return self.__getitem__(0)

                else:
                    flag_num_patches = 1
                    try:
                        image = Image.open(image_path + image_file).convert("RGB")
                    except:
                        print(f"cannot identify image file {image_file}.")
                        return self.__getitem__(0)

                    try:
                        image = self.image_processor(image)
                    except:
                        print(f"image {image_file} are broken or grayscale! we thus select 0-th sample instead!")
                        return self.__getitem__(0)

            conversations = self.multimodal_processor([data["conversations"]], flag_num_patches)
            # print(conversations)
            # exit()
        else:
            conversations = [data]

        # align with fastchat & llava here, put the conversation into a list for tokenization
        image_name = image_path + image_file
        data_dict = self.token_processor(conversations, image_name)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if isinstance(data, dict) and "image" in data:
            if image_list and image_high_list:
                data_dict["image"] = image_list
                data_dict["image_high"] = image_high_list
            else:
                data_dict["image"] = [image]
                data_dict["image_high"] = [image]
        else:
            # crop_size = self.multimodal_cfg['image_processor'].crop_size
            # data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            # Vary for two image, GOT does not use the data_dict['image]
            data_dict["image"] = [paddle.zeros([3, 1024, 1024])]
            data_dict["image_high"] = [paddle.zeros([3, 1024, 1024])]
        return data_dict