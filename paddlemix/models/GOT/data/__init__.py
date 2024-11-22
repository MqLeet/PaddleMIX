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

from dataclasses import dataclass
from functools import partial
from sys import meta_path
from typing import List, Union

import paddle
import paddlenlp
from paddle import Tensor

from paddlemix.models.GOT.data.conversation_dataset_qwen import ConversationDataset

from ..utils.constants import *

IGNORE_INDEX = -100


# helpers
def pad_sequence_paddle(sequences, padding_value=0):
    """
    Implement a function similar to PyTorch's pad_sequence in PaddlePaddle.

    Args:
    - sequences (list of Tensor): The list of sequences to be padded.
    - padding_value (float, optional): The value used for padding, default is 0.

    Returns:
    - Tensor: The result of padding all sequences to the same length.
    """
    # Calculate the maximum length
    max_len = max([seq.shape[0] for seq in sequences])

    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        # Calculate the length to pad
        padding_len = max_len - seq.shape[0]

        # Create a padding tensor
        if padding_len > 0:
            padding_tensor = paddle.full([padding_len] + list(seq.shape[1:]), padding_value, dtype=seq.dtype)
            # Concatenate the original sequence and the padding tensor
            padded_seq = paddle.concat([seq, padding_tensor], axis=0)
        else:
            padded_seq = seq

        padded_sequences.append(padded_seq)

    # Stack the padded sequences to form a batch
    padded_batch = paddle.stack(padded_sequences, axis=0)
    return padded_batch


def orig_pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    if batch_first:
        return pad_sequence_paddle(sequences, padding_value)
    else:
        assert False, "Not implemented"


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: paddlenlp.transformers.PretrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        images = [paddle.stack(instance["image"]) for instance in instances]
        images_high = [paddle.stack(instance["image_high"]) for instance in instances]
        images = list(zip(images, images_high))

        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.not_equal(paddle.to_tensor(self.tokenizer.pad_token_id)),
            images=images,
        )
        return batch


def make_supervised_data_module(interleave, with_box, tokenizer, data_args):
    assert data_args.conversation_version == "mpt"

    train_dataset = ConversationDataset(
        tokenizer=tokenizer,
        # datasets=data_args.datasets,
        meta_path=data_args.meta_path,
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high=data_args.image_processor_high,
            box_limit=data_args.box_limit,
        ),
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
