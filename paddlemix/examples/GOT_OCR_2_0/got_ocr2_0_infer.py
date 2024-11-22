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

import argparse

import paddle
from paddlenlp.transformers import QWenTokenizer

from paddlemix.models.GOT.model import GOTQwenForCausalLM

parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str, default="GOT-OCR2_0_pd", help="pretrained ckpt and tokenizer")
parser.add_argument("--image_file", type=str, default="yiyuan.jpeg")
parser.add_argument("--multi_crop", action="store_true")
parser.add_argument("--ocr_type", type=str, default="plain", choices=["ocr", "format"])
parser.add_argument("--box", type=str, default="")
parser.add_argument("--color", type=str, default="")
parser.add_argument("--render", action="store_true")

args = parser.parse_args()
model_name_or_path = args.model_name_or_path

tokenizer = QWenTokenizer.from_pretrained(model_name_or_path)
# print('tokenizer:\n', tokenizer)
# print('tokenizer.added_tokens_encoder:\n', tokenizer.added_tokens_encoder)
# print('tokenizer.added_tokens_decoder:\n', tokenizer.added_tokens_decoder)
# PretrainedTokenizer(name_or_path='',
# vocab_size=151851, model_max_len=8000, padding_side='right',
# truncation_side='right', special_tokens={
# 'pad_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False)})
model = GOTQwenForCausalLM.from_pretrained(
    model_name_or_path, dtype=paddle.bfloat16, pad_token_id=tokenizer.eos_token_id
).eval()
# print('tokenizer:\n', tokenizer)


# input test image
image_file = args.image_file
with paddle.no_grad():
    if args.multi_crop:
        # multi-crop OCR:
        res = model.chat_crop(
            tokenizer, image_file, ocr_type=args.ocr_type, render=args.render, save_render_file="./demo.html"
        )
    else:
        # plain texts OCR
        # format texts OCR
        # fine-grained OCR
        # render the formatted OCR results
        res = model.chat(
            tokenizer,
            image_file,
            ocr_type=args.ocr_type,
            ocr_box=args.box,
            ocr_color=args.color,
            render=args.render,
            save_render_file="./demo.html",
        )

    # plain texts OCR
    # res = model.chat(tokenizer, image_file, ocr_type='ocr')

    # format texts OCR:
    # res = model.chat(tokenizer, image_file, ocr_type='format')

    # fine-grained OCR:
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

    # multi-crop OCR:
    # res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
    # res = model.chat_crop(tokenizer, image_file, ocr_type='format')

    # render the formatted OCR results:
    # res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

    print(res)
