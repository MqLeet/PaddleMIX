# GOT-OCR2.0

## 1. 模型介绍

[GOT-OCR2.0](https://qwenlm.github.io/blog/qwen2-vl/) 是大规模视觉语言模型。可以以图像、文本、检测框、视频作为输入，并以文本和检测框作为输出。本仓库提供paddle版本的`GOT-OCR2.0`模型。


## 2 环境准备
- **python >= 3.10**
- **paddlepaddle-gpu 要求版本develop**
```
# 安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- paddlenlp >= 3.0.0(默认开启flash_attn，推荐源码编译安装)

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH

## 3 推理预测

1. plain texts OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py --model_name_or_path  /GOT_weights/  --image_file  /an/image/file.png  --ocr_type ocr
```

2. format texts OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py --model_name_or_path  /GOT_weights/  --image_file  /an/image/file.png  --ocr_type format
```

3. fine-grained OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py --model_name_or_path  /GOT_weights/  --image_file  /an/image/file.png  --ocr_type format/ocr --box [x1,y1,x2,y2]
```
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py --model_name_or_path  /GOT_weights/  --image_file  /an/image/file.png  --ocr_type format/ocr --color red/green/blue
```

4. multi-crop OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py --model_name_or_path  /GOT_weights/  --image_file  /an/image/file.png  --multi_crop --ocr_type format/ocr
```

4. render the formatted OCR results:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py --model_name_or_path  /GOT_weights/  --image_file  /an/image/file.png --ocr_type format --render
```

## 参考文献
```BibTeX
@article{wei2024general,
  title={General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model},
  author={Wei, Haoran and Liu, Chenglong and Chen, Jinyue and Wang, Jia and Kong, Lingyu and Xu, Yanming and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Peng, Yuang and others},
  journal={arXiv preprint arXiv:2409.01704},
  year={2024}
}
```
