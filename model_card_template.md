---
{{ card_data }}
---

# Model Card for RecNeXt-{{ model_name.split('_')[-1].upper() }}{% if distillation %} (With Knowledge Distillation){% endif %}

[![license](https://img.shields.io/github/license/suous/RecNeXt)](https://github.com/suous/RecNeXt/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2406.16004-red)](https://arxiv.org/abs/2412.19628)

<div style="display: flex; justify-content: space-between;">
    <img src="https://raw.githubusercontent.com/suous/RecNeXt/refs/heads/main/figures/RecConvA.png" alt="RecConvA" style="width: 52%;">
    <img src="https://raw.githubusercontent.com/suous/RecNeXt/refs/heads/main/figures/code.png" alt="code" style="width: 46%;">
</div>

## Model Details

- **Model Type**: Image Classification / Feature Extraction
- **Model Series**: {{ series }}
- **Model Stats**: 
    - **Parameters**: {{ config.get('params', 'N/A') }}
    - **MACs**: {{ config.get('macs', 'N/A') }}
    - **Latency**: {{ config.get('latency', 'N/A') }} (iPhone 13, iOS 18)
    - **Image Size**: {{ config.get('image_size', '224x224') }}

- **Architecture Configuration**: 
    - **Embedding Dimensions**: {{ config.get('embed_dim', 'N/A') }}
    - **Depths**: {{ config.get('depth', 'N/A') }}
    - **MLP Ratio**: {{ config.get('mlp_ratio', 2) }}

- **Paper**: [RecConv: Efficient Recursive Convolutions for Multi-Frequency Representations](https://arxiv.org/abs/2412.19628)

- **Code**: https://github.com/suous/RecNeXt

- **Dataset**: ImageNet-1K

## Model Usage

### Image Classification

```python
from urllib.request import urlopen
from PIL import Image
import timm
import torch

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('{{ model_name }}', pretrained=True, distillation={{ distillation }})
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Converting to Inference Mode

```python
import utils

# Convert training-time model to inference structure, fuse batchnorms
utils.replace_batchnorm(model)
```
## Model Comparison

### Classification

We introduce two series of models: the **A** series uses linear attention and nearest interpolation, while the **M** series employs convolution and bilinear interpolation for simplicity and broader hardware compatibility (e.g., to address suboptimal nearest interpolation support in some iOS versions). 

> **dist**: distillation; **base**: without distillation (all models are trained over 300 epochs).

| model | top_1_accuracy | params | gmacs | npu_latency | cpu_latency | throughput | fused_weights                                                                                                                                                                                                | training_logs                                                                                                                                                                                     |
|-------|----------------|--------|-------|-------------|-------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| M0    | 74.7* \| 73.2  | 2.5M   | 0.4   | 1.0ms       | 189ms       | 763        | [dist](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m0_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m0_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_m0_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_m0_without_distill_300e.txt) |
| M1    | 79.2* \| 78.0  | 5.2M   | 0.9   | 1.4ms       | 361ms       | 384        | [dist](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m1_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m1_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_m1_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_m1_without_distill_300e.txt) |
| M2    | 80.3* \| 79.2  | 6.8M   | 1.2   | 1.5ms       | 431ms       | 325        | [dist](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m2_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m2_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_m2_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_m2_without_distill_300e.txt) |
| M3    | 80.9* \| 79.6  | 8.2M   | 1.4   | 1.6ms       | 482ms       | 314        | [dist](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m3_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m3_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_m3_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_m3_without_distill_300e.txt) |
| M4    | 82.5* \| 81.4  | 14.1M  | 2.4   | 2.4ms       | 843ms       | 169        | [dist](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m4_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m4_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_m4_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_m4_without_distill_300e.txt) |
| M5    | 83.3* \| 82.9  | 22.9M  | 4.7   | 3.4ms       | 1487ms      | 104        | [dist](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m5_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m5_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_m5_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_m5_without_distill_300e.txt) |
| A0    | 75.0* \| 73.6  | 2.8M   | 0.4   | 1.4ms       | 177ms       | 4902       | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a0_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a0_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_a0_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_a0_without_distill_300e.txt) |
| A1    | 79.6* \| 78.3  | 5.9M   | 0.9   | 1.9ms       | 334ms       | 2746       | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a1_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a1_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_a1_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_a1_without_distill_300e.txt) |
| A2    | 80.8* \| 79.6  | 7.9M   | 1.2   | 2.2ms       | 413ms       | 2327       | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a2_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a2_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_a2_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_a2_without_distill_300e.txt) |
| A3    | 81.1* \| 80.1  | 9.0M   | 1.4   | 2.4ms       | 447ms       | 2206       | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a3_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a3_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_a3_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_a3_without_distill_300e.txt) |
| A4    | 82.5* \| 81.6  | 15.8M  | 2.4   | 3.6ms       | 764ms       | 1265       | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a4_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a4_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_a4_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_a4_without_distill_300e.txt) |
| A5    | 83.5* \| 83.1  | 25.7M  | 4.7   | 5.6ms       | 1376ms      | 721        | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a5_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_a5_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/logs/distill/recnext_a5_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/logs/normal/recnext_a5_without_distill_300e.txt) |

### Comparison with [LSNet](https://github.com/jameslahm/lsnet)

| model | top_1_accuracy | params | gmacs | npu_latency | cpu_latency | throughput | fused_weights                                                                                                                                                                                              | training_logs                                                                                                                                                                                               |
|-------|----------------|--------|-------|-------------|-------------|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| T     | 76.6* \| 75.1  | 12.1M  | 0.3   | 1.8ms       | 109ms       | 13878      | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_t_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_t_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/lsnet/logs/distill/recnext_t_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/lsnet/logs/normal/recnext_t_without_distill_300e.txt) |
| S     | 79.6* \| 78.3  | 15.8M  | 0.7   | 2.0ms       | 188ms       | 7989       | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_s_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_s_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/lsnet/logs/distill/recnext_s_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/lsnet/logs/normal/recnext_s_without_distill_300e.txt) |
| B     | 81.4* \| 80.3  | 19.3M  | 1.1   | 2.5ms       | 290ms       | 4450       | [dist](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_b_distill_300e_fused.pt) \| [base](https://github.com/suous/RecNeXt/releases/download/v2.0/recnext_b_without_distill_300e_fused.pt) | [dist](https://github.com/suous/RecNeXt/blob/main/lsnet/logs/distill/recnext_b_distill_300e.txt) \| [base](https://github.com/suous/RecNeXt/blob/main/lsnet/logs/normal/recnext_b_without_distill_300e.txt) |

> The NPU latency is measured on an iPhone 13 with models compiled by Core ML Tools.
> The CPU latency is accessed on a Quad-core ARM Cortex-A57 processor in ONNX format.
> And the throughput is tested on an Nvidia RTX3090 with maximum power-of-two batch size that fits in memory.


## Citation

```BibTeX
@misc{zhao2024recnext,
      title={RecConv: Efficient Recursive Convolutions for Multi-Frequency Representations},
      author={Mingshu Zhao and Yi Luo and Yong Ouyang},
      year={2024},
      eprint={2412.19628},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
