# RecNeXt: Efficient Recursive Convolutions for Multi-Frequency Representations

[![license](https://img.shields.io/github/license/suous/RecNeXt)](https://github.com/suous/RecNeXt/blob/main/LICENSE)

<div style="display: flex; justify-content: space-between;">
    <img src="figures/RecConvA.png" alt="RecConvA" style="width: 55%;">
    <img src="figures/code.png" alt="code" style="width: 45%;">
</div>

<details>
  <summary>
  <font size="+1">RecConv Details</font>
  </summary>

  ```python
  class RecConv2d(nn.Module):
      def __init__(self, in_channels, kernel_size=5, bias=False, level=1):
          super().__init__()
          self.level = level
          kwargs = {
              'in_channels': in_channels, 
              'out_channels': in_channels, 
              'groups': in_channels,
              'kernel_size': kernel_size, 
              'padding': kernel_size // 2, 
              'bias': bias, 
          }
          self.down = nn.Conv2d(stride=2, **kwargs)
          self.convs = nn.ModuleList([nn.Conv2d(**kwargs) for _ in range(level+1)])
  
      def forward(self, x):
          i = x
          features = []
          for _ in range(self.level):
              x, s = self.down(x), x.shape[2:]
              features.append((x, s))
  
          x = 0
          for conv, (f, s) in zip(self.convs, reversed(features)):
              x = nn.functional.interpolate(conv(f + x), size=s, mode='bilinear')
          return self.convs[self.level](i + x)
  ```
</details>


## Abstract

This paper introduces RecConv, a recursive decomposition strategy that efficiently constructs multi-frequency representations using small-kernel convolutions. 
RecConv establishes a linear relationship between parameter growth and decomposing levels which determines the effective kernel size $k\times 2^\ell$ for a base kernel $k$ and $\ell$ levels of decomposition, while maintaining constant FLOPs regardless of the ERF expansion. 
Specifically, RecConv achieves a parameter expansion of only $\ell+2$ times and a maximum FLOPs increase of $5/3$ times, compared to the exponential growth ($4^\ell$) of standard and depthwise convolutions.
RecNeXt-M3 outperforms RepViT-M1.1 by 1.9 $AP^{box}$ on COCO with similar FLOPs.
This innovation provides a promising avenue towards designing efficient and compact networks across various modalities.


<details>
  <summary>
  <font size="+1">Conclusion</font>
  </summary>
    This paper introduces a straightforward and versatile recursive decomposition strategy that leverages small-kernel convolutions to construct multi-frequency representations, establishing a linear relationship between parameter growth and decomposition levels.
    This innovation guarantees that the computational complexity at each decomposition level follows a geometric progression, diminishing exponentially with increasing depth.
    This recursive design is compatible with various operations and can be extended to other modalities, though this study focuses on vision tasks using basic functions.
    Leveraging this approach, we construct RecConv as a plug-and-play module that can be seamlessly integrated into existing computer vision architectures.
    To the best of our knowledge, this is the first effective convolution kernel up to $80 \times 80$ for resource-constrained vision tasks.
    Building on RecConv we introduce RecNeXt, an efficient large-kernel vision backbone optimized towards resource-constrained scenarios.
    Extensive experiments across multiple vision benchmarks demonstrate RecNeXt's superiority over current leading approaches without relying on structural reparameterization or neural architecture search.
</details>

<br/>

**UPDATES** 🔥
- **2024/06/25**: Uploaded checkpoints and training logs of recnext-M1 - M5.

## Classification on ImageNet-1K

### Models under the RepVit training strategy

We report the top-1 accuracy on ImageNet-1K with and without distillation using the same training strategy as [RepViT](https://github.com/THU-MIG/RepViT).

| Model | Top-1(distill) / Top-1 | #params | MACs | Latency |                                                                                                 Ckpt                                                                                                 |                                               Core ML                                               |                                                          Log                                                           |
|:------|:----------------------:|:-------:|:----:|:-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| M1    |      79.2 \| 78.0      |  5.2M   | 0.9G |  1.4ms  | [fused 300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m1_distill_300e_fused.pt) / [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m1_distill_300e.pth) | [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m1_distill_300e_224.mlmodel) | [distill 300e](./logs/distill/recnext_m1_distill_300e.txt) / [300e](./logs/normal/recnext_m1_without_distill_300e.txt) |
| M2    |      80.3 \| 79.2      |  6.8M   | 1.2G |  1.5ms  | [fused 300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m2_distill_300e_fused.pt) / [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m2_distill_300e.pth) | [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m2_distill_300e_224.mlmodel) | [distill 300e](./logs/distill/recnext_m2_distill_300e.txt) / [300e](./logs/normal/recnext_m2_without_distill_300e.txt) |
| M3    |      80.9 \| 79.6      |  8.2M   | 1.4G |  1.6ms  | [fused 300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m3_distill_300e_fused.pt) / [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m3_distill_300e.pth) | [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m3_distill_300e_224.mlmodel) | [distill 300e](./logs/distill/recnext_m3_distill_300e.txt) / [300e](./logs/normal/recnext_m3_without_distill_300e.txt) |
| M4    |      82.5 \| 81.1      |  14.1M  | 2.4G |  2.4ms  | [fused 300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m4_distill_300e_fused.pt) / [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m4_distill_300e.pth) | [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m4_distill_300e_224.mlmodel) | [distill 300e](./logs/distill/recnext_m4_distill_300e.txt) / [300e](./logs/normal/recnext_m4_without_distill_300e.txt) |
| M5    |      83.3 \| 81.6      |  22.9M  | 4.7G |  3.4ms  | [fused 300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m5_distill_300e_fused.pt) / [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m5_distill_300e.pth) | [300e](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m5_distill_300e_224.mlmodel) |        [distill 300e](./logs/recnext_m5_distill_300e.txt) / [300e](./logs/recnext_m5_without_distill_300e.txt)         |


```bash
# this script is used to validate the distillation results
fd txt logs/distill -x bash -c 'echo -n "{} "; jq -c ".test_acc1" {} | jq -s max' | sort | awk '{printf "%.1f %s\n", $2, $1}' 
```

<details>
  <summary>
  <font>output</font>
  </summary>

```
79.2 logs/distill/recnext_m1_distill_300e.txt
80.3 logs/distill/recnext_m2_distill_300e.txt
80.9 logs/distill/recnext_m3_distill_300e.txt
82.5 logs/distill/recnext_m4_distill_300e.txt
83.3 logs/distill/recnext_m5_distill_300e.txt
```
</details>

```bash
# this script is used to validate the results without distillation
fd txt logs/normal -x bash -c 'echo -n "{} "; jq -c ".test_acc1" {} | jq -s max' | sort | awk '{printf "%.1f %s\n", $2, $1}' 
```

<details>
  <summary>
  <font>output</font>
  </summary>

```
78.0 logs/normal/recnext_m1_without_distill_300e.txt
79.2 logs/normal/recnext_m2_without_distill_300e.txt
79.6 logs/normal/recnext_m3_without_distill_300e.txt
81.1 logs/normal/recnext_m4_without_distill_300e.txt
81.6 logs/normal/recnext_m5_without_distill_300e.txt
```
</details>


Tips: Convert a training-time RecNeXt into the inference-time structure
```
from timm.models import create_model
import utils

model = create_model('recnext_m1')
utils.replace_batchnorm(model)
```

## Latency Measurement 

The latency reported in RecNeXt for iPhone 13 (iOS 18) uses the benchmark tool from [XCode 14](https://developer.apple.com/videos/play/wwdc2022/10027/).

<details>
<summary>
RecNeXt-M1
</summary>
<img src="./figures/latency/recnext_m1_224x224.png" width=70%>
</details>

<details>
<summary>
RecNeXt-M2
</summary>
<img src="./figures/latency/recnext_m2_224x224.png" width=70%>
</details>

<details>
<summary>
RecNeXt-M3
</summary>
<img src="./figures/latency/recnext_m3_224x224.png" width=70%>
</details>

<details>
<summary>
RecNeXt-M4
</summary>
<img src="./figures/latency/recnext_m4_224x224.png" width=70%>
</details>

<details>
<summary>
RecNeXt-M5
</summary>
<img src="./figures/latency/recnext_m5_224x224.png" width=70%>
</details>

Tips: export the model to Core ML model
```
python export_coreml.py --model recnext_m1 --ckpt pretrain/recnext_m1_distill_300e.pth
```
Tips: measure the throughput on GPU
```
python speed_gpu.py --model recnext_m1
```

## ImageNet  

### Prerequisites
`conda` virtual environment is recommended. 
```
conda create -n recnext python=3.8
pip install -r requirements.txt
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The training and validation data are expected to be in the `train` folder and `val` folder respectively:

```bash
# script to extract ImageNet dataset: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
```

```
# organize the ImageNet dataset as follows:
imagenet
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── ......
│   ├── ......
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   ├── ......
│   ├── ......
```

### Training
To train RecNeXt-M1 on an 8-GPU machine:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12346 --use_env main.py --model recnext_m1 --data-path ~/imagenet --dist-eval
```
Tips: specify your data path and model name! 

### Testing 
For example, to test RecNeXt-M1:
```
python main.py --eval --model recnext_m1 --resume pretrain/recnext_m1_distill_300e.pth --data-path ~/imagenet
```

### Fused model evaluation
For example, to evaluate RecNeXt-M1 with the fused model: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suous/RecNeXt/blob/main/demo/fused_model_evaluation.ipynb)
```
python fuse_eval.py --model recnext_m1 --resume pretrain/recnext_m1_distill_300e_fused.pt --data-path ~/imagenet
```

## Downstream Tasks
[Object Detection and Instance Segmentation](detection/README.md)<br>

| Model      | $AP^b$ | $AP_{50}^b$ | $AP_{75}^b$ | $AP^m$ | $AP_{50}^m$ | $AP_{75}^m$ | Latency |                                       Ckpt                                        |                     Log                     |
|:-----------|:------:|:-----------:|:-----------:|:------:|:-----------:|:-----------:|:-------:|:---------------------------------------------------------------------------------:|:-------------------------------------------:|
| RecNeXt-M3 |  41.7  |    63.4     |    45.4     |  38.6  |    60.5     |    41.4     |  5.2ms  | [M3](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m3_coco.pth) | [M3](./detection/logs/recnext_m3_coco.json) |
| RecNeXt-M4 |  43.5  |    64.9     |    47.7     |  39.7  |    62.1     |    42.4     |  7.6ms  | [M4](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m4_coco.pth) | [M4](./detection/logs/recnext_m4_coco.json) |
| RecNeXt-M5 |  44.6  |    66.3     |    49.0     |  40.6  |    63.5     |    43.5     | 12.4ms  | [M5](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m5_coco.pth) | [M5](./detection/logs/recnext_m5_coco.json) |

```bash
# this script is used to validate the detection results
fd json detection/logs -x bash -c 'echo -n "{} "; tail -n +2 {} | jq -c ".bbox_mAP" {} | jq -s max | jq ".*100"' | sort | awk '{printf "%.1f %s\n", $2, $1}'
```

<details>
  <summary>
  <font>output</font>
  </summary>

```
41.7 detection/logs/recnext_m3_coco.json
43.5 detection/logs/recnext_m4_coco.json
44.6 detection/logs/recnext_m5_coco.json
```
</details>

[Semantic Segmentation](segmentation/README.md)

| Model      | mIoU | Latency |                                        Ckpt                                         |                       Log                        |
|:-----------|:----:|:-------:|:-----------------------------------------------------------------------------------:|:------------------------------------------------:|
| RecNeXt-M3 | 41.0 |  5.6ms  | [M3](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m3_ade20k.pth) | [M3](./segmentation/logs/recnext_m3_ade20k.json) |
| RecNeXt-M4 | 43.6 |  7.2ms  | [M4](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m4_ade20k.pth) | [M4](./segmentation/logs/recnext_m4_ade20k.json) |
| RecNeXt-M5 | 46.0 | 12.4ms  | [M5](https://github.com/suous/RecNeXt/releases/download/v1.0/recnext_m5_ade20k.pth) | [M5](./segmentation/logs/recnext_m5_ade20k.json) |

```bash
# this script is used to validate the segmentation results
fd json segmentation/logs -x bash -c 'echo -n "{} "; tail -n +2 {} | jq -c ".mIoU" {} | jq -s max | jq ".*100"' | sort | awk '{printf "%.1f %s\n", $2, $1}'
```

<details>
  <summary>
  <font>output</font>
  </summary>

```
41.0 segmentation/logs/recnext_m3_ade20k.json
43.6 segmentation/logs/recnext_m4_ade20k.json
46.0 segmentation/logs/recnext_m5_ade20k.json
```
</details>

## Ablation Study

### Overall Experiments

![ablation](figures/ablation.png)

<details>
  <summary>
  <font size="+1">Ablation Logs</font>
  </summary>

<pre>
logs/ablation
├── 224
│   ├── <a style="text-decoration:none" href="./logs/ablation/224/recnext_m1_120e_224x224_3x3_7464.txt">recnext_m1_120e_224x224_3x3_7464.txt</a>
│   ├── <a style="text-decoration:none" href="./logs/ablation/224/recnext_m1_120e_224x224_7x7_7552.txt">recnext_m1_120e_224x224_7x7_7552.txt</a>
│   ├── <a style="text-decoration:none" href="./logs/ablation/224/recnext_m1_120e_224x224_bxb_7541.txt">recnext_m1_120e_224x224_bxb_7541.txt</a>
│   ├── <a style="text-decoration:none" href="./logs/ablation/224/recnext_m1_120e_224x224_rec_3x3_7548.txt">recnext_m1_120e_224x224_rec_3x3_7548.txt</a>
│   ├── <a style="text-decoration:none" href="./logs/ablation/224/recnext_m1_120e_224x224_rec_5x5_7603.txt">recnext_m1_120e_224x224_rec_5x5_7603.txt</a>
│   └── <a style="text-decoration:none" href="./logs/ablation/224/recnext_m1_120e_224x224_rec_7x7_7567.txt">recnext_m1_120e_224x224_rec_7x7_7567.txt</a>
└── 384
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_3x3_7635.txt">recnext_m1_120e_384x384_3x3_7635.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_7x7_7742.txt">recnext_m1_120e_384x384_7x7_7742.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_bxb_7800.txt">recnext_m1_120e_384x384_bxb_7800.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_3x3_7772.txt">recnext_m1_120e_384x384_rec_3x3_7772.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_5x5_7811.txt">recnext_m1_120e_384x384_rec_5x5_7811.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_7x7_7803.txt">recnext_m1_120e_384x384_rec_7x7_7803.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_3x3_basic_7726.txt">recnext_m1_120e_384x384_rec_convtrans_3x3_basic_7726.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_5x5_basic_7787.txt">recnext_m1_120e_384x384_rec_convtrans_5x5_basic_7787.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_7x7_basic_7824.txt">recnext_m1_120e_384x384_rec_convtrans_7x7_basic_7824.txt</a>
    ├── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_7x7_group_7791.txt">recnext_m1_120e_384x384_rec_convtrans_7x7_group_7791.txt</a>
    └── <a style="text-decoration:none" href="./logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_7x7_split_7683.txt">recnext_m1_120e_384x384_rec_convtrans_7x7_split_7683.txt</a>
</pre>

```bash
# this script is used to validate the ablation results
fd txt logs/ablation -x bash -c 'echo -n "{} "; jq -c ".test_acc1" {} | jq -s max' | sort | awk '{printf "%.2f %s\n", $2, $1}' 
```

<details>
  <summary>
  <font>output</font>
  </summary>

```
74.64 logs/ablation/224/recnext_m1_120e_224x224_3x3_7464.txt
75.52 logs/ablation/224/recnext_m1_120e_224x224_7x7_7552.txt
75.41 logs/ablation/224/recnext_m1_120e_224x224_bxb_7541.txt
75.48 logs/ablation/224/recnext_m1_120e_224x224_rec_3x3_7548.txt
76.03 logs/ablation/224/recnext_m1_120e_224x224_rec_5x5_7603.txt
75.67 logs/ablation/224/recnext_m1_120e_224x224_rec_7x7_7567.txt
76.35 logs/ablation/384/recnext_m1_120e_384x384_3x3_7635.txt
77.42 logs/ablation/384/recnext_m1_120e_384x384_7x7_7742.txt
78.00 logs/ablation/384/recnext_m1_120e_384x384_bxb_7800.txt
77.72 logs/ablation/384/recnext_m1_120e_384x384_rec_3x3_7772.txt
78.11 logs/ablation/384/recnext_m1_120e_384x384_rec_5x5_7811.txt
78.03 logs/ablation/384/recnext_m1_120e_384x384_rec_7x7_7803.txt
77.26 logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_3x3_basic_7726.txt
77.87 logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_5x5_basic_7787.txt
78.24 logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_7x7_basic_7824.txt
77.91 logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_7x7_group_7791.txt
76.84 logs/ablation/384/recnext_m1_120e_384x384_rec_convtrans_7x7_split_7683.txt
```
</details>

</details>

### RecConv Variants

<div style="display: flex; justify-content: space-between;">
    <img src="figures/RecConvB.png" alt="RecConvB" style="width: 49%;">
    <img src="figures/RecConvC.png" alt="RecConvC" style="width: 49%;">
</div>


<details>
  <summary>
  <font size="+1">RecConv Variant Details</font>
  </summary>

- **RecConv using group convolutions**

```python
# RecConv Variant A
# recursive decomposition on both spatial and channel dimensions
# downsample and upsample through group convolutions
class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2):
        super().__init__()
        self.level = level
        kwargs = {'kernel_size': kernel_size, 'padding': kernel_size // 2, 'bias': bias}
        downs = []
        for l in range(level):
            i_channels = in_channels // (2 ** l)
            o_channels = in_channels // (2 ** (l+1))
            downs.append(nn.Conv2d(in_channels=i_channels, out_channels=o_channels, groups=o_channels, stride=2, **kwargs))
        self.downs = nn.ModuleList(downs)

        convs = []
        for l in range(level+1):
            channels = in_channels // (2 ** l)
            convs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, **kwargs))
        self.convs = nn.ModuleList(reversed(convs))

        # this is the simplest modification, only support resoltions like 256, 384, etc
        kwargs['kernel_size'] = kernel_size + 1
        ups = []
        for l in range(level):
            i_channels = in_channels // (2 ** (l+1))
            o_channels = in_channels // (2 ** l)
            ups.append(nn.ConvTranspose2d(in_channels=i_channels, out_channels=o_channels, groups=i_channels, stride=2, **kwargs))
        self.ups = nn.ModuleList(reversed(ups))
        
    def forward(self, x):
        i = x
        features = []
        for down in self.downs:
            x, s = down(x), x.shape[2:]
            features.append((x, s))

        x = 0
        for conv, up, (f, s) in zip(self.convs, self.ups, reversed(features)):
            x = up(conv(f + x))
        return self.convs[self.level](i + x)
```

- **RecConv using channel-wise concatenation**

```python
# recursive decomposition on both spatial and channel dimensions
# downsample using channel-wise split, followed by depthwise convolution with a stride of 2
# upsample through channel-wise concatenation
class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2):
        super().__init__()
        self.level = level
        kwargs = {'kernel_size': kernel_size, 'padding': kernel_size // 2, 'bias': bias}
        downs = []
        for l in range(level):
            channels = in_channels // (2 ** (l+1))
            downs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, stride=2, **kwargs))
        self.downs = nn.ModuleList(downs)

        convs = []
        for l in range(level+1):
            channels = in_channels // (2 ** l)
            convs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, **kwargs))
        self.convs = nn.ModuleList(reversed(convs))

 .      # this is the simplest modification, only support resoltions like 256, 384, etc
        kwargs['kernel_size'] = kernel_size + 1
        ups = []
        for l in range(level):
            channels = in_channels // (2 ** (l+1))
            ups.append(nn.ConvTranspose2d(in_channels=channels, out_channels=channels, groups=channels, stride=2, **kwargs))
        self.ups = nn.ModuleList(reversed(ups))

    def forward(self, x):
        features = []
        for down in self.downs:
            r, x = torch.chunk(x, 2, dim=1)
            x, s = down(x), x.shape[2:]
            features.append((r, s))

        for conv, up, (r, s) in zip(self.convs, self.ups, reversed(features)):
            x = torch.cat([r, up(conv(x))], dim=1)
        return self.convs[self.level](x)
```
</details>

## Acknowledgement

Classification (ImageNet) code base is partly built with [LeViT](https://github.com/facebookresearch/LeViT), [PoolFormer](https://github.com/sail-sg/poolformer), [EfficientFormer](https://github.com/snap-research/EfficientFormer),  [RepViT](https://github.com/THU-MIG/RepViT), and [MogaNet](https://github.com/Westlake-AI/MogaNet).

The detection and segmentation pipeline is from [MMCV](https://github.com/open-mmlab/mmcv) ([MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)). 

Thanks for the great implementations! 