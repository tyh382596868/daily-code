---
date: 2026-10-23
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/models/backbones/vision/base_vision.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/models/backbones/vision/base_vision.py#L108-L191
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, vision-encoder]
build_role: vision-encoder
---

# OpenVLA 视觉 backbone 契约：取倒数第二层 patch / OpenVLA Vision Backbone Contract: Take Penultimate-Layer Patches

> **一句话 / In one line**: OpenVLA 把不同 TIMM ViT 包成同一份 patch-feature 接口，并明确选择倒数第二层。 / OpenVLA wraps different TIMM ViTs behind one patch-feature interface and explicitly selects the penultimate layer.

## 为什么重要 / Why this matters

VLA 的视觉端不该把下游绑死在某个 encoder 的分类头或返回元组上。这里修整输出形状，并让 projector 只接收稳定的 patch features。

A VLA vision front end should not tie downstream code to one encoder's classifier head or tuple return. This code normalizes the output so the projector receives stable patch features.

## 代码 / The code

`openvla/openvla` — [`base_vision.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/models/backbones/vision/base_vision.py#L108-L191)

```python
self.featurizer = timm.create_model(self.timm_path_or_url, pretrained=True,
                                    num_classes=0, img_size=self.default_image_size)
self.featurizer.eval()
self.featurizer.forward = unpack_tuple(
    partial(self.featurizer.get_intermediate_layers,
            n={len(self.featurizer.blocks) - 2})
)

def forward(self, pixel_values):
    return self.featurizer(pixel_values)
```

## 逐行讲解 / What's happening

`num_classes=0` 删除分类头；`get_intermediate_layers` 把特征层选择显式化；`unpack_tuple` 消除 TIMM 返回类型的差异。最终 `forward` 保持一个简单的 tensor 契约。

`num_classes=0` removes the classifier; `get_intermediate_layers` makes layer selection explicit; `unpack_tuple` hides TIMM return-type differences. The final `forward` keeps a simple tensor contract.

## 类比 / The analogy

像相机转接环：不同镜头都接到同一种机身卡口，后面的摄影流程不用知道镜头型号。

It is a camera adapter: different lenses meet the same body mount, so the rest of the workflow need not know the lens model.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `vision-encoder` 的 cross-repo 变体：输入是预处理图像，输出是 `(B, patches, D)`；下游是 modality projector/VLM prefix。缺少它，动作 head 只能看原始像素；生产版还要处理多相机、dtype、批量和 checkpoint 加载。

This is a cross-repo `vision-encoder` variant: preprocessed images in, `(B, patches, D)` out, feeding the modality projector/VLM prefix. Without it an action head sees raw pixels; production also needs multi-camera, dtype, batching, and checkpoint handling.

## 自己跑一遍 / Try it yourself

```python
import torch
class Encoder(torch.nn.Module):
    def __init__(self): super().__init__(); self.patch = torch.nn.Conv2d(3, 8, 16, 16)
    def forward(self, x): return self.patch(x).flatten(2).transpose(1, 2)
print(Encoder()(torch.randn(2,3,32,32)).shape)
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `torch.Size([2, 4, 8])`。

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **nanoVLM projector** / **nanoVLM projector**: 接收视觉 patch，再压缩并映射到 LM 宽度。 / It receives vision patches, then compresses and maps them to LM width.

## 注意事项 / Caveats / when it breaks

- 倒数第二层是模型设计选择，换 backbone 前要重新验证特征层。 / The penultimate layer is a model choice; revalidate it when changing backbones.

## 延伸阅读 / Further reading

- [OpenVLA vision backbone source](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/models/backbones/vision/base_vision.py)
