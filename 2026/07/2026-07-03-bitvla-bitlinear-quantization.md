---
date: 2026-07-03
topic: diffusion
source: trending
repo: ustcwhy/BitVLA
file: transformers/src/transformers/models/siglip/modeling_siglip.py
permalink: https://github.com/ustcwhy/BitVLA/blob/main/transformers/src/transformers/models/siglip/modeling_siglip.py#L50-L174
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, vla, quantization, bitlinear]
---

# BitVLA BitLinear：把权重压成 {-1,0,1} / BitVLA BitLinear: Compress Weights into {-1, 0, 1}

> **一句话 / In one line**: `BitLinear` 用直通估计训练 1-bit/ternary 权重，并提供 2-bit 打包的离线量化路径。 / `BitLinear` trains 1-bit/ternary weights with a straight-through estimator and offers a packed 2-bit offline quantization path.

## 为什么重要 / Why this matters

机器人策略最终要上真实机器，显存和延迟都是真约束。BitVLA 的思路是把 VLA 里的视觉/语言线性层压到极低 bit，同时让训练仍然能反向传播。这类量化技巧会影响未来 VLA 和世界模型在边缘设备上的部署方式。

Robot policies eventually run on real hardware, where memory and latency are real constraints. BitVLA pushes VLA vision/language linear layers to very low bit widths while keeping gradients usable. This kind of quantization changes how future VLAs and world models can be deployed at the edge.

## 代码 / The code

`ustcwhy/BitVLA` — [`modeling_siglip.py`](https://github.com/ustcwhy/BitVLA/blob/main/transformers/src/transformers/models/siglip/modeling_siglip.py#L50-L174)

```python
class WeightQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = 1.0 / x.abs().mean().clamp_(min=1e-5)
        x = (x * s).round().clamp(-1, 1) / s
        return x.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ActQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x = (x * s).round().clamp(-128, 127) / s
        return x.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BitLinear(nn.Linear):
    def __init__(self, *kargs, weight_bits=8, input_bits=8, **kwargs):
        super(BitLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.enable_qlora = False

    def forward(self, input):
        weight = self.weight
        if self.input_bits == 8:
            input = ActQuant.apply(input)

        if self.enable_qlora and self.weight_bits == 1:
            weight = dequantize_from_int2(
                self.q_weight,
                self.w_step.item(),
                self.orig_shape,
                self.n_elems,
            ).type(input.dtype)
        elif self.weight_bits == 1:
            weight = WeightQuant.apply(weight)

        out = nn.functional.linear(input, weight, self.bias)
        return out
```

## 逐行讲解 / What's happening

1. **权重量化 / Weight quantization**: 中文: `abs().mean()` 给每层一个尺度，`round().clamp(-1, 1)` 把权重限制到三值。 / English: `abs().mean()` gives the layer a scale, and `round().clamp(-1, 1)` restricts weights to ternary values.
2. **激活量化 / Activation quantization**: 中文: 激活按最后一维最大值缩放到 int8 范围，再反缩放回浮点。 / English: Activations are scaled by the max over the last dimension into the int8 range, then de-scaled back to float.
3. **直通反传 / Straight-through backward**: 中文: `backward` 直接返回 `grad_output`，训练时把量化当成近似恒等映射。 / English: `backward` returns `grad_output` directly, treating quantization as an approximate identity during training.

## 类比 / The analogy

像把一套精密扳手换成三档棘轮：只能选反向、空档、正向，但如果尺度选得好，很多动作仍然够用，而且工具箱轻很多。

It is like replacing a set of precision wrenches with a three-position ratchet: reverse, neutral, forward. With a good scale, many jobs still work, and the toolbox is much lighter.

## 自己跑一遍 / Try it yourself

```python
import torch

w = torch.tensor([-0.9, -0.2, 0.1, 0.8])
scale = 1.0 / w.abs().mean().clamp(min=1e-5)
q = (w * scale).round().clamp(-1, 1) / scale
print(q)

x = torch.randn(2, 4)
s = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
xq = (x * s).round().clamp(-128, 127) / s
print(xq.shape)
```

运行 / Run with:

```bash
pip install torch
python try.py
```

预期输出 / Expected output:

```text
tensor([-0.5000, -0.0000,  0.0000,  0.5000])
torch.Size([2, 4])
```

量化后的权重只剩三种幅度，激活仍保持原 shape；部署时可以进一步打包节省显存。

The quantized weights keep only three magnitudes, while activations keep their shape. Deployment can pack the values further to save memory.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **BitNet / bitnet.cpp** / **BitNet / bitnet.cpp**: 同样围绕低 bit 线性层设计推理路径。
- **QLoRA** / **QLoRA**: 也把底座低 bit 化，但通常保留 adapter 做可训练增量。

## 注意事项 / Caveats / when it breaks

- **训练和部署内核不同** / **Training and deployment kernels differ**: Python 里的反量化方便读，但真正省时省显存需要专门 kernel。
- **量化尺度很敏感** / **The scale is sensitive**: 层内分布异常时，`abs().mean()` 可能不能代表所有通道。

## 延伸阅读 / Further reading

- [BitVLA repository](https://github.com/ustcwhy/BitVLA)
- [BitVLA paper](https://arxiv.org/abs/2506.07530)
