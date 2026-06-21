---
date: 2026-06-21
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/vera/layer.py
permalink: https://github.com/huggingface/peft/blob/036abd27e464819e0a19ddaa26093c84d5943488/src/peft/tuners/vera/layer.py#L214-L289
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, vera, lora, parameter-efficient-finetuning]
---

# VeRA：全模型共享一对冻结随机矩阵，每层只训练两个缩放向量 / VeRA: One Frozen Random Matrix Pair for the Whole Model, Two Scale Vectors Per Layer

> **一句话 / In one line**: VeRA 用一对全局共享的冻结随机矩阵 A、B 替代 LoRA 每层独立的 A、B，每层只训练一个行缩放向量 λ_d（rank 维）和一个列缩放向量 λ_b（D_out 维），将可训练参数量压缩到 LoRA 的约 1/10。 / VeRA replaces LoRA's per-layer trainable A and B matrices with one globally shared pair of frozen random matrices, training only a row-scale vector λ_d (rank) and column-scale vector λ_b (D_out) per layer — about 1/10 the trainable parameters of LoRA.

## 为什么重要 / Why this matters

LoRA 是目前最主流的参数高效微调（PEFT）方法：给每个目标层添加两个低秩矩阵 A（rank×D_in）和 B（D_out×rank），只训练这两个矩阵。但当模型层数多、rank 较大时，LoRA 的参数量仍然可观——对 LLaMA-65B 施加 rank=16 的 LoRA，可训练参数约 160M。

VeRA（Vector-based Random Matrix Adaptation）提出了一个更激进的假设：**A 和 B 不需要从数据中学习，随机初始化后冻结就够了**——只要它们足够大，随机矩阵组成的低秩空间就已经覆盖了模型需要调整的方向。真正的自由度由两个逐元素缩放向量 λ_d 和 λ_b 提供，这两个向量对整个模型而言只需要约 rank + D_out 个参数，而不是每层都分别占 rank×(D_in+D_out)。

实验显示 VeRA 在 GLUE 等基准上的质量与 LoRA 接近，而可训练参数量减少约 10 倍。这在内存受限场景（消费级 GPU 微调）和需要快速上传/切换 adapter 的场景下很有吸引力。

LoRA is the dominant PEFT method today: add two low-rank matrices A (rank×D_in) and B (D_out×rank) per target layer, train only those. But across many layers with non-trivial rank, LoRA's parameter count is still significant — rank=16 LoRA on LLaMA-65B is ~160M trainable params.

VeRA (Vector-based Random Matrix Adaptation) makes a bolder bet: **A and B don't need to be learned from data — random initialization and freezing is enough** — provided they're large enough, random matrices already span the directions the model needs to adjust. The actual degrees of freedom come from two elementwise scale vectors λ_d and λ_b, which across the whole model cost only rank + D_out parameters per layer instead of rank × (D_in + D_out).

Experiments show VeRA reaches quality close to LoRA on GLUE and similar benchmarks, with ~10× fewer trainable parameters — attractive for memory-constrained fine-tuning on consumer GPUs and for scenarios requiring fast adapter upload/switching.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/vera/layer.py`](https://github.com/huggingface/peft/blob/036abd27e464819e0a19ddaa26093c84d5943488/src/peft/tuners/vera/layer.py#L214-L289)

```python
def get_delta_weight(self, adapter) -> torch.Tensor:
    vera_A = self.vera_A[adapter]
    vera_B = self.vera_B[adapter]

    device = vera_B.device
    dtype = vera_B.dtype

    # In case users wants to merge the adapter weights that are in
    # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    lambda_d = self.vera_lambda_d[adapter]
    lambda_b = self.vera_lambda_b[adapter]

    if cast_to_fp32:
        vera_A = vera_A.float()
        vera_B = vera_B.float()
        lambda_d = lambda_d.float()
        lambda_b = lambda_b.float()

    sliced_A = vera_A[:, : self.in_features].to(lambda_d.device)
    sliced_B = vera_B[: self.out_features, :].to(lambda_d.device)
    lambda_b = lambda_b.unsqueeze(-1)
    lambda_d = lambda_d.unsqueeze(-1)
    output_tensor = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), self.fan_in_fan_out)

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

    return output_tensor

def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        orig_dtype = result.dtype
        if self.quantization_backend is not None:
            result = self.quantization_backend.maybe_clone_base_result(result)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.vera_lambda_d.keys():
                continue

            lambda_d = self.vera_lambda_d[active_adapter]
            lambda_b = self.vera_lambda_b[active_adapter]

            vera_A = self.vera_A[active_adapter]
            vera_B = self.vera_B[active_adapter]

            # As adapted layers may have different shapes and VeRA contains a single shared pair of A and B matrices,
            # we initialize these matrices with the largest required size for each dimension.
            # During the forward pass, required submatrices are sliced out from the shared vera_A and vera_B.
            sliced_A = vera_A[:, : self.in_features].to(x.device)
            sliced_B = vera_B[: self.out_features, :].to(x.device)

            dropout = self.vera_dropout[active_adapter]
            x = self._cast_input_dtype(x, lambda_d.dtype)
            result = result + lambda_b * F.linear(lambda_d * F.linear(dropout(x), sliced_A), sliced_B)
        result = result.to(orig_dtype)

    result = result.to(previous_dtype)
    return result
```

## 逐行讲解 / What's happening

1. **`get_delta_weight` 中的 `sliced_A` / `sliced_B`**:
   - 中文: 全局共享的 `vera_A` 和 `vera_B` 是按模型中**最大层**的尺寸初始化的。对于较小的层，直接切片取子矩阵：`vera_A[:, :self.in_features]` 和 `vera_B[:self.out_features, :]`。不同层共享同一对矩阵，只是切出不同大小的子块。
   - English: The globally shared `vera_A` and `vera_B` are initialized to the size of the **largest layer** in the model. For smaller layers, subarrays are sliced: `vera_A[:, :self.in_features]` and `vera_B[:self.out_features, :]`. Different layers share the same matrix pair, just taking differently sized slices.

2. **`lambda_b.unsqueeze(-1)` 和 `lambda_d.unsqueeze(-1)`**:
   - 中文: λ_b 形状原本是 `(D_out,)`，unsqueeze 后变成 `(D_out, 1)`，可以直接广播到矩阵乘法结果 `sliced_B @ (...)` 的每一列。λ_d 同理对 `sliced_A` 的每一行缩放。
   - English: λ_b originally has shape `(D_out,)`, unsqueeze makes it `(D_out, 1)` so it broadcasts across each column of the matrix product. λ_d analogously scales each row of `sliced_A`.

3. **`(lambda_b * sliced_B) @ (lambda_d * sliced_A)` — 核心公式**:
   - 中文: 这一行就是 VeRA 的全部数学：先用 λ_d 对 sliced_A 的每一行缩放（相当于对 rank 维做对角变换），再与 sliced_B 的列缩放（λ_b 对 D_out 维做对角变换）做矩阵乘。等价于 diag(λ_b) × B × diag(λ_d) × A，但只有 λ_b 和 λ_d 是可训练的。
   - English: This single line is all of VeRA's math: scale each row of `sliced_A` by λ_d (diagonal transform on the rank dimension), then matrix-multiply with `sliced_B` whose columns are scaled by λ_b (diagonal transform on D_out). Equivalent to diag(λ_b) × B × diag(λ_d) × A, but only λ_b and λ_d are trainable.

4. **`forward` 中的 `result + lambda_b * F.linear(lambda_d * F.linear(dropout(x), sliced_A), sliced_B)`**:
   - 中文: 这是 forward 的核心一行，与 LoRA 的 `result + lora_B(lora_A(dropout(x)))` 结构相同，区别在于额外的 `lambda_d *` 和 `lambda_b *` 逐元素缩放，以及 A、B 是冻结的缓冲区而非可训练参数。
   - English: This is the core forward line, structurally identical to LoRA's `result + lora_B(lora_A(dropout(x)))`, with the addition of elementwise `lambda_d *` and `lambda_b *` scaling, and A, B being frozen buffers rather than trainable parameters.

5. **CPU fp32 上转 cast**:
   - 中文: 在 CPU 上合并 bf16/fp16 权重时先转成 fp32，因为许多 CPU 的 bf16/fp16 矩阵乘很慢甚至不支持。合并完再转回原始 dtype。这个 pattern 在 PEFT 多个 adapter 里都能看到。
   - English: When merging bf16/fp16 weights on CPU, cast to fp32 first because many CPUs have slow or unsupported bf16/fp16 matmuls. Cast back to the original dtype after merging. This pattern appears in multiple PEFT adapter implementations.

## 类比 / The analogy

想象你要用一块大橡皮泥（随机矩阵 A 和 B）捏出许多不同的形状（每层的权重更新）。LoRA 的做法是：每层各给一块橡皮泥，自由塑形。VeRA 的做法是：所有层共用同一块橡皮泥，但每层可以独立调整橡皮泥在不同方向上被"拉伸"的倍数（λ_d 和 λ_b）。你不能改变橡皮泥本身，只能控制拉伸比——但这种局限的自由度往往已经足够塑造出正确的形状。

Imagine you need to sculpt many different shapes (per-layer weight updates) from clay (random matrices A and B). LoRA gives each layer its own lump of clay to shape freely. VeRA gives all layers the same shared lump, but lets each layer independently control how much to stretch it in different directions (λ_d and λ_b). You can't change the clay itself, only the stretch ratios — but this limited freedom is often enough to mold the right shape.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyVeRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank, shared_A, shared_B):
        super().__init__()
        self.base = nn.Linear(in_f, out_f, bias=False)
        # Frozen shared matrices (registered as buffers, not parameters)
        self.register_buffer("A", shared_A[:rank, :in_f])
        self.register_buffer("B", shared_B[:out_f, :rank])
        # Trainable scale vectors only
        self.lambda_d = nn.Parameter(torch.full((rank,),  0.1))
        self.lambda_b = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        base_out = self.base(x)
        vera_out = self.lambda_b * F.linear(self.lambda_d * F.linear(x, self.A), self.B)
        return base_out + vera_out

rank = 8
shared_A = torch.randn(rank, 64)   # one pair for the whole "model"
shared_B = torch.randn(32, rank)

layer = TinyVeRALinear(64, 32, rank, shared_A, shared_B)
x = torch.randn(4, 64)
print("output shape:", layer(x).shape)

trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"trainable: {trainable} / total: {total}")  # only lambda_d + lambda_b
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
output shape: torch.Size([4, 32])
trainable: 40 / total: 2088
```

中文：40 个可训练参数 = rank(8) + out_f(32)，而基础 Linear 权重有 64×32=2048 个参数。如果换成 LoRA rank=8，还需要额外的 8×64 + 32×8 = 768 个参数。VeRA 只需 40 个。

English: 40 trainable parameters = rank(8) + out_f(32), while the base Linear has 64×32=2048 parameters. LoRA rank=8 would need an additional 8×64 + 32×8 = 768 trainable params. VeRA needs only 40.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LoRA (`peft/tuners/lora/layer.py`)** / **LoRA**: 同样的双矩阵结构，但 A 和 B 是每层独立的可训练参数，没有共享和冻结。 / Same two-matrix structure, but A and B are independent trainable parameters per layer, not shared or frozen.
- **Intrinsic dimensionality / fastfood transforms** / **内禀维度研究**: Aghajanyan et al. (2020) 证明预训练模型的微调实际上发生在一个很低的内禀维度空间里。VeRA 的假设正是基于这一发现——低维更新不需要学会矩阵，只需学会缩放。 / Aghajanyan et al. (2020) showed fine-tuning of pretrained models occurs in a low intrinsic-dimension subspace. VeRA's assumption is grounded in this finding — low-dim updates don't need to learn the matrix, just the scaling.
- **随机特征方法（Random Features）** / **Random Feature methods**: Rahimi & Recht (2007) 的随机傅里叶特征：用固定随机矩阵把核函数线性化。VeRA 借用了同样的直觉：随机矩阵已经足够"丰富"，无需从数据中学习。 / Rahimi & Recht (2007) random Fourier features use fixed random matrices to linearize kernel functions. VeRA borrows the same intuition: a random matrix is already "rich enough" without learning it from data.

## 注意事项 / Caveats / when it breaks

- **rank 需要足够大** / **Rank must be large enough**: 如果 rank 太小，随机矩阵的列空间无法覆盖目标方向，λ_d 和 λ_b 无论怎么调也学不到正确的更新。VeRA 原论文建议 rank 远高于 LoRA 的典型值（如 rank=256+）。 / If rank is too small, the column space of the random matrix can't cover the needed directions, and no λ_d/λ_b values can fix it. The VeRA paper recommends much higher rank than typical LoRA (e.g. rank=256+).
- **不适合需要 per-task 矩阵的场景** / **Not for per-task matrix customization**: 若不同 adapter 需要朝不同方向调整，VeRA 的共享矩阵意味着所有 adapter 必须在同一个随机基下工作，可能限制表达能力。 / If different adapters need to adjust in very different directions, sharing the same random basis may limit expressiveness compared to per-adapter LoRA matrices.
- **初始化很重要** / **Init matters**: λ_b 初始化为 0、λ_d 初始化为 `d_initial`（默认 0.1），使初始 delta 接近零，保持训练初期的稳定性。 / λ_b initialized to 0 and λ_d to `d_initial` (default 0.1) keeps the initial delta near zero for stable early training.

## 延伸阅读 / Further reading

- [VeRA: Vector-based Random Matrix Adaptation (Kopiczko et al., 2023)](https://arxiv.org/abs/2310.11454)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
