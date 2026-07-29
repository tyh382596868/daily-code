---
date: 2026-07-29
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/unilora/layer.py
permalink: https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/unilora/layer.py#L203-L252
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, lora]
---

# PEFT UniLora：用索引表拼出 LoRA 矩阵 / PEFT UniLora: Rebuild LoRA Matrices from Index Tables

> **一句话 / In one line**: UniLora 不直接存完整 A/B 矩阵，而是从共享 `theta_d` 参数表里按索引取值，再乘缩放表组成低秩更新。 / UniLora does not store full A/B matrices directly; it indexes into a shared `theta_d` table and multiplies by scale tables to form the low-rank update.

## 为什么重要 / Why this matters

普通 LoRA 给每个 adapter 存 `A` 和 `B`。UniLora 把“参数值”和“矩阵布局”拆开：值来自共享表，布局来自 index buffer。这是一种节省参数、保留 adapter 形状灵活性的做法。

Standard LoRA stores `A` and `B` per adapter. UniLora separates "parameter values" from "matrix layout": values come from a shared table, while index buffers define the layout. That saves parameters while preserving adapter-specific matrix shapes.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/unilora/layer.py`](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/unilora/layer.py#L203-L252)

```python
def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    unilora_indices_A = self.unilora_indices_A[adapter]
    unilora_indices_B = self.unilora_indices_B[adapter]
    unilora_theta_d = self.unilora_theta_d[adapter].to(unilora_indices_A.device)
    scales_A = self.unilora_scales_A[adapter].to(unilora_indices_A.device)
    scales_B = self.unilora_scales_B[adapter].to(unilora_indices_B.device)

    if cast_to_fp32:
        unilora_theta_d = unilora_theta_d.float()
        scales_A = scales_A.float()
        scales_B = scales_B.float()

    A = unilora_theta_d[unilora_indices_A] * scales_A
    B = unilora_theta_d[unilora_indices_B] * scales_B

    return A, B

def get_delta_weight(self, adapter) -> torch.Tensor:
    device = self.unilora_indices_A[adapter].device
    dtype = self.unilora_theta_d[adapter].dtype

    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
    A, B = self._get_lora_matrices(adapter, cast_to_fp32)
    output_tensor = transpose(B @ A, self.fan_in_fan_out)
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
        for active_adapter in self.active_adapters:
            if active_adapter not in self.unilora_indices_A.keys():
                continue

            A, B = self._get_lora_matrices(active_adapter)
            x = x.to(self.unilora_theta_d[active_adapter].dtype)
            dropout = self.unilora_dropout[active_adapter]

            result = result + F.linear(F.linear(dropout(x), A), B)

    result = result.to(previous_dtype)
    return result
```

## 逐行讲解 / What's happening

1. **第 203-208 行 / Lines 203-208 (`_get_lora_matrices`)**:
   - 中文: 先拿 A/B 的索引表、共享参数表和缩放表，并把它们放到索引所在设备。
   - English: It first fetches A/B index tables, the shared parameter table, and scale tables, moving them to the index device.
2. **第 215-218 行 / Lines 215-218 (materialize A/B)**:
   - 中文: `theta_d[indices]` 像查字典一样生成矩阵，再乘 adapter 专属 scale。
   - English: `theta_d[indices]` materializes matrices by lookup, then multiplies by adapter-specific scales.
3. **第 220-229 行 / Lines 220-229 (`get_delta_weight`)**:
   - 中文: 低精度 CPU 路径先转 FP32 做矩阵乘，再转回原 dtype，避免 CPU 半精度 matmul 陷阱。
   - English: Low-precision CPU paths cast to FP32 for matmul and cast back, avoiding CPU half-precision matmul pitfalls.
4. **第 231-252 行 / Lines 231-252 (`forward`)**:
   - 中文: 未 merge 时，base layer 输出加上每个 active adapter 的 `B(A(dropout(x)))`。
   - English: When not merged, the base layer output receives each active adapter's `B(A(dropout(x)))` update.

## 类比 / The analogy

这像用一盒通用乐高零件搭不同模型：`theta_d` 是零件盒，index 表是说明书，scale 表决定每块零件在当前模型里要不要加厚。

It is like building different models from one shared box of bricks: `theta_d` is the brick box, index tables are the instructions, and scale tables decide how strongly each brick contributes.

## 自己跑一遍 / Try it yourself

```python
theta = [0.1, 0.2, 0.3, 0.4]
idx_a = [[0, 1, 2], [1, 2, 3]]
idx_b = [[3, 2], [1, 0]]
scale_a = [[1, 1, 1], [2, 2, 2]]
scale_b = [[1, 1], [0.5, 0.5]]

def gather(index, scale):
    return [[theta[i] * s for i, s in zip(row, scale_row)]
            for row, scale_row in zip(index, scale)]

A = gather(idx_a, scale_a)
B = gather(idx_b, scale_b)
delta00 = sum(B[0][k] * A[k][0] for k in range(2))
print(A)
print(B)
print(round(delta00, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0.1, 0.2, 0.3], [0.4, 0.6, 0.8]]
[[0.4, 0.3], [0.1, 0.05]]
0.16
```

这个例子展示了“矩阵不是存出来的，而是查出来的”。真正的 UniLora 只是把这里的列表换成 Tensor，并把结果接到线性层上。

The example shows that the matrices are looked up, not stored directly. Real UniLora replaces the lists with tensors and attaches the result to a linear layer.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Embedding table** / **Embedding tables**: token id 查共享 embedding 行。 / Token IDs index rows from a shared embedding table.
- **向量量化** / **Vector quantization**: codebook id 指向共享向量，样本只保存 id。 / Codebook IDs point to shared vectors while samples store IDs.

## 注意事项 / Caveats / when it breaks

- **索引设备要对齐** / **Index devices must align**: 参数表、scale 和 index 不在同一设备会触发搬运或报错。 / Parameter tables, scales, and indices must be on compatible devices.
- **merge 后 forward 逻辑不同** / **Merged forward is different**: merge 后 delta 已经加进 base weight，不再逐次计算 adapter。 / After merge, the delta is already in the base weight and is no longer recomputed every forward.

## 延伸阅读 / Further reading

- [PEFT documentation](https://huggingface.co/docs/peft)
- [Source permalink](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/unilora/layer.py#L203-L252)
