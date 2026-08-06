---
date: 2026-08-06
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/peanut/layer.py
permalink: https://github.com/huggingface/peft/blob/c702de402330f48c0d8b1f260d59413aa3110b65/src/peft/tuners/peanut/layer.py#L134-L231
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, adapter]
---

# PEFT PEANuT：delta weight 由 base weight 生成 / PEFT PEANuT: Generate Delta Weights from the Base Weight

> **一句话 / In one line**: PEANuT 不直接训练一张 LoRA 矩阵，而是让小网络读 base weight 后生成 adapter delta。 / PEANuT does not store a direct LoRA matrix; a small network reads the base weight and generates the adapter delta.

## 为什么重要 / Why this matters

常见 LoRA 是“输入乘 A 再乘 B”。PEANuT 更像 weight-aware adapter：它先把 base layer 的权重投进低秩空间，经过 encoder/decoder 残差栈，再吐出可 merge 的 delta weight。这让 adapter 能根据原层权重结构调整自己。

Classic LoRA is roughly input times A times B. PEANuT is weight-aware: it projects the base layer weight into a low-rank space, runs an encoder/decoder residual stack, and emits a mergeable delta weight adapted to that layer.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/peanut/layer.py`](https://github.com/huggingface/peft/blob/c702de402330f48c0d8b1f260d59413aa3110b65/src/peft/tuners/peanut/layer.py#L134-L231)

```python
def _compute_delta_weight(self, adapter: str, base_weight: torch.Tensor) -> torch.Tensor:
    if adapter not in self.peanut_A:
        raise ValueError(f"Adapter {adapter} not found.")

    peanut_A = self.peanut_A[adapter]
    peanut_B = self.peanut_B[adapter]
    non_linear = ACT2FN[self.act_fn[adapter]]
    scaling = self.scaling[adapter]
    res_num = self.res_num[adapter]
    peanut_encoders = self.peanut_encoders[adapter]
    peanut_decoders = self.peanut_decoders[adapter]

    base_weight_t = base_weight.transpose(0, 1).to(peanut_A.weight.dtype)
    delta_w = non_linear(torch.matmul(base_weight_t, peanut_A.weight.t()))

    residuals = []
    for i in range(res_num):
        residuals.append(delta_w)
        encoder = peanut_encoders[i]
        delta_w = non_linear(encoder(delta_w))

    for i in range(res_num):
        decoder = peanut_decoders[i]
        delta_w = non_linear(decoder(delta_w))
        delta_w = delta_w + residuals[res_num - 1 - i]

    delta_w = peanut_B(delta_w)
    return (delta_w * scaling).transpose(0, 1)

def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    adapter_names = check_adapters_to_merge(self, adapter_names)
    if not adapter_names:
        return

    base_layer = self.get_base_layer()
    merge_base_weight = self._get_base_weight_before_merge()

    for active_adapter in adapter_names:
        if active_adapter not in self.peanut_A:
            continue

        with torch.no_grad():
            delta_weight = self._compute_delta_weight(active_adapter, merge_base_weight)
            delta_weight = delta_weight.to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)

            if safe_merge:
                orig_weights = base_layer.weight.data.clone()
                orig_weights = orig_weights + delta_weight

                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = orig_weights
            else:
                base_layer.weight.data.add_(delta_weight)

        self.merged_adapters.append(active_adapter)

def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    result = self.base_layer(x, *args, **kwargs)
    if self.active_adapters:
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.peanut_A:
                continue
            delta_weight = self.get_delta_weight(active_adapter)
            x_cast = self._cast_input_dtype(x, delta_weight.dtype)
            delta = torch.matmul(x_cast, delta_weight.transpose(0, 1))
            result = result + delta.to(torch_result_dtype)
```

## 逐行讲解 / What's happening

1. **第 146-148 行 / Lines 146-148 (base-aware projection)**:
   - 中文: adapter 先读 `base_weight`，这和普通 LoRA 只读输入 `x` 不一样。
   - English: The adapter first reads `base_weight`, unlike plain LoRA which only transforms the input `x`.
2. **第 149-158 行 / Lines 149-158 (encoder/decoder residuals)**:
   - 中文: encoder 压缩变化，decoder 反向展开，并把早期残差接回来。
   - English: The encoder transforms the low-rank representation, the decoder expands it back, and skip residuals restore earlier features.
3. **第 179-196 行 / Lines 179-196 (merge)**:
   - 中文: 推理前可以把 delta 直接加进 base weight，`safe_merge` 会先检查 NaN。
   - English: Before inference, the delta can be added into the base weight; `safe_merge` validates that the merged weights stay finite.
4. **第 218-230 行 / Lines 218-230 (runtime path)**:
   - 中文: 不 merge 时，每次 forward 都算 `x @ delta_weight.T`，再加到 base layer 输出上。
   - English: Without merging, each forward computes `x @ delta_weight.T` and adds it to the base-layer output.

## 类比 / The analogy

普通 LoRA 像给每扇门贴同一种门缝条；PEANuT 会先量门框形状，再裁一条更贴合这扇门的门缝条。

Plain LoRA is like using the same weather strip on every door. PEANuT measures the door frame first, then cuts a strip that matches that specific door.

## 自己跑一遍 / Try it yourself

```python
base = [[1.0, 2.0], [3.0, 4.0]]
A = [[0.5, -0.5]]
B = [[2.0], [1.0]]
delta = []
for row in zip(*base):
    hidden = max(0.0, sum(a * b for a, b in zip(row, A[0])))
    delta.append([hidden * B[0][0], hidden * B[1][0]])
delta_t = list(map(list, zip(*delta)))
print(delta_t)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0.0, 0.0], [0.0, 0.0]]
```

这个例子故意让投影被 ReLU 截断，展示了“delta weight 是由 base weight 计算出来的”。

This toy projection is clipped by ReLU on purpose, showing that the delta weight is computed from the base weight.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DoRA** / **DoRA**: 把方向和幅值分开，而不是只加低秩 delta。 / It separates direction and magnitude instead of only adding a low-rank delta.
- **UniLoRA** / **UniLoRA**: 用共享参数和索引重建 adapter 矩阵。 / It rebuilds adapter matrices from shared parameters and indices.

## 注意事项 / Caveats / when it breaks

- **merge 依赖缓存的 base weight** / **Merge depends on cached base weight**: 多次 merge/unmerge 要非常小心，否则可能叠加错误 delta。 / Repeated merge/unmerge needs care or the wrong delta can accumulate.
- **计算比 LoRA 重** / **Heavier than LoRA**: 每层先从 base weight 生成 delta，训练和未 merge 推理都会更贵。 / Generating deltas from base weights costs more during training and unmerged inference.

## 延伸阅读 / Further reading

- [PEFT repository](https://github.com/huggingface/peft)
- [PEFT adapter methods](https://huggingface.co/docs/peft)
