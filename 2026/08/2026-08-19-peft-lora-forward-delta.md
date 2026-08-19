---
date: 2026-08-19
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/lora/layer.py
permalink: https://github.com/huggingface/peft/blob/a9d039ab503b9509a79c2770746b188a9d3f179d/src/peft/tuners/lora/layer.py#L1005-L1076
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, lora]
---

# PEFT LoRA forward：base 先算，adapter 再补差量 / PEFT LoRA Forward: Run the Base, Then Add the Adapter Delta

> **一句话 / In one line**: PEFT 的 LoRA 层先调用原始层，再把每个 active adapter 的 `B(A(dropout(x))) * scaling` 加到结果上。 / PEFT's LoRA layer runs the original layer first, then adds each active adapter's `B(A(dropout(x))) * scaling` delta.

## 为什么重要 / Why this matters

LoRA 的工程价值不只是“低秩矩阵”这个公式，而是它能在不改 base layer 权重的情况下插入、关闭、混合、合并多个 adapter。PEFT 的 forward 路径把这些状态分支写得很清楚：disable、merged、mixed batch、普通 active adapters，每种模式都不该互相污染。

LoRA's practical value is not only the low-rank formula; it is the ability to insert, disable, mix, and merge adapters without rewriting the base layer. PEFT's forward path makes those runtime states explicit: disabled adapters, merged adapters, mixed-batch adapters, and normal active adapters all take different branches.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/lora/layer.py`](https://github.com/huggingface/peft/blob/a9d039ab503b9509a79c2770746b188a9d3f179d/src/peft/tuners/lora/layer.py#L1005-L1076)

```python
def get_delta_weight(self, adapter) -> torch.Tensor:
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

    return output_tensor

def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        lora_A_keys = self.lora_A.keys()
        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = self._cast_input_dtype(x, lora_A.weight.dtype)
            if active_adapter not in self.lora_variant:  # vanilla LoRA
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                result = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x,
                    result=result,
                    **variant_kwargs,
                    **kwargs,
                )

        result = result.to(torch_result_dtype)

    return result
```

## 逐行讲解 / What's happening

1. **第 1005-1033 行 / Lines 1005-1033 (`get_delta_weight`)**:
   - 中文: 合并 adapter 时才需要显式算 `B @ A`。CPU 上的低精度矩阵乘可能慢或不支持，所以先升到 fp32。
   - English: Explicit `B @ A` is needed when merging adapters into base weights. Low-precision CPU matmul can be slow or unsupported, so the code temporarily casts to fp32.
2. **第 1040-1047 行 / Lines 1040-1047 (state branches)**:
   - 中文: adapter 关闭或已经 merge 时，forward 直接走 base layer，避免重复加 LoRA。
   - English: If adapters are disabled or already merged, the forward path calls the base layer directly to avoid double-applying LoRA.
3. **第 1049-1063 行 / Lines 1049-1063 (vanilla LoRA)**:
   - 中文: 普通路径先算 base 结果，再对每个 active adapter 加低秩残差。
   - English: The normal path computes the base result first, then adds one low-rank residual per active adapter.
4. **第 1064-1072 行 / Lines 1064-1072 (variants)**:
   - 中文: DoRA 等变体可以接管 forward，但仍复用同一个 adapter 状态框架。
   - English: Variants such as DoRA can override the adapter math while reusing the same runtime state machine.

## 类比 / The analogy

像给一张原始照片加滤镜图层。底图先完整渲染；每个 LoRA adapter 是一层透明叠加；merge 就是把滤镜烘进底图，disable 就是把图层眼睛关掉。

It is like editing a photo with layers. The base image renders first; each LoRA adapter is a transparent adjustment layer; merge bakes it into the base image, while disable hides the layer.

## 自己跑一遍 / Try it yourself

```python
def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

x = [[2.0, 1.0]]
base_w = [[1.0, 0.0], [0.0, 1.0]]
A = [[0.5], [1.0]]
B = [[2.0, -1.0]]
scaling = 0.25

base = matmul(x, base_w)
hidden = matmul(x, A)
delta = [[v * scaling for v in row] for row in matmul(hidden, B)]
out = [[b + d for b, d in zip(base[0], delta[0])]]
print(base, delta, out)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[2.0, 1.0]] [[1.0, -0.5]] [[3.0, 0.5]]
```

中文: base 输出没有被替换，只是被 adapter 残差轻量修正。

English: The base output is not replaced; it is lightly corrected by the adapter residual.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers LoRA processors** / **Diffusers LoRA processors**: attention projection 也常用 base projection + LoRA delta。 / Attention projections often use the same base-plus-LoRA-delta pattern.
- **vLLM LoRA serving** / **vLLM LoRA serving**: 推理系统会把多个请求的 adapter delta 批量化，而不是复制多份模型。 / Serving systems batch adapter deltas across requests instead of duplicating whole models.

## 注意事项 / Caveats / when it breaks

- **dtype 来回转换** / **Dtype round trips**: 输入会 cast 到 adapter dtype，最后再 cast 回 base result dtype；混合精度排查时要看这两个边界。 / Inputs are cast to adapter dtype and outputs back to base result dtype; mixed-precision bugs often live at those boundaries.
- **merge 后别重复加** / **Do not double-apply after merge**: 如果 base weight 已经包含 delta，forward 不能再走 active adapter 加法。 / Once the base weight contains the delta, the forward path must not add it again.

## 延伸阅读 / Further reading

- [PEFT LoRA layer source](https://github.com/huggingface/peft/blob/a9d039ab503b9509a79c2770746b188a9d3f179d/src/peft/tuners/lora/layer.py#L1005-L1076)
- [PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)
