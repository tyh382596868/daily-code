---
date: 2026-06-08
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/experimental/_scaled_dot_product_attention_quantized.py
permalink: https://github.com/pytorch/pytorch/blob/64bdb99cd3a3987bbe05c0eda339f2f788bf7f33/torch/nn/attention/experimental/_scaled_dot_product_attention_quantized.py#L14-L154
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, attention, fp8, quantization]
---

# PyTorch 把 FP8 attention 写进了官方:一个 154 行的 SDPA 量化 wrapper / PyTorch shipped FP8 attention to core: a 154-line quantized SDPA wrapper

> **一句话 / In one line**: query / key / value 直接进来就是 `float8_e4m3fn`,再传三张 `(B, num_kv_heads)` 的 float32 descale 张量,PyTorch 把它们一起塞进 FA3 的量化路径。 / Pass `query/key/value` already in `float8_e4m3fn` together with three `(B, num_kv_heads)` float32 descale tensors, and PyTorch routes them straight to FA3's quantized fused-attention path.

## 为什么重要 / Why this matters

到 2026 年,主流推理栈 (vLLM / SGLang / TensorRT-LLM) 都已经把 KV cache 和 attention 矩阵搬到了 FP8。PyTorch 这次把它写进官方 `torch.nn.attention.experimental` 之后,任何使用 `torch.compile` 的模型都能拿到这条 FP8 fast path——而不用自己去写一个 ext-op 或者重新链接 FA3。这份 154 行的文件展示了 PyTorch 引入新 dtype op 时的「公开 API 最小骨架」:一个 IntEnum 描述将来可能扩展的 granularity、一个 validate-and-route 工具函数、一个把参数转给 ATen op 的 thin wrapper——加起来就是新算子的标准模板。

By 2026 the mainstream inference stacks (vLLM, SGLang, TensorRT-LLM) have moved KV cache and attention matmuls to FP8. With this commit, PyTorch ships the FP8 attention fast path inside `torch.nn.attention.experimental`, so any model running through `torch.compile` can pick it up — no external op, no relinking of FA3. The 154-line file is the canonical skeleton for introducing a new datatype op into PyTorch core: an `IntEnum` that names the granularity (so the API can grow without breaking), one validate-and-route helper, and a thin wrapper that forwards into the ATen op.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/experimental/_scaled_dot_product_attention_quantized.py`](https://github.com/pytorch/pytorch/blob/64bdb99cd3a3987bbe05c0eda339f2f788bf7f33/torch/nn/attention/experimental/_scaled_dot_product_attention_quantized.py#L14-L154)

```python
class DescaleType(IntEnum):
    """Describes the scaling granularity for FP8 descale tensors.

    Used with _scaled_dot_product_attention_quantized to explicitly specify
    how the descale factors are applied to the quantized inputs.
    """

    PER_HEAD = 0
    """Per-head descaling. Descale tensor shape: (batch_size, num_kv_heads)."""


def _validate_descale(
    descale: Tensor | None,
    name: str,
    query: Tensor,
    key: Tensor,
    descale_type: DescaleType,
) -> None:
    if descale is None:
        return

    if descale.dtype != torch.float32:
        raise ValueError(f"{name}_descale must have dtype float32, got {descale.dtype}")

    if not descale.is_cuda:
        raise ValueError(f"{name}_descale must be a CUDA tensor")

    if descale_type == DescaleType.PER_HEAD:
        batch_size = query.size(0)
        # All descale tensors use num_kv_heads, even q_descale (broadcast internally)
        num_kv_heads = key.size(1)

        if descale.dim() != 2:
            raise ValueError(
                f"{name}_descale must be a 2D tensor with shape (batch_size, num_kv_heads)"
            )
        if descale.size(0) != batch_size or descale.size(1) != num_kv_heads:
            raise ValueError(f"{name}_descale shape mismatch")


def _scaled_dot_product_attention_quantized(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    q_descale: Tensor | None = None,
    k_descale: Tensor | None = None,
    v_descale: Tensor | None = None,
    q_descale_type: DescaleType = DescaleType.PER_HEAD,
    k_descale_type: DescaleType = DescaleType.PER_HEAD,
    v_descale_type: DescaleType = DescaleType.PER_HEAD,
) -> Tensor:
    """Scaled dot product attention for FP8 inputs.

    Args:
        query (Tensor): shape (N, H_q, L, E), dtype float8_e4m3fn
        key   (Tensor): shape (N, H,   S, E), dtype float8_e4m3fn
        value (Tensor): shape (N, H,   S, E_v), dtype float8_e4m3fn
        q_descale / k_descale / v_descale (Tensor): shape (N, H), dtype float32
    """
    _validate_descale(q_descale, "q", query, key, q_descale_type)
    _validate_descale(k_descale, "k", query, key, k_descale_type)
    _validate_descale(v_descale, "v", query, key, v_descale_type)

    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        warnings.warn(
            "_scaled_dot_product_attention_quantized does not support backward pass.",
            UserWarning,
        )
    # Directly call the internal flash attention operator which has descale support
    result = torch.ops.aten._scaled_dot_product_flash_attention.quantized(
        query, key, value,
        q_descale, k_descale, v_descale,
        0.0, is_causal, False, scale=scale,
    )
    return result[0]
```

## 逐行讲解 / What's happening

1. **`class DescaleType(IntEnum): PER_HEAD = 0`**
   - 中文: 把「descale 的颗粒度」做成枚举而不是 bool/string,是为了将来再加 `PER_TENSOR` / `PER_BLOCK` / `PER_TOKEN` 时 API 不破坏——这是 PyTorch 推 stable API 的标准做法。
   - English: Modelling "descale granularity" as an `IntEnum` (rather than a `bool` or string) leaves room to add `PER_TENSOR` / `PER_BLOCK` / `PER_TOKEN` later without breaking call sites — the canonical PyTorch pattern for a stable, growable API surface.

2. **`_validate_descale(descale, "q", query, key, PER_HEAD)`**
   - 中文: 验证三件事:dtype 必须是 `float32` (descale 自己不能也是 fp8,否则就没意义了)、必须在 CUDA 上、形状 `(B, num_kv_heads)`。
   - English: Three checks: dtype must be `float32` (you can't descale fp8 with an fp8 number — that would defeat the point), tensor must live on CUDA, shape must be `(B, num_kv_heads)`.

3. **`num_kv_heads = key.size(1)`** ←(注释里的关键 detail / the key detail in the comment)
   - 中文: GQA 里 `num_query_heads > num_kv_heads`,但 `q_descale` 也用 `num_kv_heads` 而不是 `num_query_heads`——q-side 的 broadcast 在 FA3 内核里做。这是 GQA 模型移植到 FP8 时最容易踩的坑。
   - English: Under GQA, `num_query_heads > num_kv_heads`. Even so, `q_descale` is still shaped on `num_kv_heads`; FA3 internally broadcasts to `num_query_heads`. Get this shape wrong and your descale factors silently misalign across heads.

4. **`if torch.is_grad_enabled() and (... requires_grad): warnings.warn(...)`**
   - 中文: 不是 raise,而是 warn——因为 fp8 attention 的 backward kernel 还没写出来,但 PyTorch 选择让你能跑前向 (推理场景) 而不是直接拒绝。这是「实验 API」的标准放行姿态。
   - English: Not a `raise` — a `warn`. The fp8 attention backward kernel doesn't exist yet, but PyTorch deliberately keeps the forward path open so inference jobs can use it today. Standard "experimental API" posture: degrade loudly, don't refuse.

5. **`torch.ops.aten._scaled_dot_product_flash_attention.quantized(...)`**
   - 中文: 直接走 ATen 上注册的 `.quantized` overload,而不是 `torch._scaled_dot_product_flash_attention`——注释解释了原因:后者跟 `torch.compile` 不兼容 (dynamo 把高层 Python 函数 inline 掉之后 graph 里就找不到 fp8 路径)。直接走 ATen op 让 Inductor 看见这一帧。
   - English: It goes straight to ATen's `.quantized` overload, not the `torch._scaled_dot_product_flash_attention` Python entry. The comment in the file explains why: the higher-level Python function gets inlined by Dynamo, after which the FP8 path is no longer visible in the graph. Calling the ATen op directly keeps the frame visible to Inductor for fusion.

6. **`return result[0]`**
   - 中文: ATen op 返回多元组 `(out, logsumexp, …)`,只取 `out` 是为了和公开的 `scaled_dot_product_attention` 接口一致——logsumexp 是给 backward 准备的,fp8 没 backward,丢掉它无所谓。
   - English: The ATen op returns `(out, logsumexp, …)`; this wrapper returns just `out` to mirror the public `scaled_dot_product_attention` signature. `logsumexp` is only needed for backward — fp8 has no backward yet, so it's safe to drop.

## 类比 / The analogy

想象你寄一个易碎的玻璃花瓶 (浮点矩阵)。直接寄过去会碎 (内存放不下);你先把它装进一个 fp8 的紧实箱子里,每个箱子外面贴一张标签 (descale tensor):「这个箱子里的数原本要除以 0.0073」。FA3 内核拆箱时,先按标签把每个箱子的数变回原来的大小,再算 attention。`PER_HEAD` 是说每颗头一个箱子标签,而不是整个 batch 一张标签——颗粒度细一点,精度损失更小,但需要的 metadata 也更多。

Picture mailing a fragile glass vase (a float tensor). Ship it as-is and it shatters (won't fit in memory). So you crush it into a tight fp8 box and stick a label on each box (the descale tensor): "the number inside was originally scaled by 0.0073." When FA3 unpacks the box, it multiplies by the label first, then runs attention. `PER_HEAD` means *one label per head*, not one for the whole batch — finer granularity, smaller precision loss, but more metadata to carry.

## 自己跑一遍 / Try it yourself

```python
# pip install torch>=2.6 ; requires CUDA + Hopper (H100) + FA3 backend
import torch
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized as fp8_sdpa,
)

B, Hq, Hkv, L, E = 1, 8, 2, 128, 64       # GQA: 8 query heads, 2 KV heads
amax = 448.0                              # e4m3 max magnitude

q = torch.randn(B, Hq,  L, E, device="cuda") / 16
k = torch.randn(B, Hkv, L, E, device="cuda") / 16
v = torch.randn(B, Hkv, L, E, device="cuda") / 16

# per-head descale: pick scale so |x_fp8 * scale| roughly recovers |x_fp32|
q_scale = q.abs().amax(dim=(0, 2, 3)) / amax           # (Hq,)
# IMPORTANT: descale uses Hkv, not Hq — broadcast happens inside FA3
q_descale = q_scale.view(B, Hq).mean(dim=1, keepdim=True).expand(B, Hkv)
k_descale = (k.abs().amax(dim=(0, 2, 3)) / amax).view(B, Hkv)
v_descale = (v.abs().amax(dim=(0, 2, 3)) / amax).view(B, Hkv)

q8 = (q / q_descale[..., None, None].unsqueeze(1).expand(-1, Hq, -1, -1)).to(torch.float8_e4m3fn)
k8 = (k / k_descale[..., None, None]).to(torch.float8_e4m3fn)
v8 = (v / v_descale[..., None, None]).to(torch.float8_e4m3fn)

out = fp8_sdpa(q8, k8, v8, is_causal=True, q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
print(out.shape, out.dtype)   # → torch.Size([1, 8, 128, 64]) torch.bfloat16
```

运行 / Run with:
```bash
pip install --pre torch
python try.py
```

预期输出 / Expected output:
```
torch.Size([1, 8, 128, 64]) torch.bfloat16
```

中文:注意输出 dtype 是 `bfloat16`——FA3 内部在 bf16 上累加,再返回 bf16,这样调用方不用再 dequant。

English: Notice the output dtype is `bfloat16` — FA3 accumulates in bf16 internally and returns bf16, so the caller never has to dequant.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NVIDIA TransformerEngine** / **NVIDIA TransformerEngine**: `te.fp8_autocast` 是把整个 transformer block 包起来,内部对每个 linear/attention 都用同样的 per-head/per-tensor descale 协议。 / `te.fp8_autocast` wraps a whole transformer block; internally each linear and attention call uses the same per-head / per-tensor descale protocol.
- **vLLM 的 W8A8 inference** / **vLLM's W8A8 inference**: vLLM 在生成 KV cache 时直接写 fp8,attention 调用同样的 FA3 quantized overload,只是从用户角度封装在 `Quantized*Attention` 类里。 / vLLM stores fp8 KV cache and calls the same FA3 quantized overload, just behind a `Quantized*Attention` class.
- **Flash-Attention 3 自己** / **Flash-Attention 3 itself**: `flash_attn_func_fp8(q, k, v, q_descale, k_descale, v_descale)` 几乎一比一对应这里的接口——PyTorch 这版只是把它搬进了官方而已。 / `flash_attn_func_fp8(q, k, v, q_descale, k_descale, v_descale)` is a near-1:1 mirror — PyTorch is just pulling it into core.

## 注意事项 / Caveats / when it breaks

- **GQA 的 q_descale 必须用 num_kv_heads** / **GQA q_descale MUST use num_kv_heads**: 用 num_query_heads 会触发形状检查报错;就算绕过去 (例如 `Hkv == Hq`),fp8 dequant 的对齐也会错。 / Using `num_query_heads` either trips the shape check, or — if `Hkv == Hq` — silently mis-aligns the dequant.
- **只支持前向** / **Forward only**: 训练时把这个函数包进 `with torch.no_grad():`,否则 warning 会被刷成噪音。 / Wrap in `torch.no_grad()` during training, or you'll get the warning on every forward.
- **e4m3 的动态范围只有 ±448** / **e4m3 dynamic range is ±448**: 选 descale 时要让 `x / descale` 落在 [−448, 448] 里,否则会 saturate 出 `inf`——这是为什么真实推理栈会维护一个动态校准的 amax 表。 / Pick `descale` so `x / descale` lands in `[−448, 448]`, otherwise you saturate to `inf`. That's why real inference stacks keep a dynamic amax calibration table per layer.
- **依赖 FA3 后端** / **Requires the FA3 backend**: 只在 H100 / B100 这种支持 FP8 的 GPU 上跑得起来;A100 跑不了。 / Only works on FP8-capable GPUs (H100 / B100). A100 will raise at op-dispatch time.

## 延伸阅读 / Further reading

- [PyTorch SDPA dispatcher overview](https://docs.pytorch.org/docs/stable/nn.attention.html)
- [Flash-Attention 3: FP8 attention paper (Tri Dao et al., 2024)](https://tridao.me/publications/flash3/flash3.pdf)
- [NVIDIA blog on fp8 e4m3 vs e5m2](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
