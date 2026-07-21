---
date: 2026-07-21
topic: infrastructure
source: tracked
repo: Dao-AILab/flash-attention
file: flash_attn/flash_attn_interface.py
permalink: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L2361-L2466
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, flash-attention, custom-op, fake-tensor]
---

# FlashAttention custom op：真实 kernel 和 fake shape 分开注册 / FlashAttention Custom Op: Register the Real Kernel and the Fake Shape Path Separately

> **一句话 / In one line**: FlashAttention 把 CUDA forward 注册成 `torch.library.custom_op`，再单独注册 fake 实现，让编译器知道输出 shape 而不用真的跑 kernel。 / FlashAttention registers the CUDA forward as a `torch.library.custom_op`, then registers a separate fake implementation so compilers can infer output shapes without launching the kernel.

## 为什么重要 / Why this matters

高性能 kernel 通常不是纯 PyTorch 图。要让 `torch.compile`、fake tensor 和导出工具理解它，项目需要告诉 PyTorch 两件事：真实执行时调用哪个 CUDA op，形状推断时应该返回什么样的空 tensor。FlashAttention 的接口层正好展示了这个边界。

High-performance kernels are usually not pure PyTorch graphs. To make `torch.compile`, fake tensors, and export tooling understand them, the project must tell PyTorch two things: which CUDA op runs at execution time, and what empty tensors represent the outputs during shape inference. FlashAttention's interface layer shows that boundary.

## 代码 / The code

`Dao-AILab/flash-attention` — [`flash_attn/flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L2361-L2466)

```python
# Simplified teaching slice, not a verbatim copy.
custom_op("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def forward(q, k, v, ...):
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
    return flash_attn_gpu.fwd(q, k, v, ...)

register_fake("flash_attn::_flash_attn_forward")
def forward_fake(q, k, v, return_softmax, ...):
    batch, seqlen_q, heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = empty_like(q)
    softmax_lse = empty((batch, heads, seqlen_q), dtype=float32)
    p = empty((batch, heads, seqlen_q, seqlen_k)) if return_softmax else empty((0,))
    return out, softmax_lse, p, rng_state
```

## 逐行讲解 / What's happening

1. **注册真实入口 / Register the real entry**: 中文: custom op 名字成为 PyTorch dispatcher 里的稳定符号。 English: the custom-op name becomes a stable symbol in the PyTorch dispatcher.
2. **声明不改参数 / Declare no input mutation**: 中文: `mutates_args=()` 告诉图变换这些输入不会被原地写坏。 English: `mutates_args=()` tells graph transforms that inputs are not modified in place.
3. **先整理 strides / Normalize strides first**: 中文: CUDA kernel 通常要求最后一维连续，所以接口层先做 contiguous 保护。 English: CUDA kernels often require contiguous inner dimensions, so the interface normalizes layout first.
4. **fake 路径只造空壳 / Fake path builds shells only**: 中文: fake 实现不算 attention，只根据 Q/K/V 的 shape 构造输出 tensor。 English: the fake implementation does not compute attention; it constructs output tensors from Q/K/V shapes.
5. **可选 softmax 输出 / Optional softmax output**: 中文: `return_softmax` 决定概率矩阵是完整形状还是空 tensor。 English: `return_softmax` decides whether the probability tensor has full shape or is empty.

## 类比 / The analogy

像港口同时有真实装卸码头和纸面报关窗口。货船来了走码头；规划系统只想知道箱子尺寸时走报关窗口，不需要真的搬箱子。

It is like a port with both a real loading dock and a paperwork desk. Ships use the dock; planning software uses the paperwork desk to learn container sizes without moving cargo.

## 自己跑一遍 / Try it yourself

```python
def fake_flash_shapes(q_shape, k_shape, return_softmax):
    batch, seqlen_q, heads, head_size = q_shape
    seqlen_k = k_shape[1]
    out = q_shape
    lse = (batch, heads, seqlen_q)
    probs = (batch, heads, seqlen_q, seqlen_k) if return_softmax else (0,)
    return out, lse, probs

print(fake_flash_shapes((2, 5, 8, 64), (2, 7, 8, 64), False))
print(fake_flash_shapes((2, 5, 8, 64), (2, 7, 8, 64), True))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
((2, 5, 8, 64), (2, 8, 5), (0,))
((2, 5, 8, 64), (2, 8, 5), (2, 8, 5, 7))
```

## 注意事项 / Caveats / when it breaks

- **fake shape 要和真实 kernel 对齐 / Fake shape must match the real kernel**: 形状错了，compile/export 会提前相信一个假合同。 / A wrong fake shape lets compile/export trust a false contract.
- **layout 不是小事 / Layout is not a detail**: kernel wrapper 里的 contiguous 处理是 ABI 的一部分。 / contiguous handling in the wrapper is part of the ABI.
- **版本分支要保留 / Keep version branches**: 老 PyTorch 没有同一套 custom-op API，所以 fallback wrapper 仍然重要。 / older PyTorch versions do not expose the same custom-op API, so fallback wrappers still matter.

## 延伸阅读 / Further reading

- [FlashAttention interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L2361-L2466)
- [PyTorch custom operators](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
