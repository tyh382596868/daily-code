---
date: 2026-09-02
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/attention.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/attention.py#L22-L170
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, dit-block, flash-attention, variable-length]
build_role: dit-block advanced variant
---

# Wan2.1 Flash Attention 包装器：先压平变长 token，再交给 kernel / Wan2.1 Flash Attention Wrapper: Flatten Variable-Length Tokens Before the Kernel

> **一句话 / In one line**: 这层 wrapper 负责把变长 q/k/v 打包成 FlashAttention 需要的形状，再在 FA2/FA3/SDPA 之间选路。 / This wrapper packs variable-length q/k/v into the shape FlashAttention wants, then routes between FA2, FA3, and SDPA.

## 为什么重要 / Why this matters

中文：DiT block 的注意力层往往面对“每个视频长度不同、每个样本 token 数不同”的现实。Wan2.1 没有把这些脏活散在上层 block 里，而是集中到一个 attention wrapper：先按长度裁掉 padding，再构造累积长度前缀，最后把输出恢复成 batch 形状。这样上层只关心语义，下层只关心 kernel。

English: A DiT block often deals with the real-world mess of variable video lengths and per-sample token counts. Wan2.1 does not scatter that logic across the higher-level block; it centralizes it in one attention wrapper. It trims padding by length, builds cumulative sequence offsets, and restores the batch shape after the kernel returns. The upper layer stays semantic; the lower layer stays kernel-specific.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/attention.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/attention.py#L22-L170)

```python
def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
```

## 逐行讲解 / What's happening

1. **第 22-52 行 / Lines 22-52**:
   - 中文: `flash_attention()` 先断言 CUDA 和半精度类型，然后决定是否需要把输入转换成统一 dtype。
   - English: `flash_attention()` first asserts CUDA and half precision, then decides whether the inputs need a dtype cast.
2. **第 59-76 行 / Lines 59-76**:
   - 中文: `q_lens` / `k_lens` 决定是直接 flatten 全 batch，还是把每个样本的有效 token 截出来再拼接。
   - English: `q_lens` / `k_lens` decide whether to flatten the full batch or to slice each sample’s valid tokens before concatenation.
3. **第 88-123 行 / Lines 88-123**:
   - 中文: 真正的 kernel 只吃 packed token 和 cumulative lengths；返回后再 `unflatten` 回 `(B, L, ...)`。
   - English: The kernel consumes packed tokens plus cumulative lengths, and the result is then `unflatten`ed back to `(B, L, ...)`.
4. **第 125-170 行 / Lines 125-170**:
   - 中文: `attention()` 负责做后端路由；如果 flash-attn 不可用，就退回 PyTorch SDPA。
   - English: `attention()` does backend routing; if flash-attn is unavailable, it falls back to PyTorch SDPA.

## 类比 / The analogy

中文：像把不同长度的火车车厢先按站台规则重新挂成一列，再交给调度中心一次性发车。调度中心不想知道每节车厢最初有多长，只想看到规整的编组表。

English: It is like re-coupling rail cars of different lengths into one train by station rules before sending the set to a dispatch center. The dispatcher does not care how long each car used to be; it only wants a clean consist sheet.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文：这是 `dit-block` 的 advanced variant。它对应的是每个 block 里“QKV 之后、输出投影之前”的注意力内核层。输入是 packed q/k/v 和长度信息，输出是恢复 batch 形状后的 attention 结果。省掉这层，block 上层就得自己处理 padding、变长长度、FA2/FA3/SDPA 兼容和 dtype 约束；生产版通常还会再补更细的 kernel 选择、更多 mask 形式和跨设备缓存策略。

English: This is an advanced variant of `dit-block`. It maps to the attention kernel inside each block, after QKV projection and before the output projection. Inputs are packed q/k/v plus length metadata; output is the attention result restored to batch shape. Without it, the higher-level block would have to manage padding, variable lengths, FA2/FA3/SDPA compatibility, and dtype constraints itself. A production version adds finer kernel selection, more mask forms, and cross-device cache policies.

## 自己跑一遍 / Try it yourself

```python
def pack(seq_lens, rows):
    packed = []
    for row, n in zip(rows, seq_lens):
        packed.extend(row[:n])
    return packed

print(pack([2, 1], [[1, 2, 3], [4, 5, 6]]))
print("backend", "flash-attn" if True else "sdpa")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 2, 4]
backend flash-attn
```

中文：最核心的信号是“长度先收口，kernel 才能高效工作”。

English: The key signal is that lengths have to be packed first before the kernel can work efficiently.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch varlen attention** / **PyTorch varlen attention**: 中文: 变长注意力也会先把有效 token 和长度前缀打包。 / English: Variable-length attention also packs valid tokens plus prefix lengths first.
- **LeRobot / openpi inference loops** / **LeRobot / openpi inference loops**: 中文: rollout 里常把变长观测整理成统一 batch 再喂模型。 / English: Rollout loops often normalize variable observations into one batch before the model sees them.

## 注意事项 / Caveats / when it breaks

- **FA3 还不支持所有选项 / FA3 does not support every option yet**: 中文: 例如 dropout 和 window_size 在注释里就被点名限制了。 / English: For example, the comments explicitly note limits around dropout and window_size.
- **`q.device.type == 'cuda'` 是硬门槛 / `q.device.type == 'cuda'` is a hard gate**: 中文: 这不是 CPU 版通用实现。 / English: This is not a general CPU implementation.

## 延伸阅读 / Further reading

- Wan2.1 source: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/attention.py
