---
date: 2026-06-03
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/varlen.py
permalink: https://github.com/pytorch/pytorch/blob/9ab94917c245d16efe77f546d30d73800c8d728d/torch/nn/attention/varlen.py#L176-L326
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, pytorch, attention, flash-attention, paged-kv-cache, gqa]
---

# PyTorch 把 vLLM 的 paged KV cache 写进了核心 attention API / PyTorch's new `varlen_attn` brings paged KV-cache, GQA and split-KV into core attention

> **一句话 / In one line**: `torch.nn.attention.varlen_attn` 一个签名同时支持 packed 变长序列、paged KV cache、滑动窗口、GQA 和"位级可重现"的 split-KV —— 不再需要为了一种 attention 写一种 wrapper。 / One `varlen_attn` signature unifies packed variable-length sequences, paged KV-cache, sliding-window, GQA and bit-deterministic split-KV — no more one-wrapper-per-attention-flavor in user code.

## 为什么重要 / Why this matters

直到最近,PyTorch 用户想拿到 Flash-Attention 的全部本事都得直接 `pip install flash-attn` 调它的私有 API。`scaled_dot_product_attention` 只覆盖了 padded 4D 张量这种最朴素的形态,vLLM 的 paged KV cache、TGI 的滑动窗口、Llama-3 的 GQA 全都要自己拼。这次 `torch.nn.attention.varlen` 把这些**一次性塞进了核心库**:packed 张量 `(T_total, H, D)`,用 `cu_seq_q/cu_seq_k` 替代 padding mask;`block_table` 让 K/V 变成"页池",任何序列从任何页里拼出来 —— 就是 vLLM 那一套;`num_splits=1` 强制关闭 split-KV,保证位级确定性(这对 batch 不变性测试和评测复现是关键)。一个 API,一份代码,从训练到推理都能跑。

Until recently, getting Flash-Attention's full feature set in PyTorch meant `pip install flash-attn` and calling its private API. The built-in `scaled_dot_product_attention` only handled padded 4D tensors — the simplest case. vLLM's paged KV-cache, TGI's sliding window, Llama-3's GQA all had to be glued together in user space. `torch.nn.attention.varlen` rolls all of this **into the core library at once**: packed tensors `(T_total, H, D)` with `cu_seq_q/cu_seq_k` instead of padding masks; `block_table` turning K/V into a "page pool" any sequence can stitch itself together from — i.e. vLLM in 30 lines; `num_splits=1` to disable split-KV for bit-deterministic outputs (important for batch-invariance testing and eval reproducibility). One API, one signature, works in training *and* inference.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/varlen.py`](https://github.com/pytorch/pytorch/blob/9ab94917c245d16efe77f546d30d73800c8d728d/torch/nn/attention/varlen.py#L176-L326)

```python
def varlen_attn(
    query: torch.Tensor,                  # [T_q, H_q, D]
    key:   torch.Tensor,                  # [T_k, H_kv, D]  OR  [total_pages, page_size, H_kv, D] if block_table
    value: torch.Tensor,                  # same shape options as key
    cu_seq_q: torch.Tensor,               # [N+1]  cumulative q offsets
    cu_seq_k: torch.Tensor | None,        # [N+1]  cumulative k/v offsets
    max_q: int,
    max_k: int,
    *,
    return_aux: AuxRequest | None = None,
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),   # (-1, -1)=full, (-1, 0)=causal, (W, 0)=sliding causal
    enable_gqa: bool = False,                  # H_kv < H_q allowed if H_q % H_kv == 0
    seqused_k: torch.Tensor | None = None,     # [N]  actual valid kv tokens per seq (KV-cache padding)
    block_table: torch.Tensor | None = None,   # [N, max_pages_per_seq] int32 (paged KV)
    num_splits: int | None = None,             # 1 = disable split-KV for bit determinism
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Compute variable-length attention using Flash Attention.

    Args:
        cu_seq_q: cumulative sequence positions for queries; shape (N+1,)
        block_table: when provided, ``key`` / ``value`` are a "pool" of
            pages of KV data; the block_table maps each sequence's
            logical chunks back to physical pages.
            ``seqused_k[i]`` tells the kernel how many tokens in sequence i
            are actually valid, since the last page is typically partial.
        num_splits: 1 disables split-KV, which enables batch invariance.
            Split-KV parallelizes the key/value sequence dimension across
            thread blocks. The split decision depends on ``max_k`` (longest
            sequence), so different batch compositions can change the
            reduction order and produce different floating-point results.
            With num_splits=1, bitwise identical outputs are guaranteed for
            a given sequence regardless of what else is in the batch.

    Example:
        >>> batch_size, max_seq_len, embed_dim, num_heads = 2, 512, 1024, 16
        >>> head_dim = embed_dim // num_heads
        >>> seq_lengths = torch.tensor([320, 192], device="cuda")
        >>> total_tokens = seq_lengths.sum().item()
        >>> q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
        >>> k = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
        >>> v = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
        >>> cu_seq = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        >>> cu_seq[1:] = seq_lengths.cumsum(0)
        >>> max_len = seq_lengths.max().item()
        >>> output = varlen_attn(q, k, v, cu_seq, cu_seq, max_len, max_len)
    """
    num_heads_q = query.size(1)
    num_heads_k = key.size(2) if block_table is not None else key.size(1)
    if not enable_gqa and num_heads_q != num_heads_k:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa and num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"Expect number of query heads to be a multiple of kv heads for GQA "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}."
        )

    is_causal = window_size == (-1, 0)
    out, lse, _ = torch.ops.torch_attn._varlen_attn(
        query, key, value,
        cu_seq_q, cu_seq_k, max_q, max_k,
        is_causal, scale, list(window_size),
        enable_gqa, seqused_k, block_table, num_splits,
    )
    if return_aux is not None and return_aux.lse:
        return out, lse
    return out
```

## 逐行讲解 / What's happening

1. **Packed shape `(T_q, H_q, D)`(没有 batch 维 / no batch dim)**:
   - 中文: 这里**没有** `B` 维度,N 条序列首尾相接拼成一个超长 tensor。`cu_seq_q = [0, T0, T0+T1, T0+T1+T2, ...]` 告诉 kernel 每条序列在哪里开始结束。少了 padding 就少了 padding mask,也少了浪费的算力,这是 Flash-Attention 当年最大的卖点之一。
   - English: there is **no batch dim** — N sequences are concatenated head-to-tail into one big tensor. `cu_seq_q = [0, T0, T0+T1, T0+T1+T2, ...]` tells the kernel where each sequence starts and ends. No padding means no padding mask and no wasted compute — one of Flash-Attention's original selling points.

2. **`block_table` + `seqused_k` —— paged KV cache 入门 / `block_table` + `seqused_k` — paged KV in two args**:
   - 中文: 当 `block_table` 是 `None` 时,`key/value` 就是普通的连续 packed 张量。一旦传了 `block_table`,K/V 变成 `[total_pages, page_size, H_kv, D]` 的**页池**,`block_table[i]` 列出序列 i 的物理页号。这就是 vLLM 的工作方式 —— 一个 16 token 的小回复和一个 2048 token 的长回复可以共享同一个池子里的页,绝不浪费显存。`seqused_k[i]` 告诉 kernel 序列 i 在最后一页只用了多少 token(因为最后一页通常是半满的)。
   - English: with `block_table=None`, `key/value` are ordinary packed contiguous tensors. Pass a `block_table` and K/V become a `[total_pages, page_size, H_kv, D]` *page pool*; `block_table[i]` lists the physical page indices for sequence i. This is exactly how vLLM works — a 16-token short reply and a 2048-token long reply share pages from the same pool, never wasting VRAM. `seqused_k[i]` tells the kernel how many tokens in sequence i's last page are actually valid, since the last page is usually partially filled.

3. **`window_size` 的三个魔法元组 / Three magic tuples in `window_size`**:
   - 中文: `(-1, -1)` 是全 attention;`(-1, 0)` 是 causal;`(W, 0)` 是 **causal + 滑动窗口大小 W**(Mistral / SWA 那一套)。一个参数同时覆盖三种最常见的 mask,而且不用真的物化 mask 张量 —— Flash kernel 在循环里直接跳过越界的 KV。
   - English: `(-1, -1)` means full attention; `(-1, 0)` means causal; `(W, 0)` means **causal + sliding window of size W** (Mistral / SWA style). One parameter spans the three most common masks, and the kernel never materializes the mask — it just skips out-of-window KV inside its loops.

4. **GQA 校验 / GQA validation**:
   - 中文: GQA(Llama-3 / Mistral)的本质是 `H_kv < H_q`,每个 KV 头被 `H_q / H_kv` 个 Q 头共享。`enable_gqa=True` 时 kernel 内部做 head replication,但 PyTorch 这层先校验整除性,失败给出可操作的错误信息("Try setting enable_gqa=True")。这种**带建议的报错**对新手友好度极高。
   - English: GQA (Llama-3 / Mistral) means `H_kv < H_q`, with each KV head shared by `H_q / H_kv` Q heads. With `enable_gqa=True` the kernel handles head replication, but the Python layer pre-validates divisibility and returns an **actionable** error ("Try setting enable_gqa=True") — beginner-friendly compared to a kernel-side `INVALID_ARGUMENT`.

5. **`num_splits=1` 的位级确定性 / `num_splits=1` for bit-determinism**:
   - 中文: split-KV 是 Flash-Attention 的一个性能 trick —— 把 K/V 序列维切成 `num_splits` 份并行,最后拿 log-sum-exp 合并。问题是切分数取决于 `max_k`,而 `max_k` 取决于这个 batch 里**最长的那条**。换 batch 组合就换 split,浮点 reduction 顺序就变,结果就在最末几位 bit 上漂移。`num_splits=1` 是新加的开关,关掉这个 trick 换回**单条序列结果与 batch 中其他序列无关**的不变性 —— 评测复现和单元测试的福音。
   - English: split-KV is a Flash-Attention performance trick — split the K/V sequence dim into `num_splits` pieces, run them in parallel, then combine via log-sum-exp. The catch: the split count depends on `max_k`, which depends on the **longest sequence in the batch**. Change the batch composition and the split changes; the floating-point reduction order changes; results drift in the last few bits. `num_splits=1` is the new escape hatch — disable the trick and the output for any single sequence is **independent of what else is in the batch**. A godsend for eval reproducibility and unit tests.

6. **底层 dispatch / The underlying dispatch**:
   - 中文: 最终调用的是 `torch.ops.torch_attn._varlen_attn`,一个 `@torch.library.custom_op` —— 这意味着它能被 `torch.compile` / Dynamo 看见,能写 fake tensor(`@_varlen_attn.register_fake`),能被 vmap、能被 autograd 集成。这套"用户 API + custom_op 内核"是 PyTorch 2.x 引入算子的标准范式。
   - English: it ultimately dispatches into `torch.ops.torch_attn._varlen_attn`, a `@torch.library.custom_op` — meaning it's visible to `torch.compile` / Dynamo, has a fake-tensor rule (`@_varlen_attn.register_fake`), composes with vmap, and integrates with autograd. The "thin Python wrapper + custom_op kernel" sandwich is the standard PyTorch 2.x recipe for shipping new ops.

## 类比 / The analogy

中文:想象一个**机场停机坪**。老式 `scaled_dot_product_attention` 是每个 batch 一架专机:有 N 个乘客就开 N 个座位的飞机,人不够拿假人填座位(padding)。`varlen_attn` 是**公共航站楼**:N 条不同长度的"航班"在同一条传送带上排队(packed),登机口名册告诉每架航班从哪个时刻起飞(`cu_seq`)。如果再把 KV 加上 `block_table`,等于把整座机场的座位拆成可拼装的**模块化舱段**(页),任何航班都可以从座位池里现拿现拼,空座位绝不浪费。`num_splits=1` 则像一条**禁止超车**的跑道规则:无论别的飞机几点起飞,你那架的落地精度不会因为同跑道有谁而变。

English: imagine an **airport apron**. The old `scaled_dot_product_attention` is a dedicated jet per batch — N passengers means an N-seat plane, and if N isn't a round number you stuff the empty seats with mannequins (padding). `varlen_attn` is a **central terminal**: N flights of different lengths queue on a single conveyor (packed), and the gate manifest (`cu_seq`) tells the kernel when each flight starts and ends. Add `block_table` to K/V and you've turned the whole airport's seats into **modular pods** (pages); any flight can assemble itself from the pool, never burning empty seats. `num_splits=1` is the **no-overtaking rule**: regardless of what other planes are landing on the same runway, your landing accuracy is independent of theirs.

## 自己跑一遍 / Try it yourself

```python
# Requires a recent PyTorch nightly with torch.nn.attention.varlen and a CUDA GPU.
import torch
from torch.nn.attention.varlen import varlen_attn

torch.manual_seed(0)
device, dtype = "cuda", torch.float16
H, D = 8, 64
lengths = torch.tensor([320, 192], device=device)
total = int(lengths.sum())

q = torch.randn(total, H, D, dtype=dtype, device=device)
k = torch.randn(total, H, D, dtype=dtype, device=device)
v = torch.randn(total, H, D, dtype=dtype, device=device)
cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
cu[1:] = lengths.cumsum(0)
max_len = int(lengths.max())

# 1) Full attention
out_full   = varlen_attn(q, k, v, cu, cu, max_len, max_len)
# 2) Causal
out_causal = varlen_attn(q, k, v, cu, cu, max_len, max_len, window_size=(-1, 0))
# 3) Causal sliding window of 128
out_swa    = varlen_attn(q, k, v, cu, cu, max_len, max_len, window_size=(128, 0))

# Bit-deterministic mode: same single-seq output regardless of batch
out_alone = varlen_attn(q[:320], k[:320], v[:320],
                        cu[:2], cu[:2], 320, 320, num_splits=1)
out_first = varlen_attn(q, k, v, cu, cu, max_len, max_len, num_splits=1)[:320]
print("max abs diff (alone vs in-batch, num_splits=1):",
      (out_alone - out_first).abs().max().item())
```

运行 / Run with:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
python try.py
```

预期输出 / Expected output:
```
max abs diff (alone vs in-batch, num_splits=1):  0.0
```

中文:`num_splits=1` 时,把第一条序列单独跑和放在 batch 里一起跑,**第一条序列的输出位级一致**。如果你把 `num_splits=1` 拿掉重跑一次,差异通常在 `1e-4` 量级 —— 不是 bug,是 split-KV 的本质。

English: with `num_splits=1`, running the first sequence alone vs. as part of a batch produces **bit-identical output** for that sequence. Drop `num_splits=1` and rerun — the difference typically sits around `1e-4`. That's not a bug; it's the nature of split-KV reduction order.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM `PagedAttention`** / **vLLM's PagedAttention**: 整套 paged KV cache 推理引擎,概念和 `block_table` 一模一样,只是它自己 ship 了 kernel。 / The entire paged-KV inference engine — `block_table` here is conceptually the same as vLLM's, just shipped in a different repo.
- **HuggingFace TGI 的 SWA** / **HuggingFace TGI's SWA**: Mistral 滑动窗口 attention 之前要单独 import flash-attn 的 varlen 接口,现在直接 `window_size=(W, 0)`。 / Mistral's sliding-window attention used to require a direct flash-attn import; now it's just `window_size=(W, 0)`.
- **`flash_attn.flash_attn_varlen_func`** / **`flash_attn.flash_attn_varlen_func`**: 上游 flash-attention 库的 packed API,`varlen_attn` 基本是它的 PyTorch-native 包装,但加了 cuDNN fallback 和确定性开关。 / The upstream flash-attention library's packed API — `varlen_attn` is basically a PyTorch-native wrapper around it, plus a cuDNN fallback and the determinism switch.

## 注意事项 / Caveats / when it breaks

- **CUDA-only / Only runs on CUDA**: Flash-Attention 路径只跑 NVIDIA GPU,CPU 上 `torch.ops.torch_attn._varlen_attn` 会报错。 / The Flash-Attention path is CUDA-only. On CPU `torch.ops.torch_attn._varlen_attn` raises.
- **`block_table` + GQA 暂时不支持 / `block_table` + GQA not supported yet**: `_varlen_attn` 内部的 cuDNN 分支不接受 GQA,Flash 分支接受但 paged + GQA 的组合在某些 PyTorch 版本里有 bug。 / cuDNN backend rejects GQA outright; the Flash backend supports both individually but paged + GQA together has been buggy in some PyTorch versions — pin a tested combo.
- **Dropout 被硬写成 0 / Dropout hard-coded to 0**: `_varlen_attn` 里 `dropout_p` 是 `0.0`,目前 API 没暴露训练 dropout。如果你需要 attention dropout,只能用旧的 SDPA。 / `dropout_p` inside `_varlen_attn` is hard-coded to `0.0`. If you need attention dropout during training, fall back to the old SDPA path for now.
- **`num_splits=1` 速度变慢 / `num_splits=1` is slower**: 它强制关掉 split-KV 的并行,batch 里只有少量很长的查询时 GPU 利用率会跌。只在评测、单元测试或需要 batch 不变性时用。 / It disables the split-KV parallelism, so GPU utilization drops when the batch has only a few long queries. Use only for eval, unit tests, or batch-invariance contracts.

## 延伸阅读 / Further reading

- [`torch/nn/attention/varlen.py` full source](https://github.com/pytorch/pytorch/blob/9ab94917c245d16efe77f546d30d73800c8d728d/torch/nn/attention/varlen.py)
- [Flash-Attention paper](https://arxiv.org/abs/2205.14135) and [Flash-Attention-2](https://arxiv.org/abs/2307.08691)
- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [PyTorch `custom_op` tutorial](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
