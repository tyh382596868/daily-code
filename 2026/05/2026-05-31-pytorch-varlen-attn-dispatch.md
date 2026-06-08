---
date: 2026-05-31
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/varlen.py
permalink: https://github.com/pytorch/pytorch/blob/f7811aa3c052ace6751fbc2f6bc93908b9ea6b9f/torch/nn/attention/varlen.py#L45-L132
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, pytorch, attention, varlen, flash-attention, cudnn, custom-op]
---

# PyTorch 把"变长序列 attention"做成一个公开算子 / PyTorch turned variable-length attention into a public op

> **一句话 / In one line**: `torch.nn.attention.varlen` 用一个 `custom_op` 把"打包变长序列 + 在 cuDNN 或 FlashAttention 后端之间分发"包成 PyTorch 原生 op,让 torch.compile 和 autograd 都能识别。 / `torch.nn.attention.varlen` wraps "packed variable-length attention dispatching between cuDNN and FlashAttention" as a `custom_op`, so torch.compile and autograd see it as native PyTorch.

## 为什么重要 / Why this matters

之前训 packed-sequence LLM 想用 varlen FlashAttention,基本只有两条路:(1) 从 `flash_attn` 包直接调 `flash_attn_varlen_func`,但它是个原生 CUDA call,torch.compile 看不进去;(2) 自己塞 padding,然后忍受巨大的浪费。这个新加的 `torch.nn.attention.varlen` 是 PyTorch 把 varlen attention 正式提升为一等公民 —— `varlen_attn(q, k, v, cu_seq_q, cu_seq_k, max_q, max_k)` 是公开 API,内部 `_varlen_attn` 是一个 `@torch.library.custom_op`,运行时根据 query 的 device 自动选 cuDNN 或 FlashAttention 后端。看懂这 90 行,你就懂了 PyTorch 是怎么"接管"一个 vendor kernel 同时还保住 compile 和 autograd 的。

Before this, packed-sequence LLM training had two options: (1) call `flash_attn_varlen_func` directly — works but torch.compile can't trace it; (2) pad to max length and waste compute. The new `torch.nn.attention.varlen` makes packed varlen attention a first-class citizen: the public `varlen_attn(q, k, v, cu_seq_q, cu_seq_k, max_q, max_k)` API delegates to a `@torch.library.custom_op` that picks cuDNN or FlashAttention at runtime. Understanding these 90 lines teaches you how PyTorch absorbs a vendor kernel while keeping compile and autograd intact.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/varlen.py`](https://github.com/pytorch/pytorch/blob/f7811aa3c052ace6751fbc2f6bc93908b9ea6b9f/torch/nn/attention/varlen.py#L45-L132)

```python
@torch.library.custom_op("torch_attn::_varlen_attn", mutates_args={})
def _varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Private custom op for variable-length attention.
    """
    window_size = _normalize_window_size(window_size)

    use_cudnn = query.is_cuda and _should_use_cudnn(query.device.index)

    if use_cudnn:
        log.info("Using cuDNN backend for varlen_attn")
        if enable_gqa:
            raise RuntimeError("GQA is not supported with the cuDNN backend.")
        if num_splits is not None:
            raise RuntimeError("num_splits is not supported with the cuDNN backend.")
        if window_size[0] != -1 or window_size[1] != -1:
            raise RuntimeError("cuDNN backend does not support window attention.")
        if seqused_k is not None or block_table is not None:
            raise RuntimeError("seqused_k/block_table is not yet supported with cuDNN.")

        result = torch.ops.aten._cudnn_attention_forward(
            query, key, value,
            None,                # attn_bias
            cu_seq_q, cu_seq_k, max_q, max_k,
            True,                # compute_log_sumexp
            0.0,                 # dropout_p
            is_causal,
            False,               # return_debug_mask
            scale=scale,
        )
        # cuDNN returns: (output, logsumexp, cum_q, cum_k, max_q, max_k,
        #                 philox_seed, philox_offset, debug_attn_mask)
        output, softmax_lse, rng_state = result[0], result[1], result[6]
    else:
        log.info("Using Flash Attention backend for varlen_attn")
        output, softmax_lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
            query, key, value,
            cu_seq_q, cu_seq_k, max_q, max_k,
            0.0,                 # dropout_p
            is_causal,
            return_debug_mask=False,
            scale=scale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            seqused_k=seqused_k,
            block_table=block_table,
            num_splits=num_splits,
        )

    rng_state_ = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    return output, softmax_lse, rng_state_
```

## 逐行讲解 / What's happening

1. **`@torch.library.custom_op("torch_attn::_varlen_attn", mutates_args={})` (line 45)**:
   - 中文: 这一行就是把这个 Python 函数注册成 PyTorch 的原生算子,命名空间 `torch_attn`、op 名 `_varlen_attn`。`mutates_args={}` 告诉 dispatcher 这是个纯函数 —— 没有任何参数会被原地改 —— autograd 引擎、torch.compile、torch.export 都能把它当作 black box 节点缓存、追踪、融合。
   - English: this decorator registers the Python function as a PyTorch op under namespace `torch_attn` with name `_varlen_attn`. `mutates_args={}` declares it pure, so autograd, torch.compile, and torch.export can treat it as a black-box node — safe to cache, trace, and fuse.

2. **打包变长输入的约定 / The packed-varlen calling convention (lines 47-53)**:
   - 中文: `query` 形状是 `(T_q, H, D)`,而不是常见的 `(B, H, T, D)` —— **没有 batch 维**。所有 batch 里的所有 token 都被拼成一根长向量。`cu_seq_q` 是长度 `(B+1,)` 的 int32,`cu_seq_q[i]:cu_seq_q[i+1]` 就是第 `i` 条序列在 `query` 里的范围。`max_q` 是 batch 里最长那条的长度 —— kernel 用它确定 block 数量。
   - English: `query` has shape `(T_q, H, D)` — **no batch dimension**. All tokens from all sequences are concatenated end-to-end. `cu_seq_q` is an `(B+1,)` int32 tensor of cumulative offsets; sequence `i` lives in `query[cu_seq_q[i]:cu_seq_q[i+1]]`. `max_q` tells the kernel the longest sequence for block sizing.

3. **设备选择 / Backend selection (line 69)**:
   - 中文: `use_cudnn = query.is_cuda and _should_use_cudnn(...)`,后者用 `@lru_cache(maxsize=8)` 缓存了"这张显卡能不能用 cuDNN"的判断,免得每次都查询 CUDA capability。在当前版本它硬编码 `return False`,意味着默认走 FlashAttention —— cuDNN 路径是为后续启用准备的占位。
   - English: `use_cudnn = query.is_cuda and _should_use_cudnn(...)`. The helper is `lru_cache`-d so the CUDA capability query happens once per device. In this version it hard-codes `return False`, so today the default is always FlashAttention — the cuDNN path is plumbed but not yet enabled.

4. **cuDNN 兼容性闸门 / cuDNN feature gating (lines 71-89)**:
   - 中文: 走 cuDNN 之前先拒绝它不支持的功能 —— GQA、num_splits、sliding window、paged KV cache。这是一个非常诚实的"特性矩阵" —— 一目了然两个后端的能力差异,可以照抄到你自己的多后端 dispatcher 里。
   - English: before calling cuDNN, explicitly reject features it doesn't support — GQA, num_splits, sliding window, paged KV cache. This is an honest "feature matrix" you can copy as a pattern for your own multi-backend dispatchers.

5. **`torch.ops.aten._cudnn_attention_forward` vs `_flash_attention_forward` (lines 91 & 110)**:
   - 中文: 注意两个后端的返回 tuple 形状不一样!cuDNN 返回 9 元 tuple `(output, lse, cum_q, cum_k, max_q, max_k, seed, offset, debug_mask)`,FlashAttention 返回 5 元 tuple `(output, lse, rng_state, _, _)`。这一层 dispatch 的核心工作就是把两种异构输出 *归一化*成统一的 `(output, lse, rng_state)`。
   - English: the two backends return *different* tuples — cuDNN gives 9 elements, FlashAttention gives 5. The dispatcher's main job is normalizing both to a unified `(output, lse, rng_state)` triple. This is the reason custom_op exists: hide vendor heterogeneity behind a clean signature.

6. **`rng_state_` 是硬编码的 (line 129)**:
   - 中文: 因为 `dropout_p=0.0` 硬编码,所以"随机数状态"在这一版本里恒为 zeros。等以后支持 dropout 时,这一格就要承接 kernel 内部的 philox 种子,用于 backward 时复现 dropout mask。
   - English: dropout is hard-coded to `0.0`, so the returned RNG state is just zeros for now. Once dropout lands, this slot will carry the kernel's philox seed/offset so the backward pass can replay the exact dropout mask.

## 类比 / The analogy

想象你是一个机场的统一安检通道。乘客来自不同航空公司(cuDNN/FlashAttention/未来的 cuTLASS),每家有自己的护照格式、自己的拒收名单(cuDNN 不收 GQA、FlashAttention 不收 PagedKV 的某些组合)。你的工作不是自己安检,而是 (1) 看护照决定送去哪个柜台,(2) 在送过去之前先核对那家柜台的拒收规则,(3) 把每家不同格式的回单整理成你的航空公司无关的统一登机牌。`_varlen_attn` 就是这个安检通道,只不过乘客是 attention 张量,登机牌是 `(output, lse, rng_state)`。

Picture an airport security desk that funnels passengers from many airlines (cuDNN, FlashAttention, future cuTLASS) through a single gate. Each airline has its own passport format and its own no-fly list (cuDNN rejects GQA; FlashAttention rejects certain paged-KV combos). Your job is not to do security yourself — it is to (1) decide which counter to send each passenger to, (2) check the counter's rejection rules first, and (3) normalize the counter's wildly different receipts into one standard boarding pass. `_varlen_attn` is that desk; the passengers are attention tensors and the boarding pass is `(output, lse, rng_state)`.

## 自己跑一遍 / Try it yourself

```python
# pip install torch>=2.6  (and a CUDA GPU with flash attention)
import torch
from torch.nn.attention.varlen import varlen_attn

# Three sequences of lengths 3, 5, 2 (total 10 tokens), H=2 heads, D=8
device = "cuda"
seq_lens = torch.tensor([3, 5, 2], device=device, dtype=torch.int32)
cu = torch.zeros(seq_lens.numel() + 1, dtype=torch.int32, device=device)
cu[1:] = seq_lens.cumsum(0)        # [0, 3, 8, 10]
T, H, D = int(seq_lens.sum()), 2, 8

q = torch.randn(T, H, D, device=device, dtype=torch.float16)
k = torch.randn(T, H, D, device=device, dtype=torch.float16)
v = torch.randn(T, H, D, device=device, dtype=torch.float16)

out = varlen_attn(q, k, v, cu, cu, max_q=int(seq_lens.max()), max_k=int(seq_lens.max()))
print("out shape:", out.shape, "dtype:", out.dtype)  # (10, 2, 8)
print("first sequence (3 tokens) head 0:", out[0:3, 0].norm().item())
print("second sequence (5 tokens) head 0:", out[3:8, 0].norm().item())
```

运行 / Run with:
```bash
pip install torch>=2.6
python try.py
```

预期输出 / Expected output:
```
out shape: torch.Size([10, 2, 8]) dtype: torch.float16
first sequence (3 tokens) head 0: <some float>
second sequence (5 tokens) head 0: <some float>
```

中文: 注意 `out.shape[0] == sum(seq_lens)` —— 没有 batch 维。每条序列只在自己 `cu[i]:cu[i+1]` 的范围里做 attention,跨序列绝对不会泄漏 token,这就是 cu_seq 的全部意义。

English: note that `out.shape[0] == sum(seq_lens)` — no batch dimension. Each sequence attends only within its own `cu[i]:cu[i+1]` slice; no cross-sequence leakage. That's the entire point of cumulative-sequence offsets.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`Dao-AILab/flash-attention` `flash_attn_varlen_func`** / **`flash_attn_varlen_func`**: 中文: 原始作者的 Python 接口,签名几乎一样,只不过没有 PyTorch custom_op 的包裹。 / English: the original Dao-AILab Python entry point; nearly identical signature without the PyTorch custom_op wrapper.
- **HuggingFace `transformers` packing trainer** / **HF Transformers packed trainer**: 中文: 训练时把多条短样本打包成一个 batch 用的就是 varlen attention —— 之前要在外面手工调 flash_attn,现在可以走 `torch.nn.attention.varlen` 让 torch.compile 看得见。 / English: training short samples concatenated into one batch needs varlen attention; you used to bypass torch.compile, now you don't have to.
- **`@torch.library.custom_op` 模式本身** / **The `@torch.library.custom_op` pattern**: 中文: 5-28 教过 flash-attention 把 NCCL collective 包成 autograd Function 的同款思路 —— 把外部 kernel 注册成原生 op,框架就能围绕它做编译、求导、追踪。 / English: same mindset as the flash-attention NCCL-as-autograd-function note from 2026-05-28 — register vendor kernels as native ops so the framework can compile, differentiate, and trace around them.

## 注意事项 / Caveats / when it breaks

- **`window_size` 的语义** / **`window_size` semantics**: 中文: `(-1, -1)` 是全注意力,`(-1, 0)` 是 causal,`(W, 0)` 是因果 + sliding window 大小 W。把 0 当 W 传就完全屏蔽 attention,容易踩坑。 / English: `(-1, -1)` = full, `(-1, 0)` = causal, `(W, 0)` = causal + sliding window of size W. Passing `0` for W silences attention entirely — easy to misfire.
- **cuDNN 路径目前是 noop** / **The cuDNN path is currently a noop**: 中文: `_should_use_cudnn` 硬编码返回 False。如果你想试 cuDNN,需要自己 monkey-patch 那个函数。 / English: `_should_use_cudnn` hard-codes `False`. To exercise the cuDNN branch you'd need to monkey-patch the helper.
- **num_splits 影响数值精度** / **`num_splits` is a numerics knob**: 中文: split-KV 把 K/V 维度分块并行,batch 组成不同时归约顺序不同,bit-精确性不再保证。设置 `num_splits=1` 关掉它换 bit-级可复现。 / English: split-KV parallelizes K/V across blocks; different batch compositions reorder reductions and break bitwise reproducibility. Set `num_splits=1` for bit-identical results.

## 延伸阅读 / Further reading

- PyTorch RFC for varlen attention: <https://github.com/pytorch/pytorch/pull/138567>
- FlashAttention 2 paper: <https://arxiv.org/abs/2307.08691>
- `torch.library.custom_op` tutorial: <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>
