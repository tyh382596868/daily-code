---
date: 2026-06-05
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/bias.py
permalink: https://github.com/pytorch/pytorch/blob/61cead47b80012071fec74bf832da489d1422a6f/torch/nn/attention/bias.py#L144-L232
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, attention, kv-cache, causal-mask]
---

# `is_causal=True` 在 KV-cache 解码里是错的:upper-left 和 lower-right 的差别 / `is_causal=True` is wrong for KV-cache decoding: the upper-left vs. lower-right story

> **一句话 / In one line**: 当 `seq_q != seq_k`(典型场景是 KV-cache 解码,query 只有 1 个 token、key/value 有 N 个),`tril` 默认的 upper-left causal mask 会让新 token 看到未来,真正想要的是 lower-right。 / When `seq_q != seq_k` (the canonical case is KV-cache decoding where query has 1 token but key/value has N), the default `tril` upper-left causal mask lets the new token attend to *future* keys; what you actually want is the lower-right variant.

## 为什么重要 / Why this matters

几乎所有人写自回归 attention 都直接 `is_causal=True` 一了百了,但这条捷径里藏着一个静默 bug:**它默认 query 和 key/value 长度相同**,也就是只在训练那种 "看 N 个,预测 N 个" 的场景下正确。一旦你做 KV-cache 推理 — query 是新进来的 1 个 token,key/value 是历史的 N 个 — 这条捷径就会算出错误的 mask。PyTorch 把这两种语义拆成了 `CausalVariant.UPPER_LEFT` 和 `LOWER_RIGHT`,并用一个绝妙的 dispatch 让 90% 的常见场景仍然走最快的 flash kernel。

Almost everyone writing autoregressive attention reaches for `is_causal=True` because it's free, but the shortcut hides a silent footgun: **it assumes query and key/value have the same length**, i.e. the training-time "see N, predict N" setting. The moment you go to KV-cache decoding — query is the single new token, key/value is the N-token history — that shortcut produces the wrong mask. PyTorch splits this into `CausalVariant.UPPER_LEFT` and `LOWER_RIGHT` and uses an elegant dispatch so 90% of the common cases still hit the fastest flash kernel.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/bias.py`](https://github.com/pytorch/pytorch/blob/61cead47b80012071fec74bf832da489d1422a6f/torch/nn/attention/bias.py#L144-L232)

```python
def _upper_left(self, device: torch.device) -> torch.Tensor:
    """Upper left causal bias"""
    return torch.tril(
        torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool)
    )

def _lower_right(self, device: torch.device) -> torch.Tensor:
    """Lower right causal bias"""
    diagonal_offset = self.seq_len_kv - self.seq_len_q
    return torch.tril(
        torch.ones(
            self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool
        ),
        diagonal=diagonal_offset,
    )

def _materialize(self, device=None):
    if device is None:
        device = torch.device("cpu")
    if self.variant == CausalVariant.UPPER_LEFT:
        return self._upper_left(device)
    elif self.variant == CausalVariant.LOWER_RIGHT:
        return self._lower_right(device)

@staticmethod
def _dispatch(query, key, value, attn_mask, dropout_p=0.0, is_causal=False,
              scale=None, enable_gqa=False):
    if is_causal:
        raise ValueError("CausalBias should not be used with causal=True")

    if (
        attn_mask.seq_len_q == attn_mask.seq_len_kv
        or attn_mask.variant == CausalVariant.UPPER_LEFT
    ):
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,  # fast path
            scale=scale,
            enable_gqa=enable_gqa,
        )
    elif attn_mask.variant == CausalVariant.LOWER_RIGHT:
        # ... dispatch to flash with is_causal=True flag *meaning* lower-right
        # ... or fall back to materializing the mask
        ...
```

## 逐行讲解 / What's happening

1. **`_upper_left`: `tril` on a `(seq_q, seq_kv)` matrix**:
   - 中文: `tril` 默认 `diagonal=0`,所以左上角是 1×1 的真,右上角全是假。`shape=(3,4)` 时长这样:
     ```
     [[1,0,0,0],
      [1,1,0,0],
      [1,1,1,0]]
     ```
     第 0 个 query 只能看到第 0 个 key — 这是从"开头对齐"的因果性。
   - English: `tril` defaults to `diagonal=0`, so a `shape=(3,4)` mask becomes
     ```
     [[1,0,0,0],
      [1,1,0,0],
      [1,1,1,0]]
     ```
     Query 0 sees only key 0. Causality is anchored at the **start** of both sequences.

2. **`_lower_right`: `diagonal=seq_kv - seq_q` 的关键偏移**:
   - 中文: 同样是 `shape=(3,4)`,但 `diagonal=1`,变成
     ```
     [[1,1,0,0],
      [1,1,1,0],
      [1,1,1,1]]
     ```
     最后一个 query 能看到所有 key — 这才是 "我现在生成 token N+3,我应该看到历史全部 N+2 个 token" 的正确语义。
   - English: same `shape=(3,4)` but `diagonal=1`:
     ```
     [[1,1,0,0],
      [1,1,1,0],
      [1,1,1,1]]
     ```
     The last query sees every key — which is the correct semantics for "I am generating token N+3, so I should see all N+2 history tokens".

3. **`_dispatch` 第一条分支(快路径)**:
   - 中文: `seq_q == seq_kv` 时两种 variant 完全相同 — 都是标准下三角。直接调底层 `is_causal=True`,走 flash kernel。这是绝大多数训练场景。同样,即使 `seq_q != seq_kv` 但用户明确选了 UPPER_LEFT,flash kernel 的 `is_causal=True` 实际上实现的就是 upper-left,可以直接转发,什么 mask 都不用 materialize。
   - English: when `seq_q == seq_kv` both variants collapse to the standard lower-triangle, so just forward to the kernel-level `is_causal=True` and stay on flash. This is the vast majority of training. Notably even if `seq_q != seq_kv`, when the user explicitly asked for UPPER_LEFT, flash's `is_causal=True` *is* upper-left semantics — so you can still forward unchanged, never materializing a mask.

4. **LOWER_RIGHT 分支(慢路径,但语义正确)**:
   - 中文: 当用户要 LOWER_RIGHT 且形状不等时,事情复杂了。代码里(本文下方截断的部分)会尝试三种后端依次回退:flash kernel 实现的 lower-right、memory-efficient kernel 通过 `custom_mask_type=2` 实现的 lower-right、最后才是 materialize 一个 4D bool mask 走通用 SDPA。这条路径告诉我们:**lower-right 是真正的 KV-cache 默认值**,但实现上比 upper-left 贵一点。
   - English: when the user wants LOWER_RIGHT *and* shapes differ, things get interesting. The full function (truncated above) walks three backends: flash with a lower-right flag, memory-efficient attention via `custom_mask_type=2`, and finally materializing a 4D bool mask through generic SDPA. The lesson: **lower-right is the correct KV-cache default**, but it costs a tiny bit more to dispatch than upper-left.

5. **`raise ValueError("CausalBias should not be used with causal=True")`**:
   - 中文: 你不能同时传 `attn_mask=CausalBias(...)` 和 `is_causal=True`,因为这两个语义会冲突。设计层面强制 fail-fast,而不是悄悄 OR 起来。
   - English: you cannot pass both `attn_mask=CausalBias(...)` and `is_causal=True` because the two ways of asking for causal would conflict. The design fails fast instead of silently OR-ing them.

## 类比 / The analogy

想象你在排队进电影院。**Upper-left 因果**就是"第 N 个进场的人只能看到前 N-1 个进场的人"——开头对齐,座位 N 看不到座位 N+1 之后。**Lower-right 因果**则是"第 N 个进场的人能看到所有比他早进场的人,包括今天上午就坐下的那批 VIP"——结尾对齐。训练时所有人同时入场,两种 variant 一样;但当你 KV-cache 解码,意味着"VIP 已经坐了一上午(cached past),新进来的客人(单个 query token)可以看到所有 VIP"——必须是 lower-right。

Imagine a movie line. **Upper-left causal** says "the N-th person in line can only see the previous N-1 people in line" — anchored at the start. **Lower-right causal** says "the N-th person can see everyone who arrived before them, *including* the VIPs who got seated this morning" — anchored at the end. During training the whole line arrives together, so the two collapse. But during KV-cache decoding the VIPs (cached past) sat down hours ago and the new arrival (single query token) must see all of them — that's lower-right, not upper-left.

## 自己跑一遍 / Try it yourself

```python
# causal_variant_demo.py — pip install torch
import torch
from torch.nn.attention.bias import causal_upper_left, causal_lower_right

# KV-cache scenario: 1 new query, 5 cached past keys
B, H, D = 1, 2, 8
q = torch.randn(B, H, 1, D)
k = torch.randn(B, H, 5, D)
v = torch.randn(B, H, 5, D)

upper = causal_upper_left(1, 5)
lower = causal_lower_right(1, 5)

print("Upper-left  mask:", upper._materialize().squeeze().tolist())
print("Lower-right mask:", lower._materialize().squeeze().tolist())

out_upper = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=upper)
out_lower = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=lower)

print("\nUpper-left output mean:", out_upper.mean().item())
print("Lower-right output mean:", out_lower.mean().item())
print("Outputs differ:", not torch.allclose(out_upper, out_lower))
```

运行 / Run with:
```bash
python causal_variant_demo.py
```

预期输出 / Expected output:
```
Upper-left  mask: [True, False, False, False, False]
Lower-right mask: [True, True, True, True, True]

Upper-left output mean: <some number close to v[:,:,0,:].mean()>
Lower-right output mean: <some other number using all 5 keys>
Outputs differ: True
```

中文:upper-left 的 mask 只让单个 query 看到第 0 个 key — 等价于忽略了所有缓存的历史!lower-right 才让 query 看到全部 5 个。如果你写 KV-cache 时偷懒用了 `is_causal=True`,你的模型其实就在 silently 忽略所有 cached past。

English: the upper-left mask lets the single query see only key 0 — equivalent to ignoring all cached history! Only lower-right lets the query attend to all 5. If you cut corners with `is_causal=True` in your KV-cache code, your model is silently discarding the cache.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM PagedAttention** / **vLLM PagedAttention**: 自己实现了 lower-right (or equivalent) 因果约束,因为 query token 数和 cached KV token 数几乎总是不等。 / Implements lower-right (or equivalent) causality natively because query token count and cached KV token count almost never match.
- **HuggingFace `generate()` 的 mask** / **HuggingFace `generate()`'s mask**: 在 transformers 里,生成时的 `attention_mask` 实际编码的是 lower-right 因果性,只是名字叫 "right-padded causal"。 / The `attention_mask` HF uses during generation is effectively lower-right causality, just named "right-padded causal".
- **Flash-Attention v2/v3 `causal` 参数** / **Flash-Attention v2/v3 `causal` flag**: 底层 flash 也有 upper-left / lower-right 两种,接口里区分得不够明显,踩坑很常见。 / Underlying flash kernels also have upper-left vs. lower-right modes; the API doesn't always make this obvious, so this footgun is common.

## 注意事项 / Caveats / when it breaks

- **训练永远是 upper-left = lower-right** / **Training: upper-left = lower-right**: `seq_q == seq_kv`,所以你不会注意到区别。问题只在推理时浮现。 / Because `seq_q == seq_kv`, the distinction never shows up. The bug only surfaces at inference.
- **chunked prefill 混合模式** / **chunked prefill mixed mode**: 当 prefill 把长序列拆成多个 chunk,某些 chunk 的 query 已经包含了上一 chunk 的 KV 缓存 — 此时既不是纯 upper-left 也不是纯 lower-right,需要每个 chunk 单独 mask。 / When you chunk prefill into multiple passes, certain chunks have queries that include past chunks' KVs — neither pure variant works; each chunk needs its own mask.
- **Dispatch fallback 会 materialize 整个 mask** / **The materialize fallback is O(seq_q · seq_kv) memory**: lower-right + 不支持 flash 的设备 → 会构造一个完整 `(seq_q, seq_kv)` bool 张量,长序列时 VRAM 涨得很快。 / Lower-right on a device that can't use flash → constructs a full `(seq_q, seq_kv)` bool tensor; long sequences will blow up VRAM.

## 延伸阅读 / Further reading

- [PyTorch SDPA docs — `is_causal` and CausalBias](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [FlashAttention v2 paper — upper-left vs. lower-right discussion](https://arxiv.org/abs/2307.08691)
- [vLLM blog: PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
