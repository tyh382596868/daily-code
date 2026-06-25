---
date: 2026-06-25
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/language_model.py
permalink: https://github.com/huggingface/nanoVLM/blob/e8087c03772704d36ad73a79b801849b32249ecf/models/language_model.py#L206-L297
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, attention, gqa, kv-cache, prefill, decode, nanovlm]
---

# nanoVLM GQA：prefill/decode 统一路径 + `is_causal` 精确谓词 / nanoVLM GQA: Unified Prefill/Decode Path + Precise `is_causal` Predicate

> **一句话 / In one line**: 一个 `is_prefill = block_kv_cache is None` 判断让同一个 forward 函数同时处理 prefill（全序列因果注意力）和 decode（单 token 拼缓存注意力），且用 `is_causal = (T_curr == T_kv and T_curr > 1)` 精确推断是否需要 causal mask。 / A single `is_prefill = block_kv_cache is None` guard lets one forward function handle both prefill (full-sequence causal attention) and decode (single-token cache-append attention), with `is_causal = (T_curr == T_kv and T_curr > 1)` as a precise causal mask predicate.

## 为什么重要 / Why this matters

GQA（Grouped Query Attention）的实现难点不在于 head 数量的缩减，而在于 **prefill 和 decode 两个阶段用的是截然不同的计算模式**：prefill 处理完整的输入序列，需要因果遮罩；decode 每次只处理一个新 token，把 KV 拼到缓存后做全序列注意力，不需要因果遮罩（单 token 看所有历史 token 天然合法）。

nanoVLM 的实现用两个干净的设计把这两种模式统一进同一段代码：①用 `block_kv_cache is None` 区分 prefill 和 decode，②用 `is_causal = (T_curr == T_kv and T_curr > 1)` 推导出正确的 mask 行为——当前序列长度等于 KV 总长度且大于 1，说明是 prefill，需要 causal mask；否则（decode 时 T_curr=1 < T_kv，或 prefill 第一个 token）不需要。这两个判断合在一起，让整个 `scaled_dot_product_attention` 调用在任意场景下都是正确的，无需任何显式 if-else 分支。

The nanoVLM implementation unifies both modes in ~90 lines without explicit branching. `block_kv_cache is None` distinguishes prefill from decode. `k.repeat_interleave(n_kv_groups, dim=1)` expands grouped keys to full head count — simpler and equally efficient compared to broadcasting tricks. And the `is_causal` predicate avoids passing wrong mask parameters to SDPA, which would either silently produce wrong attention weights or trigger a slow non-flash path in PyTorch.

## 代码 / The code

`huggingface/nanoVLM` — [`models/language_model.py`](https://github.com/huggingface/nanoVLM/blob/e8087c03772704d36ad73a79b801849b32249ecf/models/language_model.py#L206-L297)

```python
def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
    is_prefill = block_kv_cache is None                              # L227

    B, T_curr, C = x.size()

    q_curr = self.q_proj(x).view(B, T_curr, self.n_heads,    self.head_dim).transpose(1, 2)
    k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)

    # Apply rotary positional embeddings to q and k
    q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

    # KV cache: append current K/V if decoding, or initialize on first prefill
    if not is_prefill and block_kv_cache['key'] is not None:         # L239
        k = torch.cat([block_kv_cache['key'],   k_rotated], dim=2)  # L244
        v = torch.cat([block_kv_cache['value'], v_curr],    dim=2)  # L245
        block_kv_cache['key']   = k
        block_kv_cache['value'] = v
    else:
        k = k_rotated
        v = v_curr
        block_kv_cache = {'key': k, 'value': v}

    # GQA expansion: repeat K and V from n_kv_heads to n_heads
    k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)            # L255
    v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)

    T_kv = k_exp.size(2)

    # Padding mask handling: convert [B, T_kv] boolean → additive float mask
    additive_attn_mask = None
    if attention_mask is not None:
        mask_for_keys = attention_mask[:, :T_kv]
        additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min

    if self.sdpa and x.device.type != 'mps':
        # is_causal=True only when doing prefill (T_curr == T_kv) on >1 tokens   L272
        is_causal = (T_curr == T_kv and T_curr > 1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=additive_attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
    else:
        # Manual fallback (MPS or no SDPA)
        attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
        if T_curr == T_kv and T_curr > 1:
            causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
            attn = attn.masked_fill(~causal_mask_val, float('-inf'))
        if additive_attn_mask is not None:
            attn = attn + additive_attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v_exp

    y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
    y = self.out_proj(y)
    y = self.resid_dropout(y)
    return y, block_kv_cache
```

## 逐行讲解 / What's happening

1. **L227 `is_prefill = block_kv_cache is None`：零开销的阶段检测 / Zero-cost phase detection**
   - 中文：整个 prefill/decode 判断只有一行。在 prefill 时调用方传 `block_kv_cache=None`，decode 时传上一步返回的缓存 dict。这让函数签名本身就成了 API 文档：None = 我不需要缓存 = prefill；非 None = 我有历史 KV = decode。
   - English: The entire prefill/decode dispatch is one line. Callers pass `None` during prefill and the previously returned cache dict during decode. This makes the signature self-documenting: `None` means "no cache / first pass / prefill"; a dict means "I have prior KV / decode". No separate `use_cache` flag needed.

2. **L244-245 KV cache 追加 / KV cache concatenation**
   - 中文：decode 时把本步新生成的 `k_rotated`（形状 `[B, n_kv_heads, 1, head_dim]`）在序列维度（dim=2）拼到历史缓存后面，`v_curr` 同理。拼后直接 in-place 更新 `block_kv_cache`，下次 decode 直接使用。注意这里用的是 `torch.cat`（产生新 tensor）而非 in-place 填充——因为缓存不一定预分配到最大长度。
   - English: During decode, the current step's rotated key (`[B, n_kv_heads, 1, head_dim]`) is concatenated along dim=2 (the sequence axis) to the historical cache. The dict is updated in-place for the next decode step. Note the use of `torch.cat` (allocating a new tensor each step) rather than a pre-allocated buffer — nanoVLM trades allocation overhead for simpler code, appropriate for a reference implementation.

3. **L255 `k.repeat_interleave(self.n_kv_groups, dim=1)`：GQA 头展开 / GQA head expansion**
   - 中文：GQA 的 KV head 数是 `n_kv_heads`，Q head 数是 `n_heads = n_kv_heads × n_kv_groups`。`repeat_interleave(n_kv_groups, dim=1)` 把每个 KV head 重复 `n_kv_groups` 次，产生 `n_heads` 个 head，与 Q 对齐后就能做标准的 multi-head attention。与 `expand` + `reshape` 的区别是：`repeat_interleave` 保证内存连续，不会触发任何隐式 copy，对 SDPA 的后端更友好。
   - English: GQA has `n_kv_heads` K/V heads and `n_heads = n_kv_heads × n_kv_groups` query heads. `repeat_interleave(n_kv_groups, dim=1)` repeats each KV head `n_kv_groups` times, expanding from `n_kv_heads` to `n_heads` so the shapes align for dot-product attention. Unlike `expand` (which creates a view), `repeat_interleave` produces a contiguous layout that avoids subtle performance pitfalls in the SDPA backend.

4. **L272 `is_causal = (T_curr == T_kv and T_curr > 1)`：精确 causal mask 谓词 / Precise causal mask predicate**
   - 中文：这是整个函数里最微妙的一行。分三种情况：(a) prefill（全序列首次过），`T_curr = T_kv > 1`，条件成立，SDPA 启用因果遮罩；(b) decode（每步一个 token），`T_curr=1 < T_kv`，条件为假，不需要遮罩（单 token 看所有历史 token 天然合法）；(c) 第一个 token 的 prefill 或序列长度为 1，`T_curr = T_kv = 1`，`T_curr > 1` 为假，不遮罩（1×1 注意力无需遮罩）。三种情况用一个条件全覆盖。
   - English: Three cases: (a) full prefill with multiple tokens: `T_curr == T_kv > 1` → `is_causal=True`, need the lower-triangular mask; (b) decode: `T_curr=1 < T_kv` → `is_causal=False`, the single query token can attend to all historical KV tokens without masking; (c) single-token prefill: `T_curr == T_kv == 1` → `is_causal=False`, a 1×1 attention matrix needs no masking. All three covered by one expression.

## 类比 / The analogy

想象你是图书馆管理员。prefill 是你第一次把一整本书从头到尾读一遍，做笔记——每一页只能参考比它更早的页（因果遮罩）。decode 是你之后每次只新到一页，翻开笔记本，把新页和所有历史笔记对照查询——新页可以参考所有历史页，不需要因果遮罩。`block_kv_cache` 就是那本笔记本，`is_causal` 谓词就是"我现在是读第一遍还是在追更"的判断。

Think of it like a librarian with a growing notebook. Prefill is reading an entire book from scratch: each page can only reference earlier pages (causal mask). Decode is receiving new pages one at a time and consulting the notebook: the new page can reference all past pages without restriction (no causal mask). `block_kv_cache` is the notebook. The `is_causal` predicate is the librarian's check: "am I reading a full new book (prefill), or receiving today's new page (decode)?"

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn.functional as F

def gqa_forward_demo(q, k, v, n_kv_groups, kv_cache=None):
    is_prefill = kv_cache is None
    if not is_prefill and kv_cache['k'] is not None:
        k = torch.cat([kv_cache['k'], k], dim=2)
        v = torch.cat([kv_cache['v'], v], dim=2)
    kv_cache = {'k': k, 'v': v}
    k_exp = k.repeat_interleave(n_kv_groups, dim=1)
    v_exp = v.repeat_interleave(n_kv_groups, dim=1)
    T_curr, T_kv = q.shape[2], k_exp.shape[2]
    is_causal = (T_curr == T_kv and T_curr > 1)
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
    return out, kv_cache

B, n_h, n_kv, d = 1, 4, 2, 8  # 4 Q heads, 2 KV heads (n_kv_groups=2)
# prefill: 6 tokens
q_p = torch.randn(B, n_h, 6, d); k_p = torch.randn(B, n_kv, 6, d); v_p = torch.randn(B, n_kv, 6, d)
out_p, cache = gqa_forward_demo(q_p, k_p, v_p, n_kv_groups=2)
print(f"prefill output: {out_p.shape}")  # [1, 4, 6, 8]

# decode: 1 new token
q_d = torch.randn(B, n_h, 1, d); k_d = torch.randn(B, n_kv, 1, d); v_d = torch.randn(B, n_kv, 1, d)
out_d, cache = gqa_forward_demo(q_d, k_d, v_d, n_kv_groups=2, kv_cache=cache)
print(f"decode output: {out_d.shape}, cache K shape: {cache['k'].shape}")  # [1,4,1,8], [1,2,7,8]
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
prefill output: torch.Size([1, 4, 6, 8])
decode output: torch.Size([1, 4, 1, 8]), cache K shape: torch.Size([1, 2, 7, 8])
```

decode 步骤后缓存的 K 从 6 tokens 增长到 7 tokens（6 prefill + 1 decode），这就是 KV cache 自增长的机制。

After the decode step the cached K grows from 6 to 7 tokens (6 from prefill + 1 from decode). That is the KV cache growth mechanism in one demo.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`transformers` Llama 实现（`modeling_llama.py`）** / **transformers LlamaAttention**: 用 `past_key_value` 对象封装了相同的拼接逻辑，`is_causal` 同样由序列长度推导。
- **`vllm` PagedAttention** / **vllm PagedAttention**: 用 page table 替代了 `torch.cat`（避免每步分配），但 prefill/decode 分离的思路完全相同。参见 [`2026-06-11-vllm-kv-cache-watermark.md`](../2026-06-11-vllm-kv-cache-watermark.md)。
- **DeepSeek-V3 MLA absorb 的缓存** / **DeepSeek-V3 MLA absorb cache**: 今日另一篇——用压缩潜变量替代完整 K/V，但 prefill/decode 分支逻辑的思路是一样的。

## 注意事项 / Caveats / when it breaks

- **`repeat_interleave` vs `expand`** / **`repeat_interleave` vs `expand`**: `expand` 创建一个广播 view（不分配新内存），但某些 SDPA 后端（特别是 FlashAttention 内核）要求输入是连续的（`is_contiguous()=True`）。`repeat_interleave` 产生连续内存，是更安全的选择，代价是一次内存分配。
- **`block_kv_cache` 是 mutable dict** / **`block_kv_cache` is a mutable dict**: L246-247 的 in-place 更新意味着调用方必须持有同一个 dict 引用。如果你在 batch 中的不同 beam 之间共享 cache，需要手动复制，否则会有 beam 污染问题。
- **MPS 设备走 fallback 路径** / **MPS takes the fallback path**: L270 的 `x.device.type != 'mps'` 排除了 Metal Performance Shaders 后端——Apple Silicon 的 MPS 不支持 `F.scaled_dot_product_attention` 的 `is_causal` 参数，手动实现的 fallback 在 M 系列芯片上反而是正确路径。

## 延伸阅读 / Further reading

- [nanoVLM 项目页面（Hugging Face）](https://huggingface.co/blog/nanovlm)
- [GQA 原始论文（Ainslie et al. 2023）](https://arxiv.org/abs/2305.13245)
- [Flash Attention 2 的 GQA 支持](https://arxiv.org/abs/2307.08691)
