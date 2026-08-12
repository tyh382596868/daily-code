---
date: 2026-08-12
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
permalink: https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py#L446-L487
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, kv-cache]
build_role: sampler-inference advanced variant, preallocating per-layer self-attention and cross-attention caches
---

# DreamZero KV cache：先把每层的记忆槽形状定死 / DreamZero KV Cache: Fix the Per-Layer Memory Shapes First

> **一句话 / In one line**: DreamZero 为 Wan action head 预先创建每层 self-attention 与 cross-attention 的 KV cache，让闭环推理可以复用旧块。 / DreamZero pre-creates per-layer self-attention and cross-attention KV caches for its Wan action head so closed-loop inference can reuse old blocks.

## 为什么重要 / Why this matters

WAM 闭环推理不是每次都重新看完整视频历史。更实际的做法是保留前面 block 的 key/value，新一轮只处理当前 block 和动作 register。这里的关键不是算法公式，而是 cache shape：层数、batch、head 数、head dim、context 长度必须和 DiT 完全一致。

Closed-loop WAM inference should not reprocess the full video history every call. A practical implementation keeps key/value tensors for previous blocks and only processes the current block plus action registers. The important detail is the cache shape: layers, batch, heads, head dim, and context length must match the DiT exactly.

## 代码 / The code

`dreamzero0/dreamzero` — [`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`](https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py#L446-L487)

```python
def _create_kv_caches(
    self,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    frame_seqlen: int,
) -> tuple[KVCacheType, KVCacheType]:
    """
    Initialize a Per-GPU KV cache for the Wan model.
    Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
    """
    num_heads = self.model.num_heads
    head_dim = self.model.dim // num_heads
    kv_cache1: KVCacheType = []
    kv_cache_neg: KVCacheType = []
    for _ in range(self.model.num_layers):
        kv_cache1.append(
            torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
        )
        kv_cache_neg.append(
            torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
        )
    return kv_cache1, kv_cache_neg

def _create_crossattn_caches(
    self, batch_size: int, dtype: torch.dtype, device: torch.device,
) -> tuple[KVCacheType, KVCacheType]:
    """
    Initialize a Per-GPU cross-attention cache for the Wan model.
    Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
    """
    num_heads = self.model.num_heads
    head_dim = self.model.dim // num_heads
    crossattn_cache: KVCacheType = []
    crossattn_cache_neg: KVCacheType = []
    for _ in range(self.model.num_layers):
        crossattn_cache.append(
            torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
        )
        crossattn_cache_neg.append(
            torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
        )
    return crossattn_cache, crossattn_cache_neg
```

## 逐行讲解 / What's happening

1. **模型反查形状 / Derive shape from the model**:
   - 中文: `num_heads` 和 `head_dim` 从当前 DiT 读取，而不是写死；这样 5B 和 14B backbone 可以共用一套逻辑。
   - English: `num_heads` and `head_dim` are read from the current DiT instead of hard-coded, so 5B and 14B backbones share the same logic.
2. **self-attention cache / self-attention cache**:
   - 中文: `[2, B, 0, H, D]` 的第 0 维通常对应 K/V；长度先是 0，后面随着视频 block 追加。
   - English: `[2, B, 0, H, D]` uses the first dimension for K/V; sequence length starts at 0 and grows as video blocks are appended.
3. **negative cache / negative cache**:
   - 中文: `kv_cache_neg` 给 CFG 的负条件分支单独留一份，避免正负 prompt 共享可变状态。
   - English: `kv_cache_neg` keeps a separate negative-branch cache for CFG so positive and negative prompts do not share mutable state.
4. **cross-attention cache / cross-attention cache**:
   - 中文: cross-attn 的 text/context 长度固定成 512，所以一开始就分配 `[2, B, 512, H, D]`。
   - English: Cross-attention text/context length is fixed at 512, so it preallocates `[2, B, 512, H, D]`.

## 类比 / The analogy

像剧场后台的储物格：每一层 transformer 都有自己的架子，正条件和负条件各一排；self-attn 架子先空着，cross-attn 架子先放好 512 个标签位。

It is like backstage shelves in a theater: every transformer layer gets its own shelf, with one row for positive conditioning and one for negative. The self-attention shelf starts empty; the cross-attention shelf starts with 512 labeled slots.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `sampler-inference` 的高级变体。你的 nanoWAM 里可以先实现 `init_cache(num_layers, batch, heads, head_dim, text_len)`，再在每个 denoise step 把当前 video/action block 写进去。上游是 text/image encoder 和 DiT 配置；下游是带 cache 的 attention forward。没有这层，闭环推理会重复处理历史视频，延迟和显存都会失控。

This is an advanced `sampler-inference` variant. In a nanoWAM, start with `init_cache(num_layers, batch, heads, head_dim, text_len)`, then append the current video/action block during each denoising step. Upstream are text/image encoders and DiT config; downstream is cache-aware attention forward. Without this layer, closed-loop inference repeatedly reprocesses old video history and latency grows quickly.

## 自己跑一遍 / Try it yourself

```python
def make_cache(num_layers, batch, dim, heads, text_len):
    head_dim = dim // heads
    self_cache = [[[2, batch, 0, heads, head_dim] for _ in range(num_layers)]]
    text_cache = [[[2, batch, text_len, heads, head_dim] for _ in range(num_layers)]]
    return self_cache[0], text_cache[0]

self_cache, text_cache = make_cache(num_layers=3, batch=2, dim=3072, heads=24, text_len=512)
print(self_cache[0])
print(text_cache[0])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[2, 2, 0, 24, 128]
[2, 2, 512, 24, 128]
```

这里 `128` 是由 `3072 // 24` 推出来的；换 backbone 时 cache 自动变形。

Here `128` comes from `3072 // 24`; the cache adapts when the backbone changes.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM video KV cache** / **FastWAM video KV cache**: 同样把视频历史留在 cache 里，动作逐步解码。
- **LeRobot pi0 cache clone** / **LeRobot pi0 cache clone**: prefix cache 可以复用，但每次推理要隔离可变状态。

## 注意事项 / Caveats / when it breaks

- **长度策略要清楚** / **Length policy must be clear**: self cache 从 0 增长，cross cache 固定 512；两者不能混用。
- **CFG 分支不能共享 cache** / **CFG branches cannot share cache**: 正负条件如果共用同一份 K/V，会把条件污染到对方。

## 延伸阅读 / Further reading

- [DreamZero WAN action head cache creation](https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py#L446-L487)
