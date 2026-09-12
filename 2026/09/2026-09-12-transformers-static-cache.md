---
date: 2026-09-12
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/cache_utils.py
permalink: https://github.com/huggingface/transformers/blob/df04b012229d50d2b6dfba32c61c3057c3a40ea1/src/transformers/cache_utils.py#L400-L503
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, transformers, kv-cache, torch-compile, static-memory]
---

# Transformers Static Cache：先分配，再原地写入 / Transformers Static Cache: Preallocate, Then Update In Place

> **一句话 / In one line**: `StaticLayer` 用固定地址的 KV buffer 换掉动态 `cat`，让 decode 更适合 `torch.compile` 和 CUDA graphs。 / `StaticLayer` trades dynamic concatenation for fixed-address KV buffers, making decode friendlier to `torch.compile` and CUDA graphs.

## 为什么重要 / Why this matters

中文：动态 KV cache 很直观：每次生成 token 就把新 key/value 拼到旧 tensor 后面。但动态 shape 和新 tensor 分配会让编译器反复看到不同的地址或长度。`StaticLayer` 第一次收到真实输入时分配最大 buffer，之后只用位置索引原地写入，并把地址标记为静态。

English: A dynamic KV cache is intuitive: concatenate each new key/value tensor onto the old one. But changing shapes and allocations are awkward for compilation. `StaticLayer` lazily allocates a maximum-size buffer from the first real input, then writes into fixed positions in place and marks the addresses as static.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/cache_utils.py`](https://github.com/huggingface/transformers/blob/df04b012229d50d2b6dfba32c61c3057c3a40ea1/src/transformers/cache_utils.py#L400-L503)

```python
class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int, **kwargs):
        super().__init__()
        self.max_cache_len = max_cache_len
        # Very important that it's a tensor here, to avoid recompiling when we update it and use it to create positions
        self.cumulative_length = torch.tensor(0, dtype=int)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        self.keys = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.cumulative_length = self.cumulative_length.to(self.device)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.cumulative_length)

        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kv_length = key_states.shape[-2]
        cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
        self.cumulative_length.add_(kv_length)

        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

        return self.keys, self.values
```

## 逐行讲解 / What's happening

1. **第 400-417 行 / Lines 400-417 (static metadata)**:
   - 中文：`is_compileable=True` 是能力声明；`cumulative_length` 本身也是 tensor，避免每次增长时把 Python 整数带进编译图。
   - English: `is_compileable=True` declares the intended usage. `cumulative_length` is itself a tensor so length updates do not become Python-side graph changes.
2. **第 419-444 行 / Lines 419-444 (lazy allocation)**:
   - 中文：第一次 update 才知道 batch size、head 数、dtype 和 device，所以延迟分配；但一旦分配，大小直接是 `max_cache_len`。
   - English: The first update reveals batch size, head count, dtype, and device, so allocation is delayed. Once allocated, the buffer is already `max_cache_len`.
3. **第 445-453 行 / Lines 445-453 (`mark_static_address`)**:
   - 中文：固定数据指针是 CUDA graph 的关键。原地 mutation 仍然改变内容，但不会换掉 tensor 对象的地址。
   - English: A fixed data pointer is important for CUDA graphs. In-place mutation changes contents without replacing the tensor address.
4. **第 470-478 行 / Lines 470-478 (positions and cursor)**:
   - 中文：新 token 的长度决定一段连续 `cache_position`，然后用 `add_` 推进 cursor，整个操作保留在 tensor 世界里。
   - English: The new token length defines a contiguous range of positions. `add_` advances the cursor in tensor space.
5. **第 480-489 行 / Lines 480-489 (in-place write)**:
   - 中文：主路径是 `index_copy_`，MPS 等不支持时才退到普通索引赋值；两条路径都返回完整的静态 buffer。
   - English: `index_copy_` is the main path, with indexed assignment as a fallback for devices such as MPS. Both return the full static buffer.

## 类比 / The analogy

中文：像给仓库预先画好一排固定货架。每次来一箱货，只记录它该放在哪几个格子，而不是把整个仓库重新搭一遍；编译器也因此能一直盯着同一组地址。

English: Think of a warehouse with fixed shelves. Each new box is placed into the next slots instead of rebuilding the warehouse, so the compiler keeps seeing the same addresses.

## 自己跑一遍 / Try it yourself

```python
class StaticCache:
    def __init__(self, capacity):
        self.buf = [None] * capacity
        self.cursor = 0

    def update(self, values):
        end = self.cursor + len(values)
        self.buf[self.cursor:end] = values
        self.cursor = end
        return self.buf

cache = StaticCache(5)
print(cache.update(["a", "b"]))
print(cache.update(["c"]))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
['a', 'b', None, None, None]
['a', 'b', 'c', None, None]
```

中文：容量预留和 cursor 分离后，更新只改内容，不改变 buffer 的身份；这正是静态 cache 对编译器友好的原因。

English: Once capacity and cursor are separate, updates change contents without changing buffer identity. That is why static caches compose well with compilation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Paged KV cache** / **Paged KV cache**: 用 page table 管理固定大小的物理块。 / Manage fixed-size physical blocks through a page table.
- **Open-Sora VAE chunk cache** / **Open-Sora VAE chunk cache**: 预留时间边界并携带上一块的状态。 / Preserve temporal boundaries and carry state across chunks.
- **CUDA graph input buffers** / **CUDA graph input buffers**: 图捕获后，地址稳定比 Python 对象是否相同更重要。 / After capture, stable addresses matter more than Python object identity.

## 注意事项 / Caveats / when it breaks

- **必须预估最大长度** / **You must choose a maximum length**: 太小会溢出，太大会浪费显存。
- **静态长度不等于静态内容** / **Static length does not mean static content**: mask 仍然要根据真实 cursor 生成。
- **原地写入需要设备支持** / **In-place writes need device support**: fallback 路径要保留，但最好 benchmark 两条路径。

## 延伸阅读 / Further reading

- [Transformers `StaticLayer`](https://github.com/huggingface/transformers/blob/df04b012229d50d2b6dfba32c61c3057c3a40ea1/src/transformers/cache_utils.py)
- [Transformers KV cache guide](https://huggingface.co/docs/transformers/main/en/kv_cache)
