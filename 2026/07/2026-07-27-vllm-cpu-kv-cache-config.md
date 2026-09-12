---
date: 2026-07-27
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/simple_kv_offload/manager.py
permalink: https://github.com/vllm-project/vllm/blob/da99ffcc13362ab446a22a9bc6ed36c4c14942e3/vllm/v1/simple_kv_offload/manager.py#L179-L215
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, offload]
---

# vLLM CPU KV cache：按显存布局推导内存池 / vLLM CPU KV Cache: Derive the Host Pool from the GPU Layout

> **一句话 / In one line**: CPU offload 不是另起一套缓存格式，而是按 GPU KV cache 的 block 数和 tensor 布局缩放出一个同构的 CPU 配置。 / CPU offload does not invent a second cache format; it scales a CPU-side config from the GPU KV-cache block geometry and tensor layout.

## 为什么重要 / Why this matters

KV offload 最怕的是 GPU 和 CPU 两边对同一个 block 的大小、偏移和共享关系理解不一致。vLLM 这里先读取 GPU `KVCacheConfig`，再按可用 CPU 字节数计算 host 侧 block 数，让两边仍然使用同一套 `kv_cache_groups` 语义。

KV offload breaks quickly when the GPU and CPU sides disagree about block size, offsets, or sharing. This vLLM helper reads the GPU `KVCacheConfig`, computes how many host blocks fit in CPU memory, and preserves the same `kv_cache_groups` contract on both sides.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/simple_kv_offload/manager.py`](https://github.com/vllm-project/vllm/blob/da99ffcc13362ab446a22a9bc6ed36c4c14942e3/vllm/v1/simple_kv_offload/manager.py#L179-L215)

```python
    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        is_packed = any(t.block_stride for t in gpu_config.kv_cache_tensors)
        assert not is_packed or all(t.block_stride for t in gpu_config.kv_cache_tensors)
        gpu_total_bytes = (
            gpu_config.kv_cache_tensors[0].size
            if is_packed
            else sum(t.size for t in gpu_config.kv_cache_tensors)
        )
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
                offset=t.offset,
                block_stride=t.block_stride,
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )
```

## 逐行讲解 / What's happening

1. **第 179-184 行 / Lines 179-184 (``_derive_cpu_config``)**:
   - 中文: 入口只需要 GPU 配置和 CPU 容量，说明 CPU cache 是派生物，不是独立配置。
   - English: The helper only needs the GPU config and CPU capacity, so the host cache is derived rather than configured separately.
2. **第 191-197 行 / Lines 191-197 (`packed vs unpacked bytes`)**:
   - 中文: packed cache 用第一个 tensor 的 size 代表总量，普通布局则把所有 tensor size 相加。
   - English: Packed cache uses the first tensor size as the total, while the unpacked layout sums all tensor sizes.
3. **第 198-204 行 / Lines 198-204 (`block scaling`)**:
   - 中文: `num_cpu_blocks` 用容量比例缩放，并用 `max(1, ...)` 保证最小可用池。
   - English: `num_cpu_blocks` scales by capacity ratio and `max(1, ...)` guarantees a non-empty pool.
4. **第 201-215 行 / Lines 201-215 (`mirror tensors`)**:
   - 中文: 每个 CPU tensor 复制 `shared_by`、`offset`、`block_stride`，只缩放 size。
   - English: Each CPU tensor keeps `shared_by`, `offset`, and `block_stride`; only the size changes.

## 类比 / The analogy

这像给仓库做异地备份：货架编号、每格尺寸和共享规则保持一样，只是仓库面积变了，所以能放多少格要重新算。

It is like building an overflow warehouse: shelf numbering, slot shape, and sharing rules stay the same, but the floor area changes, so the number of slots must be recomputed.

## 自己跑一遍 / Try it yourself

```python
def derive(num_gpu_blocks, gpu_tensor_sizes, cpu_bytes, packed=False):
    gpu_total = gpu_tensor_sizes[0] if packed else sum(gpu_tensor_sizes)
    num_cpu_blocks = max(1, num_gpu_blocks * cpu_bytes // gpu_total)
    return [size // num_gpu_blocks * num_cpu_blocks for size in gpu_tensor_sizes]

print(derive(100, [4000, 4000], 20000))
print(derive(100, [8000], 32000, packed=True))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[10000, 10000]
[32000]
```

例子里 CPU tensor 的总大小按 block 数线性缩放，但每个 tensor 的相对布局没有变化。

In the toy example, CPU tensor sizes scale linearly with the number of blocks while the relative layout stays unchanged.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LMCache chunk keys** / **LMCache chunk keys**: 中文: 分布式 KV cache 也需要稳定的 block 地址。 / English: Distributed KV cache also needs stable block addresses.
- **vLLM allocate_slots** / **vLLM allocate_slots**: 中文: 调度器分配 slot 时依赖同一个 block 抽象。 / English: Scheduler slot allocation relies on the same block abstraction.

## 注意事项 / Caveats / when it breaks

- **packed 判断** / **Packed detection**: 中文: packed 与非 packed 布局不能混用，否则 size 估算会错。 / English: Packed and unpacked layouts cannot be mixed or the size estimate is wrong.
- **容量比例** / **Capacity ratio**: 中文: 这是容量推导，不代表 CPU 带宽足够快。 / English: This derives capacity, not bandwidth.

## 延伸阅读 / Further reading

- [vllm-project/vllm source](https://github.com/vllm-project/vllm/blob/da99ffcc13362ab446a22a9bc6ed36c4c14942e3/vllm/v1/simple_kv_offload/manager.py#L179-L215)
