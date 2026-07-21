---
date: 2026-07-08
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
permalink: https://github.com/vllm-project/vllm/blob/0ca6eee7433893e94f3d4be00cedb3e9f3d6c44f/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L124-L216
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, offloading]
---

# vLLM KV offloading：先按注意力组算对齐边界 / vLLM KV Offloading: Compute Alignment Per Attention Group First

> **一句话 / In one line**: vLLM 在调度 KV offload 前，先把每个 KV group 的 block size、sliding-window 范围和 EAGLE 易变尾块编码成统一配置。 / Before scheduling KV offload, vLLM normalizes each KV group into one config object with block size, sliding-window span, and volatile EAGLE-tail metadata.

## 为什么重要 / Why this matters

大模型服务里的 KV cache 不一定是一整块同质内存。混合架构可能同时有 full-attention、sliding-window attention、Mamba state，甚至 speculative decoding 的 draft attention group。vLLM 这段代码的价值在于：它没有把这些差异散落到调度器各处，而是在 `from_spec()` 里提前算成一组 `GroupOffloadConfig`。

KV cache in production serving is rarely one uniform buffer. Hybrid models can mix full attention, sliding-window attention, Mamba state, and draft groups for speculative decoding. This code is useful because it turns those differences into `GroupOffloadConfig` records up front, instead of leaking special cases through the scheduler.

## 代码 / The code

`vllm-project/vllm` — [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`](https://github.com/vllm-project/vllm/blob/0ca6eee7433893e94f3d4be00cedb3e9f3d6c44f/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L124-L216)

```python
@classmethod
def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
    # Determine the alignment token count from the full-attention group(s).
    # This is the offloaded_block_size of the full-attention group; load
    # hits are always aligned to this boundary, so SWA blocks earlier in
    # each segment can never serve a load hit. Relevant for hybrid
    # architectures like DeepSeek V4 (MLA + SWA groups).
    full_attn_offloaded_block_sizes: set[int] = set()
    for idx, gpu_block_size in enumerate(spec.gpu_block_size):
        kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        sw = get_sliding_window_size_in_blocks(
            kv_spec, gpu_block_size * spec.block_size_factor
        )
        if sw is None:
            full_attn_offloaded_block_sizes.add(
                gpu_block_size * spec.block_size_factor
            )

    alignment_tokens: int | None = None
    if len(full_attn_offloaded_block_sizes) == 1:
        alignment_tokens = full_attn_offloaded_block_sizes.pop()

    def _alignment_block_count(
        offloaded_block_size: int,
        sliding_window_size_in_blocks: int | None,
    ) -> int | None:
        if alignment_tokens is None or sliding_window_size_in_blocks is None:
            return None
        if alignment_tokens <= offloaded_block_size:
            return None
        per_segment = alignment_tokens // offloaded_block_size
        if sliding_window_size_in_blocks >= per_segment:
            return None
        return per_segment

    eagle_groups = {
        idx
        for idx, g in enumerate(spec.kv_cache_config.kv_cache_groups)
        if g.is_eagle_group
    }

    return cls(
        num_workers=spec.vllm_config.parallel_config.world_size,
        kv_group_configs=tuple(
            GroupOffloadConfig(
                group_idx=idx,
                gpu_block_size=gpu_block_size,
                offloaded_block_size=gpu_block_size * spec.block_size_factor,
                hash_block_size_factor=(
                    (gpu_block_size * spec.block_size_factor)
                    // spec.hash_block_size
                ),
                sliding_window_size_in_blocks=(
                    sw := get_sliding_window_size_in_blocks(
                        spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                        gpu_block_size * spec.block_size_factor,
                    )
                ),
                alignment_block_count=_alignment_block_count(
                    gpu_block_size * spec.block_size_factor, sw
                ),
                kv_event_group_spec=get_offloading_event_group_spec(
                    spec.kv_cache_config.kv_cache_groups[idx]
                ),
                is_eagle_group=idx in eagle_groups,
            )
            for idx, gpu_block_size in enumerate(spec.gpu_block_size)
        ),
```

## 逐行讲解 / What's happening

1. **第 129-142 行 / Lines 129-142 (`full_attn_offloaded_block_sizes`)**:
   - 中文: 先找 full-attention group 的 offloaded block size。后续 load hit 要按这个边界对齐，所以它是全局参考尺。
   - English: It first collects the offloaded block sizes for full-attention groups. Load hits align to this size, so it becomes the reference ruler.
2. **第 145-159 行 / Lines 145-159 (`_alignment_block_count`)**:
   - 中文: 如果 sliding-window group 的窗口小于 full-attention 对齐段，前面的 SWA blocks 永远不会成为有效命中，可以在 store/load 调度里跳过。
   - English: If a sliding-window group covers less than a full-attention alignment segment, earlier SWA blocks can never be useful load hits and can be skipped.
3. **第 161-166 行 / Lines 161-166 (`eagle_groups`)**:
   - 中文: speculative decoding 的 draft group 尾块不稳定，配置里直接打标，避免后面误 offload。
   - English: Draft groups used by speculative decoding have volatile trailing blocks, so the scheduler marks them directly in the config.

## 类比 / The analogy

这像给一个仓库装货前先量托盘：有些货箱按整托盘走，有些只能按半托盘窗口移动，还有些尾箱随时会换。调度员先拿到每类货的规格表，后面搬运就不用边搬边猜。

It is like measuring pallets before loading a warehouse truck. Some boxes move by full pallets, some only within a half-pallet window, and some tail boxes may change. Once the dispatcher has the size sheet, the loading plan stops guessing.

## 自己跑一遍 / Try it yourself

```python
def alignment_block_count(full_sizes, offloaded, sw_blocks):
    alignment = full_sizes[0] if len(set(full_sizes)) == 1 else None
    if alignment is None or sw_blocks is None:
        return None
    if alignment <= offloaded:
        return None
    per_segment = alignment // offloaded
    return None if sw_blocks >= per_segment else per_segment

print(alignment_block_count([16], offloaded=4, sw_blocks=2))
print(alignment_block_count([16], offloaded=4, sw_blocks=4))
print(alignment_block_count([16, 32], offloaded=4, sw_blocks=2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
4
None
None
```

第一行说明：full-attention 每 16 tokens 对齐一次，SWA block 只有 4 tokens，窗口只覆盖 2 个 block，所以每 4 个 SWA block 里只有对齐边界有意义。

The first line says: full attention aligns every 16 tokens, each SWA block has 4 tokens, and the window covers only 2 blocks, so only the segment boundary matters.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM block manager** / **vLLM block manager**: 先把物理块和逻辑块拆开，再让调度器只处理 block id。 / It separates physical blocks from logical blocks so the scheduler only handles block ids.
- **LMCache key design** / **LMCache key design**: 把分布式 KV chunk 的地址先规范化，再做传输和查找。 / It normalizes distributed KV chunk addresses before transfer and lookup.

## 注意事项 / Caveats / when it breaks

- **对齐前提必须一致** / **Alignment must be consistent**: 如果 full-attention group 有多个不同 block size，这段代码会放弃 alignment optimization。 / If full-attention groups disagree on block size, the optimization is disabled.
- **配置不是传输本身** / **The config is not the transfer**: 它只是给后续 store/load 调度提供边界，真正的数据搬运还在 worker 侧。 / It only prepares boundaries for later store/load scheduling; workers still perform the data movement.

## 延伸阅读 / Further reading

- [vLLM KV transfer offloading scheduler](https://github.com/vllm-project/vllm/blob/0ca6eee7433893e94f3d4be00cedb3e9f3d6c44f/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py)
