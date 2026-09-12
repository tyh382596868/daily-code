---
date: 2026-08-01
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/kv_cache_manager.py
permalink: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L271-L310
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, connector]
---

# vLLM hybrid KV connector：先问外部缓存，再分配本地 block / vLLM Hybrid KV Connector: Ask External Cache Before Allocating Local Blocks

> **一句话 / In one line**: `get_computed_blocks_for_connector` 把本地 prefix-cache 命中和外部 KV connector 命中合并成一次调度决策。 / `get_computed_blocks_for_connector` merges local prefix-cache hits with external KV-connector hits into one scheduling decision.

## 为什么重要 / Why this matters

LLM serving 里的 KV cache 不再只住在单机 GPU 内存里。生产系统会把 KV 放进 CPU、远端 cache 或跨进程 connector；调度器必须先知道“哪些 block 已经算过”，再决定还要分配多少本地 block。

KV cache in LLM serving no longer lives only in one GPU's memory. Production systems may keep KV in CPU memory, remote stores, or cross-process connectors; the scheduler must know which blocks are already computed before it allocates new local blocks.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L271-L310)

```python
def get_computed_blocks_for_connector(
    self, request: Request, block_hashes: list[BlockHash]
) -> tuple[list[KVCacheBlock], int]:
    """Get computed blocks from prefix caching and KV connector.

    Returns:
        A tuple of (blocks, num_computed_tokens). The blocks are local blocks
        with full block size and the num_computed_tokens may include tokens
        stored in the KV connector.
    """
    assert self.connector is not None
    computed_blocks, num_computed_tokens = self.get_computed_blocks(
        request, block_hashes)

    # Get num_computed_tokens from connector. This can include tokens that
    # are not cached locally.
    num_external_tokens = self.connector.get_num_new_matched_tokens(
        request, num_computed_tokens)
    if num_external_tokens > 0:
        logger.debug(
            "KV connector matched %d tokens for request %s",
            num_external_tokens,
            request.request_id,
        )
        num_computed_tokens += num_external_tokens

    # If some full blocks are external-only, allocate local placeholders for
    # them so the rest of the scheduler sees a normal block list.
    num_computed_blocks = num_computed_tokens // self.block_size
    if num_computed_blocks > len(computed_blocks):
        num_new_blocks = num_computed_blocks - len(computed_blocks)
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        computed_blocks.extend(new_blocks)
    return computed_blocks, num_computed_tokens
```

## 逐行讲解 / What's happening

1. **第 271-281 行 / Lines 271-281 (contract)**:
   - 中文: 返回值分成两件事：本地 block 列表，以及“已经可复用”的 token 数。
   - English: The return value separates two facts: the local block list and the number of reusable tokens.
2. **第 282-285 行 / Lines 282-285 (local cache first)**:
   - 中文: 先走普通 prefix cache，拿到本机已经持有的 block。
   - English: It first asks the normal prefix cache for blocks already present on this worker.
3. **第 289-299 行 / Lines 289-299 (external hit)**:
   - 中文: connector 只报告额外命中的 token 数；这批 token 可能还没有本地 block 对象。
   - English: The connector reports extra matched tokens; those tokens may not yet have local block objects.
4. **第 303-309 行 / Lines 303-309 (placeholder blocks)**:
   - 中文: 如果外部命中跨过完整 block 边界，就向 block pool 要占位 block，保持后续调度接口不变。
   - English: If external hits cover full block boundaries, it asks the block pool for placeholder blocks so downstream scheduling keeps the same interface.

## 类比 / The analogy

这像酒店前台分房：本地 cache 是已经在本楼的房间，外部 connector 是隔壁楼已经订好的房间。前台仍然要给客人一串房卡编号，后面的服务员不需要知道房间原本在哪里。

It is like hotel room assignment: local cache means rooms already in this building, while the external connector means rooms reserved next door. The front desk still hands out one sequence of room cards, so later staff do not need to know where the room came from.

## 自己跑一遍 / Try it yourself

```python
block_size = 4
local_blocks = ["L0"]
local_tokens = len(local_blocks) * block_size
external_tokens = 8
computed_tokens = local_tokens + external_tokens
need_blocks = computed_tokens // block_size
while len(local_blocks) < need_blocks:
    local_blocks.append(f"P{len(local_blocks)}")
print(local_blocks)
print(computed_tokens)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['L0', 'P1', 'P2']
12
```

本地只有一个真实 block，但外部 cache 又命中了两个完整 block，于是调度器补两个本地占位。

Only one real local block exists, but the external cache matched two more full blocks, so the scheduler adds two local placeholders.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LMCache connector** / **LMCache connector**: 外部 KV 系统通常先返回命中长度，再由 serving engine 负责本地调度状态。 / External KV systems often return a match length first, then let the serving engine maintain local scheduling state.
- **PagedAttention page table** / **PagedAttention page table**: 逻辑 token 位置和物理 block 位置分离，后续 kernel 只看统一映射。 / Logical token positions and physical blocks are separated, while kernels consume one unified mapping.

## 注意事项 / Caveats / when it breaks

- **只按完整 block 补位** / **Only full-block placeholders**: `num_computed_tokens // block_size` 会丢掉不满一块的尾部。 / `num_computed_tokens // block_size` ignores the partial tail.
- **connector 必须守合同** / **The connector must honor the contract**: 外部命中如果和本地 prefix 不一致，后续 block 映射会变脏。 / If external matches disagree with the local prefix, later block mapping becomes invalid.

## 延伸阅读 / Further reading

- [vLLM KV cache manager source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L271-L310)

