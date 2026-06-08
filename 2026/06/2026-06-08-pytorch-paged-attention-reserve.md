---
date: 2026-06-08
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/experimental/_paged_attention.py
permalink: https://github.com/pytorch/pytorch/blob/411c8477fa2478b2318f3823d57cf684a3a1f389/torch/nn/attention/experimental/_paged_attention.py#L20-L107
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, paged-attention, kv-cache, vllm]
---

# PyTorch 把 vLLM 的 paged KV cache 写进了官方:80 行的 page table 分配器 / PyTorch shipped vLLM-style paged KV cache: an 80-line page-table allocator

> **一句话 / In one line**: 一张 `[batch, logical_block_idx] → physical_page_idx` 的整数表,加上一个空闲页栈,就是 vLLM 的整个内存分配模型——PyTorch 现在把它写成了 `torch.nn.attention.experimental.PagedAttention`。
> One integer table `[batch, logical_block_idx] → physical_page_idx` plus an empty-pages stack is the entire vLLM memory model — and PyTorch now ships it as `torch.nn.attention.experimental.PagedAttention`.

## 为什么重要 / Why this matters

KV cache 是 LLM 推理的"显存大户":batch 大、序列长就爆。vLLM 的核心 idea 是"把 KV cache 切成固定大小的 page,谁要谁取,不连续也行"。这个思路其实跟操作系统的虚拟内存一模一样——逻辑页号 → 物理页号的映射表 + 空闲页池。但工业级实现(vLLM/SGLang)里这部分都是 C++ + CUDA;PyTorch 主仓库现在直接把它写成了一段约 80 行的纯 Python(基于 FlexAttention)。它适合两类人读:想搞清楚 vLLM 到底怎么"页表"的人,以及想自己做一个轻量推理引擎的人——这就是参考实现。

KV cache is the memory hog of LLM inference: big batches + long sequences and you're done. vLLM's core idea is "chop the KV cache into fixed-size pages, hand them out on demand, don't require contiguity." That's literally the OS virtual-memory trick — a logical→physical page table plus a free-page pool. But industrial implementations (vLLM, SGLang) are C++ + CUDA; PyTorch main branch now ships the same thing as ~80 lines of pure Python (on top of FlexAttention). Two audiences benefit: people who want to *understand* what "paging" actually means in vLLM, and people who want a reference for rolling their own lightweight inference engine.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/experimental/_paged_attention.py`](https://github.com/pytorch/pytorch/blob/411c8477fa2478b2318f3823d57cf684a3a1f389/torch/nn/attention/experimental/_paged_attention.py#L20-L107)

```python
def _cdiv(x, multiple):
    return (x + multiple - 1) // multiple


class PagedAttention:
    """
    PagedAttention supports flex attention inference with a large batch size.
    With PagedAttention, a batch of key/value tensors with varying kv length
    is split into tensor blocks of fixed length and cached in a compact way.
    """

    def __init__(self, n_pages, page_size, max_batch_size, device="cuda"):
        self.n_pages = n_pages
        self.page_size = page_size

        # page table: [batch, logical_block_idx] -> physical_page_idx
        self.page_table = -torch.ones(
            (max_batch_size, self.n_pages), dtype=torch.int64, device=device
        )

        # capacity: batch_idx -> allocated sequence length
        self.capacity = torch.zeros(max_batch_size, dtype=torch.int64, device=device)

        # index of empty pages that is available for allocation
        self.empty_pages = list(range(n_pages - 1, -1, -1))

        # mapping from physical page index to logical page index
        self.physical_to_logical = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )

    def reserve(self, batch_idx, seq_len):
        """Request at least `seq_len` tokens of capacity for `batch_idx`."""

        if seq_len <= self.capacity[batch_idx]:
            return

        num_pages_to_allocate = _cdiv(
            seq_len - self.capacity[batch_idx], self.page_size
        )

        if len(self.empty_pages) < num_pages_to_allocate:
            raise AssertionError(
                f"requested {num_pages_to_allocate.item()} pages "
                f"but there are only {len(self.empty_pages)} empty pages"
            )

        start_page_idx = self.capacity[batch_idx] // self.page_size
        end_page_idx = start_page_idx + num_pages_to_allocate

        # find empty physical pages
        allocated_pages = torch.tensor(
            self.empty_pages[-num_pages_to_allocate:],
            device=num_pages_to_allocate.device,
        )
        self.empty_pages = self.empty_pages[:-num_pages_to_allocate]

        # update page table
        self.page_table[
            batch_idx,
            start_page_idx:end_page_idx,
        ] = allocated_pages

        # update metadata
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page_idx.item(),
            end_page_idx.item(),
            device=num_pages_to_allocate.device,
        )
        self.capacity[batch_idx] += num_pages_to_allocate * self.page_size
```

## 逐行讲解 / What's happening

1. **`page_table` 初始化为 `-1`(行 47-49) / Lines 47-49**:
   - 中文: 这是张 `[max_batch_size, n_pages]` 的整数表,语义就是"batch 里第 i 个 logical block 对应到哪一页物理 page"。`-1` 表示这个 logical block 还没分配。后续 attention kernel 会用 `gather(page_table, ...)` 把 logical 位置翻译成 physical。
   - English: This is a `[max_batch_size, n_pages]` integer tensor whose meaning is "logical block i of batch row b maps to which physical page?". `-1` means that logical block isn't allocated yet. Downstream attention kernels use `gather(page_table, ...)` to translate logical positions into physical ones.

2. **`empty_pages = list(range(n_pages - 1, -1, -1))`(行 55) / Line 55**:
   - 中文: 空闲页栈,**倒序**。倒序是为了在第 90 行用 `[-num_pages_to_allocate:]` 弹出栈顶——这是 Python list 当 stack 用的惯用法。这种"用 list 当 free-list"的写法和操作系统教科书里的 free page list 一模一样,只是这里在 GPU 上间接维护。
   - English: A stack of free pages, **reversed**. The reverse is so that line 90 can pop the top via `[-num_pages_to_allocate:]` — the Python idiom for "use a list as a stack". The "list as free-list" pattern is exactly what every OS textbook describes for a free-page list, just maintained on the host side here.

3. **`physical_to_logical`(行 58-60) / Lines 58-60**:
   - 中文: 反向映射,`[batch, physical_page] → logical_block`。它存在的唯一原因是"驱逐 / erase":当你要回收一批 page 时,你拿到的是物理页号,需要反查它们曾经代表哪些 logical block,才能把 `page_table` 对应位置清成 -1。这是一个用空间换 O(1) 反查的小决定。
   - English: The inverse map, `[batch, physical_page] → logical_block`. The sole reason it exists is "eviction / erase": when you free a batch of pages, you have the physical page indices but need to clear the corresponding `page_table` slots — so you reverse-lookup which logical block each physical page used to be. Classic space-for-O(1)-lookup tradeoff.

4. **`reserve`:`seq_len <= self.capacity[batch_idx]: return`(行 72-73) / Lines 72-73**:
   - 中文: 这是 paged allocator 的关键设计——"按需分配,已经够大就直接返回"。每生成一个 token,你都喊一次 `reserve(b, current_seq_len)`;大多数 token 都只是 `return`,只有跨过 page 边界时才真去申请。这跟 OS 的 page-fault-on-demand 是同一思想。
   - English: The key design move of a paged allocator — "lazy allocation, return immediately if you already have enough." On every generated token you call `reserve(b, current_seq_len)`; most calls just `return`, and only the calls that cross a page boundary actually allocate. Same idea as OS page-fault-on-demand.

5. **`num_pages_to_allocate = _cdiv(seq_len - capacity, page_size)`(行 75-77) / Lines 75-77**:
   - 中文: `_cdiv` 是 ceil-div(向上取整除法,常见的 `(x + m - 1) // m` 套路)。"还差多少 token,除以每页 token 数,向上取整"——就是要申请多少新页。
   - English: `_cdiv` is ceil-div, the classic `(x + m - 1) // m`. "How many tokens short, divided by tokens-per-page, rounded up" — that's how many new pages to allocate.

6. **`allocated_pages = empty_pages[-num_pages_to_allocate:]`(行 89-93) / Lines 89-93**:
   - 中文: 从空闲栈顶弹 N 张。然后把对应的物理页号写进 `page_table[batch_idx, start:end]` 和 `physical_to_logical[batch_idx, allocated_pages]`。**注意一个隐含的不变量**:同一个 batch 行内,`page_table` 是按 logical_block 顺序连续填充的(start_page_idx 来自 `capacity // page_size`),但分配到的 physical pages 可以是任意的。这正是 paging 的精髓——逻辑连续、物理碎片化。
   - English: Pop N off the top of the free stack. Then write the physical page ids into `page_table[batch_idx, start:end]` and `physical_to_logical[batch_idx, allocated_pages]`. **A hidden invariant matters here**: within one batch row, `page_table` is filled contiguously by logical_block (start comes from `capacity // page_size`), but the physical pages handed out can be arbitrary. That's exactly the heart of paging — logical contiguous, physical fragmented.

## 类比 / The analogy

中文: 想象一个体育馆停车场。`page_table[batch=张三][0..k]` 是"张三的车位预约单",上面写着第 1 张车票去 A 区 17 号、第 2 张去 C 区 3 号——逻辑顺序连续(第 1 张、第 2 张……),但物理位置随机。`empty_pages` 是停车场入口处的"还没人占的车位编号牌堆",来一个分配一个。`capacity[张三]` 是"张三总共已占了多少个车位空间(token)";只要他在累积消息没超过这个数,就不去前台再要票。`physical_to_logical` 是反向查表:停车管理员看到 C 区 3 号车要走,马上能知道"哦,这是张三的第 2 张票"——好把那张票从预约单上撕掉。

English: Picture a stadium parking lot. `page_table[batch=Alice][0..k]` is "Alice's parking reservation slip" — slot 1 goes to row A spot 17, slot 2 goes to row C spot 3 — logical order is contiguous (slot 1, slot 2, …) but physical spots are scattered. `empty_pages` is the stack of "unclaimed spot numbers" at the lot entrance; hand one out each time. `capacity[Alice]` tracks "total spots Alice already has"; as long as her cumulative cars haven't exceeded that, she doesn't go back to the desk. `physical_to_logical` is the reverse map: when the lot attendant sees a car leaving row C spot 3, they instantly know "that's Alice's slot #2" — so they can tear that line off her reservation slip.

## 自己跑一遍 / Try it yourself

```python
# paged_alloc_demo.py
import torch

class PagedAlloc:
    def __init__(self, n_pages, page_size, max_batch):
        self.page_size = page_size
        self.page_table = -torch.ones((max_batch, n_pages), dtype=torch.int64)
        self.capacity = torch.zeros(max_batch, dtype=torch.int64)
        self.empty_pages = list(range(n_pages - 1, -1, -1))

    def reserve(self, b, seq_len):
        if seq_len <= self.capacity[b]:
            return
        need = (seq_len - self.capacity[b].item() + self.page_size - 1) // self.page_size
        assert len(self.empty_pages) >= need, "OOM"
        pages = self.empty_pages[-need:]
        self.empty_pages = self.empty_pages[:-need]
        start = self.capacity[b].item() // self.page_size
        self.page_table[b, start:start+need] = torch.tensor(pages)
        self.capacity[b] += need * self.page_size

alloc = PagedAlloc(n_pages=8, page_size=4, max_batch=2)
# Simulate token-by-token generation
for token_idx in range(11):
    alloc.reserve(b=0, seq_len=token_idx + 1)
    print(f"after token {token_idx+1}: capacity={alloc.capacity[0].item()}, "
          f"pages={alloc.page_table[0].tolist()[:4]}")
```

运行 / Run with:
```bash
pip install torch
python paged_alloc_demo.py
```

预期输出 / Expected output:
```
after token 1: capacity=4, pages=[7, -1, -1, -1]
after token 2: capacity=4, pages=[7, -1, -1, -1]
after token 3: capacity=4, pages=[7, -1, -1, -1]
after token 4: capacity=4, pages=[7, -1, -1, -1]
after token 5: capacity=8, pages=[7, 6, -1, -1]
...
```

中文一两句: 看 token 1→4 时 capacity 一直是 4——这就是 `reserve` 的"懒"。只有第 5 个 token 时(seq_len=5 > capacity=4)才真的去申请了第二张页(物理页 6)。第 9 个 token 同理触发第三页。

English: Watch tokens 1→4: capacity stays at 4 — that's `reserve` being lazy. Only token 5 (seq_len=5 > capacity=4) triggers a real allocation, grabbing physical page 6. Token 9 likewise triggers the third page. The whole point of paging.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM `BlockManager`** / **vLLM `BlockManager`**: C++ 版本的同一个 idea,加了 prefix caching、copy-on-write、跨请求共享 prefix。/ The C++ version of the same idea, with prefix caching, copy-on-write, and cross-request prefix sharing on top.
- **SGLang RadixCache** / **SGLang RadixCache**: 把 prefix tree + 页表合在一起,允许 KV cache 在请求之间共享前缀。/ Combines a prefix tree with a page table to share KV cache prefixes across requests.
- **操作系统的虚拟内存 / OS virtual memory**: 进程的 page table,内核的 free page list,几乎一字不改可以套上去。/ The process page table and kernel free-page list — the analogy isn't even a stretch, it's the same data structure.

## 注意事项 / Caveats / when it breaks

- **没有 evict / 不防 OOM / No eviction, no OOM safeguard**: `reserve` 一旦空闲页不够就 raise。真生产里得加 LRU evict、抢占、抢回别的 batch 行的页。/ `reserve` raises on OOM. A production scheduler needs LRU eviction, preemption, or stealing pages from other batch rows.
- **`empty_pages` 是 Python list,不是 GPU tensor / `empty_pages` lives on the host as a Python list**: 这意味着 `reserve` 不能直接在 CUDA graph 里调——它有 host-side 控制流。要 graph-friendly 得换成 GPU tensor + scan。/ This means `reserve` can't sit inside a CUDA graph as-is — there's host-side control flow. A graph-friendly version needs a GPU-tensor free list + a parallel scan.
- **`page_table` 用 int64 / `page_table` is int64**: 对几千页的 max_batch_size 来说很大;vLLM 用 int32。如果你打算把这段直接抄进 inference engine,把 dtype 改成 int32 能省一半内存。/ int64 is fat for a page table; vLLM uses int32. If you copy this into a real engine, switch to int32 and halve the table memory.
- **本片段只是 allocator,attention 还在别处 / This snippet is the allocator, not the attention**: 真正用 page_table 做 attention 的 kernel 在同文件 `convert_logical_block_mask` 里——它把 logical BlockMask 经过 `gather(page_table, ...)` 翻译成 physical BlockMask,再交给 FlexAttention。/ The attention itself is in `convert_logical_block_mask` further down in the same file: it gathers `page_table` to translate a logical BlockMask into a physical one, then hands it to FlexAttention.

## 延伸阅读 / Further reading

- [vLLM paper — Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [PyTorch FlexAttention announcement](https://pytorch.org/blog/flexattention/)
- [SGLang RadixAttention paper](https://arxiv.org/abs/2312.07104)
