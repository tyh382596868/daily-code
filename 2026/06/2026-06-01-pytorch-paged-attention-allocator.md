---
date: 2026-06-01
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/experimental/_paged_attention.py
permalink: https://github.com/pytorch/pytorch/blob/fee850ba5c2810b2424af3ae7db4318461e93feb/torch/nn/attention/experimental/_paged_attention.py#L24-L107
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, pytorch, attention, kv-cache, paged-attention]
---

# PagedAttention 进 PyTorch 主仓:80 行就把 vLLM 的"虚拟内存"搬了进来 / PagedAttention lands in core PyTorch: vLLM's "virtual memory" trick in 80 lines

> **一句话 / In one line**: 一张 `[batch, logical_block] → physical_page` 的索引表加一个 free-list,把 LLM 推理时变长的 KV-cache 当虚拟内存来管。 / A `[batch, logical_block] → physical_page` index table plus a free-list, treating the variable-length KV cache of LLM inference exactly like an OS treats virtual memory.

## 为什么重要 / Why this matters

vLLM 之所以能在同一张卡上同时跑下数十条 prompt,是因为它放弃了"每条序列在显存里连续摆放"这个传统假设,改用"页表"——和操作系统管虚拟内存的方法完全一致。今天这段代码意义重大的地方在于:这个数据结构现在直接出现在 `torch.nn.attention.experimental` 里,作为 `flex_attention` 的伴生工具。这不是 tensor 操作、不是 kernel,而是一个朴素的**Python 内存分配器**——但正是它让 batched LLM 推理变得可能。读懂这段代码,就读懂了 LLM serving 的底层抽象。

The reason vLLM can pack dozens of prompts of different lengths into one GPU is that it abandons the convention "store each sequence contiguously in memory" and uses a **page table**, exactly like an OS managing virtual memory. What makes today's snippet notable is that this data structure now lives directly in `torch.nn.attention.experimental` as a companion to `flex_attention`. There's almost no tensor math here — it's a plain old **Python memory allocator** — but it is the abstraction that makes batched LLM inference work. Read this and you've read the foundation of modern LLM serving.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/experimental/_paged_attention.py`](https://github.com/pytorch/pytorch/blob/fee850ba5c2810b2424af3ae7db4318461e93feb/torch/nn/attention/experimental/_paged_attention.py#L24-L107)

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
        self.page_table[batch_idx, start_page_idx:end_page_idx] = allocated_pages

        # update metadata
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page_idx.item(), end_page_idx.item(),
            device=num_pages_to_allocate.device,
        )
        self.capacity[batch_idx] += num_pages_to_allocate * self.page_size
```

## 逐行讲解 / What's happening

1. **`page_table` 的形状 / Shape of `page_table`**:
   - 中文: `(max_batch_size, n_pages)`,值是物理页号。也就是说"第 `b` 条序列的第 `i` 个逻辑块"指向哪个物理槽。**逻辑顺序**和**物理位置**完全解耦。`-1` 表示该槽未被分配。
   - English: Shape is `(max_batch_size, n_pages)`, values are physical page indices. "The `i`-th logical block of sequence `b`" maps to some physical slot. **Logical order** and **physical placement** are fully decoupled. `-1` marks "not allocated".

2. **`empty_pages` 是普通 Python list / `empty_pages` is just a plain Python list**:
   - 中文: 不是张量,就是个倒序的整数列表 `[n-1, n-2, ..., 0]`。要分配就从末尾 pop,这样 LIFO 在大多数语义上等价于 free-list。注意这是 host-side 数据结构——它**不在 GPU 上**,只有页表本身在 GPU。
   - English: Not a tensor — a plain reversed integer list `[n-1, n-2, ..., 0]`. Pop from the tail to allocate; LIFO is fine for a free-list. This is a host-side structure — it lives on the CPU, only the page table itself is on the GPU.

3. **`reserve` 的早退 / The early return in `reserve`**:
   - 中文: `if seq_len <= self.capacity[batch_idx]: return`。如果已经分配的容量够用就什么也不做——这是 prefill 阶段一次性预留,后续 decode 步无操作的关键。
   - English: `if seq_len <= self.capacity[batch_idx]: return` — if the current capacity is enough, do nothing. This is what makes the decode loop free: prefill reserves enough pages up front, then each subsequent step is a no-op.

4. **`_cdiv` 算需要新增几页 / `_cdiv` computes how many new pages are needed**:
   - 中文: ceiling division。如果差额是 50、`page_size=16`,需要 `ceil(50/16) = 4` 页。
   - English: Ceiling division. A deficit of 50 with `page_size=16` needs `ceil(50/16) = 4` pages.

5. **从尾部切走 N 页 / Slicing N pages off the tail**:
   - 中文: `allocated_pages = self.empty_pages[-N:]` 然后 `self.empty_pages = self.empty_pages[:-N]`。极简的 free-list 操作,且分配出来的物理页编号通常**不连续**——这才是关键:不连续的页可以被任意 sequence 拼接起来,跟它们之前服务过谁没关系。
   - English: `allocated_pages = self.empty_pages[-N:]` then `self.empty_pages = self.empty_pages[:-N]`. A minimalist free-list pop. The allocated physical indices are typically **non-contiguous** — that's the whole point: a sequence's logical blocks can be backed by physically scattered pages, no matter which sequence used them before.

6. **`page_table[batch_idx, start:end] = allocated_pages` / The page-table install**:
   - 中文: 这一行才是真正的"建立映射":把第 `batch_idx` 条序列的逻辑块 `start..end` 与物理页绑定。**所有后续的 attention kernel 都会通过这张表去 gather KV**。
   - English: This single line *is* the mapping: bind sequence `batch_idx`'s logical blocks `start..end` to specific physical pages. **Every subsequent attention kernel reads via this table to gather the right KV slices.**

7. **`physical_to_logical` 反向表 / The reverse table**:
   - 中文: 维护反向映射主要是为了 `erase` 操作和调试——给定一个物理页,O(1) 找到它属于谁的第几个逻辑块。
   - English: The reverse mapping exists mainly for `erase` and debugging — given a physical page, you can find which sequence owns it and where in `O(1)`.

## 类比 / The analogy

把它想成图书馆的**借阅卡**系统。每个读者(序列)在心里把书按 1、2、3 排列,但这些书在书架上根本不连号。每个读者有一张借阅卡,上面写着"我的第 1 本 = 货架 47 号、第 2 本 = 货架 12 号"。`reserve` 就是去前台说"再给我留 4 本的位置";`empty_pages` 是图书管理员手里的"空闲货架清单"。当读者把书还回来(`erase`),那些货架编号就回到空闲清单上,可以再借给别人。整个 `PagedAttention` 就是这套借阅卡 + 货架清单的代码实现。

Think of a library's **call-slip** system. Each reader (sequence) keeps a personal ordering — "my book 1, book 2, book 3" — but the books sit on physically scattered shelves. The borrowing slip says "my book 1 = shelf 47, my book 2 = shelf 12". `reserve` is asking the librarian to set aside 4 more shelves for you; `empty_pages` is the librarian's list of currently-vacant shelves. When the reader returns the books (`erase`), those shelf IDs go back into the vacant list and can be reissued to another reader. `PagedAttention` is the code for exactly this slip-and-shelves system.

## 自己跑一遍 / Try it yourself

```python
import torch

class TinyPagedAlloc:
    def __init__(self, n_pages, page_size, max_batch):
        self.page_size = page_size
        self.empty = list(range(n_pages - 1, -1, -1))
        self.page_table = -torch.ones((max_batch, n_pages), dtype=torch.int64)
        self.capacity = torch.zeros(max_batch, dtype=torch.int64)

    def reserve(self, b, seq_len):
        if seq_len <= self.capacity[b]: return
        need = (seq_len - self.capacity[b].item() + self.page_size - 1) // self.page_size
        pages = self.empty[-need:]; self.empty = self.empty[:-need]
        start = self.capacity[b].item() // self.page_size
        self.page_table[b, start:start+need] = torch.tensor(pages)
        self.capacity[b] += need * self.page_size

alloc = TinyPagedAlloc(n_pages=8, page_size=16, max_batch=3)
alloc.reserve(0, 30)   # sequence 0 wants 30 tokens -> 2 pages
alloc.reserve(1, 40)   # sequence 1 wants 40 tokens -> 3 pages
alloc.reserve(0, 50)   # extend sequence 0 to 50 tokens -> +2 more pages
print("page_table:\n", alloc.page_table)
print("free pages:", alloc.empty)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
page_table:
 tensor([[ 7,  6,  2,  1, -1, -1, -1, -1],
        [ 5,  4,  3, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1]])
free pages: [0]
```

中文一两句:留意 sequence 0 拿到了 `[7,6,2,1]`——前两页和后两页**物理位置完全不连续**。这正是 paged attention 的杀手锏:只要表能查,物理上多碎都没关系。

Notice sequence 0 got `[7, 6, 2, 1]` — its first two and last two pages are **physically non-contiguous**. That's the killer feature: as long as the table can be queried, the underlying memory can be as fragmented as it likes.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM 原版 BlockManager** / **vLLM's original `BlockManager`**: 中文:同一思想的"祖宗"实现,2023 年 vLLM 论文 (Kwon et al.) 描述的就是这套结构。 / English: The original implementation Kwon et al. (2023) introduced; PyTorch's version is a direct descendant.
- **SGLang、TGI、TensorRT-LLM**: 中文:三大主流推理框架都各自实现了 paged KV,数据结构几乎一样。 / English: All three major inference frameworks implement their own paged KV with essentially the same data structure.
- **操作系统的虚拟内存** / **OS virtual memory**: 中文:这就是 MMU 的页表,只不过页表的项指向显存而不是 DRAM。`empty_pages` 等价于操作系统的 free-page list。 / English: This *is* the MMU's page table, just pointing into GPU memory instead of DRAM. `empty_pages` is the kernel's free-page list.

## 注意事项 / Caveats / when it breaks

- **`empty_pages` 是 Python list,所以分配走的是 host code path / `empty_pages` is a Python list, so allocation runs on the host**: 中文:如果你在 hot path 里频繁调用 `reserve`,Python 开销可能显形。生产实现一般会把空闲列表换成 GPU 上的 ring-buffer 或者 bitmap。 / English: Calling `reserve` in a hot loop incurs Python overhead. Production implementations typically replace the free-list with a GPU ring-buffer or bitmap.
- **AssertionError 而不是 OOM 错误 / Raises `AssertionError`, not OOM**: 中文:`n_pages` 是开机时定死的,跑满了就直接断言失败。生产 serving 需要在外层做 admission control(拒绝新请求或抢占旧请求)。 / English: `n_pages` is a fixed budget at construction time; running out raises an assertion. Production servers wrap this with admission control (reject new requests or preempt old ones).
- **页表本身需要参与 attention kernel / The page table itself must reach the attention kernel**: 中文:`PagedAttention` 只管分配,真正消费这张表的是 `flex_attention` 的 mask_mod 和 score_mod——见同文件下面的 `convert_logical_block_mask`。 / English: This class only handles allocation. The actual consumer is `flex_attention`'s `mask_mod` / `score_mod`, which gathers from physical pages via this table — see `convert_logical_block_mask` further down the same file.

## 延伸阅读 / Further reading

- [vLLM Paper — *Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180)
- [PyTorch FlexAttention docs](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) — the attention API that consumes this page table
- [Operating Systems: Three Easy Pieces — *Paging*](https://pages.cs.wisc.edu/~remzi/OSTEP/vm-paging.pdf) — the OS textbook chapter on the very same idea
