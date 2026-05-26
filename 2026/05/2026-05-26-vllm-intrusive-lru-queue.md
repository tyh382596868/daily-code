---
date: 2026-05-26
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/kv_cache_utils.py
permalink: https://github.com/vllm-project/vllm/blob/97e4022c6ccb7b2cf1a1fc0a13a17a2a06d74f0d/vllm/v1/core/kv_cache_utils.py#L164-L327
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, data-structures, kv-cache, lru, linked-list]
---

# vLLM's Intrusive Doubly-Linked List for KV Cache LRU

> **In one line**: Why vLLM hand-rolls a linked list in pure Python instead of using `collections.deque` — and how the "sentinel node" trick eliminates every null-check branch in the hot path.

## Why this matters

When you serve an LLM with vLLM, the GPU is mostly busy storing **KV cache blocks** — chunks of 16 tokens' worth of attention keys and values from previous requests. The GPU has finite memory, so when a new request arrives and there are no free blocks, vLLM has to evict a block. The eviction policy is **LRU**: kick out whichever block was used the longest ago.

This sounds like a job for `collections.deque`. It isn't. The crucial operation isn't pop-from-front — it's **remove-an-arbitrary-block-from-the-middle**. Here's why: a free block sitting in the eviction queue might still be useful for *prefix caching*. If a new request shares its prefix with that cached block, vLLM wants to **revive** it — pull it out of the eviction queue and re-attach it to the new request. With a `deque`, locating and removing that block costs O(n). On a GPU running 1000 concurrent requests with millions of cached blocks, O(n) is a serving-stack heart attack.

The fix is an *intrusive* doubly-linked list: store the `prev`/`next` pointers directly on the `KVCacheBlock` object itself. Now any reference to a block is also a reference to its position in the queue — and removing it costs O(1). Plus the sentinel-node trick that follows is a small piece of engineering art.

## The code

`vllm-project/vllm` — [`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/97e4022c6ccb7b2cf1a1fc0a13a17a2a06d74f0d/vllm/v1/core/kv_cache_utils.py#L164-L327)

```python
class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code.
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real block
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1."""
        if (self.fake_free_list_head.next_free_block is self.fake_free_list_tail
                or self.fake_free_list_head.next_free_block is None):
            raise ValueError("No free blocks available")

        first_block = self.fake_free_list_head.next_free_block

        # Connect fake_head and the next block of first_block.
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None
        self.num_free_blocks -= 1
        return first_block

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1."""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase num_free_blocks by 1."""
        last_block = self.fake_free_list_tail.prev_free_block

        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
        self.num_free_blocks += 1
```

## What's happening

1. **"Intrusive" — `prev_free_block` and `next_free_block` live on `KVCacheBlock`, not on a wrapper node.** A regular `LinkedList` allocates a `Node(value, prev, next)` wrapper around every payload. To remove an arbitrary item you'd first need a dict mapping `item → Node`, which doubles memory and adds a hash lookup. Intrusive lists skip the wrapper: the payload object *is* the node. To remove a block you just access the block you already hold a reference to — no lookup, no allocation.

2. **The sentinel trick (`fake_free_list_head` / `fake_free_list_tail`).** Look at `append`: it unconditionally reads `self.fake_free_list_tail.prev_free_block` and writes through it. There's no `if list is empty` branch. Without sentinels, you'd need: `if last_block is None: self.head = block else: last_block.next = block`. Two cases, two paths, easy to get wrong. With sentinels, **every real block always has a valid `prev` and `next`** (in the worst case those pointers point at the sentinels themselves). One code path, no branches.

3. **No Python object allocation in the hot path.** The big docstring comment — "this class does not allocate any Python objects when manipulating the linked list" — is the headline performance claim. Compare to `deque.remove(x)`: deque is implemented in C and is fast for pop-from-front, but `remove` scans linearly and `appendleft`/`insert` allocate. The intrusive list does `block.prev.next = block.next` — four attribute assignments, zero `malloc`, zero ref-count churn beyond what's mandatory.

4. **`popleft` is `remove` of the first real block.** Look closely: `popleft` does almost exactly what `remove(first_block)` would do, just inlined and with the explicit empty-queue check. The duplication is intentional — it shaves a function call from the hottest path in the scheduler.

5. **Where the LRU semantics live.** The queue itself doesn't know about LRU. It just knows "first in is at the front, append goes to the back." The LRU policy is enforced by the *caller*: whenever a block is freed (because a request finished), the caller calls `append(block)` — pushing it to the back. So "front of the queue" naturally means "freed longest ago" = "best to evict next." The data structure is dumb and fast; the policy lives elsewhere.

## The analogy

Picture a **library returns shelf**. Patrons drop returned books at one end of the shelf (`append`); the librarian re-shelves books from the other end (`popleft`) when they have time. That's a deque — simple, fast.

Now picture this: while a book is sitting on the returns shelf, a patron walks in and says "wait, I want to check that one back out before you re-shelve it." With a regular shelf, the librarian has to scan the whole shelf to find the book. The intrusive linked list is **giving every book a Velcro strip** that sticks it to its current neighbors. When the patron grabs the book, the two neighbors snap together — the book peels right out, in O(1), no scanning. The sentinels are like permanent "bookends" at each end of the shelf: they're not real books, but they mean the librarian's grabbing arm never has to check "is this the edge of the shelf?" The bookends are always there to catch it.

## Try it yourself

Save as `try_intrusive_lru.py`:

```python
"""Strip-down version of vLLM's intrusive LRU queue (~25 lines)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Block:
    id: int
    prev: Optional["Block"] = field(default=None, repr=False)
    next: Optional["Block"] = field(default=None, repr=False)


class IntrusiveLRU:
    def __init__(self, blocks):
        self.head = Block(id=-1)  # sentinel
        self.tail = Block(id=-1)  # sentinel
        self.head.next, self.tail.prev = self.tail, self.head
        for b in blocks:
            self.append(b)

    def append(self, b):
        last = self.tail.prev
        last.next = b; b.prev = last
        b.next = self.tail; self.tail.prev = b

    def popleft(self):
        b = self.head.next
        if b is self.tail:
            raise ValueError("empty")
        self.head.next = b.next; b.next.prev = self.head
        b.prev = b.next = None
        return b

    def remove(self, b):  # O(1) middle removal — the whole point
        b.prev.next = b.next; b.next.prev = b.prev
        b.prev = b.next = None


blocks = [Block(i) for i in range(5)]
q = IntrusiveLRU(blocks)
print("evict order:", [q.popleft().id for _ in range(2)])  # 0, 1

q.append(blocks[0])   # block 0 freed → goes to back
q.remove(blocks[3])   # block 3 revived mid-queue (prefix cache hit)
q.append(blocks[3])   # then re-freed later

# Walk the queue from the head
cur, order = q.head.next, []
while cur is not q.tail:
    order.append(cur.id); cur = cur.next
print("queue now:", order)
```

Run with:
```bash
python try_intrusive_lru.py
```

Expected output:
```
evict order: [0, 1]
queue now: [2, 4, 0, 3]
```

Notice: block 3 was yanked from the *middle* of the queue and re-appended at the end — that's the prefix-cache-hit path, and it cost a constant four pointer writes.

## Where this pattern shows up elsewhere

- **Linux kernel `struct list_head`**: every linked structure in the kernel (processes, file descriptors, page-cache pages) embeds a `list_head` for intrusive linking. Same idea, in C.
- **Boost.Intrusive, `IntrusiveLinkedList<T>` in C++**: standard library equivalents for performance-critical C++.
- **LRU page cache in OS kernels**: every page frame has `lru` pointers; `mark_page_accessed()` removes it from one list and appends it to another, all O(1).
- **vLLM's [`BlockPool`](https://github.com/vllm-project/vllm/blob/97e4022c6ccb7b2cf1a1fc0a13a17a2a06d74f0d/vllm/v1/core/block_pool.py#L130) itself**: the consumer of this queue. When a request finishes, `free_blocks` walks the request's blocks in reverse and calls `append()`, encoding the LRU + "tail-of-chain-first" tiebreak.

## Caveats / when it breaks

- **Memory ownership is implicit.** A block must belong to *exactly one* list at a time. If you accidentally append the same block to two queues, the `prev`/`next` pointers get clobbered and you corrupt both lists silently. C++ Boost.Intrusive surfaces this via "hook" types; Python has nothing but discipline.
- **The four uses of `prev_free_block`/`next_free_block` are reserved for this queue.** If something else also wants to maintain a linked list of `KVCacheBlock` (say, a separate "dirty list"), it needs *its own pair* of pointer fields. Hence vLLM's deliberate naming.
- **No iteration protection.** Iterating the queue while another thread/coroutine mutates it gives you a half-stitched list. vLLM avoids this by running the scheduler single-threaded.
- **Python attribute writes are still slower than C pointer writes.** The technique closes the gap to `deque` but doesn't beat C. Rewriting this in Rust/C is a known follow-up for very large pools.

## Further reading

- vLLM PagedAttention paper, §3.2 (Block Manager): https://arxiv.org/abs/2309.06180
- vLLM v1 block pool source: [`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/97e4022c6ccb7b2cf1a1fc0a13a17a2a06d74f0d/vllm/v1/core/block_pool.py)
- Linux `list.h` (the C archetype for intrusive lists): https://github.com/torvalds/linux/blob/master/include/linux/list.h
- "Intrusive vs. non-intrusive containers" (Boost.Intrusive docs): https://www.boost.org/doc/libs/release/doc/html/intrusive/intrusive_vs_nontrusive.html
