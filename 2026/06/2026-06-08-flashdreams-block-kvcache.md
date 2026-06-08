---
date: 2026-06-08
topic: diffusion
source: trending
repo: NVIDIA/flashdreams
file: flashdreams/core/attention/kvcache.py
permalink: https://github.com/NVIDIA/flashdreams/blob/bb5fd91484bf422173072c22e7273fbd0cdb29dd/flashdreams/flashdreams/core/attention/kvcache.py#L25-L130
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, diffusion, kv-cache, streaming, cuda-graph]
---

# flashdreams 的 BlockKVCache:[sink | rolling window] 用 4 步协议讲清楚 / flashdreams's BlockKVCache: [sink | rolling window] explained as a 4-step protocol

> **一句话 / In one line**: 把 StreamingLLM 的"sink tokens + 滚动局部窗口"思路拆成 `before_update → update → cached_k → after_update` 四步,刚好对得上一个 CUDA graph 边界——这就是工业级互动视频/世界模型 KV cache 的样子。
> Take StreamingLLM's "sink + rolling window" idea and slice it into a four-step protocol `before_update → update → cached_k → after_update` that aligns perfectly with a CUDA-graph boundary — this is what an industrial interactive video / world-model KV cache looks like.

## 为什么重要 / Why this matters

NVIDIA 在 2026 年 6 月开源的 `flashdreams` 是个新东西——专门服务"互动式自回归视频/世界模型"的高性能推理库。它的目标场景:你拿着手柄玩一个 AI 生成的游戏世界,每按一次按键模型就要新生成一段视频帧,延迟必须低于 100ms。这就要求 KV cache:(1) 不能无限增长——长度封顶滚动;(2) 不能丢前几帧——前几帧是 "anchor" 用的;(3) 全部要在 CUDA graph 里跑——不能有 host-side 控制流。BlockKVCache 把 StreamingLLM 的"sink + window"做法包成一个 `@dataclass`,给 forward 设计了 4 步显式协议,每一步都是数据流上的纯函数 / 副作用,刚好能 capture 成 CUDA graph。这种"协议化"是工业代码和论文代码的根本区别。

NVIDIA's `flashdreams` (open-sourced June 2026) is fresh — a high-performance inference library specifically for "interactive autoregressive video / world models." The use case: you're holding a controller and playing an AI-generated game world; every controller input triggers a fresh batch of video frames, and the round-trip must stay under 100ms. That forces the KV cache to (1) not grow unboundedly — bound the length with rolling eviction; (2) not lose the first few frames — they're the "anchor"; (3) live entirely inside CUDA graphs — no host-side control flow allowed. BlockKVCache wraps StreamingLLM's "sink + window" recipe into a `@dataclass` and exposes a 4-step protocol such that each step is either a pure function or a clean side effect — perfectly capturable by a CUDA graph. This "protocolization" is the structural difference between research and production code.

## 代码 / The code

`NVIDIA/flashdreams` — [`flashdreams/core/attention/kvcache.py`](https://github.com/NVIDIA/flashdreams/blob/bb5fd91484bf422173072c22e7273fbd0cdb29dd/flashdreams/flashdreams/core/attention/kvcache.py#L25-L130)

```python
@dataclass
class BlockKVCache:
    """
    KV cache for causal attention with a fixed-size local window, CUDA-graph compatible.

    Layout along the rolling dim: [sink tokens | local window tokens].
    Sink tokens are never evicted; the local window rolls left as new chunks
    are added if full. Chunks are non-overlapping.

    Phases:
        - Filling: cache not yet full; tokens are written contiguously.
        - Steady-state: cache full; each new chunk triggers a left-roll of the
          local window and overwrites the rightmost positions.

    Per-step usage:
        1. before_update(chunk_idx) — prepare (roll local window if steady-state).
        2. update(k, v) — write the new chunk's keys/values into the cache.
        3. cached_k() / cached_v() — get cached keys/values for attention.
        4. after_update(chunk_idx) — update internal bookkeeping.
    """

    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    seq_dim: int
    chunk_size: int
    window_size: int
    sink_size: int = 0
    device: torch.device | str = torch.device("cuda")
    dtype: torch.dtype = torch.float16

    _prev_chunk_idx: int = -1
    _curr_chunk_idx: int | None = None
    _n_cached: int = 0

    _k: Tensor = field(init=False)
    _v: Tensor = field(init=False)

    @property
    def size(self) -> int:
        if self._curr_chunk_idx is None:
            return self._n_cached
        return self._visible_end()

    @property
    def write_end(self) -> int:
        assert self._curr_chunk_idx is not None
        return self.size

    @classmethod
    def from_tensor(cls, k: Tensor, v: Tensor, seq_dim: int) -> Self:
        """Build a single-chunk cache pre-filled with the given key and value tensors."""
        cache = cls(
            k_shape=k.shape, v_shape=v.shape, seq_dim=seq_dim,
            chunk_size=k.shape[seq_dim], window_size=k.shape[seq_dim],
            device=k.device, dtype=k.dtype,
        )
        cache.before_update(0)
        cache.update(k, v)
        cache.after_update(0)
        cache._curr_chunk_idx = 0
        return cache

    def __post_init__(self) -> None:
        assert self.k_shape[:-1] == self.v_shape[:-1]
        tensor_dim = len(self.k_shape)
        assert -tensor_dim <= self.seq_dim < tensor_dim
        # Normalize seq_dim to a non-negative index so downstream indexing math
        # doesn't have to special-case negatives.
        self.seq_dim = self.seq_dim if self.seq_dim >= 0 else self.seq_dim + tensor_dim
        assert self.sink_size >= 0
        expected_length = self.sink_size + self.window_size
        assert self.k_shape[self.seq_dim] == expected_length
        assert (self.window_size + self.sink_size) % self.chunk_size == 0

        self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
        self._v = torch.empty(self.v_shape, device=self.device, dtype=self.dtype)
```

## 逐行讲解 / What's happening

1. **`@dataclass` + buffer 分离设计 / Lines 25-95**:
   - 中文: 整个 cache 的 schema 用 `@dataclass` 声明,**所有可配置的项**(k_shape, seq_dim, chunk_size, window_size, sink_size, device, dtype)都是 dataclass 字段——可以一行 `BlockKVCache(...)` 构造,IDE 自动补全。`_k` / `_v` 用 `field(init=False)` 标记"不参与构造、`__post_init__` 里再 allocate"。这是个很干净的"配置 = 数据 + 缓冲 = 张量"分离。
   - English: The whole cache schema is declared as a `@dataclass`, **all configurable items** (k_shape, seq_dim, chunk_size, window_size, sink_size, device, dtype) become dataclass fields — instantiate in one line, IDE autocompletes everything. `_k` / `_v` are tagged `field(init=False)` meaning "not part of the constructor, allocated in `__post_init__`." A clean separation: config = data fields, buffers = tensors allocated later.

2. **`Layout: [sink tokens | local window tokens]`(docstring 行 31-33) / Lines 31-33**:
   - 中文: 这是 KV cache 内存布局的一句话总结。前 `sink_size` 个位置永远不动(StreamingLLM 论文的核心发现:模型对前几个 token 极度依赖,evict 它们会立刻崩),后面 `window_size` 是滚动局部窗口,塞满了就左移、把最老的踢出去。一个固定 size 的 buffer,但中间一段在物理位置上"流动"。
   - English: One-sentence summary of the memory layout. The first `sink_size` positions never move (StreamingLLM's core finding: the model leans heavily on the first few tokens; evicting them causes immediate collapse). The trailing `window_size` is a rolling local window; once it fills, it shifts left and drops the oldest tokens. A fixed-size buffer where the middle section "flows" through physical positions.

3. **`Phases: Filling vs Steady-state`(docstring) / Phase semantics**:
   - 中文: 这是 cache 的两种状态,**所有 dispatch 逻辑都基于这个判断**。Filling phase:`_n_cached < window_size + sink_size`,还没塞满,新 chunk 直接 append。Steady-state:塞满了,每次 update 前先把局部窗口左滚 chunk_size 位(`_roll_local_window_left` 在本片段外但 docstring 提及),腾出右边写新 chunk。State 的变化由 `is_steady_state()` 一个方法判定——没有隐式 if 散落各处。
   - English: The two cache states, **every dispatch decision keys off this distinction**. Filling phase: `_n_cached < window_size + sink_size`, not full yet, new chunks just append. Steady-state: full; each update first rolls the local window left by `chunk_size` (`_roll_local_window_left`, mentioned in the docstring but outside this snippet) to make room on the right for the new chunk. State transitions are gated by a single `is_steady_state()` method — no `if not full` checks scattered around.

4. **4 步协议(docstring 行 39-43) / 4-step protocol**:
   - 中文: `before_update(chunk_idx)` → `update(k, v)` → `cached_k() / cached_v()` → `after_update(chunk_idx)`。这是工业代码独特的招数——把一个"状态机有副作用"的对象分成 4 个原子操作,各自只负责一件事:
     - `before_update`: 决定要不要 roll、把 `_curr_chunk_idx` 标位。无副作用张量计算。
     - `update`: 纯写入,所有 index 都从 `_curr_chunk_idx` 派生。
     - `cached_k/v`: 纯读取(返回 view)。
     - `after_update`: 更新 `_prev_chunk_idx` 和 `_n_cached`。
     这种切分的妙处是:整个 attention forward 可以包成一个 CUDA graph(只 capture `update + cached_k + attention kernel`),而控制流(before/after_update 里的 Python 决策)留在 host。
   - English: `before_update(chunk_idx)` → `update(k, v)` → `cached_k() / cached_v()` → `after_update(chunk_idx)`. A distinctly industrial move — chop a stateful object with side effects into four atomic operations, each with one job:
     - `before_update`: decide whether to roll, stamp `_curr_chunk_idx`. No tensor side effects.
     - `update`: pure writes, all indices derived from `_curr_chunk_idx`.
     - `cached_k/v`: pure reads (return views).
     - `after_update`: update `_prev_chunk_idx` and `_n_cached`.
     The win of this split: the attention forward can be wrapped as a CUDA graph (capture `update + cached_k + attention kernel`) while the control flow (Python decisions in before/after_update) stays on host.

5. **`seq_dim` 标准化为 non-negative(行 73-74) / Seq-dim normalization**:
   - 中文: 用户可以传 `seq_dim=-2`(Python 风格的"倒数第二个维度"),但内部所有索引数学都基于正整数。`__post_init__` 立刻规范化:`self.seq_dim = self.seq_dim if >= 0 else self.seq_dim + tensor_dim`。一条小细节,但是它免去了后面所有 `_seq_slice` / `_roll_local_window_left` 里的 negative-axis special-case 分支。"在边界处规范化,内部代码就干净了"是个值得抄的设计哲学。
   - English: Users can pass `seq_dim=-2` (Pythonic "second-to-last"), but all internal indexing math is in positive integers. `__post_init__` normalizes immediately: `self.seq_dim = self.seq_dim if >= 0 else self.seq_dim + tensor_dim`. A small detail, but it removes the negative-axis special case from every downstream method (`_seq_slice`, `_roll_local_window_left`). "Normalize at the boundary, the interior stays clean" is a design philosophy worth stealing.

6. **`assert (window_size + sink_size) % chunk_size == 0`(行 79) / Lines 79**:
   - 中文: 这个 assert 是 CUDA-graph 兼容性的关键。如果 buffer size 不是 chunk_size 的整数倍,rolling 时会出现"半个 chunk 在头、半个在尾"的边界情况,需要 host-side 分支处理——graph 就不能纯静态了。所以作者直接 ban 掉这个情况:你必须 pad 配置让它整除。简单粗暴,但确保 graph 可 capture。
   - English: This assert is the key to CUDA-graph compatibility. If the buffer size isn't a multiple of chunk_size, rolling produces "half a chunk at the head, half at the tail" edge cases that need host-side branches — and the graph stops being purely static. The author bans the case outright: you must pad your config to divide. Crude but ensures captureability.

7. **`from_tensor` classmethod(行 50-60) / Lines 50-60**:
   - 中文: 一个 builder convenience——给定 k/v tensor,造一个 "已经填好这一个 chunk" 的 cache。这是 prefill 阶段的入口:你算完 prompt 的 KV 后,一行就能转成 cache 对象,接下去 generation 步走 update/cached/after_update 循环。注意它内部调用了 4 步协议——这就是用 `from_tensor` 测试 protocol 一致性的好处。
   - English: A builder convenience — given k/v tensors, create a cache "pre-filled with this one chunk." This is the prefill entry point: after computing the prompt KV, one line converts it to a cache object, and generation runs the update/cached/after_update loop from there. Note it invokes the full 4-step protocol internally — which is also how the protocol's self-consistency is tested.

## 类比 / The analogy

中文: 想象一家寿司店,师傅面前是一条传送带——但传送带是"环形 + 一段固定"的奇怪设计:**左边头 4 个位置固定放招牌"金枪鱼"**(sink tokens,客户人均必吃),不能动;右边一长串位置是普通寿司(local window),做新的放右端,挤满了就把最左的旧寿司(除了 4 个金枪鱼之外)扔掉。客户来了点单(attention query),师傅一眼就能看到"4 个金枪鱼 + 当前传送带上所有普通寿司"——也就是 sink + window。师傅的工作流也是严格 4 步:

1. **before_update**:"先看传送带满没满,满了就左滚一格腾位"
2. **update**: 把新做好的寿司放到右端
3. **客户看着挑(cached_k)**: 拍照、决定要哪几个 (attention)
4. **after_update**: 师傅心里记一下"我已经做到第 N 盘了"

这 4 步的顺序不能换。换了就要么写空、要么读旧。

English: Picture a sushi bar with a strange "ring + fixed segment" conveyor belt: **the first 4 positions on the left are permanently reserved for signature tuna nigiri** (sink tokens — every customer must have these) and never move; the long stretch to the right holds regular sushi (local window), new pieces go to the right end, and when full the oldest non-tuna pieces are dropped from the left of the window. A customer placing an order (the attention query) glances over and sees "the 4 tunas + whatever regular pieces are currently on the belt" — sink + window. The chef's workflow is also strictly 4 steps:

1. **before_update**: "Check if the belt is full; if so, roll left to make room."
2. **update**: Place the freshly-made piece at the right end.
3. **Customer browses (`cached_k`)**: photograph the belt, decide what to pick (attention).
4. **after_update**: The chef mentally notes "I've now made piece N."

These four steps can't be reordered — reorder them and you either write into empty space or attend to stale state.

## 自己跑一遍 / Try it yourself

```python
# block_kvcache_demo.py
import torch
from dataclasses import dataclass, field

@dataclass
class TinyBlockKVCache:
    seq_dim: int
    chunk_size: int
    window_size: int
    sink_size: int = 0
    _prev_chunk_idx: int = -1
    _n_cached: int = 0
    _k: torch.Tensor = field(init=False)

    def alloc(self, shape):
        total = self.sink_size + self.window_size
        full_shape = list(shape)
        full_shape[self.seq_dim] = total
        self._k = torch.zeros(full_shape)

    def before_update(self, chunk_idx):
        self._curr = chunk_idx
        total = self._k.shape[self.seq_dim]
        if chunk_idx > self._prev_chunk_idx and self._n_cached == total:
            # roll window left
            keep = self.window_size - self.chunk_size
            self._k.narrow(self.seq_dim, self.sink_size, keep).copy_(
                self._k.narrow(self.seq_dim, self.sink_size + self.chunk_size, keep).clone()
            )

    def update(self, k):
        total = self._k.shape[self.seq_dim]
        write_start = min(self._n_cached, total - self.chunk_size)
        self._k.narrow(self.seq_dim, write_start, self.chunk_size).copy_(k)

    def cached_k(self):
        total = self._k.shape[self.seq_dim]
        end = min(self._n_cached + self.chunk_size, total)
        return self._k.narrow(self.seq_dim, 0, end)

    def after_update(self, chunk_idx):
        total = self._k.shape[self.seq_dim]
        if chunk_idx > self._prev_chunk_idx and self._n_cached < total:
            self._n_cached += self.chunk_size
        self._prev_chunk_idx = chunk_idx

c = TinyBlockKVCache(seq_dim=1, chunk_size=2, window_size=6, sink_size=2)
c.alloc(shape=(1, 8, 4))   # total = sink(2) + window(6) = 8

for i in range(6):
    chunk = torch.full((1, 2, 4), float(i + 1))   # chunk value = i+1
    c.before_update(i)
    c.update(chunk)
    after = c.cached_k().squeeze(-1)[0, :, 0].tolist()
    c.after_update(i)
    print(f"chunk {i+1} -> cached: {after}")
```

运行 / Run with:
```bash
pip install torch
python block_kvcache_demo.py
```

预期输出 / Expected output:
```
chunk 1 -> cached: [1.0, 1.0]
chunk 2 -> cached: [1.0, 1.0, 2.0, 2.0]
chunk 3 -> cached: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
chunk 4 -> cached: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
chunk 5 -> cached: [1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
chunk 6 -> cached: [1.0, 1.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
```

中文一两句: 注意 chunk 5 之后:**1.0(sink)留下了,2.0 被踢了**——这就是 sink + rolling window 的核心。如果你把 sink_size 改成 0,你会看到 1.0 也被踢出去——退化成纯滑动窗口。

English: After chunk 5: **1.0 (sink) is preserved, 2.0 is evicted** — that's the heart of sink + rolling window. Set sink_size=0 and you'll see 1.0 also get evicted — it degenerates to a plain sliding window.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **StreamingLLM (Xiao et al. 2023)** / **StreamingLLM (Xiao et al. 2023)**: 思想的源头,论文展示了"前几个 attention sink token + 滑动窗口"为什么 work。/ The origin paper that empirically showed why "first few sink tokens + sliding window" works.
- **`NVIDIA/kvpress`** / **`NVIDIA/kvpress`**: 5/26 笔记里的 "StreamingLLM in 30 lines",研究向的实现。flashdreams 这版是工业化重写。/ Covered on 5/26 as "StreamingLLM in 30 lines" — research-grade implementation. flashdreams is the industrial rewrite.
- **vLLM `SlidingWindowAttention`** / **vLLM `SlidingWindowAttention`**: 同思想,但没有 sink token 概念,且数据结构跟 paged attention 紧耦合。/ Same intuition without the sink concept, and the data structure is tightly coupled to paged attention.
- **Mistral / Gemma 的 sliding window attention** / **Mistral / Gemma sliding window attention**: 训练时就用,推理时 cache 自然是 windowed 的。但训练 + 推理一致没 sink。/ Baked into training, so the cache is naturally windowed at inference. But neither training nor inference uses sinks.
- **`huggingface/diffusers/hooks/text_kv_cache.py`** / **`huggingface/diffusers/hooks/text_kv_cache.py`**: diffusion 模型在长 prompt 上做类似的"keep first N tokens + roll"。同样的 idea 跨域应用。/ Diffusion models do "keep first N tokens + roll" for long prompts. Same idea ported to another domain.

## 注意事项 / Caveats / when it breaks

- **`chunk_size` 必须整除 `window_size + sink_size` / `chunk_size` must evenly divide `window_size + sink_size`**: `__post_init__` 里有 assert。为了 CUDA graph 兼容性。如果用户配置不整除,你必须 pad 到整除——别去掉 assert,你会得到 silent 数值错误。/ Enforced in `__post_init__` — required for CUDA-graph compatibility. If a user config doesn't divide, pad it; don't remove the assert or you get silent numerical bugs.
- **`before_update` 必须在 `update` 之前 / `before_update` must come before `update`**: 否则 steady-state 时新数据写到还没腾出来的位置。文件内多处用 `_curr_chunk_idx is None` assert 保护这个不变量,但 release build 里 assert 是 noop。/ Otherwise in steady state you write to a position the roll hasn't vacated. The file uses `_curr_chunk_idx is None` assertions to guard the invariant, but `python -O` will strip them.
- **`_roll_local_window_left` 会 clone / `_roll_local_window_left` clones**: 滚动用的是 `self._k[src] = self._k[dst].clone()`,在 GPU 上是一个 D2D copy。如果你的 buffer 极大、roll 在 hot path 上,这是性能瓶颈。优化路径:用环形 buffer + 索引偏移代替物理 copy(但 CUDA graph capture 会更复杂)。/ The roll uses `self._k[src] = self._k[dst].clone()`, which is a D2D copy on the GPU. With a huge buffer in the hot path this is a bottleneck. Optimization: a ring buffer with an index offset instead of physical copy (but CUDA-graph capture gets more involved).
- **CUDA-graph compatibility ≠ CUDA-graph capture / "Compatible" ≠ "captured"**: 这段代码本身**没**有 `torch.cuda.graph(...)` 调用。文档里说 "CUDA-graph compatible" 意思是"你可以把它放进 graph",但实际 capture 在 `flashdreams/recipes/wan/pipeline.py` 之类的高层完成。读这段时不要以为它就是 graph。/ The snippet itself does **not** call `torch.cuda.graph(...)`. The docstring's "CUDA-graph compatible" means "you can put it in a graph"; the actual capture happens higher up in files like `flashdreams/recipes/wan/pipeline.py`. Don't mistake this snippet for the graph itself.

## 延伸阅读 / Further reading

- [StreamingLLM paper — Xiao et al. 2023](https://arxiv.org/abs/2309.17453)
- [NVIDIA flashdreams repo](https://github.com/NVIDIA/flashdreams)
- [PyTorch CUDA graphs deep dive](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [NVIDIA/kvpress StreamingLLM implementation (covered 5/26 in this archive)](https://github.com/NVIDIA/kvpress)
