---
date: 2026-09-30
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/tk-encode/src/models/bpe/merge_hot_cold_queue.rs
permalink: https://github.com/huggingface/tokenizers/blob/bbccb0513ff9afda385ca5c85c66eddb1318cfc7/tokenizers/tk-encode/src/models/bpe/merge_hot_cold_queue.rs#L34-L146
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, bpe]
---

# tokenizers BPE hot/cold queue：初始 pair 排一次，新 pair 进小堆 / tokenizers BPE Hot/Cold Queue: Sort Initial Pairs Once, Heap Only New Pairs

> **一句话 / In one line**: BPE merge 不必每步重扫整词；初始 pair 用冷队列顺序读，新生成 pair 才进热堆。 / BPE merging does not need to rescan the whole word each step: initial pairs are read from a cold sorted queue, while newly created pairs go into a hot heap.

## 为什么重要 / Why this matters

Tokenizer 的吞吐常常卡在大量短字符串和长词的重复 merge 上。这个实现把“已经知道的候选”和“merge 后才出现的候选”分层处理，少做堆操作，仍保持 BPE 的按 rank、同 rank 取左侧的规则。

Tokenizer throughput often depends on repeated merges over many short strings and occasional long words. This implementation separates candidates known at the start from candidates created by merges, reducing heap work while preserving BPE rank and left-to-right tie-breaking.

## 代码 / The code

`huggingface/tokenizers` — [`tokenizers/tk-encode/src/models/bpe/merge_hot_cold_queue.rs`](https://github.com/huggingface/tokenizers/blob/bbccb0513ff9afda385ca5c85c66eddb1318cfc7/tokenizers/tk-encode/src/models/bpe/merge_hot_cold_queue.rs#L34-L146)

```rust
/// Merges the word whose pair entries and queue keys `scratch` holds (filled by `convert_queue`),
/// writing the merged word into `symbols` as internal ids.
pub fn merge_with_queue(tables: &BpeTables, symbols: &mut Vec<u32>, scratch: &mut QueueScratch) {
    let QueueScratch { entries, queue } = scratch;
    if entries.is_empty() {
        return;
    }
    queue.sort_cold();

    // Entry 0 is the leftmost pair, and only merging the leftmost pair moves it.
    let mut leftmost = 0u32;
    // The whole word as one symbol, read back only when `leftmost` ends up `NONE`, which takes the
    // leftmost pair to have been merged.
    let mut collapsed_symbol = 0u32;

    while let Some(key) = queue.pop() {
        let index = key as u32 as usize;
        let entry = entries[index];
        // The entry was rewritten after this key was queued, so the key is stale.
        if entry.rank as u64 != key >> 32 {
            continue;
        }
        entries[index].rank = DEAD_RANK;
        if entry.left_pair_index == NONE {
            leftmost = entry.right_pair_index;
            collapsed_symbol = entry.merged_symbol;
        }
        entry.rewrite_neighbours(tables, entries, queue);
    }

    symbols.clear();
    if leftmost == NONE {
        symbols.push(collapsed_symbol);
        return;
    }
    let mut index = leftmost as usize;
    symbols.push(entries[index].left_symbol);
    loop {
        symbols.push(entries[index].right_symbol);
        match entries[index].right_pair_index {
            NONE => break,
            next => index = next as usize,
        }
    }
}

/// The buffers one word's merge needs, reused from word to word so that merging allocates nothing.
#[derive(Default)]
pub struct QueueScratch {
    /// One [`Entry`] per adjacent pair of the word, linked left/right.
    pub(crate) entries: Vec<Entry>,
    pub(crate) queue: MergeQueue,
}

/// The two-tier priority queue over pair keys: a sorted vector for the pairs the word starts with,
/// a binary min-heap for the pairs merges create. The module docs say why.
#[derive(Default)]
pub(crate) struct MergeQueue {
    /// Keys of the pairs the word starts with, in the order [`MergeQueue::sort_cold`] put them.
    cold: Vec<u64>,
    /// Keys of the pairs merges create, as a binary min-heap.
    hot: Vec<u64>,
    /// How far into `cold` [`MergeQueue::pop`] has walked.
    cold_cursor: usize,
}

impl MergeQueue {
    /// Empties both tiers for a new word, leaving room for `pairs` cold keys.
    pub(super) fn clear(&mut self, pairs: usize) {
        self.cold.clear();
        self.cold.reserve(pairs);
        self.hot.clear();
        self.cold_cursor = 0;
    }

    /// Queues one of the pairs the word starts with. Every cold key is pushed before the first
    /// [`MergeQueue::pop`], which is what lets one sort replace a heap.
    #[inline(always)]
    pub(super) fn push_cold(&mut self, key: u64) {
        self.cold.push(key);
    }

    /// Queues a pair a merge created.
    #[inline(always)]
    fn push(&mut self, key: u64) {
        heappush(&mut self.hot, key);
    }

    /// Sorts the cold tier, so that [`MergeQueue::pop`] can read it front to back. Called once,
    /// after the conversion has pushed every cold key.
    fn sort_cold(&mut self) {
        self.cold.sort_unstable();
    }

    /// The lowest key of the two tiers, `None` once both are drained.
    #[inline(always)]
    fn pop(&mut self) -> Option<u64> {
        let cold_key = self
            .cold
            .get(self.cold_cursor)
            .copied()
            .unwrap_or(EMPTY_KEY);
        let hot_key = self.hot.first().copied().unwrap_or(EMPTY_KEY);
        if cold_key <= hot_key {
            if cold_key == EMPTY_KEY {
                return None;
            }
            self.cold_cursor += 1;
            Some(cold_key)
        } else {
            Some(heappop(&mut self.hot))
        }
    }
```

## 逐行讲解 / What's happening

1. **第 34-48 行 / Lines 34-48**:
   - 中文: `merge_with_queue` 接收可复用 scratch；`entries` 是相邻 pair 的 arena，`queue` 是候选 merge 的优先级结构。
   - English: `merge_with_queue` receives reusable scratch space. `entries` is the arena of adjacent pairs, and `queue` is the priority structure.
1. **第 49-62 行 / Lines 49-62**:
   - 中文: 每次取最小 key。若 entry 的 rank 已经不等于 key 的高位，说明这是旧 key，直接跳过。
   - English: Each step pops the lowest key. If the entry rank no longer matches the high bits, the key is stale and skipped.
1. **第 64-78 行 / Lines 64-78**:
   - 中文: merge 完后不保存整棵树，而是沿 doubly linked pair 把最终 symbol 序列读出来。
   - English: After merging, the final symbol sequence is read by walking the doubly linked pair arena.
1. **第 88-146 行 / Lines 88-146**:
   - 中文: `cold` 是初始 pair 排序后的数组，`hot` 是 merge 过程中新 pair 的小堆；`pop` 比较两个队头，谁 rank 小取谁。
   - English: `cold` is a sorted vector of initial pairs; `hot` is a min-heap for newly created pairs. `pop` compares the two front keys and returns the lower one.

## 类比 / The analogy

像厨房备菜：开工前已经知道的一篮子订单先按时间排好；现场临时加的菜才放进“急单”小架子。厨师每次看两边谁更急，不需要把所有订单反复重排。

It is like a kitchen. Orders known before service are sorted once; orders created during service go into a small rush rack. The cook only compares the next normal order with the next rush order.

## 自己跑一遍 / Try it yourself

```python
import heapq
cold = [(1, 'ab'), (4, 'bc'), (9, 'cd')]
hot = []
cold_i = 0
while cold_i < len(cold) or hot:
    cold_key = cold[cold_i] if cold_i < len(cold) else (99, None)
    hot_key = hot[0] if hot else (99, None)
    if cold_key <= hot_key:
        key, pair = cold_key; cold_i += 1
    else:
        key, pair = heapq.heappop(hot)
    print(pair)
    if pair == 'ab': heapq.heappush(hot, (2, 'xb'))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
ab
xb
bc
cd
```

`xb` 是 merge 后才出现的候选，所以它进入 hot heap，并在 rank 2 时插到原始 rank 4 前面。

`xb` appears only after a merge, so it enters the hot heap and correctly runs before the original rank-4 pair.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Dijkstra / A\*** / **Dijkstra / A\***: 已确定的边和新松弛出来的候选分开维护，避免全图重排。 / Confirmed edges and newly relaxed candidates are maintained without resorting the whole graph.
- **LLM serving scheduler** / **LLM serving scheduler**: 已排队请求和新 decode token 常用不同队列，再按优先级合并。 / Queued requests and newly generated decode work are often maintained separately and merged by priority.

## 注意事项 / Caveats / when it breaks

- **stale key 必须可识别** / **Stale keys must be detectable**: 如果没有 rank 校验，旧 key 会把已经改写过的 pair 再 merge 一次。 / Without the rank check, an old key could merge a pair that has already changed.
- **tie-break 要稳定** / **Tie-breaking must be stable**: BPE 同 rank 通常按左侧优先；key 里带 entry index 就是在编码这个约定。 / BPE usually resolves equal ranks by the leftmost pair; packing the entry index into the key preserves that rule.

## 延伸阅读 / Further reading

- [Hugging Face tokenizers](https://github.com/huggingface/tokenizers)
- [Byte Pair Encoding overview](https://en.wikipedia.org/wiki/Byte_pair_encoding)
