---
date: 2026-07-13
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/src/models/bpe/trainer.rs
permalink: https://github.com/huggingface/tokenizers/blob/05741122105911e14dbe307c8f640c757fa97779/tokenizers/src/models/bpe/trainer.rs#L433-L530
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers, bpe]
---

# tokenizers BPE trainer：用堆维护下一次最值得合并的 pair / tokenizers BPE Trainer: Use a Heap for the Next Best Pair

> **一句话 / In one line**: BPE 训练先统计 pair 频次，把候选合并放进堆里，每次弹出当前最高频 pair，再校验它是否仍然有效。 / BPE training counts pair frequencies, stores merge candidates in a heap, pops the most frequent pair, then verifies that the count is still current.

## 为什么重要 / Why this matters

BPE 的朴素写法是“每合并一次就全量重扫语料”。`tokenizers` 的实现更像增量调度器：先建 `pair_counts` 和 `where_to_update`，再用 heap 只关注最高频候选。候选可能因为之前的合并变旧，所以弹出时还要和最新计数对账。

A naive BPE trainer rescans the whole corpus after every merge. `tokenizers` behaves more like an incremental scheduler: it builds `pair_counts` and `where_to_update`, then keeps the hottest candidates in a heap. A candidate can become stale after earlier merges, so the popped item is checked against the latest count.

## 代码 / The code

`huggingface/tokenizers` — [`tokenizers/src/models/bpe/trainer.rs`](https://github.com/huggingface/tokenizers/blob/05741122105911e14dbe307c8f640c757fa97779/tokenizers/src/models/bpe/trainer.rs#L433-L530)

```rust
pub fn do_train(&self, word_counts: &AHashMap<CompactString, u64>, model: &mut BPE) -> Result<Vec<AddedToken>> {
    let mut word_to_id: AHashMap<CompactString, u32> = AHashMap::with_capacity(self.vocab_size);
    let mut id_to_word: Vec<CompactString> = Vec::with_capacity(self.vocab_size);

    self.add_special_tokens(&mut word_to_id, &mut id_to_word);
    self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);

    let (mut words, counts) =
        self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);

    let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
    let mut queue = OctonaryHeap::with_capacity(pair_counts.len());
    where_to_update.drain().for_each(|(pair, pos)| {
        let count = pair_counts[&pair];
        if count > 0 {
            queue.push(Merge { pair, count: count as u64, pos });
        }
    });

    let mut merges: Vec<(Pair, u32)> = vec![];
    loop {
        if word_to_id.len() >= self.vocab_size {
            break;
        }
        let Some(mut top) = queue.pop() else {
            break;
        };
        if top.count != pair_counts[&top.pair] as u64 {
            top.count = pair_counts[&top.pair] as u64;
            queue.push(top);
            continue;
        }
        if top.count < 1 || self.min_frequency > top.count {
            break;
        }
        let part_a = &id_to_word[top.pair.0 as usize];
        let mut part_b = id_to_word[top.pair.1 as usize].as_str();
```

## 逐行讲解 / What's happening

1. **先放特殊 token 和字母表 / Seed with special tokens and alphabet**:
   - 中文: vocab 不是从空开始，特殊 token 和基础字符必须先占住 ID。
   - English: The vocabulary does not start empty; special tokens and base characters reserve IDs first.
2. **`count_pairs` 建索引 / `count_pairs` builds an index**:
   - 中文: 它不只数频次，还记录每个 pair 出现在哪些 word 位置，后续更新才不用全量扫。
   - English: It records not only frequency but also the word positions where each pair occurs, enabling localized updates.
3. **heap 里可能有旧候选 / The heap can contain stale candidates**:
   - 中文: `top.count != pair_counts[...]` 时，说明这个候选的频次已经变了，更新后重新入堆。
   - English: If `top.count` differs from `pair_counts[...]`, the candidate is stale; update it and push it back.
4. **频次阈值负责停止 / Frequency threshold stops training**:
   - 中文: 当最强候选都低于 `min_frequency`，继续合并只会制造噪声 token。
   - English: When the best remaining candidate is below `min_frequency`, further merges mostly create noisy tokens.

## 类比 / The analogy

像餐厅后厨的叫号屏：最急的菜排最上面，但如果某桌已经改菜，厨师拿到单子时要先核对当前订单。过期单不做，改完再排队。

Think of a kitchen order screen: the most urgent dish appears first, but if the table changed its order, the cook must verify the live ticket before cooking. Stale tickets are updated and requeued.

## 自己跑一遍 / Try it yourself

```python
import heapq

pair_counts = {("l", "o"): 3, ("o", "w"): 2}
heap = [(-v, k) for k, v in pair_counts.items()]
heapq.heapify(heap)

pair_counts[("l", "o")] = 1
neg_count, pair = heapq.heappop(heap)
if -neg_count != pair_counts[pair]:
    heapq.heappush(heap, (-pair_counts[pair], pair))
print(heapq.heappop(heap))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
(-2, ('o', 'w'))
```

虽然 `("l", "o")` 曾经最高频，但弹出时发现它已经过期，所以真正执行的是当前最高频候选。

Although `("l", "o")` used to be the top candidate, the pop-time check catches that it is stale, so the current best candidate wins.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Dijkstra / A\*** / **Dijkstra / A\***: 常把旧距离留在优先队列里，弹出时再判断是否过期。 / They often leave stale distances in the priority queue and discard them when popped.
- **调度系统** / **Schedulers**: 延迟校验比每次更新都删除旧节点更简单。 / Lazy validation is often simpler than deleting stale nodes on every update.

## 注意事项 / Caveats / when it breaks

- **heap 不是事实源 / the heap is not the source of truth**: 最新频次在 `pair_counts`，heap 只是候选队列。 / The latest count lives in `pair_counts`; the heap is only a candidate queue.
- **Rust 代码不是 Python API** / **the Rust code is not the Python API**: Python 用户看到的是 `tokenizers` 绑定后的 trainer，但性能关键在这里。 / Python users see bound trainer classes, but the performance-sensitive path lives here.

## 延伸阅读 / Further reading

- tokenizers BPE trainer — https://github.com/huggingface/tokenizers/blob/05741122105911e14dbe307c8f640c757fa97779/tokenizers/src/models/bpe/trainer.rs
