---
date: 2026-09-09
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/data_loader.py
permalink: https://github.com/huggingface/accelerate/blob/0f7e35fe7902cc6ba8dc01b146d6447900464b88/src/accelerate/data_loader.py#L577-L611
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, accelerate, dataloader, checkpointing]
---

# Accelerate DataLoaderShard：提前看一批，结尾才不会错 / Accelerate DataLoaderShard: Look One Batch Ahead to Get the End Right

> **一句话 / In one line**: `DataLoaderShard.__iter__` 先拿住当前 batch，再预取下一批，以便正确标记最后一批并保存可恢复状态。 / `DataLoaderShard.__iter__` holds the current batch while looking ahead to the next one, which keeps final-batch handling and resumable state correct.

## 为什么重要 / Why this matters

中文：分布式训练的数据加载器不只负责 `yield`。它还要把 batch 搬到设备、维护 checkpoint 状态、支持跳过已经消费的 batch，并准确知道什么时候到了 epoch 末尾。Accelerate 用一个很小的 one-batch lookahead 把这些边界条件集中在一个循环里。

English: A distributed data loader does more than yield tensors. It must place batches on the device, maintain checkpoint state, skip already-consumed batches, and identify the epoch boundary exactly. Accelerate’s one-batch lookahead keeps those concerns aligned in a compact iterator.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/data_loader.py`](https://github.com/huggingface/accelerate/blob/0f7e35fe7902cc6ba8dc01b146d6447900464b88/src/accelerate/data_loader.py#L577-L611)

```python
    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.synchronized_generator)
        self.begin()

        self.set_epoch(self.iteration)
        dataloader_iter = self.base_dataloader.__iter__()
        # We iterate one batch ahead to check when we are at the end
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            self.end()
            return

        batch_index = 0
        while True:
            try:
                # But we still move it to the device so it is done before `StopIteration` is reached
                if self.device is not None:
                    current_batch = send_to_device(current_batch, self.device, non_blocking=self._non_blocking)
                self._update_state_dict()
                next_batch = next(dataloader_iter)
                if batch_index >= self.skip_batches:
                    yield current_batch
                batch_index += 1
                current_batch = next_batch
            except StopIteration:
                self.end_of_dataloader = True
                self._update_state_dict()
                if batch_index >= self.skip_batches:
                    yield current_batch
                break

        self.iteration += 1
        self.end()
```

## 逐行讲解 / What's happening

1. **第 578-583 行 / Lines 578-583**:
   - 中文: 先同步随机状态、调用生命周期钩子、设置 epoch，再拿到底层 iterator。
   - English: It synchronizes RNG state, enters the loader lifecycle, sets the epoch, and creates the underlying iterator.
2. **第 584-589 行 / Lines 584-589**:
   - 中文: 第一批先放进 `current_batch`。如果底层为空，直接调用 `end()`，不会进入一个半初始化的循环。
   - English: The first batch is held in `current_batch`. An empty loader exits cleanly after `end()` instead of entering a partially initialized loop.
3. **第 594-602 行 / Lines 594-602**:
   - 中文: 当前 batch 先搬到 device、更新状态，再尝试取得下一批。只有成功拿到下一批时，当前 batch 才在正常路径上 yield。
   - English: The current batch is moved to the device and checkpointed before fetching the next batch. It is yielded on the normal path only after lookahead succeeds.
4. **第 603-611 行 / Lines 603-611**:
   - 中文: `StopIteration` 说明当前 batch 已经是最后一批；这时仍要更新状态并 yield 它，最后再结束 epoch。
   - English: `StopIteration` proves that the held batch is the last one. The iterator still updates state and yields it before closing the epoch.

## 类比 / The analogy

中文：像检票员手里始终拿着下一位乘客的票。只有看到下一位确实来了，才知道手里这张不是队尾；如果下一位没来，手里这张就是最后一张，不能丢掉。

English: It is like a gate agent holding one passenger’s ticket while checking whether another passenger arrives. A missing next ticket proves the held one is the last, but it must still be admitted.

## 自己跑一遍 / Try it yourself

```python
def lookahead(items, skip=0):
    it = iter(items)
    try:
        current = next(it)
    except StopIteration:
        return
    index = 0
    while True:
        try:
            nxt = next(it)
            if index >= skip:
                yield current
            index += 1
            current = nxt
        except StopIteration:
            if index >= skip:
                yield current
            return

print(list(lookahead(["b0", "b1", "b2"], skip=1)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['b1', 'b2']
```

中文：`skip` 不会让最后一批消失，因为“是否最后一批”的判断和“是否跳过”的判断是两件事。

English: `skip` does not erase the final batch because deciding whether a batch is last is separate from deciding whether the caller should consume it.

## 注意事项 / Caveats / when it breaks

- **lookahead 需要暂存一个 batch** / **Lookahead retains one batch**: 对超大 batch，设备搬运和内存峰值要纳入预算。
- **状态更新时机很关键** / **The update timing matters**: 把 `_update_state_dict()` 放到错误的一侧会产生“已保存但尚未 yield”或“已 yield 但未保存”的偏差。
- **跳过 batch 仍会推进底层 iterator** / **Skipped batches still advance the underlying iterator**: 恢复训练时必须让 sampler 和 loader 的状态一致。

## 延伸阅读 / Further reading

- [Accelerate data_loader.py](https://github.com/huggingface/accelerate/blob/0f7e35fe7902cc6ba8dc01b146d6447900464b88/src/accelerate/data_loader.py)
- [Accelerate DataLoader documentation](https://huggingface.co/docs/accelerate/en/package_reference/torch_wrappers)
