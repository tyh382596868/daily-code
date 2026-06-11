---
date: 2026-06-11
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/data_loader.py
permalink: https://github.com/huggingface/accelerate/blob/67bf4d644f65be28cbba615762578130a849f239/src/accelerate/data_loader.py#L213-L271
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, huggingface, accelerate, dataloader, batch-sampler, sharding]
---

# Accelerate 把"动态 batch size"塞进了多卡 sharding:一招"循环填回初始 batch"让所有进程同步收尾 / Accelerate retrofits dynamic batch sizes into multi-process sharding via the classic "ring back to initial batches" trick

> **一句话 / In one line**: 当 batch sampler 不返回固定长度的 batch(比如按 token 预算打包)时,Accelerate 的 `BatchSamplerShard._iter_with_no_split` 不能再像旧逻辑那样"按元素填回",而是必须**按整 batch 循环填回** — 这就是 PR #3969 解决的死角。 / When a batch sampler emits variable-length batches (e.g. token-budgeted packing), Accelerate's `_iter_with_no_split` can no longer top up by individual elements; it has to **cycle whole batches** back from the beginning — that's the corner case PR #3969 fixes.

## 为什么重要 / Why this matters

数据并行训练里,每个进程都要拿到**同样数量**的 batch,否则 `step` 数不齐,`gradient sync` 会卡死或者算错梯度平均。Accelerate 的 `BatchSamplerShard` 干的就是这件事 — 让 8 张 GPU 在一个 epoch 里各拿到等量的 batch。它的策略是:**正常分发**已有的 batch,然后**在末尾用前几个 batch 循环填回**,直到每个进程都拿满。原版代码假设 batch size 固定,所以填回逻辑是"按元素拼接":`batch += initial_data[cycle_index:cycle_index + batch_size - len(batch)]`。但现在越来越多的 sampler 是按 **token 预算**或**长度桶**打包的,batch size 不固定 — 比如 nanoGPT 的 packing sampler、LLaMA-Factory 的 length-balanced sampler。如果 process 0 yield 了一个 31 个样本的 batch,你能"再凑 1 个"塞进去吗?不能 — 那个样本的 token 数可能让总长超出预算。所以 PR #3969 加了**两条平行的分支**:`if self.batch_size is None:` 时整段填回的是"完整的 initial batch",不再补元素。这是一个看似微小、实际是**多进程确定性的关键**的修复。

In data-parallel training every process must yield the **same number of batches** — otherwise step counts misalign and gradient sync deadlocks or averages wrong. Accelerate's `BatchSamplerShard` handles exactly this: making 8 GPUs each pull an equal count of batches within an epoch. Its strategy is to **distribute batches normally**, then **cycle back through the first few** until every process is filled. The original logic assumed a fixed batch size, so the fill-up code stitched **per-element**: `batch += initial_data[cycle_index:cycle_index + batch_size - len(batch)]`. But more and more samplers now emit batches sized by **token budget** or **length bucket** — see nanoGPT's packing sampler, LLaMA-Factory's length-balanced sampler. If process 0 yields a 31-sample batch, you can't just "add one more" — that extra sample's token count might blow past the budget. PR #3969 adds **two parallel branches**: when `self.batch_size is None`, the cycle replays whole initial batches rather than splicing element-wise. Tiny diff, **massive correctness implication** for multi-process determinism.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/data_loader.py`](https://github.com/huggingface/accelerate/blob/67bf4d644f65be28cbba615762578130a849f239/src/accelerate/data_loader.py#L213-L271)

```python
def _iter_with_no_split(self):
    initial_data = []
    batch_to_yield = None
    for idx, batch in enumerate(self.batch_sampler):
        # We gather the initial indices in case we need to circle back at the end.
        if not self.drop_last and idx < self.num_processes:
            if self.batch_size is None:
                # If batch size is None, `batch` is considered to be a list of indices with dynamic length.
                initial_data.append(batch)
            else:
                initial_data += batch
        # We identify the batch to yield but wait until we ar sure every process gets a full batch before actually
        # yielding it.
        if idx % self.num_processes == self.process_index:
            batch_to_yield = batch
        if idx % self.num_processes == self.num_processes - 1 and (
            self.batch_size is None or len(batch) == self.batch_size
        ):
            yield batch_to_yield
            batch_to_yield = None

    # If drop_last is True, iteration is over, otherwise...
    if not self.drop_last and len(initial_data) > 0:
        if not self.even_batches:
            if batch_to_yield:
                yield batch_to_yield
        else:
            # ... we yield the complete batch we had saved before if it has the proper length
            if batch_to_yield and (self.batch_size is None or len(batch_to_yield) == self.batch_size):
                yield batch_to_yield

            # For degenerate cases where the dataset has less than num_process * batch_size samples
            _min_length_needed = (
                self.num_processes * self.batch_size if self.batch_size is not None else self.num_processes
            )
            while len(initial_data) < _min_length_needed:
                initial_data += initial_data

            # If the last batch seen was of the proper size, it has been yielded by its process so we move to the next
            if self.batch_size is None or len(batch) == self.batch_size:
                batch = []
                idx += 1

            # Make sure we yield a multiple of self.num_processes batches
            cycle_index = 0
            while idx % self.num_processes != 0 or len(batch) > 0:
                if self.batch_size is None:
                    batch = initial_data[cycle_index]
                    if idx % self.num_processes == self.process_index:
                        yield batch
                    cycle_index += 1
                else:
                    end_index = cycle_index + self.batch_size - len(batch)
                    batch += initial_data[cycle_index:end_index]
                    if idx % self.num_processes == self.process_index:
                        yield batch
                    cycle_index = end_index
                batch = []
                idx += 1
```

## 逐行讲解 / What's happening

1. **`initial_data` 的两种存法 / Two flavors of `initial_data`**:
   - 中文: 头 `num_processes` 个 batch 永远要被记下来 — 它们是"循环填回"的弹药。固定 batch size 时,把它们**摊平存元素**(`+= batch`);动态 batch size 时,把它们**整 batch 存进 list**(`.append(batch)`)。这一段决定了后面填回的粒度。
   - English: the first `num_processes` batches are always stashed — they are the cycle-back ammo. With a fixed batch size, store them flattened (`+= batch`); with dynamic size, store them as whole batches (`.append(batch)`). This single conditional decides whether refills will happen at the element level or the batch level.

2. **`if idx % self.num_processes == self.process_index: batch_to_yield = batch`**:
   - 中文: 多进程轮流"领"一个 batch — 进程 0 拿 idx=0,进程 1 拿 idx=1,…,进程 N 拿 idx=N。但这里**不立刻 yield**,而是先存到 `batch_to_yield` 里。为什么?因为我们要等到这一**轮**的最后一个 batch 也确认是"完整的"再 yield,否则你可能 yield 了一个最终发现要丢的 batch。
   - English: processes take turns claiming batches in round-robin order. But the function doesn't `yield` immediately — it stashes the claimed batch in `batch_to_yield`. Why? Because we have to wait until the *last* batch of this round is confirmed "complete" — otherwise we might yield a batch that turns out to be in a half-finished round.

3. **`if idx % num == num - 1 and (batch_size is None or len(batch) == batch_size): yield`**:
   - 中文: 一轮的最后一个 batch 必须**满**。如果 batch_size 是固定的,要 `len(batch) == self.batch_size`;如果是动态的,空字符串"按现状信任",任何 batch 都算"满"。只有这一轮所有进程都拿到 batch 且最后那个 batch 看起来正常,才整轮 yield 出去。
   - English: a round only commits if its last batch is "full." For fixed batch size, that means `len(batch) == self.batch_size`. For dynamic batch size, *any* batch counts as full (you trust the sampler's judgment). Only when both conditions hold does the round actually yield.

4. **`_min_length_needed = num_processes * batch_size if batch_size else num_processes`**:
   - 中文: 准备填回。要保证 `initial_data` 至少够填一整轮,所以如果不够,**翻倍**自己直到长度够。这是个简单的"放大池子"逻辑:`while len(initial_data) < min: initial_data += initial_data`。
   - English: prepare for refill. Make sure `initial_data` has enough material for a full round — if not, **double it in place** with `initial_data += initial_data` until it does. Simple pool-expansion logic.

5. **`if self.batch_size is None or len(batch) == self.batch_size: batch = []; idx += 1`**:
   - 中文: 如果最后看到的 batch 已经被它所属的进程 yield 掉了,那么"游标"前进一格、重置 batch。如果最后那个 batch 不满,**它在动态分支不会进入这里**(`batch_size is None` 之外的分支),会被留到下面的循环里继续填。
   - English: if the last batch we saw was a full one (its rightful process already yielded it), advance the cursor and clear `batch`. If the last batch was *partial*, in the dynamic branch we *do* fall through (since the condition collapses to `True`); in the fixed branch we keep the leftover indices for the cycle below to top up.

6. **核心循环 — 整 batch vs 按元素填回 / The core loop — whole-batch vs per-element refill**:
   - 中文: 这是 PR 真正动刀的地方。固定 batch size 的旧逻辑是 `batch += initial_data[cycle_index:end_index]` — 从摊平的元素池里**按元素**裁出 `batch_size - len(batch)` 个,补齐当前 batch。新加的 `if self.batch_size is None:` 分支用 `batch = initial_data[cycle_index]` — **整 batch 拿出来**,不再做元素切片。`cycle_index` 在动态分支是"batch 索引",在固定分支是"元素索引"。
   - English: this is where PR #3969 lives. The old fixed-size branch (`batch += initial_data[cycle_index:end_index]`) slices the flattened element pool. The new `if self.batch_size is None:` branch grabs a **whole batch** with `batch = initial_data[cycle_index]` — no element-level slicing. `cycle_index` means "batch index" in the dynamic branch and "element index" in the fixed branch — same name, two semantics, picked apart by the condition.

## 类比 / The analogy

想象 8 个朋友合伙吃自助餐,服务员一桌一桌端菜上来。**正常情况下**,菜按顺序在 8 个人之间轮转 — 每人轮到自己时把菜端走。问题在自助餐快结束的时候:菜不够 8 个人再轮一轮了。**旧方法**(固定份量)是:服务员把前几桌剩的菜按勺子重新分配,凑够 8 份。**新方法**(动态份量)是:服务员**把第一桌端出来的整桌**再原样端给最后没拿到的人 — 因为每桌的份量原本就是按某个规则定的(比如"这桌的菜不超过 2000 卡路里"),你不能拆开来重新组合,只能整桌复用。

Picture 8 friends at a buffet, with the server bringing dishes one round at a time. **Normally** each dish rotates between the 8 — each person takes theirs when their turn comes. The trouble starts at end-of-buffet: the kitchen doesn't have enough left for one more full round. The **old approach** (fixed portions): the server scoops leftovers from prior plates and assembles 8 equal portions. The **new approach** (dynamic portions): the server **brings out the very first table's entire plate again** — because each plate was sized by some rule (e.g., "no more than 2 000 calories per plate"), and you can't unmake the plate to reshuffle it. You just have to **reuse plates wholesale**.

## 自己跑一遍 / Try it yourself

```python
# pip install accelerate
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, SequentialSampler

# A "dynamic" batch sampler that emits variable-length batches by token budget
class TokenBudgetSampler:
    def __init__(self, lengths, budget=10):
        self.lengths = lengths; self.budget = budget
    def __iter__(self):
        batch, total = [], 0
        for i, L in enumerate(self.lengths):
            if total + L > self.budget and batch:
                yield batch; batch, total = [], 0
            batch.append(i); total += L
        if batch: yield batch
    def __len__(self):
        return sum(1 for _ in iter(self))

sampler = TokenBudgetSampler(lengths=[3, 4, 2, 5, 3, 4, 2, 3], budget=8)
print("native batches:", list(iter(sampler)))

for rank in range(2):
    shard = BatchSamplerShard(
        batch_sampler=sampler,
        num_processes=2,
        process_index=rank,
        split_batches=False,
        even_batches=True,
    )
    print(f"rank {rank}:", list(iter(shard)))
```

运行 / Run with:
```bash
pip install accelerate>=1.13
python try.py
```

预期输出 / Expected output:
```
native batches: [[0, 1], [2, 3], [4, 5], [6, 7]]
rank 0: [[0, 1], [4, 5]]
rank 1: [[2, 3], [6, 7]]
```

中文:每个 rank 拿到的 batch 数相等(`even_batches=True`),而且即使最后一轮没填满,Accelerate 也会用前几个 batch 循环补回 — 现在动态 batch 也能享受到这层保护。

English: each rank gets the same number of batches (`even_batches=True`), and even when the last round is short, Accelerate cycles back to the initial batches — dynamic batch sizes now get the same protection that fixed sizes have always had.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `DistributedSampler(drop_last=False)`** / **PyTorch 的分布式采样器**: 中文: 同样的"末尾填回"思路,但只支持固定 batch size。Accelerate 是它的多进程加强版。 / English: same "cycle back at the end" idea, but only supports fixed batch size — Accelerate's shard is the dynamic-aware sibling.
- **`tf.data.Dataset.repeat()` 的尾部对齐** / **TensorFlow 的 repeat 尾部对齐**: 中文: TF 的 `repeat(count=None) + take(n)` 也常用来让多 worker 拿到对齐的 step 数。 / English: TF's `repeat(None) + take(n)` is the canonical way to align step counts across workers.
- **NLP packing sampler(token budget)** / **NLP 打包采样器**: 中文: LLaMA-Factory、nanoGPT、TRL 都有按 token 预算的 packing sampler,它们 batch size 都是动态的 — 这正是 PR #3969 要支持的场景。 / English: LLaMA-Factory, nanoGPT, and TRL all ship token-budget packing samplers with variable batch sizes — exactly what PR #3969 makes Accelerate-compatible.
- **MPI 的 round-robin scatter + ring fill** / **MPI 的轮转散发 + 环填回**: 中文: 经典分布式原语;每个 rank 拿到对齐数量的工作单位,末尾不足时循环补齐。 / English: a textbook distributed primitive — give each rank an aligned chunk of work and ring-fill the tail.

## 注意事项 / Caveats / when it breaks

- **`even_batches=True` 是有代价的 / Even batches isn't free**:
  - 中文: 末尾循环填回意味着**有重复样本进梯度**。如果训练对重复样本敏感(比如 GRPO、DPO),最好用 `even_batches=False` 加上能容忍 step 不齐的同步策略。
  - English: end-of-epoch cycle-back means **duplicate samples enter your gradients**. If your training is sensitive to duplicates (GRPO, DPO, RLHF), prefer `even_batches=False` and a sync strategy that tolerates uneven step counts.
- **动态 batch size 的 BPE / token 预算谁来管? / Who owns the token budget?**:
  - 中文: Accelerate 不知道你 batch 的"逻辑大小";它信任 sampler 的判断。如果 sampler 自己出错,这里也救不了你。
  - English: Accelerate has no idea what your batch's "logical size" is — it trusts the sampler. If the sampler's budgeting is wrong, this shard layer cannot rescue it.
- **`drop_last=True` 可以完全跳过这段逻辑 / `drop_last=True` skips this whole codepath**:
  - 中文: 如果你能丢掉末尾不齐的部分,设 `drop_last=True`,整段循环填回直接不跑 — 简单干净。
  - English: if you can drop the misaligned tail, set `drop_last=True` and the entire fill-back block becomes dead code — simpler, deterministic, no duplicates.
- **`initial_data += initial_data` 的爆炸性扩张 / Pool doubling can OOM**:
  - 中文: 如果数据集比 `num_processes` 小很多(degenerate case),`initial_data` 会被翻倍多次。理论上是安全的,但 GPU OOM 之前先内存 OOM 的可能。
  - English: if the dataset is much smaller than `num_processes` (degenerate), `initial_data += initial_data` can balloon. Memory OOM may strike before you even get to GPU OOM.

## 延伸阅读 / Further reading

- [PR #3969 — "Support dynamic batch size in BatchSamplerShard"](https://github.com/huggingface/accelerate/pull/3969)
- [Accelerate docs — `Accelerator.prepare(dataloader)`](https://huggingface.co/docs/accelerate/v1.13.0/en/package_reference/accelerator#accelerate.Accelerator.prepare)
- [PyTorch `DistributedSampler` source](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py)
- [TRL packing — token-budget sampler](https://huggingface.co/docs/trl/en/sft_trainer#packing-dataset)
