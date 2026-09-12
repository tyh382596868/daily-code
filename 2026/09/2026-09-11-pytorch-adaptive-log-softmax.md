---
date: 2026-09-11
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/adaptive.py
permalink: https://github.com/pytorch/pytorch/blob/7c873562ff869461e33e92f20089595004a0c605/torch/nn/modules/adaptive.py#L217-L266
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, adaptive-softmax, large-vocabulary, sparse-compute]
---

# PyTorch Adaptive Softmax：先找桶，再算桶内概率 / PyTorch Adaptive Softmax: Find the Bucket, Then Score Inside It

> **一句话 / In one line**: `AdaptiveLogSoftmaxWithLoss` 只为每个 target 计算 head 和对应 tail cluster 的概率，而不是铺满整个词表。 / `AdaptiveLogSoftmaxWithLoss` computes the head and the target's tail cluster instead of scoring the entire vocabulary.

## 为什么重要 / Why this matters

中文：语言模型的词表可能有几十万甚至更多类别，但一个 batch 往往只触碰其中很小一部分。PyTorch 的 adaptive softmax 先用 head 判断样本属于 shortlist 还是某个 cluster，再只对命中的 tail 计算局部 log-probability，最后把两级概率相加。

English: A language model may have hundreds of thousands of classes, while a minibatch touches only a small fraction of them. Adaptive softmax first routes each target through a shortlist or tail cluster, computes only the required local probability, and then adds the head and tail log-probabilities.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/adaptive.py`](https://github.com/pytorch/pytorch/blob/7c873562ff869461e33e92f20089595004a0c605/torch/nn/modules/adaptive.py#L217-L266)

```python
        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)
                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError(
                f"Target values should be in [0, {self.n_classes - 1}], "
                f"but values in range [{target.min().item()}, {target.max().item()}] "
                "were found. "
            )

        head_output = self.head(input)
        head_logprob = F.log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()

        if not is_batched:
            output = output.squeeze(0)

        return _ASMoutput(output, loss)
```

## 逐行讲解 / What's happening

1. **第 217-221 行 / Lines 217-221**:
   - 中文: `output` 保存每个样本的局部 log-probability，`gather_inds` 保存它在 head 中对应的类别或 cluster id。
   - English: `output` stores each sample's local log-probability, while `gather_inds` records the corresponding head class or cluster id.
2. **第 223-231 行 / Lines 223-231**:
   - 中文: 每个 cutoff 区间生成一个 mask；没有样本命中的区间直接跳过。
   - English: Each cutoff interval gets a mask, and intervals with no matching targets are skipped entirely.
3. **第 234-236 行 / Lines 234-236**:
   - 中文: shortlist 中的 target 本身就是 head 类别，不需要 tail 网络。
   - English: A shortlist target is already a head class, so it needs no tail network.
4. **第 238-247 行 / Lines 238-247**:
   - 中文: tail target 先减去区间起点变成局部索引；只选出命中行，计算对应 cluster 的 logits，再 gather 目标位置。
   - English: A tail target becomes a local index by subtracting the interval start. Only matching rows go through that cluster's logits, and the target position is gathered.
5. **第 258-261 行 / Lines 258-261**:
   - 中文: head 概率代表“进入这个 shortlist/cluster”，tail 概率代表“在 cluster 里是哪一个类”；两者相加才是目标的完整 log-probability。
   - English: The head probability says “enter this shortlist or cluster,” while the tail probability says “which class inside it.” Their sum is the full target log-probability.

## 类比 / The analogy

中文：像大型仓库的分拣。先看包裹要送到哪个区域，再只打开那个区域的货架；不必把整个仓库的每个箱子都扫描一遍。

English: It is like sorting a parcel in a huge warehouse. First identify the destination zone, then inspect only that zone's shelf instead of scanning every box in the building.

## 自己跑一遍 / Try it yourself

```python
def route_targets(targets, cutoffs):
    bounds = [0, *cutoffs]
    routes = []
    for target in targets:
        for cluster, (low, high) in enumerate(zip(bounds, bounds[1:])):
            if low <= target < high:
                local = target if cluster == 0 else target - low
                routes.append((target, cluster, local))
                break
        else:
            raise ValueError(f"out of range: {target}")
    return routes

print(route_targets([2, 12, 105], [10, 100, 1000]))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[(2, 0, 2), (12, 1, 2), (105, 2, 5)]
```

中文：`12` 在第二个区间里会变成局部索引 `2`；这就是 tail cluster 为什么可以用更小的输出层。

English: Target `12` becomes local index `2` inside its cluster, which is why a tail can use a smaller output layer.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Hierarchical classifiers** / **分层分类器**: 先预测 coarse label，再预测 fine label，减少全空间计算。 / Predict a coarse label before a fine label to reduce computation over the full space.
- **Mixture-of-experts routing** / **专家路由**: gate 先选少数专家，token 只进入被选中的分支。 / A gate selects a few experts so each token enters only selected branches.
- **Sharded vocabularies** / **分片词表**: 把词表按范围或频率分片，局部计算后再合并概率。 / Partition vocabularies by range or frequency, compute locally, then combine probabilities.

## 注意事项 / Caveats / when it breaks

- **label 排序是前提** / **Label ordering is a prerequisite**: 高频类别应该排在 shortlist 前面，否则 cluster 访问频率和设计目标相反。
- **head 和 tail 要共享概率语义** / **Head and tail must share probability semantics**: tail log-probability 不能单独拿来当全局概率。
- **`log_prob` 会更贵** / **`log_prob` is more expensive**: 需要完整分布时，局部计算优势会明显下降。

## 延伸阅读 / Further reading

- [PyTorch adaptive.py](https://github.com/pytorch/pytorch/blob/7c873562ff869461e33e92f20089595004a0c605/torch/nn/modules/adaptive.py)
- [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309)
