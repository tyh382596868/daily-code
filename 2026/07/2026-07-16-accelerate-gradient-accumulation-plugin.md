---
date: 2026-07-16
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/utils/dataclasses.py
permalink: https://github.com/huggingface/accelerate/blob/665444ceb62211f2b410d0d0fdb4bc013c5effdf/src/accelerate/utils/dataclasses.py#L994-L1029
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, accelerate, gradient-accumulation, distributed]
---

# Accelerate GradientAccumulationPlugin：累积梯度也要定义同步合同 / Accelerate GradientAccumulationPlugin: Gradient Accumulation Needs a Sync Contract

> **一句话 / In one line**: Accelerate 把梯度累积的步数、scheduler 调整、dataloader 末尾同步和逐 batch 同步压进一个 dataclass。 / Accelerate compresses accumulation steps, scheduler adjustment, end-of-dataloader sync, and per-batch sync into one dataclass.

## 为什么重要 / Why this matters

梯度累积看起来只是“每 N 个 batch 才 step 一次”，但分布式训练里还有两个麻烦：scheduler 该按 micro-batch 还是 optimizer step 走，最后一个 dataloader 尾巴是否要强制同步。Accelerate 的插件把这些语义显式化，避免训练循环里散落隐式 if。

Gradient accumulation sounds like "step every N batches," but distributed training adds two details: whether the scheduler should be adjusted for accumulation, and whether the final dataloader tail forces synchronization. Accelerate makes those semantics explicit in one plugin instead of scattering implicit conditionals through the loop.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/utils/dataclasses.py`](https://github.com/huggingface/accelerate/blob/665444ceb62211f2b410d0d0fdb4bc013c5effdf/src/accelerate/utils/dataclasses.py#L994-L1029)

```python
@dataclass
class GradientAccumulationPlugin(KwargsHandler):
    """
    A plugin to configure gradient accumulation behavior. You can only pass one of
    `gradient_accumulation_plugin` or `gradient_accumulation_steps` to `Accelerator`.
    """

    num_steps: int = field(
        default=None,
        metadata={"help": "The number of steps to accumulate gradients for."},
    )
    adjust_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Whether to adjust the scheduler steps to account for the number of steps being accumulated."
        },
    )
    sync_with_dataloader: bool = field(
        default=True,
        metadata={
            "help": "Whether to synchronize setting the gradients when at the end of the dataloader."
        },
    )
    sync_each_batch: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize setting the gradients at each data batch."
        },
    )
```

## 逐行讲解 / What's happening

1. **`num_steps`**: 中文: 几个 micro-batch 合成一次 optimizer step。 English: how many micro-batches form one optimizer step.
2. **`adjust_scheduler`**: 中文: 如果 scheduler 没为累积预先缩放，就由 Accelerate 调整 step 频率。 English: if the scheduler was not pre-adjusted, Accelerate accounts for accumulation.
3. **`sync_with_dataloader`**: 中文: dataloader 结束时要不要把未满一组的梯度同步出去。 English: controls whether a partial tail at dataloader end still synchronizes.
4. **`sync_each_batch`**: 中文: 每个 batch 都同步会省一些分布式累积内存，但牺牲速度。 English: syncing each batch can reduce memory at the cost of speed.
5. **`KwargsHandler`**: 中文: 这个 dataclass 可以被 Accelerator 收集成 kwargs，而不是手写配置解析。 English: the dataclass can be collected by `Accelerator` as structured kwargs.

## 类比 / The analogy

这像拼车结算。`num_steps` 是几个人一起付一次钱；`sync_with_dataloader` 是最后车没坐满也要不要结账；`sync_each_batch` 是每上一个人就对账一次，稳但慢。

It is like settling a carpool. `num_steps` is how many riders pay together; `sync_with_dataloader` decides whether the half-full last ride settles; `sync_each_batch` reconciles after every rider, safer but slower.

## 自己跑一遍 / Try it yourself

```python
def should_step(i, n, total, sync_tail=True):
    normal = (i + 1) % n == 0
    tail = sync_tail and (i + 1) == total
    return normal or tail

print([i for i in range(5) if should_step(i, 2, 5)])
print([i for i in range(5) if should_step(i, 2, 5, sync_tail=False)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 3, 4]
[1, 3]
```

中文: 开启 tail sync 时，第 5 个 batch 即使没凑满 2 个也会 step。

English: With tail sync, the fifth batch steps even though it does not complete a pair.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DeepSpeed gradient accumulation** / **DeepSpeed gradient accumulation**: 同样把 micro-step 和 optimizer-step 分开计数。 / It also separates micro-steps from optimizer steps.
- **PyTorch DDP `no_sync`** / **PyTorch DDP `no_sync`**: 累积期间跳过同步，边界 batch 再同步。 / It skips synchronization during accumulation and syncs at boundaries.

## 注意事项 / Caveats / when it breaks

- **scheduler 不要双重调整 / Do not double-adjust schedulers**: 如果外部已经按累积步数缩放，`adjust_scheduler=True` 可能让 schedule 变慢。 / If the scheduler is already scaled, `adjust_scheduler=True` can slow it down.
- **tail sync 改变最后一步 / Tail sync changes the last step**: 数据量不能被 `num_steps` 整除时尤其明显。 / This matters when dataset length is not divisible by `num_steps`.
- **逐 batch 同步是折中 / Per-batch sync is a tradeoff**: 省内存但会增加通信。 / It saves memory but adds communication.

## 延伸阅读 / Further reading

- [Accelerate `GradientAccumulationPlugin`](https://github.com/huggingface/accelerate/blob/665444ceb62211f2b410d0d0fdb4bc013c5effdf/src/accelerate/utils/dataclasses.py)
- [Accelerate gradient accumulation guide](https://github.com/huggingface/accelerate/blob/665444ceb62211f2b410d0d0fdb4bc013c5effdf/docs/source/usage_guides/gradient_accumulation.md)
