---
date: 2026-08-07
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/checkpoint.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L794-L918
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, checkpointing, autograd]
---

# PyTorch checkpoint early-stop：重算够了就停 / PyTorch Checkpoint Early-Stop: Stop Recompute Once Enough Is Rebuilt

> **一句话 / In one line**: 非 reentrant checkpoint 会在重算拿到所需 saved tensor 后提前停止，而不是完整重跑 forward。 / Non-reentrant checkpoint can stop recomputation as soon as the needed saved tensors are rebuilt.

## 为什么重要 / Why this matters

Activation checkpointing 用重算换显存。朴素做法是 backward 时把整段 forward 再跑一遍；PyTorch 的 early-stop 更细：只要被 autograd 需要的 tensor 已经重建，就可以中断后续无关计算。

Activation checkpointing trades recompute for memory. The naive version reruns the whole forward during backward; PyTorch's early-stop is finer: once the tensors needed by autograd have been rebuilt, unrelated later work can be skipped.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/checkpoint.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L794-L918)

```python
_enable_checkpoint_early_stop = None

@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    global _enable_checkpoint_early_stop
    prev = _enable_checkpoint_early_stop
    _enable_checkpoint_early_stop = enable
    try:
        yield
    finally:
        _enable_checkpoint_early_stop = prev
```

## 逐行讲解 / What's happening

1. **第 794-802 行 / Lines 794-802 (rule)**:
   - 中文: 注释明确规则：重算到“已知需要的 tensor 数量”就抛内部异常停止。
   - English: The rule says recomputation stops when the known needed tensors have been rebuilt.
2. **第 866-885 行 / Lines 866-885 (user switch)**:
   - 中文: `set_checkpoint_early_stop(False)` 只需要包住 forward；backward 发生在外面也会继承这个选择。
   - English: `set_checkpoint_early_stop(False)` only needs to wrap forward; backward can happen later and still follows the choice.
3. **第 886-892 行 / Lines 886-892 (restore)**:
   - 中文: 全局开关用 `try/finally` 恢复，避免影响外层训练代码。
   - English: The global switch is restored with `try/finally` so surrounding training code is not polluted.
4. **第 902-918 行 / Lines 902-918 (frame state)**:
   - 中文: `_CheckpointFrame` 记录 recompute 函数、holder、计数器和 `early_stop` 标志。
   - English: `_CheckpointFrame` stores the recompute function, holders, counters, and the `early_stop` flag.

## 类比 / The analogy

像回放监控录像找钥匙：找到钥匙掉在哪一帧后，不必把剩下两个小时都看完。

It is like replaying security footage to find dropped keys: once the needed frame is found, the rest of the video can be skipped.

## 自己跑一遍 / Try it yourself

```python
needed = {"b"}
recomputed = []
for name in ["a", "b", "c", "d"]:
    recomputed.append(name)
    if needed.issubset(recomputed):
        break
print(recomputed)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['a', 'b']
```

这就是 early-stop 的核心：不是不重算，而是不多重算。

That is the core of early-stop: recompute, but do not over-recompute.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Selective checkpoint policy** / **Selective checkpoint policy**: 保存/重算决策也可以按 op 精细化。 / Save/recompute choices can also be made per op.
- **Compiler dead-code elimination** / **Compiler dead-code elimination**: 只执行对最终需求有贡献的部分。 / Execute only the work that contributes to the required result.

## 注意事项 / Caveats / when it breaks

- **副作用 forward 不适合** / **Side-effectful forward is risky**: 提前停止意味着后半段副作用不会重放。 / Early-stop means later side effects are not replayed.
- **调试时可关闭** / **Disable for debugging**: 某些复杂 autograd 场景需要完整重算来排查。 / Some complex autograd cases are easier to debug with full recomputation.

## 延伸阅读 / Further reading

- [PyTorch checkpoint internals](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L794-L918)

