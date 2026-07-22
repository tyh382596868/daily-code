---
date: 2026-07-10
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/checkpoint.py
permalink: https://github.com/pytorch/pytorch/blob/2966664bcc06ec75384ba37ac6b6485216f734c2/torch/utils/checkpoint.py#L1260-L1352
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, checkpointing, dispatch]
---

# PyTorch selective checkpoint：给每个 op 一张保存或重算的票 / PyTorch Selective Checkpoint: Give Each Op a Save-or-Recompute Ticket

> **一句话 / In one line**: selective activation checkpointing 让 policy function 决定每个 op 的输出是保存、重算，还是 CPU offload。 / Selective activation checkpointing lets a policy function decide whether each op output should be saved, recomputed, or offloaded to CPU.

## 为什么重要 / Why this matters

普通 activation checkpoint 是一刀切：中间激活少存，反向时重算。PyTorch 这段把选择权下放到 op 级别，矩阵乘这种贵的 op 可以保存，便宜的 elementwise op 可以重算，于是显存和算力之间可以更细地交易。

Plain activation checkpointing is coarse: save fewer activations and recompute during backward. This code moves the decision to the op level, so expensive matmuls can be saved while cheap elementwise ops are recomputed, giving a finer memory/compute tradeoff.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/checkpoint.py`](https://github.com/pytorch/pytorch/blob/2966664bcc06ec75384ba37ac6b6485216f734c2/torch/utils/checkpoint.py#L1260-L1352)

```python
class SelectiveCheckpointContext:
    """
    Context passed to policy function during selective checkpointing.
    """
    def __init__(self, *, is_recompute, op_output=None) -> None:
        self.is_recompute = is_recompute
        self.op_output = op_output


class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.
    """
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3
    MUST_CPU_OFFLOAD = 4
    PREFER_CPU_OFFLOAD = 5


def _policy_from_bool(b):
    # For backward compatibility
    return CheckpointPolicy.MUST_SAVE if b else CheckpointPolicy.PREFER_RECOMPUTE


SAC_IGNORED_OPS = {
    # AC inserts different number of detach during forward and recompute.
    torch.ops.aten.detach.default,
    # AC's determinism check invokes additional metadata ops during forward.
    # With subclasses involved, these metadata ops become dispatchable, this
    # can result in incorrectness if these ops are selected cached.
    torch.ops.prim.device.default,
} | set(torch._subclasses.functional_tensor.FunctionalTensor.metadata_fns)  # type: ignore[has-type]


def _sac_storage_key(func, args):
    """Compute the SAC storage key for a given op.

    For inductor_compiled_code, each compiled region gets its own FIFO queue
    keyed by the callable's unique idx.  Without this, all compiled regions
    share one queue and a cache-hit that skips a region during recompute
    causes the queue to return the wrong entry (see gh-175258).
    """
    from torch._higher_order_ops.wrap import (
        _resolve_inductor_callable,
        inductor_compiled_code as _inductor_compiled_code,
    )
    if func is _inductor_compiled_code and args:
        return (func, _resolve_inductor_callable(args[0]).idx)
    return func
```

## 逐行讲解 / What's happening

1. **第 1260-1287 行 / Lines 1260-1287 (`SelectiveCheckpointContext`)**:
   - 中文: policy function 不只拿到 op，还能看到 `op_output`，所以可以按输出形状、dtype 或大小做判断。
   - English: The policy function sees not only the op but also `op_output`, so it can decide based on shape, dtype, or output size.
2. **第 1290-1320 行 / Lines 1290-1320 (`CheckpointPolicy`)**:
   - 中文: `MUST_*` 是硬约束，`PREFER_*` 是给编译器或其他系统留余地的偏好。
   - English: `MUST_*` is a hard constraint, while `PREFER_*` leaves room for compilers or other subsystems.
3. **第 1323-1325 行 / Lines 1323-1325 (`_policy_from_bool`)**:
   - 中文: 旧的 bool policy 仍然能用，`True` 等价强制保存，`False` 等价倾向重算。
   - English: Older boolean policies still work: `True` means must save, `False` means prefer recompute.
4. **第 1328-1352 行 / Lines 1328-1352 (ignored ops and storage key)**:
   - 中文: 一些 metadata/detach op 不能随便缓存；compiled region 还要用唯一 id 区分 FIFO 队列。
   - English: Some metadata or detach ops cannot be safely cached; compiled regions also need a unique id to separate FIFO queues.

## 类比 / The analogy

这像厨房备餐：慢炖汤提前留一锅，切葱这种小活现做就行。policy 就是厨师长贴在每道工序旁边的标签。

It is like kitchen prep: keep the slow soup ready, but chop scallions on demand. The policy is the chef's label on each step.

## 自己跑一遍 / Try it yourself

```python
from enum import Enum

class Policy(Enum):
    MUST_SAVE = 0
    PREFER_RECOMPUTE = 3

def policy(op_name):
    return Policy.MUST_SAVE if op_name in {"matmul", "attention"} else Policy.PREFER_RECOMPUTE

for op in ["relu", "matmul", "add", "attention"]:
    print(op, policy(op).name)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
relu PREFER_RECOMPUTE
matmul MUST_SAVE
add PREFER_RECOMPUTE
attention MUST_SAVE
```

这个 toy policy 把昂贵 op 保存，把便宜 op 留给反向时重算。

This toy policy saves expensive ops and leaves cheap ops for recomputation during backward.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TorchDispatchMode** / **TorchDispatchMode**: selective checkpoint 靠 dispatch mode 观察每个 op。 / Selective checkpointing uses dispatch modes to observe each op.
- **Rematerialization in JAX** / **Rematerialization in JAX**: 也通过策略控制哪些中间值要重算。 / It also uses policy-like choices for which intermediates to recompute.

## 注意事项 / Caveats / when it breaks

- **保存太多不等于不开 checkpoint** / **Saving everything is not the same as no checkpoint**: 注释明确说 `PREFER_SAVE` every op 会额外保存不一定需要的 tensor。 / The comments note that saving every op can store tensors that gradients do not actually need.
- **op 顺序必须稳定** / **Op order must stay stable**: recompute 阶段靠相同 key 和调用计数取缓存，非确定性控制流会破坏匹配。 / The recompute pass retrieves cache entries by key and call index, so nondeterministic control flow breaks matching.

## 延伸阅读 / Further reading

- [PyTorch selective checkpoint policy](https://github.com/pytorch/pytorch/blob/2966664bcc06ec75384ba37ac6b6485216f734c2/torch/utils/checkpoint.py#L1260-L1352)
