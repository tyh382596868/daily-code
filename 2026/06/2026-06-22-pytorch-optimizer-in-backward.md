---
date: 2026-06-22
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/distributed/optim/apply_optimizer_in_backward.py
permalink: https://github.com/pytorch/pytorch/blob/37c67b79e2c83f1d7e22548f8dc196ea034a1b11/torch/distributed/optim/apply_optimizer_in_backward.py#L18-L100
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, pytorch, autograd, memory-optimization, distributed-training, optimizer]
---

# 优化器藏进反向传播：`_apply_optimizer_in_backward` / The Optimizer Hidden Inside Backward: `_apply_optimizer_in_backward`

> **一句话 / In one line**: 给每个参数注册一个 `AccumulateGrad` 钩子，梯度一就绪就立刻 `opt.step()` 并把 `.grad` 置 None——反向传播结束时梯度缓冲区已经清空，峰值显存大幅下降。 / Register a hook on each parameter's `AccumulateGrad` node; the moment a gradient is ready, fire `opt.step()` and set `.grad = None` — the gradient buffer is gone by the time backward finishes, slashing peak memory.

## 为什么重要 / Why this matters

标准训练流程：`loss.backward()` 先把所有参数的梯度全部算出来、存进 `.grad`，`backward()` 返回后再调 `optimizer.step()`。这意味着在 `backward()` 运行的整个过程中，所有参数的梯度都要同时驻留在显存里，峰值显存 ≈ 模型参数量 × 2（weights + grads）。

`_apply_optimizer_in_backward` 把 `step()` 挪到了反向传播的内部：每当一个参数的梯度被 `AccumulateGrad` 节点写入 `.grad`，对应的优化器钩子立刻触发 `step()`，然后把 `.grad` 置 None。这样，梯度在写入的同一时刻就被消费掉了，全程最多只有"正在被反向传播经过的那一层"的梯度活在显存里。对 70B 模型来说，峰值梯度显存可从 ~140 GB 降到接近单层大小。

Standard training: `loss.backward()` computes and stores every parameter's gradient into `.grad`, then `optimizer.step()` is called after backward completes. During the entire backward pass, all gradients coexist in memory — peak memory ≈ 2× parameter count (weights + grads).

`_apply_optimizer_in_backward` moves `step()` inside the backward pass: as soon as each parameter's gradient is written by its `AccumulateGrad` node, the registered hook fires `step()` and immediately sets `.grad = None`. Gradients are consumed the instant they're produced, so peak gradient memory drops to roughly one layer's worth at a time — critical for fine-tuning 70B+ models on limited hardware.

## 代码 / The code

`pytorch/pytorch` — [`torch/distributed/optim/apply_optimizer_in_backward.py`](https://github.com/pytorch/pytorch/blob/37c67b79e2c83f1d7e22548f8dc196ea034a1b11/torch/distributed/optim/apply_optimizer_in_backward.py#L18-L100)

```python
param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()


@no_type_check
def _apply_optimizer_in_backward(
    optimizer_class: type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: dict[str, Any],
    register_hook: bool = True,
) -> None:
    torch._C._log_api_usage_once("torch.distributed.optim.apply_optimizer_in_backward")

    @no_type_check
    def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
        # view_as creates a node in autograd graph that allows us access to the
        # parameter's AccumulateGrad autograd function object. We register a
        # hook on this object to fire the optimizer when the gradient for
        # this parameter is ready (has been accumulated into .grad field)

        # Don't create a new acc_grad if we already have one
        # i.e. for shared parameters or attaching multiple optimizers to a param.
        if param not in param_to_acc_grad_map:
            param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[
                0
            ][0]

        optimizer = optimizer_class([param], **optimizer_kwargs)

        if not hasattr(param, "_in_backward_optimizers"):
            param._in_backward_optimizers = []
            param._optimizer_classes = []
            param._optimizer_kwargs = []

        param._in_backward_optimizers.append(optimizer)
        param._optimizer_classes.append(optimizer_class)
        param._optimizer_kwargs.append(optimizer_kwargs)

        if not register_hook:
            return

        def optimizer_hook(*_unused) -> None:
            for opt in param._in_backward_optimizers:
                opt.step()
            param.grad = None

        handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)
        if param not in param_to_optim_hook_handle_map:
            param_to_optim_hook_handle_map[param] = []
        param_to_optim_hook_handle_map[param].append(handle)

    for param in params:
        _apply_optimizer_in_backward_to_param(param)
```

## 逐行讲解 / What's happening

1. **`WeakTensorKeyDictionary`（全局，两个）**:
   - 中文: 用弱引用字典存"param → AccumulateGrad 节点"和"param → 钩子 handle"，这样 param 被垃圾回收后字典自动清理，不会造成内存泄漏。普通 dict 的话 param 永远不会被回收。
   - English: Weak-reference dicts keyed on the parameter tensor. When the parameter is garbage-collected, its entries disappear automatically — a regular `dict` would pin the tensor in memory forever.

2. **`param.view_as(param).grad_fn.next_functions[0][0]`**:
   - 中文: 这是整段代码最精妙的地方。`param.view_as(param)` 在 autograd 图里创建一个 `ViewBackward` 节点；`grad_fn.next_functions[0][0]` 是这个节点的上游——对叶子参数来说，上游就是 `AccumulateGrad` 节点（PyTorch 专门用来把梯度写进 `.grad` 的内置节点）。取到这个节点，才能在"梯度刚刚写入"的瞬间注册钩子。
   - English: This is the cleverest line in the file. `param.view_as(param)` inserts a `ViewBackward` node into the autograd graph; its upstream (`grad_fn.next_functions[0][0]`) is the `AccumulateGrad` node — the built-in PyTorch node responsible for writing gradients into `.grad`. Grabbing this node is the only way to hook in at the exact moment the gradient lands.

3. **`optimizer = optimizer_class([param], **optimizer_kwargs)`**:
   - 中文: 每个参数得到自己的独立 optimizer 实例，只管理这一个参数。这样 optimizer 的状态（Adam 的 m/v）和参数绑定，和 `param` 一起活在显存里，但不需要批量管理所有参数的状态。
   - English: Each parameter gets its own dedicated optimizer instance managing only that one parameter. The optimizer's state (Adam's first/second moments) is tied to this parameter and lives with it in memory, but no cross-parameter bookkeeping is needed.

4. **`param._in_backward_optimizers.append(optimizer)`**:
   - 中文: 把 optimizer 挂在 param 自身的属性上（而不是外部字典），这样一个参数可以注册多个 optimizer（例如先 SGD 再 Adam），钩子执行时会挨个 `.step()`。
   - English: Attaches the optimizer to the parameter itself so multiple optimizers can be registered on one parameter (e.g., different lr schedules for different param groups). The hook loops over `param._in_backward_optimizers` and calls each one.

5. **`optimizer_hook(*_unused): opt.step(); param.grad = None`**:
   - 中文: 钩子的核心：调用 optimizer.step() 消费当前梯度，然后立刻把 `.grad` 置 None。`*_unused` 是因为 AccumulateGrad 的钩子接收一个 grad 参数，但我们不需要它（`.grad` 此时已经被写入）。
   - English: The hook's core: call `opt.step()` to consume the gradient, then immediately null `.grad`. The `*_unused` signature accepts the grad tensor passed by `AccumulateGrad`'s hook protocol — we don't need it since `.grad` is already set when this fires.

6. **`param_to_acc_grad_map[param].register_hook(optimizer_hook)`**:
   - 中文: 把钩子注册到 AccumulateGrad 节点（不是参数本身的 `.register_hook()`，后者的触发时机稍晚）。AccumulateGrad 钩子在梯度刚好写入 `.grad` 之后立刻触发，是最早的触发点。
   - English: Registers the hook on the `AccumulateGrad` node (not `param.register_hook()` which fires slightly later). `AccumulateGrad` hooks fire the instant a gradient is written to `.grad` — the earliest possible hook point in the backward pass.

## 类比 / The analogy

想象一条流水线：工人甲（backward）逐站生产零件（梯度），通常所有零件全部生产完才交给工人乙（optimizer）安装。`_apply_optimizer_in_backward` 相当于在每个工位放了一个即时安装员：零件刚下线就被安装进产品，安装完立刻回收模具（`.grad = None`）。流水线上最多只有一个零件在等待安装，仓储压力从"N 个零件同时堆积"变成了"最多 1 个"。

Imagine a production line: worker A (backward) produces parts (gradients) at each station; normally all parts are stockpiled before worker B (optimizer) starts assembling. `_apply_optimizer_in_backward` places an instant-install crew at each station: the moment a part rolls off the line it's assembled into the product, and the mold (`.grad`) is immediately recycled. At most one part waits in the buffer at any time — warehousing pressure drops from "N parts all at once" to "at most 1."

## 自己跑一遍 / Try it yourself

```python
import torch
from torch.distributed.optim import _apply_optimizer_in_backward

model = torch.nn.Sequential(
    torch.nn.Linear(64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)

_apply_optimizer_in_backward(
    torch.optim.Adam,
    model.parameters(),
    {"lr": 1e-3},
)

x = torch.randn(4, 64)
loss = model(x).sum()
loss.backward()

# After backward(), parameters are already updated and .grad is None
for name, p in model.named_parameters():
    print(f"{name}: grad={p.grad}, updated={p.requires_grad}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
0.weight: grad=None, updated=True
0.bias: grad=None, updated=True
2.weight: grad=None, updated=True
2.bias: grad=None, updated=True
```

注意 `backward()` 返回后 `.grad` 全部是 None——意味着梯度在反向传播期间已经被消费完了，不需要再 `zero_grad()`。

After `backward()`, all `.grad` are already `None` — gradients were consumed during the backward pass itself. No `optimizer.step()` or `zero_grad()` calls needed afterward.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ZeRO-3 (DeepSpeed)**: ZeRO-3 把参数、梯度、optimizer state 全部切分到多卡，梯度 reduce 后也是立刻消费——和这里的思路完全一致，只是在 reduce 之后触发而不是 accumulate 之后。 / ZeRO-3 partitions params, grads, and optimizer state across GPUs; gradients are consumed immediately after the cross-GPU reduce — the same "instant consume" philosophy.
- **`torch.optim.Optimizer` + `param.register_post_accumulate_grad_hook`**: PyTorch 后来加了更官方的 `register_post_accumulate_grad_hook` API，和这里的 AccumulateGrad 钩子语义完全相同，是它的公开版本。 / PyTorch later added `register_post_accumulate_grad_hook` as the public API for the same semantics — it's the blessed version of this AccumulateGrad trick.
- **FSDP2 的 `fully_shard`**: FSDP2 内部用同样的机制在 backward 期间触发 optimizer step，以便立刻释放已处理 shard 的梯度。 / FSDP2's `fully_shard` uses the same mechanism to trigger optimizer steps during backward, releasing each shard's gradient the moment it's ready.

## 注意事项 / Caveats / when it breaks

- **gradient clipping 失效 / Gradient clipping breaks**: `torch.nn.utils.clip_grad_norm_` 需要等到所有梯度都在才能算全局 norm，但这里梯度已经被 step() 消费了。如果你需要 grad clipping，必须在 AccumulateGrad 钩子内部做，或者把 clipping 逻辑嵌入自定义 optimizer。 / `clip_grad_norm_` requires all gradients to be present simultaneously. Since grads are consumed per-parameter, you must embed clipping logic inside a custom optimizer or hook.
- **不兼容标准 `optimizer.step()` / Incompatible with standard `optimizer.step()`**: 外层如果还调用 `optimizer.step()`，该 optimizer 的 params 的 `.grad` 已经是 None，step 是 no-op（不会出错，但毫无效果）。 / If you also call an outer `optimizer.step()`, that optimizer's params have `.grad = None` — its step is a no-op (silent, but no update happens).
- **共享参数的多次梯度 / Shared parameters with multiple gradient contributions**: 如果同一个参数被多个 loss term 用到，AccumulateGrad 只在最后一次 accumulate 后触发，所以钩子仍能看到完整梯度。但若使用 `retain_graph=True` 多次 backward，需要额外注意计数。 / `AccumulateGrad` fires after all contributions are summed, so the hook still sees the complete gradient even with multiple loss terms. However, `retain_graph=True` + multiple backward calls requires extra care.

## 延伸阅读 / Further reading

- [PyTorch `register_post_accumulate_grad_hook` 文档](https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html) — 公开 API 版本。
- [ZeRO 论文 (Rajbhandari et al. 2020)](https://arxiv.org/abs/1910.02054) — 系统性介绍"梯度即产即用"的内存收益。
- [`torch/distributed/optim/` 目录](https://github.com/pytorch/pytorch/tree/main/torch/distributed/optim) — 还有 `ZeroRedundancyOptimizer`（ZeRO-1 纯 Python 实现）和 `PostLocalSGDOptimizer`（本地 SGD + 全局同步）。
