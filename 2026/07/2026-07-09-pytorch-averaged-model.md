---
date: 2026-07-09
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/swa_utils.py
permalink: https://github.com/pytorch/pytorch/blob/e3f5bf0b18585511e6cd7d7a574ebf82f465e5ae/torch/optim/swa_utils.py#L268-L386
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, swa, ema]
---

# PyTorch AveragedModel：第一次复制，之后才平均 / PyTorch AveragedModel: Copy First, Average Later

> **一句话 / In one line**: `AveragedModel.update_parameters()` 用 `n_averaged` 区分初始化 copy 和后续 SWA/EMA 更新，并按 device/dtype 分组走 foreach 快路径。 / `AveragedModel.update_parameters()` uses `n_averaged` to separate the first copy from later SWA/EMA updates, then groups tensors by device and dtype for foreach fast paths.

## 为什么重要 / Why this matters

SWA/EMA 常被描述成一行公式，但在框架里要处理 device、dtype、buffer、第一次更新和自定义平均函数。PyTorch 这段代码展示了一个稳健的影子模型实现：先 deepcopy 模型，再把“第一次同步”和“之后更新”拆开。

SWA/EMA is often shown as a one-line formula, but a framework implementation must handle devices, dtypes, buffers, first update semantics, and custom averaging functions. This code is a compact example of a robust shadow-model implementation.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/swa_utils.py`](https://github.com/pytorch/pytorch/blob/e3f5bf0b18585511e6cd7d7a574ebf82f465e5ae/torch/optim/swa_utils.py#L268-L386)

```python
class AveragedModel(Module):
    n_averaged: Tensor

    def __init__(
        self,
        model: Module,
        device: int | torch.device | None = None,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = None,
        multi_avg_fn: Callable[[PARAM_LIST, PARAM_LIST, Tensor | int], None]
        | None = None,
        use_buffers=False,
    ) -> None:
        super().__init__()
        if avg_fn is not None and multi_avg_fn is not None:
            raise AssertionError(
                "Only one of avg_fn and multi_avg_fn should be provided"
            )
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer(
            "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
        )
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

    def update_parameters(self, model: Module) -> None:
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers
            else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers
            else model.parameters()
        )
        self_param_detached: list[Tensor | None] = []
        model_param_detached: list[Tensor | None] = []
        copy_param = bool(self.n_averaged == 0)
        for p_averaged, p_model in zip(self_param, model_param, strict=False):
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if copy_param:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                grouped_tensors = _group_tensors_by_device_and_dtype(
                    [self_param_detached, model_param_detached]
                )
                for (device, _), ([self_params, model_params], _) in grouped_tensors.items():
                    if self.multi_avg_fn:
                        self.multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                    elif device is not None and device.type in _get_foreach_kernels_supported_devices():
                        multi_avg_fn = get_swa_multi_avg_fn()
                        multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                    else:
                        avg_fn = get_swa_avg_fn()
                        n_averaged = self.n_averaged.to(device)
                        for p_averaged, p_model in zip(self_params, model_params, strict=True):
                            p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
            else:
                for p_averaged, p_model in zip(self_param_detached, model_param_detached, strict=True):
                    n_averaged = self.n_averaged.to(p_averaged.device)
                    p_averaged.detach().copy_(self.avg_fn(p_averaged.detach(), p_model, n_averaged))

        if not self.use_buffers:
            for b_swa, b_model in zip(self.module.buffers(), model.buffers(), strict=True):
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        self.n_averaged += 1
```

## 逐行讲解 / What's happening

1. **第 282-285 行 / Lines 282-285 (exclusive functions)**:
   - 中文: 单参数 `avg_fn` 和批量 `multi_avg_fn` 只能选一个，否则语义会冲突。
   - English: A scalar `avg_fn` and batched `multi_avg_fn` are mutually exclusive because they define the same update slot.
2. **第 286-292 行 / Lines 286-292 (shadow model)**:
   - 中文: 影子模型是 `deepcopy(model)`，`n_averaged` 是 buffer，所以会进 state dict。
   - English: The shadow model is a `deepcopy(model)`, and `n_averaged` is a buffer so it is checkpointed.
3. **第 312-319 行 / Lines 312-319 (first copy)**:
   - 中文: 第一次不是平均，而是直接 copy 当前模型，避免把初始随机权重混进平均。
   - English: The first update is a direct copy, not an average, so initial random shadow weights never enter the running average.
4. **第 323-346 行 / Lines 323-346 (grouped fast path)**:
   - 中文: 后续更新按 device/dtype 分组；能 foreach 就批量更新，不能就逐参数 fallback。
   - English: Later updates are grouped by device/dtype; foreach handles supported devices, with a per-parameter fallback otherwise.

## 类比 / The analogy

这像做一锅长期炖汤：第一天先把锅装满真正的汤底，之后每天才按比例加入新汤。不能先拿一锅清水参与平均。

It is like maintaining a long-running soup base. Day one fills the pot with real broth; only after that do you blend in each new batch. You do not average with plain water.

## 自己跑一遍 / Try it yourself

```python
avg = None
n = 0
for current in [10.0, 14.0, 20.0]:
    if n == 0:
        avg = current
    else:
        avg = avg + (current - avg) / (n + 1)
    n += 1
    print(round(avg, 2), n)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
10.0 1
12.0 2
14.67 3
```

注意第一步是 copy，所以第二步才开始出现真正的 running average。

Notice that step one is a copy; the running average starts on the second call.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **EMA teacher models** / **EMA teacher models**: teacher 第一次同步 student，之后才指数滑动。 / A teacher first syncs from the student, then follows with exponential updates.
- **BatchNorm post-update** / **BatchNorm post-update**: `update_bn` 也是先重置统计量，再重新扫数据。 / `update_bn` resets stats before scanning data again.

## 注意事项 / Caveats / when it breaks

- **buffer 策略要明确** / **Buffer policy must be explicit**: `use_buffers=False` 时 buffer 不是平均，而是同步源模型。 / With `use_buffers=False`, buffers are synced from the source model rather than averaged.
- **自定义函数不会进 state dict** / **Custom functions are not checkpointed**: 恢复训练时要重新传入 `avg_fn` 或 `multi_avg_fn`。 / Resumed training must pass `avg_fn` or `multi_avg_fn` again.

## 延伸阅读 / Further reading

- [PyTorch `AveragedModel.update_parameters`](https://github.com/pytorch/pytorch/blob/e3f5bf0b18585511e6cd7d7a574ebf82f465e5ae/torch/optim/swa_utils.py#L268-L386)
