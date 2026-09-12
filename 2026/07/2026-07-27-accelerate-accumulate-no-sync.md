---
date: 2026-07-27
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/accelerator.py
permalink: https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/accelerator.py#L1255-L1297
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, accelerate, gradient-accumulation]
---

# Accelerate accumulate：梯度同步是一个上下文选择 / Accelerate accumulate: Gradient Sync Is a Context Choice

> **一句话 / In one line**: `Accelerator.accumulate` 每个 step 先判断这次 backward 是否该同步，再为每个 model 选择 `no_sync` 或普通上下文。 / `Accelerator.accumulate` first decides whether this backward should synchronize, then wraps each model in either `no_sync` or a plain context.

## 为什么重要 / Why this matters

分布式梯度累积的难点不是少写几行 `if`，而是让“哪些 microbatch 不 all-reduce、哪一个 microbatch 必须 all-reduce”稳定地落在同一套状态机里。Accelerate 把这个判断封装成上下文，训练代码保持普通形状。

The hard part of distributed gradient accumulation is not removing an `if`; it is keeping the “skip all-reduce for these microbatches, sync on this one” rule in one state machine. Accelerate hides that decision in a context manager so the training loop stays ordinary.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/accelerator.py`](https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/accelerator.py#L1255-L1297)

```python
    def accumulate(self, *models):
        """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            *models (list of `torch.nn.Module`):
                PyTorch Modules that were prepared with `Accelerator.prepare`. Models passed to `accumulate()` will
                skip gradient syncing during backward pass in distributed training

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=1)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, output in dataloader:
        ...     with accelerator.accumulate(model):
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
        ...         optimizer.zero_grad()
        ```
        """
        self._do_sync()

        allow_gradient_sync = (
            self.sync_gradients  # must sync if sync gradients need to complete an optimizer step
            or (
                # the no_sync context stops the gradients from reducing during distributed training
                # bringing speedup (potentially at some costs). Here, no_sync can be prevented
                # by setting sync_each_batch = True.
                self.use_distributed  # only relevant in distributed settings
                and self.gradient_state.plugin_kwargs.get("sync_each_batch", False)
            )
        )
        with contextlib.ExitStack() as cm_stack:
            for m in models:
                cm_stack.enter_context(contextlib.nullcontext() if allow_gradient_sync else self.no_sync(m))
            yield
```

## 逐行讲解 / What's happening

1. **第 1255-1263 行 / Lines 1255-1263 (`context contract`)**:
   - 中文: `accumulate(*models)` 明确接收被 `prepare` 包过的模型，并承诺在累积阶段跳过同步。
   - English: `accumulate(*models)` takes prepared models and promises to skip sync during accumulation windows.
2. **第 1282 行 / Lines 1282 (``_do_sync()``)**:
   - 中文: 进入上下文前先推进内部 step，并更新 `self.sync_gradients`。
   - English: Before entering the context, `_do_sync()` advances the internal step and updates `self.sync_gradients`.
3. **第 1284-1292 行 / Lines 1284-1292 (`sync decision`)**:
   - 中文: 到达同步步，或配置要求每个 batch 都同步时，`allow_gradient_sync` 才为真。
   - English: `allow_gradient_sync` becomes true only on a sync step or when the plugin forces sync on every batch.
4. **第 1294-1297 行 / Lines 1294-1297 (`ExitStack`)**:
   - 中文: 多个 model 会被统一装进 `ExitStack`，每个 model 独立选择 `no_sync` 或空上下文。
   - English: Multiple models enter one `ExitStack`, and each model chooses either `no_sync` or a null context.

## 类比 / The analogy

这像拼车收费站：前几段路只记账不停车，到了结算出口才统一刷卡；但如果规则要求每段都结算，也可以每段都停。

It is like a toll road for a carpool: early segments are recorded without stopping, and the final gate charges once; if policy requires it, every segment can still stop and pay.

## 自己跑一遍 / Try it yourself

```python
class Accumulator:
    def __init__(self, steps):
        self.steps = steps
        self.step = 0
    def enter(self):
        self.step += 1
        sync = self.step % self.steps == 0
        return "sync" if sync else "no_sync"

acc = Accumulator(3)
for _ in range(6):
    print(acc.enter())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
no_sync
no_sync
sync
no_sync
no_sync
sync
```

训练循环外观看起来每个 batch 都 backward，但底层只有第 3、6 个 microbatch 做同步。

The training loop appears to backward every batch, but only microbatches 3 and 6 synchronize underneath.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DeepSpeed accumulation** / **DeepSpeed accumulation**: 中文: DeepSpeed 也把累积步数交给 engine 状态管理。 / English: DeepSpeed also lets engine state own accumulation steps.
- **DDP no_sync** / **DDP no_sync**: 中文: PyTorch DDP 的原语是 `no_sync`；Accelerate 把它包成更高层训练接口。 / English: PyTorch DDP exposes `no_sync`; Accelerate wraps it as a higher-level training interface.

## 注意事项 / Caveats / when it breaks

- **多模型** / **Multiple models**: 中文: 每个传入 model 都要进入同样的同步窗口。 / English: Every passed model must enter the same sync window.
- **sync_each_batch** / **`sync_each_batch`**: 中文: 打开后会放弃累积阶段的 no-sync 省通信效果。 / English: Enabling it gives up the communication savings of no-sync accumulation.

## 延伸阅读 / Further reading

- [huggingface/accelerate source](https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/accelerator.py#L1255-L1297)
