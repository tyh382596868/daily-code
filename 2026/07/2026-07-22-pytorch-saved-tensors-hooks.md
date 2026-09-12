---
date: 2026-07-22
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/autograd/graph.py
permalink: https://github.com/pytorch/pytorch/blob/801c3e6b41427fde03004f2afa4691919d4ee005/torch/autograd/graph.py#L269-L348
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, pytorch, autograd, saved-tensors-hooks]
---

# PyTorch saved_tensors_hooks：反向传播保存什么，由你接管 / PyTorch saved_tensors_hooks: Take Control of What Backward Saves

> **一句话 / In one line**: `saved_tensors_hooks` 用一对 `pack` / `unpack` hook 拦截 autograd 保存的中间张量，让你把显存压力换成自定义存储策略。 / `saved_tensors_hooks` intercepts autograd's saved tensors with `pack` / `unpack` hooks, letting you trade GPU memory for a custom storage strategy.

## 为什么重要 / Why this matters

反向传播不是只靠参数和输入，它还要保存前向里的中间结果。大模型训练时，这些 saved tensors 往往就是显存峰值来源。PyTorch 把“保存前怎么打包、用时怎么解包”暴露成上下文管理器，使 offload、压缩、调试都能复用同一个机制。

Backward does not run only from parameters and inputs; it also needs intermediates from forward. In large models, those saved tensors often drive peak memory. PyTorch exposes "how to pack before saving, how to unpack before use" as a context manager, so offloading, compression, and debugging can reuse the same mechanism.

## 代码 / The code

`pytorch/pytorch` — [`torch/autograd/graph.py`](https://github.com/pytorch/pytorch/blob/801c3e6b41427fde03004f2afa4691919d4ee005/torch/autograd/graph.py#L269-L348)

```python
# Simplified teaching slice, preserving the public contract.
class saved_tensors_hooks:
    def __init__(self, pack_hook, unpack_hook) -> None:
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        torch._C._autograd._pop_saved_tensors_default_hooks()
```

## 逐行讲解 / What's happening

1. **两个 hook 是一组 / The two hooks are a pair**: 中文: `pack_hook(tensor)` 的输出必须能被 `unpack_hook(obj)` 还原成等值张量。 English: whatever `pack_hook(tensor)` returns must be restorable by `unpack_hook(obj)`.
2. **`__enter__` push 到 C++ autograd 栈 / `__enter__` pushes into the C++ autograd stack**: 中文: 真正拦截保存动作的是 autograd 引擎，不是 Python wrapper 自己。 English: the real interception happens in the autograd engine, not in the Python wrapper itself.
3. **`__exit__` pop 回旧状态 / `__exit__` restores the previous state**: 中文: 上下文结束后，后续 forward 不再使用这对 hook。 English: after the context exits, later forwards stop using this hook pair.
4. **只允许最内层生效 / Only the innermost pair applies**: 中文: 嵌套时，PyTorch 选择当前最内层策略，避免多个 pack 策略互相套娃。 English: when nested, PyTorch uses the innermost policy to avoid stacking incompatible packing schemes.

## 类比 / The analogy

像搬家公司的临时仓库。`pack_hook` 决定家具是原样放仓库、拆成零件、还是拍照登记；`unpack_hook` 必须在搬回家时还原出能用的家具。

It is like a moving company's temporary storage. `pack_hook` decides whether furniture is stored as-is, disassembled, or cataloged; `unpack_hook` must reconstruct usable furniture when it comes back.

## 自己跑一遍 / Try it yourself

```python
saved = []

def pack(x):
    saved.append(("pack", x))
    return ("cpu-copy", x)

def unpack(obj):
    tag, x = obj
    saved.append(("unpack", x))
    return x

payload = pack([1, 2, 3])
print(unpack(payload))
print(saved)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 2, 3]
[('pack', [1, 2, 3]), ('unpack', [1, 2, 3])]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`save_on_cpu`** / **`save_on_cpu`**: 中文: 它就是基于同一 hook 机制，把 saved tensors 放到 CPU。 / English: it builds on the same hook mechanism to move saved tensors to CPU.
- **activation checkpointing** / **activation checkpointing**: 中文: 也是在“保存中间值”和“重算中间值”之间做内存权衡。 / English: it also trades between saving intermediates and recomputing them.
- **offload optimizers** / **offload optimizers**: 中文: 参数、梯度和中间激活都可以用类似的 pack/unpack 心智模型分析。 / English: parameters, gradients, and activations can all be analyzed with a similar pack/unpack mental model.

## 注意事项 / Caveats / when it breaks

- **不要返回原 tensor 引用 / Do not return the original tensor reference**: 这会形成引用环；官方建议用 `detach()` 或外部存储。 / That can create reference cycles; the official guidance is to use `detach()` or external storage.
- **不要原地改 hook 输入 / Do not mutate hook inputs in-place**: autograd 期望解包后数值、shape、dtype、device 仍能匹配原张量。 / autograd expects the unpacked value, shape, dtype, and device to match the original tensor.
- **调试时先小图验证 / Validate on a small graph first**: 错误的 unpack 可能到 backward 才爆。 / A wrong unpack path may not fail until backward.

## 延伸阅读 / Further reading

- [PyTorch graph.py saved_tensors_hooks](https://github.com/pytorch/pytorch/blob/801c3e6b41427fde03004f2afa4691919d4ee005/torch/autograd/graph.py#L269-L348)
- [PyTorch autograd graph docs](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks)

