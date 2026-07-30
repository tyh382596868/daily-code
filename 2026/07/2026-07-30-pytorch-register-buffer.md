---
date: 2026-07-30
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/module.py
permalink: https://github.com/pytorch/pytorch/blob/21f65597612350dfe196739da0c4aa334b2d4f14/torch/nn/modules/module.py#L528-L590
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, module-state, buffer]
---

# PyTorch register_buffer：状态不是参数也要被模块管理 / PyTorch register_buffer: State Without Parameters Still Belongs to the Module

> **一句话 / In one line**: `register_buffer` 把非参数 Tensor 纳入 module 状态，并用 `persistent=False` 控制它是否进入 `state_dict`。 / `register_buffer` puts non-parameter tensors under module state and uses `persistent=False` to decide whether they enter `state_dict`.

## 为什么重要 / Why this matters

很多模型状态不该训练，却必须跟着 `.to(device)`、`.cuda()` 和 checkpoint 管理走，例如 BatchNorm running mean、RoPE cache、mask cache。`register_buffer` 正是给这类状态用的窄门。

Many model states should not be trained but must still follow `.to(device)`, `.cuda()`, and checkpoint handling: BatchNorm running means, RoPE caches, mask caches. `register_buffer` is the narrow API for that kind of state.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/module.py`](https://github.com/pytorch/pytorch/blob/21f65597612350dfe196739da0c4aa334b2d4f14/torch/nn/modules/module.py#L528-L590)

```python
    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(
                f"buffer name should be a string. Got {torch.typename(name)}"
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not (
            isinstance(tensor, torch.Tensor) or hasattr(tensor, "__torch_function__")
        ):
            raise TypeError(
                f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
                "(torch Tensor or None required)"
            )
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)
```

## 逐行讲解 / What's happening

1. **第 559-560 行 / Lines 559-560 (ScriptModule guard)**:
   - 中文: TorchScript 不支持 non-persistent buffer，因此这里提前拒绝。
   - English: TorchScript does not support non-persistent buffers, so this path rejects them early.
2. **第 562-573 行 / Lines 562-573 (name checks)**:
   - 中文: buffer 名字不能绕过 `Module.__init__`，不能含点号，也不能覆盖已有非 buffer 属性。
   - English: Buffer names cannot bypass `Module.__init__`, contain dots, or overwrite an existing non-buffer attribute.
3. **第 574-580 行 / Lines 574-580 (type checks)**:
   - 中文: 值必须是 Tensor、`None`，或实现了 `__torch_function__` 的张量类对象。
   - English: The value must be a Tensor, `None`, or a tensor-like object implementing `__torch_function__`.
4. **第 582-590 行 / Lines 582-590 (storage and persistence)**:
   - 中文: 真正的注册只做两件事：写入 `_buffers`，再维护 `_non_persistent_buffers_set`。
   - English: The actual registration does two things: store into `_buffers`, then update `_non_persistent_buffers_set`.

## 类比 / The analogy

这像办公室里的白板：它不是员工薪资表里的“人”，但搬办公室时必须一起搬；有些白板内容还不需要归档。

It is like a whiteboard in an office: it is not an employee in payroll, but it must move with the office; some whiteboard contents do not need archiving.

## 自己跑一遍 / Try it yourself

```python
class TinyModule:
    def __init__(self):
        self._buffers = {}
        self._non_persistent_buffers_set = set()
    def register_buffer(self, name, value, persistent=True):
        if '.' in name or not name:
            raise KeyError('bad buffer name')
        self._buffers[name] = value
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)
    def state_dict(self):
        return {k:v for k,v in self._buffers.items() if k not in self._non_persistent_buffers_set}

m = TinyModule()
m.register_buffer('running_mean', [0.0, 1.0])
m.register_buffer('rope_cache', [42], persistent=False)
print(m._buffers)
print(m.state_dict())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'running_mean': [0.0, 1.0], 'rope_cache': [42]}
{'running_mean': [0.0, 1.0]}
```

`rope_cache` 被 module 管理，但不会进入 `state_dict`。这就是 persistent flag 的核心效果。

`rope_cache` is managed by the module but does not enter `state_dict`. That is the core effect of the persistent flag.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **BatchNorm buffers** / **BatchNorm buffers**: `running_mean` 和 `running_var` 不参与梯度，但必须保存。 / `running_mean` and `running_var` do not train through gradients, but must be saved.
- **RoPE caches** / **RoPE caches**: 预计算 cos/sin 表经常是 non-persistent buffer。 / Precomputed cos/sin tables are often non-persistent buffers.

## 注意事项 / Caveats / when it breaks

- **不要放可训练权重** / **Do not store trainable weights here**: 需要 optimizer 更新的 Tensor 应该是 `Parameter`。 / Tensors updated by the optimizer should be `Parameter`s.
- **名字是 API** / **Names are API**: buffer 名会进入 `state_dict` key，改名会影响 checkpoint 兼容。 / Buffer names become `state_dict` keys; renaming affects checkpoint compatibility.

## 延伸阅读 / Further reading

- [PyTorch Module source](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py)
- [Source permalink](https://github.com/pytorch/pytorch/blob/21f65597612350dfe196739da0c4aa334b2d4f14/torch/nn/modules/module.py#L528-L590)
