---
date: 2026-10-13
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/backend_registration.py
permalink: https://github.com/pytorch/pytorch/blob/41b4c95e20fc641ec83037111acb05bf011dfbb4/torch/utils/backend_registration.py#L190-L274
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, privateuse1, backend-registration]
---

# PyTorch PrivateUse1：给自定义设备补上 `.foo()` 方法 / PyTorch PrivateUse1: Add `.foo()` Methods for a Custom Device

> **一句话 / In one line**: PyTorch 会给 PrivateUse1 后端动态挂上 `Tensor.foo()`、`Tensor.is_foo` 和 `Module.foo()`。 / PyTorch dynamically attaches `Tensor.foo()`, `Tensor.is_foo`, and `Module.foo()` for a PrivateUse1 backend.

## 为什么重要 / Why this matters

新硬件后端需要的不只是 kernel，还要用户能像调用 `.cuda()` 一样把 tensor 和 module 搬到设备上。

A new hardware backend needs more than kernels; users need to move tensors and modules as naturally as calling `.cuda()`.

## 代码 / The code

`pytorch/pytorch` - [`torch/utils/backend_registration.py`](https://github.com/pytorch/pytorch/blob/41b4c95e20fc641ec83037111acb05bf011dfbb4/torch/utils/backend_registration.py#L190-L274)

```python
def _generate_tensor_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.Tensor) -> bool:
        if has_torch_function_unary(self):
            # TODO mypy doesn't support @property, see: https://github.com/python/mypy/issues/6185
            return handle_torch_function(wrap_tensor_backend.__get__, (self,), self)  # type: ignore[attr-defined]
        return self.device.type == custom_backend_name

    _check_register_once(torch.Tensor, f"is_{custom_backend_name}")
    wrap_tensor_backend.fget.__name__ = f"is_{custom_backend_name}"  # type: ignore[attr-defined]
    setattr(torch.Tensor, f"is_{custom_backend_name}", wrap_tensor_backend)

    def wrap_tensor_to(
        self: torch.Tensor,
        device: int | torch.device | None = None,
        non_blocking=False,
        **kwargs,
    ) -> torch.Tensor:
        r"""Perform Tensor device conversion. Call the to operator implementation.

        .. note::
            If the ``self`` Tensor already
            has the correct :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`torch.device`.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                wrap_tensor_to,
                (self,),
                self,
                device=device,
                non_blocking=False,
                **kwargs,
            )
        device_idx = _normalization_device(custom_backend_name, device)
        return self.to(
            device=torch.device(f"{custom_backend_name}:{device_idx}"),
            non_blocking=non_blocking,
            **kwargs,
        )

    _check_register_once(torch.Tensor, custom_backend_name)
    wrap_tensor_to.__name__ = custom_backend_name
    setattr(torch.Tensor, custom_backend_name, wrap_tensor_to)


def _generate_module_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # Generate Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(torch.Tensor, custom_backend_name):
        raise RuntimeError(
            f"Can not automatically generate {custom_backend_name}() method for torch.nn.Module."
            f" Because torch.Tensor doesn't have the method {custom_backend_name}()."
            f" For this error, you can try setting for_tensor=True."
        )

    def wrap_module_to(
        # pyrefly: ignore [invalid-type-var]
        self: torch.nn.modules.module.T,
        device: int | torch.device | None = None,
    ) -> torch.nn.modules.module.T:  # pyrefly: ignore [invalid-type-var]
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        # pyrefly: ignore [missing-attribute]
        return self._apply(lambda t: getattr(t, custom_backend_name)(device))

    _check_register_once(torch.nn.Module, custom_backend_name)
    setattr(torch.nn.Module, custom_backend_name, wrap_module_to)
```

## 逐行讲解 / What's happening

1. **第 190-200 行 / Lines 190-200 (`is_foo`)**:
   - 中文: 动态生成 property，检查 `device.type`。
   - English: It dynamically creates a property that checks `device.type`.
2. **第 202-240 行 / Lines 202-240 (`foo()` tensor move)**:
   - 中文: 把设备编号标准化成 `foo:N`，再委托给 `Tensor.to`。
   - English: It normalizes the device index into `foo:N`, then delegates to `Tensor.to`.
3. **第 243-274 行 / Lines 243-274 (`Module.foo()`)**:
   - 中文: module 方法依赖 tensor 方法，避免半注册状态。
   - English: The module method depends on tensor methods, avoiding half-registered state.

## 类比 / The analogy

这像给新机场加地铁站牌：机场已经建好，但没有站牌乘客就不知道怎么去。

It is like adding subway signs for a new airport: the airport may exist, but passengers need a route to reach it.

## 自己跑一遍 / Try it yourself

```python
class Tensor:
    def __init__(self, device="cpu"):
        self.device = device
    def to(self, device):
        return Tensor(device)

def generate(name):
    setattr(Tensor, f"is_{name}", property(lambda self: self.device.startswith(name)))
    setattr(Tensor, name, lambda self, i=0: self.to(f"{name}:{i}"))

generate("foo")
x = Tensor().foo(2)
print(x.device, x.is_foo)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
foo:2 True
```

中文: 这里复用现有对象，只是把新设备入口挂上去。

English: This reuses existing objects and adds a new device entry point.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Dynamo backend registry** / **Dynamo backend registry**: Dynamo backend registry。 / Dynamo backend registry.
- **Device modules** / **Device modules**: Device modules。 / Device modules.

## 注意事项 / Caveats / when it breaks

- **只能注册一次** / **Registration must not overwrite existing methods**: 只能注册一次。 / Registration must not overwrite existing methods.
- **kernel 仍然必须存在** / **The Python method does not replace real backend kernels**: kernel 仍然必须存在。 / The Python method does not replace real backend kernels.

## 延伸阅读 / Further reading
- [PyTorch backend registration](https://github.com/pytorch/pytorch/blob/41b4c95e20fc641ec83037111acb05bf011dfbb4/torch/utils/backend_registration.py#L190-L274)
