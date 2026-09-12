---
date: 2026-07-20
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/lazy.py
permalink: https://github.com/pytorch/pytorch/blob/2f86d10bc3a0c71a4972c900e48e900ed7a0a26b/torch/nn/modules/lazy.py#L252-L275
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, lazy-module, hooks, materialization]
---

# PyTorch LazyModule：第一次 forward 后把自己变成真模块 / PyTorch LazyModule: Become a Real Module After the First Forward

> **一句话 / In one line**: `LazyModuleMixin` 用 forward pre-hook 根据输入 materialize 参数，然后移除 hook，并可把 `__class__` 切换成最终模块类。 / `LazyModuleMixin` uses a forward pre-hook to materialize parameters from input shape, removes the hooks, and can switch `__class__` to the final module type.

## 为什么重要 / Why this matters

Lazy layer 让你不用提前写 `in_features`，但这只是构造阶段的便利。第一次看到真实输入后，模块必须尽快变成普通模块，否则 optimizer、state dict、DataParallel 都会遇到“不知道形状”的参数。PyTorch 把这个转换做成一次性 hook。

Lazy layers let you skip `in_features` at construction time, but that convenience cannot last forever. After the first real input, the module must become an ordinary module so optimizers, state dicts, and parallel wrappers see real parameters. PyTorch implements that transition as a one-shot hook.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/lazy.py`](https://github.com/pytorch/pytorch/blob/2f86d10bc3a0c71a4972c900e48e900ed7a0a26b/torch/nn/modules/lazy.py#L252-L275)

```python
def _infer_parameters(self: _LazyProtocol, module, args, kwargs=None):
    r"""Infers the size and initializes the parameters according to the provided input batch.

    Given a module that contains parameters that were declared inferable
    using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
    in the complete module using the provided input to initialize all the parameters
    as needed.
    The module is set into evaluation mode before running the forward pass in order
    to avoid saving statistics or calculating gradients
    """
    kwargs = kwargs if kwargs else {}
    module.initialize_parameters(*args, **kwargs)
    if module.has_uninitialized_params():
        raise RuntimeError(f'module {self._get_name()} has not been fully initialized')
    module._initialize_hook.remove()
    module._load_hook.remove()
    delattr(module, '_initialize_hook')
    delattr(module, '_load_hook')
    if module.cls_to_become is not None:
        module.__class__ = module.cls_to_become
```

## 逐行讲解 / What's happening

1. **kwargs 归一化 / Normalize kwargs**: 中文: hook 可能收到 `None`，先变成空 dict。 English: the hook may receive `None`, so it is normalized to an empty dict.
2. **用输入推形状 / Infer shape from input**: 中文: `initialize_parameters` 是子类实现的真正 materialize 入口。 English: `initialize_parameters` is the subclass hook that actually materializes parameters.
3. **立刻检查残留 lazy 参数 / Check for leftover lazy parameters**: 中文: 还有未初始化参数就报错，不让半初始化模块继续跑。 English: if any lazy parameter remains, the module fails instead of running half-initialized.
4. **移除一次性 hook / Remove one-shot hooks**: 中文: 初始化和 load hook 都不再需要。 English: both initialize and load hooks are no longer needed.
5. **切换类身份 / Switch class identity**: 中文: `LazyLinear` 可以在完成后表现成 `Linear`。 English: `LazyLinear` can behave as `Linear` after initialization.

## 类比 / The analogy

像买一套可伸缩书架。搬家前它不知道墙宽；第一次贴墙安装时量尺寸、固定螺丝，然后它就不再是“可伸缩状态”，而是一张普通书架。

It is like buying an adjustable bookshelf. Before moving in it does not know the wall width; during first installation it measures, locks the screws, and becomes a normal fixed shelf.

## 自己跑一遍 / Try it yourself

```python
class LazyBox:
    cls_to_become = None
    def __init__(self):
        self.weight = None
        self.hook = True
    def initialize_parameters(self, x):
        self.weight = [0] * len(x)
    def has_uninitialized_params(self):
        return self.weight is None
    def infer(self, x):
        self.initialize_parameters(x)
        if self.has_uninitialized_params():
            raise RuntimeError("not initialized")
        self.hook = False

box = LazyBox()
print(box.weight, box.hook)
box.infer([1, 2, 3])
print(box.weight, box.hook)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
None True
[0, 0, 0] False
```

第一次输入之后，参数从未知变成具体长度，hook 也被关掉。

After the first input, the parameter becomes concrete and the hook is disabled.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Keras build-on-call layers** / **Keras build-on-call layers**: 先拿输入 shape 再创建权重。 / Weights are created after the first input shape is known.
- **JAX lazy init wrappers** / **JAX lazy init wrappers**: 用 shape-only 或 dummy input 初始化参数树。 / Shape-only or dummy inputs initialize parameter trees.

## 注意事项 / Caveats / when it breaks

- **optimizer 要等 dry run / Optimizer must wait for dry run**: 参数没 materialize 前不要创建 optimizer。 / Do not create the optimizer before parameters are materialized.
- **DataParallel 不能吃 lazy 参数 / DataParallel cannot consume lazy params**: 多卡复制需要真实 shape。 / Multi-device replication needs real shapes.
- **随机初始化顺序会变 / Random init order can change**: lazy 层延后初始化会影响 RNG 消耗顺序。 / Delayed initialization changes the RNG consumption order.

## 延伸阅读 / Further reading

- [PyTorch lazy.py](https://github.com/pytorch/pytorch/blob/2f86d10bc3a0c71a4972c900e48e900ed7a0a26b/torch/nn/modules/lazy.py#L252-L275)
- [PyTorch Lazy Modules docs](https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html)
