---
date: 2026-07-20
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/lora/inc.py
permalink: https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/tuners/lora/inc.py#L29-L75
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, lora, dispatch]
---

# PEFT INC LoRA：只给支持的量化 Linear 换适配层 / PEFT INC LoRA: Dispatch Only Supported Quantized Linear Layers

> **一句话 / In one line**: PEFT 的 INC dispatch 先剥出 base layer，只有遇到 `PatchedLinear` 才包成 `IncLoraLinear`，并明确拒绝尚未实现的 merge/unmerge。 / PEFT's INC dispatch unwraps the base layer, wraps only `PatchedLinear`, and explicitly rejects unimplemented merge/unmerge paths.

## 为什么重要 / Why this matters

LoRA 注入不是“看到 Linear 就替换”这么简单。量化后端可能有自己的 Linear 包装，权重格式、merge 语义和 kernel 都不同。PEFT 用 dispatch 函数把后端识别和适配层创建隔离开：能安全处理就返回新模块，不能处理就交给下一个 dispatcher。

LoRA injection is not just "replace every Linear". Quantization backends can wrap Linear with different weight formats, merge semantics, and kernels. PEFT isolates backend recognition in dispatch functions: return a new module only when safe, otherwise let the next dispatcher try.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/lora/inc.py`](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/tuners/lora/inc.py#L29-L75)

```python
if is_inc_available():

    class IncLoraLinear(Linear):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            config: LoraConfig,
            **kwargs,
        ):
            super().__init__(base_layer, adapter_name, config=config, **kwargs)

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            raise NotImplementedError("Merging LoRA with INC layers is not yet implemented")

        def unmerge(self) -> None:
            raise NotImplementedError("Unmerging LoRA from INC layers is not yet implemented")


def dispatch_inc(target: torch.nn.Module, adapter_name: str, config: LoraConfig, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_inc_available():
        from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import (
            PatchedLinear,
        )

        if isinstance(target_base_layer, PatchedLinear):
            new_module = IncLoraLinear(target, adapter_name, config=config, **kwargs)

    return new_module
```

## 逐行讲解 / What's happening

1. **依赖存在才定义类 / Define the class only when dependency exists**: 中文: 没装 INC 时不提前 import 后端模块。 English: without INC installed, backend modules are not imported.
2. **继承通用 LoRA Linear / Reuse generic LoRA Linear**: 中文: `IncLoraLinear` 复用 PEFT 已有的 LoRA 参数和 forward 逻辑。 English: `IncLoraLinear` reuses PEFT's existing LoRA parameters and forward logic.
3. **merge/unmerge 明确报错 / Merge and unmerge fail explicitly**: 中文: 量化权重合并语义没实现时，宁可早报错。 English: when quantized weight merging is undefined, failing early is safer.
4. **先剥 base layer / Unwrap the base layer first**: 中文: 如果目标已经是 tuner layer，就取里面真正的底层模块。 English: if the target is already a tuner layer, inspect the true wrapped module.
5. **只识别 PatchedLinear / Recognize only PatchedLinear**: 中文: dispatcher 命中才返回新模块，否则返回 `None`。 English: the dispatcher returns a module only on a `PatchedLinear` hit; otherwise it returns `None`.

## 类比 / The analogy

像给相机配镜头转接环。普通镜头能直接装，特殊卡口要专用转接环；如果还不知道这个卡口能不能安全锁紧，就不要硬拧上去。

It is like mounting lenses on a camera. Normal lenses fit directly, special mounts need adapters, and if the adapter cannot lock safely, you should not force it.

## 自己跑一遍 / Try it yourself

```python
class BaseTunerLayer:
    def __init__(self, base): self.base = base
    def get_base_layer(self): return self.base
class PatchedLinear: pass
class IncLoraLinear:
    def __init__(self, target): self.target = target

def dispatch(target):
    base = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
    return IncLoraLinear(target) if isinstance(base, PatchedLinear) else None

print(type(dispatch(BaseTunerLayer(PatchedLinear()))).__name__)
print(dispatch(object()))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
IncLoraLinear
None
```

返回 `None` 不是失败，而是让 PEFT 的其他 dispatcher 继续尝试。

Returning `None` is not a failure; it lets PEFT's other dispatchers try.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PEFT bitsandbytes dispatch** / **PEFT bitsandbytes dispatch**: 不同量化 Linear 走不同 LoRA 包装类。 / Different quantized Linear classes route to different LoRA wrappers.
- **Transformers AutoClass** / **Transformers AutoClass**: 先识别配置/类型，再派发到具体实现。 / Identify config or type first, then dispatch to a concrete implementation.

## 注意事项 / Caveats / when it breaks

- **不要假设 merge 可用 / Do not assume merge is available**: INC LoRA 这里显式不支持 merge/unmerge。 / INC LoRA explicitly does not support merge or unmerge here.
- **后端 import 要延迟 / Backend imports should be lazy**: 否则没装 INC 的用户也会被 import error 卡住。 / Otherwise users without INC would fail at import time.
- **BaseTunerLayer 要解包 / BaseTunerLayer must be unwrapped**: 二次注入时看 wrapper 类型会误判。 / During repeated injection, inspecting the wrapper type can misroute.

## 延伸阅读 / Further reading

- [PEFT inc.py](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/tuners/lora/inc.py#L29-L75)
- [PEFT repository](https://github.com/huggingface/peft)
