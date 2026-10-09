---
date: 2026-10-09
topic: diffusion
source: trending
repo: AlibabaResearch/SparkDiffusion
file: sparkdiffusion/networks/sparse_attn_registry.py
permalink: https://github.com/AlibabaResearch/SparkDiffusion/blob/6149ac5873fbb1283906bbe13d05b984042eed56/sparkdiffusion/networks/sparse_attn_registry.py#L36-L67
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, sparse-attention-registry]
---
# SparkDiffusion sparse registry：让 attention 变体插件化 / SparkDiffusion Sparse Registry: Make Attention Variants Pluggable

> **一句话 / In one line**: 一个装饰器把 sparse attention 类注册成字符串 key，并额外记录哪些新参数需要在微调阶段解冻。 / A decorator registers sparse-attention classes by string key and records which new parameters should stay trainable.

## 为什么重要 / Why this matters

做视频 DiT 加速时，attention 变体很容易把主干代码改成一堆 `if/else`。SparkDiffusion 这里用了一个很小的 registry：核心网络只按名字取类，具体 sparse attention 在自己的模块里注册，同时声明 Stage-1 微调要训练的新参数名。

Video DiT acceleration often grows into a mess of attention-specific branches. SparkDiffusion uses a tiny registry instead. The core network looks up a class by name; each sparse-attention variant registers itself and declares which new parameter names should remain trainable during stage-one finetuning.

## 代码 / The code

`AlibabaResearch/SparkDiffusion` — [`sparkdiffusion/networks/sparse_attn_registry.py`](https://github.com/AlibabaResearch/SparkDiffusion/blob/6149ac5873fbb1283906bbe13d05b984042eed56/sparkdiffusion/networks/sparse_attn_registry.py#L36-L67)

```python
def register_sparse_attn(name: str, sparse_param_names: Optional[List[str]] = None):
    """Register a sparse-attention class into the global registry.

    Args:
        name: the string key referenced via attn_variant in an experiment.
        sparse_param_names: trainable param names added by this variant (substring
            match). Finetuning Stage-1 adds these to the freeze whitelist so the
            custom params are not frozen. Optional.
    """
    def _decorator(cls):
        if name in SPARSE_ATTN_REGISTRY and SPARSE_ATTN_REGISTRY[name] is not cls:
            raise ValueError(f"sparse attention variant '{name}' is already registered as {SPARSE_ATTN_REGISTRY[name]}")
        SPARSE_ATTN_REGISTRY[name] = cls
        SPARSE_ATTN_PARAM_NAMES[name] = list(sparse_param_names or [])
        return cls
    return _decorator


def get_sparse_attn_class(name: str) -> Type:
    """Return the registered sparse-attention class by name; raise a clear error if unregistered."""
    if name not in SPARSE_ATTN_REGISTRY:
        raise KeyError(
            f"unknown attn_variant='{name}'. Registered: {sorted(SPARSE_ATTN_REGISTRY)}. "
            f"Register your sparse-attention class with @register_sparse_attn('{name}') "
            f"and make sure its module is imported (see the sparkdiffusion/networks/sparse_attn_registry.py docstring)."
        )
    return SPARSE_ATTN_REGISTRY[name]


def get_sparse_param_names(name: str) -> List[str]:
    """Return the trainable sparse param names declared by a variant (empty list if none)."""
    return list(SPARSE_ATTN_PARAM_NAMES.get(name, []))
```

## 逐行讲解 / What's happening

1. **第 29-33 行 / Lines 29-33 (global tables)**:
   - 中文: 一个表存名字到类，另一个表存这个变体新增的可训练参数名。
   - English: One table maps names to classes; the other stores trainable parameter-name hints.
2. **第 36-51 行 / Lines 36-51 (`register_sparse_attn`)**:
   - 中文: 装饰器检查重复注册，写入两个表，然后原样返回类。
   - English: The decorator checks duplicate registration, fills both tables, and returns the class unchanged.
3. **第 54-62 行 / Lines 54-62 (`get_sparse_attn_class`)**:
   - 中文: 未注册时报一个带操作建议的错误，而不是让后面构造模型时神秘失败。
   - English: Missing registrations raise an actionable error instead of failing mysteriously during model construction.

## 类比 / The analogy

像工具箱里的标签抽屉。你把“短螺丝刀”贴到某个格子上，之后别人只要说标签名就能取到工具；旁边还贴着“这把工具需要单独保养”的说明。

It is like labeled drawers in a toolbox. Registering “short screwdriver” puts the tool behind that label, and a side note says which parts need special maintenance.

## 自己跑一遍 / Try it yourself

```python
REGISTRY = {}
PARAMS = {}

def register(name, params=None):
    def deco(cls):
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise ValueError(f"{name} already registered")
        REGISTRY[name] = cls
        PARAMS[name] = list(params or [])
        return cls
    return deco

@register("window", ["window_q", "window_k"])
class WindowAttention: pass

print(REGISTRY["window"].__name__)
print(PARAMS["window"])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
WindowAttention
['window_q', 'window_k']
```

装饰器没有改变类本身，只是把它放进可查询的表里。

The decorator does not modify the class; it just places it in a lookup table.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch backend registry** / **PyTorch backend registries**: 字符串 key 到 callable 的映射能隔离核心框架和扩展实现。 / String-to-callable registries separate core frameworks from extensions.
- **LoRA / adapter dispatch** / **LoRA / adapter dispatch**: 变体注册也常用于按配置选择 adapter 类型。 / Variant registries are also common when selecting adapter implementations by config.

## 注意事项 / Caveats / when it breaks

- **模块必须被 import** / **Modules must be imported**: 只有执行过装饰器的模块才会注册。 / Registration happens only after the module containing the decorator is imported.
- **全局表要避免名字冲突** / **Global names can collide**: 代码只允许同名同类重复注册，不允许同名不同类。 / The code permits re-registering the same class, but rejects a different class under the same name.

## 延伸阅读 / Further reading

- [SparkDiffusion sparse attention registry](https://github.com/AlibabaResearch/SparkDiffusion/blob/6149ac5873fbb1283906bbe13d05b984042eed56/sparkdiffusion/networks/sparse_attn_registry.py#L36-L67)
- [SparkDiffusion repository](https://github.com/AlibabaResearch/SparkDiffusion)
