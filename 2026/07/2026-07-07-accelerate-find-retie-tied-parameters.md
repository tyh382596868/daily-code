---
date: 2026-07-07
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/utils/modeling.py
permalink: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py#L568-L627
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, tied-parameters]
---

# Accelerate tied parameters：先找共享权重，再把断掉的引用接回去 / Accelerate Tied Parameters: Find Shared Weights, Then Retie Broken References

> **一句话 / In one line**: Accelerate 用 `named_parameters(remove_duplicate=False/True)` 的差集找 tied weights，再用点路径把同一个参数对象重新挂回模型。 / Accelerate finds tied weights by diffing `named_parameters(remove_duplicate=False/True)`, then reattaches the same parameter object by dotted paths.

## 为什么重要 / Why this matters

大模型加载、device map、hook 注入和 meta device 初始化都可能意外打断“两个层共享同一个参数对象”的关系。语言模型里 embedding 和 lm_head 常常 tied，如果引用断了，显存会多一份，训练语义也会变。Accelerate 这段代码就是给模型手术后的“缝合线”。

Large-model loading, device maps, hooks, and meta-device initialization can accidentally break the relationship where two layers share the exact same parameter object. In language models, embeddings and `lm_head` are often tied; if that tie breaks, memory usage and training semantics change. This code is the stitch-up step after model surgery.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/utils/modeling.py`](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py#L568-L627)

```python
def find_tied_parameters(model: torch.nn.Module, **kwargs) -> list[list[str]]:
    all_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=False)}
    no_duplicate_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=True)}

    tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())

    tied_param_groups = {}
    for tied_param_name in tied_param_names:
        tied_param = all_named_parameters[tied_param_name]
        for param_name, param in no_duplicate_named_parameters.items():
            if param is tied_param:
                if param_name not in tied_param_groups:
                    tied_param_groups[param_name] = []
                tied_param_groups[param_name].append(tied_param_name)

    return [sorted([weight] + list(set(tied))) for weight, tied in tied_param_groups.items()]


def retie_parameters(model, tied_params):
    for tied_group in tied_params:
        param_to_tie = None
        for param_name in tied_group:
            module = model
            splits = param_name.split(".")
            for split in splits[:-1]:
                module = getattr(module, split)
            param = getattr(module, splits[-1])
            if param_to_tie is None and param.device != torch.device("meta"):
                param_to_tie = param
                break
        if param_to_tie is not None:
            for param_name in tied_group:
                module = model
                splits = param_name.split(".")
                for split in splits[:-1]:
                    module = getattr(module, split)
                setattr(module, splits[-1], param_to_tie)
```

## 逐行讲解 / What's happening

1. **第 2-3 行 / Lines 2-3 (`remove_duplicate`)**:
   - 中文: 同一组参数遍历两遍：一遍保留重复名字，一遍只保留唯一对象。
   - English: the same model is traversed twice: once keeping duplicate names, once keeping only unique objects.
2. **第 5 行 / Line 5 (`set` difference)**:
   - 中文: 差集就是“名字存在，但因为共享对象被去重掉”的参数名。
   - English: the set difference is the list of names that disappeared because their objects were duplicates.
3. **第 8-14 行 / Lines 8-14 (identity grouping)**:
   - 中文: 用 `param is tied_param` 比对象身份，不比数值。
   - English: grouping uses object identity, `param is tied_param`, not tensor values.
4. **第 20-29 行 / Lines 20-29 (choose anchor)**:
   - 中文: retie 时先找一个不是 meta device 的真实参数作为锚点。
   - English: retie first finds a real non-meta parameter as the anchor.
5. **第 30-36 行 / Lines 30-36 (`setattr`)**:
   - 中文: 顺着点路径走到父模块，再把目标属性设为同一个参数对象。
   - English: it walks the dotted path to the parent module, then sets the target attribute to the same parameter object.

## 类比 / The analogy

像两张门禁卡本来指向同一个办公室。装修后其中一张卡被换成了临时卡，Accelerate 先找出这些“本该同门”的卡，再统一绑定回同一个门锁。

Imagine two access cards that should open the same office. After renovation one card points to a temporary lock. Accelerate finds cards that should belong together and binds them back to the same lock.

## 自己跑一遍 / Try it yourself

```python
class P:
    def __init__(self, name, device="cpu"):
        self.name, self.device = name, device
class M:
    def __init__(self):
        self.a = P("shared")
        self.b = self.a
    def named_parameters(self, remove_duplicate=True):
        seen = set()
        for name in ["a", "b"]:
            p = getattr(self, name)
            if remove_duplicate and id(p) in seen:
                continue
            seen.add(id(p)); yield name, p
m = M()
all_names = dict(m.named_parameters(False))
unique = dict(m.named_parameters(True))
print(set(all_names) - set(unique), m.a is m.b)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'b'} True
```

这个最小例子复现了 `remove_duplicate=False/True` 差集为什么能定位 tied 参数。

This minimal example shows why the `remove_duplicate=False/True` difference identifies tied parameters.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers weight tying** / **Transformers weight tying**: token embedding 和输出头常共享权重。 / Token embeddings and output heads often share weights.
- **FSDP / device-map loading** / **FSDP / device-map loading**: 包装或拆分模块后需要重新确认 tied references。 / After wrapping or sharding modules, tied references need to be checked again.

## 注意事项 / Caveats / when it breaks

- **只适用于同对象共享** / **Only object sharing counts**: 数值相同但对象不同，不会被识别为 tied。 / Equal values with different objects are not considered tied.
- **路径必须可 `getattr`** / **Paths must be `getattr`-walkable**: 非标准容器可能需要额外处理。 / Non-standard containers may need special handling.

## 延伸阅读 / Further reading

- [Accelerate modeling utilities](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py)
