---
date: 2026-07-30
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/experimental/utils.py
permalink: https://github.com/huggingface/trl/blob/185f73824abf332b5576afe0d14abe7f30e8fc03/trl/experimental/utils.py#L972-L1049
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, trl, reference-model]
---

# TRL reference model：冻结一份策略作比较 / TRL Reference Model: Freeze a Policy for Comparison

> **一句话 / In one line**: `create_reference_model` 为 DPO/GRPO 这类算法创建静态 reference model，并可让前几层共享存储来省显存。 / `create_reference_model` creates a static reference model for algorithms like DPO/GRPO and can share early-layer storage to save memory.

## 为什么重要 / Why this matters

偏好优化不只看当前策略，还要和一个不动的 reference 比。最朴素做法是 deepcopy 全模型；TRL 这段代码再进一步，允许把底层共享且冻结，只复制后面的可比较部分。

Preference optimization compares the current policy against a fixed reference. The naive solution is a full deepcopy; this TRL code goes further by optionally sharing and freezing lower layers while copying the comparable upper part.

## 代码 / The code

`huggingface/trl` — [`trl/experimental/utils.py`](https://github.com/huggingface/trl/blob/185f73824abf332b5576afe0d14abe7f30e8fc03/trl/experimental/utils.py#L972-L1049)

```python
def create_reference_model(
    model: nn.Module, num_shared_layers: int | None = None, pattern: str | None = None
) -> nn.Module:
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model ([`nn.Module`]): The model to be copied.
        num_shared_layers (`int`, *optional*):
            The number of initial layers that are shared between both models and kept frozen. Shared layers reference
            the same storage as the source model, so they are not duplicated in memory.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns:
        [`nn.Module`]
    """
    if is_deepspeed_zero3_enabled():
        raise ValueError(
            "DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoModelForCausalLM.from_pretrained()`."
        )

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any(pattern_candidate in name for name in parameter_names):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, _param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)

    # Freeze the shared layers in the source model, then point the reference parameter at the same
    # storage instead of keeping the `deepcopy` duplicate. The shared (frozen) layers are thus held in
    # memory only once; because they are frozen in the model, the reference stays static during training.
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False

        ref_param = ref_model.get_parameter(param_name)
        ref_param.data = param.data
        ref_param.requires_grad = False

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    return ref_model.eval()
```

## 逐行讲解 / What's happening

1. **第 989-992 行 / Lines 989-992 (ZeRO-3 guard)**:
   - 中文: ZeRO-3 会分片参数，`deepcopy` 加共享存储的假设不成立，所以直接拒绝。
   - English: ZeRO-3 shards parameters, breaking the assumptions behind `deepcopy` plus storage sharing, so the function rejects it.
2. **第 994-1002 行 / Lines 994-1002 (full copy path)**:
   - 中文: 没有共享层时，复制整模型并把 reference 参数全部 `requires_grad=False`。
   - English: Without shared layers, it copies the full model and freezes every reference parameter.
3. **第 1004-1015 行 / Lines 1004-1015 (pattern discovery)**:
   - 中文: 共享边界不是硬编码架构，而是用 layer-name pattern 找到第 N 层。
   - English: The sharing boundary is not hard-coded to one architecture; a layer-name pattern identifies the Nth layer.
4. **第 1030-1039 行 / Lines 1030-1039 (storage sharing)**:
   - 中文: 共享参数在 source 和 reference 之间指向同一份 `.data`，同时两边都冻结。
   - English: Shared parameters point to the same `.data` in source and reference, and both sides are frozen.

## 类比 / The analogy

这像考试时保留一份标准答案：前几章教材完全一样就共用同一本，后几章才各自复印一份。

It is like keeping an answer key for an exam: if the first chapters are identical, share one textbook; only photocopy the later chapters separately.

## 自己跑一遍 / Try it yourself

```python
class Param:
    def __init__(self, value): self.data, self.requires_grad = value, True

source = {'layer0.w': Param([1]), 'layer1.w': Param([2]), 'head.w': Param([3])}
ref = {k: Param(v.data.copy()) for k, v in source.items()}
pattern = 'layer1'
sharing = True
for name in source:
    if pattern in name:
        sharing = False
    if sharing:
        source[name].requires_grad = False
        ref[name].data = source[name].data
    ref[name].requires_grad = False

source['layer0.w'].data.append(99)
print(ref['layer0.w'].data, ref['head.w'].data)
print([p.requires_grad for p in ref.values()])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 99] [3]
[False, False, False]
```

共享层真的共用同一份 data；reference 的所有参数都冻结。

The shared layer really uses the same data, and every reference parameter is frozen.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DPO reference policy** / **DPO reference policy**: DPO loss 需要当前 policy 和 reference policy 的 log-probs。 / DPO loss needs log-probs from both the current policy and the reference policy.
- **EMA teacher** / **EMA teacher**: 另一个静态/慢更新目标模型也常用于稳定训练。 / Another static or slowly updated target model is often used to stabilize training.

## 注意事项 / Caveats / when it breaks

- **共享层必须冻结** / **Shared layers must be frozen**: 如果 source 共享层还训练，reference 就不再静态。 / If source shared layers keep training, the reference is no longer static.
- **名称 pattern 很脆** / **Name patterns are brittle**: 模型改了 layer 命名，自动匹配可能找不到边界。 / If a model changes layer names, automatic matching may fail.

## 延伸阅读 / Further reading

- [TRL repository](https://github.com/huggingface/trl)
- [Source permalink](https://github.com/huggingface/trl/blob/185f73824abf332b5576afe0d14abe7f30e8fc03/trl/experimental/utils.py#L972-L1049)
