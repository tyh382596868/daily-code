---
date: 2026-10-10
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/compilation/passes/fusion/rope_kvcache_fusion.py
permalink: https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/compilation/passes/fusion/rope_kvcache_fusion.py#L46-L76
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache]
---
# vLLM fused RoPE：用空 tensor 保住 KV cache 副作用顺序 / vLLM Fused RoPE: Preserve KV-Cache Side Effects with an Empty Tensor

> **一句话 / In one line**: 这段 fused op 真正返回的是一个空 tensor，但它把 RoPE 和 KV cache update 的副作用钉进 `torch.compile` 图里。 / This fused op returns an empty tensor, but that empty value anchors RoPE and KV-cache mutation inside the compiled graph.

## 为什么重要 / Why this matters

KV cache 写入是副作用，不是普通纯函数。vLLM 把 layer context、slot mapping 和 cache update 包进 custom op，再返回 dummy tensor 当依赖边，提醒编译器这件事必须发生。

A KV-cache write is a side effect, not a pure function. vLLM wraps layer context, slot mapping, and cache update in a custom op, then returns a dummy tensor as the dependency edge that keeps ordering visible.

## 代码 / The code

`vllm-project/vllm` — [`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/compilation/passes/fusion/rope_kvcache_fusion.py#L46-L76)

```python
def fused_rope_and_unified_kv_cache_update_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """This impl fetches the KV cache and slot mapping from the forward context,
    then calls the layer impl's `AttentionImpl.do_rope_and_kv_cache_update` method.
    It also returns a dummy tensor, similar to `Attention.unified_kv_cache_update`,
    that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_rope_and_kv_cache_update(
            attn_layer,
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )

    return query.new_empty(0)
```

## 逐行讲解 / What's happening

1. **入口 / Entry**:
   - 中文: 先看函数签名和输入，它定义了这个模块承担的边界职责。
   - English: Start from the signature and inputs; they define the module boundary.
2. **核心状态 / Core state**:
   - 中文: 中间变量保存的是工程约束，例如 cache、rank、bin 或时间线。
   - English: The intermediate variables encode engineering constraints such as cache, rank, bins, or timeline state.
3. **返回值 / Return value**:
   - 中文: 输出不是孤立结果，而是给下游模块继续消费的契约。
   - English: The output is not an isolated result; it is the contract consumed downstream.

## 类比 / The analogy

像仓库扫码入库：扫码枪没有把货物带回来，但那一声记录让后面的出库流程知道货物已经到位。

It is like scanning boxes into a warehouse. The scanner does not return the box, but the record tells shipping that the box has arrived.

## 自己跑一遍 / Try it yourself

```python
def fused_update(key, value, slots, cache):
    if slots is not None:
        for i, slot in enumerate(slots):
            cache[slot] = (key[i] + "|rope", value[i])
    return []
cache = {}
print(fused_update(["k0"], ["v0"], [3], cache))
print(cache)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[]
{3: ('k0|rope', 'v0')}
```

这个小例子保留了源码里最重要的控制结构，方便你先写最小版再回到工程实现。

The small example keeps the most important control structure from the source, so you can write the minimal version before returning to the production implementation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **训练/推理边界** / **Training/inference boundaries**: 这类代码常把数学步骤转换成工程契约。 / This kind of code turns a mathematical step into an engineering contract.
- **nano 系统实现** / **Nano-system implementation**: 从这些片段抽象出的接口可以直接变成你自己的最小模块。 / The interface abstracted from these snippets can become your own minimal module.

## 注意事项 / Caveats / when it breaks

- **边界条件** / **Boundary cases**: 空输入、短 batch、短视频或越界动作通常最容易出 bug。 / Empty inputs, short batches, short videos, or out-of-range actions are the easiest places to break.
- **契约要写测试** / **Test the contract**: 这些函数依赖调用方遵守 shape、顺序和状态生命周期。 / These functions rely on callers respecting shapes, ordering, and state lifetime.

## 延伸阅读 / Further reading

- [vllm-project/vllm source](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/compilation/passes/fusion/rope_kvcache_fusion.py#L46-L76)
