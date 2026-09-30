---
date: 2026-09-30
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/config/vllm.py
permalink: https://github.com/vllm-project/vllm/blob/a8e069f81b4842723a04a84fad2cd5f757820af3/vllm/config/vllm.py#L205-L252
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kernel-fusion]
---

# vLLM fusion gate：先验条件通过了，才把 RoPE 和 KV cache 合成一个 op / vLLM Fusion Gate: Fuse RoPE and KV Cache Only After the Preconditions Pass

> **一句话 / In one line**: 这些小函数把“能不能融合”写成显式门禁，避免编译器在不安全的图切分里硬塞自定义 op。 / These tiny gates make fusion eligibility explicit, so the compiler does not force a custom op into an unsafe graph partition.

## 为什么重要 / Why this matters

高性能 serving 里，RoPE、Q/K norm、KV cache update 往往挨在一起；如果能融合，少一次读写和 kernel launch。但融合不是“越多越好”：后端、custom op 开关、图切分策略都要同时满足，否则会破坏编译图或落到不可用 kernel。

In high-throughput serving, RoPE, Q/K norm, and KV-cache updates often sit next to each other. Fusing them can remove memory traffic and kernel launches. But fusion is not a blanket win: backend availability, custom-op switches, and graph partitioning all have to line up.

## 代码 / The code

`vllm-project/vllm` — [`vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/a8e069f81b4842723a04a84fad2cd5f757820af3/vllm/config/vllm.py#L205-L252)

```python
def enable_rope_kvcache_fusion(cfg: "VllmConfig") -> bool:
    """Enable if rotary embedding custom op is active and
    use_inductor_graph_partition is enabled.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    return (
        rocm_aiter_ops.is_enabled()
        and cfg.compilation_config.is_custom_op_enabled("rotary_embedding")
        and (
            cfg.compilation_config.use_inductor_graph_partition
            or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
        )
    )


def enable_rope_kvcache_mla_fusion(cfg: "VllmConfig") -> bool:
    """Enable if use_inductor_graph_partition is enabled."""
    return (
        cfg.compilation_config.use_inductor_graph_partition
        or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
    )


def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
    """Enable if using AITER RMSNorm and hidden size is 2880 i.e. gpt-oss."""
    return (
        cfg.kernel_config.ir_op_priority.fused_add_rms_norm[0] == "aiter"
        and cfg.model_config is not None
        and cfg.model_config.get_hidden_size() == 2880
    )


def enable_mla_dual_rms_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable MLA dual RMS norm fusion on ROCm with AITER."""
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.is_enabled()


def enable_qk_norm_rope_kvcache(cfg: "VllmConfig") -> bool:
    """Enable fused QK-norm + RoPE/MRoPE + KV cache update with AITER."""
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.is_enabled():
        return False
    return cfg.compilation_config.is_custom_op_enabled("rotary_embedding")

```

## 逐行讲解 / What's happening

1. **第 205-218 行 / Lines 205-218**:
   - 中文: `enable_rope_kvcache_fusion` 同时检查 ROCm AITER 是否启用、`rotary_embedding` custom op 是否允许，以及 KV cache update 是否能留在同一个可融合区域里。
   - English: `enable_rope_kvcache_fusion` checks the ROCm AITER backend, the `rotary_embedding` custom-op switch, and whether KV-cache updates can remain in a fusible region.
1. **第 221-226 行 / Lines 221-226**:
   - 中文: MLA 变体只关心图分区条件；如果 KV cache update 已经不在 splitting ops 里，或使用 Inductor graph partition，就可以放行。
   - English: The MLA variant only gates on graph partitioning: it passes when graph partitioning is active or KV-cache updates are no longer splitting ops.
1. **第 229-235 行 / Lines 229-235**:
   - 中文: `enable_norm_pad_fusion` 展示了另一个门禁形态：不是泛化判断，而是针对 `gpt-oss` hidden size 的精确开关。
   - English: `enable_norm_pad_fusion` shows another style of gate: a deliberately narrow switch for the `gpt-oss` hidden size.
1. **第 245-252 行 / Lines 245-252**:
   - 中文: QK norm + RoPE + KV cache update 的大融合先拒绝非 AITER，再检查 RoPE custom op，顺序很直接。
   - English: The larger QK-norm + RoPE + KV-cache fusion rejects non-AITER first, then checks the RoPE custom op.

## 类比 / The analogy

像机场转机的“最短换乘”规则：两个航班相邻不代表能无缝换乘，还要看是不是同航站楼、安检是否重走、登机口是否支持直通。

It is like a tight airport connection. Two flights may be adjacent on the schedule, but the transfer is only valid if the terminal, security path, and gate rules all support it.

## 自己跑一遍 / Try it yourself

```python
class Compile:
    def __init__(self, custom, partition, split):
        self.custom, self.partition, self.split = custom, partition, split
    def is_custom_op_enabled(self, name): return name in self.custom
    def splitting_ops_contain_kv_cache_update(self): return self.split

def can_fuse(cfg, aiter=True):
    return (aiter and cfg.is_custom_op_enabled('rotary_embedding')
            and (cfg.partition or not cfg.splitting_ops_contain_kv_cache_update()))

for cfg in [Compile({'rotary_embedding'}, False, True), Compile({'rotary_embedding'}, False, False)]:
    print(can_fuse(cfg))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
False
True
```

第一组虽然打开了 RoPE custom op，但 KV cache update 仍会切图，所以拒绝融合。第二组不再被切开，可以放行。

The first configuration enables the RoPE custom op, but KV-cache update still splits the graph, so fusion is rejected. The second no longer splits there, so the gate opens.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch SDPA backend 选择** / **PyTorch SDPA backend selection**: 先检查 dtype、device、mask 形状，再决定 Flash、cuDNN 或 math backend。 / It checks dtype, device, and mask shape before selecting Flash, cuDNN, or math.
- **TensorRT plugin registry** / **TensorRT plugin registry**: plugin 只有在 shape、dtype、capability 全匹配时才替换原图。 / A plugin replaces graph nodes only when shape, dtype, and capability all match.

## 注意事项 / Caveats / when it breaks

- **门禁要保守** / **Gates should be conservative**: 漏掉一次融合只是少一点性能，错误放行会变成 silent wrong result 或编译失败。 / Missing a fusion costs performance; admitting an invalid one can produce wrong results or compiler failures.
- **硬编码条件要有出口** / **Hard-coded conditions need an exit**: 像 hidden size 2880 这种特例，要随着模型族和 kernel 能力变化定期复查。 / Special cases such as hidden size 2880 need periodic review as model families and kernels evolve.

## 延伸阅读 / Further reading

- [vLLM repository](https://github.com/vllm-project/vllm)
- [PyTorch custom operators tutorial](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
