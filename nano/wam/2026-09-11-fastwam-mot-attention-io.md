---
date: 2026-09-11
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/mot.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py#L151-L211
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, dit-block, mixture-of-transformers, rope, qkv]
build_role: dit-block advanced variant
---

# FastWAM MoT：每个 expert 自己准备 QKV，再共享 attention / FastWAM MoT: Per-Expert QKV, Shared Attention

> **一句话 / In one line**: FastWAM 的 MoT 在每个 expert 内完成 modulation、QKV 和 RoPE，再把结果交给共享的混合 attention。 / FastWAM's MoT performs modulation, QKV projection, and RoPE inside each expert before handing the result to shared mixed attention.

## 为什么重要 / Why this matters

中文：视频 token 和动作 token 可以共享一部分计算，但它们的输入分布、时间步和参数化方式不一定相同。MoT 的关键边界是：每个 expert 先用自己的 block 做归一化、时间调制和 QKV 投影，之后才把 token 放进同一个 attention 空间。

English: Video and action tokens can share computation, but their input distributions, timesteps, and parameterizations may differ. MoT's key boundary is that each expert performs its own normalization, timestep modulation, and QKV projection before the tokens enter a shared attention space.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/mot.py`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py#L151-L211)

```python
    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )
```

## 逐行讲解 / What's happening

1. **第 169-188 行 / Lines 169-188**:
   - 中文: 返回值不只有 Q/K/V，还把 residual、MLP 调制量和 checkpoint 标记一起带出，供 attention 后半段继续使用。
   - English: The return value carries more than Q/K/V: residual state, MLP modulation values, and checkpoint metadata continue into the post-attention half.
2. **第 190-191 行 / Lines 190-191**:
   - 中文: `t_mod` 先拆成 MSA 和 MLP 各自的 shift/scale/gate，再只把 MSA 的 shift/scale 应到 attention 输入。
   - English: `t_mod` is split into MSA and MLP shift/scale/gate values; only the MSA shift/scale is applied to the attention input here.
3. **第 193-195 行 / Lines 193-195**:
   - 中文: Q 和 K 额外做 norm，V 保持普通投影；这让 attention 的几何输入更稳定。
   - English: Q and K receive extra normalization while V uses a regular projection, stabilizing the geometry of attention inputs.
4. **第 197-198 行 / Lines 197-198**:
   - 中文: RoPE 只作用于 Q/K，因为它编码的是 query-key 之间的相对位置关系。
   - English: RoPE is applied only to Q/K because it encodes relative positional structure between queries and keys.
5. **第 200-210 行 / Lines 200-210**:
   - 中文: checkpointing 能力作为元数据返回，让外层决定是否对 post-block 做重算，而不把训练策略硬塞进 QKV 构建。
   - English: Checkpointing capability leaves as metadata so the outer layer can choose recomputation without mixing training policy into QKV construction.

## 类比 / The analogy

中文：像两支乐队先各自调音。视频乐队和动作乐队用不同的调音器完成自己的准备，等音高统一后才进入同一个混音台。

English: Imagine two bands tuning separately. The video and action bands use different tuners, then enter the same mixing console only after their signals share a compatible format.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文：这是 `dit-block` 的 advanced variant，依赖已经存在的 patch/token 输入和时间条件。上游是 video/action token、RoPE 频率和 timestep modulation；本模块输出统一 hidden width 的 Q/K/V 加 residual bookkeeping；下游是 shared attention、cross-attention 和 MLP residual。省掉 expert-local preparation 会把不同模态的归一化和时间条件混在一起。生产版还要补齐不同 head 数的兼容策略、mask shape 校验、KV cache 生命周期和 flash kernel 的 layout 契约。

English: This is an advanced `dit-block` variant on top of tokenization and timestep conditioning. Upstream supplies video/action tokens, RoPE frequencies, and timestep modulation; this module emits compatible-width Q/K/V plus residual bookkeeping; downstream is shared attention, cross-attention, and the MLP residual path. Without expert-local preparation, modality-specific normalization and time conditioning get conflated. Production code also needs policies for differing head counts, mask-shape validation, KV-cache lifetime, and flash-kernel layout contracts.

## 自己跑一遍 / Try it yourself

```python
def prepare(tokens, shift, scale, width):
    normalized = [(x - 0.5) for x in tokens]
    modulated = [x * (1 + scale) + shift for x in normalized]
    q = [[x] * width for x in modulated]
    k = [[x + 0.1] * width for x in modulated]
    v = [[x - 0.1] * width for x in modulated]
    return q, k, v

q, k, v = prepare([0.2, 0.8], shift=0.1, scale=0.5, width=2)
rounded = [[[round(value, 2) for value in row] for row in group] for group in (q, k, v)]
print(rounded)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[[[-0.35, -0.35], [0.55, 0.55]], [[-0.25, -0.25], [0.65, 0.65]], [[-0.45, -0.45], [0.45, 0.45]]]
```

中文：这个例子没有真的做 attention，但保留了重要顺序：先调制，再产生 QKV，最后才有资格进入共享 kernel。

English: The miniature does not run attention, but it preserves the important order: modulate first, build QKV second, and only then enter a shared kernel.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LingBot-VA action mode** / **LingBot-VA action mode**: 动作和视频用不同 input/output adapter，共享中间 transformer。 / Action and video use different input/output adapters while sharing the middle transformer.
- **Wan2.1 DiT blocks** / **Wan2.1 DiT blocks**: AdaLN modulation、QKV 和 RoPE 也在 block 内组成固定顺序。 / AdaLN modulation, QKV, and RoPE follow a fixed order inside the block.
- **OpenWAM shared action backbone** / **OpenWAM shared action backbone**: action backbone 把 state/action encoder 和 attention backend 做成可替换组件。 / The action backbone makes state/action encoders and attention backends replaceable components.

## 注意事项 / Caveats / when it breaks

- **Q/K/V layout 要一致** / **Q/K/V layouts must agree**: 共享 attention 前，head 数和 head dim 的契约不能含糊。
- **RoPE 频率要对齐 token 序列** / **RoPE frequencies must align with the token sequence**: action/video token 数错位会造成位置编码污染。
- **缓存的是特征，不是所有状态** / **The cache stores features, not every state**: modulation、mask 和 cross-attention context 仍可能随 step 变化。

## 延伸阅读 / Further reading

- [FastWAM MoT](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py)
- [FastWAM repository](https://github.com/yuantianyuan01/FastWAM)
