---
date: 2026-06-26
topic: robotics
source: trending
repo: hao-ai-lab/FastVideo
file: fastvideo/attention/layer.py
permalink: https://github.com/hao-ai-lab/FastVideo/blob/f9b8e30ff33f1845b81c71958511d2c344fdba78/fastvideo/attention/layer.py#L163-L236
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, robotics, video-generation, sequence-parallelism, sparse-attention, distributed, vsa]
---

# FastVideo 的视频稀疏注意力序列并行：把 gate_compress 打包进 all-to-all，省掉一次通信 / FastVideo VSA Sequence Parallelism: Bundle gate_compress into the all-to-all and Save One Communication Round

> **一句话 / In one line**: `DistributedAttention_VSA` 把 Video Sparse Attention（VSA）的 gate_compress 张量和 qkv 拼成一个 4× 大的张量，用一次 all-to-all 集体通信同时完成"scatter heads + gather seq"的序列并行，避免了第二次通信。 / `DistributedAttention_VSA` concatenates VSA's gate_compress tensor with qkv into a single 4× tensor, performing a single all-to-all collective to do both "scatter heads + gather seq" for sequence parallelism, avoiding a second communication round.

## 为什么重要 / Why this matters

长视频生成是当前 diffusion 模型最难啃的工程骨头：一段 10 秒的 720P 视频展开后有几百万个 token，单卡放不下，必须跨卡做序列并行（sequence parallelism）。序列并行的关键操作是 `all-to-all`：把每张卡上的部分头（full seq, shard heads）换成每张卡上的部分序列（shard seq, full heads），才能在卡内做完整的注意力计算。

FastVideo 在此基础上加了 Video Sparse Attention（VSA）：一个可学习的 gate 张量（`gate_compress`）决定哪些视频 token 参与稀疏注意力，哪些被跳过，大幅降低长视频的注意力计算量。但 `gate_compress` 和 `q/k/v` 一样，也需要经过 all-to-all 重新分布。朴素做法需要两次 all-to-all（先 qkv，再 gate_compress），FastVideo 把它们拼在一起，一次就搞定——通信代价减半。

Long-video generation is the hardest engineering challenge for current diffusion models: a 10-second 720P video flattened to tokens can have millions of tokens, far beyond a single GPU. Sequence parallelism splits this across GPUs via `all-to-all`: swap "full sequence, shard heads" for "shard sequence, full heads" so each GPU can compute complete attention within its shard.

FastVideo adds Video Sparse Attention (VSA) on top: a learnable gate tensor (`gate_compress`) determines which video tokens participate in sparse attention and which are skipped, dramatically reducing attention compute for long videos. But `gate_compress` needs the same all-to-all redistribution as `q/k/v`. The naive approach would run two all-to-all collectives (one for qkv, one for gate). FastVideo bundles them together and does it in one — halving the communication cost.

## 代码 / The code

`hao-ai-lab/FastVideo` — [`fastvideo/attention/layer.py`](https://github.com/hao-ai-lab/FastVideo/blob/f9b8e30ff33f1845b81c71958511d2c344fdba78/fastvideo/attention/layer.py#L163-L236)

```python
class DistributedAttention_VSA(DistributedAttention):
    """Distributed attention layer with VSA support."""

    @_maybe_compiler_disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_seq_len: int,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        gate_compress: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.

        Args:
            q/k/v: [batch_size, seq_len, num_heads, head_dim]
            original_seq_len: Original (unpadded) full sequence length
            gate_compress: [batch_size, seq_len, num_heads, head_dim] — VSA gate
        """
        assert replicated_q is None and replicated_k is None and replicated_v is None, \
            "Replicated QKV is not supported for VSA now"
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        batch_size, seq_len, num_heads, head_dim = q.shape

        # THE KEY TRICK: bundle gate_compress as a 4th "virtual batch" alongside qkv
        qkvg = torch.cat([q, k, v, gate_compress], dim=0)
        # Shape before: [4*batch, shard_seq_len, num_heads, head_dim]
        # all-to-all: scatter heads (dim=2) and gather seq (dim=1)
        qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)
        # Shape after: [4*batch, full_seq_len, shard_num_heads, head_dim]

        # Trim padding that may have been added to make seq len divisible
        pad_seq_len = qkvg.shape[1] - original_seq_len
        qkvg = qkvg[:, :original_seq_len, :, :]

        # Apply rotary embeddings to q and k only (first 2*batch slices)
        if freqs_cis is not None:
            cos, sin = freqs_cis
            qkvg[:batch_size * 2] = _apply_rotary_emb(
                qkvg[:batch_size * 2], cos, sin, is_neox_style=False
            )

        qkvg = self.attn_impl.preprocess_qkv(qkvg, ctx_attn_metadata)

        # Split back into q, k, v, gate_compress
        q, k, v, gate_compress = qkvg.chunk(4, dim=0)

        # Sparse attention: gate_compress controls which tokens participate
        output = self.attn_impl.forward(q, k, v, gate_compress, ctx_attn_metadata)

        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        # Re-pad, then inverse all-to-all: scatter seq (dim=1), gather heads (dim=2)
        output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad_seq_len))
        output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)

        return output, None  # replicated_output is None (text tokens not yet supported)
```

## 逐行讲解 / What's happening

1. **`qkvg = torch.cat([q, k, v, gate_compress], dim=0)`**
   - 中文: 把四个 `[B, S, H, D]` 张量沿 batch 维度（dim=0）拼成一个 `[4B, S, H, D]` 张量。这样 all-to-all 只需要一次，会同时处理 q、k、v 和 gate_compress 四个分量，等效于并行执行了 4 次单独的 all-to-all，但通信批次只有 1 次。
   - English: Concatenate four `[B, S, H, D]` tensors along the batch dimension (dim=0) to get `[4B, S, H, D]`. The single all-to-all then processes all four components simultaneously — equivalent to 4 separate all-to-all calls but with only 1 communication round.

2. **`sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)`**
   - 中文: 这是序列并行的核心集体通信：scatter heads（把 `num_heads` 沿 dim=2 分散到各卡）同时 gather seq（把各卡上的局部序列沿 dim=1 收集成完整序列）。每张卡变换后拥有：完整序列长度、但只有 `num_heads / world_size` 个头，可以在卡内做完整的 multi-head attention。
   - English: This is the core collective for sequence parallelism: scatter heads (split `num_heads` along dim=2 across cards) while gathering sequence (collect partial sequences along dim=1 into the full sequence). After the all-to-all, each card holds the full sequence length but only `num_heads / world_size` heads — enough to compute complete multi-head attention locally.

3. **`pad_seq_len = qkvg.shape[1] - original_seq_len; qkvg = qkvg[:, :original_seq_len, :, :]`**
   - 中文: all-to-all 要求序列长度能被 world_size 整除，所以在此之前可能已经 pad 了若干位置。这里把 padding 去掉，只保留真实的序列长度，避免稀疏注意力在 padding token 上浪费计算。最后输出时会重新 pad 回去再做逆 all-to-all。
   - English: all-to-all requires sequence length to be divisible by world_size, so padding may have been added earlier. This strips that padding back to the real sequence length, preventing sparse attention from wasting compute on pad tokens. The output is re-padded before the inverse all-to-all.

4. **`qkvg[:batch_size * 2]` 做 RoPE，不动 v 和 gate**
   - 中文: RoPE（旋转位置编码）只应用于 q 和 k，不应用于 v 或 gate_compress。`qkvg[:batch_size * 2]` 精确索引出拼合张量里 q 和 k 对应的前两段（`q = qkvg[:B]`，`k = qkvg[B:2B]`），原地修改后 v 和 gate 不受影响。
   - English: RoPE applies only to q and k — not v or gate_compress. `qkvg[:batch_size * 2]` indexes exactly the q and k slices (`q = qkvg[:B]`, `k = qkvg[B:2B]`) from the stacked tensor, modifying them in-place while v and gate remain untouched.

5. **`q, k, v, gate_compress = qkvg.chunk(4, dim=0)` → `attn_impl.forward(q, k, v, gate_compress, ...)`**
   - 中文: 解包回四个独立张量，然后送入底层稀疏注意力实现（`attn_impl`，可以是 FlashAttention with VSA mask 或 Triton kernel）。`gate_compress` 在这里控制哪些 token 参与注意力计算，实现视频稀疏注意力的核心逻辑。
   - English: Unpack back to four independent tensors, then pass to the backend sparse attention implementation (`attn_impl`, which can be FlashAttention with a VSA mask or a Triton kernel). `gate_compress` controls which tokens participate in attention here — this is where the core VSA logic lives.

6. **`@_maybe_compiler_disable` 装饰器**
   - 中文: 根据环境变量 `FASTVIDEO_DISABLE_ATTENTION_COMPILE` 决定是否对这个 forward 函数调用 `torch.compiler.disable()`。对于某些 attention 后端（如使用了 CUDA graph 的稀疏 kernel），`torch.compile` 会引入兼容问题，这个装饰器提供了无需修改代码就能关闭编译的机制。
   - English: Conditionally applies `torch.compiler.disable()` to this forward function based on the `FASTVIDEO_DISABLE_ATTENTION_COMPILE` env var. Some attention backends (e.g. sparse kernels using CUDA graphs) are incompatible with `torch.compile`; this decorator provides a code-change-free escape hatch.

## 类比 / The analogy

想象 8 个快递员（GPU）分拣一批 4 种物品（q、k、v、gate_compress）的包裹。朴素做法：先分拣 q/k/v（一次交接），再分拣 gate_compress（第二次交接）。FastVideo 的做法：把这 4 种物品装进同一辆大卡车（`torch.cat`），只跑一趟完成所有物品的重新分配（一次 all-to-all），到达目的地后再拆箱（`.chunk(4)`）分给各自的工位。通信总量不变，但往返次数减半，在高延迟网络（跨机器 NVLink 或 InfiniBand）上效果显著。

Imagine 8 delivery workers (GPUs) sorting packages of 4 item types (q, k, v, gate_compress). Naive: sort q/k/v (one handoff round), then sort gate_compress (second handoff round). FastVideo's approach: load all 4 types into one large truck (`torch.cat`), make a single trip to redistribute everything (one all-to-all), then unpack the crates (`.chunk(4)`) at the destination. Total data moved is the same, but the number of round trips is halved — significant on high-latency networks (cross-node NVLink or InfiniBand).

## 自己跑一遍 / Try it yourself

```python
import torch

def mock_all_to_all(tensor, scatter_dim, gather_dim, world_size=2, rank=0):
    """Simulate all-to-all: scatter along scatter_dim, gather along gather_dim."""
    # For demonstration: just split and reassemble (no real MPI)
    chunks = tensor.chunk(world_size, dim=scatter_dim)
    # Each rank "receives" one chunk from each peer; here we just return the full tensor
    return tensor  # simplified mock

B, S, H, D = 2, 64, 8, 32
q = torch.randn(B, S, H, D)
k = torch.randn(B, S, H, D)
v = torch.randn(B, S, H, D)
gate = torch.randn(B, S, H, D)

# The FastVideo trick: bundle as one tensor
qkvg = torch.cat([q, k, v, gate], dim=0)
print(f"bundled shape: {qkvg.shape}")  # [4*B, S, H, D]

# After all-to-all (mocked): scatter heads, gather seq
qkvg_after = mock_all_to_all(qkvg, scatter_dim=2, gather_dim=1, world_size=2, rank=0)

# Split back
q2, k2, v2, gate2 = qkvg_after.chunk(4, dim=0)
print(f"q shape after split: {q2.shape}")   # [B, S, H, D]
print(f"gate shape: {gate2.shape}")          # [B, S, H, D]
assert torch.allclose(q, q2) and torch.allclose(gate, gate2)
print("bundled all-to-all roundtrip OK")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
bundled shape: torch.Size([8, 64, 8, 32])
q shape after split: torch.Size([2, 64, 8, 32])
gate shape: torch.Size([2, 64, 8, 32])
bundled all-to-all roundtrip OK
```

中文：这个 mock 展示了"拼合 → all-to-all → 拆分"的核心思路，虽然没有真实的分布式通信，但张量维度变换的逻辑是完整的。真实情况下 all-to-all 之后每张卡只持有 `H/world_size` 个头，这里为了可运行简化了。

English: This mock shows the core "bundle → all-to-all → split" idea without real distributed communication, but the tensor dimension logic is complete. In reality each card holds only `H/world_size` heads after the all-to-all; the mock skips that redistribution for simplicity.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Ring Attention（Berkeley）** / **Ring Attention (Berkeley)**: 另一种序列并行方案——不做 all-to-all 重新分布 heads，而是让每张卡保留自己的 heads，把 kv 在卡间滚动传递成环形，每张卡只接收一段 kv 就完成一次"片段注意力"。通信量相同，但结构不同。 / An alternative sequence parallelism: instead of all-to-all redistributing heads, each card keeps its own heads and rotates kv blocks in a ring; each card runs a "fragment attention" per rotation. Same communication volume, different topology.
- **Megatron 的 Tensor + Sequence 并行组合** / **Megatron's Tensor + Sequence Parallelism combined**: Megatron 用 tensor parallelism（切 heads）+ sequence parallelism（切 seq），两种并行同时生效，但 all-to-all 的组织方式和 FastVideo 有所不同——Megatron 在 `ColumnParallelLinear` 层之前 scatter seq，在 `RowParallelLinear` 之后 gather seq。 / Megatron uses tensor parallelism (cut heads) + sequence parallelism (cut seq) simultaneously, but organizes all-to-all differently — Megatron scatters seq before `ColumnParallelLinear` and gathers it after `RowParallelLinear`.
- **FastVideo 的父类 `DistributedAttention`** / **FastVideo's parent class `DistributedAttention`**: 基础版只处理 qkv（3× bundle），VSA 变体在此基础上增加了第 4 个 gate_compress 张量，其余结构完全复用父类逻辑。 / The base class only bundles qkv (3× batch); the VSA variant adds the 4th gate_compress tensor, reusing the parent class structure for everything else.

## 注意事项 / Caveats / when it breaks

- **Replicated QKV 暂不支持** / **Replicated QKV not yet supported**: 代码里有明确的 `assert replicated_q is None ...` 检查。Replicated QKV 是给文本 token 用的路径（文本 token 复制到所有卡，不参与序列并行），VSA 版本暂不支持这条路径。 / The code has an explicit `assert replicated_q is None ...`. The replicated QKV path is for text tokens (replicated across all cards, not sequence-parallelized); VSA doesn't support this path yet.
- **`gate_compress` 必须不为 None** / **`gate_compress` must not be None**: 如果 `gate_compress=None` 而代码尝试执行 `torch.cat([..., None], dim=0)` 会报错。调用者必须确保 VSA 路径只在 gate_compress 有效时使用，否则应退回基类的 `DistributedAttention.forward`。 / If `gate_compress=None`, `torch.cat([..., None], dim=0)` will raise. Callers must ensure VSA is only invoked when gate_compress is valid; otherwise fall back to the parent `DistributedAttention.forward`.
- **seq_len 必须能被 world_size 整除** / **seq_len must be divisible by world_size**: all-to-all scatter 要求序列维度能整除 world_size。padding 是在进入 all-to-all 之前加的，这段代码里只做了 trim（去除推理时的 pad），如果 padding 未在上游正确计算，可能导致 shape mismatch。 / The all-to-all scatter requires seq_len divisible by world_size. Padding must be added upstream before the all-to-all; this code only trims (removes) padding, so incorrect upstream padding causes shape mismatches.

## 延伸阅读 / Further reading

- [FastVideo GitHub repo](https://github.com/hao-ai-lab/FastVideo)
- [VSA (Video Sparse Attention) — to be linked when paper publishes]
- [Ring Attention paper (arXiv:2310.01889)](https://arxiv.org/abs/2310.01889)
- [Megatron sequence parallelism paper (arXiv:2205.05198)](https://arxiv.org/abs/2205.05198)
