---
date: 2026-06-25
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/76e9427657c74f7d063f6c50fce9529900c59e28/wan/modules/model.py#L493-L560
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, wam, dit, adaln, video-generation, wan2, patchify, modulation]
build_role: dit-block (cross-repo variant — WanModel full forward showing 6-vector adaLN conditioning + per-video variable-length patchify)
---

# Wan2.1 WanModel：一个 Linear 产生 6 个 adaLN 调制向量 + 逐视频变长 patchify / Wan2.1 WanModel: One Linear Produces 6 adaLN Modulation Vectors + Per-Video Variable-Length Patchify

> **一句话 / In one line**: `time_projection(e).unflatten(1, (6, dim))` 把一个时间嵌入向量展开成 6 个 adaLN 调制向量，供所有 DiT block 的 shift/scale/gate（自注意力 + 交叉注意力 + MLP 各两个）使用；每个视频单独过 3D Conv patchify，天然支持可变分辨率和时长。 / `time_projection(e).unflatten(1, (6, dim))` expands a single time embedding into 6 adaLN modulation vectors for all DiT blocks (shift/scale/gate for self-attn + cross-attn + MLP); each video is patchified independently via 3D Conv, natively supporting variable resolution and duration.

## 为什么重要 / Why this matters

DiT（Diffusion Transformer）的核心是 adaLN（adaptive Layer Norm）条件化：在每个 transformer block 里，把时间嵌入转化为 shift/scale/gate 三元组，分别作用于自注意力前、MLP 前（和它们的残差门控）。Wan2.1 对经典 DiT 的两个扩展值得深挖：

1. **6 个调制向量一次生成**：标准 DiT（如 Meta 的原始实现）每个 block 独立有一个 `adaLN_modulation` MLP。Wan 把这 6 个向量（自注意力的 shift/scale/gate + MLP 的 shift/scale/gate）统一用一个顶层 `time_projection` 线性层生成，通过 `unflatten(1, (6, dim))` 分成 6 份分发给每个 block——参数更少，但 block 之间的条件化完全共享（每层 adaLN 是一样的，区别只在 block 内部如何应用这 6 个向量）。
2. **逐视频 patchify**：不同视频可能有不同的帧数、分辨率。Wan 对每个视频独立调用 `patch_embedding`（一个 3D Conv），然后 flatten 成 token 序列，最后 padding 到统一长度再 batch 处理。这比强制所有视频等长要自然得多。

Wan2.1's WanModel.forward is a clean production reference for "how adaLN conditioning actually flows through a video DiT." The `unflatten(1, (6, dim))` trick packs all 6 modulation signals into one shared output, then dispatches them to blocks — fewer parameters, cleaner code. The per-video 3D Conv patchify is the correct approach for variable-length video: pad after patchify, not before, so the Conv always sees real frames.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/76e9427657c74f7d063f6c50fce9529900c59e28/wan/modules/model.py#L493-L560)

```python
def forward(
    self,
    x,          # List[Tensor]: each [C_in, F, H, W] — one per video in the batch
    t,          # Tensor [B]: diffusion timesteps
    context,    # List[Tensor]: text embeddings, each [L, C]
    seq_len,    # int: maximum sequence length for positional encoding
    clip_fea=None,  # optional: CLIP image features for i2v mode
    y=None,         # optional: clean reference frames for i2v mode
):
    if self.model_type == 'i2v' or self.model_type == 'flf2v':
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # i2v: prepend clean reference frame to each noisy video
    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]   # concat on frame axis

    # Per-video patchify via 3D Conv (supports variable F, H, W per video)
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]       # each → [1, C, f, h, w]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]               # each → [1, S_i, C]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len

    # Pad shorter sequences to seq_len, then batch
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])  # [B, seq_len, C]

    # Time embedding → 6 adaLN modulation vectors (the key trick)
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))   # [B, 6, dim]
        # e0[:, 0] = shift_attn,  e0[:, 1] = scale_attn, e0[:, 2] = gate_attn
        # e0[:, 3] = shift_mlp,   e0[:, 4] = scale_mlp,  e0[:, 5] = gate_mlp
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # Text context: pad to uniform length, then embed
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )
    # (forward continues: CLIP i2v conditioning, transformer blocks, head — not shown)
```

## 逐行讲解 / What's happening

1. **i2v 帧拼接（`torch.cat([u, v], dim=0)`）：在帧轴上拼参考帧 / i2v frame concatenation on the frame axis**
   - 中文：`x` 列表里的每个张量形状是 `[C, F, H, W]`（C=通道，F=帧数，H/W=空间维度）。`dim=0` 是 C 通道轴而不是帧轴——等等，这里的 `dim=0` 实际上是帧轴，因为这些是未 unsqueeze 的视频张量（没有 batch 维度），`[C, F, H, W]` 中 C 是第 0 维。不，实际上 `torch.cat([u, v], dim=0)` 把 clean frame 拼到 noisy frames 的通道维度——这是 Wan i2v 的条件化方式：参考帧被 channel-concat 到噪声视频里，而不是拼在时间轴上。
   - English: Each `x[i]` has shape `[C, F, H, W]` where C is the channel dim (dim=0). `torch.cat([u, v], dim=0)` concatenates the clean reference frame along the **channel** axis — Wan's image-to-video conditioning stacks the clean reference frame's channels alongside the noisy video's channels, not on the temporal axis. The 3D Conv patchify then processes the doubled-channel tensor.

2. **逐视频 patchify（`self.patch_embedding(u.unsqueeze(0))`）：变长支持的关键 / Per-video patchify for variable-length support**
   - 中文：`self.patch_embedding` 是一个 3D Conv（kernel `(patch_t, patch_h, patch_w)`）。对每个视频单独调用，`unsqueeze(0)` 加上 batch 维度。这样不同的 `(F, H, W)` 都能正常处理——不需要提前把所有视频 pad 到相同大小，而是 patchify 之后再 pad token 序列。这个顺序很重要：pad 前 patchify，避免 Conv 看到填充像素。
   - English: `self.patch_embedding` is a 3D Conv. Calling it independently on each video (with `unsqueeze(0)` adding the batch dim) means videos can have different `(F, H, W)` — no pre-padding to a fixed size. Padding happens *after* patchify, so the Conv always processes real pixels. The resulting sequence lengths are recorded in `seq_lens` for masking in the attention blocks.

3. **`time_projection(e).unflatten(1, (6, self.dim))`：6 个调制向量一次生成 / Generating all 6 modulation vectors at once**
   - 中文：`time_projection` 是 `nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))`，输出形状 `[B, dim*6]`。`unflatten(1, (6, dim))` 把第 1 维从 `dim*6` 分成 `(6, dim)` 两级，得到 `[B, 6, dim]`。这 6 个向量对应 DiT block 里 adaLN 的 6 个调制信号：自注意力的 shift/scale/gate，MLP 的 shift/scale/gate。每个 block 接收同样的 `e0`，但内部如何应用这 6 个向量由各 `WanAttentionBlock` 决定。
   - English: `time_projection` is `nn.Sequential(nn.SiLU(), nn.Linear(dim, dim*6))`. Its output `[B, dim*6]` is reshaped by `unflatten(1, (6, dim))` to `[B, 6, dim]`. The 6 slices are: shift/scale/gate for self-attention (indices 0-2) and shift/scale/gate for MLP (indices 3-5). All blocks receive the **same** `e0` — the shared conditioning — but each `WanAttentionBlock` applies the slices to its own layernorm parameters. This is a parameter-efficient design: one `time_projection` drives all blocks instead of per-block MLPs.

4. **`amp.autocast(dtype=torch.float32)` 包裹时间嵌入 / FP32 autocast for time embedding**
   - 中文：时间嵌入的 sinusoidal 函数和 SiLU 激活在 BF16 下可能损失精度（尤其是频率低的分量）。强制 FP32 计算后再传给 blocks，保证调制信号的精度，而 block 内部仍然可以用 BF16/FP16。
   - English: Sinusoidal embeddings and their SiLU activations can lose precision in BF16 (low-frequency components become numerically indistinct). Computing in FP32 here ensures the conditioning signal `e0` is accurate before it flows into the blocks, even if the block internals run in BF16. The `assert` on line 550 guards against silent dtype downcast.

## 类比 / The analogy

想象 WanModel 是一个管弦乐团的指挥（时间嵌入就是节拍器）。`time_projection(e).unflatten(1, (6, dim))` 就是指挥把节拍器的速度信号分解成 6 根指挥棒（给弦乐的力度、速度、休止，和给管乐的力度、速度、休止），每个乐手（DiT block）拿到这 6 根信号棒后，按自己的方式调整演奏。逐视频 patchify 就是每个乐手先独立把乐谱（视频帧）切成小节（patch），然后才上台合奏——不需要提前把所有乐谱统一格式。

Think of WanModel as an orchestra conductor whose metronome is the timestep. `time_projection(e).unflatten(1, (6, dim))` is the conductor splitting one tempo signal into 6 baton gestures — shift/scale/gate for strings (self-attention) and shift/scale/gate for brass (MLP). Each musician (DiT block) receives these 6 gestures and adapts their playing accordingly. Per-video patchify is each musician cutting their own sheet music (video frames) into bars (patches) before they arrive on stage — no need to standardize all sheet music to the same length beforehand.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 nanoWAM 构建计划中 **dit-block** 组件的 Wan2.1 全模型视角变体。之前的 dit-block 笔记聚焦于单个 block 的 adaLN 结构；这里展示的是在模型顶层如何用一个 `time_projection` 生成所有 block 共享的 6 个调制向量——这是产品级实现的标准做法。

This is the **dit-block** component in the nanoWAM build plan — a Wan2.1 full-model view. Earlier dit-block notes covered what one block does internally; this note shows how the top-level model produces the shared conditioning `e0` in one shot.

In your nanoWAM:
- The `time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))` pattern is production-proven and easy to replicate.
- `unflatten(1, (6, dim))` is the cleanest way to split the 6 modulation signals — more readable than manually slicing `e[:, :dim]`, `e[:, dim:2*dim]`, etc.
- For variable-length video, adopt the per-item patchify + padding-after-patchify pattern: `[patch_embedding(u.unsqueeze(0)) for u in x]` then `torch.cat` with zero-padding.
- Upstream components needed: VAE encoder (produces `x`), noise scheduler (produces `t`), text encoder (produces `context`).

Production additions: (a) RoPE frequencies for the 3D spatial/temporal positions (the `self.freqs` buffer); (b) cross-attention for text/CLIP conditioning inside `WanAttentionBlock`; (c) paged attention / sequence-parallel for long videos.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn

dim = 64  # small for demo

time_embedding = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

t_emb = torch.randn(2, dim)
e  = time_embedding(t_emb)               # [B, dim]
e0 = time_projection(e).unflatten(1, (6, dim))  # [B, 6, dim]

print(f"e0 shape: {e0.shape}")           # [2, 6, 64]

# Simulate how a DiT block uses the 6 modulation signals
shift_attn, scale_attn, gate_attn = e0[:, 0], e0[:, 1], e0[:, 2]
shift_mlp,  scale_mlp,  gate_mlp  = e0[:, 3], e0[:, 4], e0[:, 5]

# adaLN applied to some intermediate tensor x
x = torch.randn(2, 16, dim)  # [B, seq, dim]
norm = nn.LayerNorm(dim)
x_normed = norm(x)
x_modulated = x_normed * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
print(f"modulated tensor shape: {x_modulated.shape}")  # [2, 16, 64]
print("All 6 adaLN signals ready for attention and MLP.")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
e0 shape: torch.Size([2, 6, 64])
modulated tensor shape: torch.Size([2, 16, 64])
All 6 adaLN signals ready for attention and MLP.
```

`unflatten(1, (6, dim))` 的妙处：代码里不需要出现任何 `e[:, :dim]` 这样的硬编码切片，6 个信号都有了有意义的名字，代码可读性大幅提升。

`unflatten(1, (6, dim))` eliminates all hardcoded `e[:, :dim]`, `e[:, dim:2*dim]` slices. All six signals get named, and the code reads like documentation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Meta DiT 原版（`DiT.py`）** / **Meta DiT original**: 每个 block 有自己的 `adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))`，但同样输出 6 个信号——Wan 把这个 MLP 提升到顶层共享，参数少很多。
- **CogVideoX（昨日 WAM 笔记）** / **CogVideoX (yesterday's WAM note)**: 类似的 adaLN 结构，参见 [`2026-06-24-cogvideo-dpmpp2m-sampler.md`](../../2026/06/2026-06-24-cogvideo-dpmpp2m-sampler.md)。
- **Psi0 AdaLayerNormZero（今日 trending 笔记）** / **Psi0 AdaLayerNormZero (today's trending note)**: 和 Wan 用的是同一种 6-way split adaLN，只是 Psi0 把 split 放在 block 内部，Wan 放在模型顶层。

## 注意事项 / Caveats / when it breaks

- **`e0` 是所有 block 共享的** / **`e0` is shared across all blocks**: 这意味着不同 block 接收完全相同的调制信号，与每 block 独立的 `adaLN_modulation` 相比，表达力更弱。Wan 的实验证明这在视频生成中足够，但对精细控制需求高的任务（如高保真单帧图像生成）可能需要 per-block adaLN。
- **per-video patchify 的效率** / **Per-video patchify efficiency**: `[patch_embedding(u.unsqueeze(0)) for u in x]` 是一个串行 Python 循环，对大 batch 有性能损失。生产实现可以改用 `pad_sequence` 统一 patchify，或用自定义 CUDA kernel 处理变长输入。
- **FP32 时间嵌入的显存** / **FP32 time embedding memory**: 用 `amp.autocast(dtype=torch.float32)` 强制 FP32 会让 `e` 和 `e0` 占用更多显存（BF16 的 2 倍）。对于 dim=5120 的大模型，这个开销不可忽视。

## 延伸阅读 / Further reading

- [Wan2.1 技术报告](https://arxiv.org/abs/2503.20314)
- [DiT 原始论文（Peebles & Xie, 2022）](https://arxiv.org/abs/2212.09748)
- [Wan2.1 昨日笔记 — FlowDPMSolverMultistepScheduler](2026-06-24-wan21-flow-dpm-solver.md)
