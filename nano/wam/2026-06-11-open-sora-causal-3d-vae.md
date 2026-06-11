---
date: 2026-06-11
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/hunyuan_vae/unet_causal_3d_blocks.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/unet_causal_3d_blocks.py#L50-L158
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, wam, temporal-compression, causal-3d-vae, hunyuan-vae, open-sora]
build_role: temporal-compression (cross-repo variant — Open-Sora HunyuanVAE causal-3D primitives, complements Wan2.1 streaming-cache version)
---

# 让 3D VAE "听不到未来":Open-Sora 的三个因果原语 / Making a 3D VAE deaf to the future: Open-Sora's three causal primitives

> **一句话 / In one line**: 三件套合起来让一个 3D VAE 变成"时间因果"的:(1) 一个**分块三角 attention mask**,让每一帧只看得到自己及之前的帧;(2) 一个**不对称的 (T-1, 0) Conv3d padding**,让时间卷积只向前看;(3) 一个**首帧二维上采样**的小技巧,因为第一帧没有"过去"可以插值。 / Three primitives compose to make a 3D VAE temporally causal: (1) a **block-triangular attention mask** so each frame attends only to itself and past frames; (2) an **asymmetric (T-1, 0) Conv3d padding** so temporal convs only look backward; (3) a **first-frame-only 2D upsampling** trick because the first frame has no past to interpolate against.

## 为什么重要 / Why this matters

视频世界模型(WAM)推理时一定是**流式**的:你给它一帧,它出一帧;再给一帧,再出一帧。它不能像离线训练那样**"先看完整段再编码"**,否则 latency 会爆。所以 WAM 用的 3D VAE 必须**时间因果** — 任何时刻 t 的输出**只能依赖 ≤ t 的输入**,跟自回归 LM 在文本上是一回事。Open-Sora 的 HunyuanVAE 把这件事拆得**特别干净**:每个时间维相关的算子(attention / Conv3d / Upsample)各有一个对应的因果改造。读完这段你会有种"啊原来 3D 因果是这么便宜实现"的清爽感。

5 月 29 日我们讲过 Wan2.1 的 `feat_cache` — 那是**推理时的流式 cache**(像 LM 的 KV cache 一样)。今天讲的 Open-Sora 是**训练时就让网络结构本身因果**(像 GPT 的 causal mask 一样)。两条路线、同一个目标:让 VAE 在时序上单向。这是 `temporal-compression` 这个 build slot 的两个互补视角。

Video world models (WAM) must run **streaming** at inference: you feed one frame, get one frame; feed another, get another. Unlike offline training, they can't **"see the whole clip and then encode"** — latency would tank. So the 3D VAE inside a WAM must be **temporally causal** — output at time t depends only on inputs at ≤ t, just like an autoregressive LM in text. Open-Sora's HunyuanVAE achieves this with **three small, mutually orthogonal modifications** — one per time-aware operator (attention, Conv3d, upsample). Reading this leaves you with the cheerful realization "ah, so 3D causality is *that* cheap."

On 2026-05-29 we taught Wan2.1's `feat_cache` — that's a **streaming inference cache** (analogous to LM's KV cache). Today's Open-Sora piece is the **other angle**: building causality into the network *itself*, at training time, like GPT's causal mask in text. Two routes, same destination — VAE that's one-way in time. These are the two complementary views of the `temporal-compression` build slot.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/hunyuan_vae/unet_causal_3d_blocks.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/unet_causal_3d_blocks.py#L50-L158)

```python
def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class CausalConv3d(nn.Module):
    """
    Implements a causal 3D convolution layer where each position only depends on previous timesteps and current spatial locations.
    This maintains temporal causality in video generation tasks.
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode="replicate",
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size - 1,
            0,
        )  # W, H, T
        self.time_causal_padding = padding

        self.conv = ChannelChunkConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class UpsampleCausal3D(nn.Module):
    """
    A 3D upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        bias=True,
        upsample_factor=(2, 2, 2),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample_factor = upsample_factor
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        assert input_tensor.shape[1] == self.channels
        hidden_states = input_tensor

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # interpolate H & W only for the first frame; interpolate T & H & W for the rest
        T = hidden_states.size(2)
        first_h, other_h = hidden_states.split((1, T - 1), dim=2)
        # process non-1st frames
        if T > 1:
            other_h = chunk_nearest_interpolate(other_h, scale_factor=self.upsample_factor)
        # process 1st frame
        first_h = first_h.squeeze(2)
        first_h = chunk_nearest_interpolate(first_h, scale_factor=self.upsample_factor[1:])
        first_h = first_h.unsqueeze(2)
        # concat together
        if T > 1:
            hidden_states = torch.cat((first_h, other_h), dim=2)
        else:
            hidden_states = first_h

        hidden_states = self.conv(hidden_states)
        return hidden_states
```

## 逐行讲解 / What's happening

1. **`prepare_causal_attention_mask` — 分块三角 mask / Block-triangular mask**:
   - 中文: 假设 video latent 是 `(T 帧, H*W 个位置)`,展平后 `seq_len = T * (H*W)`。这个 mask 是 `(seq_len, seq_len)` 的下三角,但**以帧为单位**:位置 `i` 在第 `i_frame = i // n_hw` 帧上,它能看到 `< (i_frame + 1) * n_hw` 的所有位置 — 也就是**当前帧的所有 H*W 位置 + 所有过去帧的所有位置**。同帧内 H*W 之间是全连接的(空间是非因果的),跨帧是严格因果的。这就叫 **block-triangular** — 不是元素级三角,是帧块级三角。
   - English: video latent shape is `(T frames, H*W tokens per frame)` flattened to `seq_len = T * (H*W)`. The mask is `(seq_len, seq_len)` lower-triangular **at the frame block level**: position `i` lives in frame `i_frame = i // n_hw`, and may attend to anything in frames `< i_frame + 1` — i.e., **all spatial positions of the current frame + all positions of all earlier frames**. Within a frame, spatial attention is dense (space is not causal); across frames it's strictly one-way. That's **block-triangular** — not element-level, but block-level.

2. **`CausalConv3d` 的不对称 padding `(kernel - 1, 0)`**:
   - 中文: 正常的 Conv3d 在时间维 padding 是 `(kernel // 2, kernel // 2)` — 对称,左右各填一半。因果 3D conv 改成 `(kernel - 1, 0)` — **所有 padding 全塞在过去那一侧**,未来那一侧零 padding。这样卷积窗口在时间 t 的输出只依赖 `[t - kernel + 1, t]` 这段,**永远不碰未来**。space 维 (H, W) 仍然对称 padding,因为空间没有"过去/未来"的概念。
   - English: a normal Conv3d uses `(kernel // 2, kernel // 2)` time padding — symmetric, half on each side. Causal Conv3d uses `(kernel - 1, 0)` — **all padding stuffed on the *past* side**, zero on the *future* side. The output at time t now depends only on inputs `[t - kernel + 1, t]`, **never future**. Spatial (H, W) padding stays symmetric — space has no past/future.

3. **`UpsampleCausal3D` 的"首帧 2D / 其余 3D" 拆分**:
   - 中文: 这是整段代码最巧妙的一块。3D 上采样要在时间 + 空间一起放大,但**第一帧没有"前一帧"可以插值** — 如果你硬要 3D 插值,第一帧的时间维会去取一个不存在的"过去帧"。Open-Sora 的解法是:**把第一帧切出来 (`first_h, other_h = split((1, T-1))`),它只做 2D(H, W)放大,其余帧做 3D(T, H, W)放大,然后再拼回去**。这一招让首帧的因果性得以保留 — 它不会"假装"自己有过去。
   - English: this is the cleverest piece. 3D upsampling enlarges time *and* space together — but **the first frame has no past frame to interpolate against**. Naive 3D interpolation would pretend the first frame had a non-existent past. Open-Sora's fix: **split off the first frame (`first_h, other_h = split((1, T-1))`), upsample it in 2D only (H, W), upsample the rest in full 3D, then concat them back together along T**. Causality at frame 0 is preserved — no fictional past.

4. **`chunk_nearest_interpolate` 分块插值 / Chunked interpolation**:
   - 中文: 不是这里的重点,但值得提一句:大体积 3D tensor 的 `F.interpolate` 会触发 CUDA 的 numel 上限,所以代码把它沿 batch 维分块,一块一块跑完再 cat。这是工程上的一个小坑。
   - English: not the main story, but worth noting: large 3D tensors hit CUDA's numel cap inside `F.interpolate`, so the code chunks along batch dim and concatenates. A small engineering trap worth being aware of.

5. **`for i in range(seq_len): mask[i, :(i_frame+1)*n_hw] = 0` 的复杂度**:
   - 中文: 这是 O(seq_len) 的 Python loop,但 `seq_len` 是 `T * H*W` 可以很大(64 帧 × 32×32 = 65536)。生产里通常用 `arange + comparison` 写成纯 tensor 版本,而不是 Python loop。代码里这么写是为了清晰。
   - English: this is an O(seq_len) Python loop, and `seq_len` is `T * H * W` — can be 65 536 for 64 × 32 × 32. Production code typically replaces it with a vectorized `arange + comparison`. The loop form here favors readability.

6. **`pad_mode="replicate"` 不是"zero" / Replicate, not zero**:
   - 中文: 时间维 padding 用 `replicate`(复制最早一帧)而不是 zero。如果用 zero,模型会以为视频从一片黑开始,影响低层特征。replicate 等于"假装视频在第 -1 帧时和第 0 帧一样" — 是更自然的边界条件。
   - English: time padding uses `replicate` (copy the earliest real frame) instead of zero. Zero padding would tell the model "video starts from black," polluting low-level features. Replicate effectively asserts "the video was the same at frame -1 as at frame 0" — a far more natural boundary.

## 类比 / The analogy

想象你在剪一段录像:你想给每一帧加滤镜,但**只能用这一帧和它之前的画面信息** — 比如稳定化算法不能看到下一帧,否则就不是实时算法了。

- **mask** 像剪辑师戴上的"时光眼罩":看第 3 帧时,只能盯着第 0、1、2、3 帧的画面,后面用黑布盖住。
- **CausalConv3d** 像一个"前向积分器":每输出一帧,只读最近 K 帧的画面,从不预读未来。
- **UpsampleCausal3D 的首帧特殊处理** 像是片头:第一帧没有前传,所以你不能用"运动差分"放大它,只能用单帧的空间细节放大;之后每一帧都可以借助"上一帧是什么样"来更聪明地放大。

三件套合起来,就是把一个**离线 3D 网络**变成可以**逐帧推进、永远不偷看未来**的网络。

Picture editing a live video stream where each frame may use **only itself and earlier frames** — like a stabilization algorithm that can't peek at frame t+1 if it wants to stay realtime.

- The **mask** is the editor's "time blinder": watching frame 3, you only see frames 0–3; later frames are blacked out.
- **CausalConv3d** is a "forward integrator": each output frame reads only the last K real frames — never the future.
- **UpsampleCausal3D's first-frame trick** is like the title sequence: the first frame has no predecessor, so you can't use motion-difference upsampling on it; you must upsample by spatial detail alone. Later frames can lean on "what frame t-1 looked like."

Together they turn an **offline 3D net** into one that **runs frame by frame, never peeking forward**.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

**`temporal-compression` 是 nanoWAM 课程里 build plan 末尾(`depends_on: [vae-encoder-decoder]`)**。它的角色是**让你的 3D VAE 可以在推理时按帧流式跑**,不需要等整段视频。如果你跳过这个组件,只能拿离线 encode 过的 latent 训 DiT,推理时也要先攒齐 T 帧再编码 — latency 完全爆掉。

你需要的输入是一个**已经训好的 3D VAE**(`vae-encoder-decoder` slot 的产物)。输出是**同一个 VAE 但是时间因果**。两条改造路线:

1. **训练时改造**(Open-Sora 路线 — 今天这段代码):mask + asymmetric padding + first-frame split,**网络结构本身因果**。优点:推理 latency 干净,无 cache 状态需要管理。缺点:重训整个 VAE,昂贵。
2. **推理时改造**(Wan2.1 路线 — 2026-05-29 已讲):**保留对称 padding 的 VAE**,在推理时维护一个 `feat_cache`(像 LM 的 KV cache),每帧推进只算新的部分,缓存旧的部分。优点:可以复用已训好的非因果 VAE。缺点:cache 状态管理复杂,内存占用大。

**生产级实现还要补**:`prepare_causal_attention_mask` 改成 `torch.arange` 向量化版本(O(1) GPU);时间维 stride > 1 的下采样(downsample)同样要因果改造,本文没贴出来;首帧 mask 还要处理"视频开头是 padding 还是真实首帧"的 metadata;再加上一个 streaming inference loop 把这套组件串起来(就是 sampler-inference slot)。

**`temporal-compression` sits late in the nanoWAM build plan (`depends_on: [vae-encoder-decoder]`)**. Its role: **make your 3D VAE runnable frame-by-frame at inference**, without waiting for the full clip. Skip this and you must batch up T frames before encoding — latency explodes.

Inputs: a trained 3D VAE (the `vae-encoder-decoder` output). Outputs: **the same VAE, but temporally causal**. Two routes:

1. **Train-time** (Open-Sora — today's code): mask + asymmetric padding + first-frame split, **causality built into the network**. Pro: clean inference latency, no cache state. Con: retrain the VAE from scratch — expensive.
2. **Inference-time** (Wan2.1 — covered 2026-05-29): **keep the symmetric-padding VAE**, maintain a `feat_cache` (like an LM's KV cache) at inference, advance one frame at a time, reuse cached features. Pro: works with off-the-shelf non-causal VAEs. Con: complex cache state, high memory.

**Production also needs**: vectorize `prepare_causal_attention_mask` with `torch.arange` (O(1) on GPU); apply the same asymmetric padding to time-stride downsamples (not shown); handle metadata about whether "frame 0" is padding or a true keyframe; and wire a streaming inference loop on top (that's the `sampler-inference` slot).

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv3d(c_in, c_out, k)
    def forward(self, x):
        # asymmetric time padding: all on the past side
        x = F.pad(x, (self.k//2, self.k//2, self.k//2, self.k//2, self.k-1, 0), mode="replicate")
        return self.conv(x)

def causal_mask(T, n_hw):
    L = T * n_hw
    m = torch.full((L, L), float("-inf"))
    idx = torch.arange(L) // n_hw
    valid = (torch.arange(L)[None, :] < (idx[:, None] + 1) * n_hw)
    return torch.where(valid, 0.0, m)

# verify: poking a future frame should not move past outputs
conv = CausalConv3d(1, 1, k=3)
x = torch.zeros(1, 1, 5, 8, 8)
y0 = conv(x).detach().clone()
x[..., 4, :, :] = 1.0  # poke the last (future-most) frame only
y1 = conv(x).detach()
print("past frames unchanged?:", torch.allclose(y0[..., :3, :, :], y1[..., :3, :, :]))
print("mask shape:", causal_mask(T=4, n_hw=4).shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
past frames unchanged?: True
mask shape: torch.Size([16, 16])
```

中文:第一句确认了因果性 — 戳一下最后一帧,之前的帧输出**纹丝不动**,真的看不到未来。如果你把 padding 换回 `(k//2, k//2)`,这个断言就会失败,你可以亲手验证。

English: the first line confirms causality — poking the final frame leaves earlier outputs **untouched**, proving the network really can't see the future. Swap padding back to `(k//2, k//2)` and the assertion flips — a hands-on demonstration of what asymmetric padding buys you.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 的 `feat_cache` 流式推理**(2026-05-29 已讲) / **Wan2.1 streaming cache (covered 2026-05-29)**: 中文: 同一个 slot 的"推理时"路线,跟今天的"训练时"路线互补。 / English: the inference-side version of the same slot — Wan2.1 keeps a non-causal VAE and adds a streaming cache; Open-Sora bakes causality into the net.
- **GPT 的 causal attention mask** / **GPT 的文本因果掩码**: 中文: 完全同构 — 一个对 token,一个对帧块。 / English: structurally identical — one masks tokens, the other masks frame blocks.
- **WaveNet 的 dilated causal Conv1d** / **WaveNet 的扩张因果 1D 卷积**: 中文: WaveNet 在音频领域早就用过 `(kernel-1, 0)` 这种不对称 padding,Open-Sora 把它推广到 3D。 / English: WaveNet used the same `(kernel-1, 0)` trick on audio years ago — Open-Sora generalizes the idea to 3D.
- **LightX2V 的 streaming WAM 推理**(2026-06-08 / 11 trending 都见过) / **LightX2V's streaming WAM inference**: 中文: 同样靠"网络本身因果 + 帧级推进"完成实时推理。 / English: same "network-level causality + per-frame advance" recipe for realtime inference.

## 注意事项 / Caveats / when it breaks

- **空间维度不能也因果化 / Don't try to make space causal**:
  - 中文: 空间没有"过去/未来"的物理意义,如果你把 H/W 也搞成因果,会破坏整张图的对称性,效果会暴跌。
  - English: spatial dims have no "past/future" meaning — adding causal masking there ruins spatial symmetry and tanks quality.
- **First-frame 2D 上采样会引入伪影 / First-frame 2D upsample leaves visible artifacts**:
  - 中文: 因为第一帧的时间维只有它自己,放大后的细节质感会和其他帧不太一致。生产里有时会刻意 padding 一帧或让首帧多走一个 refine 网络。
  - English: with only itself to interpolate from, the first frame's upsampled texture often looks slightly off from later frames. Production sometimes pads a synthetic past frame or routes the first frame through a refine net.
- **`prepare_causal_attention_mask` 的 Python loop 很慢 / Python loop kills throughput**:
  - 中文: 那个 `for i in range(seq_len)` 在 65k seq_len 下是真的会慢。一定要用 `arange` 改成 tensor 版。
  - English: that `for i in range(seq_len)` is genuinely slow at seq_len = 65k. Vectorize with `arange + broadcasting`.
- **Downsample 也要改 / Downsample paths need the same treatment**:
  - 中文: encoder 里时间维下采样(stride=2)也要因果改造,否则下采样窗口跨了未来。本文片段没贴 downsample,但同名文件里有 `DownsampleCausal3D`。
  - English: time-stride-2 downsamples in the encoder need the same asymmetric treatment — otherwise the downsample window straddles future frames. Not in today's snippet, but `DownsampleCausal3D` lives in the same file.

## 延伸阅读 / Further reading

- [Hunyuan VAE — *HunyuanVideo: A Systematic Framework For Large Video Generation Model*](https://arxiv.org/abs/2412.03603)
- [Wan2.1 feat_cache (2026-05-29 note)](2026-05-29-wan21-resample-streaming-cache.md)
- [WaveNet — *A Generative Model for Raw Audio* (van den Oord et al.)](https://arxiv.org/abs/1609.03499)
- [Open-Sora design doc — HunyuanVAE integration](https://github.com/hpcaitech/Open-Sora/blob/main/docs/zh_CN/hunyuan_video.md)
