---
date: 2026-07-22
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/hunyuan_vae/unet_causal_3d_blocks.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/unet_causal_3d_blocks.py#L40-L128
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, wam, temporal-compression, causal-3d-conv]
build_role: temporal-compression
---

# Open-Sora CausalConv3d：时间维只向过去补 padding / Open-Sora CausalConv3d: Pad Time Only Toward the Past

> **一句话 / In one line**: `CausalConv3d` 在时间维只给过去补 `(kernel_size - 1)` 帧，让视频 VAE 压缩时不会偷看未来。 / `CausalConv3d` pads only the past side of the time dimension with `(kernel_size - 1)` frames, so a video VAE does not peek into future frames during compression.

## 为什么重要 / Why this matters

WAM 的 VAE 不只是把图片压小，它还要把视频沿时间维压成 latent。如果卷积在时间维左右对称 padding，当前 latent 会混入未来帧信息，在线预测或动作条件 rollout 时就会泄漏答案。因果 3D 卷积是 temporal compression 的底层安全阀。

A WAM VAE does more than compress images; it compresses video along time into latents. If temporal convolution pads symmetrically, the current latent can mix in future frames, leaking answers during online prediction or action-conditioned rollout. Causal 3D convolution is the low-level safety valve for temporal compression.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/hunyuan_vae/unet_causal_3d_blocks.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/unet_causal_3d_blocks.py#L40-L128)

```python
# Simplified teaching slice, preserving the padding idea.
def prepare_causal_attention_mask(n_frame, n_hw, dtype, device):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    return mask

class CausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = (
            kernel_size // 2, kernel_size // 2,
            kernel_size // 2, kernel_size // 2,
            kernel_size - 1, 0,
        )  # W, H, T
        self.time_causal_padding = padding
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        return self.conv(x)
```

## 逐行讲解 / What's happening

1. **attention mask 也按帧因果 / The attention mask is causal by frame**: 中文: 第 `i` 个 token 只能看自己所在帧及之前的所有空间 token。 English: token `i` can see only its own frame and earlier frames' spatial tokens.
2. **空间 padding 仍然对称 / Spatial padding stays symmetric**: 中文: 宽高方向可以看邻域左右上下，因为这不会泄漏未来时间。 English: width and height can use left/right and up/down context because that does not leak future time.
3. **时间 padding 是 `(kernel_size - 1, 0)` / Temporal padding is `(kernel_size - 1, 0)`**: 中文: 只在过去一侧补帧，卷积窗口不会覆盖未来帧。 English: frames are padded only on the past side, so the convolution window never covers future frames.
4. **`replicate` 补第一帧 / `replicate` pads from the first frame**: 中文: 视频开头没有历史帧，就复制边界，避免引入全零突变。 English: the start of the video has no history, so boundary replication avoids an artificial zero jump.

## 类比 / The analogy

像写行车记录。你可以参考当前时刻以前的记录，也可以看同一时刻的左右车道，但不能翻到明天的记录再决定今天怎么开。

It is like writing a driving log. You may use records up to the current moment and nearby lanes at the same moment, but you cannot read tomorrow's log to decide how to drive today.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `temporal-compression` 组件。你的 nanoWAM 可以先实现一个小 `CausalConv3dEncoder`：输入 `[B, C, T, H, W]` 视频和动作条件，输出更短的 `[B, Z, T/r, H/s, W/s]` latent。上游是原始视频帧和可能的动作 token；下游是 DiT 或 WAM backbone。如果省掉因果性，离线重建可能更好看，但在线世界模型会把未来泄漏进当前状态。

This is the `temporal-compression` component. A nanoWAM can start with a small `CausalConv3dEncoder`: input `[B, C, T, H, W]` video plus optional action conditioning, output a shorter `[B, Z, T/r, H/s, W/s]` latent. Upstream are raw frames and possible action tokens; downstream is the DiT or WAM backbone. Without causality, offline reconstruction may look better, but an online world model leaks future information into the present state.

## 自己跑一遍 / Try it yourself

```python
def causal_window(frames, t, kernel=3):
    left = kernel - 1
    padded = [frames[0]] * left + frames
    start = t
    return padded[start:start + kernel]

frames = ["f0", "f1", "f2", "f3"]
for t in range(len(frames)):
    print(t, causal_window(frames, t))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 ['f0', 'f0', 'f0']
1 ['f0', 'f0', 'f1']
2 ['f0', 'f1', 'f2']
3 ['f1', 'f2', 'f3']
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-Sora causal VAE** / **Open-Sora causal VAE**: 中文: 编码器和解码器都要维持时间方向的因果性。 / English: both encoder and decoder need to preserve temporal causality.
- **Wan2.1 temporal VAE chunks** / **Wan2.1 temporal VAE chunks**: 中文: 长视频编码时用时间块和 cache 控制显存与连续性。 / English: long-video encoding uses temporal chunks and cache to control memory and continuity.
- **streaming world models** / **streaming world models**: 中文: 在线 rollout 的 latent 只能来自过去观测和当前动作。 / English: online rollout latents must come only from past observations and current actions.

## 注意事项 / Caveats / when it breaks

- **kernel size 假设要明确 / Kernel-size assumptions must be explicit**: 原文件的 padding 写法假设整型 kernel；tuple kernel 需要分别处理。 / The original padding shape assumes an integer kernel; tuple kernels need per-axis handling.
- **边界复制不是唯一选择 / Replication is not the only boundary rule**: 也可以用零 padding 或 learned initial state，但统计分布不同。 / Zero padding or a learned initial state are possible, but they change statistics.
- **重建质量和因果性会拉扯 / Reconstruction quality and causality can trade off**: 对称卷积可能更容易重建，但不适合在线预测。 / Symmetric convolution may reconstruct better, but it is unsuitable for online prediction.

## 延伸阅读 / Further reading

- [Open-Sora CausalConv3d](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/unet_causal_3d_blocks.py#L40-L128)

