---
date: 2026-08-07
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L421-L507
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, patchify-positional]
build_role: patchify-positional advanced variant, 3D patch embedding plus manually managed RoPE frequency table
---

# Wan2.1 patch/freq buffer：视频切块后还要带上三维坐标尺 / Wan2.1 Patch/Freq Buffer: Patch Video Tokens Need 3D Rulers

> **一句话 / In one line**: Wan2.1 先用 `Conv3d` 把视频变成 token，再维护一张按时间/高度/宽度拆分的 RoPE 频率表。 / Wan2.1 turns video into tokens with `Conv3d`, then keeps a RoPE frequency table split across time, height, and width.

## 为什么重要 / Why this matters

WAM 的 DiT 主干不是直接看像素帧，而是看 latent patch token。每个 token 来自某个时间、行、列位置；如果坐标频率表和 patch 网格不对齐，attention 会失去时空顺序感。

A WAM DiT backbone does not read frames directly; it reads latent patch tokens. Each token comes from a time, row, and column position. If the frequency table and patch grid are misaligned, attention loses spacetime order.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L421-L507)

```python
self.patch_embedding = nn.Conv3d(
    in_dim, dim, kernel_size=patch_size, stride=patch_size
)

d = dim // num_heads
self.freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
], dim=1)

x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
grid_sizes = torch.stack([torch.tensor(u.shape[2:]) for u in x])
```

## 逐行讲解 / What's happening

1. **第 421-423 行 / Lines 421-423 (patchify)**:
   - 中文: `Conv3d` 的 kernel 和 stride 都等于 `patch_size`，所以一次卷积就是一次无重叠时空切块。
   - English: `Conv3d` uses `patch_size` as both kernel and stride, so one convolution performs non-overlapping spacetime patching.
2. **第 441-449 行 / Lines 441-449 (frequency table)**:
   - 中文: 频率表没有用 `register_buffer`，注释说明是为了避免 `.to()` 改 dtype。
   - English: The frequency table is not registered as a buffer; the comment says this avoids dtype changes from `.to()`.
3. **第 487-489 行 / Lines 487-489 (device move)**:
   - 中文: forward 时手动把 `freqs` 挪到 patch embedding 的 device。
   - English: Forward manually moves `freqs` to the patch embedding device.
4. **第 493-507 行 / Lines 493-507 (grid and padding)**:
   - 中文: 每个视频先独立 patchify，再记录 `[F,H,W]`，最后补到统一 `seq_len`。
   - English: Each video is patchified independently, records `[F,H,W]`, then pads to a shared `seq_len`.

## 类比 / The analogy

像把视频胶片切成小格：切完不能只知道“第几个格子”，还要知道它来自第几帧、第几行、第几列。

It is like cutting film into small cells: after cutting, you need more than a flat index; you need frame, row, and column.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，它是 `patchify-positional` 的核心入口。VAE latent 进入 DiT 前，先变成 `[token, channel]` 序列，同时生成后续 RoPE attention 需要的三维 grid。这个边界决定模型是否真的知道“动作发生在视频里的什么位置”。

In a nanoWAM, this is the core `patchify-positional` entry. Before VAE latents enter the DiT, they become a `[token, channel]` sequence and carry the 3D grid needed by later RoPE attention. This boundary determines whether the model knows where events happen in the video.

## 自己跑一遍 / Try it yourself

```python
def patch_grid(frames, height, width, pt, ph, pw):
    return (frames // pt, height // ph, width // pw)

grid = patch_grid(16, 32, 48, 4, 8, 8)
tokens = grid[0] * grid[1] * grid[2]
print(grid, tokens)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(4, 4, 6) 96
```

96 个 token 不是一条无意义序列，而是 4 x 4 x 6 的时空网格。

The 96 tokens are not a meaningless flat list; they are a 4 x 4 x 6 spacetime grid.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT patch embed** / **DiT patch embed**: 图像扩散模型也先 patchify，只是少了时间维。 / Image diffusion models also patchify first, but without the time axis.
- **Open-Sora PatchEmbed3D** / **Open-Sora PatchEmbed3D**: 同样把视频 tubelet 变成 transformer token。 / It also turns video tubelets into transformer tokens.

## 注意事项 / Caveats / when it breaks

- **seq_len 必须够大** / **`seq_len` must be large enough**: 代码会断言最大 token 数不超过限制。 / The code asserts that the maximum token count fits the limit.
- **手动 buffer 要小心 dtype/device** / **Manual buffers need dtype/device care**: 不注册 buffer 就要自己处理迁移。 / If a table is not registered as a buffer, migration is your job.

## 延伸阅读 / Further reading

- [Wan2.1 model patch embedding](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L421-L507)

