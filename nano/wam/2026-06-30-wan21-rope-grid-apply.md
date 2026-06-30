---
date: 2026-06-30
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L26-L65
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, patchify-positional, rope]
build_role: patchify-positional
---

# nanoWAM 的 3D 坐标层：RoPE 不只是一维位置 / nanoWAM's 3D Coordinate Layer: RoPE Is Not Just One-Dimensional Position

> **一句话 / In one line**: Wan2.1 的 `rope_apply` 把每个视频 token 的 `(frame, height, width)` 坐标变成复数旋转。 / Wan2.1's `rope_apply` turns each video token's `(frame, height, width)` coordinate into a complex rotation.

## 为什么重要 / Why this matters

WAM 要预测“动作之后视频会怎样变”，因此 token 必须知道自己处在第几帧和画面哪个位置。没有三维位置，模型只能看到一串扁平 token，很难区分“下一帧同一位置”和“同一帧相邻位置”。

A WAM predicts how video changes after actions, so each token needs to know its frame and spatial location. Without 3D position, the model only sees a flat token string and struggles to distinguish "same location next frame" from "neighboring location in the same frame."

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L26-L65)

```python
@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()
```

## 逐行讲解 / What's happening

1. **第 29-32 行 / Lines 29-32**:
   - 中文: `torch.polar` 把角度表变成单位复数，后面乘一下就是旋转。
   - English: `torch.polar` turns phase angles into unit complex numbers, so later application is multiplication.
2. **第 39 行 / Line 39**:
   - 中文: 通道数按三维坐标拆分，时间轴拿到余下的主通道。
   - English: Channels are split across the three axes, with the time axis receiving the remainder.
3. **第 47-53 行 / Lines 47-53**:
   - 中文: 每个轴先单独 expand，再 cat 成每个 token 的完整坐标频率。
   - English: Each axis is expanded independently, then concatenated into the full frequency vector for each token.

## 类比 / The analogy

这像电影院座位票：排号、列号、场次都写在票上，只写一个流水号当然也能入场，但找座位会慢很多。

It is like a cinema ticket with row, seat, and showtime. A single serial number could identify the seat, but the structured coordinates make lookup much easier.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `patchify-positional` 层。VAE 或 patch embedder 先把视频压成 latent tokens；RoPE 再给这些 tokens 标注三维坐标；后面的 DiT block 才能用 attention 比较正确的时空邻居。

This is the `patchify-positional` layer. A VAE or patch embedder first turns video into latent tokens; RoPE tags those tokens with 3D coordinates; later DiT blocks can then compare the right spatiotemporal neighbors.

## 自己跑一遍 / Try it yourself

```python
import torch

F, H, W = 2, 3, 4
coords = torch.stack(torch.meshgrid(torch.arange(F), torch.arange(H), torch.arange(W), indexing="ij"), dim=-1)
flat = coords.reshape(-1, 3)
print(flat[:5].tolist())
print(flat.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0]]
torch.Size([24, 3])
```

RoPE 做的事情就是把这些整数坐标变成 attention 可用的旋转相位。

RoPE's job is to convert these integer coordinates into rotary phases usable by attention.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 self-attention** / **Wan2.1 self-attention**: Q/K 在进入 flash attention 前调用同一个 `rope_apply`。 / Q/K call the same `rope_apply` before entering flash attention.
- **Open-Sora multi-axis RoPE** / **Open-Sora multi-axis RoPE**: 同样把视频的时空结构显式塞进 token。 / It also explicitly injects video spatiotemporal structure into tokens.

## 注意事项 / Caveats / when it breaks

- **padding token 要排除** / **Padding tokens must be excluded**: `seq_len = f * h * w` 后面的 padding 被拼回去但不旋转。 / Tokens after `seq_len = f * h * w` are concatenated back without rotation.
- **坐标顺序要和 patchify 一致** / **Coordinate order must match patchify order**: flatten 顺序错了，位置会系统性错配。 / If flattening order differs, positions are systematically mismatched.

## 延伸阅读 / Further reading

- [Wan2.1 RoPE implementation](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py)

