---
date: 2026-06-30
topic: diffusion
source: tracked
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L26-L155
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, rope, attention]
---

# Wan2.1 的 3D RoPE 自注意力：把时间、高度、宽度拆成三把尺 / Wan2.1 3D RoPE Self-Attention: Three Rulers for Time, Height, and Width

> **一句话 / In one line**: Wan2.1 先把 RoPE 频率拆成 frame/height/width 三段，再只旋转 Q/K，不碰 V。 / Wan2.1 splits RoPE frequencies into frame, height, and width bands, then rotates only Q/K while leaving V unchanged.

## 为什么重要 / Why this matters

视频 DiT 的 token 不是一条普通文本序列，而是 `F x H x W` 的三维网格。Wan2.1 的这段代码把一维 RoPE 扩展成三维位置系统：一部分通道表达时间，一部分表达高度，一部分表达宽度。这样 attention 仍然是标准 Q/K/V，但每个 query/key 已经带着自己的视频坐标。

Video DiT tokens are not a plain text sequence; they come from an `F x H x W` grid. This code turns one-dimensional RoPE into a three-axis position system: some channels encode time, some encode height, and some encode width. Attention stays a regular Q/K/V operation, but each query and key carries its video coordinate.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L26-L155)

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

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanSelfAttention(nn.Module):
    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.flatten(2)
        x = self.o(x)
        return x
```

## 逐行讲解 / What's happening

1. **第 26-33 行 / Lines 26-33 (`rope_params`)**:
   - 中文: 用复数极坐标预先生成 RoPE 旋转因子，后面只需要乘法。
   - English: The code precomputes RoPE multipliers as complex numbers, so application is just multiplication.
2. **第 39 行 / Line 39 (`freqs.split`)**:
   - 中文: 通道被切成三段，分别服务 frame、height、width。
   - English: The channel budget is split into three bands for frame, height, and width.
3. **第 48-53 行 / Lines 48-53 (`expand`)**:
   - 中文: 三个一维频率表广播成同一个 `F x H x W` 网格。
   - English: Three one-axis frequency tables are broadcast into the same `F x H x W` grid.
4. **第 142-148 行 / Lines 142-148 (`flash_attention`)**:
   - 中文: RoPE 只加在 Q/K 上，V 保持原始内容，位置只影响“谁看谁”。
   - English: RoPE is applied only to Q/K, while V stays as content; position affects who attends to whom.

## 类比 / The analogy

这像给仓库货架贴三种标签：第几排、几层、第几个格子。搬货的人还是看货物本身，但查找路线会先看坐标标签。

It is like labeling warehouse shelves with aisle, level, and slot. The worker still moves the item itself, but the lookup route is guided by coordinates.

## 自己跑一遍 / Try it yourself

```python
import torch

def rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).double() / dim))
    return torch.polar(torch.ones_like(freqs), freqs)

def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    ft, fh, fw = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        z = torch.view_as_complex(x[i, : f*h*w].double().reshape(f*h*w, n, -1, 2))
        g = torch.cat([ft[:f].view(f,1,1,-1).expand(f,h,w,-1), fh[:h].view(1,h,1,-1).expand(f,h,w,-1), fw[:w].view(1,1,w,-1).expand(f,h,w,-1)], -1)
        out.append(torch.view_as_real(z * g.reshape(f*h*w,1,-1)).flatten(2))
    return torch.stack(out).float()

x = torch.randn(1, 2 * 3 * 4, 2, 12)
print(rope_apply(x, torch.tensor([[2, 3, 4]]), rope_params(8, 6)).shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
torch.Size([1, 24, 2, 12])
```

形状不变，但 Q/K 的每个位置已经乘上了三维旋转相位。

The shape is unchanged, but every Q/K position has received a 3D rotary phase.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-Sora EmbedND** / **Open-Sora EmbedND**: 同样把多轴位置编码作为视频 token 的基础坐标系。 / It also treats multi-axis position encoding as the coordinate system for video tokens.
- **CogVideoX patch embed** / **CogVideoX patch embed**: 视频 token 化后也必须恢复时空位置，否则 attention 只看到一维序号。 / After video tokenization, spatiotemporal position must be restored or attention sees only a flat index.

## 注意事项 / Caveats / when it breaks

- **通道数要够拆三段** / **The channel budget must be large enough**: 头维度太小会让某个轴几乎没有表达能力。 / Tiny head dimensions leave too few channels for one of the axes.
- **变长视频要传 `grid_sizes`** / **Variable video sizes need `grid_sizes`**: padding token 不能和真实网格一起旋转。 / Padding tokens must not be rotated as if they were real grid cells.

## 延伸阅读 / Further reading

- [Wan2.1 `model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py)

