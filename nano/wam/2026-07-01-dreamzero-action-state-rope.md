---
date: 2026-07-01
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/wan2_1_submodule.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan2_1_submodule.py#L99-L159
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, rope]
build_role: action/state positional conditioning for a video-action DiT
---

# DreamZero action/state RoPE：把控制 token 接进视频坐标系 / DreamZero Action/State RoPE: Splice Control Tokens into the Video Coordinate System

> **一句话 / In one line**: 它先为 action/state register 取一维 RoPE，再拼到视频 token 的 RoPE 后面，让整条混合序列共享旋转位置编码。 / It takes 1D RoPE for action/state registers, appends it after video-token RoPE, and lets the mixed sequence share rotary positions.

## 为什么重要 / Why this matters

WAM 不只预测视频，也要读动作和状态。DreamZero 这段代码在 RoPE 应用前扩展 `freqs`：先算 action register 覆盖几个 chunk，再取 action 和 state 的频率片段拼到原视频频率后面，最后按标准 RoPE 旋转 Q/K。

A WAM does not only predict video; it also reads actions and state. This DreamZero code extends `freqs` before applying RoPE: compute how many chunks the action registers cover, slice action and state frequency segments, append them after video frequencies, then rotate Q/K with standard RoPE.

## 代码 / The code

`dreamzero0/dreamzero` — [`groot/vla/model/dreamzero/modules/wan2_1_submodule.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan2_1_submodule.py#L99-L159)

```python
def rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    B, seq_len, n, D = x.shape

    if action_register_length is not None:
        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)
        freqs_1d_action = freqs_action[:chunk_size * num_action_per_block]
        freqs_1d_state = freqs_state[:chunk_size * num_state_per_block]
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)

    # Reshape freqs to be broadcastable: (1, seq_len, 1, D)
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    x0, x1 = x.chunk(2, dim=-1)
    freqs_cos, freqs_sin = freqs.chunk(2, dim=-1)

    rotated_x0 = x0 * freqs_cos - x1 * freqs_sin
    rotated_x1 = x1 * freqs_cos + x0 * freqs_sin
    x_rotated = torch.cat((rotated_x0, rotated_x1), dim=-1)

    return x_rotated


# @amp.autocast(enabled=False)
def rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert num_action_per_block is not None
        assert num_state_per_block is not None

        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)

        freqs_1d_action = freqs_action[:chunk_size * num_action_per_block].view(chunk_size * num_action_per_block, 1, -1)
        freqs_1d_state = freqs_state[:chunk_size * num_state_per_block].view(chunk_size * num_state_per_block, 1, -1)
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x
```

## 逐行讲解 / What's happening

1. **第 110-115 行 / Lines 110-115**: 中文: 非 polar 路径先按 `action_register_length` 反推出 chunk 数，再把 action/state 频率拼到 `freqs` 后。 / English: The non-polar path derives chunk count from `action_register_length`, then appends action/state frequencies to `freqs`.
2. **第 119-124 行 / Lines 119-124**: 中文: 把通道拆成实部/虚部，手写二维旋转公式。 / English: It splits channels into real/imaginary halves and applies the 2D rotation formula manually.
3. **第 142-158 行 / Lines 142-158**: 中文: polar 路径把最后一维变成 complex，用复数乘法完成同一件事。 / English: The polar path views the last dimension as complex numbers and performs the same rotation with complex multiplication.

## 类比 / The analogy

像给视频帧、动作和状态都发座位号：视频坐前排，动作和状态接在后排；注意力看见的是同一个剧场里的有序座位。

It is like assigning seats to video frames, actions, and state: video sits in the front rows, actions and state continue behind them, and attention sees ordered seats in one theater.


## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文: 这属于 `action-conditioning` 和 `patchify-positional` 的交界层。视频 token 已经有 RoPE 坐标，动作/state register 也需要自己的坐标，否则 DiT 只能看到“有一串额外 token”，却不知道这些 token 在 action horizon 里的顺序。生产级实现还要把 action/state 的频率尺度和视频帧率、控制频率对齐。

English: This sits between `action-conditioning` and `patchify-positional`. Video tokens already have RoPE coordinates; action and state registers need their own coordinates too. Otherwise the DiT only sees extra tokens without knowing their order in the action horizon. A production version must align action/state frequency scales with video frame rate and control rate.


## 自己跑一遍 / Try it yourself

```python
import torch
video = torch.arange(3).view(3,1)
action = torch.arange(4).view(4,1) + 10
state = torch.arange(2).view(2,1) + 20
freqs = torch.cat([video, action, state], dim=0)
print(freqs.squeeze().tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
[0, 1, 2, 10, 11, 12, 13, 20, 21]
```

中文: 这个小例子保留了源码里的关键控制流，但把依赖压到最低，便于你直接观察形状、索引或状态变化。

English: The miniature keeps the original control-flow idea while stripping dependencies down so the shape, index, or state change is visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 3D RoPE** / **Wan2.1 3D RoPE**: 中文: 视频 token 按时间、高度、宽度拆频率。 / English: Video tokens split frequencies across time, height, and width.
- **VLA action tokens** / **VLA action tokens**: 中文: 离散动作 token 也需要位置，否则 horizon 顺序会丢。 / English: Discrete action tokens also need positions or horizon order is lost.

## 注意事项 / Caveats / when it breaks

- **长度必须匹配 / Lengths must match**: 中文: 拼接后的 `freqs` 长度必须覆盖混合 token 序列。 / English: The concatenated `freqs` must cover the mixed-token sequence.
- **频率尺度不是随便选 / Frequency scale is not arbitrary**: 中文: action/control rate 和 video fps 差很多时，频率设计会影响泛化。 / English: When action/control rate differs from video FPS, frequency design affects generalization.

## 延伸阅读 / Further reading

- Source permalink above.
- Project repository linked from the frontmatter.
