---
date: 2026-07-24
topic: infrastructure
source: tracked
repo: meta-pytorch/torchtune
file: torchtune/modules/position_embeddings.py
permalink: https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/position_embeddings.py#L44-L107
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, infrastructure, torchtune, rope, attention]
---

# torchtune RoPE cache：把旋转角提前烤好 / torchtune RoPE Cache: Bake Rotation Angles Ahead of Time

> **一句话 / In one line**: `RotaryPositionalEmbeddings` 先把每个位置的 cos/sin 表缓存起来，forward 时只按位置切片并广播到 attention head。 / `RotaryPositionalEmbeddings` caches the cos/sin table for every position, then slices and broadcasts it over attention heads during `forward`.

## 为什么重要 / Why this matters

RoPE 本质上是给 Q/K 的每两个通道做二维旋转。训练大模型时，这个旋转每天会被调用无数次；torchtune 把角频率和 cos/sin 表提前注册成 buffer，既减少重复计算，也让 packed sample 的 `input_pos` 能走同一套代码。

RoPE rotates every pair of Q/K channels. In large-model training this runs constantly, so torchtune stores the frequency table and cos/sin cache as buffers. That removes repeated math while still supporting packed samples through `input_pos`.

## 代码 / The code

`meta-pytorch/torchtune` — [`torchtune/modules/position_embeddings.py`](https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/position_embeddings.py#L44-L107)

```python
    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
```

## 逐行讲解 / What's happening

1. **第 45-50 行 / Lines 45-50 (`theta`)**: 中文: 每两个 hidden 通道共用一个角频率，频率按几何级数下降。 English: each pair of hidden channels gets one angular frequency, decreasing geometrically.
2. **第 51 行 / Line 51 (`register_buffer`)**: 中文: `theta` 跟着 device/dtype 迁移，但不是可训练参数。 English: `theta` moves with the module's device and dtype, but is not trainable.
3. **第 60-68 行 / Lines 60-68 (`idx_theta`)**: 中文: 外积一次生成 `[position, frequency]` 的角度表。 English: one outer product builds the `[position, frequency]` angle table.
4. **第 72-73 行 / Lines 72-73 (`cache`)**: 中文: cos 和 sin 放在最后一维，后面可以像复数乘法一样套公式。 English: cos and sin live in the last dimension, so the forward path can apply complex-rotation algebra.
5. **第 86-88 行 / Lines 86-88 (`input_pos`)**: 中文: 普通训练切前 `seq_len` 个位置，packed sample 则按显式 position id gather。 English: normal training slices the first `seq_len` positions; packed samples gather explicit position ids.

## 类比 / The analogy

像餐厅提前把一天会用到的酱汁按编号装进小碟。真正出餐时，厨师只拿对应编号的碟子，不再现场从头调味。

It is like a restaurant preparing numbered sauce cups before service. During service, the cook grabs the right cup instead of mixing the sauce from scratch.

## 自己跑一遍 / Try it yourself

```python
import math

dim = 4
theta = [1 / (10000 ** (i / dim)) for i in range(0, dim, 2)]
cache = [[(round(math.cos(pos * t), 3), round(math.sin(pos * t), 3)) for t in theta] for pos in range(3)]

x0, x1 = 1.0, 0.0
cos, sin = cache[2][0]
rotated = (x0 * cos - x1 * sin, x1 * cos + x0 * sin)
print(cache)
print(rotated)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[(1.0, 0.0), (1.0, 0.0)], [(0.54, 0.841), (1.0, 0.01)], [(-0.416, 0.909), (1.0, 0.02)]]
(-0.416, 0.909)
```

中文: 注意位置 2 的第一对通道被旋转到接近单位圆上的 `(-0.416, 0.909)`。 English: notice how position 2 rotates the first channel pair to roughly `(-0.416, 0.909)` on the unit circle.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 3D RoPE** / **Wan2.1 3D RoPE**: 视频模型会把时间、高、宽三套位置频率组合起来。 / video models combine separate temporal, height, and width frequencies.
- **DINOv3 RoPE attention** / **DINOv3 RoPE attention**: 图像 prefix token 可以不旋转，只旋转 patch token。 / image prefix tokens can stay unrotated while patch tokens get RoPE.

## 注意事项 / Caveats / when it breaks

- **超过缓存长度 / Past cache length**: 如果实际序列超过 `max_seq_len`，必须重建 cache。 / if the real sequence exceeds `max_seq_len`, the cache has to be rebuilt.
- **head_dim 必须成对 / Paired head dimensions**: RoPE 按两个通道一组旋转，奇数维会让 reshape 契约变复杂。 / RoPE rotates channel pairs, so odd dimensions complicate the reshape contract.
- **packed sample 要传 `input_pos` / Packed samples need `input_pos`**: 否则不同样本拼在一起后会共享错误的位置编号。 / without `input_pos`, concatenated samples inherit wrong position ids.

## 延伸阅读 / Further reading

- [torchtune position_embeddings.py](https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/position_embeddings.py#L44-L107)

