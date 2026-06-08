---
date: 2026-06-08
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L31-L44
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, dit, rope, positional-encoding]
build_role: patchify + positional encoding — a single module that does N-dimensional RoPE; same code drives text (1D), image (2D), or video (3D)
---

# 一份 14 行的 EmbedND:文本 / 图像 / 视频共用一个 RoPE 模块 / 14 lines of EmbedND: text, image, and video share one RoPE module

> **一句话 / In one line**: Open-Sora 的 MMDiT 把 N 维 RoPE 抽象成「每个轴给一段 channel,for 循环里调同一个 `rope()`,沿 channel 维 concat」,一个 module 同时处理 1D / 2D / 3D。 / Open-Sora's MMDiT abstracts N-D RoPE into "give each axis a slice of the head channels, loop and call the same `rope()`, then concat along the channel dim" — one module covers 1D, 2D, and 3D.

## 为什么重要 / Why this matters

之前在 2026-05-29 那期我们看过 Wan-Video/Wan2.1 的 3D RoPE:`f_freqs / h_freqs / w_freqs` 三个硬编码的 buffer 各自算,最后 concat。能用,但写死了 3 维:今天来一段纯文字 prompt 怎么办?来一张静态图片怎么办?Open-Sora 借了 Flux / Black Forest Labs 那套优雅抽象——`axes_dim: list[int]` 是一个列表,告诉模块「每个轴占用 head_dim 里的多少 channel」。视频是 `[16, 24, 24]`(时间 16 channel,H/W 各 24),纯文本就是 `[head_dim]`,纯图像就是 `[head_dim // 2, head_dim // 2]`。同一个 `EmbedND.forward()` 跑遍三种场景,且 `axes_dim` 总和等于 head_dim 就行。这是 nanoWAM 课程里 `patchify-positional` 这一栏的「优雅版」。

The 2026-05-29 lesson covered Wan-Video/Wan2.1's 3D RoPE: three hardcoded buffers (`f_freqs / h_freqs / w_freqs`), one per axis, computed separately and concatenated. Works, but it's locked at 3D — what if your prompt is text-only? What about a still image? Open-Sora borrows the Flux / Black Forest Labs abstraction: `axes_dim: list[int]` lists how many head channels each axis gets. Video uses `[16, 24, 24]` (time gets 16, H and W get 24 each). Text-only uses `[head_dim]`. Image uses `[head_dim // 2, head_dim // 2]`. The same `EmbedND.forward()` handles all three — as long as `sum(axes_dim) == head_dim`. This is the elegant version of the `patchify-positional` slot in the nanoWAM curriculum.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L31-L44)

```python
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)
```

底层的 `rope()` 和 `apply_rope()`(同目录的 [`math.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/math.py#L50-L65)):

The underlying `rope()` and `apply_rope()` (sibling file [`math.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/math.py#L50-L65)):

```python
def rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
```

## 逐行讲解 / What's happening

### `EmbedND.forward(ids)`

输入:`ids` 形状 `(B, seq_len, n_axes)`。对于视频,`n_axes = 3`,`ids[b, n] = (t_index, h_index, w_index)`。对于纯文本,`n_axes = 1`,`ids[b, n] = (n,)`。

Input: `ids` of shape `(B, seq_len, n_axes)`. For video, `n_axes = 3`, `ids[b, n] = (t_idx, h_idx, w_idx)`. For plain text, `n_axes = 1`, `ids[b, n] = (n,)`.

1. **`n_axes = ids.shape[-1]`**
   - 中文: 不在 `__init__` 里固定轴数,而是从 ids tensor 里看——同一个模块可以在 batch 内动态变维度。
   - English: Number of axes isn't fixed at `__init__` time; it's read from the `ids` tensor at runtime, so the same module can dynamically handle different dimensionalities.

2. **`[rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)]`**
   - 中文: 对第 i 个轴,只取那一维的 position id,调用 `rope()` 得到该轴专属的 (cos, -sin / sin, cos) 旋转矩阵——形状 `(B, seq_len, axes_dim[i]//2, 2, 2)`。
   - English: For axis `i`, slice out just that axis's position ids, call `rope()`, get axis-`i`'s (cos, -sin / sin, cos) rotation tensors — shape `(B, seq_len, axes_dim[i]//2, 2, 2)`.

3. **`torch.cat(..., dim=-3)`**
   - 中文: 沿倒数第三个维度(`axes_dim[i]//2`)拼接。最终 emb 形状 `(B, seq_len, head_dim // 2, 2, 2)`——head_dim 因为正好等于 `sum(axes_dim)` 而对齐。
   - English: Concatenate along axis -3 (`axes_dim[i] // 2`). Final shape: `(B, seq_len, head_dim // 2, 2, 2)`. The total head dim is exactly `sum(axes_dim)`, so the channels line up.

4. **`emb.unsqueeze(1)`**
   - 中文: 在 head 这一维补一个 size-1,后续会 broadcast 到所有 attention head 上。
   - English: Inserts a size-1 axis for "head" so the embedding broadcasts across attention heads later.

### `rope(pos, dim, theta)` — 旋转矩阵 / The rotation matrix

5. **`scale = arange(0, dim, 2) / dim;  omega = 1 / theta**scale`**
   - 中文: 标准 RoPE 频率: `omega_k = 1 / theta^(2k/dim)`。 `theta=10000` 是 RoPE 论文的默认,Open-Sora MMDiT 也照搬。
   - English: Standard RoPE frequencies `omega_k = 1 / theta^(2k/dim)`. `theta = 10000` is the RoPE paper default, kept as-is.

6. **`out = einsum("...n,d->...nd", pos, omega)`**
   - 中文: 外积:位置 × 频率 = 每个位置对每个频率的相位。
   - English: Outer product — position × frequency = the phase angle of each (position, frequency) pair.

7. **`torch.stack([cos, -sin, sin, cos], dim=-1)` + `rearrange(... "b n d (i j) -> b n d i j", i=2, j=2)`**
   - 中文: 把 4 个标量包装成一个 2x2 旋转矩阵 `[[cos, -sin], [sin, cos]]`。最终形状 `(B, seq_len, dim//2, 2, 2)`——每对相邻 channel 一个 2x2 矩阵。
   - English: Pack the 4 scalars into a 2x2 rotation matrix `[[cos, -sin], [sin, cos]]`. Final shape: `(B, seq_len, dim//2, 2, 2)` — one 2x2 matrix per channel pair.

### `apply_rope(xq, xk, freqs_cis)` — 旋转的应用 / Apply the rotation

8. **`xq_ = xq.reshape(..., -1, 1, 2)`** + **`freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]`**
   - 中文: 把 q 的最后一维看成「每两个 channel 一组」,跟旋转矩阵相乘。具体是把 `[xq_0, xq_1]` 这一对转 angle θ:`xq_0' = cos·xq_0 - sin·xq_1`,`xq_1' = sin·xq_0 + cos·xq_1`。代码里实际写的是 `freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]`,因为 stack 的顺序里 `freqs_cis[..., 0]` 已经包含了 `[cos, -sin]`,自动凑成正确的旋转。
   - English: Reshape `q`'s last axis as "every two channels form one pair," then matmul with the rotation. For each pair `[xq_0, xq_1]` rotate by angle θ: `xq_0' = cos·xq_0 - sin·xq_1`, `xq_1' = sin·xq_0 + cos·xq_1`. The actual code reads `freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]` because the `stack` order packed `[cos, -sin]` into `freqs_cis[..., 0]`, so the formula collapses to one weighted sum.

## 类比 / The analogy

想象一个有 N 个旋钮的调音台——每个旋钮控制一段频率范围。Wan2.1 是「一台调音台,3 个旋钮,焊死」(time, H, W)。Open-Sora 的 EmbedND 是「一个可配置的调音台,你买的时候告诉我:文字版只有 1 个大旋钮,图像版 2 个旋钮各占一半,视频版 3 个旋钮按 16/24/24 分」。旋钮的「内部电路」(rope 函数)是一样的;不同的只是面板布局(axes_dim)。

Picture a mixing console with N knobs — each knob controls a band of frequencies. Wan2.1 is a console with three knobs welded in place (time, H, W). Open-Sora's EmbedND is a configurable console where you choose at purchase time: text mode = 1 fat knob, image mode = 2 knobs each owning half, video mode = 3 knobs split 16/24/24. The internal circuitry of each knob (`rope`) is identical; only the front-panel layout (`axes_dim`) changes.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `nano_wam.patchify-positional` 这一栏的「优雅版」(Wan2.1 的硬编码版是另一种实现风格)。无前置依赖:positional encoding 是图块/帧产生 token 之后的第一道工序。下游消费者:`dit-block`——DiT 的 attention 内部要拿这个 `freqs_cis` 调 `apply_rope(q, k)`。

This is the *elegant* implementation of the `nano_wam.patchify-positional` slot (Wan2.1's hardcoded version covers a different style). No prerequisites: positional encoding is the first step after patch tokens are produced. Downstream consumer: `dit-block` — the DiT's attention calls `apply_rope(q, k)` on the `freqs_cis` this module returns.

在 nanoWAM 里,这个模块对应:

In nanoWAM, this module corresponds to:

输入: / Inputs:
- `ids: (B, T*H*W, 3)` ——每个 patch token 的 (t, h, w) 索引,你的 patchifier 负责生成。 / Per-token `(t, h, w)` indices generated by your patchifier.

输出: / Output:
- `freqs_cis: (B, 1, T*H*W, head_dim//2, 2, 2)` ——一个旋转矩阵张量,DiT 的 attention 内部调 `apply_rope(q, k, freqs_cis)` 把它"乘"进 q/k。 / A rotation-matrix tensor; DiT's attention calls `apply_rope(q, k, freqs_cis)` to "rotate" `q` and `k` before the dot product.

最小实现需要:**(1)** `EmbedND(dim, theta, axes_dim=[t_dim, h_dim, w_dim])` 实例,**(2)** 一个 patchifier 负责产生 `ids` 张量,**(3)** 在 DiT block 的 attention 里写 `q, k = apply_rope(q, k, freqs_cis)`。省略 RoPE 的话,你的 nanoWAM 会很快地学到「生成静态视频」——因为它根本没办法区分 t=5 和 t=20 这两帧。生产级 nanoWAM 还要加:**(a)** 把 `rope()` 换成 `liger_rope()`(Triton-fused cos/sin)以省 kernel launch overhead;**(b)** 给 t 轴用更长的 wavelength 以处理几百帧的长视频(`axes_dim[0]` 增大或 `theta` 增大)。

Minimum implementation needs **(1)** one `EmbedND(dim, theta, axes_dim=[t_dim, h_dim, w_dim])` instance, **(2)** a patchifier that produces the `ids` tensor, **(3)** `q, k = apply_rope(q, k, freqs_cis)` inside the DiT block's attention. Omit RoPE and your nanoWAM will rapidly learn to "generate a still video," because it has no way to distinguish frame `t=5` from frame `t=20`. For production: **(a)** swap `rope()` for `liger_rope()` (Triton-fused cos/sin) to save kernel-launch overhead, **(b)** widen `axes_dim[0]` or raise `theta` along the time axis to handle long videos (hundreds of frames).

## 自己跑一遍 / Try it yourself

```python
# pip install torch einops
import torch
from einops import rearrange

def rope(pos, dim, theta=10000):
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega.to(pos.device))
    out = torch.stack([out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

class EmbedND(torch.nn.Module):
    def __init__(self, axes_dim, theta=10000):
        super().__init__()
        self.axes_dim = axes_dim; self.theta = theta
    def forward(self, ids):
        return torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])],
            dim=-3,
        ).unsqueeze(1)

# Same module, three uses:
emb = EmbedND(axes_dim=[16])                       # text-only: 1 axis, 16 channels
ids_text = torch.arange(32).view(1, 32, 1).float()
print("text emb:",  emb(ids_text).shape)           # (1, 1, 32, 8, 2, 2)

emb2 = EmbedND(axes_dim=[32, 32])                  # image: 2 axes
ids_img = torch.cartesian_prod(torch.arange(8), torch.arange(8)).view(1, 64, 2).float()
print("image emb:", emb2(ids_img).shape)           # (1, 1, 64, 32, 2, 2)

emb3 = EmbedND(axes_dim=[16, 24, 24])              # video: 3 axes (t, h, w), sums to 64
ids_vid = torch.cartesian_prod(torch.arange(4), torch.arange(4), torch.arange(4)).view(1, 64, 3).float()
print("video emb:", emb3(ids_vid).shape)           # (1, 1, 64, 32, 2, 2)
```

运行 / Run with:
```bash
pip install torch einops
python try.py
```

预期输出 / Expected output:
```
text emb:  torch.Size([1, 1, 32, 8, 2, 2])
image emb: torch.Size([1, 1, 64, 32, 2, 2])
video emb: torch.Size([1, 1, 64, 32, 2, 2])
```

中文:三种调用,同一个类,只是 `axes_dim` 变了。这就是「one module, N dimensions」的具体含义。

English: Three different calls, one class, just `axes_dim` changes. That's what "one module, N dimensions" looks like operationally.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Black Forest Labs / Flux** / **Black Forest Labs / Flux**: Open-Sora 的代码直接抄自 Flux,文件头的 `# Modified from Flux` 注释承认这点。Flux 用 `[16, 56, 56]` 处理 1024x1024 图像。 / Open-Sora's file literally has `# Modified from Flux` at the top. Flux uses `[16, 56, 56]` for 1024x1024 images.
- **Stable Diffusion 3 / MMDiT** / **Stable Diffusion 3 / MMDiT**: 同样的 EmbedND 模式,但 axes_dim 配置不同(没有 time 维度)。 / Same EmbedND pattern, different `axes_dim` (no time axis).
- **`Wan-Video/Wan2.1` 的 `WanRotaryPosEmbed`** / **`Wan-Video/Wan2.1`'s `WanRotaryPosEmbed`**: 同一个 patchify-positional 槽,但写法是「3 个硬编码 buffer」+「polar(ones, freqs)」复数表示。功能等价,但不能自然推广到 1D / 2D。 / Same slot, but written as three hardcoded buffers using `torch.polar(ones, freqs)` complex representation. Functionally equivalent, but doesn't generalize cleanly to 1D / 2D.
- **`facebookresearch/DiT`** / **`facebookresearch/DiT`**: 用的是 absolute sin-cos position embedding,不是 RoPE——同一个槽位的另一种解法(更简单,但对超出训练 resolution 的输入泛化更差)。 / Uses an absolute sin-cos position embedding instead of RoPE — a different solution for the same slot, simpler but generalizes worse to resolutions unseen at train time.

## 注意事项 / Caveats / when it breaks

- **`sum(axes_dim) == head_dim`** / **`sum(axes_dim) == head_dim`**: 不满足这个条件,emb 的 channel 数会和 q/k 对不上,broadcast 会报错。 / If this doesn't hold, the embedding's channel count won't match q/k and broadcasting will raise.
- **每个 `axes_dim[i]` 必须偶数** / **Every `axes_dim[i]` must be even**: 因为 RoPE 是「每对儿 channel 一个旋转」,奇数会被 `assert dim % 2 == 0` 拒掉。 / Because RoPE rotates pairs of channels — odd values trip the `assert dim % 2 == 0`.
- **`ids` 是 float 还是 long?** / **Are `ids` float or long?**: 这里是 float——因为 `einsum` 要做乘法。如果你想保持索引语义,先 `ids.float()`。 / Here, `float` — `einsum` needs to multiply. If you want index semantics, cast with `ids.float()` first.
- **`apply_rope` 拒绝奇数 head_dim** / **`apply_rope` rejects odd head_dim**: 把 head_dim 设成奇数(比如 65)会让 `reshape(..., -1, 1, 2)` 报 size 错。 / Setting `head_dim` to an odd value (say 65) breaks `reshape(..., -1, 1, 2)` with a size error.

## 延伸阅读 / Further reading

- [RoFormer / RoPE 原始论文 (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [Black Forest Labs Flux 代码 (axes_dim source)](https://github.com/black-forest-labs/flux)
- [MMDiT / Stable Diffusion 3 paper](https://arxiv.org/abs/2403.03206)
- [Open-Sora technical report](https://arxiv.org/abs/2503.09642)
