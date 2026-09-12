---
date: 2026-07-24
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L321-L350
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, wam, wan2.1, output-head, modulation]
build_role: output-head advanced variant
---

# Wan2.1 Head：最后一层也吃条件向量 / Wan2.1 Head: The Last Layer Is Conditioned Too

> **一句话 / In one line**: `Head` 不是简单 Linear；它先用时间条件调制 normalized token，再投影成每个 patch 的输出通道。 / `Head` is not just a Linear layer; it modulates normalized tokens with the timestep condition before projecting each token into patch channels.

## 为什么重要 / Why this matters

WAM 的 DiT backbone 输出的还是 token。真正要交给 VAE decoder 的，是按 patch 排列的 latent 预测。Wan2.1 在最后一步继续使用 adaLN 风格调制，让噪声时间步不仅影响中间 block，也影响最终输出头。

A WAM DiT backbone still outputs tokens. What the VAE decoder needs is a patch-arranged latent prediction. Wan2.1 keeps adaLN-style modulation in the final step, so the diffusion timestep affects not only middle blocks but also the output head.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L321-L350)

```python
class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x
```

## 逐行讲解 / What's happening

1. **第 330 行 / Line 330 (`math.prod(patch_size)`)**: 中文: 一个 token 要预测整个 patch 内的所有 latent 数值，所以输出维度乘上 patch 体积。 English: one token predicts every latent value inside its patch, so output dimension is multiplied by patch volume.
2. **第 331-332 行 / Lines 331-332 (`norm` + `Linear`)**: 中文: 先归一化 token，再线性投影到 patch space。 English: normalize token features first, then project them into patch space.
3. **第 335 行 / Line 335 (`modulation`)**: 中文: 可训练偏置提供每个 head 自己的默认 shift/scale。 English: the trainable parameter gives the head its own default shift/scale.
4. **第 344-346 行 / Lines 344-346 (`chunk(2)`)**: 中文: 时间条件拆成两份：一份做 shift，一份做 scale。 English: the timestep condition is split into two pieces: one shift and one scale.
5. **第 346 行 / Line 346 (`norm(x) * (1 + e[1]) + e[0]`)**: 中文: 这是最后一层的 adaLN 注入。 English: this is adaLN-style injection at the final layer.

## 类比 / The analogy

像打印照片前的最后一次色彩校正。前面的编辑已经完成构图，但输出到纸张前，打印机会根据纸张和墨水状态再调一次亮度和对比度。

It is like the final color correction before printing a photo. The composition is already done, but the printer still adjusts brightness and contrast for the paper and ink state.

## 在 nanoWAM 中的位置 / Where this fits in nanoWAM

这是 `output-head`。上游是 DiT backbone 的视频 token 和 timestep embedding，下游是 unpatchify 与 VAE decode。若省掉条件化，模型最后一层只能用同一个投影处理所有噪声阶段；生产级实现通常会把 patch size、latent channel 和 mixed precision 都写进这个契约。

This is the `output-head`. Upstream provides video tokens and the timestep embedding from the DiT backbone; downstream unpatchifies and decodes through the VAE. Without conditioning, the final projection uses the same mapping for every noise level. A production implementation usually bakes patch size, latent channels, and mixed precision into this contract.

## 自己跑一遍 / Try it yourself

```python
token = [2.0, 4.0]
mean = sum(token) / len(token)
var = sum((x - mean) ** 2 for x in token) / len(token)
normed = [(x - mean) / (var ** 0.5) for x in token]
shift = [0.5, -0.5]
scale = [0.1, 0.2]
modulated = [n * (1 + s) + b for n, s, b in zip(normed, scale, shift)]
print([round(x, 3) for x in normed])
print([round(x, 3) for x in modulated])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[-1.0, 1.0]
[-0.6, 0.7]
```

中文: 同一个 normalized token 会被条件向量重新平移和缩放，然后才进入输出投影。 English: the same normalized token is shifted and scaled by the condition before output projection.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT FinalLayer** / **DiT FinalLayer**: 经典 DiT 也在输出头里用条件向量做 shift/scale。 / classic DiT also uses condition-driven shift/scale in its output head.
- **Wan2.1 block modulation** / **Wan2.1 block modulation**: 中间 block 用 6 路调制，最终 head 缩成 2 路。 / middle blocks use six modulation paths; the final head reduces that to two.

## 注意事项 / Caveats / when it breaks

- **输出还不是视频 / Output is not video yet**: 这一步输出 patch latent，需要 unpatchify 和 VAE decode。 / this outputs patch latents, not pixels; unpatchify and VAE decode still follow.
- **`e` 必须是 float32 / `e` must be float32**: 源码显式 assert，并在 autocast 中保持数值稳定。 / the source asserts this and uses autocast for numerical stability.
- **patch 契约要一致 / Patch contract must match**: `math.prod(patch_size) * out_dim` 必须和后续还原形状对上。 / `math.prod(patch_size) * out_dim` must match the later restore shape.

## 延伸阅读 / Further reading

- [Wan2.1 Head](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L321-L350)

