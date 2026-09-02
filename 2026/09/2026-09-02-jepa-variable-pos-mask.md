---
date: 2026-09-02
topic: diffusion
source: tracked
repo: facebookresearch/jepa
file: src/models/vision_transformer.py
permalink: https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/models/vision_transformer.py#L112-L227
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, vit, positional-embedding, masking]
---

# V-JEPA 可变位置编码：输入变了，位置格子也跟着变 / V-JEPA Variable Positional Embedding: Let the Grid Follow the Input

> **一句话 / In one line**: V-JEPA 先按输入大小重算位置编码，再把不需要的 patch token 直接遮掉。 / V-JEPA first re-scales positional encodings to the input shape, then drops masked patch tokens.

## 为什么重要 / Why this matters

中文：视觉模型常常希望“输入分辨率可变，token 形状不乱”。这里的关键不是只做 patch embed，而是把位置网格也按当前输入重插值，再在进入 Transformer block 前把不该看的 token 去掉。这样模型既能吃固定权重，又不会把位置感知绑死在训练分辨率上。

English: Vision models often need variable input resolution without changing the token contract. The trick here is not just patch embedding; it also re-interpolates the positional grid to the current input and removes masked tokens before the Transformer blocks. That keeps the weights fixed while avoiding a hard dependency on the training resolution.

## 代码 / The code

`facebookresearch/jepa` — [`src/models/vision_transformer.py`](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/models/vision_transformer.py#L112-L227)

```python
def _init_pos_embed(self, pos_embed):
    embed_dim = pos_embed.size(-1)
    grid_size = self.input_size // self.patch_size
    if self.is_video:
        grid_depth = self.num_frames // self.tubelet_size
        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            grid_size,
            grid_depth,
            cls_token=False,
            uniform_power=self.uniform_power
        )
    else:
        sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
    pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

def forward(self, x, masks=None):
    """
    :param x: input image/video
    :param masks: indices of patch tokens to mask (remove)
    """

    if masks is not None and not isinstance(masks, list):
        masks = [masks]

    # Tokenize input
    pos_embed = self.pos_embed
    if pos_embed is not None:
        pos_embed = self.interpolate_pos_encoding(x, pos_embed)
    x = self.patch_embed(x)
    if pos_embed is not None:
        x += pos_embed
    B, N, D = x.shape

    # Mask away unwanted tokens (if masks provided)
    if masks is not None:
        x = apply_masks(x, masks)
        masks = torch.cat(masks, dim=0)

    # Fwd prop
    outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x, mask=masks)
        if self.out_layers is not None and i in self.out_layers:
            outs.append(self.norm(x))

    if self.out_layers is not None:
        return outs

    if self.norm is not None:
        x = self.norm(x)

    return x

def interpolate_pos_encoding(self, x, pos_embed):

    _, N, dim = pos_embed.shape

    if self.is_video:
        # If pos_embed already corret size, just return
        _, _, T, H, W = x.shape
        if H == self.input_size and W == self.input_size and T == self.num_frames:
            return pos_embed

        # Convert depth, height, width of input to be measured in patches
        # instead of pixels/frames
        T = T // self.tubelet_size
        H = H // self.patch_size
        W = W // self.patch_size

        # Compute the initialized shape of the positional embedding measured
        # in patches
        N_t = self.num_frames // self.tubelet_size
        N_h = N_w = self.input_size // self.patch_size
        assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

        # Compute scale factor for spatio-temporal interpolation
        scale_factor = (T/N_t, H/N_h, W/N_w)

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
            scale_factor=scale_factor,
            mode='trilinear')
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed

    else:
        # If pos_embed already corret size, just return
        _, _, H, W = x.shape
        if H == self.input_size and W == self.input_size:
            return pos_embed

        # Compute scale factor for spatial interpolation
        npatch = (H // self.patch_size) * (W // self.patch_size)
        scale_factor = math.sqrt(npatch / N)

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=scale_factor,
            mode='bicubic')
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed
```

## 逐行讲解 / What's happening

1. **第 112-117 行 / Lines 112-117**:
   - 中文: `_init_pos_embed()` 先按 2D/3D 网格生成固定 sin/cos 位置编码，视频版本会多带一个 depth 维。
   - English: `_init_pos_embed()` builds a fixed 2D/3D sin/cos grid, with video mode carrying an extra depth axis.
2. **第 159-180 行 / Lines 159-180**:
   - 中文: `forward()` 先 patchify，再加位置编码，再用 `apply_masks()` 删掉不想让 block 看到的 token。
   - English: `forward()` patchifies, adds position embeddings, then uses `apply_masks()` to remove tokens the blocks should not see.
3. **第 197-227 行 / Lines 197-227**:
   - 中文: `interpolate_pos_encoding()` 把原始网格缩放到新输入尺寸，视频走 trilinear，图像走 bicubic。
   - English: `interpolate_pos_encoding()` rescales the original grid to the new input size, using trilinear for video and bicubic for images.

## 类比 / The analogy

中文：像把一张老地图放大到新城市边界，再把被施工围挡挡住的街区直接从导航里删掉。地图还能用，但不会把旧尺度硬套到新道路上。

English: It is like enlarging an old map to match a new city boundary, then deleting the blocked streets behind construction fences. The map still works, but you are not forcing an old scale onto new roads.

## 自己跑一遍 / Try it yourself

```python
import math

def resize_1d(n_old, n_new):
    scale = math.sqrt(n_new / n_old)
    return [round(i * scale, 2) for i in range(n_old)]

tokens = ["t0", "t1", "t2", "t3"]
mask = [0, 1, 0, 1]
kept = [t for t, m in zip(tokens, mask) if m == 0]
print(resize_1d(4, 9))
print(kept)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.0, 1.5, 3.0, 4.5]
['t0', 't2']
```

中文：这里最重要的是两件事分开做：先把位置坐标放到新网格，再决定哪些 token 保留。

English: The important part is splitting the job in two: first move positions onto the new grid, then decide which tokens survive.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DINOv3 RoPE coordinate augmentation** / **DINOv3 RoPE coordinate augmentation**: 中文: 位置编码也能在训练时做坐标抖动。 / English: Positional encodings can also absorb coordinate jitter at training time.
- **Wan2.1 patch/freq buffer** / **Wan2.1 patch/freq buffer**: 中文: 视频 token 一旦变长，坐标和频率表也得一起对齐。 / English: Once video tokens become variable length, coordinate and frequency tables must stay aligned too.

## 注意事项 / Caveats / when it breaks

- **patch/tubelet 必须整除输入尺寸 / Input size must be divisible by patch/tubelet size**: 中文: 否则 reshape 和插值的尺度就会错位。 / English: Otherwise the reshape and interpolation scales will drift apart.
- **mask 不是后处理装饰 / Masking is not a cosmetic afterthought**: 中文: 它直接改变 block 看到的 token 集合。 / English: It changes the token set the block actually sees.

## 延伸阅读 / Further reading

- V-JEPA source: https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/models/vision_transformer.py
