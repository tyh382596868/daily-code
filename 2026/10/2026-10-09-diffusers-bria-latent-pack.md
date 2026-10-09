---
date: 2026-10-09
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/pipelines/bria/pipeline_bria.py
permalink: https://github.com/huggingface/diffusers/blob/d961a388fd02e4db38d17350c8dd9b8abe642e05/src/diffusers/pipelines/bria/pipeline_bria.py#L412-L446
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, latent-packing]
---
# Diffusers latent packing：把 2x2 小格折成 token / Diffusers Latent Packing: Fold Each 2x2 Cell into a Token

> **一句话 / In one line**: BRIA pipeline 把 `[C,H,W]` latent 重排成 patch tokens，同时保留可还原的反向路径和二维坐标 id。 / The BRIA pipeline rearranges image latents into patch tokens, with a matching unpack path and coordinate ids.

## 为什么重要 / Why this matters

很多 DiT/Flux 类模型不直接吃 `[B,C,H,W]` 图像 latent，而是吃 `[B,N,D]` token 序列。这段代码展示了最干净的一种桥接：`view` 把 2x2 空间块显式拆出来，`permute` 把空间块移到 token 维，最后 `reshape` 把四个像素位置的通道拼进 token channel。

Many DiT-style pipelines consume `[B, N, D]` tokens rather than `[B, C, H, W]` image latents. This helper is a compact bridge between the two. It exposes 2x2 spatial cells, moves them into the token axis, and folds the four positions into the channel dimension.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/pipelines/bria/pipeline_bria.py`](https://github.com/huggingface/diffusers/blob/d961a388fd02e4db38d17350c8dd9b8abe642e05/src/diffusers/pipelines/bria/pipeline_bria.py#L412-L446)

```python
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
```

## 逐行讲解 / What's happening

1. **第 412-415 行 / Lines 412-415 (`_pack_latents`)**:
   - 中文: 先把高宽各拆成 `H//2,2,W//2,2`，再把小格子维度挪到 channel 后面，形成 token。
   - English: Height and width are split into coarse coordinates plus 2x2 offsets, then the offsets are folded into the token channel.
2. **第 420-431 行 / Lines 420-431 (`_unpack_latents`)**:
   - 中文: 解包是 pack 的逆操作：token 序列先恢复成粗网格，再把 2x2 offset 展回空间维。
   - English: Unpacking reverses the process: token sequence to coarse grid, then 2x2 offsets back into spatial dimensions.
3. **第 434-446 行 / Lines 434-446 (`_prepare_latent_image_ids`)**:
   - 中文: 每个 latent token 获得 `(batch-ish, row, col)` 坐标，给模型一个稳定的位置标识。
   - English: Each latent token receives row and column ids, giving the model stable positional metadata.

## 类比 / The analogy

像把一张照片切成贴纸。每张 2x2 小贴纸被装进一个信封，信封上写着第几行第几列；要还原照片时，再按信封地址贴回去。

It is like cutting a photo into stickers. Each 2x2 sticker goes into an envelope labeled with its row and column; reconstruction just places each envelope back at its address.

## 自己跑一遍 / Try it yourself

```python
def pack(latents, batch, channels, height, width):
    x = latents
    x = [[[[x[c][r+i][col+j] for i in range(2) for j in range(2)]
           for c in range(channels)]
          for col in range(0, width, 2)]
         for r in range(0, height, 2)]
    return x

image = [[[r * 10 + c for c in range(4)] for r in range(4)]]
packed = pack(image, 1, 1, 4, 4)
print(packed[0][0])
print(packed[1][1])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[0, 1, 10, 11]]
[[22, 23, 32, 33]]
```

每个 token 里装的是一个 2x2 局部邻域，而不是单个像素。

Each token stores a 2x2 local neighborhood, not a single pixel.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Flux/DiT pipelines** / **Flux/DiT pipelines**: 图像 latent 经常被打平成 patch token 后送进 transformer。 / Image latents are often flattened into patch tokens before entering a transformer.
- **VAE latent grids** / **VAE latent grids**: 坐标 id 让模型知道 token 来自哪一块空间。 / Coordinate ids tell the model where each token came from.

## 注意事项 / Caveats / when it breaks

- **高宽要能整除 2** / **Height and width must be divisible by 2**: 这里直接使用 `height // 2` 和 `width // 2`。 / The implementation assumes clean 2x2 packing.
- **`view` 需要布局兼容** / **`view` needs compatible layout**: 上游 tensor 通常要保持连续或至少 stride 合法。 / The upstream tensor must have a layout compatible with `view`.

## 延伸阅读 / Further reading

- [Diffusers BRIA pipeline latent helpers](https://github.com/huggingface/diffusers/blob/d961a388fd02e4db38d17350c8dd9b8abe642e05/src/diffusers/pipelines/bria/pipeline_bria.py#L412-L446)
- [Diffusers pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview)
