---
date: 2026-07-13
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L520-L595
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, patchify-positional]
build_role: variable-length video patchify and unpatchify path for a from-scratch nanoWAM
---

# Wan2.1 forward：逐视频 patchify，再补齐到同一长度 / Wan2.1 forward: Patchify Each Video, Then Pad to One Length

> **一句话 / In one line**: Wan2.1 先按每个视频自己的时空网格做 patch embedding，记录真实 `seq_lens` 和 `grid_sizes`，再 padding 成 batch。 / Wan2.1 patch-embeds each video on its own spatiotemporal grid, records real `seq_lens` and `grid_sizes`, then pads them into a batch.

## 为什么重要 / Why this matters

视频生成经常遇到不同帧数、不同分辨率、I2V/T2V 条件不同的输入。简单做法是强行裁成同一形状；生产做法是记录每个样本的真实网格，把 transformer 计算和最后 unpatchify 都建立在这些元数据上。

Video generation often sees different frame counts, resolutions, and conditioning modes. The simple approach forces every sample into one shape; the production approach records each sample's real grid and uses that metadata for transformer computation and unpatchify.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L520-L595)

```python
if y is not None:
    x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

# embeddings
x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
grid_sizes = torch.stack(
    [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
x = [u.flatten(2).transpose(1, 2) for u in x]
seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
assert seq_lens.max() <= seq_len
x = torch.cat([
    torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
              dim=1) for u in x
])

with amp.autocast(dtype=torch.float32):
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).float())
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

context = self.text_embedding(
    torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ]))

kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
              freqs=self.freqs, context=context, context_lens=None)
for block in self.blocks:
    x = block(x, **kwargs)

x = self.head(x, e)
x = self.unpatchify(x, grid_sizes)
return [u.float() for u in x]
```

## 逐行讲解 / What's happening

1. **I2V 条件拼到通道维 / I2V condition joins channels**:
   - 中文: 如果有条件视频 `y`，先和噪声 latent `x` 在通道维拼接。
   - English: If conditional video `y` exists, it is concatenated with noisy latent `x` along channels.
2. **逐样本 patchify / Patchify per sample**:
   - 中文: 每个视频单独过 `patch_embedding`，所以不同网格大小可以先独立存在。
   - English: Each video passes through `patch_embedding` separately, so different grid sizes can exist before batching.
3. **保存 `grid_sizes` 和 `seq_lens` / Save `grid_sizes` and `seq_lens`**:
   - 中文: padding 后的 batch 会丢掉真实长度信息，这两个 tensor 是后续 attention 和 unpatchify 的账本。
   - English: Padding hides true lengths; these tensors are the ledger for attention and reconstruction.
4. **最后按原网格还原 / Reconstruct by original grids**:
   - 中文: `unpatchify(x, grid_sizes)` 用真实三维网格把 token 还原成视频 latent。
   - English: `unpatchify(x, grid_sizes)` restores tokens into video latents using the true 3D grids.

## 类比 / The analogy

像快递站把不同大小的包裹装进统一周转箱。箱子尺寸统一方便运输，但每个包裹原来的长宽高必须贴在标签上，否则到站就不知道怎么还原。

It is like putting different-sized packages into standardized shipping bins. The bin shape helps transport, but each package still needs its original dimensions on the label for unpacking.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `patchify-positional` 的高级版本，连接 VAE latent、DiT backbone、3D RoPE 和输出 head。nanoWAM 最小实现可以先固定视频大小；一旦支持变长视频或多分辨率 batch，就需要 `seq_lens`、`grid_sizes` 和 padding mask 这套元数据通道。

This is an advanced `patchify-positional` path connecting VAE latents, the DiT backbone, 3D RoPE, and the output head. A minimal nanoWAM can start with fixed video size; variable-length or multi-resolution batches need `seq_lens`, `grid_sizes`, and padding metadata.

## 自己跑一遍 / Try it yourself

```python
videos = [["a", "b", "c"], ["x", "y"]]
seq_len = max(len(v) for v in videos)
seq_lens = [len(v) for v in videos]
padded = [v + ["<pad>"] * (seq_len - len(v)) for v in videos]
restored = [row[:n] for row, n in zip(padded, seq_lens)]
print(padded)
print(restored)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[['a', 'b', 'c'], ['x', 'y', '<pad>']]
[['a', 'b', 'c'], ['x', 'y']]
```

padding 只是 batch 运输格式，真实长度才决定哪些 token 属于原视频。

Padding is only the batch transport format; true lengths decide which tokens belong to the original video.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NLP attention masks** / **NLP attention masks**: 文本 batch 也会 padding，同时保留真实长度或 mask。 / Text batches also pad sequences while retaining lengths or masks.
- **Variable-length FlashAttention** / **Variable-length FlashAttention**: 常用 cumulative lengths 告诉 kernel 每个样本真实边界。 / It often uses cumulative lengths to tell kernels the true sample boundaries.

## 注意事项 / Caveats / when it breaks

- **padding token 不能参与有效 attention** / **padding tokens must not count as real tokens**: 否则模型会看见不存在的视频块。 / Otherwise the model attends to nonexistent video patches.
- **unpatchify 必须用同一套 patch size** / **unpatchify must use the same patch size**: patchify 和还原参数不一致会直接错位。 / Mismatched patchify and reconstruction parameters misalign the video.

## 延伸阅读 / Further reading

- Wan2.1 model source — https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py

