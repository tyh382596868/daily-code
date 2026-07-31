---
date: 2026-07-31
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/vae/utils.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/vae/utils.py#L59-L109
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae-encoder-decoder, conv3d]
build_role: vae-encoder-decoder memory-bounded Conv3d
---

# Open-Sora channel chunk Conv3d：卷积不变，只把通道分块 / Open-Sora Channel-Chunk Conv3d: Same Convolution, Split Channels

> **一句话 / In one line**: `channel_chunk_conv3d` 在输入通道和输出通道上拆块，累加局部卷积，最后拼回完整输出。 / `channel_chunk_conv3d` splits input and output channels, accumulates partial convolutions, then concatenates the full output.

## 为什么重要 / Why this matters

WAM 的视频 VAE 会遇到巨大 5D tensor。直接 `Conv3d` 可能爆显存，但改模型结构又会改变数学结果。Open-Sora 这段代码选择第三条路：保持卷积等价，只把计算拆成更小的通道块。

A WAM video VAE deals with large 5D tensors. A direct `Conv3d` can run out of memory, but changing the model architecture changes the math. This Open-Sora code takes a third route: keep the convolution equivalent and split the computation into smaller channel chunks.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/vae/utils.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/vae/utils.py#L59-L109)

```python
def get_conv3d_n_chunks(numel: int, n_channels: int, numel_limit: int):
    n_chunks = math.ceil(numel / numel_limit)
    n_chunks = ceil_to_divisible(n_chunks, n_channels)
    return n_chunks

def channel_chunk_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list,
    padding: list,
    dilation: list,
    groups: int,
    numel_limit: int,
):
    out_channels, in_channels = weight.shape[:2]
    kernel_size = weight.shape[2:]
    output_shape = get_conv3d_output_shape(input.shape, out_channels, kernel_size, stride, padding, dilation)
    n_in_chunks = get_conv3d_n_chunks(input.numel(), in_channels, numel_limit)
    n_out_chunks = get_conv3d_n_chunks(np.prod(output_shape), out_channels, numel_limit)
    if n_in_chunks == 1 and n_out_chunks == 1:
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)
    input_shards = input.chunk(n_in_chunks, dim=1)
    weight_chunks = weight.chunk(n_out_chunks)
    output_list = []
    if bias is not None:
        bias_chunks = bias.chunk(n_out_chunks)
    else:
        bias_chunks = [None] * n_out_chunks
    for weight_, bias_ in zip(weight_chunks, bias_chunks):
        weight_shards = weight_.chunk(n_in_chunks, dim=1)
        o = None
        for x, w in zip(input_shards, weight_shards):
            if o is None:
                o = F.conv3d(x, w, None, stride, padding, dilation, groups).float()
            else:
                o += F.conv3d(x, w, None, stride, padding, dilation, groups).float()
        o = o.to(input.dtype)
        if bias_ is not None:
            o += bias_[None, :, None, None, None]
        output_list.append(o)
    return torch.cat(output_list, dim=1)
```

## 逐行讲解 / What's happening

1. **第 59-62 行 / Lines 59-62 (`get_conv3d_n_chunks`)**:
   - 中文: 先按元素数估计块数，再把块数调到能整除通道数，避免 chunk 形状不齐。
   - English: It estimates chunk count from element count, then adjusts it to divide the channel count cleanly.
2. **第 75-83 行 / Lines 75-83 (shape planning)**:
   - 中文: 输入和输出各自算块数，因为输入激活和输出激活的显存压力不同。
   - English: Input and output chunk counts are computed separately because input and output activations stress memory differently.
3. **第 88-99 行 / Lines 88-99 (input/output channel split)**:
   - 中文: 外层遍历输出通道块，内层遍历输入通道块；每个局部结果加到同一个输出块。
   - English: The outer loop walks output-channel chunks, while the inner loop walks input-channel chunks; each partial result is accumulated into the same output block.
4. **第 103-109 行 / Lines 103-109 (dtype and concat)**:
   - 中文: 累加时用 float，最后转回原 dtype，加 bias，再沿通道拼接。
   - English: Accumulation uses float, then casts back to the original dtype, adds bias, and concatenates along channels.

## 类比 / The analogy

这像把一张大桌子分成几块搬：桌子还是同一张桌子，只是搬运时先拆成能进电梯的尺寸。

It is like moving a large table in sections: the table is still the same table, but it is split into elevator-sized pieces during transport.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这属于 `vae-encoder-decoder` 的生产化实现。上游是视频帧 tensor，下游是 latent 或重建帧；nano 版可以先直接 `Conv3d`，但长视频、高清分辨率或低显存 GPU 上需要这种等价分块层。

This belongs in the production version of the `vae-encoder-decoder` component. Upstream is the video-frame tensor; downstream is either latents or reconstructed frames. A nano version can start with direct `Conv3d`, but long clips, high resolution, or low-memory GPUs need this equivalent chunked layer.

## 自己跑一遍 / Try it yourself

```python
def chunked_linear(x, w, out_chunks=2, in_chunks=2):
    xs = [x[i::in_chunks] for i in range(in_chunks)]
    rows = [w[i::out_chunks] for i in range(out_chunks)]
    outs = []
    for row_block in rows:
        acc = None
        for j, x_part in enumerate(xs):
            w_part = [r[j::in_chunks] for r in row_block]
            part = [sum(a*b for a, b in zip(r, x_part)) for r in w_part]
            acc = part if acc is None else [a+b for a, b in zip(acc, part)]
        outs.extend(acc)
    return outs

x = [1, 2, 3, 4]
w = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
print(chunked_linear(x, w))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 3, 2, 4]
```

示例为了展示分块，输出通道按块顺序排列；真实实现用 `torch.cat` 拼回输出通道块。

The toy example exposes the chunk ordering; the real implementation concatenates output-channel blocks with `torch.cat`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Tensor parallel linear/conv** / **Tensor parallel linear/conv**: 大矩阵乘也常按输入或输出通道拆分，再归并结果。 / Large matrix multiplies are also split by input or output channels and then reduced.
- **Tiled VAE inference** / **Tiled VAE inference**: 另一种常见做法是按空间/时间 tile 拆分，而这里按 channel 拆。 / Another common approach tiles space or time; this one tiles channels.

## 注意事项 / Caveats / when it breaks

- **分组卷积要小心** / **Grouped convolution needs care**: `groups` 会改变输入输出通道对应关系，分块策略必须与之兼容。 / `groups` changes how input and output channels correspond, so chunking must remain compatible.
- **速度未必更快** / **Not necessarily faster**: 目标是降低峰值显存，不是减少总 FLOPs。 / The goal is lower peak memory, not fewer total FLOPs.

## 延伸阅读 / Further reading

- [Open-Sora VAE utils source permalink](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/vae/utils.py#L59-L109)
