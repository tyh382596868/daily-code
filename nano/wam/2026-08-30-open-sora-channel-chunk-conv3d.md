---
date: 2026-08-30
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/vae/utils.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/vae/utils.py#L65-L109
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae, conv3d, chunking]
build_role: vae-encoder-decoder advanced variant
---

# Open-Sora chunked Conv3D：大卷积分块算，语义不变 / Open-Sora Chunked Conv3D: Split a Large Convolution Without Changing Semantics

> **一句话 / In one line**: `channel_chunk_conv3d` 把输入通道和输出通道分块，逐块卷积再拼回输出，用更小的中间张量完成同一个 3D convolution。 / `channel_chunk_conv3d` splits input and output channels, convolves chunks, and concatenates the outputs, computing the same 3D convolution with smaller intermediate tensors.

## 为什么重要 / Why this matters

WAM 的 VAE 经常要处理长视频 latent。普通 `Conv3d` 在大 batch、大通道或长时间轴上可能撞到内存或 kernel 限制。Open-Sora 这里保留数学语义，只改变执行方式：把一次大卷积分解成多次小卷积和累加。

A WAM VAE often processes long video latents. A regular `Conv3d` can hit memory or kernel limits with large batches, many channels, or long temporal axes. Open-Sora keeps the math intact and changes only the execution: one large convolution becomes many smaller convolutions plus accumulation.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/vae/utils.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/vae/utils.py#L65-L109)

```python
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
    n_out_chunks = get_conv3d_n_chunks(
        np.prod(output_shape),
        out_channels,
        numel_limit,
    )
    if n_in_chunks == 1 and n_out_chunks == 1:
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)
    # output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    # outputs = output.chunk(n_out_chunks, dim=1)
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
        # inplace operation cannot be used during training
        # output_.copy_(o)
        output_list.append(o)
    return torch.cat(output_list, dim=1)
```

## 逐行讲解 / What's happening

1. **第 75-83 行 / Lines 75-83**:
   - 中文: 先根据权重和输入形状推导输出形状，再分别估算输入通道和输出通道需要切成几块。
   - English: The function derives the output shape from weights and input, then estimates how many chunks are needed along input and output channels.
2. **第 84-85 行 / Lines 84-85**:
   - 中文: 如果没有超过限制，直接走普通 `F.conv3d`，避免给小样本增加开销。
   - English: If no limit is exceeded, it calls normal `F.conv3d`, avoiding overhead on small tensors.
3. **第 88-96 行 / Lines 88-96**:
   - 中文: 输入按 channel 切，输出权重也按 output channel 切；每个输出块再拿到对应的输入权重 shards。
   - English: Inputs are split by channel, and weights are split by output channel; each output block then gets matching input-channel shards.
4. **第 97-109 行 / Lines 97-109**:
   - 中文: 同一个输出块会累加所有输入 shard 的卷积结果，最后加 bias，并把输出块拼回 channel 维。
   - English: Each output block accumulates convolutions over all input shards, adds bias, and finally concatenates output blocks along the channel dimension.

## 类比 / The analogy

这像搬一张很大的会议桌进房间：不改变桌子的最终形状，只是先拆成桌腿和桌面，分批搬进去，再按原样装回去。

It is like moving a huge conference table into a room: the final table shape does not change, but you detach the legs and tabletop, carry pieces in batches, and reassemble them exactly.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，这属于 `vae-encoder-decoder` 的生产级高级变体。上游是视频帧或 video latent，下游是 DiT/flow model 使用的压缩 latent。nano 版本可以先用普通 `Conv3d`；当你开始支持长视频、高分辨率或多卡训练时，这类 chunked execution 会变成必要的工程层。

In a nanoWAM, this is a production-grade advanced variant of `vae-encoder-decoder`. Upstream is video frames or video latents; downstream is the compressed latent consumed by the DiT or flow model. A nano version can start with plain `Conv3d`; once you support long videos, high resolution, or multi-GPU training, chunked execution becomes an important engineering layer.

## 自己跑一遍 / Try it yourself

```python
def chunked_linear(x, weight, chunks):
    outs = []
    for row_block in range(0, len(weight), chunks):
        rows = weight[row_block : row_block + chunks]
        block = [sum(a * b for a, b in zip(x, row)) for row in rows]
        outs.extend(block)
    return outs

x = [1, 2, 3]
weight = [[1, 0, 0], [0, 1, 1], [2, 0, 1]]
print(chunked_linear(x, weight, chunks=2))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[1, 5, 5]
```

这个 toy 用矩阵乘法模拟 output-channel chunking：分块改变执行顺序，不改变最终结果。

This toy uses matrix multiplication to mimic output-channel chunking: chunking changes execution order, not the final result.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 VAE chunking** / **Wan2.1 VAE chunking**: 长视频 encode/decode 会按时间块处理，并维护 cache。 / Long-video encode/decode processes temporal chunks while maintaining cache.
- **vLLM KV cache paging** / **vLLM KV cache paging**: 大块内存被切成可管理的 block。 / Large memory is split into manageable blocks.
- **tensor parallel linear layers** / **tensor parallel linear layers**: 输入或输出 channel 被切给不同设备，再聚合结果。 / Input or output channels are split across devices and then aggregated.

## 注意事项 / Caveats / when it breaks

- **groups 让切分更敏感** / **Groups make splitting sensitive**: grouped convolution 下通道切分必须和组结构兼容。 / For grouped convolution, channel chunks must remain compatible with group structure.
- **训练中避免 inplace** / **Avoid inplace during training**: 源码也注释了 inplace copy 在训练时不安全。 / The source comments note that inplace copy is unsafe during training.
- **分块太多会慢** / **Too many chunks slow things down**: kernel launch 和 Python 循环会抵消省内存收益。 / Kernel launches and Python loops can eat the memory-saving benefit.

## 延伸阅读 / Further reading

- Open-Sora VAE utils: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/vae/utils.py
- PyTorch `conv3d`: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html
