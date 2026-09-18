---
date: 2026-09-18
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/hunyuan_vae/vae.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/vae.py#L40-L150
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae-encoder-decoder, causal-video, temporal-compression]
build_role: vae-encoder-decoder advanced variant
---

# Open-Sora causal 3D VAE：时间压缩不能偷看未来 / Open-Sora Causal 3D VAE: Temporal Compression Must Not See the Future

> **一句话 / In one line**: Open-Sora 的 `EncoderCausal3D` 同时安排空间/时间下采样，并在中间 attention block 使用 causal mask，让每个 latent 只依赖当前和过去帧。 / Open-Sora's `EncoderCausal3D` schedules spatial and temporal downsampling while applying a causal mask in the middle attention block so each latent depends only on present and past frames.

## 为什么重要 / Why this matters

WAM 的视频 latent 不是普通图片 latent 加一维时间。只要模型在 rollout 中逐帧或分块生成，encoder 如果看到了未来帧，训练时得到的表示就和推理时不一致，动作条件也会被“偷看答案”污染。

A WAM video latent is not just an image latent with one more axis. If a rollout generates frames incrementally, an encoder that sees future frames creates a train/inference mismatch and lets action conditioning cheat by reading the answer.

这段代码把两个容易混在一起的问题拆开：`downsample_stride` 决定空间和时间压缩比例；`prepare_attention_mask` 负责 attention 的因果性。最终 `forward` 只需要把 5D video tensor 依次过 `conv_in`、down blocks、mid block 和 output projection。

The code separates two concerns that are easy to conflate: `downsample_stride` controls spatial and temporal compression, while `prepare_attention_mask` controls causality. The `forward` path then stays simple: input convolution, down blocks, middle block, and output projection.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/hunyuan_vae/vae.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/vae.py#L40-L150)

```python
class EncoderCausal3D(nn.Module):
    r"""
    The `EncoderCausal3D` layer of a variational autoencoder that encodes its input into a latent representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, _ in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block
                )
            elif time_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_spatial_downsample_layers)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}.")

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)
            down_block = DownEncoderBlockCausal3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=dropout,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )

            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

    def prepare_attention_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        attention_mask = prepare_causal_attention_mask(
            T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B
        )
        return attention_mask

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions"
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        if self.mid_block.add_attention:
            attention_mask = self.prepare_attention_mask(sample)
        else:
            attention_mask = None
        sample = auto_grad_checkpoint(self.mid_block, sample, attention_mask)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
```

## 逐行讲解 / What's happening

1. **第 45-57 行 / Lines 45-57 (compression contract)**:
   - 中文: `time_compression_ratio` 和 `spatial_compression_ratio` 是 latent geometry 的全局合同，后面的 DiT patchify 必须使用相同的缩放结果。
   - English: The time and spatial compression ratios define the latent geometry contract; later DiT patchification must agree with the resulting scale.
2. **第 62-88 行 / Lines 62-88 (separate strides)**:
   - 中文: `downsample_stride_T` 与 `downsample_stride_HW` 分开计算，再合成 `(T, H, W)` stride，避免把时间压缩误当成普通 2D pooling。
   - English: Temporal and spatial strides are computed independently and then combined into a `(T, H, W)` stride, rather than treating time as ordinary 2D pooling.
3. **第 75-84 行 / Lines 75-84 (supported schedules)**:
   - 中文: time ratio 为 4 和 8 时，时间下采样层的位置不同；未知 ratio 直接报错，避免生成 silent shape mismatch。
   - English: Ratios 4 and 8 place temporal downsampling at different blocks; unknown ratios fail early instead of creating silent shape mismatches.
4. **第 121-126 行 / Lines 121-126 (causal mask)**:
   - 中文: mask 的 token 数是 `T * H * W`，意味着同一时间帧的空间 token 可以互相 attention，但未来时间位置被屏蔽。
   - English: The mask covers `T * H * W` tokens, allowing spatial interaction within a frame while blocking future time positions.
5. **第 128-150 行 / Lines 128-150 (forward path)**:
   - 中文: 主干路径很干净，真正的 causal 语义由 `CausalConv3d`、`prepare_causal_attention_mask` 和 checkpointed mid block 共同提供。
   - English: The forward path stays clean; causal behavior comes from `CausalConv3d`, the causal attention mask, and the checkpointed middle block together.

## 类比 / The analogy

它像一个剪辑师处理直播录像：可以看已经播出的片段，不能提前翻到未来时间轴。空间上可以同时看当前画面的左上和右下，但时间上必须遵守“只向后看”的规则。

It is like an editor cutting a live broadcast: previously aired frames are available, but the editor cannot scrub into the future. Spatial regions in the current frame can interact, while temporal access only moves backward.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `vae-encoder-decoder` 组件的 encoder 半边，位于原始视频帧和 DiT token 化之间。输入是 `[B, C, T, H, W]`，输出是压缩后的 latent 参数，`double_z=True` 时通常还包含 mean/log-variance 所需的双通道。上游是视频读取和归一化，下游是 `patchify-positional`、`dit-block` 以及 sampler。省掉它，DiT 只能直接处理高分辨率像素，序列长度和显存都会失控；如果去掉 causal 约束，训练和逐块生成会发生未来信息泄露。生产版还要补 tiled/chunked encode、latent scaling、跨 chunk cache、decoder 对称性和不同帧率的时间对齐。

English: This is the encoder half of the `vae-encoder-decoder` component between raw video frames and DiT tokenization. It accepts `[B, C, T, H, W]` and emits compressed latent parameters; with `double_z=True`, the output typically carries the two halves needed for mean/log-variance. Upstream is video loading and normalization; downstream are `patchify-positional`, `dit-block`, and the sampler. Without it, the DiT would process full-resolution pixels and sequence length would explode. Without causality, training and chunked generation would leak future information. Production code adds tiled/chunked encoding, latent scaling, cross-chunk cache state, decoder symmetry, and frame-rate alignment.

## 自己跑一遍 / Try it yourself

```python
def compressed_shape(shape, time_ratio, spatial_ratio):
    b, c, t, h, w = shape
    return b, c, max(1, t // time_ratio), max(1, h // spatial_ratio), max(1, w // spatial_ratio)

def causal_visible_tokens(t, h, w):
    tokens_per_frame = h * w
    return [[j for j in range(t * tokens_per_frame) if j // tokens_per_frame <= i]
            for i in range(t * tokens_per_frame)]

print(compressed_shape((1, 3, 16, 64, 64), 4, 8))
print(causal_visible_tokens(3, 1, 1))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
(1, 3, 4, 8, 8)
[[0], [0, 1], [0, 1, 2]]
```

中文: 第一个结果展示 latent geometry，第二个结果展示第 3 帧不能反过来影响第 1 帧。  
English: The first result shows latent geometry; the second shows that frame 3 cannot influence frame 1.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 causal VAE** / **Wan2.1 causal VAE**: 中文: 通过 causal Conv3d 和时间 cache 把长视频拆成可流式处理的片段。 / English: It uses causal Conv3d and temporal cache state to process long videos in streaming chunks.
- **DreamZero VideoVAE** / **DreamZero VideoVAE**: 中文: 以固定帧块携带 cache，强调时间压缩与 chunk 边界。 / English: It carries cache across fixed-size frame chunks, emphasizing temporal compression and chunk boundaries.
- **Open-Sora chunked Conv3D** / **Open-Sora chunked Conv3D**: 中文: 把大卷积拆成可控的块，解决显存峰值而不改变卷积语义。 / English: It splits large convolutions into manageable chunks to reduce peak memory without changing convolution semantics.

## 注意事项 / Caveats / when it breaks

- **stride 要和 decoder 对称** / **Strides must match the decoder**: 中文: encoder 压缩比例和 decoder 上采样比例不一致时，视频尺寸会在重建处漂移。 / English: Mismatched encoder and decoder ratios make reconstructed video dimensions drift.
- **causal mask 只解决 attention** / **The mask only covers attention**: 中文: 还要确认 causal convolution、chunk cache 和数据切分没有偷偷读取未来帧。 / English: Causal attention alone is insufficient; causal convolution, chunk cache, and data slicing must also avoid future frames.
- **5D shape 不是可选项** / **The 5D shape is a contract**: 中文: `[B,C,T,H,W]` 的轴顺序一旦错，压缩和 mask 都会指向错误维度。 / English: If `[B,C,T,H,W]` axes are mixed up, both compression and masking target the wrong dimensions.

## 延伸阅读 / Further reading

- [Open-Sora Hunyuan causal VAE](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/vae.py)
- [Wan2.1 VAE modules](https://github.com/Wan-Video/Wan2.1)
- [Diffusers AutoencoderKLWan](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_wan.py)
