---
date: 2026-08-05
topic: wam
source: wam
repo: huggingface/lerobot
file: src/lerobot/policies/fastwam/wan/adapters.py
permalink: https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/fastwam/wan/adapters.py#L25-L108
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, fastwam, wan-vae, latent-standardization]
build_role: vae-encoder-decoder advanced variant, adapter contract around a fixed Wan video VAE
---

# FastWAM Wan VAE adapter：raw latent 进出都要标准化 / FastWAM Wan VAE Adapter: Standardize Raw Latents on Both Sides

> **一句话 / In one line**: `WanVideoVAE38` 把 diffusers 的 Wan VAE 包成 FastWAM 需要的契约，并显式应用 `latents_mean/std`。 / `WanVideoVAE38` wraps the diffusers Wan VAE into the contract FastWAM expects and explicitly applies `latents_mean/std`.

## 为什么重要 / Why this matters

WAM 训练通常不直接在像素上建模，而是在视频 VAE 的 latent 空间里预测未来。问题是不同 VAE 对 latent 的缩放约定不一样：有的返回已经标准化的 latent，有的返回 raw latent。FastWAM 这里把约定写进 adapter：encode 后减均值除标准差，decode 前乘回标准差加均值，让后面的 DiT 看到稳定分布。

WAM training usually predicts in video-VAE latent space instead of pixel space. The catch is that VAEs differ in their latent scaling conventions: some return standardized latents, others return raw latents. This FastWAM adapter makes the convention explicit: subtract mean and divide by std after encode, then undo that transform before decode, so the downstream DiT sees a stable distribution.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/fastwam/wan/adapters.py`](https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/fastwam/wan/adapters.py#L25-L108)

```python
class WanVideoVAE38(torch.nn.Module):
    """FastWAM VAE contract over `diffusers.AutoencoderKLWan` (Wan2.2-TI2V-5B).

    16x spatial / 4x temporal compression, 48 latent channels. diffusers'
    `AutoencoderKLWan` returns *raw* latents (it does not apply `latents_mean`/
    `latents_std`), so `encode`/`decode` here apply the same standardization the
    Wan reference uses — `(latents - mean) / std` — done in fp32 for stability.
    `encode` uses the deterministic posterior mode, matching the original VAE
    which returned the latent mean `mu`.
    """

    upsampling_factor = 16
    temporal_downsample_factor = 4
    z_dim = 48

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        *,
        pretrained: AutoencoderKLWan,
    ) -> None:
        super().__init__()
        # The Wan2.2 VAE is a fixed pretrained model — it is never trained from scratch,
        # so a real `AutoencoderKLWan` (with weights) must always be supplied (loaded from
        # the diffusers repo by `load_pretrained_wan_vae`). No random/offline build path.
        self.vae = pretrained.to(device=device, dtype=dtype)

        # Read the standardization stats from the VAE's own config (diffusers populates
        # these from vae/config.json) — single source of truth, no local copy. diffusers'
        # encode/decode return *raw* latents, so we apply (latent - mean) / std ourselves.
        # Non-persistent: kept out of state_dict.
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean).view(1, self.z_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.vae.config.latents_std).view(1, self.z_dim, 1, 1, 1),
            persistent=False,
        )

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.vae.parameters())
        return param.device, param.dtype

    def encode(
        self,
        videos: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del device, tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE encoding is not supported by the FastWAM adapter.")
        if isinstance(videos, (list, tuple)):
            videos = torch.stack(list(videos))
        dev, dtype = self._device_dtype()
        mu = self.vae.encode(videos.to(device=dev, dtype=dtype)).latent_dist.mode().float()
        mean = self.latents_mean.float().to(mu.device)
        std = self.latents_std.float().to(mu.device)
        return (mu - mean) / std

    def decode(
        self,
        hidden_states: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del device, tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE decoding is not supported by the FastWAM adapter.")
        if isinstance(hidden_states, (list, tuple)):
            hidden_states = torch.stack(list(hidden_states))
        dev, dtype = self._device_dtype()
        z = hidden_states.float()
        z = z * self.latents_std.float().to(z.device) + self.latents_mean.float().to(z.device)
        out = self.vae.decode(z.to(device=dev, dtype=dtype)).sample
        return out.float().clamp_(-1.0, 1.0)
```

## 逐行讲解 / What's happening

1. **第 25-38 行 / Lines 25-38 (contract constants)**:
   - 中文: 类属性说明这个 VAE 的空间压缩、时间压缩和 latent channel 数，后续模块可以按这些常量推 shape。
   - English: Class attributes state spatial compression, temporal compression, and latent channel count, so downstream modules can derive shapes.
2. **第 48-66 行 / Lines 48-66 (fixed pretrained VAE)**:
   - 中文: adapter 不创建随机 VAE，而是要求传入带权重的 `AutoencoderKLWan`；标准化统计从 VAE config 读，并注册为非持久 buffer。
   - English: The adapter does not create a random VAE. It requires a weighted `AutoencoderKLWan`; standardization stats come from the VAE config and are registered as non-persistent buffers.
3. **第 72-89 行 / Lines 72-89 (encode raw to normalized)**:
   - 中文: list 输入先 stack，视频被送到 VAE 所在 device/dtype；取 deterministic posterior mode 后转 fp32，再做 `(mu - mean) / std`。
   - English: List inputs are stacked, videos move to the VAE device/dtype, the deterministic posterior mode is cast to fp32, and `(mu - mean) / std` is applied.
4. **第 91-108 行 / Lines 91-108 (decode normalized to pixels)**:
   - 中文: decode 前先反标准化，再交给 VAE decode，最后 clamp 到像素范围 `[-1, 1]`。
   - English: Decode first unnormalizes the latent, then calls the VAE decoder, and finally clamps pixels to `[-1, 1]`.

## 类比 / The analogy

这像给不同国家的电器接转换插头。墙上的电压和插头形状不一定符合设备预期，adapter 先把电压和接口变成统一规格，用完再转换回去。

It is like using a travel adapter for electronics. Wall voltage and plug shape may not match the device, so the adapter converts them into the expected convention and converts back afterward.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `vae-encoder-decoder` 的 advanced variant。上游是像素视频；中间层是固定视频 VAE 和 latent 标准化 adapter；下游是 DiT/WAM 在标准化 latent 空间上的训练和采样。如果省掉 adapter，模型可能在错误尺度的 latent 上学习，loss 看似下降但 decode 出来发灰、过曝或动作视频不稳定。生产级实现还要处理 tiling、长视频 chunk、mixed precision、VAE 版本兼容和 distributed cache。

This is an advanced variant of `vae-encoder-decoder`. Upstream is pixel video; the middle is a fixed video VAE plus latent-standardization adapter; downstream is DiT/WAM training and sampling in normalized latent space. Without the adapter, the model may learn at the wrong latent scale: loss can fall while decoded videos look washed out, overexposed, or unstable. A production version also needs tiling, long-video chunks, mixed precision, VAE-version compatibility, and distributed caching.

## 自己跑一遍 / Try it yourself

```python
mean, std = 10.0, 2.0

def encode(raw_latent):
    return [(x - mean) / std for x in raw_latent]

def decode(normalized):
    return [x * std + mean for x in normalized]

z = encode([8.0, 10.0, 14.0])
print(z)
print(decode(z))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[-1.0, 0.0, 2.0]
[8.0, 10.0, 14.0]
```

中文: encode/decode 必须互为反函数，否则 latent 空间和像素空间会慢慢错位。

English: Encode and decode must be inverse transforms, or latent space and pixel space drift apart.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion VAE scale factor** / **Stable Diffusion VAE scale factor**: SD 系列也需要在 latent 和 VAE 之间应用固定缩放。 / Stable Diffusion also applies a fixed scale between latents and the VAE.
- **Wan2.1 VAE temporal chunk cache** / **Wan2.1 VAE temporal chunk cache**: 同一 VAE slot 的另一个生产问题是长视频分块时保持时间连续。 / Another production issue in the same VAE slot is preserving temporal continuity across chunks.

## 注意事项 / Caveats / when it breaks

- **统计必须来自同一个 VAE / Stats must come from the same VAE**: mean/std 换错版本，decode 会系统性偏色或偏尺度。 / Mean/std from the wrong VAE version causes systematic scale or color shifts.
- **fp32 不是多余的 / fp32 is not cosmetic**: 标准化统计常在小差值上工作，低精度更容易放大误差。 / Standardization works on small differences, so low precision can amplify errors.

## 延伸阅读 / Further reading

- FastWAM Wan VAE adapter: https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/fastwam/wan/adapters.py#L25-L108
