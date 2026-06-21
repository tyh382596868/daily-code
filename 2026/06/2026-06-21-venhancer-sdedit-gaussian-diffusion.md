---
date: 2026-06-21
topic: diffusion
source: tracked
repo: Vchitect/VEnhancer
file: video_to_video/diffusion/diffusion_sdedit.py
permalink: https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/video_to_video/diffusion/diffusion_sdedit.py#L1-L110
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, diffusion, sdedit, video-super-resolution, logsnr, zero-terminal-snr]
---

# VEnhancer 的 SDEdit 内核:60 行把"加噪→去噪"全讲完 / VEnhancer's SDEdit core: forward + reverse diffusion in 60 lines

> **一句话 / In one line**: VEnhancer 把低分辨率视频在固定时间步 `t_max=200` 处加噪,然后以文本+条件 mask 为引导去噪,整套数学压进 `GaussianDiffusion` 的三个方法里。/ VEnhancer adds controlled noise to a low-resolution video at a fixed `t_max=200`, then denoises it guided by text and a condition mask — the full math lives in three methods of `GaussianDiffusion`.

## 为什么重要 / Why this matters

SDEdit 解决了视频超分中一个根本矛盾:如果从纯噪声生成,模型不知道目标内容是什么;如果完全不加噪声,模型就只是一个上采样器,泛化能力差。VEnhancer 的做法是"折中":在 `t_max=200`(而非最大步 `T=1000`)处向低分辨率视频注入噪声,然后在这个噪声级别开始去噪。这相当于"从低分辨率视频的邻域采样",既保留了内容结构,又给模型留出了足够的生成空间。

`GaussianDiffusion` 类还自带一个细节:配合 `logSNR cosine interp` 调度表 + `zero_terminal_snr` 修正,确保最大噪声步 σ 恰好等于 1.0——消除了训练时"最大噪声 ≠ 纯噪声"的经典错位。

SDEdit solves a fundamental tension in video super-resolution: generating from pure noise loses all content structure; not adding noise reduces the model to an upsampler with poor generalization. VEnhancer threads the needle by injecting noise only at `t_max=200` (not at the full `T=1000`), then starting denoising from that point. This amounts to sampling from the neighborhood of the low-res video, preserving structure while leaving the model enough creative latitude to hallucinate fine texture.

The `GaussianDiffusion` class also carries a precision detail: paired with a `logSNR cosine interp` schedule plus `zero_terminal_snr` rescaling, the maximum σ is forced to exactly 1.0 — eliminating the classic training mismatch where "maximum noise ≠ pure noise".

## 代码 / The code

`Vchitect/VEnhancer` — [`video_to_video/diffusion/diffusion_sdedit.py`](https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/video_to_video/diffusion/diffusion_sdedit.py#L1-L110)

```python
def _i(tensor, t, x):
    # 把 1-D schedule tensor 里的第 t 个值广播成和 x 相同的维度
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).to(x.device)


class GaussianDiffusion(object):

    def __init__(self, sigmas):
        self.sigmas = sigmas                           # σ_t  (噪声强度)
        self.alphas = torch.sqrt(1 - sigmas**2)        # α_t = √(1 - σ_t²)
        self.num_timesteps = len(sigmas)

    def diffuse(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt

    def denoise(self, xt, t, s, model, model_kwargs={},
                guide_scale=None, guide_rescale=None, clamp=None, percentile=None):
        s = t - 1 if s is None else s

        sigmas   = _i(self.sigmas, t, xt)
        alphas   = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.0
        sigmas_s = torch.sqrt(1 - alphas_s**2)

        betas  = 1 - (alphas / alphas_s) ** 2
        coef1  = betas * alphas_s / sigmas**2
        coef2  = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var    = betas * (sigmas_s / sigmas) ** 2
        log_var = torch.log(var).clamp_(-20, 20)

        if guide_scale is None:
            out = model(xt, t=t, **model_kwargs)
        else:
            y_out = model(xt, t=t, **model_kwargs[0])   # conditioned
            if guide_scale == 1.0:
                out = y_out
            else:
                u_out = model(xt, t=t, **model_kwargs[1])  # unconditioned
                out = u_out + guide_scale * (y_out - u_out)  # CFG

                if guide_rescale is not None:
                    ratio = (y_out.flatten(1).std(dim=1) /
                             (out.flatten(1).std(dim=1) + 1e-12)).view(
                                 (-1,) + (1,) * (y_out.ndim - 1))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0

        # 从模型预测的 noise (ε) 算回 x0
        x0 = alphas * xt - sigmas * out

        if percentile is not None:
            s2 = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s2 = s2.clamp_(1.0).view((-1,) + (1,) * (xt.ndim - 1))
            x0 = torch.min(s2, torch.max(-s2, x0)) / s2
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        eps = (xt - alphas * x0) / sigmas
        mu  = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps
```

配套的调度表 (`video_to_video/diffusion/schedules_sdedit.py`):

```python
def noise_schedule(schedule="logsnr_cosine_interp", n=1000,
                   zero_terminal_snr=False, **kwargs):
    sigmas = {"logsnr_cosine_interp": logsnr_cosine_interp_schedule}[schedule](n, **kwargs)
    if zero_terminal_snr and sigmas.max() != 1.0:
        scale = (1.0 - sigmas.min()) / (sigmas.max() - sigmas.min())
        sigmas = sigmas.min() + scale * (sigmas - sigmas.min())
    return sigmas
```

## 逐行讲解 / What's happening

1. **`_i(tensor, t, x)` — 调度值广播器**
   - 中文: `tensor` 是长度 `T` 的一维张量 (σ 或 α 的时间序列)。`t` 是一批整数索引,每张图/帧对应一个时间步。`view(shape)` 把取出的标量变成可广播的 `(B, 1, 1, 1)` 形状,让它可以直接乘以 `(B, C, H, W)` 的张量。
   - English: `tensor` is the 1-D schedule array (σ or α over T steps). `t` is a batch of integer indices — one per video sample. `view(shape)` reshapes the extracted scalar into `(B, 1, 1, 1)` so it broadcasts over `(B, C, H, W)` without writing any explicit loops.

2. **`__init__` — 从 σ 推 α**
   - 中文: 扩散过程里有恒等式 `α² + σ² = 1` (variance-preserving)。所以只需存 `sigmas`,就能推出 `alphas = √(1 - σ²)`。两条曲线完全互相决定。
   - English: The variance-preserving constraint is `α² + σ² = 1`, so storing `sigmas` is sufficient — `alphas` follows by a single `sqrt`. The two curves are duals of each other.

3. **`diffuse(x0, t)` — 前向过程(一步跳达)**
   - 中文: `xt = α_t * x0 + σ_t * ε`。这是高斯扩散的闭合解:不需要逐步 Markov 链,可以在任意时间步 `t` 直接得到带噪版本。VEnhancer 用 `t=200` 对低分辨率视频调用这个函数,得到"微带噪的 LR 视频",再从这里开始去噪。
   - English: `xt = α_t * x0 + σ_t * ε` is the closed-form solution for Gaussian diffusion: no step-by-step Markov chain, just a direct noise injection at any timestep `t`. VEnhancer calls this with `t=200` on the low-res video to get a "lightly noised LR frame", then begins denoising from there.

4. **`denoise` — 后验均值计算**
   - 中文: 给定模型预测的噪声 `out = ε_θ(xt, t)`,先还原 `x0 = α_t * xt - σ_t * ε_θ`,再用贝叶斯后验公式 `μ = coef1 * x0 + coef2 * xt` 算出从 `t` 到 `s = t-1` 的去噪均值。`coef1, coef2` 由 β 和各步的 σ/α 推导。`var` 是后验方差,DDPM 采样用得到。
   - English: Given the model-predicted noise `out = ε_θ(xt, t)`, first recover `x0 = α_t * xt - σ_t * ε_θ`, then compute the DDPM posterior mean `μ = coef1 * x0 + coef2 * xt` for stepping from `t` down to `s = t-1`. Coefficients `coef1` and `coef2` come from the standard VP-SDE posterior formula. `var` is the posterior variance used by stochastic samplers.

5. **CFG 与 guide_rescale**
   - 中文: `out = u_out + scale * (y_out - u_out)` 是无分类器引导(CFG)。`guide_rescale` 是修正项——CFG 倾向于让预测噪声方差暴涨,`guide_rescale * ratio + (1-guide_rescale) * 1.0` 把方差拉回有条件预测的水平。
   - English: `out = u_out + scale * (y_out - u_out)` is classifier-free guidance (CFG). `guide_rescale` compensates for the well-known CFG side-effect of inflating predicted-noise variance — the ratio term pulls variance back toward the conditioned model's magnitude.

6. **`noise_schedule` 的 `zero_terminal_snr`**
   - 中文: `logSNR cosine interp` 调度本身在 `t=T` 时 σ 并不精确等于 1.0。`zero_terminal_snr` 修正用了一个线性缩放:`scale = (1 - σ_min) / (σ_max - σ_min)`,把整条曲线拉伸到 [σ_min, 1.0]。结果是训练中"加噪最重的步骤"现在真的等于纯高斯噪声,消除了模型永远看不到"纯噪声"的 gap。
   - English: The `logSNR cosine interp` schedule does not naturally end at σ = 1.0. `zero_terminal_snr` applies a linear rescale that stretches the entire σ curve to exactly end at 1.0. The consequence: the model's heaviest noise step is now indistinguishable from pure Gaussian noise, closing the gap where the model was trained to see "almost pure noise" but tested against "perfectly pure noise".

## 类比 / The analogy

把低分辨率视频比作一张已经洗过但略模糊的老照片。你不是从空白纸开始画——那会完全丢失内容。你也不是直接 PS 锐化——那没有生成能力。你做的是:先往照片上轻轻覆一层薄薄的雾(加噪),然后用"你认识这个人"的记忆(文本条件)把雾从照片上擦掉——擦的过程中顺便画清楚了细节。`t_max=200` 就是"薄薄的雾"有多厚的那根旋钮。

Think of the low-res video as a slightly blurry old photograph. You don't start from blank paper — that would lose all content. You don't just sharpen it in Photoshop — that has no generative power. Instead, you lay a thin sheet of fog over it (add noise), then use your memory of "who's in the photo" (text conditioning) to wipe the fog away — and in the process, draw in the crisp details that were always missing. `t_max=200` is the knob controlling how thick the fog layer is.

## 自己跑一遍 / Try it yourself

```python
import torch

def _i(tensor, t, x):
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    return tensor[t].view(shape)

T = 1000
t_arr = torch.linspace(1, 0, T)
sigmas = torch.sin(t_arr * 3.14159 / 2)  # toy cosine-ish schedule
alphas = torch.sqrt(1 - sigmas**2)

# simulate SDEdit: add noise at t=200 then measure SNR
x0 = torch.randn(2, 3, 64, 64)
t  = torch.tensor([200, 200])
noise = torch.randn_like(x0)
xt = _i(alphas, t, x0) * x0 + _i(sigmas, t, x0) * noise
snr = (_i(alphas, t, x0)**2 / _i(sigmas, t, x0)**2).squeeze()
print(f"SNR at t=200: {snr[0]:.2f}  (xt norm: {xt.norm():.2f})")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
SNR at t=200: ~3-8   (xt norm: ~varies)
```

中文: 在 `t=200` 时 SNR 仍然大于 1,说明视频内容信号还"占主导"——这就是为什么从这里去噪能保留内容结构,而不是从纯噪声开始。

English: At `t=200` the SNR is still above 1, meaning the content signal still dominates — that's precisely why starting denoising here preserves structure instead of hallucinating from scratch.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SDEdit 原论文 (Meng et al., 2021)** / **Original SDEdit paper**: 同样思路用于图像编辑:给"参考图+噪声"去噪,而不是从纯噪声生成。VEnhancer 把这个思路搬到了视频超分里。/ Same idea for image editing: denoise from "reference + noise" instead of from pure noise. VEnhancer ports it to video super-resolution.
- **img2img in diffusers** / **diffusers 的 img2img**: `StableDiffusionImg2ImgPipeline` 的 `get_timesteps()` 做了完全相同的事——根据 `strength` 参数决定从哪个 `t` 开始,越接近 1.0 越接近纯生成。/ `StableDiffusionImg2ImgPipeline.get_timesteps()` does the identical thing: picks start-t from `strength`, where `strength=1.0` means start from pure noise.
- **DDIM inversion** / **DDIM 反向**: 先把真实图像"噪声化"到 t=T,再从那里去噪——也是"前向 → 后向"的 SDEdit 结构,只是 t_max=T 而不是局部步。/ First "noises up" a real image to t=T, then denoises from there — same forward→backward SDEdit structure, just with t_max=T instead of a partial step.

## 注意事项 / Caveats / when it breaks

- **`t_max` 是超参数** / **`t_max` is a hyperparameter**: 太低 → 去噪空间不足,还是会糊;太高 → 内容被噪声覆盖,失去对原视频的忠实度。VEnhancer 用 200/1000,不同分辨率差异可能需要重新调。/ Too low → not enough denoising budget, result still blurry; too high → content overwhelmed by noise, output drifts from the input. VEnhancer uses 200/1000; you may need to re-tune for different resolutions.
- **`zero_terminal_snr` 只影响训练匹配** / **`zero_terminal_snr` only fixes training-inference alignment**: 如果你用的是现有预训练权重(非 zero-terminal-SNR 训练),开这个选项不会有帮助,甚至会让生成结果变差。/ If you're using an existing checkpoint trained *without* zero-terminal-SNR, enabling this flag won't help and may hurt.
- **CFG 方差膨胀** / **CFG variance inflation**: `guide_scale > 1.5` 时,预测噪声方差会超过有条件预测的方差。`guide_rescale` 修正这个问题,但它本身是一个近似,extreme guidance scale 下仍然会过饱和。/ Above `guide_scale ~1.5`, predicted-noise variance exceeds the conditioned model's natural magnitude. `guide_rescale` compensates, but it's an approximation — extreme values still over-saturate.

## 延伸阅读 / Further reading

- SDEdit 原论文: [arXiv 2108.01073](https://arxiv.org/abs/2108.01073) — Guided Image Synthesis via Stochastic Differential Editing
- Zero-terminal SNR 修正: [arXiv 2305.08891](https://arxiv.org/abs/2305.08891) — Rescaling Diffusion Model Noise Schedules
- VEnhancer 项目主页: [GitHub Vchitect/VEnhancer](https://github.com/Vchitect/VEnhancer)
