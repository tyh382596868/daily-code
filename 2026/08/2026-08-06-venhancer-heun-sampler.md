---
date: 2026-08-06
topic: diffusion
source: tracked
repo: Vchitect/VEnhancer
file: video_to_video/diffusion/solvers_sdedit.py
permalink: https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/video_to_video/diffusion/solvers_sdedit.py#L24-L64
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, heun-sampler]
---

# VEnhancer Heun sampler：先试走一步，再用平均斜率修正 / VEnhancer Heun Sampler: Take a Trial Step, Then Correct with the Average Slope

> **一句话 / In one line**: Heun sampler 把一次 denoise 调用变成 predictor-corrector，让视频 SDEdit 采样少一点偏航。 / The Heun sampler turns one denoising step into a predictor-corrector update, reducing drift in video SDEdit sampling.

## 为什么重要 / Why this matters

扩散采样不是简单地把噪声一点点减掉，而是在一条由模型预测出的轨道上积分。Euler 只看当前位置的斜率，便宜但容易偏；Heun 先走一个试探步，再在新位置再问一次模型，把两个斜率平均后更新。

Diffusion sampling is numerical integration over a trajectory predicted by the model. Euler uses only the slope at the current point; Heun probes the next point too, then averages both slopes before committing the update.

## 代码 / The code

`Vchitect/VEnhancer` — [`video_to_video/diffusion/solvers_sdedit.py`](https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/video_to_video/diffusion/solvers_sdedit.py#L24-L64)

```python
def get_scalings(sigma):
    c_out = -sigma
    c_in = 1 / (sigma**2 + 1.0**2) ** 0.5
    return c_out, c_in


@torch.no_grad()
def sample_heun(noise, model, sigmas, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, show_progress=True):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.0
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float("inf"):
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if sigmas[i] == float("inf"):
            # Euler method
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            _, c_in = get_scalings(sigma_hat)
            denoised = model(x * c_in, sigma_hat)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                _, c_in = get_scalings(sigmas[i + 1])
                denoised_2 = model(x_2 * c_in, sigmas[i + 1])
                d_2 = (x_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    return x
```

## 逐行讲解 / What's happening

1. **第 24-27 行 / Lines 24-27 (`get_scalings`)**:
   - 中文: `c_in` 把输入按 sigma 归一化，让模型在不同噪声强度下看到稳定尺度。
   - English: `c_in` normalizes the input by sigma so the model sees a stable scale across noise levels.
2. **第 35-43 行 / Lines 35-43 (noise churn)**:
   - 中文: 采样可以在指定 sigma 区间额外注入一点随机性，用 `gamma` 拉高当前 sigma。
   - English: The sampler can add controlled stochasticity in a sigma window by inflating the current sigma with `gamma`.
3. **第 49-52 行 / Lines 49-52 (first slope)**:
   - 中文: 模型先给当前点的 denoised 估计，`(x - denoised) / sigma_hat` 就是沿噪声轴回退的斜率。
   - English: The model predicts the denoised point; `(x - denoised) / sigma_hat` becomes the slope for stepping along the noise axis.
4. **第 58-63 行 / Lines 58-63 (Heun correction)**:
   - 中文: 先用 `d` 走到 `x_2`，再计算 `d_2`，最后用平均斜率更新，减少单点估计的误差。
   - English: It first predicts `x_2` with slope `d`, recomputes slope `d_2`, then updates with their average to reduce single-slope error.

## 类比 / The analogy

像开车转弯时先按当前方向打一把方向盘，但你不会闭眼开完整个弯；你会看车头的新方向，再微调方向盘。Heun 就是这个“看一眼再修正”的采样器。

It is like steering through a curve: you turn based on the current heading, look at where the car now points, then correct. Heun is that look-and-correct loop for sampling.

## 自己跑一遍 / Try it yourself

```python
def model(x, sigma):
    return x * (1 - 0.25 * sigma)

sigmas = [1.0, 0.5, 0.0]
x = 2.0 * sigmas[0]
for i in range(len(sigmas) - 1):
    sigma = sigmas[i]
    denoised = model(x, sigma)
    d = (x - denoised) / sigma
    dt = sigmas[i + 1] - sigma
    if sigmas[i + 1] == 0:
        x = x + d * dt
    else:
        x2 = x + d * dt
        denoised2 = model(x2, sigmas[i + 1])
        d2 = (x2 - denoised2) / sigmas[i + 1]
        x = x + ((d + d2) / 2) * dt
print(round(x, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
1.5449
```

这个 toy 例子把 tensor 换成 float，但保留了 predictor-corrector 的结构。

The toy version replaces tensors with floats, but keeps the predictor-corrector structure intact.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Karras EDM 采样器** / **Karras EDM samplers**: 用 sigma schedule 加二阶修正提升低步数质量。 / Sigma schedules plus second-order correction improve quality at low step counts.
- **Wan2.1 FlowDPMSolver** / **Wan2.1 FlowDPMSolver**: 多步 solver 用历史模型输出做更高阶修正。 / The multistep solver uses historical model outputs for higher-order correction.

## 注意事项 / Caveats / when it breaks

- **多一次模型调用** / **One extra model call**: 非终点 step 会调用两次 denoiser，速度比 Euler 慢。 / Non-terminal steps call the denoiser twice, so it is slower than Euler.
- **sigma 为 0 要特殊处理** / **Zero sigma needs special handling**: 最后一步不能除以 0，所以退回 Euler 更新。 / The final step cannot divide by zero, so it falls back to Euler.

## 延伸阅读 / Further reading

- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models"
- [VEnhancer repository](https://github.com/Vchitect/VEnhancer)
