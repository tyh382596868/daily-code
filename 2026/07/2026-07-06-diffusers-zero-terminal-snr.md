---
date: 2026-07-06
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
permalink: https://github.com/huggingface/diffusers/blob/cbdb63798bc627e35ca6656265b7185e7618ad92/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L88-L116
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusion-scheduler]
---

# Diffusers zero-terminal SNR：把最后一步真的推到纯噪声 / Diffusers Zero-Terminal SNR: Make the Last Step Truly Pure Noise

> **一句话 / In one line**: 这段函数重标定 beta schedule，让累计信号强度在最后一个 timestep 变成 0。 / This function rescales a beta schedule so the cumulative signal strength reaches 0 at the final timestep.

## 为什么重要 / Why this matters

扩散模型的噪声表不只是训练细节，它决定了模型在最末端看到的是“几乎纯噪声”还是“还残留一点图像”。zero-terminal-SNR 的思路是保留起点强度，把终点强度平移到 0，再反推新的 beta。这样 scheduler 的边界条件更干净。

A diffusion noise schedule is not just bookkeeping; it decides whether the final timestep is nearly pure noise or still carries residual image signal. Zero-terminal SNR keeps the starting strength, shifts the ending strength to 0, then converts the curve back into betas. The scheduler gets cleaner boundary conditions.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`](https://github.com/huggingface/diffusers/blob/cbdb63798bc627e35ca6656265b7185e7618ad92/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L88-L116)

```python
def rescale_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas
```

## 逐行讲解 / What's happening

1. **第 3-5 行 / Lines 3-5 (`cumprod`)**:
   - 中文: beta 先转成每步保留率 `alpha`，再累计成整条时间线的信号保留率。
   - English: betas become per-step signal retention `alpha`, then a cumulative signal-retention curve.
2. **第 8-9 行 / Lines 8-9 (`clone`)**:
   - 中文: 记录原来的起点和终点，后面要用它们做仿射重标定。
   - English: the original start and end values are saved for the affine rescale.
3. **第 12-15 行 / Lines 12-15 (shift and scale)**:
   - 中文: 先把终点挪到 0，再把起点拉回原来的高度。
   - English: first move the endpoint to 0, then stretch the curve so the start returns to its old height.
4. **第 18-21 行 / Lines 18-21 (back to betas)**:
   - 中文: scheduler 最终还是吃 beta，所以把累计曲线拆回逐步 alpha，再转成 beta。
   - English: schedulers consume betas, so the cumulative curve is decomposed back into per-step alphas and then betas.

## 类比 / The analogy

像把一段音频做归一化：结尾要真正静音，但开头音量不能变。你先减掉结尾残留的底噪，再整体放大回原来的开头音量。

It is like normalizing an audio clip whose ending should be truly silent while the beginning keeps its original volume. You subtract the ending noise floor, then scale the whole clip back up.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

betas = np.linspace(0.01, 0.1, 5)
alphas = 1 - betas
abar = np.sqrt(np.cumprod(alphas))
start, end = abar[0], abar[-1]
shifted = (abar - end) * start / (start - end)
new_abar = shifted ** 2
new_alphas = np.r_[new_abar[:1], new_abar[1:] / new_abar[:-1]]
print(np.round(1 - new_alphas, 4))
print(round(shifted[-1], 6))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[0.01   0.3485 0.5194 0.7497 1.    ]
0.0
```

最后一个累计信号强度被压到 0，所以最后一步 beta 可以到 1。

The final cumulative signal strength reaches 0, so the last beta can become 1.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DDIM / DPM schedulers** / **DDIM / DPM schedulers**: 很多 scheduler 都共享 beta、alpha、alpha-bar 这三个视角。 / Many schedulers move among beta, alpha, and alpha-bar views.
- **视频扩散采样器** / **Video diffusion samplers**: 边界条件越干净，少步数采样时越不容易留下亮度偏差。 / Cleaner boundary conditions help short samplers avoid brightness bias.

## 注意事项 / Caveats / when it breaks

- **不要在训练和推理随意混用 schedule** / **Do not casually mix schedules between train and inference**: 模型学到的是某条噪声曲线。 / The model was trained against a particular noise curve.
- **重标定会改变中间步密度** / **Rescaling changes middle-step density**: 不是只改最后一步。 / This changes the whole curve, not only the last step.

## 延伸阅读 / Further reading

- [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)
