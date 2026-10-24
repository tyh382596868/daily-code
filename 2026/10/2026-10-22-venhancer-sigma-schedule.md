---
date: 2026-10-22
topic: diffusion
source: tracked
repo: Vchitect/VEnhancer
file: video_to_video/diffusion/schedules_sdedit.py
permalink: https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/video_to_video/diffusion/schedules_sdedit.py#L8-L20
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, noise-schedule]
---

# VEnhancer 噪声刻度：在 beta、sigma 与 log-SNR 间切换 / VEnhancer Noise Coordinates: Move Between Beta, Sigma, and Log-SNR

> **一句话 / In one line**: 同一条扩散时间线可用三种坐标描述，转换函数让采样器不必绑死在训练参数化上。 / One diffusion timeline has several useful coordinate systems; these conversions keep a sampler independent of training parameterization.

## 为什么重要 / Why this matters

训练常以 beta 表达每步加多少噪声，采样器却更适合直接用 sigma；log-SNR 则把极端区间拉开，便于设计平滑 schedule。这里所有转换都从保留信号能量与噪声能量互补这一约束推出。

Training often uses beta while samplers want sigma. Log-SNR spreads out the difficult ends of the trajectory, making schedules easier to shape. Each conversion follows from signal and noise energy being complementary.

## 代码 / The code

`Vchitect/VEnhancer` — [`schedules_sdedit.py`](https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/video_to_video/diffusion/schedules_sdedit.py#L8-L20)

```python
def betas_to_sigmas(betas):
    return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))

def sigmas_to_betas(sigmas):
    square_alphas = 1 - sigmas**2
    betas = 1 - torch.cat([square_alphas[:1], square_alphas[1:] / square_alphas[:-1]])
    return betas

def logsnrs_to_sigmas(logsnrs):
    return torch.sqrt(torch.sigmoid(-logsnrs))
```

## 逐行讲解 / What's happening

1. **累计乘积 / Cumulative product**：`cumprod(1 - betas)` 是还留下的信号方差；`1 -` 它就是噪声方差。 / It is retained signal variance; its complement is noise variance.
2. **反向转换 / Reverse conversion**：相邻 alpha-bar 相除恢复单步 alpha，再取补集得到 beta。 / Adjacent alpha-bars recover one-step alpha, then its complement gives beta.
3. **log-SNR / log-SNR**：`sigmoid(-logsnr)` 正好是噪声方差，数值范围自然落在 `(0, 1)`。 / It directly yields noise variance in `(0, 1)`.

## 类比 / The analogy

这像同一段山路的公里数、海拔和坡度：选哪一个取决于你要导航、画地图还是控制车速。 / It is like distance, altitude, and slope for the same road: choose the coordinate suited to navigation, mapping, or speed control.

## 自己跑一遍 / Try it yourself

```python
import torch
b = torch.tensor([.01, .02, .03])
s = torch.sqrt(1 - torch.cumprod(1 - b, 0))
print(s.round(decimals=4))
```

运行 / Run with: `python try.py`  
预期输出 / Expected output: 递增且小于 1 的三个 sigma。 / Three increasing sigmas below 1.

## 注意事项 / Caveats / when it breaks

- **范围 / Range**：`sigmas_to_betas` 假设 sigma 对应单调、合法的离散轨迹。 / It assumes a valid monotone discrete trajectory.

## 延伸阅读 / Further reading

- [VEnhancer source](https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/video_to_video/diffusion/schedules_sdedit.py)
