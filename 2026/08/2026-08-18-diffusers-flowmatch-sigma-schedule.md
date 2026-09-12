---
date: 2026-08-18
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
permalink: https://github.com/huggingface/diffusers/blob/9284607295a09f759aadd65ed08f48b35feea6d9/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L241-L377
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers]
---

# Diffusers FlowMatch schedule：sigma 时间表先变形再补终点 / Diffusers FlowMatch Schedule: Shape Sigmas, Then Append the Endpoint

> **一句话 / In one line**: `FlowMatchEulerDiscreteScheduler.set_timesteps` 先构造 sigma 序列，再按 shift、terminal、Karras/exponential/beta 规则变形，最后追加采样终点。 / `FlowMatchEulerDiscreteScheduler.set_timesteps` builds sigmas, reshapes them with shift, terminal, and Karras/exponential/beta rules, then appends the sampling endpoint.

## 为什么重要 / Why this matters

扩散采样质量很大一部分来自“每一步走在哪些噪声水平上”。这段代码把默认步数、自定义 timesteps、自定义 sigmas 和不同 schedule 变体都收敛到同一条 `self.sigmas` 数组，后续 `step()` 只需要按 index 前进。

A large part of diffusion sampling quality comes from the noise levels chosen for each step. This code funnels default steps, custom timesteps, custom sigmas, and schedule variants into one `self.sigmas` array, so `step()` can simply advance by index.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/9284607295a09f759aadd65ed08f48b35feea6d9/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L346-L377)

```python
if self.config.use_dynamic_shifting:
    sigmas = self.time_shift(mu, 1.0, sigmas)
else:
    sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

if self.config.shift_terminal:
    sigmas = self.stretch_shift_to_terminal(sigmas)

if self.config.use_karras_sigmas:
    sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
elif self.config.use_exponential_sigmas:
    sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
elif self.config.use_beta_sigmas:
    sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
timesteps = sigmas * self.config.num_train_timesteps
```

## 逐行讲解 / What's happening

1. **第 346-351 行 / Lines 346-351**: 中文: sigma 先经过 dynamic shift 或固定 shift，改变采样点在高噪声/低噪声区域的密度。 / English: Sigmas first pass through dynamic or fixed shifting, changing how densely steps cover high- and low-noise regions.
2. **第 353-355 行 / Lines 353-355**: 中文: 如果配置了 terminal，时间表会被拉伸到指定终点，避免最后一步停在不想要的 sigma。 / English: If a terminal value is configured, the schedule is stretched so the last point lands where requested.
3. **第 357-363 行 / Lines 357-363**: 中文: Karras、exponential、beta 三种变体互斥，都是在同一条初始 sigma 轨道上重新分布步点。 / English: Karras, exponential, and beta variants are mutually exclusive ways to redistribute points along the initial sigma track.
4. **第 365-377 行 / Lines 365-377**: 中文: 代码把 sigma 转成 tensor，并追加 `0` 或 `1` 作为隐式终点，让每一步都能读到 next sigma。 / English: Sigmas become tensors, then `0` or `1` is appended as the implicit endpoint so every step can read a next sigma.

## 类比 / The analogy

像给登山路线安排休息站：先定总路线，再决定前半段站点密一点还是后半段密一点，最后一定要把终点站补上。

It is like placing rest stops on a hiking route: define the route, decide whether to pack stops early or late, and always append the final stop.

## 自己跑一遍 / Try it yourself

```python
def shifted_sigmas(sigmas, shift):
    return [round(shift * s / (1 + (shift - 1) * s), 4) for s in sigmas]

def append_terminal(sigmas, invert=False):
    return sigmas + ([1.0] if invert else [0.0])

base = [1.0, 0.75, 0.5, 0.25]
shifted = shifted_sigmas(base, 1.5)
print(shifted)
print(append_terminal(shifted))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1.0, 0.8182, 0.6, 0.3333]
[1.0, 0.8182, 0.6, 0.3333, 0.0]
```

shift 改变了中间采样点，终点则作为额外 sigma 被补上。

The shift moves the intermediate sampling points, and the endpoint is appended as an extra sigma.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **image-to-image restart** / **image-to-image restarts**: 中途开始采样时，需要 schedule 能准确找到当前 sigma index。 / Mid-schedule starts need the scheduler to locate the current sigma index precisely.
- **video diffusion** / **video diffusion**: 长视频通常更依赖高质量 schedule，因为每一步误差会跨帧放大。 / Long video generation depends heavily on good schedules because errors propagate across frames.

## 注意事项 / Caveats / when it breaks

- **自定义长度要匹配** / **Custom lengths must match**: 同时传 `sigmas` 和 `timesteps` 时长度必须一致。 / If both `sigmas` and `timesteps` are passed, their lengths must match.
- **schedule 变体互斥** / **Schedule variants are exclusive**: 同时打开多种 sigma 转换会在初始化阶段报错。 / Enabling multiple sigma conversions is rejected during initialization.

## 延伸阅读 / Further reading

- [Diffusers source](https://github.com/huggingface/diffusers/blob/9284607295a09f759aadd65ed08f48b35feea6d9/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L241-L377)
