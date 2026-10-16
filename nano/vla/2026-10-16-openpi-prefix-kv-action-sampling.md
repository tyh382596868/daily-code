---
date: 2026-10-16
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models_pytorch/pi0_pytorch.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py#L377-L455
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-head-continuous, kv-cache]
build_role: continuous action head, advanced inference variant
---

# openpi 动作采样：前缀算一次，动作去噪多步复用 / openpi Action Sampling: Compute the Prefix Once, Reuse It Across Denoising Steps

> **一句话 / In one line**: π₀ 先缓存图像和语言的 KV，再只对动作 suffix 做 Euler 流匹配积分。 / π₀ caches image-and-language KV once, then performs Euler flow-matching integration only over the action suffix.

## 为什么重要 / Why this matters

动作 chunk 去噪要调用模型多次，但视觉和语言条件在同一个控制 tick 内不变。缓存稳定前缀把昂贵的重复计算从每一步挪到一次。

Action-chunk denoising calls the model repeatedly, but visual and language context are stable within one control tick. Caching that stable prefix moves expensive repeated work out of every step.

## 代码 / The code

`Physical-Intelligence/openpi` — [`pi0_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py#L377-L455)

```python
_, past_key_values = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
    past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
)
dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
x_t = noise
time = torch.tensor(1.0, dtype=torch.float32, device=device)
while time >= -dt / 2:
    v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
    x_t = x_t + dt * v_t
    time += dt
return x_t
```

## 逐行讲解 / What's happening

1. 中文: 前缀 forward 开 `use_cache=True`，把相机和语言的注意力记忆存为 `past_key_values`。
   English: Prefix forward enables `use_cache=True`, storing camera and language attention memory as `past_key_values`.
2. 中文: 从噪声 `x_t` 开始，时间从 1 走向 0；`denoise_step` 只需拼新动作 token。
   English: Sampling starts at noisy `x_t` and walks time from 1 to 0; `denoise_step` only appends fresh action tokens.
3. 中文: `x_t = x_t + dt * v_t` 是显式 Euler 积分，返回整段动作 horizon。
   English: `x_t = x_t + dt * v_t` is explicit Euler integration and returns the full action horizon.

## 类比 / The analogy

像先把导航地图展开一次，再对同一条路线连续微调方向盘；不必每转一度就重画整座城市。

It is unfolding the navigation map once and repeatedly adjusting the steering wheel, not redrawing the whole city for every degree.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `action-head-continuous` 的进阶推理实现。上游的视觉/语言投影器和 VLM backbone 生成 prefix；本模块输入状态与噪声动作，输出连续 action chunk；下游是限幅、反归一化和执行器。省掉 cache 不会改变语义，却会让多步采样的延迟随 prefix 长度被重复支付。生产版还需处理 cache 生命周期、batch 异构长度、提前中断和安全 watchdog。

English: This is an advanced `action-head-continuous` inference implementation. Vision/language projectors and the VLM backbone produce the prefix; this module consumes state plus noisy actions and emits a continuous action chunk for unnormalization and execution. Omitting the cache preserves semantics but repays prefix latency on every step. Production code also needs cache lifetime rules, heterogeneous lengths, cancellation, and a safety watchdog.

## 自己跑一遍 / Try it yourself

```python
def sample(x, steps):
    dt = -1 / steps
    t = 1.0
    while t > 0:
        velocity = -x  # toy flow toward zero
        x = x + dt * velocity
        t += dt
    return round(x, 3)

print(sample(1.0, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
2.441
```

中文: 这个玩具只演示积分结构；真正 velocity 由条件模型预测。  
English: The toy only shows the integration structure; a conditioned model predicts the real velocity.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot diffusion action head** / **LeRobot diffusion action head**: 中文: 同样多步还原动作，但采用扩散参数化。 / English: It also restores actions over multiple steps, with a diffusion parameterization.
- **FastWAM action cache** / **FastWAM action cache**: 中文: 稳定视频 KV 跨动作步骤复用。 / English: Stable video KV is reused across action steps.

## 注意事项 / Caveats / when it breaks

- 中文: 条件一变，旧 `past_key_values` 必须丢弃。 / English: A changed condition invalidates old `past_key_values`.
- 中文: Euler 步数和步长是质量/延迟权衡。 / English: Euler count and step size trade quality against latency.

## 延伸阅读 / Further reading

- [openpi sample_actions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py#L377-L455)
