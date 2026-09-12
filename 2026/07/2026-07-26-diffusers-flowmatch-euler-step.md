---
date: 2026-07-26
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
permalink: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, flow-matching]
---

# Diffusers FlowMatch Euler：一步就是沿速度场走一小段 / Diffusers FlowMatch Euler: One Step Walks Along the Velocity Field

> **一句话 / In one line**: Flow-matching scheduler 把模型输出当速度，用相邻 sigma 的差值更新 sample。 / A flow-matching scheduler treats model output as velocity and updates the sample by the delta between neighboring sigmas.

## 为什么重要 / Why this matters

扩散采样器常被讲成黑盒。Euler flow-matching 的核心很朴素：当前样本在噪声时间 `sigma` 上，模型预测“往干净方向怎么走”，scheduler 用下一步 sigma 的差值把它推进一点。

Diffusion samplers are often treated as black boxes. Euler flow matching is simple at its core: the current sample sits at a noise level `sigma`, the model predicts a direction, and the scheduler advances it by the sigma delta.

## 代码 / The code

`huggingface/diffusers` — [`scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)

```python
def step(self, model_output, timestep, sample, return_dict=True):
    if self.step_index is None:
        self._init_step_index(timestep)

    sample = sample.to(torch.float32)
    sigma = self.sigmas[self.step_index]
    sigma_next = self.sigmas[self.step_index + 1]

    prev_sample = sample + (sigma_next - sigma) * model_output
    prev_sample = prev_sample.to(model_output.dtype)

    self._step_index += 1

    if not return_dict:
        return (prev_sample,)
    return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
```

## 逐行讲解 / What's happening

1. **`_init_step_index`**
   - 中文: 第一次调用时根据传入 timestep 找到当前 scheduler 位置，之后每步自增。
   - English: The first call maps the external timestep to an internal scheduler index; later calls simply increment it.
2. **`sample.to(torch.float32)`**
   - 中文: 即便模型输出是半精度，更新公式也先用 float32，减少长采样链的数值误差。
   - English: Even if the model runs in half precision, the update is computed in float32 to reduce accumulated numerical error.
3. **`sigma_next - sigma`**
   - 中文: 这个差值就是 Euler 步长，sigma 往 0 走，所以它通常是负数。
   - English: This delta is the Euler step size; sigma moves toward zero, so it is often negative.
4. **`prev_sample`**
   - 中文: 模型输出不是直接结果，而是速度场；结果来自“当前位置 + 步长 * 速度”。
   - English: The model output is not the final sample; it is a velocity field integrated by one step.

## 类比 / The analogy

这像在地图上沿风向走路：风向告诉你方向，两个等高线之间的距离告诉你这一步走多远。

It is like walking on a map with wind arrows: the arrow gives direction, and the distance between contour lines tells you how long the step is.

## 自己跑一遍 / Try it yourself

```python
sigmas = [1.0, 0.6, 0.2, 0.0]
sample = 10.0
for i, velocity in enumerate([2.0, 1.0, 0.5]):
    sample = sample + (sigmas[i + 1] - sigmas[i]) * velocity
    print(round(sample, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
9.2
8.8
8.7
```

中文: sigma 差值越大，同样的模型速度会造成更大的样本移动。
English: A larger sigma gap makes the same model velocity move the sample farther.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DDIM** / **DDIM**: 中文: 也是根据相邻噪声级别把样本从一步搬到下一步。 / English: It also moves a sample between neighboring noise levels.
- **rectified flow VLA heads** / **Rectified-flow VLA heads**: 中文: action head 常把动作去噪写成同类速度场积分。 / English: Action heads often formulate denoising as integrating a velocity field.

## 注意事项 / Caveats / when it breaks

- **timestep 顺序** / **Timestep order**: 中文: scheduler 的内部索引必须和外部 timestep 对齐。 / English: The internal index must match the external timestep order.
- **dtype 回写** / **dtype casting back**: 中文: 更新后要回到模型 dtype，否则后续 kernel 可能多耗显存。 / English: Casting back avoids unexpected memory growth in later kernels.

## 延伸阅读 / Further reading

- [Diffusers scheduler source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)
- [Diffusers schedulers overview](https://huggingface.co/docs/diffusers/using-diffusers/schedulers)
