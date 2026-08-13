---
date: 2026-08-13
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_scm.py
permalink: https://github.com/huggingface/diffusers/blob/614ae4bb9df07d102ce1187777ef1c62f8aab0e8/src/diffusers/schedulers/scheduling_scm.py#L237-L302
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, scheduler]
---

# Diffusers SCM step：用三角参数化从噪声回到样本 / Diffusers SCM Step: Walk from Noise to Sample with Trigflow

> **一句话 / In one line**: `SCMScheduler.step()` 先从当前 `sample` 和模型输出恢复 `pred_x0`，多步采样时再用下一个 timestep 混入新噪声。 / `SCMScheduler.step()` reconstructs `pred_x0` from the current `sample` and model output, then remixes it with fresh noise at the next timestep.

## 为什么重要 / Why this matters

scheduler 是 diffusion pipeline 里最容易被误读的部分：模型预测向量场，真正决定下一步 sample 怎么来的，是 scheduler。这段 SCM 代码清楚展示了 trigflow 里的 `cos/sin` 更新。

Schedulers are easy to misread in diffusion pipelines: the model predicts a vector field, but the scheduler decides how the next sample is formed. This SCM code clearly shows the `cos/sin` trigflow update.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_scm.py`](https://github.com/huggingface/diffusers/blob/614ae4bb9df07d102ce1187777ef1c62f8aab0e8/src/diffusers/schedulers/scheduling_scm.py#L237-L302)

```python
def step(
    self,
    model_output: torch.FloatTensor,
    timestep: float,
    sample: torch.FloatTensor,
    generator: torch.Generator | None = None,
    return_dict: bool = True,
) -> SCMSchedulerOutput | tuple:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator for reproducible sampling.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_scm.SCMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_scm.SCMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_scm.SCMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # 2. compute alphas, betas
    t = self.timesteps[self.step_index + 1]
    s = self.timesteps[self.step_index]

    # 4. Different Parameterization:
    parameterization = self.config.prediction_type

    if parameterization == "trigflow":
        pred_x0 = torch.cos(s) * sample - torch.sin(s) * model_output
    else:
        raise ValueError(f"Unsupported parameterization: {parameterization}")

    # 5. Sample z ~ N(0, I), For MultiStep Inference
    # Noise is not used for one-step sampling.
    if len(self.timesteps) > 1:
        noise = (
            randn_tensor(model_output.shape, device=model_output.device, generator=generator)
            * self.config.sigma_data
        )
        prev_sample = torch.cos(t) * pred_x0 + torch.sin(t) * noise
    else:
        prev_sample = pred_x0

    self._step_index += 1

    if not return_dict:
        return (prev_sample, pred_x0)

    return SCMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)
```

## 逐行讲解 / What's happening

1. **第 266-272 行 / Lines 266-272: `set_timesteps()` 是前置条件，第一次 step 会初始化内部 step index。 / `set_timesteps()` is required, and the first step initializes the internal step index.**
2. **第 274-283 行 / Lines 274-283: 当前角度 `s` 把 sample 和 model output 组合成 `pred_x0`。 / Current angle `s` combines sample and model output into `pred_x0`.**
3. **第 286-295 行 / Lines 286-295: 多步采样按下一个角度 `t` 把 `pred_x0` 和新噪声重新混合。 / Multistep sampling remixes `pred_x0` and fresh noise using next angle `t`.**
4. **第 297-302 行 / Lines 297-302: index 前进，并返回 next sample 与 clean estimate。 / The index advances and returns both the next sample and clean estimate.**

## 类比 / The analogy

像调音台两个旋钮：一个旋钮把当前混音还原成主旋律，另一个旋钮再按下一拍混入背景噪声。

It is like two knobs on an audio mixer: one recovers the melody from the current mix, and the other blends in background noise for the next beat.

## 自己跑一遍 / Try it yourself

```python
import math

def scm_step(sample, model_output, s, t):
    pred_x0 = math.cos(s) * sample - math.sin(s) * model_output
    noise = 0.0
    prev = math.cos(t) * pred_x0 + math.sin(t) * noise
    return prev, pred_x0

prev, x0 = scm_step(sample=2.0, model_output=0.5, s=1.0, t=0.3)
print(round(x0, 4), round(prev, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.6599 0.6304
```

噪声固定为 0 后，可以直接看到 `t` 对下一步 sample 的缩放作用。

With noise fixed at zero, you can directly see how `t` scales the next sample.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FlowMatch Euler** / **FlowMatch Euler**: 也是 scheduler 接管一步更新，但用 ODE 形式推进 sample。 / The scheduler also owns the step update, but advances with an ODE-style rule.
- **DDPM scheduler** / **DDPM scheduler**: 经典 scheduler 同样先估计原图，再采下一步。 / Classic schedulers also estimate the original sample before drawing the next state.

## 注意事项 / Caveats / when it breaks

- **必须先设 timesteps** / **Timesteps must be initialized**: 没有时间表，`step_index` 无意义。 / Without a schedule, `step_index` has no meaning.
- **prediction_type 要匹配训练** / **`prediction_type` must match training**: 源码只接受 `trigflow`。 / The source accepts only `trigflow`.

## 延伸阅读 / Further reading

- [huggingface/diffusers source](https://github.com/huggingface/diffusers/blob/614ae4bb9df07d102ce1187777ef1c62f8aab0e8/src/diffusers/schedulers/scheduling_scm.py#L237-L302)
