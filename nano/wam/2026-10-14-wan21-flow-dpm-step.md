---
date: 2026-10-14
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/utils/fm_solvers.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L707-L799
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, dpm-solver]
build_role: sampler-inference advanced variant, multistep flow-matching solver step
---

# Wan2.1 Flow DPM step：历史预测决定这一脚怎么迈 / Wan2.1 Flow DPM Step: Past Predictions Decide the Next Move

> **一句话 / In one line**: Wan2.1 的 flow DPM solver 把模型输出转成 solver 形式，滚动保存历史，再按一阶、二阶或三阶更新 latent。 / Wan2.1's flow DPM solver converts model output, rolls a history buffer, then updates the latent with first-, second-, or third-order logic.

## 为什么重要 / Why this matters

WAM 推理不只是“模型跑很多次”；采样器决定每一步如何使用当前预测和历史预测，直接影响速度、稳定性和视频动作轨迹。

WAM inference is not just running the model repeatedly. The sampler decides how each step uses current and past predictions, which directly affects speed, stability, and the generated action-video trajectory.

## 代码 / The code

`Wan-Video/Wan2.1` - [`wan/utils/fm_solvers.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L707-L799)

```python
# Modified from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.step
def step(
    self,
    model_output: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    sample: torch.Tensor,
    generator=None,
    variance_noise: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[SchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
    the multistep DPMSolver.
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # Improve numerical stability for small number of steps
    lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
        self.config.euler_at_final or
        (self.config.lower_order_final and len(self.timesteps) < 15) or
        self.config.final_sigmas_type == "zero")
    lower_order_second = ((self.step_index == len(self.timesteps) - 2) and
                          self.config.lower_order_final and
                          len(self.timesteps) < 15)

    model_output = self.convert_model_output(model_output, sample=sample)
    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
    self.model_outputs[-1] = model_output

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)
    if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"
                                     ] and variance_noise is None:
        noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=torch.float32)
    elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
        noise = variance_noise.to(
            device=model_output.device,
            dtype=torch.float32)  # pyright: ignore
    else:
        noise = None

    if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
        prev_sample = self.dpm_solver_first_order_update(
            model_output, sample=sample, noise=noise)
    elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
        prev_sample = self.multistep_dpm_solver_second_order_update(
            self.model_outputs, sample=sample, noise=noise)
    else:
        prev_sample = self.multistep_dpm_solver_third_order_update(
            self.model_outputs, sample=sample)

    if self.lower_order_nums < self.config.solver_order:
        self.lower_order_nums += 1

    # Cast sample back to expected dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    self._step_index += 1  # pyright: ignore

    if not return_dict:
        return (prev_sample,)

    return SchedulerOutput(prev_sample=prev_sample)
```

## 逐行讲解 / What's happening

1. **第 739-745 行 / Lines 739-745 (scheduler state)**:
   - 中文: 没有先 `set_timesteps` 就报错；第一次调用时用当前 timestep 初始化 step index。
   - English: It errors if `set_timesteps` was not called; on the first step it initializes the step index from the current timestep.
2. **第 747-759 行 / Lines 747-759 (history buffer)**:
   - 中文: 最后几步或步数很少时会降阶；然后把当前预测滚入 `model_outputs` 历史队列。
   - English: The solver drops order near the tail or for short schedules, then rolls the current prediction into the `model_outputs` history buffer.
3. **第 761-775 行 / Lines 761-775 (precision and SDE noise)**:
   - 中文: 更新前把 sample 升到 fp32；SDE 变体需要额外噪声，ODE 变体不需要。
   - English: The sample is upcast to fp32 before the update; SDE variants need extra noise, while ODE variants do not.
4. **第 777-799 行 / Lines 777-799 (order selection)**:
   - 中文: 冷启动先用低阶，历史够了才升到二阶/三阶；最后 cast 回模型 dtype 并推进 step index。
   - English: Cold start uses lower order, then upgrades to second or third order once enough history exists; the result is cast back and the step index advances.

## 类比 / The analogy

这像开车过弯：刚进弯时只看当前方向盘，连续几帧以后可以根据前几次转向趋势更平滑地修正。

It is like steering through a curve: at entry you only trust the current steering angle, but after a few moments you can use previous steering trends for a smoother correction.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `sampler-inference` 的高级版本，位于 DiT/flow 模型输出之后、VAE decode 之前。输入是当前 latent、timestep 和模型预测；输出是下一步 latent。生产实现还要管理 CFG、offload、分布式并行和中间帧预览。

English: This is an advanced `sampler-inference` component between the DiT/flow model output and VAE decoding. It takes the current latent, timestep, and model prediction, then returns the next latent. A production version also manages CFG, offload, distributed execution, and preview frames.

## 自己跑一遍 / Try it yourself

```python
history = []
order = 3

def step(x, pred):
    history.append(pred)
    del history[:-order]
    if len(history) == 1:
        update = history[-1]
    elif len(history) == 2:
        update = 1.5 * history[-1] - 0.5 * history[-2]
    else:
        update = (23 * history[-1] - 16 * history[-2] + 5 * history[-3]) / 12
    return x - 0.1 * update

x = 1.0
for p in [0.8, 0.6, 0.4]:
    x = step(x, p)
    print(round(x, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.92
0.87
0.84
```

中文: 历史预测越多，更新就越像“带惯性的轨迹修正”。

English: With more past predictions, the update behaves more like a trajectory correction with momentum.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers DPMSolverMultistepScheduler** / **Diffusers DPMSolverMultistepScheduler**: 原始 DPM-Solver++ 也滚动保存模型输出。 / The original DPM-Solver++ also keeps a rolling model-output history.
- **FastWAM scheduler** / **FastWAM scheduler**: 连续 flow scheduler 也把训练时间和推理步长分开处理。 / Continuous flow schedulers also separate training time from inference step size.

## 注意事项 / Caveats / when it breaks

- **必须先设 timesteps** / **Timesteps must be set first**: 否则 solver 不知道自己在采样轨迹的哪一步。 / Otherwise the solver does not know where it is in the sampling trajectory.
- **短 schedule 要降阶** / **Short schedules need lower order**: 历史不足时强行三阶会放大数值误差。 / Forcing third order without enough history can amplify numerical error.

## 延伸阅读 / Further reading

- [Wan2.1 flow DPM solver step](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L707-L799)
