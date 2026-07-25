---
date: 2026-07-25
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/utils/fm_solvers.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L227-L293
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, noise-scheduler]
build_role: noise-scheduler advanced variant, inference-time sigma schedule
---

# Wan2.1 Flow Scheduler：采样前先重写 sigma 时间表 / Wan2.1 Flow Scheduler: Rewrite the Sigma Schedule Before Sampling

> **一句话 / In one line**: flow-matching 采样不是只数 step, 还要把 sigma、timestep、solver history 和 step index 一起重置。 / Flow-matching sampling is not just counting steps, it resets sigmas, timesteps, solver history, and step indices together.

## 为什么重要 / Why this matters

视频生成的采样器要在推理前决定一条噪声时间轴。`set_timesteps` 支持外部给定 `sigmas`, 也能按步数线性生成, 再通过 static 或 dynamic shift 改形。最后它重置 multistep solver 的历史, 避免上一段采样污染下一段。

A video sampler must choose a noise timeline before inference. `set_timesteps` accepts custom `sigmas` or builds them from the step count, reshapes them with static or dynamic shifting, and resets multistep solver history so one sampling run cannot leak into the next.

## 代码 / The code

`Wan-Video/Wan2.1` -- [`wan/utils/fm_solvers.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L227-L293)

```python
    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                " you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min,
                                 num_inference_steps +
                                 1).copy()[:-1]  # pyright: ignore

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)  # pyright: ignore
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 +
                                       (shift - 1) * sigmas)  # pyright: ignore

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) /
                          self.alphas_cumprod[0])**0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]
                                ]).astype(np.float32)  # pyright: ignore

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        self._step_index = None
        self._begin_index = None
        # self.sigmas = self.sigmas.to(
        #     "cpu")  # to avoid too much CPU/GPU communication

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
```

## 逐行讲解 / What's happening

1. **第 245-248 行 / Lines 245-248 (`dynamic shift guard`)**:
   - 中文: 启用动态 shifting 时必须传 `mu`, 否则时间轴没有分辨率条件。
   - English: Dynamic shifting requires `mu`; without it the schedule lacks its resolution-dependent control.
2. **第 250-253 行 / Lines 250-253 (`生成 sigma`)**:
   - 中文: 没有外部 sigmas 时, 从 `sigma_max` 到 `sigma_min` 线性取 `num_inference_steps` 个点。
   - English: Without custom sigmas, it linearly samples points from `sigma_max` to `sigma_min`.
3. **第 255-261 行 / Lines 255-261 (`shift 时间表`)**:
   - 中文: 动态或静态 shift 都是在改变 sigma 曲线, 不是改变模型结构。
   - English: Dynamic and static shifts reshape the sigma curve, not the model.
4. **第 263-279 行 / Lines 263-279 (`终点和 timestep`)**:
   - 中文: 最后追加 `sigma_last`, 并把 sigmas 乘训练步数得到离散 timestep。
   - English: It appends the final sigma and converts sigmas into discrete timesteps by scaling with training steps.
5. **第 281-289 行 / Lines 281-289 (`清 solver 状态`)**:
   - 中文: 步数、历史 model outputs、低阶计数和索引都归零。
   - English: It resets inference length, stored model outputs, lower-order counters, and indices.

## 类比 / The analogy

这像给长途拍摄排一张航拍时间表。每个航点的高度先排好, 再按天气修正, 起飞前还要清掉上一趟飞行的日志状态。

It is like planning a drone-shot schedule. You choose altitudes for each waypoint, adjust them for conditions, and clear the previous flight log before takeoff.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

在 nanoWAM 中, 这是 `NoiseScheduler` 或 `Sampler` 的推理入口。上游是目标帧数、分辨率和采样步数, 下游是每一步 denoiser/DiT forward。省掉这个组件, 训练时间轴和推理时间轴会混乱, multistep solver 也会错误复用旧预测。生产级实现还要支持不同分辨率的 `mu` 策略、Karras/exponential sigma 和断点续采样。

In a nanoWAM, this is the inference entry of the `NoiseScheduler` or `Sampler`. Upstream provides frame count, resolution, and step count; downstream is each denoiser or DiT forward. Without it, training and inference timelines drift, and a multistep solver may reuse stale predictions. A production implementation adds resolution-dependent `mu`, Karras or exponential sigmas, and resume support.

## 自己跑一遍 / Try it yourself

```python
def shifted_sigmas(steps, sigma_max=1.0, sigma_min=0.0, shift=3.0):
    raw = [sigma_max + (sigma_min - sigma_max) * i / steps for i in range(steps)]
    sigmas = [shift * s / (1 + (shift - 1) * s) for s in raw]
    timesteps = [int(s * 1000) for s in sigmas]
    return sigmas + [0.0], timesteps

sigmas, timesteps = shifted_sigmas(4)
print([round(x, 3) for x in sigmas])
print(timesteps)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[1.0, 0.9, 0.75, 0.5, 0.0]
[1000, 900, 750, 500]
```

`shift > 1` 会把中间 sigma 往高噪声端推, 让少步采样更重视早期粗结构。

A `shift > 1` pushes middle sigmas toward higher noise, making low-step sampling spend more effort on early coarse structure.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers FlowMatchEuler**: 同样把 sigmas 当作推理主时间轴。 / It also treats sigmas as the primary inference timeline.
- **Wan2.1 denoise loop**: 采样循环逐个消费这里生成的 timesteps。 / Wan2.1 denoising loops consume the timesteps generated here.

## 注意事项 / Caveats / when it breaks

- **自定义 sigmas 要单调**: 乱序 sigma 会破坏求解器假设。 / Custom sigmas must be monotonic enough for solver assumptions.
- **状态必须重置**: 多次调用 sampler 时, `model_outputs` 不能跨 run 复用。 / Sampler runs must not share stale `model_outputs`.

## 延伸阅读 / Further reading

- Wan2.1 repository: https://github.com/Wan-Video/Wan2.1
- DPM-Solver paper: https://huggingface.co/papers/2206.00927
