---
date: 2026-08-20
topic: robotics
source: trending
repo: NVIDIA/flashdreams
file: flashdreams/flashdreams/infra/diffusion/scheduler/fm_euler.py
permalink: https://github.com/NVIDIA/flashdreams/blob/abb468e500bb814bdb0cbcb790589b03c6ddc5b6/flashdreams/flashdreams/infra/diffusion/scheduler/fm_euler.py#L45-L222
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, world-model-serving, flow-matching]
---

# FlashDreams Euler scheduler：时间表要固定，buffer 要保真 / FlashDreams Euler Scheduler: Pin the Schedule, Preserve the Buffers

> **一句话 / In one line**: FlashDreams 的 flow-matching Euler scheduler 支持固定 distilled timestep 表，并重写 `_apply` 防止 `.to(bf16)` 破坏 fp32 sigma buffer。 / FlashDreams' flow-matching Euler scheduler supports fixed distilled timesteps and overrides `_apply` so `.to(bf16)` does not corrupt fp32 sigma buffers.

## 为什么重要 / Why this matters

交互式 world model serving 常用少步蒸馏模型，时间表不是随便线性切 4 段就行。FlashDreams 把固定 timestep 作为配置入口，同时保护 `timesteps` 和 `sigmas` 始终保持 fp32，避免模型整体转 bf16 时把采样轨迹悄悄改掉。

Interactive world-model serving often uses few-step distilled models, where the timestep schedule is not just an arbitrary 4-way linspace. FlashDreams exposes fixed timesteps in config and keeps `timesteps` and `sigmas` in fp32 even when the module is cast to bf16, preventing silent schedule drift.

## 代码 / The code

`NVIDIA/flashdreams` — [`flashdreams/flashdreams/infra/diffusion/scheduler/fm_euler.py`](https://github.com/NVIDIA/flashdreams/blob/abb468e500bb814bdb0cbcb790589b03c6ddc5b6/flashdreams/flashdreams/infra/diffusion/scheduler/fm_euler.py#L45-L222)

```python
@dataclass(kw_only=True)
class FlowMatchEulerDiscreteSchedulerConfig(SchedulerConfig):
    _target: type["FlowMatchEulerDiscreteScheduler"] = field(
        default_factory=lambda: FlowMatchEulerDiscreteScheduler
    )

    num_inference_steps: int = 4
    shift: float = 5.0
    num_train_timesteps: int = 1000
    fixed_timesteps: tuple[float, ...] | None = None
    enable_tqdm: bool = False


class FlowMatchEulerDiscreteScheduler(Scheduler):
    timesteps: Tensor
    sigmas: Tensor

    def __init__(self, config: FlowMatchEulerDiscreteSchedulerConfig) -> None:
        super().__init__(config)
        self.config: FlowMatchEulerDiscreteSchedulerConfig = config

        N = config.num_inference_steps
        N_train = config.num_train_timesteps
        assert N > 0, f"num_inference_steps must be > 0 (got {N})"

        if config.fixed_timesteps is not None:
            # Distilled / hand-tuned schedule path. Caller supplies an
            # N+1-entry timestep list; sigmas follow trivially as
            # timesteps/N_train. The trailing entry is typically 0.0 so
            # the last Euler step reaches the clean latent.
            ft = config.fixed_timesteps
            assert len(ft) == N + 1, (
                f"fixed_timesteps length {len(ft)} must equal "
                f"num_inference_steps + 1 = {N + 1}"
            )
            timesteps_np = np.asarray(ft, dtype=np.float32)
            sigmas_np = (timesteps_np / N_train).astype(np.float32)
        else:
            train_alphas = np.linspace(1.0, 1.0 / N_train, N_train)[::-1].copy()
            train_sigmas = 1.0 - train_alphas
            sigma_min, sigma_max = float(train_sigmas[-1]), float(train_sigmas[0])
            inf_sigmas = np.linspace(sigma_max, sigma_min, N + 1)[:-1]
            inf_sigmas = (
                config.shift * inf_sigmas / (1.0 + (config.shift - 1.0) * inf_sigmas)
            )
            sigmas_np = np.concatenate([inf_sigmas, [0.0]]).astype(np.float32)
            timesteps_np = (sigmas_np * N_train).astype(np.float32)

        self.register_buffer(
            "timesteps",
            torch.from_numpy(timesteps_np),
            persistent=False,
        )
        self.register_buffer(
            "sigmas",
            torch.from_numpy(sigmas_np),
            persistent=False,
        )

    _FP32_BUFFERS = ("timesteps", "sigmas")

    def _apply(self, fn, recurse=True):
        """Move buffers with the parent ``.to(...)`` but keep them fp32."""
        saved = {name: getattr(self, name) for name in self._FP32_BUFFERS}
        super()._apply(fn, recurse=recurse)
        for name, original in saved.items():
            target_device = getattr(self, name).device
            setattr(self, name, original.to(device=target_device))
        return self

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        input_dtype = initial_noise.dtype
        N = self.config.num_inference_steps

        noisy = initial_noise
        for i in tqdm(
            range(N),
            disable=not self.config.enable_tqdm,
            desc="FlowMatchEulerDiscreteScheduler",
        ):
            timestep = self.timesteps[i].to(dtype=input_dtype)
            dt = (self.sigmas[i + 1] - self.sigmas[i]).to(dtype=input_dtype)

            flow = predict_flow(noisy, timestep)
            noisy = noisy + dt * flow

        return noisy.to(input_dtype)
```

## 逐行讲解 / What's happening

1. **第 60-83 行 / Lines 60-83 (config knobs)**:
   - 中文: 默认是 4 步、shift 5.0，也允许传入 `fixed_timesteps` 走蒸馏模型的手写时间表。
   - English: The default is 4 steps with shift 5.0, but `fixed_timesteps` allows a hand-tuned distilled schedule.
2. **第 127-153 行 / Lines 127-153 (schedule construction)**:
   - 中文: 固定路径直接用 timestep 除以训练步数得到 sigma；普通路径按 flow matching 的 shifted sigma 表生成。
   - English: The fixed path converts timesteps to sigmas by division; the normal path builds a shifted flow-matching sigma grid.
3. **第 155-168 行 / Lines 155-168 (buffers)**:
   - 中文: `timesteps` 和 `sigmas` 注册成 non-persistent buffer，跟着 device 走但不进 checkpoint。
   - English: `timesteps` and `sigmas` are non-persistent buffers, so they move with device but do not enter checkpoints.
4. **第 170-182 行 / Lines 170-182 (`_apply`)**:
   - 中文: 先保存 fp32 原件，再让父类 `.to(...)` 执行，最后只移动 device、不改变 dtype。
   - English: It snapshots fp32 originals, lets parent `.to(...)` run, then restores the buffers with device movement only.
5. **第 184-222 行 / Lines 184-222 (`sample`)**:
   - 中文: 每步调用 `predict_flow`，按 `noisy = noisy + (sigma_next - sigma) * flow` 显式 Euler 更新。
   - English: Each step calls `predict_flow` and applies the explicit Euler update `noisy = noisy + (sigma_next - sigma) * flow`.

## 类比 / The analogy

像烤箱菜谱。温度曲线必须按菜谱走，不能因为你把整个厨房切到“节能模式”就把温度表四舍五入；否则同样的食材会烤出不同结果。

It is like an oven recipe. The temperature curve must follow the recipe; switching the whole kitchen into a lower-precision mode should not round the temperature table and change the dish.

## 自己跑一遍 / Try it yourself

```python
timesteps = [1000.0, 960.0, 888.8889, 727.2728, 0.0]
sigmas = [t / 1000.0 for t in timesteps]
noisy = 1.0

def predict_flow(x, timestep):
    return 0.5

for i in range(4):
    dt = sigmas[i + 1] - sigmas[i]
    noisy = noisy + dt * predict_flow(noisy, timesteps[i])
print(round(noisy, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.5
```

中文: 因为 sigma 从 1 走到 0，常数 flow 0.5 会把样本沿负方向推进 0.5。

English: Because sigma moves from 1 to 0, a constant flow of 0.5 moves the sample by -0.5.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers FlowMatchEulerDiscreteScheduler** / **Diffusers FlowMatchEulerDiscreteScheduler**: 同样在 sigma 空间做 flow matching Euler 更新。 / It performs the same kind of flow-matching Euler update in sigma space.
- **Wan few-step inference** / **Wan few-step inference**: 蒸馏视频模型常固定少步 timestep，而不是每次重新推导。 / Distilled video models often pin a few-step timestep list instead of deriving it each run.

## 注意事项 / Caveats / when it breaks

- **`fixed_timesteps` 长度必须是 N+1** / **`fixed_timesteps` must be N+1**: 少了终点就不知道最后一步该落到哪个 sigma。 / Without the terminal entry, the final step has no target sigma.
- **fp32 buffer 不是性能瓶颈** / **fp32 buffers are not the bottleneck**: 时间表很小，保留 fp32 比追求 bf16 节省更重要。 / The schedule is tiny; preserving fp32 is more important than saving a negligible amount of memory.

## 延伸阅读 / Further reading

- [FlashDreams Euler scheduler](https://github.com/NVIDIA/flashdreams/blob/abb468e500bb814bdb0cbcb790589b03c6ddc5b6/flashdreams/flashdreams/infra/diffusion/scheduler/fm_euler.py#L45-L222)
- [NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams)
