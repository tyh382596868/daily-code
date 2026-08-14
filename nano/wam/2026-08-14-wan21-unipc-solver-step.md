---
date: 2026-08-14
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/utils/fm_solvers_unipc.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers_unipc.py#L657-L741
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference]
build_role: sampler-inference advanced variant, UniPC predictor-corrector state machine for flow matching
---

# Wan2.1 UniPC step：采样器要记住前几步 / Wan2.1 UniPC Step: A Sampler Needs Memory of Previous Steps

> **一句话 / In one line**: `step` 先按当前模型输出做 corrector，再滚动保存历史输出和 timestep，最后用 predictor 生成 `prev_sample`。 / `step` first runs a corrector from the current model output, rolls historical outputs and timesteps, then uses a predictor to produce `prev_sample`.

## 为什么重要 / Why this matters

WAM 推理不是“模型 forward 一次就结束”。采样器负责把噪声 latent 一步步变成未来视频 latent，而 UniPC 这类多步 solver 会用前几步的模型输出提高稳定性。这里的核心不是某个公式，而是状态机：什么时候 warmup、什么时候 correct、什么时候滚动历史。

WAM inference is not a single model forward. The sampler turns noisy latents into future video latents over multiple steps, and a multistep solver such as UniPC uses previous model outputs for stability. The key lesson here is the state machine: when to warm up, when to correct, and when to roll history forward.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/utils/fm_solvers_unipc.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers_unipc.py#L657-L741)

```python
def step(self,
         model_output: torch.Tensor,
         timestep: Union[int, torch.Tensor],
         sample: torch.Tensor,
         return_dict: bool = True,
         generator=None) -> Union[SchedulerOutput, Tuple]:
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    use_corrector = (
        self.step_index > 0 and
        self.step_index - 1 not in self.disable_corrector and
        self.last_sample is not None
    )

    model_output_convert = self.convert_model_output(
        model_output, sample=sample)
    if use_corrector:
        sample = self.multistep_uni_c_bh_update(
            this_model_output=model_output_convert,
            last_sample=self.last_sample,
            this_sample=sample,
            order=self.this_order,
        )

    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
        self.timestep_list[i] = self.timestep_list[i + 1]

    self.model_outputs[-1] = model_output_convert
    self.timestep_list[-1] = timestep

    if self.config.lower_order_final:
        this_order = min(self.config.solver_order,
                         len(self.timesteps) -
                         self.step_index)
    else:
        this_order = self.config.solver_order

    self.this_order = min(this_order,
                          self.lower_order_nums + 1)  # warmup for multistep
    assert self.this_order > 0

    self.last_sample = sample
    prev_sample = self.multistep_uni_p_bh_update(
        model_output=model_output,
        sample=sample,
        order=self.this_order,
    )

    if self.lower_order_nums < self.config.solver_order:
        self.lower_order_nums += 1

    self._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return SchedulerOutput(prev_sample=prev_sample)
```

## 逐行讲解 / What's happening

1. **第 683-690 行 / Lines 683-690**: 中文: 没有先 `set_timesteps` 就拒绝采样；第一次 step 会把外部 timestep 对齐到内部 index。 / English: Sampling is rejected until `set_timesteps` is called; the first step aligns the external timestep with the internal index.
2. **第 691-705 行 / Lines 691-705**: 中文: corrector 只在有历史样本、未禁用、且不是第一步时启用。 / English: The corrector is enabled only when history exists, correction is not disabled, and this is not the first step.
3. **第 707-713 行 / Lines 707-713**: 中文: 历史模型输出和 timestep 像队列一样左移，最新值放到末尾。 / English: Historical model outputs and timesteps shift left like a queue, with the newest values appended.
4. **第 714-733 行 / Lines 714-733**: 中文: solver order 会在开头 warmup，在末尾可降阶，避免历史不够还强行高阶。 / English: Solver order warms up at the beginning and can lower near the end, avoiding high-order updates without enough history.

## 类比 / The analogy

像开车过弯：第一秒只能看当前方向盘，后面可以参考前几秒的轨迹修正路线；快到出口时又要保守一点，不要继续做激进预测。

It is like driving through a curve: at the first moment you only know the current steering angle, later you can use recent trajectory history to correct the path, and near the exit you become conservative again.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoWAM 里，这属于 `sampler-inference`。上游是 DiT/flow model 给出的 velocity 或 denoised prediction，下游是 VAE decode 前的未来 latent。最小实现可以先写 Euler step；生产级 WAM 需要这种带历史、可降阶、可 corrector 的 solver，才能在较少采样步下保持视频 rollout 稳定。

In a nanoWAM, this is the `sampler-inference` component. Upstream is a velocity or denoised prediction from the DiT/flow model; downstream is the future latent before VAE decoding. A minimal build can start with Euler, but a production WAM needs a history-aware solver with lower-order fallback and correction to keep rollouts stable with fewer sampling steps.

## 自己跑一遍 / Try it yourself

```python
history = [None, None]
lower_order = 0
for step, out in enumerate([10, 8, 5]):
    history[0], history[1] = history[1], out
    order = min(2, lower_order + 1)
    pred = out if order == 1 else round((history[0] + history[1]) / 2, 1)
    lower_order = min(2, lower_order + 1)
    print(step, history, order, pred)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 [None, 10] 1 10
1 [10, 8] 2 9.0
2 [8, 5] 2 6.5
```

第一步只能一阶，后面有历史后才进入二阶估计。

The first step is first-order; only after history exists can the update become second-order.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers multistep schedulers** / **Diffusers multistep schedulers**: 同样维护模型输出历史，并在 warmup 阶段限制阶数。 / They also keep model-output history and limit order during warmup.
- **Open-Sora RFLOW sampler** / **Open-Sora RFLOW sampler**: 采样器统一管理 timestep、CFG 和 latent 更新。 / The sampler owns timesteps, CFG, and latent updates in one place.

## 注意事项 / Caveats / when it breaks

- **必须先设时间表** / **Timesteps must be set first**: `num_inference_steps is None` 会直接报错。 / `num_inference_steps is None` raises immediately.
- **历史状态不能跨 episode 污染** / **History must not leak across episodes**: 新视频 rollout 前要重置 `last_sample`、`model_outputs`、`step_index`。 / Before a new video rollout, reset `last_sample`, `model_outputs`, and `step_index`.

## 延伸阅读 / Further reading

- [Wan-Video/Wan2.1 source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers_unipc.py#L657-L741)
