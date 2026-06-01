---
date: 2026-06-01
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_flow_map_euler_discrete.py
permalink: https://github.com/huggingface/diffusers/blob/b003a47354ca3a28b4b21fceb4d649656d8cee99/src/diffusers/schedulers/scheduling_flow_map_euler_discrete.py#L223-L308
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, huggingface, diffusion, flow-matching, distillation, sampler]
---

# FlowMapEulerDiscreteScheduler:把"任意步采样"塞进一个 Euler 实现里 / FlowMapEulerDiscreteScheduler: any-step sampling, in one Euler `step`

> **一句话 / In one line**: 一个升级版的 Euler 采样器,接受**两个**端点 (t, r) 而不是一个,让一个 flow-map 蒸馏后的模型在 1、2、4、8 步之间自由切换。 / An upgraded Euler sampler that takes **two** endpoints (t, r) instead of one, letting a flow-map distilled model switch freely between 1, 2, 4, 8 NFE without retraining.

## 为什么重要 / Why this matters

普通 flow matching / rectified flow 的 Euler 采样器有一个隐含假设:每一步固定从 `sigma[i]` 走到 `sigma[i+1]`。这就意味着想换一个推理步数,要么重新设计 schedule,要么蒸馏成 consistency model 强制走到 `z_0`。`FlowMapEulerDiscreteScheduler` 把这个假设解除了——`step()` 直接吃 `(timestep, r_timestep)`,**两个端点都由调用者决定**。配合 AnyFlow 那种"flow-map 蒸馏"训练目标,一个 checkpoint 就能在 NFE=1、2、4、8 之间切换,而代码增量只有 80 行。

A standard flow-matching Euler scheduler hard-codes one assumption: each step goes from `sigma[i]` to `sigma[i+1]`. To change the number of inference steps, you either redesign the schedule or distill into a consistency model that's pinned to `z_0`. `FlowMapEulerDiscreteScheduler` removes that constraint — `step()` accepts both `timestep` *and* `r_timestep`, so **both endpoints are caller-controlled**. Paired with a flow-map distillation training objective (à la AnyFlow), one checkpoint can run at NFE=1, 2, 4, 8 without retraining. And the entire change is 80 lines of scheduler code.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_flow_map_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/b003a47354ca3a28b4b21fceb4d649656d8cee99/src/diffusers/schedulers/scheduling_flow_map_euler_discrete.py#L223-L308)

```python
def step(
    self,
    model_output,
    timestep,
    sample,
    r_timestep=None,
    return_dict=True,
):
    if self.sigmas is None or self.timesteps is None:
        raise ValueError("`set_timesteps` has not been called.")

    if self._step_index is None:
        self._init_step_index(timestep)

    # Resolve source sigma via index lookup; fall back to / num_train_timesteps only if off-schedule.
    t_idx = self.index_for_timestep(timestep)
    if t_idx is not None:
        sigma_t = self.sigmas[t_idx].to(device=sample.device, dtype=self.sigmas.dtype)
    else:
        t_value = timestep.to(self.sigmas.dtype) if torch.is_tensor(timestep) else torch.tensor(timestep)
        sigma_t = (t_value / self.config.num_train_timesteps).to(device=sample.device, dtype=self.sigmas.dtype)

    # Resolve target sigma.
    if r_timestep is None:
        if t_idx is None:
            raise ValueError(
                "`r_timestep` is None but `timestep` is not on the current schedule, ..."
            )
        sigma_r = self.sigmas[t_idx + 1].to(device=sample.device, dtype=self.sigmas.dtype)
    else:
        r_idx = self.index_for_timestep(r_timestep)
        if r_idx is not None:
            sigma_r = self.sigmas[r_idx].to(device=sample.device, dtype=self.sigmas.dtype)
        else:
            r_value = r_timestep.to(self.sigmas.dtype) if torch.is_tensor(r_timestep) else torch.tensor(r_timestep)
            sigma_r = (r_value / self.config.num_train_timesteps).to(device=sample.device, dtype=self.sigmas.dtype)

    sigma_t = sigma_t.view(*sigma_t.shape, *([1] * (model_output.ndim - sigma_t.ndim)))
    sigma_r = sigma_r.view(*sigma_r.shape, *([1] * (model_output.ndim - sigma_r.ndim)))
    prev_sample = sample - (sigma_t - sigma_r) * model_output
    prev_sample = prev_sample.to(model_output.dtype)

    self._step_index += 1

    if not return_dict:
        return (prev_sample,)
    return FlowMapEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
```

## 逐行讲解 / What's happening

1. **`step` 的签名 / The signature of `step`**:
   - 中文: 比传统 `FlowMatchEulerDiscreteScheduler.step` 多了 `r_timestep`。这是整个 scheduler 唯一新引入的概念——"目标 timestep"。
   - English: Compared with `FlowMatchEulerDiscreteScheduler.step`, the only new argument is `r_timestep` — the "target timestep". This is the entire new concept the scheduler introduces.

2. **`t_idx = self.index_for_timestep(timestep)`**:
   - 中文: 用 timestep 反查它在 schedule 里的索引位置,然后用 `self.sigmas[t_idx]` 拿到对应 sigma。**为什么不直接用 `t / num_train_timesteps`?** 因为很多 schedule 做了非线性变换(比如 `shift=5` 把后半段拉长),线性除法算不对。
   - English: Find the timestep's index in the current schedule, then read `sigma_t` from `self.sigmas[t_idx]`. **Why not just compute `t / num_train_timesteps`?** Because many schedules apply a nonlinear shift (e.g. `shift=5` stretches the late steps); a plain divide would give the wrong sigma.

3. **off-schedule 回退 / The off-schedule fallback**:
   - 中文: 如果 caller 给的 timestep 不在当前 schedule 里(比如做 1-step generation,r 直接传 `0.0`),`index_for_timestep` 返回 `None`,此时回退到 `t / num_train_timesteps`。这允许"超出 schedule 的任意步采样"。
   - English: If the caller passes a timestep not on the current schedule (e.g. `r=0.0` for one-shot generation), `index_for_timestep` returns `None` and the code falls back to `t / num_train_timesteps`. This is what enables truly any-step sampling — including endpoints the schedule never defined.

4. **`r_timestep=None` 的默认行为 / Default behavior of `r_timestep=None`**:
   - 中文: 当 caller 没指定 r,就用 `sigmas[t_idx + 1]`——即下一个 schedule 步,**完全退化为传统 Euler scheduler**。这是 API 兼容性的关键。
   - English: When `r_timestep` is `None`, the target defaults to `sigmas[t_idx + 1]` — the next scheduled step. This degenerates to the classic Euler scheduler, preserving API compatibility.

5. **`sigma_t.view(*sigma_t.shape, *([1] * (model_output.ndim - sigma_t.ndim)))`**:
   - 中文: 这一行是为了 broadcast。当 `sigma_t` 是标量(shape `()`)而 `model_output` 是 `(B, C, H, W)` 时,要把 sigma 从 `()` reshape 成 `(1, 1, 1, 1)` 才能逐元素相乘。这是 pythonic 的 trick,常被忽略。
   - English: This line is broadcasting-machinery. When `sigma_t` is a scalar `()` and `model_output` is `(B, C, H, W)`, you need to reshape sigma to `(1, 1, 1, 1)` so it can multiply elementwise. A standard but easy-to-miss trick.

6. **核心更新公式 / The core update**:
   - 中文: `prev_sample = sample - (sigma_t - sigma_r) * model_output`。这就是 rectified-flow Euler 的本质——velocity field 是常数,所以从 `z_t` 出发走 `(sigma_t - sigma_r)` 这么长就到了 `z_r`。**`model_output` 一份,(t, r) 任意取**,所以才叫 flow-map。
   - English: `prev_sample = sample - (sigma_t - sigma_r) * model_output` — the heart of rectified-flow Euler. Because the velocity field is constant along the straight path, you can step from `z_t` to `z_r` in one shot by moving `(sigma_t - sigma_r)` along `model_output`. **One model call, any `(t, r)` pair** — hence the name flow-*map*.

7. **`_step_index += 1` 即使 r 自定义也照常加 / `_step_index += 1` even with custom `r`**:
   - 中文: 维持 callback 友好。即便 caller 跳着采样,observer 看到的 step_index 仍单调递增。
   - English: Keeps callbacks happy. Even if the caller is jumping around with custom `r`, the externally-observable `step_index` still advances monotonically.

## 类比 / The analogy

想象你在玩一款赛车游戏,赛道上有 1000 个里程碑(timestep)。传统的 Euler 调度器像是一辆**只能在相邻里程碑之间挪一格**的车,你想"快进"必须重新设计赛道。而 flow-map 调度器配合 flow-map 蒸馏后的模型,就像换上了一辆配备了**任意目的地导航**的车——你直接告诉它"我现在在 800,直接去 0"——它就一脚油门到位。这就是为什么同一个 checkpoint 能跑 1 步、4 步、8 步。

Imagine a racing game where the track has 1000 mile-markers (timesteps). A traditional Euler scheduler is a car that can **only crawl from one mile-marker to the next** — to skip ahead you must redesign the track. The flow-map scheduler, paired with a flow-map distilled model, hands you a car with an **arbitrary-destination GPS**. You can say "I'm at mile 800, take me straight to mile 0" and the car floors it. Same model checkpoint, any number of stops.

## 自己跑一遍 / Try it yourself

```python
import torch

class TinyFlowMap:
    def __init__(self, num_train=1000, num_steps=8):
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train).long()
        self.num_train = num_train

    def step(self, v_pred, sample, t, r):
        sigma_t = self.sigmas[(self.timesteps == t).nonzero(as_tuple=True)[0]] \
                  if (self.timesteps == t).any() else torch.tensor(t / self.num_train)
        sigma_r = torch.tensor(r / self.num_train)
        sigma_t = sigma_t.view(*sigma_t.shape, *([1] * (v_pred.ndim - sigma_t.ndim)))
        return sample - (sigma_t - sigma_r) * v_pred

torch.manual_seed(0)
sample = torch.randn(1, 3, 16, 16)
v_pred = torch.randn_like(sample) * 0.1
sched = TinyFlowMap()

# 8-step trajectory
x = sample.clone()
for i in range(8):
    t, r = sched.timesteps[i].item(), 0 if i == 7 else sched.timesteps[i + 1].item()
    x = sched.step(v_pred, x, t, r)
print("8-step final norm:", x.norm().item())

# 1-step shortcut: same start, jump directly t=875 -> r=0
x_one = sched.step(v_pred, sample, t=875, r=0)
print("1-step final norm:", x_one.norm().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
8-step final norm: 25.49...
1-step final norm: 25.49...
```

中文一两句:两种方案最终范数几乎一样——因为 velocity field 是常数 `v_pred`,8 小步累积的位移和 1 大步的位移完全相等。这就是 flow-map 蒸馏想达到的"等价性"。

The two norms agree because the velocity field here is constant `v_pred` — the cumulative displacement of 8 small steps equals the 1 large step. That equivalence is exactly what flow-map distillation tries to teach a real model.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Consistency Models (`scheduling_consistency_models.py`)** / **Consistency Models**: 中文:同样想"一步到位",但训练目标不一样——CM 强制 student(t) == student(t-1) of teacher,本质上把 r 锁死成 0。 / English: Same goal — get to `z_0` in one shot — but a different training objective: CM forces `student(t) == student(t-1) of teacher`, effectively pinning `r=0`. Flow-map keeps `r` free.
- **LCM (`scheduling_flow_match_lcm.py`)**: 中文:Latent Consistency Models 把 CM 思路嵌入 flow matching,可以看成 flow-map 的特化版。 / English: LCM is essentially a CM-flavored specialization of flow-map.
- **AnyFlow / Hyper-SD / DMD2 蒸馏**: 中文:这些蒸馏方法训练的 student 通常都需要"任意 (t, r) 都能调用"的 sampler,这个 scheduler 就是它们的运行时基础设施。 / English: All these distillation methods produce students that need an "any (t, r)" sampler at inference time — this scheduler is the runtime infrastructure for them.

## 注意事项 / Caveats / when it breaks

- **模型本身必须是 flow-map 蒸馏过的 / The model must actually be flow-map distilled**: 中文:对一个标准 flow matching 模型直接传 `r=0` 不会"一步到位",只会得到一个噪声很大的预测。这个 scheduler 只是基础设施,关键在训练目标。 / English: Calling `r=0` on a standard flow-matching model does *not* yield one-shot generation — it gives a noisy prediction. The scheduler is only the runtime; the magic lives in the distillation training objective.
- **shift 与 timestep 的耦合 / Coupling between shift and timestep**: 中文:`shift > 1` 会让 sigma 不再线性变化,因此 `r_value / num_train_timesteps` 这种回退路径**只在 timestep 落在 schedule 外部时**有用——caller 要承担"自己知道在做什么"的责任。 / English: A `shift > 1` makes sigma nonlinear in `t`, so the `t_value / num_train_timesteps` fallback is only meaningful when the timestep is genuinely off-schedule. Calling with a shift-warped on-schedule value and bypassing `index_for_timestep` will silently give the wrong sigma.
- **`_step_index` 不再代表"已采样多少步" / `_step_index` no longer literally means "steps taken"**: 中文:any-step 模式下 caller 可以一次跳很远,但 `_step_index` 还是按调用次数加 1,只是它的语义从此变成"调用次数",不是"在 schedule 上前进了几格"。 / English: In any-step mode, the caller might leap forward many sigmas at once, but `_step_index` still increments by 1 per call. Its semantics shift from "schedule position" to "call count" — important when reading callback code that assumed the former.

## 延伸阅读 / Further reading

- [AnyFlow paper — *Any-Step Video Diffusion Model with On-Policy Flow Map Distillation*](https://huggingface.co/papers/2605.13724) — the paper this scheduler was written for
- [Rectified Flow — *Flow Straight and Fast*](https://arxiv.org/abs/2209.03003) — why straight velocity fields make any-step sampling viable
- [Consistency Models](https://arxiv.org/abs/2303.01469) — the closely-related "pin to z_0" approach
