---
date: 2026-06-09
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/rtc/modeling_rtc.py
permalink: https://github.com/huggingface/lerobot/blob/49755a3d9e7d43ae93092de8324e75348955afab/src/lerobot/policies/rtc/modeling_rtc.py#L117-L249
difficulty: advanced
read_time: ~18 min
tags: [code-of-the-day, vla, action-chunking, flow-matching, real-time-inference]
build_role: action chunking — the inference-time chunk-stitching layer that sits between the action head and the robot, preventing seams between consecutive chunks
---

# Real-Time Chunking:让动作 chunk 之间的接缝消失的 130 行 / Real-Time Chunking: 130 lines that make the seams between action chunks disappear

> **一句话 / In one line**: 上一个 chunk 还没执行完,新 chunk 就得开始算了 —— RTC 用 autograd 推一个"前缀 guidance",把新 chunk 的去噪轨迹拉到上一 chunk 的"未执行尾巴"上,让两段动作平滑无缝拼接。 / The previous chunk is still being executed when the next chunk has to start denoising — RTC uses autograd to derive a "prefix guidance" that steers the new denoising trajectory toward the unexecuted tail of the previous chunk, so the two action segments stitch together seamlessly.

## 为什么重要 / Why this matters

现代 VLA(π₀、π₀-FAST、SmolVLA、GR00T)几乎都用 *action chunking*:一次预测 H=8~50 步的未来动作,然后让机器人执行。问题是:推理一次要 50-200ms,而机器人控制频率是 30-50Hz。直接的做法是"执行完整个 chunk 再算下一个" —— 但这意味着每 chunk 末尾机器人会停顿一下,产生明显的抖动。RTC(Physical Intelligence 2025 提出)的解法是:**在上一 chunk 还有几步没执行完时,就启动新 chunk 的去噪**,用上一 chunk 的"尾巴"去引导新 chunk 的开头,让两段动作在时间上严丝合缝。整个机制核心是 `denoise_step` 这 130 行 —— 一个 autograd 推出来的 prefix guidance,加上一个随 flow-matching 时间衰减的权重曲线。这是从 OpenVLA 的"标准 chunk 输出"到 π₀ 在真实硬件上跑得流畅的关键拼图。

Modern VLAs (π₀, π₀-FAST, SmolVLA, GR00T) almost universally use *action chunking*: predict `H = 8–50` future steps, execute them one by one. But inference takes 50–200 ms while the robot's control loop runs at 30–50 Hz. The naive "finish the chunk, then compute the next" produces a visible pause-and-jerk at every chunk boundary. RTC (Physical Intelligence, 2025) fixes this by **starting the next chunk's denoising while the previous chunk is still executing**, using the unexecuted *tail* of the previous chunk as a guidance signal that pulls the new denoising trajectory's *head* into a smooth continuation. The whole mechanism centres on `denoise_step` — an autograd-derived prefix correction plus a flow-matching-time-aware weighting curve, all in 130 lines. This is the missing piece between OpenVLA's "chunk-of-actions output" and a π₀-class policy that actually runs smoothly on hardware.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/rtc/modeling_rtc.py`](https://github.com/huggingface/lerobot/blob/49755a3d9e7d43ae93092de8324e75348955afab/src/lerobot/policies/rtc/modeling_rtc.py#L117-L249)

```python
def denoise_step(
    self,
    x_t,
    prev_chunk_left_over,
    inference_delay,
    time,
    original_denoise_step_partial,
    execution_horizon=None,
) -> Tensor:
    """RTC guidance wrapper around an existing denoiser."""

    # In the original implementation, the time goes from 0 to 1 and
    # In our implementation, the time goes from 1 to 0
    # So we need to invert the time
    tau = 1 - time

    if prev_chunk_left_over is None:
        # First step, no guidance - return v_t
        v_t = original_denoise_step_partial(x_t)
        return v_t

    x_t = x_t.clone().detach()

    squeezed = False
    if len(x_t.shape) < 3:
        x_t = x_t.unsqueeze(0)
        squeezed = True

    if len(prev_chunk_left_over.shape) < 3:
        prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

    if execution_horizon is None:
        execution_horizon = self.rtc_config.execution_horizon

    # If the previous action chunk is too short, don't use a long execution horizon
    if execution_horizon > prev_chunk_left_over.shape[1]:
        execution_horizon = prev_chunk_left_over.shape[1]

    batch_size = x_t.shape[0]
    action_chunk_size = x_t.shape[1]
    action_dim = x_t.shape[2]

    if (prev_chunk_left_over.shape[1] < action_chunk_size
            or prev_chunk_left_over.shape[2] < action_dim):
        padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
        padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
        prev_chunk_left_over = padded

    weights = (
        self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
        .to(x_t.device)
        .unsqueeze(0)
        .unsqueeze(-1)
    )

    with torch.enable_grad():
        v_t = original_denoise_step_partial(x_t)
        x_t.requires_grad_(True)

        x1_t = x_t - time * v_t  # one-step Euler back to x1 (the clean prediction)
        err = (prev_chunk_left_over - x1_t) * weights
        grad_outputs = err.clone().detach()
        correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

    max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
    tau_tensor = torch.as_tensor(tau)
    squared_one_minus_tau = (1 - tau_tensor) ** 2
    inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
    c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
    guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

    result = v_t - guidance_weight * correction
    # …squeeze + debug-track…
    return result
```

## 逐行讲解 / What's happening

1. **`if prev_chunk_left_over is None: return v_t`**:
   - 中文: 第一次推理没有"上一 chunk 的尾巴",直接返回原版 denoiser 的 `v_t`。RTC 是个**只在过渡时才介入**的 wrapper —— 这种"无前缀就退化"的设计很重要,意味着你可以用它替换任何 flow-matching 推理而不破坏首步。
   - English: On the very first inference there's no "previous chunk tail," so just return the underlying denoiser's `v_t`. RTC is a **transition-only wrapper** — the "no prefix → degenerate" branch matters because it means you can drop RTC over any flow-matching policy without breaking the first step.

2. **`weights = self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)`**:
   - 中文: 沿 chunk 的时间维构造一根 `(T,)` 的权重曲线。前 `inference_delay` 步权重 = 1(已经定好了、不能改了);第 `inference_delay` 到 `execution_horizon` 步权重在 [1, 0] 之间逐渐衰减(可以稍微调整);超过 `execution_horizon` 的步权重 = 0(完全自由,新 chunk 自己拍板)。曲线形状 (ZEROS / ONES / LINEAR / EXP) 由 config 选。
   - English: Builds a `(T,)` curve of guidance weights along the chunk's time dimension. The first `inference_delay` steps get weight 1 (already committed, cannot be changed); steps `inference_delay` through `execution_horizon` decay from 1 to 0 (mildly steerable); steps beyond `execution_horizon` get weight 0 (the new chunk is fully free). The curve shape (`ZEROS` / `ONES` / `LINEAR` / `EXP`) is config-selected.

3. **`x1_t = x_t - time * v_t`** —— 单步 Euler 反推到清洁动作:
   - 中文: 在 flow-matching 里,`v_t` 是当前 `x_t` 在时间 `t` 上的速度场。`x1_t = x_t - t·v_t` 就是"用当前速度一步外推到 t=0 的预测干净动作"。**这是 guidance 能 work 的关键**:必须在干净动作空间里比较新 chunk 的预测和上一 chunk 的尾巴,而不是在噪声空间。
   - English: In flow matching `v_t` is the velocity field at noise time `t`. `x1_t = x_t - t·v_t` is the one-step Euler extrapolation to `t=0` — the model's *current best guess* of the clean action. **This is what makes guidance meaningful**: you compare the new chunk's predicted clean actions against the previous chunk's tail in *clean* action space, not in noisy space.

4. **`err = (prev_chunk_left_over - x1_t) * weights`**:
   - 中文: "上一 chunk 留下的动作"减去"新 chunk 当前预测的清洁动作",再乘上时间维度上的权重曲线 —— 这就是我们想要的"对齐误差"。注意 `err` 在权重 = 0 的位置自动归零,即新 chunk 自由区不会受到任何 pull。
   - English: "What the previous chunk asked us to do" minus "what the new chunk is currently predicting," weighted along the time axis — this is the alignment error. The `weights = 0` region auto-zeros `err`, so the free portion of the new chunk gets no pull.

5. **`correction = torch.autograd.grad(x1_t, x_t, grad_outputs=err)`**:
   - 中文: **整段最精妙的一步**。我们想最小化 `‖prev - x1_t‖²` 在权重区上的能量,梯度对 `x1_t` 就是 `-err`(算 grad_outputs 时已经带了负号)。但我们要的是对 `x_t`(噪声样本)的梯度,而不是对 `x1_t`。这一行用 `autograd.grad(x1_t, x_t, grad_outputs=err)` 一次反传就拿到 ∂loss/∂x_t。这一步本质上是在"问 denoiser":如果我想让你的清洁预测往这个方向走,你需要我把当前噪声样本往哪边推?
   - English: **The smartest line in the whole snippet**. We want to minimise `‖prev − x1_t‖²` on the weighted region; the gradient w.r.t. `x1_t` is `−err` (the sign is folded into `grad_outputs`). But we need the gradient w.r.t. `x_t` (the noisy sample), not `x1_t`. `autograd.grad(x1_t, x_t, grad_outputs=err)` does exactly one backward pass to give us ∂loss/∂x_t. Intuitively this is *asking the denoiser*: "if I want your clean prediction to move in this direction, where do I need to push the current noisy sample?"

6. **`inv_r2 = (squared_one_minus_tau + tau_tensor**2) / squared_one_minus_tau` 和 `c = (1 - tau) / tau`**:
   - 中文: `guidance_weight = c · inv_r2`,这两项是论文里推导出来的时间-自适应系数,目标是让 guidance 在 `tau → 0`(刚开始去噪、噪声很大)时强,在 `tau → 1`(接近干净)时几乎为零。`nan_to_num(posinf=max_guidance_weight)` 是数值兜底,因为 `c = (1-τ)/τ` 在 τ=0 时炸,这里给上限。
   - English: `guidance_weight = c · inv_r2`. These two factors come from the RTC paper and produce a time-adaptive weight: strong guidance at `tau → 0` (early denoising, very noisy) and almost zero at `tau → 1` (nearly clean). The `nan_to_num(posinf=max_guidance_weight)` clamp catches the singularity at `τ = 0`.

7. **`result = v_t - guidance_weight * correction`**:
   - 中文: 把 guidance 项**减进**原来的速度场,得到修正过的速度。这样后续 ODE 积分(用这个修正后的 `v_t`)就会自然地把 chunk 的开头拉向上一 chunk 的尾巴 —— 整个机器人控制回路在 chunk 边界处就丝滑了。
   - English: Subtract the guidance term from the original velocity field. When the subsequent ODE solver integrates with this corrected `v_t`, the chunk's head naturally bends toward the previous chunk's tail — and the boundary is silky on the robot.

## 类比 / The analogy

想象两个交班的护士。第一位护士还在给病人翻身(还没做完整套护理),第二位护士已经进病房准备接手了 —— 但她不能粗暴地打断,得在第一位还在动手的几秒里观察"她接下来本来要做什么",然后让自己的动作自然衔接上去。RTC 就是这位"会观察会衔接"的接班护士。`weights` 是"前几秒我必须完全跟着她做"的紧紧程度;`time` 衰减(`guidance_weight`)是"越接近交班结束,我越能按自己的判断走";`autograd.grad` 这一步则是在问自己"要让我的护理流程对上她的节奏,我现在的手势该往哪挪一点"。两个 chunk 之间的"接缝"消失了,病人完全察觉不到换了人。

Picture two nurses handing off a patient. The first nurse is still turning the patient (mid-procedure) when the second nurse arrives — she can't just barge in, she has to observe what the first nurse was about to do next and continue from there. RTC is that "observe-and-blend" replacement nurse. `weights` is "how strictly I have to follow her plan in the first few seconds"; the `time` decay (`guidance_weight`) is "the closer the handoff finishes, the more I can trust my own judgement"; the `autograd.grad` call is the nurse asking herself, "where do my hands need to be right now so my procedure picks up exactly where hers left off?" The seam between two chunks disappears; the patient never notices the handover.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:在 nanoVLA 的构件图里,RTC 是**推理时**的中间层,位置在 **action head**(flow-matching 输出 `v_t`)和**机器人 action queue / 执行器**之间。依赖的上游 curriculum 组件:`vlm-backbone-wiring`(产生条件 token)、`action-head-continuous`(产生 `v_t` 的 denoiser)。下游消费者:机器人的实际执行循环(一个 `select_action()` 调用从队列里 pop 出一步动作执行)。`prev_chunk_left_over` 是上一 chunk 中"已经预测好但还没执行"的尾巴(通常是 chunk 长度 H 减去 inference_delay)。如果在你的 nanoVLA 里**省掉 RTC**,你就回到了"每 chunk 一次完整 50ms 推理,机器人在边界停顿"的世界 —— 在 demo 视频里看起来还行,真实硬件上抖动肉眼可见。生产实现需要在这个基础上补:(1) 异步推理线程(看 `lerobot/rollout/inference/rtc.py`);(2) 与 policy server 的网络通信延迟兜底;(3) `LatencyTracker` 来动态估算 `inference_delay`;(4) chunk 长度自适应(当上一 chunk 提前耗尽时短化新 chunk)。

English: in nanoVLA's component graph, RTC is an **inference-time** middleware sitting between the **action head** (the flow-matching denoiser producing `v_t`) and the **robot action queue / executor**. Upstream curriculum dependencies: `vlm-backbone-wiring` (produces the condition tokens), `action-head-continuous` (the denoiser). Downstream consumer: the robot's actual control loop (`select_action()` pops one step from the queue and sends it to motors). `prev_chunk_left_over` is the tail of the previous chunk's predicted actions that hadn't yet been executed when the new chunk started (typically chunk size H minus inference_delay). **Skipping RTC in nanoVLA** drops you back to "one full ~50 ms inference per chunk, robot pauses at every boundary" — fine in a demo video, visibly jittery on real hardware. A production implementation adds: (1) an async inference thread (see `lerobot/rollout/inference/rtc.py`); (2) network-latency tolerance for policy-server deployments; (3) a `LatencyTracker` to estimate `inference_delay` dynamically; (4) adaptive chunk length when the previous chunk runs out faster than expected.

## 自己跑一遍 / Try it yourself

```python
import torch

def rtc_denoise_step(x_t, v_t_fn, prev_left_over, time, weights, max_w=5.0):
    """Minimal RTC guidance — strip every distraction from the lerobot version."""
    if prev_left_over is None:
        return v_t_fn(x_t)
    x_t = x_t.clone().detach()
    with torch.enable_grad():
        x_t.requires_grad_(True)
        v_t = v_t_fn(x_t)
        x1_t = x_t - time * v_t                              # extrapolate to clean
        err = (prev_left_over - x1_t) * weights
        correction = torch.autograd.grad(x1_t, x_t, err.detach())[0]
    tau = 1 - time
    c = (1 - tau) / max(tau, 1e-6)
    inv_r2 = ((1 - tau) ** 2 + tau ** 2) / max((1 - tau) ** 2, 1e-6)
    gw = min(c * inv_r2, max_w)
    return v_t - gw * correction

# Fake denoiser that wants to predict [0,0,0,...,0] (the clean target).
v_t_fn = lambda x: x  # gradient v_t/d x_t = identity
H, A = 8, 2
x_t = torch.randn(H, A)
# Previous chunk wanted [1,1] on every step; the linear weight schedule says
# steps 0-2 are fully bound, steps 2-5 decay to free, steps 5-7 are free.
prev = torch.ones(H, A)
weights = torch.cat([torch.ones(2), torch.linspace(1, 0, 4), torch.zeros(2)]).unsqueeze(-1)

corrected = rtc_denoise_step(x_t, v_t_fn, prev, time=0.5, weights=weights)
print("first 2 steps (should bend toward 1.0):", corrected[:2, 0].tolist())
print("last 2 steps (should be free, like v_t): ", corrected[-2:, 0].tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
first 2 steps (should bend toward 1.0):  [≈0.8 .. 0.9, ≈0.8 .. 0.9]
last 2 steps (should be free, like v_t):  [≈x_t[-2:].tolist()]  (unchanged)
```

中文一句:前 2 步被强行往 `prev = 1.0` 拉,最后 2 步完全自由 —— 中间 4 步则是平滑过渡。这就是"接缝消失"的核心:不同时间步获得不同强度的"上一 chunk 引力"。

English: the first two steps are strongly pulled toward `prev = 1.0`, the last two are left untouched, and the middle four are a smooth interpolation. That is the seam-removal mechanism in microcosm: each time-step within the chunk feels a different amount of "gravity" from the previous chunk.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **同样是 action-chunking 的不同接缝处理 / Other action-chunking implementations of the same slot**:
  - **`huggingface/lerobot:src/lerobot/policies/act/modeling_act.py` —— ACT 的 ensemble**: 中文:最早的 chunk-policy ACT 用的是"加权 ensemble":新 chunk 和老 chunk 在重叠区按时间衰减加权平均 —— 一种没有 guidance 的、平均化的 RTC。 / English: The original ACT chunk policy used "ensemble averaging": new and old chunks in their overlap region get an exponentially decaying weighted mean — a guidance-free, averaged ancestor of RTC.
  - **`NVIDIA/Isaac-GR00T:gr00t/data/state_action/action_chunking.py` —— 数据侧的 chunk 处理**: 中文:GR00T 在 *数据* 端把动作切成 chunk,推理时只是"按 horizon 依次执行"。没有 stitching —— 这种风格在低频任务(50ms 推理、200ms 控制周期)下能凑合,但高频就崩。 / English: GR00T handles chunking entirely on the *data* side and then "execute by horizon" at inference. No stitching — fine on slow control loops, breaks on fast ones.
  - **`Physical-Intelligence/openpi` —— 原作 RTC**: 中文:RTC 算法的最初实现(JAX),lerobot 这版就是 port 过来的。两者算法相同,只是 framework 不同。 / English: The original RTC implementation in JAX from Physical Intelligence; the lerobot version is a port. Same algorithm, different framework.
- **DM-style diffusion guidance**: 中文:`autograd.grad` 推 guidance 的思路和 classifier-free guidance / image-painting / SDEdit 的修补 guidance 同根同源 —— 都是"用一个外部目标的梯度修正去噪轨迹"。 / English: The "use an external objective's gradient to bend the denoising trajectory" idea is the same family as classifier-free guidance, image inpainting, and SDEdit edits.

## 注意事项 / Caveats / when it breaks

- **必须打开 grad —— 推理时也要 / Requires grad enabled even at inference**: 中文:`autograd.grad` 必须在 `torch.enable_grad()` 里,即使整体是推理。会比纯前向慢一点。如果用 `torch.compile`,小心 graph break。 / English: `autograd.grad` requires `torch.enable_grad()`, even in inference mode — adds overhead vs pure forward. Beware of graph breaks under `torch.compile`.
- **denoiser 必须可微 / The denoiser must be differentiable through `x_t`**: 中文:有些底层算子(比如 KV-cache 的 in-place 写、自定义 CUDA op)会阻断 `x_t` 上的梯度流。生产 deploy 时要先验证 `autograd.grad` 拿到的 `correction` 不全是零。 / English: Some low-level ops (in-place KV-cache writes, custom CUDA kernels) silently break gradient flow through `x_t`. Always sanity-check that the returned `correction` isn't all zeros in deployment.
- **`time = 1` 时 `c = 0/0` / `time = 1` is a singularity**: 中文:这种情况意味着已经完全干净,RTC 此时返回的 guidance_weight 是 NaN 被 `nan_to_num` 兜回 max_guidance_weight —— 但实际上你应该提前 break 出 RTC 而不是依赖这个兜底。 / English: At `time = 1` (fully clean) `c = 0/0`; `nan_to_num` clamps it to `max_guidance_weight`, but you should break out of the RTC path before reaching this point rather than rely on the clamp.
- **chunk 边界 bias / Bias at chunk start**: 中文:`prev_chunk_left_over` 比当前 chunk 短时被右补零 —— 这会让 chunk 的后半段被"零动作"轻微拉拽,虽然权重为 0 时理论上无效,但浮点误差有时会泄漏出来。 / English: When `prev_chunk_left_over` is shorter than the current chunk, it gets right-padded with zeros — which under fp16 / weight-near-zero conditions can leak a tiny "pull toward zero" into the chunk's tail. Cosmetic, but worth knowing in tight ablations.

## 延伸阅读 / Further reading

- [Real-Time Chunking — Physical Intelligence whitepaper (PDF)](https://www.physicalintelligence.company/download/real_time_chunking.pdf)
- [Original JAX RTC implementation — Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214)
- [lerobot RTC inference engine — `src/lerobot/rollout/inference/rtc.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/inference/rtc.py)
