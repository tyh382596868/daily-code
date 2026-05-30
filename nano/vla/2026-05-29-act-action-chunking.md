---
date: 2026-05-29
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/act/modeling_act.py
permalink: https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/policies/act/modeling_act.py#L101-L165
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking, temporal-ensemble, inference]
build_role: Action chunking — predict a horizon of actions at once, then unroll them with a queue or temporal ensemble
---

# Action chunking:一次预测一串,用队列吐给环境 / Action chunking: predict a chunk, drip-feed it to the env via a queue

> **一句话 / In one line**: ACT 一次前向预测未来 `chunk_size` 步动作,但不是一次全执行 —— 用一个 `deque` 把它们排队,环境每要一个动作就 `popleft` 一个,队列空了才重新前向,既减少推理频率又保证动作连贯。 / ACT predicts `chunk_size` future actions in one forward, but doesn't execute them all at once — it queues them in a `deque`, pops one per environment step, and only re-runs the forward when the queue empties. Fewer inferences, smoother motion.

## 为什么重要 / Why this matters

机器人控制频率通常 30-50Hz,但 VLA 一次前向要几十到几百毫秒,做不到每步都推理。Action chunking 是这个矛盾的标准解:**一次预测未来一整段动作**(比如 100 步),然后慢慢执行。它还顺带解决了"行为不连贯"问题 —— 单步策略容易在相邻时刻抖动,chunk 预测天然平滑。这段代码展示两种 chunk 消费策略:(1) **队列**(`deque`,简单,执行 `n_action_steps` 步后重新预测);(2) **temporal ensemble**(每步都重新预测,但把多个 overlapping chunk 的预测加权平均,最平滑但最贵)。这是几乎所有现代 VLA(ACT、π₀、GR00T、OpenVLA-OFT)都用的件。

Robots run at 30-50 Hz, but a VLA forward takes tens to hundreds of milliseconds — you can't infer every step. Action chunking is the standard fix: **predict a whole horizon of future actions** (say 100 steps) and execute them gradually. It also fixes jittery behaviour — single-step policies wobble between adjacent timesteps, while chunk predictions are inherently smooth. This code shows two chunk-consumption strategies: (1) a **queue** (`deque`, simple, re-predict after `n_action_steps`), and (2) **temporal ensemble** (re-predict every step but weight-average overlapping chunks — smoothest but priciest). Nearly every modern VLA (ACT, π₀, GR00T, OpenVLA-OFT) ships this.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/act/modeling_act.py`](https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/policies/act/modeling_act.py#L101-L165)

```python
@torch.no_grad()
def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    """Select a single action given environment observations.

    Works by managing the actions in a queue and only calling the model when the
    queue is empty.
    """
    self.eval()

    if self.config.temporal_ensemble_coeff is not None:
        actions = self.predict_action_chunk(batch)
        action = self.temporal_ensembler.update(actions)
        return action

    # Action queue logic for n_action_steps > 1. When the queue is depleted,
    # populate it by querying the policy.
    if len(self._action_queue) == 0:
        actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

        # model.forward returns (batch, n_action_steps, action_dim), but the queue
        # effectively has shape (n_action_steps, batch, *), hence the transpose.
        self._action_queue.extend(actions.transpose(0, 1))
    return self._action_queue.popleft()

@torch.no_grad()
def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
    """Predict a chunk of actions given environment observations."""
    self.eval()
    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
    actions = self.model(batch)[0]
    return actions
```

And the queue is created in `reset()`:

```python
self._action_queue = deque([], maxlen=self.config.n_action_steps)
```

## 逐行讲解 / What's happening

1. **`deque([], maxlen=n_action_steps)` / The action queue**:
   - 中文:一个固定容量的双端队列。`chunk_size` 是模型一次预测多少步(比如 100),`n_action_steps` 是实际从这串里取出来执行多少步(比如 50)再重新预测。`n_action_steps < chunk_size` 意味着预测有重叠余量,提升鲁棒性。
   - English: a fixed-capacity double-ended queue. `chunk_size` is how many steps the model predicts at once (say 100); `n_action_steps` is how many of those get executed (say 50) before re-predicting. `n_action_steps < chunk_size` leaves predictive headroom, improving robustness.

2. **`if len(self._action_queue) == 0:` / Lazy re-prediction**:
   - 中文:只有队列空了才重新跑模型。这就是"减少推理频率"的核心 —— 50 步只推理一次,推理频率降到控制频率的 1/50。
   - English: the model only re-runs when the queue empties. This is the whole "fewer inferences" win — one forward per 50 steps drops inference frequency to 1/50 of control frequency.

3. **`predict_action_chunk(batch)[:, :n_action_steps]` / Slice the usable prefix**:
   - 中文:模型预测 `chunk_size` 步,但只取前 `n_action_steps` 步放进队列,后面的丢掉(留作余量,下次重新预测会覆盖)。
   - English: the model predicts `chunk_size` steps but only the first `n_action_steps` go in the queue; the rest are discarded (headroom, overwritten by the next prediction).

4. **`actions.transpose(0, 1)` / Batch-time transpose**:
   - 中文:模型输出 `[batch, n_action_steps, action_dim]`,但队列要按时间步逐个 pop,所以转成 `[n_action_steps, batch, action_dim]` 再 `extend` —— 队列里每个元素是"某一时间步、全 batch 的动作"。
   - English: the model outputs `[batch, n_action_steps, action_dim]`, but the queue pops per timestep, so transpose to `[n_action_steps, batch, action_dim]` before `extend` — each queue element is "one timestep's action across the whole batch".

5. **`popleft()` / One action per env step**:
   - 中文:环境每要一个动作,从队头取一个。队列像一个"动作缓冲区",把"低频预测"和"高频执行"解耦。
   - English: the env gets one action per call from the queue head. The queue is an action buffer decoupling low-frequency prediction from high-frequency execution.

6. **`temporal_ensemble_coeff` 分支 / Temporal ensemble alternative**:
   - 中文:另一条路 —— 每步都重新预测一整个 chunk,然后用 `temporal_ensembler` 把"在当前时刻、来自多个不同起点 chunk 的预测"做指数加权平均。最平滑(每步都有最新观测 + 多次预测投票),但每步都推理,最贵。ACT 论文里用 `coeff` 控制权重衰减:`w_i = exp(-coeff * i)`,越老的 chunk 权重越小。
   - English: the alternative — re-predict a full chunk every step, then `temporal_ensembler` exponentially-weight-averages predictions for the current timestep coming from multiple chunks with different start points. Smoothest (fresh observation every step + voting across predictions) but priciest (inference every step). The ACT paper uses `coeff` for weight decay: `w_i = exp(-coeff * i)`, older chunks weigh less.

## 类比 / The analogy

像点外卖:你不会每饿一次就下一单(推理太慢),而是一次点一周的备餐(预测一个 chunk),放冰箱排队(deque),每天取一份吃(popleft)。吃到还剩两天份时(队列快空)再下下一周的单。Temporal ensemble 则像每天都重新规划整周菜单,但今天吃啥取"昨天计划的今天" + "今天计划的今天"的折中 —— 更新鲜,但每天都得规划一次。

Like meal-prepping. You don't order food every time you're hungry (inference too slow) — you cook a week's meals at once (predict a chunk), stack them in the fridge queue (deque), and eat one a day (popleft). When two days' worth remain (queue near-empty) you cook next week's. Temporal ensemble is re-planning the whole week's menu daily but eating a blend of "yesterday's plan for today" and "today's plan for today" — fresher, but you plan every single day.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:在 nanoVLA 里这是 `nano/vla/runtime/action_queue.py` —— 推理运行时的核心,夹在"策略模型"和"机器人执行器"之间。上游:action head(离散 token 解码 或 flow matching 积分)产出一个 `[B, chunk_size, action_dim]` 的动作块;下游:机器人控制器,每个控制周期 `select_action` 拿一个。如果省掉 chunking、每步都推理:控制频率会被 VLA 推理延迟卡死(比如 VLA 100ms/次 → 最多 10Hz,远低于机器人需要的 30-50Hz),而且动作抖。chunking 是把"慢推理"用在"快控制"上的必备适配层。生产实现要补:(1) **temporal ensemble 的数值稳定**(权重归一化、处理 chunk 边界);(2) **异步推理**(后台线程提前算下一个 chunk,见今天的 inference-loop 笔记);(3) **chunk 拼接平滑**(两个 chunk 交界处可能跳变,要做插值或重叠加权)。chunk_size、n_action_steps 这俩超参直接决定延迟-平滑度权衡。

English: in nanoVLA this is `nano/vla/runtime/action_queue.py` — the inference runtime core, sitting between the policy model and the robot actuator. Upstream: the action head (discrete token decode or flow-matching integration) emits a `[B, chunk_size, action_dim]` chunk. Downstream: the robot controller, which calls `select_action` once per control cycle. Skip chunking and infer every step: control frequency gets capped by VLA inference latency (100 ms/forward → max 10 Hz, far below the 30-50 Hz robots need) and motion jitters. Chunking is the mandatory adapter that lets slow inference drive fast control. Production additions: (1) **numerically stable temporal ensemble** (weight normalisation, chunk-boundary handling), (2) **async inference** (a background thread computes the next chunk early — see today's inference-loop note), (3) **chunk-seam smoothing** (the junction between two chunks can jump; interpolate or overlap-weight). The `chunk_size` and `n_action_steps` hyperparameters directly set the latency-smoothness tradeoff.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
from collections import deque
import torch

class ChunkedPolicy:
    def __init__(self, chunk_size=8, n_action_steps=4, action_dim=7):
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self._queue = deque([], maxlen=n_action_steps)
        self.infer_count = 0

    def _model_forward(self, obs):
        # pretend the model predicts a full chunk from the observation
        self.infer_count += 1
        return torch.randn(1, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def select_action(self, obs):
        if len(self._queue) == 0:
            actions = self._model_forward(obs)[:, : self.n_action_steps]   # [1, n_action_steps, dim]
            self._queue.extend(actions.transpose(0, 1))                    # per-timestep elements
        return self._queue.popleft()

policy = ChunkedPolicy(chunk_size=8, n_action_steps=4)
for step in range(12):                       # simulate 12 env steps
    a = policy.select_action(obs=None)
print("env steps        :", 12)
print("model inferences :", policy.infer_count)   # 3, not 12 — 12 / n_action_steps(4)
print("inference saved  :", f"{(1 - policy.infer_count/12)*100:.0f}%")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
env steps        : 12
model inferences : 3
inference saved  : 75%
```

中文:12 个控制步只推理了 3 次(`12 / n_action_steps`),省了 75% 的推理。`n_action_steps` 越大,推理越省,但对环境变化的响应越慢 —— 这就是核心权衡。

English: 12 control steps trigger only 3 inferences (`12 / n_action_steps`), saving 75%. Larger `n_action_steps` saves more inference but reacts slower to environment changes — the core tradeoff.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ACT / Aloha (原始出处)** / **ACT / Aloha (the origin)**: 中文 — temporal ensemble 就是 ACT 论文提出的,这份 lerobot 代码是它的官方实现。 / English — temporal ensemble was introduced in the ACT paper; this lerobot code is its canonical implementation.
- **π₀ / GR00T / OpenVLA-OFT** / **π₀ / GR00T / OpenVLA-OFT**: 中文 — 全部用 action chunking,只是 chunk 的产生方式不同(flow matching vs 离散 token vs 并行解码)。 / English — all use action chunking; only the chunk-generation differs (flow matching vs discrete tokens vs parallel decoding).
- **Diffusion Policy receding horizon** / **Diffusion Policy receding horizon**: 中文 — 同样的"预测一段、执行一段、滚动重规划"思路,机器人领域叫 receding horizon control。 / English — the same "predict a horizon, execute part, re-plan" idea; robotics calls it receding horizon control.
- **LLM speculative decoding** / **LLM speculative decoding**: 中文 — 远亲:一次产出多个 token 再验证消费,也是"批量产出 + 队列消费"减少前向次数。 / English — a distant cousin: produce several tokens at once then verify-consume, also "batch produce + queue consume" to cut forward passes.

## 注意事项 / Caveats / when it breaks

- **n_action_steps 太大反应迟钝** / **Large `n_action_steps` reacts slowly**: 中文 — 执行步数越多,越久才看一次新观测,环境突变时来不及响应(比如物体被碰掉)。动态环境要小 chunk。 / English — more executed steps means longer between fresh observations; the policy can't react to sudden changes (e.g. an object knocked over). Dynamic environments need small chunks.
- **chunk 边界跳变** / **Chunk-seam discontinuity**: 中文 — 队列用尽重新预测时,新 chunk 第一步可能和旧 chunk 最后一步不连续。temporal ensemble 能缓解,纯队列不能。 / English — when the queue refills, the new chunk's first step may not be continuous with the old chunk's last. Temporal ensemble smooths this; a plain queue doesn't.
- **temporal ensemble 贵** / **Temporal ensemble is expensive**: 中文 — 每步都推理,失去了 chunking 省推理的好处,只换平滑。资源紧时用队列。 / English — it infers every step, forfeiting the inference savings for smoothness. Use the queue when compute-bound.
- **transpose 别搞错维度** / **Get the transpose right**: 中文 — `[batch, time, dim]` → `[time, batch, dim]` 才能让 deque 按时间步 pop;搞反会 pop 出整个 batch 的错位动作。 / English — `[batch, time, dim]` → `[time, batch, dim]` is required so the deque pops per timestep; reversing it pops misaligned batch actions.

## 延伸阅读 / Further reading

- [ACT / Aloha: Action Chunking with Transformers (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy — receding horizon control](https://arxiv.org/abs/2303.04137)
- [lerobot ACT policy](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act)
- [Today's VLA action survey doc](./README-action-survey.md)
