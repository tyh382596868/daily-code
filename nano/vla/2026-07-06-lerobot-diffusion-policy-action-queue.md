---
date: 2026-07-06
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/diffusion/modeling_diffusion.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/modeling_diffusion.py#L95-L159
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking]
build_role: action-chunking advanced variant
---

# LeRobot DiffusionPolicy：一次生成一段，只执行一个 / LeRobot DiffusionPolicy: Generate a Chunk, Execute One Action

> **一句话 / In one line**: `select_action` 缓存观测历史和动作队列，动作队列空了才重新跑 diffusion sampler。 / `select_action` caches observation history and an action queue, rerunning the diffusion sampler only when the queue is empty.

## 为什么重要 / Why this matters

VLA 或 diffusion policy 往往一次预测多个未来动作，但机器人控制环每次只需要一个动作。这个函数把“批量生成”和“逐步执行”接起来：观测进入短历史队列，模型生成 horizon，当前控制周期只 `popleft()` 一个动作。

A VLA or diffusion policy often predicts a future action chunk, while the robot control loop needs exactly one action now. This function bridges the two: observations enter a short history queue, the model generates a horizon, and the current control tick consumes one action with `popleft()`.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/diffusion/modeling_diffusion.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/modeling_diffusion.py#L95-L159)

```python
@torch.no_grad()
def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
    queues_populated = any(len(q) > 0 for q in self._queues.values())
    if queues_populated:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
    else:
        batch = dict(batch)
        if self.config.image_features:
            for key in self.config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
    actions = self.diffusion.generate_actions(batch, noise=noise)
    return actions

@torch.no_grad()
def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
    if ACTION in batch:
        batch.pop(ACTION)

    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
    self._queues = populate_queues(self._queues, batch)

    if len(self._queues[ACTION]) == 0:
        actions = self.predict_action_chunk(batch, noise=noise)
        self._queues[ACTION].extend(actions.transpose(0, 1))

    action = self._queues[ACTION].popleft()
    return action
```

## 逐行讲解 / What's happening

1. **第 3-5 行 / Lines 3-5 (`queues_populated`)**:
   - 中文: 在线推理时，batch 会被替换成内部队列里最近几帧观测。
   - English: during online inference, the incoming batch is replaced with the recent observations stored in queues.
2. **第 7-12 行 / Lines 7-12 (image stack)**:
   - 中文: 多相机图像被堆到统一的 `OBS_IMAGES` 字段，给模型一个固定入口。
   - English: multiple camera images are stacked into one `OBS_IMAGES` field, giving the model a stable input.
3. **第 25-27 行 / Lines 25-27 (refill action queue)**:
   - 中文: 只有 action 队列空了才生成新 chunk，避免每个控制周期都跑完整采样。
   - English: a new chunk is generated only when the action queue is empty, avoiding a full sampling loop on every control tick.
4. **第 29 行 / Line 29 (`popleft`)**:
   - 中文: 控制器每次只拿队首动作，队列把慢模型和快控制频率解耦。
   - English: the controller consumes only the first queued action, decoupling a slow model from a faster control loop.

## 类比 / The analogy

像电饭煲一次煮一锅饭，但你每餐只盛一碗。锅空了才重新煮，吃饭这个动作不需要每次从淘米开始。

It is like cooking a full pot of rice but serving one bowl per meal. You cook again only when the pot is empty; each meal does not start from washing rice.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-chunking` 的高级变体，依赖上游的 observation encoder 和 action head。nanoVLA 里可以把它写成 `PolicyRuntime`：维护 `obs_queue` 和 `action_queue`，当 `action_queue` 为空时调用模型生成 `[H, action_dim]`，否则只弹出下一步。生产级实现还要处理队列重置、延迟补偿和 chunk 重叠融合。

This is an advanced `action-chunking` variant, downstream of the observation encoder and action head. In nanoVLA, it can be a `PolicyRuntime` object that owns `obs_queue` and `action_queue`: when the action queue is empty, call the model for `[H, action_dim]`; otherwise pop the next step. A production version also needs reset handling, latency compensation, and chunk overlap smoothing.

## 自己跑一遍 / Try it yourself

```python
from collections import deque

obs_queue, action_queue = deque(maxlen=2), deque()

def model_generate(obs_hist):
    last = obs_hist[-1]
    return [last + 0.1, last + 0.2, last + 0.3]

for obs in [1.0, 2.0, 3.0, 4.0]:
    obs_queue.append(obs)
    if not action_queue:
        action_queue.extend(model_generate(list(obs_queue)))
    print(round(action_queue.popleft(), 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
1.1
1.2
1.3
4.1
```

前三步来自同一个 chunk；队列耗尽后，模型才用最新观测重新生成。

The first three steps come from one chunk; only after the queue is drained does the model regenerate from the latest observation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Real-Time Chunking** / **Real-Time Chunking**: 会进一步融合重叠 chunk，减少动作跳变。 / It further blends overlapping chunks to reduce action discontinuities.
- **openpi async inference** / **openpi async inference**: 把策略服务器和机器人控制循环分离，也是同一类缓冲思想。 / It separates the policy server from the robot loop, using the same buffering idea.

## 注意事项 / Caveats / when it breaks

- **环境 reset 必须清队列** / **Environment reset must clear queues**: 否则新 episode 会执行旧动作。 / Otherwise a new episode can execute stale actions.
- **长 chunk 会增加反应延迟** / **Long chunks increase reaction latency**: 机器人可能继续执行过时计划。 / The robot may keep executing an outdated plan.

## 延伸阅读 / Further reading

- [LeRobot diffusion policy](https://github.com/huggingface/lerobot)
