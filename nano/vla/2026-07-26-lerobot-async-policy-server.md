---
date: 2026-07-26
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/async_inference/policy_server.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/async_inference/policy_server.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, async]
build_role: inference-loop advanced variant, server-side observation-to-action boundary
---

# LeRobot PolicyServer：把异步观测变成可控推理节拍 / LeRobot PolicyServer: Turn Async Observations into a Controlled Inference Beat

> **一句话 / In one line**: policy server 把网络来的 observation 排队、批量转设备、调用 policy，再把 action 发回控制端。 / The policy server queues incoming observations, moves them to device, calls the policy, and sends actions back to the controller.

## 为什么重要 / Why this matters

真实机器人推理不是 `policy(obs)` 一行。相机、网络、GPU 和控制频率都有自己的节拍，server 的责任是把这些异步事件整理成稳定的 observation-to-action 合约。

Real robot inference is not just `policy(obs)`. Cameras, networks, GPUs, and control loops run at different rates; the server turns those asynchronous events into a stable observation-to-action contract.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/async_inference/policy_server.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/async_inference/policy_server.py)

```python
class PolicyServer:
    def __init__(self, policy, device, max_queue_size=1):
        self.policy = policy
        self.device = device
        self.observation_queue = queue.Queue(maxsize=max_queue_size)
        self.action_queue = queue.Queue(maxsize=max_queue_size)

    def receive_observation(self, observation):
        if self.observation_queue.full():
            self.observation_queue.get_nowait()
        self.observation_queue.put_nowait(observation)

    @torch.no_grad()
    def run_once(self):
        observation = self.observation_queue.get(timeout=1.0)
        batch = {k: v.to(self.device) for k, v in observation.items()}
        action = self.policy.select_action(batch)
        if self.action_queue.full():
            self.action_queue.get_nowait()
        self.action_queue.put_nowait(action.cpu())
```

## 逐行讲解 / What's happening

1. **两个 queue / Two queues**
   - 中文: observation 和 action 分成两个缓冲区，GPU 慢一点时也不会阻塞相机线程太久。
   - English: Separate observation and action queues let GPU latency avoid blocking the camera thread for too long.
2. **满队列丢旧帧 / Drop old frames when full**
   - 中文: 控制机器人更需要新鲜观测，过期帧保真但不保实时。
   - English: Robot control values fresh observations; stale frames may be accurate but no longer timely.
3. **`torch.no_grad()`**
   - 中文: 推理 server 不保留 autograd 图，省显存也避免长期服务泄漏。
   - English: The inference server avoids autograd graphs, saving memory and preventing long-lived leaks.
4. **`action.cpu()`**
   - 中文: 发回控制端前离开 GPU，避免网络线程持有 CUDA tensor。
   - English: Actions leave the GPU before being returned so network threads do not hold CUDA tensors.

## 类比 / The analogy

这像餐厅出餐窗口：服务员不断送新单，厨房只保留最新要做的单，菜做好后放到另一个窗口等取餐。

It is like a restaurant pass: waiters bring new orders, the kitchen keeps the freshest pending order, and finished dishes wait at another window.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `inference-loop` 组件的生产级外壳。上游是相机/机器人状态编码，下游是动作 chunk broker 或底层控制器。如果没有这个层，你的 nanoVLA 只能离线跑 batch，不能稳定接实时机器人。

English: This is the production shell around the `inference-loop` component. Upstream are camera and robot-state encoders; downstream are an action chunk broker or low-level controller. Without this layer, a nanoVLA can run offline batches but cannot reliably serve a live robot.

## 自己跑一遍 / Try it yourself

```python
from queue import Queue

obs_q, act_q = Queue(maxsize=1), Queue(maxsize=1)
def put_latest(q, x):
    if q.full(): q.get_nowait()
    q.put_nowait(x)

def policy(obs):
    return {"move": obs["x"] * 2}

put_latest(obs_q, {"x": 1})
put_latest(obs_q, {"x": 3})
put_latest(act_q, policy(obs_q.get()))
print(act_q.get())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'move': 6}
```

中文: 第一帧被丢掉了，因为实时控制更需要最新状态。
English: The first frame was dropped because real-time control prefers the newest state.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi policy server** / **openpi policy server**: 中文: 同样把外部请求和模型推理解耦。 / English: It similarly decouples external requests from model inference.
- **ACT temporal ensemble** / **ACT temporal ensemble**: 中文: 后续动作缓存也在处理“不同时钟”问题。 / English: Later action caches solve the same multi-clock problem.

## 注意事项 / Caveats / when it breaks

- **丢帧策略** / **Drop policy**: 中文: 高速任务可能要记录被丢帧的时间戳。 / English: High-speed tasks may need timestamps for dropped frames.
- **线程安全** / **Thread safety**: 中文: 真实实现要处理异常、断连和超时。 / English: Real servers must handle exceptions, disconnects, and timeouts.

## 延伸阅读 / Further reading

- [LeRobot async inference](https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference)
- [LeRobot policy server source](https://github.com/huggingface/lerobot/blob/main/src/lerobot/async_inference/policy_server.py)
