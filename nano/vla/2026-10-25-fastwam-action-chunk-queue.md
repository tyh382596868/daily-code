---
date: 2026-10-25
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/fastwam/modeling_fastwam.py
permalink: https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/fastwam/modeling_fastwam.py#L284-L324
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking, queue]
build_role: action-chunking (advanced variant, policy-owned chunk queue)
---

# FastWAM 动作队列：预测一段，控制器逐拍取 / FastWAM Action Queue: Predict a Chunk, Serve One Control Tick at a Time

> **一句话 / In one line**: FastWAM 将一次生成的 action chunk 转置入队，让控制循环每次只弹出当前一步。 / FastWAM transposes one generated action chunk into a queue so the control loop pops exactly one step per tick.

## 为什么重要 / Why this matters

VLA 通常比机器人控制频率慢：如果每个控制 tick 都重新推理，延迟和抖动会放大。这个小接口把“昂贵的 chunk 预测”和“便宜的单步执行”解耦，并把缓存所有权放在 policy 内部。

VLAs are commonly slower than robot control rates. Re-running inference at every tick magnifies latency and jitter. This small interface separates expensive chunk prediction from cheap one-step execution, keeping cache ownership inside the policy.

## 代码 / The code

`huggingface/lerobot` — [`modeling_fastwam.py`](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/fastwam/modeling_fastwam.py#L284-L324)

```python
@torch.no_grad()
def predict_action_chunk(self, batch, *, noise=None):
    self.eval()
    infer_kwargs = _batch_to_infer_kwargs(batch=batch, config=self.config)
    infer_kwargs["noise"] = noise
    action = _action_from_model_output(self.model.infer_action(**infer_kwargs))
    return action.to(device=batch_device(batch), dtype=torch.float32)

@torch.no_grad()
def select_action(self, batch, *, noise=None):
    self.eval()
    if len(self._action_queue) == 0:
        actions = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
        self._action_queue.extend(actions.transpose(0, 1))
    return self._action_queue.popleft()
```

## 逐行讲解 / What's happening

1. **第 298-316 行 / Lines 298-316**：先把通用 LeRobot batch 翻译为 world-action model 的输入，再规范化输出设备和 dtype。 / The generic LeRobot batch is translated to world-action-model inputs, then the output device and dtype are normalized.
2. **第 321-323 行 / Lines 321-323**：队列为空才推理；`transpose(0, 1)` 让时间维成为队首顺序。 / Inference only happens for an empty queue; transposition makes the time axis become queue order.
3. **第 324 行 / Line 324**：控制 API 永远返回一条 action，而不把 horizon 细节泄漏给执行器。 / The control API always returns one action, hiding horizon details from the actuator.

## 类比 / The analogy

像面包房一次烤一盘面包、柜台每次卖一个：烤炉昂贵，递出一个面包很便宜。 / It is a bakery that bakes one tray at a time but serves one roll per customer: the oven is expensive, serving is cheap.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 `action-chunking` 组件的进阶变体。上游是 observation/language encoder 与连续 action head，输入是当前 batch，输出是 `[B, H, A]` chunk；下游是实时执行器，每 tick 消费 `[B, A]`。省掉它就必须按控制频率调用大模型；生产实现还要加时间戳、观测变更失效与安全急停。

This is an advanced `action-chunking` component. Upstream are observation/language encoders and a continuous action head; it receives a batch and emits `[B, H, A]`, while the real-time executor consumes `[B, A]` each tick. Without it, the large model runs at control frequency. Production versions also need timestamps, invalidation on observation change, and an emergency stop.

## 自己跑一遍 / Try it yourself

```python
from collections import deque

queue = deque()
def select():
    if not queue:
        queue.extend([[1, 0], [2, 0], [3, 0]])
    return queue.popleft()
print(select(), select(), select())
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `[1, 0] [2, 0] [3, 0]`。

队列只解决复用，不保证旧动作仍安全。 / A queue enables reuse; it does not guarantee that old actions are still safe.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot SmolVLA / LeRobot SmolVLA**：同样以队列把 horizon 变成 control ticks。 / It also turns a horizon into control ticks with a queue.
- **openpi ActionChunkBroker / openpi ActionChunkBroker**：在 client 侧管理 chunk 补给。 / It manages chunk refills on the client side.

## 注意事项 / Caveats / when it breaks

- **观测过期 / Stale observations**：接触、滑移或人类介入后必须丢弃剩余队列。 / Contact, slip, or human intervention must invalidate remaining actions.
- **批次语义 / Batch semantics**：多样本路径需要确保每个样本使用正确的 chunk。 / Multi-sample paths must ensure every sample receives its own chunk.

## 延伸阅读 / Further reading

- [FastWAM policy source](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/fastwam/modeling_fastwam.py)
