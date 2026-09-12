---
date: 2026-08-04
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/lingbot_va/modeling_lingbot_va.py
permalink: https://github.com/huggingface/lerobot/blob/1e3a158e1395db7e5ac7639f993902ed85748a57/src/lerobot/policies/lingbot_va/modeling_lingbot_va.py#L403-L436
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, observation-memory, action-chunking]
build_role: short-term-observation-memory advanced variant, keyframe feedback for chunked inference
---

# LeRobot LingBot-VA keyframe buffer：执行动作时顺手攒观测 / LeRobot LingBot-VA Keyframe Buffer: Collect Observations While Executing Actions

> **一句话 / In one line**: `select_action` 一边消费 action queue，一边按 stride 缓存真实观测，等当前 chunk 用完后把 keyframes 反馈给下一次预测。 / `select_action` consumes an action queue while buffering real observations at a stride, then feeds those keyframes into the next prediction once the current chunk is exhausted.

## 为什么重要 / Why this matters

chunked VLA 推理有一个实际问题：模型一次预测多步动作，但环境每执行一步都会产生新的观测。如果完全忽略这些观测，下一段动作只基于过期世界；如果每步都重新推理，延迟又太高。这里的折中是短期观测记忆：执行期间采样 keyframes，chunk 结束后一次性反馈。

Chunked VLA inference has a practical tension: the model predicts several actions at once, but the environment emits a new observation after each step. Ignoring those observations makes the next chunk stale; re-running inference every step is expensive. This code takes the middle path: sample short-term keyframes during execution and feed them back when the chunk is done.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/lingbot_va/modeling_lingbot_va.py`](https://github.com/huggingface/lerobot/blob/1e3a158e1395db7e5ac7639f993902ed85748a57/src/lerobot/policies/lingbot_va/modeling_lingbot_va.py#L403-L436)

```python
@torch.no_grad()
def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
    """Return one action, refilling the chunk (and feeding back observed keyframes) as needed.

    Mirrors the upstream LIBERO client loop (``evaluation/libero/client.py``): the first obs is
    the conditioning frame; every observation produced afterwards is buffered as a keyframe and,
    once the chunk's actions are exhausted, the buffered frames + executed actions are fed back
    into the KV cache before the next chunk is predicted.
    """
    self.eval()
    self._ensure_frozen_modules()
    self._maybe_init_prompt(batch)

    if not self._started:
        # First call: this observation conditions the first chunk (it is *not* a keyframe).
        self._started = True
        actions = self.predict_action_chunk(batch)  # [B, chunk_size, n_used]
        self._action_queue.extend(actions.transpose(0, 1))  # [chunk_size, B, n_used]
        self._obs_buffer = []
        self._exec_step = 0
    else:
        # This observation is the result of the previously executed action -> a candidate
        # keyframe. Buffer it on the sub-step boundary the upstream client samples on.
        if (self._prev_j + 1) % self._keyframe_stride == 0:
            self._obs_buffer.append(self._extract_raw_obs(batch))
        if len(self._action_queue) == 0:
            # All actions for the current chunk have been executed; feed the observed
            # keyframes + executed actions back and predict the next chunk.
            actions = self.predict_action_chunk(None)
            self._action_queue.extend(actions.transpose(0, 1))
            self._exec_step = 0

    self._prev_j = self._exec_step % self.config.action_per_frame
    self._exec_step += 1
    return self._action_queue.popleft()
```

## 逐行讲解 / What's happening

1. **第 411-413 行 / Lines 411-413 (inference setup)**:
   - 中文: 进入 eval/no-grad 路径，确保冻结模块和 prompt 已准备好，避免在线控制时改动训练态。
   - English: The method enters eval/no-grad mode and ensures frozen modules plus prompt state are ready, avoiding training-mode changes during control.
2. **第 415-421 行 / Lines 415-421 (first chunk)**:
   - 中文: 第一次观测只负责 condition 第一段动作，不进入 keyframe buffer，因为它已经被当前 chunk 看见了。
   - English: The first observation conditions the first chunk and is not buffered as a keyframe because the current chunk has already seen it.
3. **第 423-431 行 / Lines 423-431 (feedback boundary)**:
   - 中文: 后续观测按 `_keyframe_stride` 入 buffer；动作队列耗尽时，用这些 keyframes 触发下一段预测。
   - English: Later observations enter the buffer at `_keyframe_stride`; when the action queue is empty, those keyframes drive the next chunk prediction.

## 类比 / The analogy

这像厨师一次做出一盘 8 个小菜，服务员每上两道就拍一张客人反馈照片。等这一盘上完，厨师根据照片调整下一盘，而不是每上一道都重新开会。

It is like a chef preparing eight small dishes at once while the server takes a feedback photo every two dishes. After the plate is served, the chef uses those photos to adjust the next plate instead of holding a meeting after every dish.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `short-term-observation-memory` 的 advanced variant。上游是机器人环境每步返回的 observation；中间层是 `_obs_buffer`、`_action_queue` 和执行步计数；下游是下一次 `predict_action_chunk`。如果省掉它，chunked policy 会在多个控制步内失去真实反馈。生产级实现还要处理多环境 batch、失败 reset、相机时间戳和 action/observation 对齐。

This is an advanced variant of `short-term-observation-memory`. Upstream is the per-step environment observation; the middle state is `_obs_buffer`, `_action_queue`, and execution counters; downstream is the next `predict_action_chunk`. Without it, a chunked policy loses real feedback for several control steps. A production version also needs batched environments, failure resets, camera timestamps, and action/observation alignment.

## 自己跑一遍 / Try it yourself

```python
from collections import deque

queue, obs_buffer = deque(), []
started, stride, prev_j, exec_step = False, 2, 0, 0

for obs in ["o0", "o1", "o2", "o3", "o4"]:
    if not started:
        started = True
        queue.extend(["a0", "a1"])
    else:
        if (prev_j + 1) % stride == 0:
            obs_buffer.append(obs)
        if not queue:
            print("feedback", obs_buffer)
            obs_buffer = []
            queue.extend(["next0", "next1"])
            exec_step = 0
    prev_j = exec_step % 2
    exec_step += 1
    print("act", queue.popleft())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
act a0
act a1
feedback ['o2']
act next0
act next1
feedback ['o4']
act next0
```

中文: keyframe 不是每步都收，而是在和模型时间压缩匹配的边界上收。
English: Keyframes are not collected every step; they are sampled at boundaries that match the model's temporal compression.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot pi0 denoise loop** / **LeRobot pi0 denoise loop**: 同样把多步动作作为一个 chunk 生成。 / It also predicts actions in multi-step chunks.
- **RTC action queue** / **RTC action queue**: 实时控制常把旧动作队列和新观测一起纳入调度。 / Real-time control often schedules old action queues together with new observations.

## 注意事项 / Caveats / when it breaks

- **stride 必须匹配模型压缩率** / **Stride must match compression**: stride 错了会让 VAE/KV cache 看到错误帧数。 / A wrong stride gives the VAE or KV cache the wrong number of frames.
- **reset 很关键** / **Reset is critical**: 新 episode 不清空 buffer 和 queue，会把上一个任务的记忆带进来。 / If a new episode does not clear buffers and queues, it inherits memory from the previous task.

## 延伸阅读 / Further reading

- LeRobot LingBot-VA source: https://github.com/huggingface/lerobot/blob/1e3a158e1395db7e5ac7639f993902ed85748a57/src/lerobot/policies/lingbot_va/modeling_lingbot_va.py#L403-L436
