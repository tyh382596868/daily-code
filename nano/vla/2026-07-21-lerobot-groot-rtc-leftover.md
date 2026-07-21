---
date: 2026-07-21
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/modeling_groot.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/groot/modeling_groot.py#L1932-L2028
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, lerobot, groot, real-time-chunking, action-chunking]
component: action-chunking
variant: advanced
---

# LeRobot GR00T RTC：先裁掉 padding，再把旧动作变成约束 / LeRobot GR00T RTC: Strip Padding, Then Turn Old Actions into Constraints

> **一句话 / In one line**: GR00T 的 RTC wrapper 会从上一段 leftover 动作里找出真实前缀，裁剪/补齐到模型 action 维度，再把它作为下一次预测的 overlap 约束。 / GR00T's RTC wrapper finds the real prefix in leftover actions, trims or pads it to the model action dimension, then feeds it as the overlap constraint for the next prediction.

## 为什么重要 / Why this matters

机器人策略常常一次预测一整段动作，但控制器只执行前几步。下一次预测时，如果完全丢掉上一段剩余动作，动作 chunk 的交界会抖；如果把 padding 也当真实动作，模型又会被零动作误导。这个 wrapper 处理的就是这条边界。

Robot policies often predict a chunk of actions while the controller executes only the first few. On the next call, dropping the leftover chunk causes boundary jitter; treating padding as real actions misleads the model. This wrapper handles that boundary.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/modeling_groot.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/groot/modeling_groot.py#L1932-L2028)

```python
# Simplified teaching slice, not a verbatim copy.
valid = abs(prev_actions).sum(dim=(batch_dim, action_dim)) > 0
if not any(valid):
    return inputs, None

prev_actions = prev_actions[:, :last_valid_step(valid), :]
prev_actions = prev_actions[:, -model_action_horizon:, :]
prev_actions = pad_or_trim_action_dim(prev_actions, max_action_dim)

overlap_steps = min(action_horizon, execution_horizon)
frozen_steps = clamp(inference_delay, 0, overlap_steps)

inputs["action"] = prev_actions.to(state.device, state.dtype)
options = {
    "action_horizon": action_horizon,
    "rtc_overlap_steps": overlap_steps,
    "rtc_frozen_steps": frozen_steps,
    "rtc_ramp_rate": rtc_ramp_rate,
}
```

## 逐行讲解 / What's happening

1. **先识别真实 leftover / Detect real leftovers first**: 中文: 全零行是固定 shape padding，不应该当动作约束。 English: all-zero rows are fixed-shape padding, not action constraints.
2. **超长前缀取尾部 / Keep the latest prefix**: 中文: 如果 leftover 比模型 horizon 长，只保留最接近当前时刻的尾部。 English: if leftovers exceed the model horizon, keep the most recent tail.
3. **动作维度对齐 / Align action dimension**: 中文: 维度太多就裁，太少就补零到模型最大动作维。 English: trim extra dimensions or pad missing ones to the model maximum.
4. **overlap 受执行步数限制 / Overlap is limited by execution horizon**: 中文: 模型只需要约束将和新 chunk 重叠的步。 English: the model only constrains steps that overlap the new chunk.
5. **frozen steps 表示延迟 / Frozen steps encode delay**: 中文: 推理延迟期间已经承诺的动作不能随便改。 English: actions already committed during inference delay should not be changed freely.

## 在 nanoVLA 中的位置 / Where this fits in nanoVLA

中文: 这是 `action-chunking` 层的生产版细节。nanoVLA 可以先实现“预测 H 步、执行 E 步”，再加入 leftover prefix 约束，最后加入 padding 清理和 frozen-step 逻辑。

English: This is a production detail of the `action-chunking` layer. A nanoVLA can start with "predict H steps, execute E steps", then add leftover-prefix constraints, and finally add padding cleanup plus frozen-step handling.

## 类比 / The analogy

像乐队换下一小节：上一小节最后几个音还在延音，新乐句必须接住它们；但谱纸上为了排版补的空白小节不能当成真的音乐。

It is like a band entering the next bar: the tail of the previous phrase is still ringing and the new phrase must respect it; blank measures inserted for layout are not real music.

## 自己跑一遍 / Try it yourself

```python
def trim_leftover(rows, horizon):
    valid_until = 0
    for i, row in enumerate(rows):
        if any(x != 0 for x in row):
            valid_until = i + 1
    real = rows[:valid_until]
    return real[-horizon:]

prev = [[0.1, 0.2], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]]
print(trim_leftover(prev, horizon=3))
print(trim_leftover(prev, horizon=1))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0.1, 0.2], [0.3, 0.4]]
[[0.3, 0.4]]
```

## 注意事项 / Caveats / when it breaks

- **零动作可能是真动作 / Zero action may be real**: 如果机器人合法动作常常全零，需要额外 mask，而不能只靠数值判断。 / if all-zero is a legitimate action, carry an explicit mask instead of relying on values.
- **维度补零要和归一化一致 / Padding must match normalization**: 补零前后要确认动作空间的零点含义。 / confirm what zero means in the normalized action space.
- **延迟估计会影响平滑 / Delay estimates affect smoothness**: `frozen_steps` 太大太小都会改变响应性。 / too many or too few frozen steps changes responsiveness.

## 延伸阅读 / Further reading

- [LeRobot GR00T RTC code](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/groot/modeling_groot.py#L1932-L2028)
- [LeRobot repository](https://github.com/huggingface/lerobot)
