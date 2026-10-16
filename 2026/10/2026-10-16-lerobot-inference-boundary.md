---
date: 2026-10-16
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/common/control_utils.py
permalink: https://github.com/huggingface/lerobot/blob/297774fca4e3ad0aa34e7425fbf937111637e77b/src/lerobot/common/control_utils.py#L44-L92
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, inference-boundary]
---

# LeRobot 推理边界：把一次动作预测封进可控的安全壳 / LeRobot Inference Boundary: Put One Action Prediction in a Controlled Shell

> **一句话 / In one line**: `predict_action` 先复制观测、再统一预处理/推理/后处理，最后才把动作交给机器人。 / `predict_action` copies the observation, runs one normalized inference pipeline, and only then releases an action to the robot.

## 为什么重要 / Why this matters

机器人控制最容易在边界处出错：相机数组、任务文本、设备、精度和模型输出的形状都在这里相遇。把它们收进一个函数能让调用端只面对“观测进、动作出”的稳定契约。

Robotics failures often happen at boundaries, where camera arrays, task text, device placement, precision, and output shape meet. A single function gives callers a stable “observation in, action out” contract.

## 代码 / The code

`huggingface/lerobot` — [`control_utils.py`](https://github.com/huggingface/lerobot/blob/297774fca4e3ad0aa34e7425fbf937111637e77b/src/lerobot/common/control_utils.py#L44-L92)

```python
observation = copy(observation)
with (
    torch.inference_mode(),
    torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
):
    observation = prepare_observation_for_inference(observation, device, task, robot_type)
    observation = preprocessor(observation)
    action = policy.select_action(observation)
    action = postprocessor(action)
return action
```

## 逐行讲解 / What's happening

1. 中文: `copy` 防止预处理器就地改写上游传入的观测；`inference_mode` 则明确这条路径不建反向图。
   English: `copy` protects the caller’s observation from in-place processors, while `inference_mode` makes the no-gradient contract explicit.
2. 中文: AMP 只在 CUDA 启用；随后先把原始观测转成 policy 所需张量，再运行可组合的前处理链。
   English: AMP is CUDA-only. Raw observations become policy tensors before the composable preprocessing chain runs.
3. 中文: `select_action` 只负责 policy；后处理才负责反归一化、限幅或设备格式。
   English: `select_action` belongs to the policy; denormalization, clipping, and actuator formatting belong downstream.

## 类比 / The analogy

像机场安检：旅客先被分流、过检查、进入航班，最终才到登机口；不要让每位乘客自己决定流程。

It is airport security: route, inspect, board—rather than letting each passenger invent the process.

## 自己跑一遍 / Try it yourself

```python
def predict(obs, preprocess, policy, postprocess):
    obs = dict(obs)
    return postprocess(policy(preprocess(obs)))

obs = {"state": [1, 2]}
action = predict(obs, lambda x: {**x, "state": [v / 2 for v in x["state"]]},
                 lambda x: sum(x["state"]), lambda a: max(-1, min(1, a)))
print(obs, action)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'state': [1, 2]} 1
```

中文: 原始观测保持不变，而动作在最后一道门被约束。  
English: The input stays intact and the action is constrained at the final gate.

## 注意事项 / Caveats / when it breaks

- 中文: `inference_mode` 不适用于需要梯度的在线自适应。 / English: `inference_mode` is wrong for online adaptation that needs gradients.
- 中文: 后处理必须与训练时的动作表示匹配。 / English: Postprocessing must match the action representation used in training.

## 延伸阅读 / Further reading

- [LeRobot control utilities](https://github.com/huggingface/lerobot/blob/297774fca4e3ad0aa34e7425fbf937111637e77b/src/lerobot/common/control_utils.py#L44-L92)
