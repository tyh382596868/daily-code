---
date: 2026-07-19
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/scripts/lerobot_eval.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, evaluation, processors, rollout]
---

# LeRobot eval rollout：把环境、policy 和处理器接成闭环 / LeRobot eval rollout: Close the Loop Between Env, Policy, and Processors

> **一句话 / In one line**: LeRobot 的 eval rollout 不是直接把 observation 塞进 policy，而是先后经过环境处理器、policy 处理器、policy 后处理器和环境后处理器。 / LeRobot's eval rollout does not feed observations straight into the policy; it routes them through environment and policy processors on both input and output.

## 为什么重要 / Why this matters

机器人评估最容易出错的地方是坐标系和 schema。仿真环境给的是 numpy、环境自己的键名和动作约定；policy 期待的是 LeRobot 的 batch schema。这个 rollout 把边界都集中在 processor 层，policy 只看到干净的 observation，环境只收到干净的 action。

Robot evaluation often fails at coordinate systems and schemas. A simulator emits numpy arrays, environment-specific keys, and its own action convention; the policy expects a LeRobot batch. This rollout keeps those boundaries in processors, so the policy sees clean observations and the environment receives clean actions.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/scripts/lerobot_eval.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py)

```python
while not np.all(done) and step < max_steps:
    observation = preprocess_observation(observation)

    try:
        observation["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            observation["task"] = list(env.call("task"))
        except (AttributeError, NotImplementedError):
            observation["task"] = [""] * env.num_envs

    observation = env_preprocessor(observation)
    observation = preprocessor(observation)

    with torch.inference_mode():
        action = policy.select_action(observation)

    action = postprocessor(action)
    action_transition = {ACTION: action}
    action_transition = env_postprocessor(action_transition)
    action = action_transition[ACTION]

    action_numpy: np.ndarray = action.to("cpu").numpy()
    assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
    observation, reward, terminated, truncated, info = env.step(action_numpy)
```

## 逐行讲解 / What's happening

1. **先统一 observation / Normalize the observation first**: 中文: `preprocess_observation` 把环境返回值转成 policy 能继续处理的基础形态。 English: `preprocess_observation` converts raw env output into the base format the policy stack can consume.
2. **任务文本从环境取 / Task text comes from the env**: 中文: 代码先试 `task_description`，再试 `task`，最后回退成空字符串。 English: the code tries `task_description`, falls back to `task`, then uses empty strings.
3. **两层输入处理 / Two input processors**: 中文: `env_preprocessor` 处理环境习惯，`preprocessor` 处理模型习惯。 English: `env_preprocessor` handles env quirks, while `preprocessor` handles model expectations.
4. **推理无梯度 / Inference has no gradients**: 中文: `torch.inference_mode()` 明确这是部署路径，不是训练路径。 English: `torch.inference_mode()` makes this a deployment path, not a training path.
5. **两层输出处理 / Two output processors**: 中文: policy 输出先反归一化，再变成环境动作。 English: policy output is unnormalized, then converted into the env action contract.

## 类比 / The analogy

这像国际机场的转机通道。乘客不是直接从 A 国跑到 B 国登机口，中间要过护照、安检、登机牌和海关格式转换。每个关口只处理自己的规则。

It is like an international airport transfer. A passenger does not walk straight from country A to country B's gate; passport checks, security, boarding passes, and customs each handle their own contract.

## 自己跑一遍 / Try it yourself

```python
def env_pre(obs):
    obs["state"] = [x / 10 for x in obs["state"]]
    return obs

def policy_pre(obs):
    return {"x": obs["state"], "task": obs.get("task", "")}

def policy(batch):
    return [round(sum(batch["x"]), 2)]

def policy_post(action):
    return [a * 10 for a in action]

def env_post(action):
    return {"action": [max(-1, min(1, action["action"][0]))]}

obs = {"state": [2, 3], "task": "push"}
batch = policy_pre(env_pre(obs))
action = env_post({"action": policy_post(policy(batch))})
print(batch)
print(action)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'x': [0.2, 0.3], 'task': 'push'}
{'action': [1]}
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi transforms** / **openpi transforms**: 用 transform group 把 dataset schema 和 policy schema 分开。 / Transform groups separate dataset schema from policy schema.
- **LeRobot record loop** / **LeRobot record loop**: 数据采集也用 processor 管理 teleop action、robot action 和 observation。 / Data recording also routes teleop action, robot action, and observation through processors.

## 注意事项 / Caveats / when it breaks

- **processor 顺序不能乱 / Processor order matters**: 先归一化再转坐标，和先转坐标再归一化，可能不是同一个动作。 / Normalize-then-transform and transform-then-normalize may produce different actions.
- **batch 维要保留 / Keep the batch dimension**: 代码断言 action 是 `(batch, action_dim)`。 / The code asserts the action is `(batch, action_dim)`.
- **task 文本缺失要显式处理 / Missing task text must be explicit**: 空字符串比缺 key 更容易调试。 / An empty string is easier to debug than a missing key.

## 延伸阅读 / Further reading

- [LeRobot eval script](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py)
- [LeRobot env processor docs](https://github.com/huggingface/lerobot/blob/main/docs/source/env_processor.mdx)
