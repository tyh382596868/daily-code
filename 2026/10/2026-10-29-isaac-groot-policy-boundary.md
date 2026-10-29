---
date: 2026-10-29
topic: robotics
source: tracked
repo: NVIDIA/Isaac-GR00T
file: gr00t/policy/policy.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/c56f54128d92f743889b7e4cecfc1ceafd88739e/gr00t/policy/policy.py#L80-L106
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, policy-contract]
---

# Isaac GR00T policy 边界 / Isaac GR00T Policy Boundary

> **一句话 / In one line**: `get_action` 把一次机器人决策夹在输入和输出校验之间。 / `get_action` places one robot decision between input and output validation.

## 为什么重要 / Why this matters

机器人 policy 的危险不只在模型预测错，也在观测字段或动作形状悄悄漂移；把校验放在公开入口处，调用者不会绕过契约。

A policy can fail because its interface drifts as well as because its model is wrong. Keeping validation at the public boundary prevents callers from bypassing the contract.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/policy/policy.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/c56f54128d92f743889b7e4cecfc1ceafd88739e/gr00t/policy/policy.py#L80-L106)

```python
def get_action(self, observation, options=None):
    if self.strict:
        self.check_observation(observation)
    action, info = self._get_action(observation, options)
    if self.strict:
        self.check_action(action)
    return action, info
```

## 逐行讲解 / What's happening

1. 中文：`strict` 让同一个实现既能在集成测试中严查，也能在经过验证的热路径中减少额外工作。
   English: `strict` lets the same implementation be defensive in integration tests and lighter on a trusted hot path.
2. 中文：子类只实现 `_get_action`；基类保留验证和返回值的统一语义。
   English: Subclasses implement only `_get_action`; the base class owns validation and a uniform return contract.

## 类比 / The analogy

这像机场登机口：安检在进入登机桥前，落地后还要核对行李标签。

It is an airport gate: screening happens before boarding, and the baggage tag is checked again at the destination.

## 自己跑一遍 / Try it yourself

```python
def act(obs):
    assert obs["state"] >= 0
    action = {"delta": obs["state"] + 1}
    assert action["delta"] <= 3
    return action
print(act({"state": 1}))
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `{'delta': 2}`

中文：把第二个断言删掉，坏动作会跨过模型边界。 English: Remove the second assertion and an invalid action crosses the model boundary.

## 注意事项 / Caveats / when it breaks

- 中文：真实部署应报告可诊断错误，而不是只抛 `assert`。 / English: Production deployment should emit diagnosable errors, not only `assert` failures.

## 延伸阅读 / Further reading

- [Isaac GR00T policy interface](https://github.com/NVIDIA/Isaac-GR00T/blob/c56f54128d92f743889b7e4cecfc1ceafd88739e/gr00t/policy/policy.py#L80-L106)
