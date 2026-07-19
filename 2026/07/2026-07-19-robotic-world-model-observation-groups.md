---
date: 2026-07-19
topic: robotics
source: trending
repo: leggedrobotics/robotic_world_model
file: source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py
permalink: https://github.com/leggedrobotics/robotic_world_model/blob/master/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, world-model, observation-groups, isaac-lab]
---

# Robotic World Model：把状态、动作和接触拆成预测组 / Robotic World Model: Split State, Action, and Contacts into Prediction Groups

> **一句话 / In one line**: `robotic_world_model` 用 Isaac Lab 的 observation groups 明确哪些量是 dynamics 输入，哪些量是 world model 要预测的输出头。 / `robotic_world_model` uses Isaac Lab observation groups to state which signals feed dynamics and which become world-model prediction heads.

## 为什么重要 / Why this matters

机器人 world model 不只是预测下一帧图像。腿式机器人更关心速度、姿态、关节、力矩、接触和终止信号。把这些量拆成组，训练代码就能给不同 head 配不同 loss，也能在 imagined rollout 里只取策略真正需要的状态。

A robotic world model is not only a next-frame predictor. For legged robots, velocity, orientation, joints, torque, contacts, and termination matter. Splitting these signals into groups lets training assign different losses to different heads and lets imagined rollouts read only the state a policy needs.

## 代码 / The code

`leggedrobotics/robotic_world_model` — [`flat_env_cfg.py`](https://github.com/leggedrobotics/robotic_world_model/blob/master/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py)

```python
@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):
    @configclass
    class SystemStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
```

## 逐行讲解 / What's happening

1. **pretrain 用专门 observation config / Pretraining gets its own observation config**: 中文: world model 需要的信号和普通 policy observation 不完全一样。 English: a world model needs signals that are not identical to a normal policy observation.
2. **state group 聚合动力学状态 / The state group collects dynamics state**: 中文: 线速度、角速度、重力方向、关节位置和速度都进同一组。 English: linear velocity, angular velocity, gravity, joint position, and joint velocity share one group.
3. **action group 用上一动作 / The action group uses the previous action**: 中文: dynamics 预测必须知道系统刚吃了什么控制量。 English: dynamics prediction needs to know which control was just applied.
4. **关闭 corruption / Corruption is disabled**: 中文: 训练 world model 时目标信号应尽量干净。 English: world-model targets should stay clean.
5. **拼接 terms / Terms are concatenated**: 中文: 每组最后变成一个 dense vector，方便接 MLP/RNN dynamics。 English: each group becomes a dense vector for an MLP or recurrent dynamics model.

## 类比 / The analogy

像给维修技师分仪表盘：发动机转速、车速、油温和刹车状态分门别类摆好。技师不用从一堆杂乱传感器里猜哪个是诊断目标。

It is like arranging a mechanic's dashboard: RPM, speed, temperature, and brake state are grouped by purpose. The mechanic does not guess which sensor belongs to which diagnosis.

## 自己跑一遍 / Try it yourself

```python
signals = {
    "base_lin_vel": [1.0, 0.0],
    "joint_pos": [0.1, -0.2],
    "last_action": [0.3, 0.4],
}

groups = {
    "system_state": ["base_lin_vel", "joint_pos"],
    "system_action": ["last_action"],
}

packed = {name: sum((signals[k] for k in keys), []) for name, keys in groups.items()}
print(packed)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'system_state': [1.0, 0.0, 0.1, -0.2], 'system_action': [0.3, 0.4]}
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DROID timestep processor** / **DROID timestep processor**: 单步机器人数据先整理成稳定训练契约。 / One robot step is normalized into a stable training contract.
- **LeRobot processors** / **LeRobot processors**: observation 和 action 边界也被显式处理。 / Observation and action boundaries are also explicit.

## 注意事项 / Caveats / when it breaks

- **组名就是 loss 合同 / Group names are loss contracts**: 改名会影响训练配置和 checkpoint。 / Renaming groups affects training configs and checkpoints.
- **action 要和 state 对齐 / Action must align with state**: 用错上一动作会让 dynamics 学到假因果。 / A mismatched previous action teaches false causality.
- **contact 是高价值监督 / Contacts are high-value targets**: 只预测连续 state 可能漏掉失败和摔倒。 / Predicting only continuous state can miss failure and falling signals.

## 延伸阅读 / Further reading

- [robotic_world_model flat env config](https://github.com/leggedrobotics/robotic_world_model/blob/master/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py)
- [robotic_world_model repository](https://github.com/leggedrobotics/robotic_world_model)
