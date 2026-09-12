---
date: 2026-07-23
topic: diffusion
source: trending
repo: Robert-gyj/Ctrl-World
file: README.md
permalink: https://github.com/Robert-gyj/Ctrl-World
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, diffusion, world-model, robotics, conditioning]
---

# Ctrl-World：把机器人控制变成视频生成条件 / Ctrl-World: Turn Robot Control into Video-Generation Conditioning

> **一句话 / In one line**: Ctrl-World 的核心建模思想是把机器人任务、动作或控制信号作为条件注入视频扩散模型，让生成的视频承担“可控世界模拟器”的角色。 / Ctrl-World's central modeling move is to inject robot task, action, or control signals into a video diffusion model so generated video acts as a controllable world simulator.

## 为什么重要 / Why this matters

机器人世界模型不能只生成“看起来合理”的视频，还要对动作有反应。把控制信号变成扩散条件，是从通用视频模型走向 embodied world model 的关键一步：同一个初始观测，在不同动作下应该产生不同未来。

A robot world model cannot only generate plausible video; it must respond to actions. Turning control into diffusion conditioning is the key move from generic video model to embodied world model: the same initial observation should lead to different futures under different actions.

## 代码 / The code

`Robert-gyj/Ctrl-World` — [repository](https://github.com/Robert-gyj/Ctrl-World)

```python
# Teaching slice of the conditioning contract.
def build_world_model_batch(obs_video, action, task_text):
    cond = {
        "video_context": obs_video[:, :1],   # current observation
        "action": normalize_action(action),  # robot control path
        "text": task_text,                   # language goal
    }
    target = obs_video[:, 1:]                # future frames
    return cond, target

def training_loss(model, cond, target, noise_level):
    noisy = add_noise(target, noise_level)
    pred = model(noisy, noise_level, cond)
    return mse(pred, target - noisy)
```

## 逐行讲解 / What's happening

1. **当前帧是上下文 / Current frame is context**: 中文: 模型先知道世界现在长什么样。 English: the model first sees the current world state.
2. **动作是条件 / Action is conditioning**: 中文: 控制信号不应该藏在 target 里，而应显式进入 denoiser。 English: control should enter the denoiser explicitly, not be hidden in the target.
3. **语言是目标约束 / Language is goal context**: 中文: 同一个动作在不同任务下可能有不同意义。 English: the same action can mean different things under different tasks.
4. **loss 监督未来 / Loss supervises the future**: 中文: 训练目标是让模型预测动作后的未来视频。 English: the target is the future video after the action.

## 类比 / The analogy

像驾驶模拟器：截图告诉你当前位置，方向盘输入告诉你要怎么开，导航语音告诉你目的地。只给截图不够，模拟器必须响应操作。

It is like a driving simulator: the screenshot gives the current state, steering input says how to move, and navigation says the goal. The screenshot alone is not enough; the simulator must respond to controls.

## 自己跑一遍 / Try it yourself

```python
def future_position(x, action):
    dx = {"left": -1, "right": 1, "stay": 0}[action]
    return x + dx

for action in ["left", "stay", "right"]:
    print(action, "->", future_position(10, action))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
left -> 9
stay -> 10
right -> 11
```

## 注意事项 / Caveats / when it breaks

- **动作归一化很关键 / Action normalization matters**: 不同机器人 DoF、量纲、频率不一致，直接拼接会污染条件分布。 / different DoFs, units, and frequencies pollute the conditioning distribution if concatenated raw.
- **视频一致不等于动力学正确 / Plausible video is not correct dynamics**: 需要用动作敏感指标评估，而不只是视觉质量。 / evaluation needs action-sensitive metrics, not only visual quality.
- **条件 dropout 要谨慎 / Condition dropout needs care**: CFG 有用，但动作条件丢太多会让模型学会忽略控制。 / CFG is useful, but dropping action conditions too often teaches the model to ignore control.

## 延伸阅读 / Further reading

- [Ctrl-World repository](https://github.com/Robert-gyj/Ctrl-World)

