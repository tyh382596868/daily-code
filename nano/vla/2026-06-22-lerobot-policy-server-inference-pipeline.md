---
date: 2026-06-22
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/async_inference/policy_server.py
permalink: https://github.com/huggingface/lerobot/blob/73782447f2ca420f8d71c9bd0a169ece5968d2d6/src/lerobot/async_inference/policy_server.py#L312-L407
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference, gRPC, deployment, action-chunk]
build_role: inference-loop — the deployment shell that wraps a trained VLA policy in a production gRPC server decoupled from the real-time robot controller
---

# 生产级 VLA 推理流水线：gRPC 策略服务器的 5 步内核 / Production VLA Inference Pipeline: the 5-Step Core of the gRPC Policy Server

> **一句话 / In one line**: `_predict_action_chunk` 是一个 5 步流水线，把机器人传感器帧变成打了时间戳的动作列表——这是把训练好的 VLA 模型接进 50 Hz 机器人控制循环的"部署外壳"。 / `_predict_action_chunk` is a 5-step pipeline that converts a raw robot sensor frame into a list of timestamped actions — the "deployment shell" that plugs a trained VLA into a 50 Hz robot control loop.

## 为什么重要 / Why this matters

训练一个 VLA 模型只是第一步，真正难的是把它部署到实机上：机器人控制器要求确定性的 50 Hz 实时循环，而 GPU 推理是有延迟的突发操作。LeRobot 的解决方案是一个 gRPC 服务器，把推理端和控制端彻底解耦。控制器不断发送观测帧、接收动作列表；服务器异步跑 GPU，预测一整个"动作块"（chunk）——未来 N 步的动作——让控制器可以在等下一次推理的同时继续执行已经预测好的动作。

`_predict_action_chunk` 是这个服务器的心脏：它把从 gRPC 收到的原始传感器数据，经过 5 个步骤变成机器人可以直接执行的 `TimedAction` 列表，每个 `TimedAction` 都带有精确的执行时刻（真实世界时间戳）。

Training a VLA model is only step one. The hard part is deploying it to real hardware: robot controllers demand a deterministic 50 Hz real-time loop, while GPU inference is a high-latency burst operation. LeRobot decouples the two with a gRPC server. The controller streams observations in, receives lists of actions back; the GPU server predicts a whole "action chunk" — N future steps — so the controller can execute already-predicted actions while waiting for the next inference. `_predict_action_chunk` is that server's core: it takes raw gRPC sensor data through 5 steps and returns a `TimedAction` list the robot executes directly.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/async_inference/policy_server.py`](https://github.com/huggingface/lerobot/blob/73782447f2ca420f8d71c9bd0a169ece5968d2d6/src/lerobot/async_inference/policy_server.py#L312-L407)

```python
def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
    """Turn a chunk of actions into a list of TimedAction instances,
    with the first action corresponding to t_0 and the rest corresponding to
    t_0 + i*environment_dt for i in range(len(action_chunk))
    """
    return [
        TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
        for i, action in enumerate(action_chunk)
    ]

def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get an action chunk from the policy. The chunk contains only"""
    chunk = self.policy.predict_action_chunk(observation)
    if chunk.ndim != 3:
        chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

    return chunk[:, : self.actions_per_chunk, :]

def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
    """Predict an action chunk based on an observation.

    Pipeline:
    1. Convert raw observation to LeRobot format
    2. Apply preprocessor (tokenization, normalization, batching, device placement)
    3. Run policy inference to get action chunk
    4. Apply postprocessor (unnormalization, device movement)
    5. Convert to TimedAction list
    """
    """1. Prepare observation"""
    observation: Observation = raw_observation_to_observation(
        observation_t.get_observation(),
        self.lerobot_features,
        self.policy_image_features,
    )

    """2. Apply preprocessor"""
    observation = self.preprocessor(observation)
    self.last_processed_obs: TimedObservation = observation_t

    """3. Get action chunk"""
    action_tensor = self._get_action_chunk(observation)

    """4. Apply postprocessor"""
    _, chunk_size, _ = action_tensor.shape
    processed_actions = []
    for i in range(chunk_size):
        single_action = action_tensor[:, i, :]
        processed_action = self.postprocessor(single_action)
        processed_actions.append(processed_action)

    action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
    action_tensor = action_tensor.detach().cpu()

    """5. Convert to TimedAction list"""
    action_chunk = self._time_action_chunk(
        observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
    )
    return action_chunk
```

## 逐行讲解 / What's happening

1. **`raw_observation_to_observation(...)`（步骤 1）**:
   - 中文: 把 gRPC 传进来的原始字节（相机图像、关节角度、gripper 状态……）转换成 LeRobot 的标准 `Observation` 字典。转换的具体规则由 `lerobot_features` 和 `policy_image_features` 定义——包括哪些键要保留、图像分辨率是否要 resize、通道顺序（HWC→CHW）等。这一步与硬件相关，是唯一需要按机器人型号定制的部分。
   - English: Converts the raw bytes arriving via gRPC (camera images, joint angles, gripper state…) into LeRobot's standard `Observation` dict. Conversion rules come from `lerobot_features` and `policy_image_features` — which keys to keep, whether to resize images, channel reordering (HWC→CHW). This is the only hardware-specific step and the one piece you'd customize per robot.

2. **`self.preprocessor(observation)`（步骤 2）**:
   - 中文: 对观测做归一化、batch 维度扩展（`unsqueeze(0)`），并把张量移到 GPU。`preprocessor` 是一个可配置的 callable，内部通常封装了图像归一化（减 ImageNet mean/std）和关节角度归一化（减 dataset mean 除以 std）。这一步把硬件传来的"真实世界单位"变成网络期望的"归一化张量"。
   - English: Applies normalization, adds a batch dimension (`unsqueeze(0)`), and moves tensors to GPU. The `preprocessor` callable typically wraps image normalization (subtract ImageNet mean/std) and joint normalization (subtract dataset mean, divide by std). This converts real-world units from the robot into the normalized tensors the policy expects.

3. **`self.policy.predict_action_chunk(observation)` + `chunk[:, :actions_per_chunk, :]`（步骤 3）**:
   - 中文: 这是唯一真正调用神经网络的地方。`predict_action_chunk` 返回 `(B, full_chunk_size, action_dim)`；`[:, :actions_per_chunk, :]` 截取前 N 步，N 由配置控制（允许模型预测更长的块但只执行前几步）。如果模型只返回 2D 张量（`(B, action_dim)`，即单步模型），`unsqueeze(0)` 先补上 chunk 维度。
   - English: The only call to the neural network. `predict_action_chunk` returns `(B, full_chunk_size, action_dim)`; `[:, :actions_per_chunk, :]` trims it to the first N steps — allowing the model to predict a longer horizon than the controller actually executes. If the model returns a 2D tensor (single-step policy), `unsqueeze(0)` adds the chunk dimension first.

4. **`for i in range(chunk_size): postprocessor(action_tensor[:, i, :])`（步骤 4）**:
   - 中文: `postprocessor` 期望每次处理一个时间步的动作（形状 `(B, action_dim)`），所以要逐步拆开处理再 stack 回去。后处理做的是归一化的反操作：把网络输出的"归一化关节角度"变回"真实世界关节弧度"，以及把张量搬回 CPU。如果不做后处理直接发给机器人，关节会转到错误位置。
   - English: The `postprocessor` expects one timestep at a time `(B, action_dim)`, so the chunk is unrolled per-step, processed, then stacked back. Post-processing inverts the normalization — converting the network's "normalized joint outputs" back to real-world joint radians — and moves tensors back to CPU. Skipping this and sending normalized values directly to the robot would move joints to the wrong positions.

5. **`_time_action_chunk(t_0, list(action_tensor), i_0)`（步骤 5）**:
   - 中文: 给每一步动作附上精确的执行时刻：`timestamp = t_0 + i * environment_dt`（真实挂钟时间），`timestep = i_0 + i`（控制循环计数）。机器人控制器用 `timestamp` 做精确定时执行，而不是"收到就执行"——这样即使网络推理时间有抖动，动作序列也能以恒定频率执行。
   - English: Stamps each action with its exact execution time: `timestamp = t_0 + i * environment_dt` (wall-clock time) and `timestep = i_0 + i` (control-loop counter). The robot controller uses `timestamp` for precise timed execution rather than "execute on receipt" — so even if GPU inference has jitter, the action sequence plays back at a steady frequency.

## 类比 / The analogy

想象一个翻译团队给联合国现场口译：演讲者说话（传感器输入），口译员飞速把一整段话翻译好（GPU 批量推理动作块），翻译稿按时间戳分发给各代表团（`TimedAction` 列表）。代表们拿到翻译稿后，即使口译员还在处理下一段，他们也知道"第 5 秒读第 3 句"——不会因为翻译速度不均匀而失步。

Picture a UN simultaneous interpreter: the speaker talks (sensor input), the interpreter translates a whole paragraph in one burst (GPU batch inference of an action chunk), and the translation is distributed to delegations with timestamps (`TimedAction` list). Even if the interpreter is still working on the next paragraph, each delegation knows exactly "read sentence 3 at second 5" — the uneven translation speed doesn't cause anyone to fall out of sync.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 nanoVLA 构建计划中 **`inference-loop`（部署外壳）** 这一模块的生产实现，属于整个 nanoVLA 的最外层接口。

依赖关系：所有其他 nanoVLA 组件都在这之前：`vision-encoder`（步骤 1 用到图像特征化）→ `modality-projector`（对齐 vision 和 language 空间）→ `language-backbone`（LLM 推理主干）→ `action-head`（输出动作分布）→ `training-loop`（得到训练好的权重）。这个 `inference-loop` 是在训练完成后，把 `policy.predict_action_chunk()` 包起来让机器人可以实际用到。

输入：一个 `TimedObservation`（带时间戳的传感器帧）。输出：`list[TimedAction]`（带执行时刻的动作序列）。如果省掉这个模块，你的模型只能在 Python 脚本里手动调用、手动处理归一化，无法接入真实机器人的实时控制循环。生产级实现还需要补充：gRPC 服务端注册（`servicer`）、观测队列去重（`_obs_sanity_checks`）、异步 worker 线程、断线重连逻辑，以及将 `TimedAction` 序列化为 protobuf 消息的编解码层。

This is the **`inference-loop` (deployment shell)** component of the nanoVLA build plan — the outermost interface of the entire system.

Dependencies: every other nanoVLA component comes before this. `vision-encoder` (Step 1 uses image featurization) → `modality-projector` (vision-language alignment) → `language-backbone` (LLM inference) → `action-head` (action distribution output) → `training-loop` (produces trained weights). The `inference-loop` takes a trained `policy.predict_action_chunk()` and wraps it so a real robot can use it.

Inputs: a `TimedObservation` (timestamped sensor frame). Outputs: `list[TimedAction]` (timestamped action sequence). Without this module your model can only be called manually in a script with hand-wired normalization — it can't plug into a real robot's real-time control loop. A production system adds: gRPC servicer registration, observation-queue deduplication (`_obs_sanity_checks`), async worker threads, reconnection logic, and protobuf serialization for `TimedAction`.

## 自己跑一遍 / Try it yourself

```python
import torch
from dataclasses import dataclass

@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: torch.Tensor

def time_action_chunk(t_0, action_chunk, i_0, environment_dt=0.02):
    return [
        TimedAction(timestamp=t_0 + i * environment_dt, timestep=i_0 + i, action=a)
        for i, a in enumerate(action_chunk)
    ]

def predict_action_chunk(obs):
    # Stub: returns (B=1, chunk_size=10, action_dim=7) tensor
    return torch.randn(1, 10, 7)

# Simulate one inference call
obs = {"image": torch.randn(1, 3, 224, 224), "joints": torch.randn(1, 7)}
chunk = predict_action_chunk(obs)
chunk = chunk[:, :5, :]                    # trim to 5 executed steps
action_list = [chunk[0, i] for i in range(chunk.shape[1])]

timed = time_action_chunk(t_0=1000.0, action_chunk=action_list, i_0=50)
for ta in timed:
    print(f"t={ta.timestamp:.3f}s  step={ta.timestep}  action_norm={ta.action.norm():.3f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
t=1000.000s  step=50  action_norm=2.xxx
t=1000.020s  step=51  action_norm=2.xxx
t=1000.040s  step=52  action_norm=2.xxx
t=1000.060s  step=53  action_norm=2.xxx
t=1000.080s  step=54  action_norm=2.xxx
```

注意时间戳以 20ms（50 Hz）为间隔递增，即使推理本身花了 100ms——控制器用这些时间戳驱动精确定时执行，而不是收到就发。

The timestamps increment in 20ms (50 Hz) steps even if inference took 100ms. The controller uses these timestamps for timed playback — the chunk acts as a prefetch buffer that absorbs inference latency.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi (`server.py`)**: openpi 的推理服务器也是 gRPC + 5 步流水线，但用 JAX/Flax 做推理、`pi0` 的 flow-matching action head 输出连续动作分布，后处理里多一个采样步骤。 / openpi's inference server uses the same gRPC + 5-step pattern but with JAX/Flax inference and pi0's flow-matching head — postprocessing adds a sampling step.
- **Isaac GR00T (`gr00t/inference/`)**:  GR00T 的服务器把 `predict_action_chunk` 换成了 diffusion policy 的 DDPM 采样循环，其余 5 步结构完全相同。 / GR00T's server replaces `predict_action_chunk` with DDPM sampling from a diffusion policy; the surrounding 5-step shell is identical.
- **ROS 2 action server**: ROS 2 中同样的解耦思路：action server 异步处理长时任务，客户端用 `feedback` 消息获取中间结果——和 gRPC + TimedAction 列表的结构如出一辙。 / The same decoupling in ROS 2: action servers handle long async tasks, clients poll via `feedback` — structurally identical to gRPC + TimedAction.

## 注意事项 / Caveats / when it breaks

- **后处理必须与训练归一化统计量一致 / Postprocessor must match training normalization stats**: 如果 `preprocessor` 和 `postprocessor` 使用的 mean/std 与训练数据集不一致，关节角度会有系统性偏移，机器人动作会持续跑偏。部署前要验证归一化统计量来源于同一 dataset config。 / If `preprocessor`/`postprocessor` use different mean/std than training, joint positions will have systematic offset. Always verify normalization stats come from the same dataset config used in training.
- **`actions_per_chunk` 截断会丢弃长程预测 / `actions_per_chunk` truncation discards long-horizon predictions**: 模型可能预测了 20 步但只有前 5 步被执行。这是有意为之（减少误差积累），但意味着你为没用到的计算付了 GPU 代价。可以通过减小模型 `chunk_size` 来避免。 / The model may predict 20 steps but only 5 get executed. This is intentional (reduces error accumulation) but wastes GPU compute on unused steps. Consider training with a smaller model chunk_size to match `actions_per_chunk`.
- **gRPC 与 50 Hz 控制循环的时间对齐 / gRPC timing vs. 50 Hz control loop**: 如果推理耗时超过 `actions_per_chunk × environment_dt`（例如推理 200ms，但只预测了 2 步 × 20ms = 40ms），控制器会耗尽动作缓冲区并停下等待。需要增大 `actions_per_chunk` 或减小推理延迟。 / If inference takes longer than `actions_per_chunk × environment_dt` (e.g., 200ms inference for 2 steps × 20ms = 40ms), the controller drains the action buffer and stalls. Fix by increasing `actions_per_chunk` or reducing inference latency.

## 延伸阅读 / Further reading

- [LeRobot async_inference 目录](https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference) — 完整的 gRPC 服务端，包含观测队列、OBS sanity check、servicer 注册。
- [openpi `server.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/server.py) — 同模式，JAX + flow-matching action head。
- [Action Chunking with Transformers (ACT)](https://arxiv.org/abs/2304.13705) — 证明动作块预测（而非单步预测）大幅降低 compounding error 的论文，`actions_per_chunk` 设计的理论基础。
