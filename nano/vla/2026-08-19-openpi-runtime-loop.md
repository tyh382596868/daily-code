---
date: 2026-08-19
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: packages/openpi-client/src/openpi_client/runtime/runtime.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/runtime/runtime.py#L32-L92
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, runtime]
build_role: inference-loop advanced variant
---

# openpi Runtime loop：观测、决策、执行、记录一拍完成 / openpi Runtime Loop: Observe, Decide, Act, Log on Every Tick

> **一句话 / In one line**: openpi 把部署侧控制循环写成固定合同：reset episode，按频率 step，读 observation，问 agent，apply action，再通知 subscribers。 / openpi turns deployment control into a fixed contract: reset the episode, step at a target rate, read observation, query the agent, apply action, then notify subscribers.

## 为什么重要 / Why this matters

VLA 真正部署时，模型 forward 只是闭环中的一段。你还要管 episode 生命周期、机器人 reset、控制频率、日志/可视化订阅、最大步数和环境自己宣布结束。`Runtime` 把这些边界集中到一个小类里，让 policy、environment 和 subscriber 都只实现自己的接口。

When a VLA is deployed, model forward is only one part of the closed loop. You also need episode lifecycle, robot reset, control frequency, logging/visualization subscribers, max-step limits, and environment termination. `Runtime` centralizes those boundaries in one small class so policy, environment, and subscribers each implement only their own interface.

## 代码 / The code

`Physical-Intelligence/openpi` — [`packages/openpi-client/src/openpi_client/runtime/runtime.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/runtime/runtime.py#L32-L92)

```python
def run(self) -> None:
    """Runs the runtime loop continuously until stop() is called or the environment is done."""
    for _ in range(self._num_episodes):
        self._run_episode()

    # Final reset, this is important for real environments to move the robot to its home position.
    self._environment.reset()

def run_in_new_thread(self) -> threading.Thread:
    """Runs the runtime loop in a new thread."""
    thread = threading.Thread(target=self.run)
    thread.start()
    return thread

def mark_episode_complete(self) -> None:
    """Marks the end of an episode."""
    self._in_episode = False

def _run_episode(self) -> None:
    """Runs a single episode."""
    logging.info("Starting episode...")
    self._environment.reset()
    self._agent.reset()
    for subscriber in self._subscribers:
        subscriber.on_episode_start()

    self._in_episode = True
    self._episode_steps = 0
    step_time = 1 / self._max_hz if self._max_hz > 0 else 0
    last_step_time = time.time()

    while self._in_episode:
        self._step()
        self._episode_steps += 1

        # Sleep to maintain the desired frame rate
        now = time.time()
        dt = now - last_step_time
        if dt < step_time:
            time.sleep(step_time - dt)
            last_step_time = time.time()
        else:
            last_step_time = now

    logging.info("Episode completed.")
    for subscriber in self._subscribers:
        subscriber.on_episode_end()

def _step(self) -> None:
    """A single step of the runtime loop."""
    observation = self._environment.get_observation()
    action = self._agent.get_action(observation)
    self._environment.apply_action(action)

    for subscriber in self._subscribers:
        subscriber.on_step(observation, action)

    if self._environment.is_episode_complete() or (
        self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
    ):
        self.mark_episode_complete()
```

## 逐行讲解 / What's happening

1. **第 32-38 行 / Lines 32-38 (`run`)**:
   - 中文: 连跑多个 episode，最后再 reset 一次，把真实机器人带回安全初始状态。
   - English: The runtime runs multiple episodes and performs a final reset to return a real robot to a safe home state.
2. **第 50-60 行 / Lines 50-60 (episode start)**:
   - 中文: 环境、agent、subscriber 都收到 episode 边界，不把 reset 逻辑散落到 policy 内部。
   - English: Environment, agent, and subscribers all see the episode boundary, so reset logic does not leak into the policy.
3. **第 63-74 行 / Lines 63-74 (Hz control)**:
   - 中文: 每步后按 `max_hz` 补 sleep，慢了就不追赶，只更新时间戳。
   - English: After each step, the loop sleeps to respect `max_hz`; if the step was slow, it does not try to catch up.
4. **第 80-92 行 / Lines 80-92 (`_step`)**:
   - 中文: 单步合同非常清楚：观测 -> 决策 -> 执行动作 -> 通知订阅者 -> 检查结束。
   - English: The single-step contract is explicit: observe -> decide -> act -> notify subscribers -> check termination.

## 类比 / The analogy

像一个实验室节拍器。每一拍先读传感器，再让研究员决定动作，再让机械臂执行，旁边的记录员同步记日志；到达次数或实验结束信号后，节拍器停下。

It is like a lab metronome. On each tick it reads sensors, asks the researcher for an action, commands the robot arm, and lets a logger record the step. When the trial ends or the max count is reached, the metronome stops.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoVLA 里，这是 `inference-loop` 的部署外壳。上游是环境 observation provider，下游是真实 robot API 或 simulator；中间的 agent 可以包本地 policy、远端 websocket policy 或 action chunk broker。省掉这个 runtime，你会把 reset、限速、日志和终止条件塞进 policy，导致训练和部署接口混在一起。生产级还要补异常恢复、硬实时控制、异步 sensor timestamp 对齐和安全急停。

In a nanoVLA, this is the deployment shell for the `inference-loop`. Upstream is the observation provider; downstream is a real robot API or simulator; the agent can wrap a local policy, remote websocket policy, or action chunk broker. Without this runtime, reset, rate limiting, logging, and termination logic drift into the policy and mix training with deployment. A production version needs error recovery, hard real-time control, async sensor timestamp alignment, and emergency stop handling.

## 自己跑一遍 / Try it yourself

```python
class Env:
    def __init__(self): self.t = 0
    def reset(self): self.t = 0
    def get_observation(self): return {"t": self.t}
    def apply_action(self, action): self.t += action["dt"]
    def is_episode_complete(self): return self.t >= 3

class Agent:
    def reset(self): pass
    def get_action(self, obs): return {"dt": 1}

env, agent, log = Env(), Agent(), []
env.reset(); agent.reset()
while not env.is_episode_complete():
    obs = env.get_observation()
    action = agent.get_action(obs)
    env.apply_action(action)
    log.append((obs, action))
print(log)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[({'t': 0}, {'dt': 1}), ({'t': 1}, {'dt': 1}), ({'t': 2}, {'dt': 1})]
```

中文: 最小闭环里，policy 不知道 episode 何时结束；环境负责终止条件，runtime 负责调度。

English: In the minimal loop, the policy does not own termination; the environment owns the stop condition and the runtime owns orchestration.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Gymnasium loop** / **Gymnasium loop**: `obs -> action -> env.step(action)` 是同一个核心合同。 / The classic `obs -> action -> env.step(action)` loop is the same core contract.
- **LeRobot eval rollout** / **LeRobot eval rollout**: 评测时也需要把 env、policy、processor 和 recorder 接成闭环。 / Evaluation also connects env, policy, processors, and recorders into a closed loop.

## 注意事项 / Caveats / when it breaks

- **`time.sleep` 不是硬实时** / **`time.sleep` is not hard real time**: 桌面系统调度会抖动，真实高速控制要下沉到实时控制器。 / Desktop scheduling jitters; high-speed robot control should move into a real-time controller.
- **慢 step 不补偿** / **Slow steps are not compensated**: 如果 policy forward 超时，循环不会追赶丢掉的 tick。 / If policy forward is too slow, the loop does not catch up missed ticks.

## 延伸阅读 / Further reading

- [openpi runtime loop source](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/runtime/runtime.py#L32-L92)
- [openpi client runtime package](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/runtime)
