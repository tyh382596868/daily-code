---
date: 2026-05-29
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/async_inference/policy_server.py
permalink: https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/async_inference/policy_server.py#L214-L260
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference, async, action-chunk, deployment]
build_role: Inference loop — a client-server split that streams observations in and timed action chunks out
---

# 异步推理:把"慢推理"和"快控制"拆成 client-server / Async inference: split slow inference and fast control into client-server

> **一句话 / In one line**: lerobot 把 VLA 部署拆成两端 —— 机器人 client 高频推观测进 `observation_queue`,policy server 拿最新观测推一个 action chunk、给每个动作打上未来时间戳(`TimedAction`)流回去,client 按时间戳执行,推理延迟被异步吸收。 / lerobot splits VLA deployment into two ends — the robot client streams observations into an `observation_queue`, the policy server pops the latest, predicts an action chunk, timestamps each action (`TimedAction`), and streams it back; the client executes by timestamp while inference latency is absorbed asynchronously.

## 为什么重要 / Why this matters

前面 chunking 笔记解决了"少推理几次",但还有个更狠的问题:**VLA 推理那 100-500ms 期间,机器人在干嘛?** 如果同步等待,机器人会卡住不动,动作变成"走走停停"。生产 VLA 部署的标准解是**异步 client-server**:机器人 client 一边执行上一个 chunk、一边在后台请求下一个 chunk;policy server 是独立进程(甚至独立机器/GPU),专门跑推理。这样推理延迟被"边走边算"掩盖。这段代码是 lerobot 的 server 端核心 —— 它教你怎么把 chunking + 时间戳 + 队列拼成一个真正能跑在真实机器人上的推理服务。这是从"能预测动作"到"能控制机器人"的最后一公里。

The chunking note cut *how often* you infer, but there's a nastier problem: **what's the robot doing during the 100-500 ms of VLA inference?** Synchronous waiting freezes the robot into stop-and-go motion. The standard production answer is **async client-server**: the robot client executes the previous chunk *while* requesting the next one in the background; the policy server is a separate process (even a separate machine/GPU) dedicated to inference. Inference latency is hidden behind "compute while moving". This code is lerobot's server core — how to assemble chunking + timestamps + queues into an inference service that runs on a real robot. It's the last mile from "can predict actions" to "can control a robot".

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/async_inference/policy_server.py`](https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/async_inference/policy_server.py#L214-L260)

```python
def GetActions(self, request, context):
    """Returns actions to the robot client as a single chunk of multiple actions."""
    try:
        # 1) Pull the MOST RECENT observation (queue, not a blocking call)
        obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps.add(obs.get_timestep())

        # 2) Run the (slow) policy forward to get a whole action chunk
        start_time = time.perf_counter()
        action_chunk = self._predict_action_chunk(obs)        # list[TimedAction]
        inference_time = time.perf_counter() - start_time

        # 3) Serialize and stream the chunk back to the robot client
        actions_bytes = pickle.dumps(action_chunk)
        actions = services_pb2.Actions(data=actions_bytes)

        # 4) Pace the loop to a target inference latency
        time.sleep(max(0, self.config.inference_latency
                          - max(0, time.perf_counter() - getactions_starts)))
        return actions

    except Empty:   # no observation arrived within the timeout
        return services_pb2.Empty()
```

And the crucial timestamping that makes async work — `_time_action_chunk`:

```python
def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int):
    """Attach FUTURE timestamps t_0 + i*environment_dt to each action in the chunk,
    so the client knows WHEN to execute each one."""
    return [
        TimedAction(timestamp=t_0 + i * self.config.environment_dt,
                    action=action, timestep=i_0 + i)
        for i, action in enumerate(action_chunk)
    ]
```

## 逐行讲解 / What's happening

1. **`observation_queue.get(timeout=...)` / Pull the latest observation**:
   - 中文:server 不是被动等单次请求,而是从一个观测队列里取**最新**的观测。client 高频往队列塞观测(`SendObservations` 是另一个 stream),server 推理时只取队头最新那个 —— 旧观测自然丢弃,保证推理基于最新世界状态。
   - English: the server doesn't wait for a single request — it pops the *latest* observation from a queue. The client streams observations in at high frequency (`SendObservations` is a separate stream); the server takes only the freshest, naturally dropping stale ones so inference reflects the current world state.

2. **`_predicted_timesteps` 加锁 / Locked timestep set**:
   - 中文:`_predicted_timesteps_lock` 保护一个"已经为哪些 timestep 预测过"的集合。因为 server 是多线程的(gRPC),要防止对同一观测重复推理。这是并发服务的细节,但暴露了"异步"的本质 —— 多个请求可能同时在飞。
   - English: `_predicted_timesteps_lock` guards the set of "which timesteps we've already predicted for". Since the server is multithreaded (gRPC), it prevents duplicate inference on the same observation. A concurrency detail, but it reveals the async nature — multiple requests may be in flight.

3. **`_predict_action_chunk(obs)` / The slow part, isolated**:
   - 中文:这一行是唯一慢的部分(VLA 前向 + flow matching 积分 / token 解码)。把它隔离在 server 端意味着机器人 client 完全不被它阻塞 —— client 那边正在执行上一个 chunk。
   - English: this is the only slow line (VLA forward + flow-matching integration / token decode). Isolating it on the server means the robot client is never blocked by it — the client is busy executing the previous chunk.

4. **`pickle.dumps(action_chunk)` / Serialize for the wire**:
   - 中文:action chunk 序列化成 bytes 通过 gRPC 传回。chunk 里是 `TimedAction` 列表,每个带时间戳。
   - English: the chunk is serialized to bytes and sent back over gRPC. The chunk is a list of `TimedAction`, each with a timestamp.

5. **`_time_action_chunk` 打未来时间戳 / Future timestamps are the magic**:
   - 中文:这是异步能 work 的**关键**。每个动作被标上 `t_0 + i * environment_dt` —— 即"这个动作应该在未来第 i 个控制周期执行"。client 收到一整个带时间戳的 chunk 后,按墙钟时间到点执行对应动作。推理花了多久不重要,只要在动作该执行的时间点之前算完就行。
   - English: this is **the key** to async working. Each action is stamped `t_0 + i * environment_dt` — "execute this action at the i-th future control cycle". The client receives the timestamped chunk and fires each action when wall-clock time reaches its stamp. How long inference took doesn't matter, as long as it finished before the action's scheduled time.

6. **`time.sleep(inference_latency - elapsed)` / Pace the loop**:
   - 中文:server 故意 sleep 把每次循环对齐到一个固定的 `inference_latency`,避免忙等浪费 GPU、也让节奏可预测。这是工程上的节流。
   - English: the server deliberately sleeps to align each loop to a fixed `inference_latency`, avoiding busy-waiting that wastes GPU and keeping the cadence predictable. An engineering throttle.

7. **`except Empty` / Graceful starvation**:
   - 中文:如果 `obs_queue_timeout` 内没有新观测(client 卡了),返回空而不是崩。鲁棒的服务必须容忍上游断流。
   - English: if no fresh observation arrives within `obs_queue_timeout` (client stalled), return empty rather than crash. A robust service tolerates upstream starvation.

## 类比 / The analogy

像餐厅后厨和传菜员的关系。传菜员(client)不停把新点单(观测)贴到出菜口,大厨(server)看最新一张单子做一整套菜(action chunk),每盘菜贴上"几点上桌"的标签(时间戳),一次性推出窗口。传菜员按标签时间端菜上桌。大厨做菜慢没关系 —— 只要在那盘菜该上桌前做完,客人(机器人)感觉是连续流畅的上菜,而不是"等一道做一道"的卡顿。

Like the relationship between a kitchen and a runner. The runner (client) keeps posting new orders (observations) at the pass; the chef (server) reads the latest ticket and cooks a whole course (action chunk), labelling each plate with a "serve at" time (timestamp), pushing them all out at once. The runner serves each plate when its time arrives. The chef being slow is fine — as long as each plate is ready before its serve time, the diner (robot) experiences smooth continuous service, not stop-and-go "wait for each dish".

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:在 nanoVLA 里这是 `nano/vla/runtime/policy_server.py` + `robot_client.py` —— 整个部署的运行时,是 nanoVLA 课程图的**最后一个节点**(依赖前面所有:vision encoder、VLM backbone、action head、chunking 全部就位才有东西可 serve)。上游:训练好并存了 dataset statistics 的 nanoVLA 模型;下游:真实机器人的电机控制器。如果省掉异步、用同步推理循环:机器人会在每次推理时卡住 100-500ms,动作断断续续,精细任务直接失败。这一层是"实验室 demo"到"真机部署"的分界。生产实现要补:(1) **chunk 拼接策略**(新 chunk 到了,旧 chunk 还没执行完,怎么平滑切换 —— 通常丢弃旧 chunk 未来部分);(2) **观测-动作时间对齐**(动作时间戳和机器人时钟要同步,否则执行错位);(3) **网络抖动容错**(chunk 没及时到,要 fallback 到旧 chunk 或安全停止);(4) **多 GPU 批处理**(一个 server 同时服务多个机器人时把请求 batch 起来)。

English: in nanoVLA this is `nano/vla/runtime/policy_server.py` + `robot_client.py` — the deployment runtime, the **final node** of the nanoVLA build graph (it depends on everything: vision encoder, VLM backbone, action head, chunking all in place before there's anything to serve). Upstream: a trained nanoVLA with saved dataset statistics. Downstream: the real robot's motor controller. Skip async and use a synchronous loop: the robot freezes for 100-500 ms per inference, motion stutters, fine tasks fail. This layer is the boundary between "lab demo" and "real-robot deployment". Production additions: (1) **chunk-splice strategy** (a new chunk arrives before the old one finishes — usually discard the old chunk's future portion), (2) **observation-action clock alignment** (action timestamps must sync to the robot clock or execution misaligns), (3) **network-jitter tolerance** (a late chunk needs a fallback to the old chunk or a safe stop), (4) **multi-GPU batching** (one server serving several robots batches requests).

## 自己跑一遍 / Try it yourself

```python
# pip install (stdlib only)
import time, threading, queue, random

environment_dt = 0.02                          # 50 Hz control
obs_queue = queue.Queue()
action_buffer = []                             # timed actions the client will execute

def policy_server():
    """Pops latest obs, 'infers' a chunk (slow!), stamps future times, returns."""
    while True:
        try:
            t0, obs = obs_queue.get(timeout=0.5)
        except queue.Empty:
            return
        time.sleep(random.uniform(0.05, 0.15))     # simulate 50-150ms inference
        chunk = [f"act@step{obs}+{i}" for i in range(8)]   # 8-step chunk
        timed = [(t0 + i * environment_dt, a) for i, a in enumerate(chunk)]
        action_buffer.extend(timed)

# client streams a few observations, server runs async in a thread
server = threading.Thread(target=policy_server)
server.start()
for step in range(3):
    obs_queue.put((time.perf_counter(), step))
    time.sleep(0.1)                            # client keeps moving meanwhile
time.sleep(0.5); obs_queue.put(None) if False else None
server.join(timeout=1)

print(f"actions buffered while 'moving': {len(action_buffer)}")
print("first 3 timed actions:")
for ts, a in action_buffer[:3]:
    print(f"  exec_at={ts:.3f}  action={a}")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
actions buffered while 'moving': 24       # 3 chunks x 8 actions
first 3 timed actions:
  exec_at=...  action=act@step0+0
  exec_at=...  action=act@step0+1
  exec_at=...  action=act@step0+2
```

中文:client 在 0.1s 间隔里持续推观测,server 在后台异步推理并把带未来时间戳的动作塞进 buffer。关键观察:推理延迟(50-150ms 随机)完全没阻塞 client 的循环。

English: the client streams observations every 0.1 s while the server infers asynchronously and fills the buffer with future-timestamped actions. Key observation: the variable inference latency (50-150 ms) never blocks the client loop.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **π₀ / GR00T 部署栈** / **π₀ / GR00T deployment stacks**: 中文 — 同样的 client-server + action chunk streaming,openpi 的 `websocket_policy_server` 几乎一模一样。 / English — same client-server + action-chunk streaming; openpi's `websocket_policy_server` is nearly identical.
- **今天的 action chunking 笔记** / **Today's action chunking note**: 中文 — chunking 是这个异步服务的前提:没有 chunk,异步就没有"提前算好的动作"可缓冲。两篇配对看。 / English — chunking is the prerequisite: without a chunk, async has no "pre-computed actions" to buffer. Read them paired.
- **自动驾驶的 planning-control 分层** / **Autonomous-driving planning-control split**: 中文 — 高频控制器执行低频规划器的轨迹,完全同构。 / English — a high-frequency controller executing a low-frequency planner's trajectory; structurally identical.
- **游戏服务器的 client-side prediction** / **Game-server client-side prediction**: 中文 — 远亲:用时间戳和缓冲掩盖网络/计算延迟。 / English — a distant cousin: timestamps and buffering hide network/compute latency.

## 注意事项 / Caveats / when it breaks

- **时间戳必须和机器人时钟同步** / **Timestamps must sync to the robot clock**: 中文 — `t_0 + i*dt` 里的 `t_0` 要用机器人执行端的时钟基准,server 和 client 时钟不同步会让动作执行错位。 / English — `t_0` in `t_0 + i*dt` must use the robot's execution clock base; clock skew between server and client misaligns execution.
- **新旧 chunk 切换要平滑** / **Splice old and new chunks carefully**: 中文 — 新 chunk 到达时旧 chunk 可能还剩几步没执行,直接切换会跳变。常见做法:丢弃旧 chunk 未来部分,或在重叠区加权。 / English — when a new chunk arrives, the old one may have steps left; a hard switch jumps. Common fix: discard the old chunk's future or weight the overlap.
- **观测队列要丢旧的** / **Drop stale observations**: 中文 — server 必须取最新观测,否则基于过期世界状态预测,机器人会"追着影子动"。 / English — the server must take the freshest observation, or it predicts on a stale world state and the robot chases shadows.
- **网络断流的安全行为** / **Define safe behaviour on stream loss**: 中文 — chunk 没及时到时,机器人要有 fallback(保持当前姿态 / 安全停止),不能盲目执行过期动作。 / English — if a chunk is late, the robot needs a fallback (hold pose / safe stop), never blindly executing stale actions.

## 延伸阅读 / Further reading

- [lerobot async inference](https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference)
- [Real-Time Chunking for VLAs (π₀ RTC)](https://www.physicalintelligence.company/research/real_time_chunking)
- [Today's action chunking note](./2026-05-29-act-action-chunking.md)
- [Today's VLA action survey doc](./README-action-survey.md)
