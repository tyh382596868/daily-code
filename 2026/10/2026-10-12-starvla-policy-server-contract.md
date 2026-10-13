---
date: 2026-10-12
topic: robotics
source: tracked
repo: starVLA/starVLA
file: deployment/model_server/server_policy.py
permalink: https://github.com/starVLA/starVLA/blob/985c2238627481f25476fbee478083812dd82a21/deployment/model_server/server_policy.py#L15-L77
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, policy-server, deployment-contract]
---

# StarVLA policy server：把训练契约带到推理入口 / StarVLA Policy Server: Carry the Training Contract into Serving

> **一句话 / In one line**: 这个入口在启动 websocket policy 前，把 seed、config override、action metadata 和训练/测试一致性检查都放到同一个边界里。 / This entry point starts a websocket policy only after binding seed, config overrides, action metadata, and train/test consistency checks to one serving boundary.

## 为什么重要 / Why this matters

VLA 部署最怕的 bug 往往不是模型崩溃，而是模型“正常返回错误动作”：相机顺序换了、state 维度变了、unnorm key 用错了，代码还能跑，但成功率悄悄掉下去。StarVLA 这段 server 入口的价值在于，它把这些隐性契约变成启动时日志和 handshake metadata，让 client 有机会在第一拍之前检查自己是否按训练时的方式喂数据。

The dangerous VLA deployment failures are often silent. The model still returns actions, but camera order, state layout, or unnormalization stats no longer match training. This server entry point makes those hidden contracts visible as startup logs and handshake metadata, so clients can verify the interface before the first control step.

## 代码 / The code

`starVLA/starVLA` - [`deployment/model_server/server_policy.py`](https://github.com/starVLA/starVLA/blob/985c2238627481f25476fbee478083812dd82a21/deployment/model_server/server_policy.py#L15-L77)

```python
def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    seed = getattr(args, "seed", None)
    if not isinstance(seed, int):
        seed = None
    if seed is not None:
        set_seed_everywhere(seed)

    config_overrides = getattr(args, "config_override", [])
    if config_overrides:
        override_keys = [item.split("=", 1)[0] for item in config_overrides]
        logging.info("Applying config override keys: %s", override_keys)
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        config_overrides=config_overrides,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # =========================================================================
    # !!! TRAIN / TEST CONSISTENCY — READ BEFORE SERVING !!!
    # -------------------------------------------------------------------------
    # This server replays the *training-time* observation contract. The eval
    # client MUST feed observations exactly as the model saw them at TRAIN time,
    # otherwise the success rate silently drops (no error is raised):
    #   - state   : is proprioceptive state used? (use_state) and its dim/order
    #   - img size: resize / crop resolution (e.g. 224x224)
    #   - img num : how many camera views are fed
    #   - img order: the ordering of those camera views
    #   - action normalization: unnorm_key / dataset stats must match training
    # `wrapper.metadata` (logged below and sent at handshake) exposes
    # action_chunk_size / state_keys / action_keys — cross-check these against
    # the training config used to produce `args.ckpt_path`.
    # =========================================================================
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s — verify eval observations "
        "(state / image size / image count / image order / action normalization) match "
        "the training config. metadata=%s",
        args.ckpt_path,
        wrapper.metadata,
    )

    # start websocket server; wrapper.metadata is sent at handshake.
    metadata = dict(wrapper.metadata)
    metadata["seed"] = seed
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("server running ... metadata=%s", metadata)
    server.serve_forever()
```

## 逐行讲解 / What's happening

1. **第 22-27 行 / Lines 22-27 (`seed`)**:
   - 中文: 入口只接受真正的 `int` seed，其他值都当作 `None`，避免字符串 `"0"` 这种 CLI 假值误导复现实验。
   - English: The entry point accepts only a real `int` seed. Other values become `None`, which avoids accidental reproducibility claims from CLI-shaped values like `"0"`.

2. **第 28-37 行 / Lines 28-37 (`PolicyServerWrapper`)**:
   - 中文: config override 在构造 wrapper 前记录 key，wrapper 则统一负责 checkpoint、device、bf16 和配置改写。
   - English: Override keys are logged before wrapper construction. The wrapper then owns checkpoint loading, device choice, bf16 mode, and config edits as one object.

3. **第 43-64 行 / Lines 43-64 (consistency warning)**:
   - 中文: 这里不是普通注释，它把训练时 observation 契约列成一张检查清单。VLA client 必须确认 state、图像尺寸、相机数量、相机顺序和 action normalization 都没有漂移。
   - English: This is more than a comment. It is a checklist for the training-time observation contract: state layout, image size, camera count, camera order, and action normalization must not drift.

4. **第 66-75 行 / Lines 66-75 (`metadata`)**:
   - 中文: server 把 `wrapper.metadata` 拷贝出来并加上 seed，交给 websocket server。这样 client 可以在握手阶段拿到 action chunk size、state keys、action keys。
   - English: The server copies `wrapper.metadata`, adds the seed, and passes it into the websocket layer. The client can read action chunk size, state keys, and action keys during handshake.

## 类比 / The analogy

这像机场登机口的“行李尺寸框”。飞机当然能飞，但如果登机口不先确认箱子尺寸，到了舱门才发现放不进去就晚了。这里的 metadata 就是那个尺寸框，先把数据契约量一遍。

It is like the baggage-size frame at an airport gate. The plane can fly either way, but discovering at the cabin door that the bag does not fit is too late. The metadata is that frame for the serving data contract.

## 自己跑一遍 / Try it yourself

```python
def build_handshake(wrapper_meta, seed):
    required = {"action_chunk_size", "state_keys", "action_keys"}
    missing = required - wrapper_meta.keys()
    if missing:
        raise ValueError(f"metadata missing: {sorted(missing)}")
    metadata = dict(wrapper_meta)
    metadata["seed"] = seed if isinstance(seed, int) else None
    return metadata

meta = build_handshake(
    {"action_chunk_size": 8, "state_keys": ["joint"], "action_keys": ["arm"]},
    seed="not-an-int",
)
print(meta)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'action_chunk_size': 8, 'state_keys': ['joint'], 'action_keys': ['arm'], 'seed': None}
```

注意 seed 被清成了 `None`，但 metadata 的关键字段保留下来。这个小例子复刻的是“服务端先暴露契约，再服务动作”的思路。

Notice that the seed becomes `None`, while the important metadata survives. The small example mirrors the serving pattern: expose the contract before serving actions.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi policy server** / **openpi policy server**: `openpi` 也把远程 policy 包成像本地函数一样的接口，重点是让 client 不关心模型进程在哪里。 / `openpi` also wraps remote policies so the client calls them like local functions, hiding where the model process lives.
- **LeRobot async inference** / **LeRobot async inference**: LeRobot 的异步 server 更强调请求排队和推理节拍，StarVLA 这段则更强调训练/测试契约。 / LeRobot's async server focuses on request queues and inference cadence, while this StarVLA entry point focuses on the train/test contract.

## 注意事项 / Caveats / when it breaks

- **metadata 只是提醒，不是强校验** / **Metadata warns, it does not enforce**: client 如果忽略 handshake，仍然可能喂错相机顺序。生产系统可以把 metadata 做成硬 schema 校验。 / A client can still ignore the handshake and feed cameras in the wrong order. A production system can turn metadata into a hard schema check.
- **override key 只记录 key，不记录值** / **Override logging records keys, not values**: 这能减少日志泄漏，但排查配置差异时还需要可控的完整 config dump。 / This reduces log leakage, but debugging config drift still needs a controlled full config dump.

## 延伸阅读 / Further reading

- [StarVLA repository](https://github.com/starVLA/starVLA)
- [Source permalink](https://github.com/starVLA/starVLA/blob/985c2238627481f25476fbee478083812dd82a21/deployment/model_server/server_policy.py#L15-L77)

