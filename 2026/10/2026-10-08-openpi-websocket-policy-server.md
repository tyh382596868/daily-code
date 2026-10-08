---
date: 2026-10-08
topic: robotics
source: tracked
repo: Physical-Intelligence/openpi
file: src/openpi/serving/websocket_policy_server.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/serving/websocket_policy_server.py#L15-L88
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, policy-serving]
---

# openpi WebSocket policy server：远端 policy 也要像本地函数一样稳 / openpi WebSocket Policy Server: Make Remote Policy Calls Feel Like Local Functions

> **一句话 / In one line**: 这个 server 把机器人观测从 WebSocket 收进来，调用同一个 `policy.infer()`，再把动作和延迟指标打包回去。 / This server receives robot observations over WebSocket, calls the same `policy.infer()`, then returns actions plus timing metadata.

## 为什么重要 / Why this matters

机器人部署时，policy 往往跑在 GPU 机器上，控制端跑在机器人或笔记本上。这里的关键不是 WebSocket 本身，而是它把网络边界收敛成一个稳定的 request/response 循环：先发 metadata，循环收 observation，调用 policy，补上 timing，再处理断连和异常。

In robot deployment, the policy often runs on a GPU box while the control client runs on the robot or an operator laptop. The important bit is not WebSocket as a buzzword; it is the stable request/response boundary: send metadata, receive observations, run policy, attach timing, and handle disconnects and failures.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/serving/websocket_policy_server.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/serving/websocket_policy_server.py#L15-L88)

```python
class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK
")
```

## 逐行讲解 / What's happening

1. **第 21-32 行 / Lines 21-32**:
   - 中文: 构造函数只保存 policy、地址和 metadata；真正的机器人逻辑仍然待在 `policy.infer()` 里，server 不偷业务逻辑。
   - English: The constructor stores the policy, bind address, and metadata. Robot intelligence stays inside `policy.infer()`; the server does not absorb policy logic.
1. **第 37-46 行 / Lines 37-46**:
   - 中文: `_server.serve` 把 `_handler` 注册成每个连接的处理器，同时关闭压缩、放开消息大小，并接上 `/healthz`。
   - English: `_server.serve` registers `_handler` per connection, disables compression, removes the message-size cap, and wires in `/healthz`.
1. **第 48-72 行 / Lines 48-72**:
   - 中文: 每轮循环做四件事：收 observation、跑推理、写入 `infer_ms` 和上一轮 total latency、发回 action。
   - English: Each loop receives an observation, runs inference, records `infer_ms` and the previous total latency, then sends the action back.
1. **第 74-83 行 / Lines 74-83**:
   - 中文: 正常断连只记录日志；真实异常先把 traceback 发给客户端，再用 internal-error close code 关闭连接。
   - English: A normal disconnect is just logged. A real exception sends the traceback first, then closes with an internal-error code.

## 类比 / The analogy

像餐厅的传菜窗口：厨师不用知道外卖小哥骑什么车，窗口只负责接单、把菜交给厨房、贴上出餐时间，再把结果递回去。

It is like a restaurant pass-through window. The chef does not need to know how the courier arrived; the window receives the order, hands it to the kitchen, stamps timing, and passes the result back.

## 自己跑一遍 / Try it yourself

```python
import time

class Policy:
    def infer(self, obs):
        return {"action": obs["x"] * 2}

prev_total = None
policy = Policy()
for obs in [{"x": 3}, {"x": 5}]:
    start = time.monotonic()
    t0 = time.monotonic()
    action = policy.infer(obs)
    action["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
    if prev_total is not None:
        action["server_timing"]["prev_total_ms"] = prev_total * 1000
    print(action)
    prev_total = time.monotonic() - start
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'action': 6, 'server_timing': {'infer_ms': ...}}
{'action': 10, 'server_timing': {'infer_ms': ..., 'prev_total_ms': ...}}
```

第二次响应才有 `prev_total_ms`，因为总耗时要等上一轮发送完成后才知道。

The second response is the first one with `prev_total_ms`, because total latency is only known after the previous send finishes.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot async policy server** / **LeRobot async policy server**: 同样把 policy 推理隔离在服务边界后面。 / It also hides policy inference behind a service boundary.
- **Triton inference server** / **Triton inference server**: 生产服务通常把模型调用、序列化、健康检查和 latency telemetry 放在同一层。 / Production serving stacks often keep model calls, serialization, health checks, and latency telemetry in one layer.

## 注意事项 / Caveats / when it breaks

- **控制频率要留余量** / **Leave control-rate headroom**: `infer_ms` 只是模型推理时间，不包含客户端传输和解码开销。 / `infer_ms` only measures policy inference, not client transport or decoding overhead.
- **异常回传别暴露生产秘密** / **Tracebacks can leak secrets**: 调试期回传 traceback 很方便，生产环境通常要脱敏或改成错误码。 / Sending tracebacks is useful during debugging, but production systems usually redact them or return structured error codes.

## 延伸阅读 / Further reading

- [openpi repository](https://github.com/Physical-Intelligence/openpi)
- [websockets server API](https://websockets.readthedocs.io/)
