---
date: 2026-08-14
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: packages/openpi-client/src/openpi_client/websocket_client_policy.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L12-L58
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop]
build_role: inference-loop advanced variant, deployment-side policy adapter over websocket
---

# openpi WebsocketClientPolicy：把远端服务伪装成本地 policy / openpi WebsocketClientPolicy: Make a Remote Server Look Like a Local Policy

> **一句话 / In one line**: `WebsocketClientPolicy` 实现同一个 `BasePolicy` 接口，但把 `infer(obs)` 序列化后发给 websocket server，再把返回 bytes 解包成动作。 / `WebsocketClientPolicy` implements the same `BasePolicy` interface, but serializes `infer(obs)` over a websocket server and unpacks the returned bytes into actions.

## 为什么重要 / Why this matters

生产 VLA 往往不会和机器人控制进程跑在同一个 Python 解释器里。这个 adapter 把“远端模型服务”藏在 `BasePolicy` 后面，让上层控制器仍然只调用 `infer(obs)`，部署形态可以从本地模型切到远端 GPU 服务。

Production VLAs often do not run in the same Python interpreter as the robot control loop. This adapter hides a remote model server behind `BasePolicy`, so the controller still calls `infer(obs)` while deployment can move from an in-process model to a remote GPU service.

## 代码 / The code

`Physical-Intelligence/openpi` — [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L12-L58)

```python
class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
```

## 逐行讲解 / What's happening

1. **第 18-27 行 / Lines 18-27**: 中文: 构造函数规范化 URI，准备 msgpack packer，并在初始化时等待服务端握手。 / English: The constructor normalizes the URI, prepares a msgpack packer, and waits for the server handshake at initialization.
2. **第 32-44 行 / Lines 32-44**: 中文: 连接失败不会立刻崩，而是每 5 秒重试；连接成功后第一帧消息就是 server metadata。 / English: Connection refusal triggers a retry every five seconds; after connection, the first server message becomes metadata.
3. **第 47-54 行 / Lines 47-54**: 中文: `infer` 的合同没变：输入 observation dict，输出 action dict，只是中间跨了 websocket。 / English: The `infer` contract stays the same: observation dict in, action dict out, with a websocket hop in the middle.
4. **第 51-53 行 / Lines 51-53**: 中文: 如果服务端返回字符串，客户端把它当错误文本处理，而不是误解包成动作。 / English: If the server returns a string, the client treats it as an error message rather than unpacking it as actions.

## 类比 / The analogy

像餐厅的点餐窗口：顾客还是说“我要这份套餐”，但厨房可以在墙后很远的地方。窗口保证菜单和取餐格式不变。

It is like a takeout window: the customer still orders the same meal, but the kitchen may be far behind the wall. The window keeps the menu and pickup format stable.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoVLA 里，这属于 `inference-loop` 的部署层 adapter。上游是机器人控制器给出的 `obs`，下游是本地或远端 policy server 产生的 action。nano 版本可以先直接调用本地模型；一旦模型太大、需要独占 GPU、或要多机器人共享服务，就需要这种 transport adapter。

In a nanoVLA, this is a deployment adapter inside the `inference-loop`. Upstream is the robot controller's `obs`; downstream is an action from a local or remote policy server. A minimal build can call the model in-process first, but once the model is large, GPU-bound, or shared by multiple robots, this transport adapter becomes the clean boundary.

## 自己跑一遍 / Try it yourself

```python
class FakeSocket:
    def __init__(self):
        self.last = None
    def send(self, data):
        self.last = data
    def recv(self):
        return {"actions": [1, 2, 3], "saw": self.last}

class ClientPolicy:
    def __init__(self, ws):
        self.ws = ws
    def infer(self, obs):
        self.ws.send(dict(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return response

print(ClientPolicy(FakeSocket()).infer({"image": "frame0"}))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'actions': [1, 2, 3], 'saw': {'image': 'frame0'}}
```

调用者只看见 `infer`，不用知道动作来自本地模型还是远端服务。

The caller only sees `infer`; it does not need to know whether actions came from a local model or a remote service.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot async policy server** / **LeRobot async policy server**: 同样把推理节拍和机器人控制节拍隔离开。 / It also separates model inference timing from robot control timing.
- **vLLM / model serving** / **vLLM / model serving**: 大模型部署常把模型进程放到独立服务里，客户端只保留稳定 API。 / Large-model deployment often isolates the model process behind a stable client API.

## 注意事项 / Caveats / when it breaks

- **reset 是空实现** / **`reset` is a no-op**: 如果服务端有 episode 状态，生产实现需要把 reset 也发过去。 / If the server owns episode state, a production client should forward reset too.
- **网络延迟进控制环** / **Network latency enters the control loop**: 高频控制必须配合 action chunk、缓存或超时策略。 / High-rate control needs action chunks, caching, or timeout policies.

## 延伸阅读 / Further reading

- [Physical-Intelligence/openpi source](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L12-L58)
