---
date: 2026-07-13
topic: robotics
source: tracked
repo: Physical-Intelligence/openpi
file: src/openpi/policies/policy.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/policies/policy.py#L18-L115
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, policy-inference]
---

# openpi Policy：同一个 infer 包住 JAX 和 PyTorch / openpi Policy: One infer Wrapper for JAX and PyTorch

> **一句话 / In one line**: openpi 把输入变换、batch 维、采样后处理和计时都包进 `Policy.infer`，让 JAX/PyTorch 模型暴露同一个机器人推理接口。 / openpi wraps transforms, batching, sampling, post-processing, and timing inside `Policy.infer`, giving JAX and PyTorch policies the same robot-facing API.

## 为什么重要 / Why this matters

机器人策略部署时，模型框架不应该泄漏到控制循环里。控制端只想给一份 observation，拿一段 action；至于是 JAX module 还是 PyTorch module、输入要不要转 tensor、输出要不要搬回 CPU，都应该藏在 policy wrapper 后面。

In robot deployment, the control loop should not care which framework trained the model. It should pass an observation and receive actions; conversion to JAX arrays or PyTorch tensors, RNG handling, device movement, and output transforms belong behind the policy wrapper.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/policies/policy.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/policies/policy.py#L18-L115)

```python
class Policy(BasePolicy):
    def __init__(self, model, *, rng=None, transforms=(), output_transforms=(),
                 sample_kwargs=None, metadata=None, pytorch_device="cpu", is_pytorch=False):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)
```

## 逐行讲解 / What's happening

1. **组合输入输出变换 / Compose transforms**:
   - 中文: `compose(transforms)` 把归一化、重命名、相机预处理等步骤压成一个函数。
   - English: `compose(transforms)` turns normalization, key remapping, and camera preprocessing into one callable.
2. **框架分支只出现一次 / Framework branching is localized**:
   - 中文: 初始化时决定 JAX 走 `module_jit`，PyTorch 走 `.eval()` 和 device placement。
   - English: Initialization decides whether to JIT the JAX sampler or place a PyTorch model on the target device.
3. **推理时统一加 batch 维 / Add a batch dimension uniformly**:
   - 中文: 单环境 observation 进入模型前变成 batch size 1，输出再把第 0 维去掉。
   - English: A single-environment observation becomes a batch of one before sampling, then the wrapper removes that dimension from the output.
4. **可选 noise 进入 sample_kwargs / Optional noise joins sample kwargs**:
   - 中文: flow matching 或 diffusion policy 可以显式传入噪声，方便复现实验或做调试。
   - English: Flow-matching or diffusion policies can receive explicit noise for reproducibility and debugging.

## 类比 / The analogy

像万能充电器：墙上可能是不同插座，设备可能要不同电压，但使用者只看见一个 USB-C 口。`Policy.infer` 就是机器人控制循环前面的那只适配器。

Think of a universal charger: the wall outlet and voltage conversion vary, but the user sees one USB-C port. `Policy.infer` is that adapter in front of the robot control loop.

## 自己跑一遍 / Try it yourself

```python
def compose(fs):
    def run(x):
        for f in fs:
            x = f(x)
        return x
    return run

class TinyPolicy:
    def __init__(self):
        self.in_tf = compose([lambda o: {**o, "state": [v / 10 for v in o["state"]]}])
        self.out_tf = compose([lambda o: {**o, "actions": [a * 100 for a in o["actions"]]}])
    def infer(self, obs):
        x = self.in_tf(obs)
        y = {"state": x["state"], "actions": [sum(x["state"])]}
        return self.out_tf(y)

print(TinyPolicy().infer({"state": [2, 4, 6]}))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'state': [0.2, 0.4, 0.6], 'actions': [120.0]}
```

这个小例子保留了核心结构：输入先标准化，模型只看标准形状，输出再变回控制端需要的尺度。

The toy version keeps the core contract: normalize inputs, let the model see a standard shape, then convert outputs back to actuator scale.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot `PreTrainedPolicy`** / **LeRobot `PreTrainedPolicy`**: 同样把 `select_action` 当作机器人侧稳定入口。 / It similarly exposes `select_action` as the stable robot-facing entry point.
- **RL serving systems** / **RL serving systems**: 通常也会把 observation normalization 和 action denormalization 固定在 policy wrapper 里。 / They usually keep observation normalization and action denormalization inside the policy wrapper.

## 注意事项 / Caveats / when it breaks

- **transform 会改输入** / **transforms may mutate inputs**: 源码先复制 tree，是为了避免 transform 原地改坏调用方的数据。 / The source copies the tree first so transforms do not mutate caller-owned observations.
- **同步计时不等于端到端延迟** / **model timing is not full latency**: 真机还要算传感器采集、网络和控制器执行时间。 / Real robot latency also includes sensors, transport, and controller execution.

## 延伸阅读 / Further reading

- openpi policy source — https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/policies/policy.py

