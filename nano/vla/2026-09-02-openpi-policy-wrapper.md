---
date: 2026-09-02
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/policies/policy.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py#L24-L106
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, policy-wrapper, jax, pytorch]
build_role: inference-loop advanced variant
---

# openpi Policy：一份 `infer()` 同时兼容 JAX 和 PyTorch / openpi Policy: One `infer()` for Both JAX and PyTorch

> **一句话 / In one line**: 这个 wrapper 把观测变换、设备搬运、噪声注入和动作后处理都包进一个统一的 `infer()`。 / This wrapper packs observation transforms, device moves, noise injection, and action postprocessing into one unified `infer()`.

## 为什么重要 / Why this matters

中文：做 VLA 推理时，真正麻烦的常常不是模型本体，而是“前后处理怎么别把 JAX 和 PyTorch 搞成两套”。openpi 直接把这些差异收进 Policy 封装里：输入先过 transform，再按后端打包成 batch，必要时注入 noise，最后统一把动作和 timing 收回来。

English: In VLA inference, the hard part is often not the model itself but keeping JAX and PyTorch from becoming two separate codepaths. openpi folds those differences into one Policy wrapper: inputs pass through transforms, are batched per backend, optionally receive injected noise, and finally return actions plus timing in one shape.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/policies/policy.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py#L24-L106)

```python
class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
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

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
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
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
```

## 逐行讲解 / What's happening

1. **第 24-65 行 / Lines 24-65**:
   - 中文: 构造函数先把输入/输出变换链拼好，再按 `is_pytorch` 决定用 JAX 还是 PyTorch 路径。
   - English: The constructor first composes the input/output transform chains, then picks JAX or PyTorch based on `is_pytorch`.
2. **第 68-88 行 / Lines 68-88**:
   - 中文: `infer()` 会复制观测、防止原地修改，然后把 batch 维补上，并把可选噪声塞进 `sample_kwargs`。
   - English: `infer()` clones the observation to avoid in-place mutation, adds a batch dimension, and threads optional noise into `sample_kwargs`.
3. **第 90-106 行 / Lines 90-106**:
   - 中文: 真正的动作输出只占很少代码；后处理、反归一化和 timing 全都集中在 wrapper 里。
   - English: The action output itself is tiny; postprocessing, denormalization, and timing all live in the wrapper.

## 类比 / The analogy

中文：像一个机场登机口。乘客先过安检（transform），再按不同航班登机（JAX/PyTorch），落地后还要把行李标签换回本地格式（output transform）。

English: It is like an airport gate. Passengers go through security first (transforms), board different flights (JAX/PyTorch), and then get their luggage tags converted back to local format on arrival (output transforms).

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文：这是 `inference-loop` 的 advanced variant。它在你的 nanoVLA 里就像“政策外壳”：上游喂进观测，内部统一做 batch、device 和 noise 处理，下游吐出动作和 timing。省掉这层，JAX 版和 PyTorch 版会各写一遍 rollout glue，维护成本会直接翻倍；生产版通常还会再补并发排队、缓存复用、RPC/HTTP 适配和异常回退。

English: This is an advanced variant of `inference-loop`. In your nanoVLA it acts like the policy shell: observations come in upstream, batching/device/noise handling happens inside, and actions plus timing come out downstream. If you omit it, JAX and PyTorch would each need their own rollout glue, which doubles maintenance. A production version adds queueing, cache reuse, RPC/HTTP adapters, and failure fallback.

## 自己跑一遍 / Try it yourself

```python
def infer(obs, noise=None, backend="jax"):
    x = dict(obs)
    x["state"] = x["state"] * 2
    actions = [v + (noise or 0) for v in x["action_seed"]]
    return {"state": x["state"], "actions": actions, "backend": backend}

print(infer({"state": 3, "action_seed": [1, 2]}, backend="jax"))
print(infer({"state": 3, "action_seed": [1, 2]}, noise=0.5, backend="torch"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'state': 6, 'actions': [1, 2], 'backend': 'jax'}
{'state': 6, 'actions': [1.5, 2.5], 'backend': 'torch'}
```

中文：最值得注意的是，真正复杂的地方不是 `model.sample_actions`，而是把它周围的输入输出契约统一起来。

English: The important part is that the hard problem is not `model.sample_actions`; it is standardizing the input/output contract around it.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot PolicyServer** / **LeRobot PolicyServer**: 中文: 也是把观测收进统一推理接口，再吐出连续动作。 / English: It also wraps observations into one inference interface and emits continuous actions.
- **OpenVLA predict_action** / **OpenVLA predict_action**: 中文: 先解码动作 token，再统一反归一化。 / English: It decodes action tokens first, then unnormalizes them through one consistent path.

## 注意事项 / Caveats / when it breaks

- **batch 维不是可选项 / The batch dimension is mandatory**: 中文: 这层 wrapper 默认服务的是 rollout 不是单个标量。 / English: This wrapper is built for rollouts, not scalar calls.
- **PyTorch 分支要显式 `.to(device)` / PyTorch branch must call `.to(device)`**: 中文: 不然动作会在错误设备上生成。 / English: Otherwise actions may be produced on the wrong device.

## 延伸阅读 / Further reading

- openpi source: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py
