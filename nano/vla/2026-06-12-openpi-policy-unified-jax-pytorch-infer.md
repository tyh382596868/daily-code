---
date: 2026-06-12
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/policies/policy.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/policies/policy.py#L24-L106
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, jax-pytorch-bridge, openpi]
build_role: inference-loop (cross-repo variant — the unified rollout wrapper)
---

# 同一份 `infer()` 跑 JAX 和 PyTorch 两套 VLA:openpi 的 80 行统一推理器 / One `infer()` for both JAX and PyTorch VLAs: openpi's 80-line unified rollout wrapper

> **一句话 / In one line**: `jax.tree.map` 在 torch tensor 上也能用,所以 openpi 用一个 `Policy` 类把 "JIT'd JAX 模型" 和 "eager PyTorch 模型" 的输入预处理、batch/unbatch、设备搬运、noise 注入、timing 全部统一成同一段 80 行代码 / `jax.tree.map` works on torch tensors too — so openpi packs a JIT'd JAX model and an eager PyTorch model into one `Policy` class whose 80-line `infer()` handles input transforms, batch/unbatch, device transfer, noise injection, and timing identically for both.

## 为什么重要 / Why this matters

我们前面已经讲过 pi0-FAST 的 stop-signal 解码循环、MEM 的短期视觉记忆 —— 这些都是"单一框架内部的推理细节"。但当你真要把模型部署到机器人上,你会发现 90% 的代码不是模型本身,而是 **"观测进来 → 预处理 → 喂模型 → 后处理 → 动作出去"** 这个外壳 —— 这层壳必须支持你今天用 JAX,明天换 PyTorch,后天换自己写的 nano 模型。openpi 的 `Policy` 是我见过最干净的一份 "推理外壳" 实现,80 行解决了 `JAX ↔ PyTorch` 互通、`jax.tree.map` 当通用 tree 操作、noise 注入与 timing 收集。这是一份你可以照搬到自己 nanoVLA 里的 `infer` 接口。

We've already covered pi0-FAST's stop-signal decode loop and MEM's short-term visual memory — both are "inside-the-model inference details". But once you actually deploy a model to a robot, 90% of the code isn't the model itself — it's the **"observation in → preprocess → model → postprocess → action out"** shell, and the shell has to accommodate JAX today, PyTorch tomorrow, your own nano model the day after. openpi's `Policy` is the cleanest "inference shell" I've seen: 80 lines that solve `JAX ↔ PyTorch` interop, `jax.tree.map` as a universal tree op, noise injection, and timing collection. It's a contract you can drop directly into your own nanoVLA.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/policies/policy.py`](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/policies/policy.py#L24-L106)

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
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
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
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs
```

## 逐行讲解 / What's happening

1. **构造函数的双路分支 / The two-branch constructor (lines 50-65)**:
   - 中文: 接收一个 `model`,根据 `is_pytorch` 走两条不同初始化路径。PyTorch 这边 `.to(device).eval()` 把模型搬上 GPU、冻结 dropout/BN;然后 `self._sample_actions = model.sample_actions` 直接拿 unbound method 准备调。JAX 这边用 `nnx_utils.module_jit` 给 `sample_actions` 套上 jit —— 注意 jit 在构造期间就触发了 tracing,所以第一次 `infer` 已经是 warm 的。
   - English: Takes a `model`, branches on `is_pytorch`. PyTorch side: `.to(device).eval()` moves to GPU and freezes dropout/BN; then `self._sample_actions = model.sample_actions` grabs the unbound method directly. JAX side: `nnx_utils.module_jit` wraps `sample_actions` in jit — tracing fires at construction time, so the first `infer` call is already warm.

2. **`inputs = jax.tree.map(lambda x: x, obs)` 这个看起来没用的拷贝 / `inputs = jax.tree.map(lambda x: x, obs)` — the deceptively useless copy (line 70)**:
   - 中文: 看起来啥也没做,实际是"按 pytree 结构复制一层"。`obs` 是嵌套 dict + ndarray,如果让下游 transform 原地改了 `obs["image"]`,调用者会很懵 —— 这一行隔离了上下游。
   - English: Looks like a no-op but is actually a "copy by pytree structure". `obs` is a nested dict + ndarray; if a downstream transform mutates `obs["image"]` in place, the caller would be confused. This single line walls off upstream from downstream.

3. **`self._input_transform(inputs)` 的 compose 链 / The `self._input_transform(inputs)` compose chain (line 71)**:
   - 中文: `_transforms.compose(transforms)` 把多个 `DataTransformFn` 串成一条流水线。一个典型的 transform 是 `ResizeImage((224, 224))`、`Normalize(mean, std)`、`UnifiedActionFormat`。注意 transform 函数签名是 `dict → dict`,所以可以任意插拔。
   - English: `_transforms.compose(transforms)` chains several `DataTransformFn` into a pipeline. A typical transform is `ResizeImage((224, 224))`, `Normalize(mean, std)`, `UnifiedActionFormat`. Each transform takes a `dict` and returns a `dict`, so pieces snap together freely.

4. **`jax.tree.map(...)` 同时作用在 jnp 和 torch 上 / `jax.tree.map(...)` walking both jnp and torch trees (lines 72-79)**:
   - 中文: 这是整个文件最妙的一招。`jax.tree.map` 是个"按 pytree 结构 walk + apply"工具,叶子节点是什么类型完全无所谓 —— 在 JAX 分支它把每个 ndarray 转成 `jnp.asarray(x)[np.newaxis, ...]`(批维度从无到有),PyTorch 分支转成 `torch.from_numpy(...).to(device)[None, ...]`,两条路径**走同一个 tree 结构**。换句话说,JAX 和 PyTorch 共享 "如何递归" 的代码,只在 leaf 层定制 "怎么变换".
   - English: The slickest move in the file. `jax.tree.map` is a "walk-the-pytree + apply" tool, totally agnostic about leaf types — in the JAX branch each ndarray becomes `jnp.asarray(x)[np.newaxis, ...]` (batch dim from none); in the PyTorch branch it becomes `torch.from_numpy(...).to(device)[None, ...]`. Both branches **walk the same tree structure** — they share the "how to recurse" logic and only differ in the leaf-level transform.

5. **RNG 和 device 共用一个变量名 / RNG and device share one variable name (lines 75, 79)**:
   - 中文: `sample_rng_or_pytorch_device` 这个变量名是个小宝藏 —— JAX 那边塞一个 `PRNGKey`,PyTorch 那边塞一个 `str` 设备字符串。这是因为 `model.sample_actions(rng_or_device, obs, **kwargs)` 的接口约定第一个参数"该是什么就是什么",由模型自己解释。
   - English: The variable name `sample_rng_or_pytorch_device` is a small treasure — JAX puts a `PRNGKey` here, PyTorch puts a `str` device. The contract is that `model.sample_actions(rng_or_device, obs, **kwargs)` accepts whichever the model itself expects in slot 1.

6. **noise 注入的两条平行实现 / The two parallel noise injections (lines 83-88)**:
   - 中文: `noise` 是机器人侧 rollout 时常常需要的外部信号(比如同一个噪声跑多个模型对比、复现实验)。注意它**没有走 `_input_transform`**,而是直接进 `sample_kwargs` —— 因为它不属于 observation,是一个超参式的、控制采样器随机性的额外通道。这条设计让"noise"和"obs"在数据流上正交。
   - English: `noise` is an external signal often needed during robot rollouts (e.g. run multiple models with the same noise for fair comparison, or reproduce experiments). Notice it **does not flow through `_input_transform`** — it goes straight to `sample_kwargs`. That's because noise isn't observation; it's a sampler-randomness channel. The design keeps "noise" and "obs" orthogonal in the data flow.

7. **timing 收集只裹模型那一层 / Timing wraps only the model call (lines 91-105)**:
   - 中文: `start_time = time.monotonic()` 紧贴 `_sample_actions` 调用,**transform 和 unbatch 都不算进去** —— 这是个有意为之的决定:报给上游的 `infer_ms` 是"模型本身的延迟",CPU 前处理慢不算模型的锅。
   - English: `start_time = time.monotonic()` hugs the `_sample_actions` call. **Transforms and unbatching are deliberately excluded**. The `infer_ms` you report upstream is "model latency alone" — slow CPU preprocessing isn't blamed on the model.

8. **unbatch 又是一次 `jax.tree.map` / Unbatching is another `jax.tree.map` (lines 97-100)**:
   - 中文: 输出里所有 leaf 都做 `x[0, ...]`(去掉前面加的 batch 维度)然后转 ndarray。PyTorch 分支多一句 `.detach().cpu()`,JAX 直接 `np.asarray(x[0, ...])` —— 两条路径仍然共享同一棵 tree。
   - English: All leaves get `x[0, ...]` (peel off the batch dim added earlier) then cast to ndarray. PyTorch branch adds `.detach().cpu()`, JAX uses `np.asarray(x[0, ...])`. The two branches still share the same tree.

## 类比 / The analogy

想象你开了一家**多品牌汽车经销店**,你必须接待开本田来的客户,也得接待开特斯拉来的客户。两边的"协议"完全不同:本田用机械钥匙,特斯拉用手机 app。但流程是一样的 —— 接车 → 拍照建档 → 跑诊断 → 出报告。聪明的做法不是写两套接待流程,而是写**一套接待流程**,在"塞钥匙到诊断仪"这一步插一个 if:本田就插机械钥匙,特斯拉就调 API。openpi 的 `Policy.infer` 就是这种"插钥匙"的设计:`jax.tree.map` 是接待流程,`sample_rng_or_pytorch_device` 就是那个"钥匙槽",不管插的是 RNG key 还是 device 字符串。

Imagine you run a **multi-brand car dealership**: you receive customers who drive in with Hondas, and customers who drive in with Teslas. The two protocols are totally different — Hondas use a mechanical key, Teslas use a phone app. But the workflow is identical: check the car in → photograph it → run diagnostics → print the report. Smart move: don't write two reception workflows, write **one** workflow with one `if` at "insert key into diagnostic tool" — Honda branch inserts the mechanical key, Tesla branch hits the API. openpi's `Policy.infer` is exactly this "key-insert" design: `jax.tree.map` is the workflow, `sample_rng_or_pytorch_device` is the key slot, agnostic about whether you're plugging in an RNG key or a device string.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:这个组件是 nanoVLA 课程里的 **`inference-loop`** —— 也是你 nanoVLA 整体架构里"最外面那一层"。它依赖前面已经覆盖的 `vlm-backbone-wiring` 和 `action-head-continuous`,因为它假设你已经把模型本身写好了,这一层只负责**外壳与适配**。在你自己的 nanoVLA 里,这个 `Policy.infer` 文件对应 `nanovla/runtime/policy.py`,被两类人调用:(1) 训练时的 evaluator/rollout 脚本;(2) 推理时的 policy server(robot 控制器)。它的输入契约是 `obs: dict[str, np.ndarray]`,输出契约是 `{"actions": np.ndarray, "policy_timing": {...}}`。上游是 `obs_buffer`(机器人发来的最新一帧),下游是 `actuator` 或者 `action_queue` —— `noise` 是给做"同噪声 A/B 对比"留的口子,可以不用。

English: This component is the **`inference-loop`** slot in your nanoVLA curriculum — and the outermost layer of the whole architecture. It depends on `vlm-backbone-wiring` and `action-head-continuous` (both already covered) because it assumes the model is built; this layer only handles **shell and adaptation**. In your own codebase this maps to `nanovla/runtime/policy.py`, called by (1) the evaluator/rollout script during training, and (2) the policy server during deployment. Input contract: `obs: dict[str, np.ndarray]`. Output contract: `{"actions": np.ndarray, "policy_timing": {...}}`. Upstream it reads the latest frame from `obs_buffer`; downstream it feeds either the actuator or the `action_queue`. The `noise` slot exists for fair A/B comparison and can be ignored in vanilla rollouts.

中文:如果你**省掉**这一层会发生什么?每个 rollout 脚本都要自己处理"加 batch 维 → 搬到 GPU → 模型 forward → 去 batch 维 → 转回 numpy"这五步,加上 framework dispatch,你的 evaluator、policy_server、jupyter notebook 就会出现五份大体相同但又各有 bug 的复制粘贴版。生产级实现至少要在这个 80 行模板上加:(a) async future-based 调用(让模型 forward 在后台跑、CPU 同时准备下一帧);(b) action queue + replanning interval(不是每个 obs 都重新跑模型,而是 chunk action 每 8 步 replan 一次);(c) observation/action 的版本号同步,避免 race condition。这些都属于"包装层"的事,不该污染模型本体。

English: If you **omit** this layer, every rollout script has to repeat "add batch dim → device transfer → model forward → strip batch dim → cast back to numpy" by hand, plus its own framework-dispatch — and your evaluator, policy server, and Jupyter notebook will each grow a slightly different, slightly buggy copy. A production-grade implementation needs to add at least: (a) async future-based calls so the model can forward in the background while CPU prepares the next frame; (b) an action queue with a replanning interval so you don't re-run the model every frame, but instead replan a chunk every 8 steps; (c) version-stamping observation/action pairs to prevent race conditions. All of these belong in the shell layer and shouldn't pollute the model itself.

## 自己跑一遍 / Try it yourself

```python
import time
import numpy as np
import jax, jax.numpy as jnp

class FakeJAXModel:
    sample_actions = staticmethod(lambda rng, obs: jnp.zeros((1, 8, 7)) + obs["state"][..., None, :])

class FakePyTorchModel:
    import torch
    sample_actions = staticmethod(lambda dev, obs: FakePyTorchModel.torch.zeros(1, 8, 7) + obs["state"][..., None, :])
    def to(self, d): return self; eval = lambda self: None

def compose(fns):
    def go(x):
        for f in fns: x = f(x)
        return x
    return go

class MiniPolicy:
    def __init__(self, model, *, is_pytorch=False, transforms=()):
        self.model, self.is_pt = model, is_pytorch
        self.tf = compose(transforms)
        if not is_pytorch:
            self.sample = jax.jit(model.sample_actions); self.rng = jax.random.key(0)
        else:
            self.sample = model.sample_actions
    def infer(self, obs):
        obs = jax.tree.map(lambda x: x, obs)        # structural copy
        obs = self.tf(obs)
        if not self.is_pt:
            obs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], obs)
            self.rng, k = jax.random.split(self.rng); arg = k
        else:
            import torch
            obs = jax.tree.map(lambda x: torch.from_numpy(np.array(x))[None, ...], obs); arg = "cpu"
        t0 = time.monotonic()
        out = {"actions": self.sample(arg, obs)}
        return {"actions": np.asarray(out["actions"][0]), "infer_ms": (time.monotonic() - t0) * 1000}

obs = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])}
print("JAX:    ", MiniPolicy(FakeJAXModel()).infer(obs))
print("PyTorch:", MiniPolicy(FakePyTorchModel(), is_pytorch=True).infer(obs))
```

运行 / Run with:
```bash
pip install jax torch numpy
python try.py
```

预期输出 / Expected output:
```
JAX:     {'actions': array([[0.1, 0.2, ...]] x8), 'infer_ms': 23.4}
PyTorch: {'actions': array([[0.1, 0.2, ...]] x8), 'infer_ms': 1.2}
```

中文:注意两个 backend 的 `actions` 内容完全一样,只有 `infer_ms` 不同 —— JAX 第一次 jit compile 慢,PyTorch eager 直接跑。这就是统一接口的意义:换 backend 不换 API。

English: Notice both backends produce identical `actions` and differ only in `infer_ms` — JAX pays JIT compilation cost on the first call; PyTorch eager runs immediately. That's the point of the unified interface: swap backends, keep the API.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot 的 `async_inference/policy_server.py` / lerobot's `async_inference/policy_server.py`**: 同样的 `obs → predict_action → put on queue` 外壳,只是 framework dispatch 在 `policy.config.type` 上 / Same `obs → predict_action → put on queue` shell, with framework dispatch driven by `policy.config.type`.
- **HuggingFace `pipeline()` / HuggingFace `pipeline()`**: `pipeline("image-classification")` 也是这种"统一外壳 + 后端可换"的设计,只不过 backend 选择是隐藏的 / `pipeline("image-classification")` is the same "unified shell + swappable backend" idea, with the backend selection hidden inside.
- **AlphaFold3 的 inference loop / AlphaFold3's inference loop**: 同样 jit 模型 + numpy 转换 + tree.map 的外壳;甚至变量名都很像 / Same jit'd model + numpy adapter + tree.map shell — even the variable names look similar.
- **MuJoCo MJX policy harness / MuJoCo MJX policy harness**: MJX 也是 JAX-first,policy 部分写成 `jax.jit` 包模型 + `tree.map` 整理 obs,几乎是这份代码的兄弟 / MJX is also JAX-first; the policy layer is a `jax.jit`-wrapped model plus a `tree.map` obs cleanup, almost a sibling of this code.

## 注意事项 / Caveats / when it breaks

- **PyTorch 分支会复制两次数据 / The PyTorch branch copies data twice**: `np.array(x)` 然后 `torch.from_numpy(...).to(device)` —— 第一次是 numpy 内存拷贝,第二次是 H2D。优化版应该用 `torch.as_tensor(x, device=device, dtype=...)` 一步到位 / `np.array(x)` then `torch.from_numpy(...).to(device)` does a numpy copy *and* an H2D copy. Optimized code should use `torch.as_tensor(x, device=device, dtype=...)` to merge both.
- **JAX 第一次 infer 会编译 / JAX's first infer triggers compilation**: `nnx_utils.module_jit` 是 lazy 编译,第一次 `infer()` 会有几百毫秒甚至几秒的 jit 开销。建议在服务启动后跑一次 warm-up call / `nnx_utils.module_jit` compiles lazily, so the first `infer()` pays hundreds of ms to seconds of jit overhead. Always do a warm-up call on server start.
- **`jax.tree.map` 默认把 dict / tuple / list 当 pytree / `jax.tree.map` treats dict / tuple / list as pytrees by default**: 如果你的 obs 里塞了 dataclass 或者自定义类,要么先 `flax.struct.dataclass` 注册,要么先转 dict —— 否则 `tree.map` 直接走 leaf 而不递归 / If your obs contains a dataclass or custom class, either register it via `flax.struct.dataclass` or convert to a dict first — otherwise `tree.map` will treat it as a leaf and skip recursion.
- **`noise` 维度顺序硬编码 / `noise` dim order is hardcoded**: 第 87 行的 `if noise.ndim == 2: noise = noise[None, ...]` 假设 noise 是 `(horizon, action_dim)`,如果你换了 layout(比如 batched noise)会出错 / Line 87's `if noise.ndim == 2: noise = noise[None, ...]` assumes `(horizon, action_dim)`. If you change the layout (e.g. batched noise) it breaks silently.

## 延伸阅读 / Further reading

- [openpi GitHub repo](https://github.com/Physical-Intelligence/openpi)
- [`jax.tree.map` docs (pytree walk)](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree.map.html)
- [Lerobot's `async_inference/policy_server.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/async_inference/policy_server.py)
- [PI's pi0 paper (the model behind this policy)](https://arxiv.org/abs/2410.24164)
