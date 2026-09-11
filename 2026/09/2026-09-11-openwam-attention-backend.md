---
date: 2026-09-11
topic: robotics
source: trending
repo: OpenWAM-Official/OpenWAM
file: openwam/model/action_backbone/components.py
permalink: https://github.com/OpenWAM-Official/OpenWAM/blob/d8dd33d8576b475f5a5cdc6fb8ca902778a199b3/openwam/model/action_backbone/components.py#L352-L396
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, trending, attention-backend, fallback, runtime-selection]
---

# OpenWAM attention backend：一个 tensor 契约，多个 kernel / OpenWAM Attention Backend: One Tensor Contract, Many Kernels

> **一句话 / In one line**: OpenWAM 允许显式指定 attention backend，也能按可用性自动从 FlashAttention 3 退到 SDPA。 / OpenWAM supports explicit backend selection and can automatically fall back from FlashAttention 3 to native SDPA.

## 为什么重要 / Why this matters

中文：生产级机器人模型经常跑在不同机器上：有的装了 FlashAttention，有的只有 PyTorch SDPA，有的还需要 xFormers 或 SageAttention。OpenWAM 把这些差异压在一个 `(q, k, v) -> out` 契约之后，启动时选择一次并缓存函数，模型主体不用到处写环境判断。

English: Production robotics models run across machines with different kernel stacks. Some have FlashAttention, others only native SDPA, and some use xFormers or SageAttention. OpenWAM hides those differences behind one `(q, k, v) -> out` contract, selects once at startup, and caches the callable.

## 代码 / The code

`OpenWAM-Official/OpenWAM` — [`openwam/model/action_backbone/components.py`](https://github.com/OpenWAM-Official/OpenWAM/blob/d8dd33d8576b475f5a5cdc6fb8ca902778a199b3/openwam/model/action_backbone/components.py#L352-L396)

```python
_BACKEND_MAP = {
    "flash3": _try_flash_attn_3,
    "flash2": _try_flash_attn_2,
    "sage": _try_sage_attention,
    "xformers": _try_xformers,
    "sdpa": lambda: _sdpa,
}

_AUTO_PRIORITY = ["flash3", "flash2", "sage", "xformers", "sdpa"]


def get_attention_fn() -> Callable:
    """Return the best available attention function.

    Signature: ``(q, k, v) -> out`` where tensors are ``(B, H, S, D)``.
    """
    global _ATTENTION_FN
    if _ATTENTION_FN is not None:
        return _ATTENTION_FN

    override = os.environ.get("WAM_ATTENTION_IMPL", "").strip().lower()

    if override:
        if override not in _BACKEND_MAP:
            raise ValueError(f"Unknown WAM_ATTENTION_IMPL='{override}'. Choose from: {list(_BACKEND_MAP.keys())}")
        fn = _BACKEND_MAP[override]()
        if fn is None:
            logger.warning(
                "WAM_ATTENTION_IMPL='%s' requested but not available, falling back to auto-detect",
                override,
            )
        else:
            logger.info("Using attention backend: %s (explicit)", override)
            _ATTENTION_FN = fn
            return _ATTENTION_FN

    for name in _AUTO_PRIORITY:
        fn = _BACKEND_MAP[name]()
        if fn is not None:
            logger.info("Using attention backend: %s (auto-detected)", name)
            _ATTENTION_FN = fn
            return _ATTENTION_FN

    _ATTENTION_FN = _sdpa
    return _ATTENTION_FN
```

## 逐行讲解 / What's happening

1. **第 352-358 行 / Lines 352-358**:
   - 中文: `_BACKEND_MAP` 把用户可见名字映射到“尝试加载 backend 的工厂函数”，而不是直接保存可能导入失败的 kernel。
   - English: `_BACKEND_MAP` maps user-facing names to backend factory functions, avoiding imports of optional kernels before they are needed.
2. **第 360 行 / Line 360**:
   - 中文: 自动优先级是策略的一部分：更快的 fused kernel 在前，原生 SDPA 最后兜底。
   - English: The priority list is part of the policy: faster fused kernels come first, native SDPA is the final fallback.
3. **第 368-370 行 / Lines 368-370**:
   - 中文: `_ATTENTION_FN` 做进程级缓存，避免每一层、每一个 batch 重复探测环境。
   - English: `_ATTENTION_FN` is cached process-wide so every layer and batch does not rediscover the environment.
4. **第 372-386 行 / Lines 372-386**:
   - 中文: 环境变量是显式 override；如果用户指定的可选 backend 不存在，代码记录 warning 后继续自动探测。
   - English: The environment variable is an explicit override. If the requested optional backend is missing, the code warns and continues with auto-detection.
5. **第 388-396 行 / Lines 388-396**:
   - 中文: 自动扫描第一个可用实现；如果全部失败，仍返回 `_sdpa`，让功能正确性优先于性能。
   - English: Auto-detection returns the first available implementation; if all optional paths fail, `_sdpa` keeps correctness ahead of performance.

## 类比 / The analogy

中文：像机器人控制柜里的电机驱动选择器。控制层只规定“给我一个速度输入、返回一个执行结果”，启动时根据硬件选择 CAN、EtherCAT 或模拟器驱动。

English: It is like a motor-driver selector in a robot control cabinet. The control layer defines one input/output contract, while startup chooses CAN, EtherCAT, or a simulator driver based on the machine.

## 自己跑一遍 / Try it yourself

```python
import os

BACKENDS = {"fast": lambda: "fast-kernel", "safe": lambda: "safe-kernel"}
PRIORITY = ["fast", "safe"]

def select():
    override = os.environ.get("BACKEND", "").strip()
    if override:
        if override not in BACKENDS:
            raise ValueError(override)
        return BACKENDS[override]()
    return next(BACKENDS[name]() for name in PRIORITY)

print(select())
os.environ["BACKEND"] = "safe"
print(select())
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
fast-kernel
safe-kernel
```

中文：选择逻辑只跑一次、返回稳定 callable，是把“部署差异”从模型计算图里移出去的关键。

English: Running selection once and returning a stable callable is what keeps deployment differences out of the model's computation graph.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch SDPA dispatch** / **PyTorch SDPA dispatch**: 根据 device、dtype 和输入形状挑选 attention 实现。 / Device, dtype, and input shape choose an attention implementation.
- **vLLM attention backends** / **vLLM attention backends**: runtime backend 选择和 kernel capability 绑定。 / Runtime backend selection is tied to kernel capability.
- **OpenWAM architecture registry** / **OpenWAM architecture registry**: Hydra 配置先选组件，运行时再把组件装进完整 WAM。 / Hydra selects components first, then runtime composes them into a full WAM.

## 注意事项 / Caveats / when it breaks

- **shape 契约必须一致** / **The shape contract must stay identical**: fallback 不能只在名字上兼容，输入 layout 和 mask 语义也要一致。
- **全局缓存会影响测试** / **Global caching affects tests**: 测试切换环境变量前要清空 `_ATTENTION_FN`。
- **fallback 可能隐藏性能回退** / **Fallbacks can hide performance regressions**: 启动日志或 metrics 应记录最终 backend。

## 延伸阅读 / Further reading

- [OpenWAM attention components](https://github.com/OpenWAM-Official/OpenWAM/blob/d8dd33d8576b475f5a5cdc6fb8ca902778a199b3/openwam/model/action_backbone/components.py)
- [OpenWAM repository](https://github.com/OpenWAM-Official/OpenWAM)
