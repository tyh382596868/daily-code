---
date: 2026-08-04
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/utils/random.py
permalink: https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/utils/random.py#L81-L165
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, accelerate, rng]
---

# Accelerate RNG sync：随机数状态也要广播 / Accelerate RNG Sync: Broadcast the Random State Too

> **一句话 / In one line**: `synchronize_rng_state` 从主进程取 RNG state，按分布式后端广播，再写回每个设备的随机数生成器。 / `synchronize_rng_state` reads RNG state from the main process, broadcasts it through the active distributed backend, and writes it back to each device generator.

## 为什么重要 / Why this matters

多卡训练里，随机数不是小事。dropout、随机采样、数据增强和 generator 都可能让 rank 之间走向不同轨迹。Accelerate 把“取哪个 RNG、怎么广播、写回哪里”统一封装，让上层训练循环不用手写每种设备后端的分支。

In multi-device training, randomness is not a detail. Dropout, sampling, augmentation, and custom generators can make ranks diverge. Accelerate wraps “which RNG to read, how to broadcast it, and where to write it back” so the training loop does not need backend-specific branches.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/utils/random.py`](https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/utils/random.py#L81-L165)

```python
def synchronize_rng_state(rng_type: Optional[RNGType] = None, generator: Optional[torch.Generator] = None):
    # Get the proper rng state
    if rng_type == RNGType.TORCH:
        rng_state = torch.get_rng_state()
    elif rng_type == RNGType.CUDA:
        rng_state = torch.cuda.get_rng_state()
    elif rng_type == RNGType.XLA:
        assert is_torch_xla_available(), "Can't synchronize XLA seeds as torch_xla is unavailable."
        rng_state = torch.tensor(xm.get_rng_state())
    elif rng_type == RNGType.NPU:
        assert is_npu_available(), "Can't synchronize NPU seeds on an environment without NPUs."
        rng_state = torch.npu.get_rng_state()
    elif rng_type == RNGType.GENERATOR:
        assert generator is not None, "Need a generator to synchronize its seed."
        rng_state = generator.get_state()

    # Broadcast the rng state from device 0 to other devices
    state = AcceleratorState()
    if state.distributed_type == DistributedType.XLA:
        rng_state = rng_state.to(xm.xla_device())
        xm.collective_broadcast([rng_state])
        xm.mark_step()
        rng_state = rng_state.cpu()
    elif (
        state.distributed_type in CUDA_DISTRIBUTED_TYPES
        or state.distributed_type == DistributedType.MULTI_MLU
        or state.distributed_type == DistributedType.MULTI_SDAA
        or state.distributed_type == DistributedType.MULTI_MUSA
        or state.distributed_type == DistributedType.MULTI_NPU
        or state.distributed_type == DistributedType.MULTI_XPU
        or state.distributed_type == DistributedType.MULTI_HPU
        or state.distributed_type == DistributedType.MULTI_NEURON
    ):
        rng_state = rng_state.to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        rng_state = rng_state.cpu()
    elif state.distributed_type == DistributedType.MULTI_CPU:
        torch.distributed.broadcast(rng_state, 0)

    # Set the broadcast rng state
    if rng_type == RNGType.TORCH:
        torch.set_rng_state(rng_state)
    elif rng_type == RNGType.CUDA:
        torch.cuda.set_rng_state(rng_state)
    elif rng_type == RNGType.NPU:
        torch.npu.set_rng_state(rng_state)
    elif rng_type == RNGType.XLA:
        xm.set_rng_state(rng_state.item())
    elif rng_type == RNGType.GENERATOR:
        generator.set_state(rng_state)


def synchronize_rng_states(rng_types: list[Union[str, RNGType]], generator: Optional[torch.Generator] = None):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)
```

## 逐行讲解 / What's happening

1. **第 83-113 行 / Lines 83-113 (read state by kind)**:
   - 中文: 不同设备的 RNG state API 不一样，函数先把它们统一成 `rng_state` 这个 tensor-like 对象。
   - English: Different devices expose different RNG state APIs, so the function first normalizes them into one `rng_state` object.
2. **第 116-137 行 / Lines 116-137 (broadcast by backend)**:
   - 中文: XLA 用 `xm.collective_broadcast`，CUDA/MLU/NPU/XPU 等走 `torch.distributed.broadcast`，CPU 分布式也单独处理。
   - English: XLA uses `xm.collective_broadcast`; CUDA-like backends use `torch.distributed.broadcast`; multi-CPU is handled separately.
3. **第 139-160 行 / Lines 139-160 (write back)**:
   - 中文: 广播完还必须写回对应 generator，否则只是同步了一个临时 tensor。
   - English: After broadcast, the state must be written back to the matching generator; otherwise only a temporary tensor was synchronized.

## 类比 / The analogy

这像拍电影前给每台摄像机同步 timecode：不是只告诉大家“现在开始”，而是把主机的精确时间码写进每台机器，这样剪辑时画面才能对齐。

It is like syncing timecode across cameras before a shoot: you do not merely say “start now”; you write the master clock into every camera so the footage aligns later.

## 自己跑一遍 / Try it yourself

```python
import random

master = random.Random(7)
state = master.getstate()

workers = [random.Random(), random.Random()]
for w in workers:
    w.setstate(state)

print([w.randint(0, 99) for w in workers])
print([w.randint(0, 99) for w in workers])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[41, 41]
[19, 19]
```

中文: 两个 worker 的随机序列完全一致，因为它们拿到了同一个 state。
English: Both workers produce the same random sequence because they received the same state.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DistributedSampler seed** / **DistributedSampler seed**: 数据顺序也要按 epoch 和 rank 可复现。 / Data order also needs reproducible epoch and rank-dependent seeds.
- **Diffusers generator lists** / **Diffusers generator lists**: 多样本生成常显式传入 generator 来控制每个样本随机性。 / Multi-sample generation often passes explicit generators to control randomness per sample.

## 注意事项 / Caveats / when it breaks

- **只同步 state，不同步业务逻辑** / **State sync is not logic sync**: 如果不同 rank 调用随机数次数不同，后面还是会分叉。 / If ranks consume a different number of random values, they will diverge again.
- **后端必须已初始化** / **Backend must be initialized**: `AcceleratorState()` 依赖当前分布式环境。 / `AcceleratorState()` depends on the current distributed environment.

## 延伸阅读 / Further reading

- Accelerate `random.py`: https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/utils/random.py#L81-L165
