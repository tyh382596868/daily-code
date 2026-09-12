---
date: 2026-08-19
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/_utils/worker.py
permalink: https://github.com/pytorch/pytorch/blob/bb046f4988cc6ad2d9e51a848c23055538f5e9fd/torch/utils/data/_utils/worker.py#L188-L241
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader, seeding]
---

# PyTorch worker seed：给 NumPy 单独洗一副随机牌 / PyTorch Worker Seed: Shuffle a Separate Deck for NumPy

> **一句话 / In one line**: DataLoader worker 不直接把同一个 seed 塞给 NumPy，而是把 `base_seed` 和 `worker_id` 混成 4 个 32-bit 状态数。 / A DataLoader worker does not hand NumPy the same seed; it mixes `base_seed` and `worker_id` into four 32-bit state values.

## 为什么重要 / Why this matters

多进程 DataLoader 最怕“看起来随机，实际各 worker 走同一条随机序列”。PyTorch 已经给 torch RNG 做了 worker seed，但 NumPy 和 Python `random` 也常被 dataset transform 使用。这里的 `_generate_state` 专门为 NumPy 生成独立状态，避免不同 RNG 库因为相同 seed 和相似算法撞车。

Multiprocess data loading can silently fail when workers appear random but actually follow correlated streams. PyTorch seeds its own RNG, but datasets often use NumPy or Python `random` inside transforms. `_generate_state` builds a separate NumPy state so worker-local randomness does not collide with the torch/Python seed path.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/_utils/worker.py`](https://github.com/pytorch/pytorch/blob/bb046f4988cc6ad2d9e51a848c23055538f5e9fd/torch/utils/data/_utils/worker.py#L188-L241)

```python
# This function generates an array of int32 as the seed for
# `numpy.random`, in order to prevent state collision due to same
# seed and algorithm for `numpy.random` and `random` modules.
# TODO: Implement `SeedSequence` like object for `torch.random`
def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state
```

## 逐行讲解 / What's happening

1. **第 193-200 行 / Lines 193-200 (constants)**:
   - 中文: 这些十六进制常量用于 hash 和 mix，让低位、高位都参与扰动。
   - English: These constants drive the hash and mix steps so both low and high bits affect the result.
2. **第 202 行 / Line 202 (`entropy`)**:
   - 中文: 输入不是一个整数，而是 worker id、seed 低 32 位、seed 高 32 位和一个占位槽。
   - English: The entropy input is not one integer; it includes worker id plus low and high halves of the base seed.
3. **第 222-230 行 / Lines 222-230 (pool mixing)**:
   - 中文: 先各自 hash，再让 pool 中每个位置和其他位置互相混合，避免只改一个输入位只影响一个输出位。
   - English: Values are first hashed, then each pool slot is mixed with the others so one changed input bit can affect every output word.
4. **第 232-241 行 / Lines 232-241 (state output)**:
   - 中文: 第二组常量再洗一次，最后产出 NumPy 可吃的 4 个 `int32` 状态。
   - English: A second constant stream scrambles the pool once more and emits four `int32` state words for NumPy.

## 类比 / The analogy

像给四个摄影师发门票。不能只把同一张票复印四份，否则入口系统可能以为是同一个人；要把姓名、日期、座位区混成四张不同但可追踪的票。

It is like issuing tickets to four camera operators. Copying the same ticket four times creates collisions at the gate; mixing name, date, and section produces distinct but reproducible tickets.

## 自己跑一遍 / Try it yourself

```python
def tiny_state(base_seed, worker_id):
    mask = 0xFFFFFFFF
    entropy = [worker_id, base_seed & mask, base_seed >> 32, 0]
    state = []
    x = 0x43B0D7E5
    for value in entropy:
        x = (x * 0x931E8875) & mask
        y = ((value ^ x) * (x | 1)) & mask
        state.append((y ^ (y >> 16)) & mask)
    return state

print(tiny_state(1234567890123, 0))
print(tiny_state(1234567890123, 1))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[2854989754, 2736501125, 3779864178, 1641700067]
[4224731704, 2736501125, 3779864178, 1641700067]
```

中文: 只改 `worker_id`，第一个状态词就变了；真实 PyTorch 版本会进一步让所有位置互相扩散。

English: Changing only `worker_id` changes the first state word; the real PyTorch version diffuses that difference across the full pool.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NumPy `SeedSequence`** / **NumPy `SeedSequence`**: 同样把熵扩散成可复现的多 word 状态。 / It similarly expands entropy into reproducible multi-word state.
- **分布式训练 rank seed** / **Distributed rank seeding**: 常见写法是 `base_seed + rank`，但严肃系统会额外 hash，避免相邻 rank 相关。 / Serious systems hash rank seeds instead of only adding the rank.

## 注意事项 / Caveats / when it breaks

- **不是加密随机** / **Not cryptographic randomness**: 这里追求可复现和低碰撞，不适合安全用途。 / This is for reproducibility and collision avoidance, not security.
- **worker 外部状态仍要管** / **External worker state still matters**: 如果 dataset 自己连接数据库或服务端随机源，这个函数不会替你隔离那些状态。 / If a dataset talks to databases or remote random sources, this function does not isolate those systems.

## 延伸阅读 / Further reading

- [PyTorch DataLoader worker seeding source](https://github.com/pytorch/pytorch/blob/bb046f4988cc6ad2d9e51a848c23055538f5e9fd/torch/utils/data/_utils/worker.py#L188-L241)
- [PyTorch data loading randomness docs](https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading)
