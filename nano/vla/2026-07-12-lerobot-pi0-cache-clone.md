---
date: 2026-07-12
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/pi0/modeling_pi0.py
permalink: https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/pi0/modeling_pi0.py#L132-L140
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop]
build_role: inference-loop cache isolation for a from-scratch nanoVLA
---

# LeRobot pi0 cache clone：预填前缀可以复用，但不能共享可变状态 / LeRobot pi0 Cache Clone: Reuse Prefix Prefill Without Sharing Mutable State

> **一句话 / In one line**: pi0 把 prefix prefill 得到的 `DynamicCache` 深拷贝一份，给后续编译 denoising 循环使用。 / pi0 deep-copies the `DynamicCache` produced by prefix prefill so the compiled denoising loop can use it without mutating the original.

## 为什么重要 / Why this matters

VLA 推理里，语言和图像前缀通常只需要编码一次，后面每个 denoising step 复用这段 KV cache。但 cache 是可变对象，如果多个 step 或编译图共享同一份引用，后续 append 会污染基线前缀。这几行代码的价值就在于：复用计算结果，但隔离状态。

In VLA inference, language and image prefixes are often encoded once and reused by every denoising step. But a cache is mutable; if compiled loops or repeated steps share the same object, later appends can corrupt the prefix baseline. These few lines reuse the computation while isolating state.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/pi0/modeling_pi0.py`](https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/pi0/modeling_pi0.py#L132-L140)

```python
def clone_past_key_values(past_key_values):
    """Clone the DynamicCache returned by prefix prefill for compiled denoising."""
    return DynamicCache(
        tuple(
            (keys.clone(), values.clone(), sliding_window) for keys, values, sliding_window in past_key_values
        )
    )
```

## 逐行讲解 / What's happening

1. **输入是 prefill cache / The input is prefill cache**:
   - 中文: `past_key_values` 保存前缀 token 的 K/V，通常来自图像、语言、state 这些不随 denoising step 改变的部分。
   - English: `past_key_values` stores K/V for prefix tokens, usually from image, language, and state inputs that do not change across denoising steps.
2. **逐层 clone tensor / Clone tensors per layer**:
   - 中文: 每层的 `keys` 和 `values` 都调用 `.clone()`，新 cache 有自己的底层存储。
   - English: Each layer's `keys` and `values` call `.clone()`, so the new cache owns separate storage.
3. **保留 sliding window 元数据 / Preserve sliding-window metadata**:
   - 中文: `sliding_window` 是配置型元数据，不需要 tensor clone，但必须一起带过去。
   - English: `sliding_window` is configuration-like metadata; it does not need tensor cloning, but it must travel with the cache.

## 类比 / The analogy

像考试时老师发草稿纸：题目已经印好，可以复用同一份模板；但每个学生必须拿自己的纸写推导，不能在公共模板上涂改。

It is like handing out exam scratch paper: everyone can reuse the printed prompt, but each student must write on their own copy rather than modifying the shared master sheet.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `inference-loop` 组件的一部分，依赖前面的 `vlm-backbone-wiring` 和 `action-head-continuous`。在 nanoVLA 中，你可以先对图像和指令做一次 prefix prefill，随后每个 action denoising step 复制 cache、追加 noisy action token、预测 velocity。如果省掉 clone，最常见的问题是第二个 step 读到第一个 step 追加过的 KV。

This belongs in the `inference-loop` component, after `vlm-backbone-wiring` and `action-head-continuous`. In a nanoVLA, you prefill image and instruction tokens once, then each action denoising step clones the cache, appends noisy action tokens, and predicts velocity. Without cloning, the second step can accidentally read KV entries appended by the first step.

## 自己跑一遍 / Try it yourself

```python
cache = [([1, 2], [10, 20], None)]

def bad_reuse(c):
    c[0][0].append(3)

def good_clone(c):
    return [(list(k), list(v), w) for k, v, w in c]

copy_cache = good_clone(cache)
bad_reuse(copy_cache)
print("original", cache)
print("copy", copy_cache)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
original [([1, 2], [10, 20], None)]
copy [([1, 2, 3], [10, 20], None)]
```

这个玩具例子说明：浅复用会污染前缀，深拷贝让每个 denoising 分支自己承担修改。

This toy example shows the core point: shallow reuse corrupts the prefix, while deep copy lets each denoising branch own its mutations.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers generation cache** / **Transformers generation cache**: beam search 常要复制 cache，避免不同 beam 的追加互相串扰。 / Beam search often copies cache so beams do not contaminate each other.
- **vLLM prefix cache** / **vLLM prefix cache**: prefix 可以共享，但请求级 append 状态必须独立管理。 / Prefixes can be shared, while request-level append state must be managed independently.

## 注意事项 / Caveats / when it breaks

- **clone 有显存成本** / **clone costs memory**: 长上下文和多层模型会让复制成本明显上升。 / Long contexts and deep models make copying expensive.
- **只 clone tensor，不 clone策略** / **clone tensors, not policy**: sliding-window 或 cache eviction 规则仍然要和原模型一致。 / Sliding-window and eviction rules still need to match the source model.

## 延伸阅读 / Further reading

- LeRobot pi0 source — https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/pi0/modeling_pi0.py
