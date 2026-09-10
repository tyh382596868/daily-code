---
date: 2026-09-10
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/cache_utils.py
permalink: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1208-L1266
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, transformers, kv-cache, offload]
---

# Transformers Cache offload：下一层预取，上一层下放 / Transformers Cache Offload: Prefetch Next, Offload Current

> **一句话 / In one line**: `Cache.update()` 在 offloading 模式下等待预取流、更新当前层 K/V、再把当前层 cache 下放回 CPU。 / In offloading mode, `Cache.update()` waits for the prefetch stream, updates the current layer K/V, then offloads that layer's cache back to CPU.

## 为什么重要 / Why this matters

中文：长上下文生成时，KV cache 可能比模型权重更难放。Transformers 的 `Cache` 不只是 list 容器；它在每层更新前预取下一层，更新后把当前层移走。这样 GPU 上只常驻即将使用的 cache，代价是更多 CPU/GPU 传输和 stream 同步。

English: During long-context generation, the KV cache can be harder to fit than the model weights. Transformers' `Cache` is not just a list container; before updating a layer it prefetches the next offloaded layer, and after updating it moves the current layer away. GPU residency is narrowed to what is about to be used, at the cost of transfer and stream synchronization.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1208-L1266)

```python
def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
    is_offloaded = [
        not is_linear and not (only_non_sliding and is_sliding)
        for is_linear, is_sliding in zip(self.is_linear, self.is_sliding)
    ]
    try:
        layer_idx = layer_idx + is_offloaded[layer_idx:].index(True)
    except ValueError:
        layer_idx = is_offloaded.index(True)
    with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
        self.layers[layer_idx].prefetch()

def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
    if self.layer_class_to_replicate is not None:
        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())
    if self.offloading:
        torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
        self.prefetch(layer_idx + 1, self.only_non_sliding)

    keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    if self.offloading:
        self.offload(layer_idx, self.only_non_sliding)

    return keys, values
```

## 逐行讲解 / What's happening

1. **第 1216-1219 行 / Lines 1216-1219**:
   - 中文: 只把值得 offload 的层纳入预取队列；linear attention 不走同样 K/V cache，sliding 层默认也可常驻。
   - English: Only layers worth offloading enter the prefetch queue. Linear-attention layers do not use the same K/V path, and sliding layers can stay resident by default.
2. **第 1220-1228 行 / Lines 1220-1228**:
   - 中文: 从当前层之后找下一个 offloaded 层；找不到就绕回开头，并在非默认 stream 上预取。
   - English: It searches for the next offloaded layer after the current point, wraps around if needed, and prefetches on a non-default stream.
3. **第 1252-1255 行 / Lines 1252-1255**:
   - 中文: lazy cache 会按 `layer_idx` 自动补齐层对象，避免初始化时必须知道完整层数。
   - English: Lazy caches append layer objects up to `layer_idx`, so construction does not need every layer materialized upfront.
4. **第 1256-1266 行 / Lines 1256-1266**:
   - 中文: update 前等待预取完成并启动下一层预取；当前层更新完后立即 offload。
   - English: Before update, it waits for prefetch and launches prefetch for the next layer; after the current update, it offloads the current layer.

## 类比 / The analogy

中文：像流水线上的工具车。工人开始装第 N 层前，下一层工具已经推到旁边；第 N 层用完的工具马上推回仓库。

English: It is like a tool cart on an assembly line. Before the worker starts layer N, the next layer's tools are already nearby; after layer N is done, its tools go back to storage.

## 自己跑一遍 / Try it yourself

```python
class Cache:
    def __init__(self):
        self.gpu = set()
    def prefetch(self, layer):
        self.gpu.add(layer)
        print("prefetch", layer, "gpu=", sorted(self.gpu))
    def offload(self, layer):
        self.gpu.discard(layer)
        print("offload", layer, "gpu=", sorted(self.gpu))
    def update(self, layer):
        self.prefetch((layer + 1) % 3)
        print("update", layer)
        self.offload(layer)

c = Cache()
c.prefetch(0)
for layer in range(3):
    c.update(layer)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
prefetch 0 gpu= [0]
prefetch 1 gpu= [0, 1]
update 0
offload 0 gpu= [1]
prefetch 2 gpu= [1, 2]
update 1
offload 1 gpu= [2]
prefetch 0 gpu= [0, 2]
update 2
offload 2 gpu= [0]
```

中文：这个模式用传输换显存；如果 PCIe 传输太慢，吞吐可能反而下降。

English: This pattern trades transfer for memory. If CPU/GPU transfer is too slow, throughput can drop.

## 注意事项 / Caveats / when it breaks

- **stream 同步不能省** / **Stream synchronization is required**: 默认流必须等预取完成，否则当前层会读到未就绪 cache。
- **sliding/linear 层有特殊规则** / **Sliding and linear layers have special rules**: 不是每层都应该被下放。
- **循环预取依赖完整层属性** / **Wraparound prefetch needs layer metadata**: `is_linear` 和 `is_sliding` 必须和实际 cache 层对齐。

## 延伸阅读 / Further reading

- [Transformers cache_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
- [Transformers KV cache guide](https://huggingface.co/docs/transformers/kv_cache)
