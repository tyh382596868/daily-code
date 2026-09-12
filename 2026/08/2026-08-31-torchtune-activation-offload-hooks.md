---
date: 2026-08-31
topic: infrastructure
source: tracked
repo: pytorch/torchtune
file: torchtune/training/_activation_offloading.py
permalink: https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/training/_activation_offloading.py#L24-L72
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, activation-offloading, saved-tensors-hooks]
---

# torchtune activation offload：用 hooks 给反向传播腾显存 / torchtune Activation Offload: Use Hooks to Free VRAM for Backward

> **一句话 / In one line**: `saved_tensors_hooks` 让 forward 保存的激活先去 CPU，backward 要用时再搬回来。 / `saved_tensors_hooks` lets forward-saved activations move to CPU and come back only when backward needs them.

## 为什么重要 / Why this matters

大模型训练经常不是算力先爆，而是激活显存先爆。torchtune 这段代码把 PyTorch autograd 的“保存张量”入口接管掉：够大的 GPU 激活被复制到 CPU，原来的 GPU 内存就有机会释放；反传读到同一个 tensor id 时再搬回 GPU。

Large-model training often hits activation memory before compute. This code takes over PyTorch autograd's saved-tensor path: large GPU activations are copied to CPU so GPU memory can be reused, then restored by tensor id during backward.

## 代码 / The code

`pytorch/torchtune` — [`torchtune/training/_activation_offloading.py`](https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/training/_activation_offloading.py#L24-L72)

```python
class OffloadActivations(saved_tensors_hooks):
    """Context manager under which activation tensors created in the forward pass will be offloaded.

    Enable the memory efficiency technique of activation offloading, where activations bigger than
    min_offload_size bytes will be offloaded to CPU in the forward and brought back in the backward.
    This is in contrast to maintaining the activation on GPU VRAM throughout the program.

    This manager contains the option of using one additional CUDA stream to handle the communication
    between CUDA and CPU, which is intended to overlap with the default computation stream to improve
    runtime. We designed synchronization with a few heuristics for optimizing the tradeoff between
    runtime vs memory usage.
    """

    def __init__(
        self,
        use_pin_memory: bool = True,
        use_streams: bool = True,
        max_fwd_stash_size: int = 5,
        min_offload_size: int = 1024,
    ) -> None:
        self.use_streams: bool = use_streams
        self.min_tensor_size_bytes = min_offload_size
        self.tracker = {}
        self.tensor_id: int = 0
        self.is_first_forward_call = True
        self.is_first_backward_call = True
        self.is_first_forward_pass = True
```

## 逐行讲解 / What's happening

1. **第 24 行 / Line 24 (`saved_tensors_hooks`)**:
   - 中文: 这个类不是普通 context manager，而是直接继承 autograd 保存张量 hook。forward 中“本来要存起来给 backward 用”的张量都会走它的 pack/unpack。
   - English: This is not just a normal context manager. It inherits autograd's saved-tensor hook API, so tensors saved for backward pass through its pack/unpack path.
2. **第 27-29 行 / Lines 27-29 (CPU offload contract)**:
   - 中文: 设计目标很直接：超过阈值的 activation 不长期占 GPU，而是 forward 后放到 CPU。
   - English: The contract is direct: activations above the threshold should not occupy GPU memory for the whole step; they move to CPU after forward.
3. **第 31-34 行 / Lines 31-34 (extra stream)**:
   - 中文: 数据搬运可以放到额外 CUDA stream，让拷贝和默认 stream 上的计算尽量重叠。
   - English: Copies can run on an additional CUDA stream so communication overlaps with default-stream compute.
4. **第 68-72 行 / Lines 68-72 (knobs)**:
   - 中文: 这几个参数是工程取舍：pin memory 更快但占稀缺资源，stash 越大越容易重叠但越吃内存，阈值太小会浪费带宽。
   - English: These knobs encode engineering tradeoffs: pinned memory is faster but scarce, a bigger stash improves overlap but costs memory, and too small a threshold wastes bandwidth.

## 类比 / The analogy

像考试时桌面只能放几本书。你把暂时不用的大部头放到旁边书架，等做题真用到那一页再拿回来。桌面空了，但来回拿书也有时间成本。

Think of a small exam desk. You move bulky books to a nearby shelf and bring one back only when the problem needs it. The desk is freer, but each fetch has a cost.

## 自己跑一遍 / Try it yourself

```python
import torch
from torch.autograd.graph import saved_tensors_hooks

store = {}

def pack(t):
    key = len(store)
    store[key] = t.detach().cpu()
    return key

def unpack(key):
    return store.pop(key)

x = torch.randn(4, requires_grad=True)
with saved_tensors_hooks(pack, unpack):
    y = (x * x).sum()
    y.backward()

print(x.grad.tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
four gradient values, each equal to 2 * x
```

中文: 关键现象是 backward 没有直接拿原始 tensor，而是通过 `unpack` 把 forward 保存的东西还原出来。

English: The key observation is that backward does not directly hold the original tensor; it restores what forward saved through `unpack`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch checkpoint** / **PyTorch checkpointing**: 中文: 另一种省 activation 的方式是不保存，backward 时重算。 / English: Another way to save activation memory is to avoid saving and recompute during backward.
- **ZeRO-Offload** / **ZeRO-Offload**: 中文: 同样把 GPU 压力转移到 CPU，只是对象常常是 optimizer state 或参数分片。 / English: It also moves pressure from GPU to CPU, usually for optimizer state or parameter shards.

## 注意事项 / Caveats / when it breaks

- **PCIe 带宽会变成瓶颈 / PCIe bandwidth can bottleneck**: 中文: 如果每一步都搬太多 tensor，省下的显存会被拷贝时间吃掉。 / English: If every step moves too many tensors, copy time can erase the memory win.
- **小 tensor 不值得搬 / Small tensors are not worth moving**: 中文: 所以实现里有 `min_offload_size`。 / English: That is why the implementation has `min_offload_size`.

## 延伸阅读 / Further reading

- PyTorch saved tensors hooks: https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks
- torchtune activation offloading source: https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/training/_activation_offloading.py
