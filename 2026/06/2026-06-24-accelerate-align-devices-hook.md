---
date: 2026-06-24
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/hooks.py
permalink: https://github.com/huggingface/accelerate/blob/main/src/accelerate/hooks.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, accelerate, cpu-offload, meta-device, memory-management]
---

# AlignDevicesHook：70B 模型如何用 meta device 实现零显存加载 / AlignDevicesHook: How 70B Models Load with Zero GPU Memory via the Meta Device

> **一句话 / In one line**: `AlignDevicesHook` 把每一层的权重生命周期压成三段——初始化时上 meta（零显存）、forward 前加载到 GPU、forward 后归还 meta——这是 `device_map="auto"` 和磁盘 offload 的核心机制。/ `AlignDevicesHook` compresses each layer's weight lifecycle into three phases — meta on init (zero GPU memory), load to GPU before forward, back to meta after — the core mechanism behind `device_map="auto"` and disk offloading.

## 为什么重要 / Why this matters

加载一个 70B 参数的模型，权重本身就有 ~140 GB。多数机器的 GPU 显存不够。Accelerate 的解决方案不是"不加载"，而是"随用随加载随释放"：用 `meta` device 作为占位符（只有形状和 dtype 信息，不占实际内存），在每层的 forward 前把权重从 CPU / 磁盘搬进 GPU，forward 完立刻送回 meta。`AlignDevicesHook` 是实现这个生命周期的钩子类。

Loading a 70B parameter model means ~140 GB of weights alone — more than most GPU setups can hold. Accelerate's solution is not "don't load them"; it's "load them just-in-time and free them immediately after". It uses the `meta` device as a zero-memory placeholder (shape and dtype, no actual storage), loads each layer's weights from CPU/disk to GPU just before its forward pass, then immediately returns them to meta. `AlignDevicesHook` is the hook that implements this lifecycle.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/hooks.py`](https://github.com/huggingface/accelerate/blob/main/src/accelerate/hooks.py)

```python
class AlignDevicesHook(ModelHook):
    """
    A hook that ensures inputs and model weights are on the same device
    for forward passes of a given model, potentially offloading the weights
    after the forward pass.
    """

    def __init__(
        self,
        execution_device=None,
        offload=False,
        io_same_device=False,
        weights_map=None,
        offload_buffers=False,
        place_submodules=False,
        skip_keys=None,
        tied_params_map=None,
    ):
        self.execution_device = execution_device
        self.offload = offload
        self.io_same_device = io_same_device
        self.weights_map = weights_map        # maps param name → on-disk or CPU tensor
        self.offload_buffers = offload_buffers
        self.place_submodules = place_submodules
        self.skip_keys = skip_keys
        self.tied_params_map = tied_params_map  # shared-weight tracking to avoid double-offload

    def init_hook(self, module):
        """Called once when the hook is attached to the module."""
        if self.offload:
            # Move all parameters and buffers to 'meta' — zero GPU memory.
            # Their values are preserved in self.weights_map (CPU dict or disk mmap).
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
            ):
                # Skip tied (shared) parameters already tracked to avoid duplication.
                if (
                    self.tied_params_map is not None
                    and id(rgetattr(module, name)) in self.tied_params_map
                ):
                    continue
                set_module_tensor_to_device(module, name, "meta")
        return module

    def pre_forward(self, module, *args, **kwargs):
        """Called just before module.forward()."""
        if self.offload:
            # Restore weights from weights_map to the execution device (GPU/CPU).
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
            ):
                # Handle tied weights: only move the first time we see this object.
                if self.tied_params_map is not None:
                    param_id = id(rgetattr(module, name))
                    if param_id in self.tied_params_map:
                        if self.tied_params_map[param_id] is not None:
                            # Already loaded for another tied module — point at same tensor.
                            set_module_tensor_to_device(
                                module, name, self.execution_device,
                                value=self.tied_params_map[param_id],
                            )
                            continue
                # Load the actual weight from the map and place it on the GPU.
                value = self.weights_map[name]
                set_module_tensor_to_device(
                    module, name, self.execution_device,
                    value=value,
                    fp16_statistics=getattr(value, "fp16_statistics", None),
                )
                # Track for tied-weight deduplication.
                if self.tied_params_map is not None:
                    self.tied_params_map[id(rgetattr(module, name))] = rgetattr(module, name)

        # Also move inputs to the execution device.
        if self.io_same_device:
            self.input_device = find_device(args) or find_device(kwargs)
        if self.execution_device is not None:
            args, kwargs = send_to_device((args, kwargs), self.execution_device)
            args, kwargs = args[0], args[1]
        return args, kwargs

    def post_forward(self, module, output):
        """Called just after module.forward() returns."""
        if self.offload:
            # Return all parameters to meta — free the GPU memory.
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
            ):
                if self.tied_params_map is not None:
                    param_id = id(rgetattr(module, name))
                    if param_id in self.tied_params_map:
                        # Reset the tracking entry for next forward.
                        self.tied_params_map[param_id] = None
                set_module_tensor_to_device(module, name, "meta")

        if self.io_same_device and self.input_device is not None:
            output = send_to_device(output, self.input_device)
        return output

    def detach_hook(self, module):
        """Called when the hook is removed from the module."""
        if self.offload:
            # Restore weights from the map so the module is usable stand-alone again.
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
            ):
                value = self.weights_map.get(name)
                if value is not None:
                    set_module_tensor_to_device(module, name, value.device, value=value)
        return module
```

## 逐行讲解 / What's happening

1. **`weights_map`**:
   - 中文: 一个字典（或 offload_index 指向的磁盘 mmap），以参数名为键，值是实际的 CPU tensor 或磁盘文件 view。这是权重的"冷存储"。
   - English: A dict (or a disk-backed mmap via `offload_index`) mapping param names to actual CPU tensors or disk-file views — the "cold storage" for weights.

2. **`init_hook`: `set_module_tensor_to_device(module, name, "meta")`**:
   - 中文: 把参数替换成 meta tensor。meta device 是 PyTorch 的零内存占位符：只保留形状和 dtype，不分配任何存储空间。这让整个模型在 CPU/GPU 上只占几 KB。
   - English: Replaces each parameter with a meta tensor. The meta device is PyTorch's zero-memory placeholder — shape and dtype only, no storage. This makes the entire model take only a few KB on CPU/GPU.

3. **`pre_forward`: `value = self.weights_map[name]`**:
   - 中文: 从"冷存储"里取出这一层的权重（CPU 读到内存 / 磁盘 mmap），然后 `set_module_tensor_to_device` 把它放到 GPU（或指定设备）。
   - English: Fetches this layer's weights from cold storage (CPU dict read or disk mmap), then `set_module_tensor_to_device` places them on the GPU (or specified execution device).

4. **`tied_params_map`（绑定权重去重）**:
   - 中文: Transformer 模型常见 token embedding 和 output projection 权重共享（tied weights）。如果两处指向同一 tensor，offload/restore 两次会出错。`tied_params_map` 用 object id 做记录，确保相同 tensor 只移动一次。
   - English: Transformer models often tie the token embedding and output projection weights to the same tensor. Moving it twice would corrupt things. `tied_params_map` tracks object ids to ensure shared tensors are only moved once per direction.

5. **`post_forward`: 再次 `set_module_tensor_to_device(module, name, "meta")`**:
   - 中文: forward 完成，立刻把权重送回 meta，释放 GPU 显存。下一层的 pre_forward 会再次从 weights_map 加载。
   - English: As soon as forward finishes, weights go back to meta, freeing GPU memory. The next layer's `pre_forward` will load from `weights_map` again.

6. **`send_to_device(args, ..., self.execution_device)` 在 pre_forward 里**:
   - 中文: 除了权重，输入激活值也要放到正确设备——这是"Align Devices"名称的来源。模型拆分到多 GPU / CPU 时，每一层的输入需要对齐到该层所在设备。
   - English: Beyond weights, input activations also need to be on the right device — this is the "Align" in `AlignDevicesHook`. When a model is split across multiple GPUs/CPU, each layer's input must be moved to that layer's device.

## 类比 / The analogy

想象一个仓库里的工人：每本书（参数）平时锁在书库（meta device，不占台面）。顾客来借书（forward 开始），工人去书库取出来放在阅览桌（GPU）；顾客读完（forward 结束），书立刻收回书库，腾出桌面给下一本书。桌面（显存）永远只放当前正在读的那本书，而不是把整个图书馆搬出来。

Imagine a library where every book (parameter) is normally locked in the storeroom (meta device — no desk space needed). When a reader arrives (forward starts), the librarian retrieves the book and places it on the reading desk (GPU); when the reader finishes (forward ends), the book immediately goes back to storage, freeing the desk for the next book. The desk (GPU memory) only ever holds the book currently being read — not the entire library.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn
from accelerate.hooks import AlignDevicesHook, add_hook_to_module

fc = nn.Linear(64, 64)
weights_map = {
    "weight": fc.weight.detach().clone(),
    "bias":   fc.bias.detach().clone(),
}
hook = AlignDevicesHook(execution_device="cpu", offload=True, weights_map=weights_map)
add_hook_to_module(fc, hook)
hook.init_hook(fc)

print("After init_hook, weight device:", fc.weight.device)  # meta

x = torch.randn(2, 64)
y = fc(x)  # pre_forward loads, post_forward offloads

print("Output shape:", y.shape)
print("Weight device after forward:", fc.weight.device)  # meta again
```

运行 / Run with:
```bash
pip install accelerate torch
python try.py
```

预期输出 / Expected output:
```
After init_hook, weight device: meta
Output shape: torch.Size([2, 64])
Weight device after forward: meta
```

中文：forward 之前和之后权重都在 meta device，只有 forward 执行期间才短暂出现在 CPU。真实场景换成 `execution_device="cuda:0"` 即可实现 GPU offload。

Before and after forward, weights are on the meta device. They only briefly exist on CPU during the forward pass itself. In a real scenario, set `execution_device="cuda:0"` for GPU offloading.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`CpuOffloadHook`（同文件）** / **`CpuOffloadHook` (same file)**: 轻量版本，只做输入设备对齐，不做权重 offload / lightweight variant: only aligns input devices, no weight offloading.
- **`LayerwiseCastingHook`（同文件）** / **`LayerwiseCastingHook` (same file)**: 同样的三段生命周期，但换成"cast to low-precision → forward → cast back" / same three-phase lifecycle, but the action is "cast to low-precision → forward → cast back" instead of offloading.
- **bitsandbytes `LinearFP4` / GPTQ** / **bitsandbytes / GPTQ**: 量化权重存储同样依赖 pre/post forward 的 dequant-on-the-fly，思路完全一致 / quantized-weight storage relies on the same dequant-on-the-fly pre/post-forward pattern.

## 注意事项 / Caveats / when it breaks

- **IO 带宽成为瓶颈** / **IO bandwidth becomes the bottleneck**: 每层 forward 前从磁盘/CPU 搬运权重，带宽不够时推理速度会大幅下降（比全显存慢 5-20x）/ loading weights from disk/CPU before each layer forward makes inference 5–20× slower than full-GPU when bandwidth is limited.
- **`tied_params_map` 必须正确初始化** / **`tied_params_map` must be correctly initialized**: 如果省略，LLaMA 类模型的 embedding/lm_head 共享权重会被 double-offload，forward 中途 tensor 数据丢失 / if omitted, shared embedding/lm_head weights in LLaMA-like models get double-offloaded and silently corrupted mid-forward.
- **只支持叶子模块（或显式设置 `place_submodules=True`）** / **Only handles leaf modules by default (or set `place_submodules=True`)**: 嵌套 module 要逐层挂 hook，`attach_align_device_hook_on_blocks` 会递归帮你做 / nested modules need hooks attached layer-by-layer; `attach_align_device_hook_on_blocks` does this recursively.

## 延伸阅读 / Further reading

- [Accelerate: CPU Offloading](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [PyTorch Meta Device explained](https://pytorch.org/docs/stable/meta.html)
- [HuggingFace Big Model Inference](https://huggingface.co/blog/accelerate-large-models)
