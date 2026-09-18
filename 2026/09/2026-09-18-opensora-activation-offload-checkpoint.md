---
date: 2026-09-18
topic: diffusion
source: tracked
repo: hpcaitech/Open-Sora
file: opensora/acceleration/checkpoint.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/acceleration/checkpoint.py#L17-L85
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, activation-checkpointing, cpu-offload, pinned-memory]
---

# Open-Sora activation offload：把 checkpoint 的中间结果暂存到 CPU / Open-Sora Activation Offload: Park Checkpoint Activations on CPU

> **一句话 / In one line**: Open-Sora 用一个 pinned-CPU arena 和后进先出的 tensor 栈，把 checkpoint 激活从 GPU 显存挪出去，再在 backward 前按顺序搬回来。 / Open-Sora combines a pinned CPU arena with a LIFO tensor stack to move checkpoint activations out of GPU memory and restore them in backward order.

## 为什么重要 / Why this matters

视频扩散模型的 activation 很大，尤其是 3D VAE、DiT 和长视频训练。普通 activation checkpointing 已经会用“重新计算”换显存，但保存到 CPU 的版本还能进一步把峰值显存压下来。代价是 PCIe 或 NVLink 传输，以及对保存顺序的严格要求。

Video diffusion activations are enormous, especially inside 3D VAEs, DiTs, and long-clip training. Ordinary activation checkpointing trades memory for recomputation; this variant goes further by parking saved tensors on CPU. The price is transfer bandwidth and a strict requirement that reloads follow the same stack order as offloads.

这段实现值得学的地方不只是 `pin_memory=True`。它把一块连续 buffer 当成 activation arena，用 offset 做 bump allocation，再用 `tensor_id_queue` 记录“谁最后被搬走”。这让多个 checkpoint 共享输入时也能避免重复搬运。

The interesting part is not just `pin_memory=True`. The code treats one contiguous buffer as an activation arena, uses an offset as a bump allocator, and records the last offloaded tensor in `tensor_id_queue`. That gives shared inputs a way to avoid being copied twice.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/acceleration/checkpoint.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/acceleration/checkpoint.py#L17-L85)

```python
class ActivationManager:
    def __init__(self):
        self.enable = False
        self.buffer = None
        self.total_size = 0
        self.avail_offset = 0
        self.tensor_id_queue = []
        self.ignore_tensor_id_set = set()

    def setup_buffer(self, numel: int, dtype: torch.dtype):
        self.buffer = torch.empty(numel, dtype=dtype, pin_memory=True)
        self.total_size = numel
        self.enable = True

    def offload(self, x: torch.Tensor) -> None:
        if not self.enable or id(x) in self.ignore_tensor_id_set:
            return
        size = x.numel()
        if self.avail_offset + size > self.total_size:
            raise RuntimeError("Activation buffer is full")
        assert x.dtype == self.buffer.dtype, f"Wrong dtype of offload tensor"
        cpu_x = self.buffer[self.avail_offset : self.avail_offset + size].view_as(x)
        cpu_x.copy_(x)
        x.data = cpu_x
        self.avail_offset += size
        self.tensor_id_queue.append(id(x))

    def onload(self, x: torch.Tensor) -> None:
        if not self.enable or id(x) in self.ignore_tensor_id_set:
            return
        assert self.tensor_id_queue[-1] == id(x), f"Wrong order of offload/onload"
        assert x.data.is_pinned()
        x.data = x.data.to(get_current_device(), non_blocking=True)
        self.tensor_id_queue.pop()
        self.avail_offset -= x.numel()
        if len(self.tensor_id_queue) == 0:
            self.ignore_tensor_id_set.clear()

    def add_ignore_tensor(self, x: torch.Tensor) -> None:
        self.ignore_tensor_id_set.add(id(x))

    def is_top_tensor(self, x: torch.Tensor) -> bool:
        return len(self.tensor_id_queue) > 0 and self.tensor_id_queue[-1] == id(x)


GLOBAL_ACTIVATION_MANAGER = ActivationManager()


class CheckpointFunctionWithOffload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        for x in args[::-1]:
            if GLOBAL_ACTIVATION_MANAGER.is_top_tensor(x):
                GLOBAL_ACTIVATION_MANAGER.onload(x)
                GLOBAL_ACTIVATION_MANAGER.add_ignore_tensor(x)
        out = CheckpointFunction.forward(ctx, run_function, preserve_rng_state, *args)
        for x in args:
            if torch.is_tensor(x):
                GLOBAL_ACTIVATION_MANAGER.offload(x)
        return out

    @staticmethod
    def backward(ctx, *args):
        for tensor in ctx.saved_tensors[::-1]:
            GLOBAL_ACTIVATION_MANAGER.onload(tensor)
        return CheckpointFunction.backward(ctx, *args)
```

## 逐行讲解 / What's happening

1. **第 17-24 行 / Lines 17-24 (`ActivationManager.__init__`)**:
   - 中文: `buffer` 是共享的 CPU 存储，`avail_offset` 是下一个空闲位置；`tensor_id_queue` 则把激活保存成一个栈。
   - English: `buffer` is shared CPU storage, `avail_offset` is the next free slot, and `tensor_id_queue` turns saved activations into a stack.
2. **第 26-29 行 / Lines 26-29 (`setup_buffer`)**:
   - 中文: pinned memory 允许 CUDA 使用更高效的异步 host-to-device copy。这里先分配整块空间，后续只做切片和复制。
   - English: Pinned memory enables more efficient asynchronous host-to-device copies. The manager allocates one arena up front and later uses slices into it.
3. **第 31-42 行 / Lines 31-42 (`offload`)**:
   - 中文: 先检查容量和 dtype，再把 `x` 拷贝到 arena 的对应切片，并把 `x.data` 指向 CPU view。`offset` 单调增加，像一个很轻量的 bump allocator。
   - English: The method checks capacity and dtype, copies `x` into an arena slice, and points `x.data` at the CPU view. The offset advances like a tiny bump allocator.
4. **第 44-54 行 / Lines 44-54 (`onload`)**:
   - 中文: 这里故意要求当前 tensor 必须是栈顶元素。恢复后 offset 回退，意味着这块空间可以立刻被下一个 activation 重用。
   - English: The method deliberately requires the tensor to be the stack top. After restoring it, the offset moves backward, so that storage can be reused immediately.
5. **第 66-85 行 / Lines 66-85 (`CheckpointFunctionWithOffload`)**:
   - 中文: forward 先处理 checkpoint 之间共享的输入，再执行普通 checkpoint，最后把输入放回 CPU；backward 反向遍历 saved tensors，严格遵守 LIFO。
   - English: Forward handles inputs shared across checkpoints, runs the regular checkpoint, then offloads tensors. Backward reloads saved tensors in reverse order, preserving LIFO discipline.

## 类比 / The analogy

可以把 GPU 显存想成厨房的操作台，把 pinned CPU buffer 想成旁边的一排带编号抽屉。厨师做菜时只把当前需要的食材放在台面上，暂时不用的食材按顺序放进抽屉。取回时必须先拿最后放进去的那一盒，否则编号和内容就对不上。

Think of GPU memory as a kitchen counter and the pinned CPU buffer as numbered drawers beside it. Ingredients not needed right now go into drawers in order. When the cook asks for them back, the last drawer used must be opened first, or the labels no longer match the computation.

## 自己跑一遍 / Try it yourself

```python
class Arena:
    def __init__(self, capacity):
        self.buf = [None] * capacity
        self.top = 0
        self.stack = []

    def offload(self, name, values):
        end = self.top + len(values)
        if end > len(self.buf):
            raise RuntimeError("buffer full")
        self.buf[self.top:end] = values
        self.stack.append((name, self.top, end))
        self.top = end

    def onload(self, name):
        saved, start, end = self.stack.pop()
        assert saved == name
        values = self.buf[start:end]
        self.top = start
        return values

arena = Arena(8)
arena.offload("a", [1, 2])
arena.offload("b", [3, 4, 5])
print(arena.onload("b"), arena.onload("a"), arena.top)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[3, 4, 5] [1, 2] 0
```

中文: 注意 `top` 在最后一次恢复后回到 0；这就是“栈式释放”带来的空间复用。  
English: Notice that `top` returns to zero after the final reload; stack-shaped release makes the arena reusable.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch activation checkpointing** / **PyTorch activation checkpointing**: 中文: 通过重算减少保存量，但默认不负责把 activation 放到 CPU。 / English: It reduces saved state through recomputation, but does not by itself provide this CPU arena.
- **torchtune activation offload** / **torchtune activation offload**: 中文: 用 saved-tensor hooks 在保存和恢复边界搬运 activation，抽象更通用。 / English: It uses saved-tensor hooks around save and restore boundaries, offering a more general integration point.
- **FlashAttention workspace reuse** / **FlashAttention workspace reuse**: 中文: 同样依赖预分配 workspace 和明确的生命周期，只是保存对象从 activation 换成 kernel 中间量。 / English: It similarly relies on preallocated workspace and explicit lifetimes, but manages kernel intermediates instead of autograd activations.

## 注意事项 / Caveats / when it breaks

- **只能容纳预估大小** / **Capacity is a hard limit**: 中文: `setup_buffer` 太小会在训练中途直接报错，生产实现应根据模型配置和 checkpoint 深度做容量规划。 / English: An undersized arena fails during training; production code should size it from model configuration and checkpoint depth.
- **dtype 必须一致** / **Dtypes must match**: 中文: 这份实现不在 offload 时转换 dtype，否则 view 的元素数和字节布局会失配。 / English: The implementation does not convert dtype during offload, so mismatches would invalidate the view and byte layout.
- **LIFO 不是装饰** / **LIFO is a contract**: 中文: checkpoint 嵌套、共享输入或异常退出时若破坏栈序，恢复的 tensor 可能完全错误。 / English: Nested checkpoints, shared inputs, or exceptional exits must preserve stack order or the wrong tensor can be restored.
- **`.data` 要谨慎** / **Use `.data` carefully**: 中文: 这是低层内存优化代码，不能照搬到普通业务模块；要配合 autograd 语义和充分的梯度测试。 / English: This is low-level memory code, not a general application pattern; it needs autograd-aware design and gradient tests.

## 延伸阅读 / Further reading

- [Open-Sora checkpoint implementation](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/acceleration/checkpoint.py)
- [PyTorch activation checkpoint documentation](https://pytorch.org/docs/stable/checkpoint.html)
- [Pinned memory and asynchronous CUDA copies](https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers)
