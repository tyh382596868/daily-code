---
date: 2026-09-20
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/worker/workspace.py
permalink: https://github.com/vllm-project/vllm/blob/1c3ef2ad4cabfbc2e5d74494b0b2bf8c3cb3f936/vllm/v1/worker/workspace.py#L116-L225
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vllm, gpu-memory, workspace, batching]
---

# vLLM workspace lanes：一块平坦 buffer 服务多个临时张量 / vLLM Workspace Lanes: One Flat Buffer for Many Temporary Tensors

> **一句话 / In one line**: vLLM 把同一个 micro-batch/lane 的临时张量放进可增长的对齐 byte buffer，再返回多个 typed view，减少反复申请 GPU workspace。 / vLLM stores temporary tensors for one micro-batch/lane in an aligned, growable byte buffer and returns typed views, reducing repeated GPU workspace allocations.

## 为什么重要 / Why this matters

Serving 系统里，workspace 不是模型参数，却会在每个 kernel 或 attention 路径里反复出现。若每次都申请不同 shape 的临时 tensor，CUDA caching allocator 可能留下越来越多 reserved segment，峰值显存和碎片一起上升。`WorkspaceManager` 用一次分配承载同时存在的多个 view，把“小块申请”变成“按 lane 懒增长”。

In a serving system, workspace is not model state, but it appears repeatedly in kernels and attention paths. Allocating a different temporary tensor each time can leave reserved segments in the CUDA caching allocator, increasing both peak memory and fragmentation. `WorkspaceManager` turns many small allocations into one lazily growing allocation per lane.

这里最值得学的不是 byte buffer 本身，而是三个契约：每个 view 按 256 bytes 对齐、增长只影响请求中的 `(ubatch, lane)`、锁定后任何新需求都必须显式失败。这样别的 lane 仍能持有旧 view，而系统也能在 warmup 阶段把容量冻结。

The important lesson is not merely the byte buffer. It is the contract: each view is aligned to 256 bytes, growth touches only the requesting `(ubatch, lane)`, and any new request after locking fails loudly. Other lanes can keep their old views, while warmup can freeze capacity before production.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/worker/workspace.py`](https://github.com/vllm-project/vllm/blob/1c3ef2ad4cabfbc2e5d74494b0b2bf8c3cb3f936/vllm/v1/worker/workspace.py#L116-L225)

```python
def get_simultaneous(
    self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
) -> list[torch.Tensor]:
    """Get multiple workspace tensors simultaneously from a single allocation.

    Args:
        *shapes_and_dtypes: One or more (shape, dtype) tuples.

    Returns:
        List of tensor views into the workspace buffer, one per shape/dtype pair.

    """
    actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
    aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
    total_bytes = sum(aligned_bytes)

    # Calculate cumulative offsets using itertools.accumulate
    offsets = list(accumulate([0] + aligned_bytes[:-1]))

    current_workspace = self._ensure_workspace_size(total_bytes)

    return [
        current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
        .view(shapes_and_dtypes[i][1])
        .reshape(shapes_and_dtypes[i][0])
        for i in range(len(shapes_and_dtypes))
    ]

def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
    """Ensure workspace is allocated and large enough, return current workspace."""
    ubatch_id = dbo_current_ubatch_id()
    lane = _workspace_lane.get()
    if lane >= self._num_lanes:
        raise RuntimeError(
            f"Workspace lane {lane} is not configured; manager has "
            f"{self._num_lanes} lane(s)."
        )
    workspace_id = ubatch_id * self._num_lanes + lane
    current_workspace = self._current_workspaces[workspace_id]
    current_size = self._workspace_size_bytes(current_workspace)

    if current_size < required_bytes:
        if self._locked:
            raise AssertionError(
                f"Workspace is locked but allocation requires "
                f"{required_bytes / _MB:.2f} MB, current size is "
                f"{current_size / _MB:.2f} MB. "
                "Workspace growth is not allowed after locking."
            )

        # Only resize the requesting ubatch/lane workspace. Other slots
        # resize lazily on their next get_simultaneous call.
        self._current_workspaces[workspace_id] = None
        del current_workspace
        torch.accelerator.empty_cache()
        self._current_workspaces[workspace_id] = torch.empty(
            (required_bytes,), dtype=torch.uint8, device=self._device
        )
        current_workspace = self._current_workspaces[workspace_id]

    return current_workspace
```

## 逐行讲解 / What's happening

1. **实际大小和对齐大小 / Actual versus aligned bytes**:
   - 中文: `_compute_bytes` 只计算真实 payload，`round_up(..., 256)` 给相邻 view 留出稳定的对齐边界。
   - English: `_compute_bytes` measures the real payload, while `round_up(..., 256)` creates stable aligned boundaries between neighboring views.
2. **累积 offset / Cumulative offsets**:
   - 中文: `accumulate` 把多个 allocation 变成一个线性地址表；每个 slice 只取自己的真实 bytes，再恢复 dtype 和 shape。
   - English: `accumulate` turns many allocations into a linear address table; each slice takes only its real bytes, then restores dtype and shape.
3. **lane 隔离 / Lane isolation**:
   - 中文: `workspace_id = ubatch_id * num_lanes + lane` 把 micro-batch 和并行 lane 映射到独立槽位，避免 resize 互相踢掉 view。
   - English: `workspace_id = ubatch_id * num_lanes + lane` maps each micro-batch/lane pair to its own slot, so one resize does not invalidate another lane's view.
4. **懒增长 / Lazy growth**:
   - 中文: 只有当前请求的槽位增长，其他槽位等到下一次真正需要时再增长。
   - English: Only the slot used by the current request grows; other slots wait until they actually need more capacity.
5. **释放 allocator segment / Releasing allocator segments**:
   - 中文: 重新分配前调用 `empty_cache()`，让已释放的 segment 回到 CUDA caching allocator，避免连续 resize 抬高 reserved memory。
   - English: `empty_cache()` before reallocating returns freed segments to the CUDA caching allocator and avoids reserved-memory growth across repeated resizes.
6. **锁定契约 / Locking contract**:
   - 中文: warmup 完成后可锁住 manager；任何漏估的 workspace 需求都会以调用方信息报错，而不是静默抖动显存。
   - English: After warmup, the manager can be locked; an underestimated workspace request reports its caller instead of silently perturbing memory usage.

## 类比 / The analogy

把 workspace 想成仓库里每条生产线自己的托盘。托盘先按最大的已知订单扩容，订单内部的零件只是在同一个托盘上划分位置；不同生产线不抢同一个托盘，生产稳定后还可以封存尺寸。

Think of workspace as a dedicated tray for each production lane. The tray grows to the largest known order, individual parts occupy slices of the same tray, and different lanes never steal each other's tray. Once production is stable, the tray size can be frozen.

## 自己跑一遍 / Try it yourself

```python
import torch

def views(*specs):
    sizes = [torch.empty(shape, dtype=dtype).numel() * torch.empty((), dtype=dtype).element_size()
             for shape, dtype in specs]
    aligned = [((n + 255) // 256) * 256 for n in sizes]
    buf = torch.empty(sum(aligned), dtype=torch.uint8)
    offset = 0
    out = []
    for (shape, dtype), n in zip(specs, sizes):
        out.append(buf[offset:offset + n].view(dtype).reshape(shape))
        offset += ((n + 255) // 256) * 256
    return out

a, b = views(((4, 8), torch.float32), ((3,), torch.int64))
print(a.shape, b.dtype, a.untyped_storage().data_ptr() <= b.untyped_storage().data_ptr())
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
torch.Size([4, 8]) torch.int64 True
```

中文: 两个 tensor 共享一个 byte buffer，但拥有各自的 dtype、shape 和对齐起点。
English: Both tensors share one byte buffer while keeping independent dtype, shape, and aligned starting offsets.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch memory planning** / **PyTorch memory planning**: 中文: 编译器也会把临时值安排到复用的 storage 区间，核心思想都是“生命周期不重叠就复用”。 / English: Compilers similarly place temporaries into reusable storage intervals; the common idea is to reuse memory when lifetimes do not overlap.
- **FlashAttention workspace** / **FlashAttention workspace**: 中文: fused kernel 常把多个中间结果打包进一个临时区域，减少 launch 之间的 allocator 交互。 / English: Fused kernels often pack intermediates into one temporary region to reduce allocator interaction between launches.
- **CUDA graph capture** / **CUDA graph capture**: 中文: capture 要求地址和 shape 稳定，因此预分配 workspace 是把动态图变成可复用执行图的基础。 / English: Capture needs stable addresses and shapes, so preallocated workspace is a foundation for reusable CUDA graphs.

## 注意事项 / Caveats / when it breaks

- **view 生命周期** / **View lifetime**: 中文: resize 会替换当前 lane 的 backing tensor；不要让旧 view 跨越 resize 继续使用。 / English: Resizing replaces the backing tensor for that lane; do not use an old view across a resize.
- **对齐不是形状契约** / **Alignment is not a shape contract**: 中文: 256-byte 对齐解决地址布局，不会自动保证 view 的维度或 dtype 合法。 / English: 256-byte alignment solves address layout, not the validity of a view's dimensions or dtype.
- **锁定前要完整 warmup** / **Warm up before locking**: 中文: 生产流量遇到新 shape 后才发现容量不足，会从可控的 assert 变成线上错误。 / English: Discovering a new shape only after production traffic starts turns a controlled assertion into an outage.

## 延伸阅读 / Further reading

- [vLLM workspace manager](https://github.com/vllm-project/vllm/blob/1c3ef2ad4cabfbc2e5d74494b0b2bf8c3cb3f936/vllm/v1/worker/workspace.py#L116-L225)
- [PyTorch CUDA memory management](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [CUDA Graphs](https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)
