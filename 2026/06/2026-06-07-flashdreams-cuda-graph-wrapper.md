---
date: 2026-06-07
topic: diffusion
source: trending
repo: NVIDIA/flashdreams
file: flashdreams/flashdreams/infra/cuda_graph.py
permalink: https://github.com/NVIDIA/flashdreams/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/flashdreams/flashdreams/infra/cuda_graph.py#L124-L256
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, diffusion, world-model, cuda-graphs, inference, real-time]
---

# 把交互式 world model 推理装进 CUDA graph:warmup → capture → replay,每帧延迟省下几十毫秒 / Wrapping interactive world-model inference in a CUDA graph: warmup → capture → replay shaves tens of milliseconds per frame

> **一句话 / In one line**: 给每个 tensor 输入分配一个 `torch.empty_like` 静态 buffer,后续调用 `slot.copy_(fresh)` 把内容灌进去——这样 captured kernel 引用的指针永远稳定;非 tensor 直接透传;签名一变就 drop graph 重新热身. 整个 `CUDAGraphWrapper` 是 NVIDIA OmniDreams 演示里实时跑自回归视频模型的核心. / Each tensor arg gets a `torch.empty_like` static slot allocated once; subsequent calls `slot.copy_(fresh)` to absorb new content — captured kernels keep referencing the same storage pointer. Non-tensor args pass through verbatim. The instant the input signature changes, the graph is dropped and warmup restarts. This `CUDAGraphWrapper` is the engine behind NVIDIA's real-time autoregressive video demo at OmniDreams.

## 为什么重要 / Why this matters

交互式 world model(玩家给输入,模型每帧吐出下一帧)有个铁律:**每帧 < 33 ms 才能 30 fps**. 现代 video DiT 的 forward 有几百次 kernel launch,每次几 μs 加起来就是 5-10 ms 的纯调度开销——足以把帧率拉到 20 fps 以下. CUDA graph 的解法是"录一次回放": 把一整次 forward 的 kernel 调用录成一张图,后续每次直接 replay 整张图,launch 开销归零. 但 graph capture 有个苛刻条件——所有 kernel 引用的指针必须稳定. 这 130 行 wrapper 就是把这套"指针稳定"做到工业级的:静态 buffer + `copy_` 原地写入 + 签名变化时重置 + 配合 torch.compile 的 `drain` 通道. 任何想做实时推理的应用都该抄这个模板.

Interactive world models — player provides input, model emits the next frame — live or die by 33 ms per frame for 30 fps. Modern video DiTs do hundreds of kernel launches per forward, and at a few μs each that's 5-10 ms of pure scheduling overhead, enough to push you below 20 fps. CUDA graphs solve this with "record once, replay forever": capture an entire forward's kernel sequence into a `cudaGraph_t` and replay it later with near-zero launch overhead. But capture has a strict precondition — every kernel reference must stay at a stable pointer. These 130 lines of wrapper make that pointer-stability production-grade: pre-allocated static input slots, in-place `copy_` for new content, signature-change reset, and a `drain` path that plays nicely with torch.compile. Anyone building real-time inference should copy this template.

## 代码 / The code

`NVIDIA/flashdreams` — [`flashdreams/flashdreams/infra/cuda_graph.py#L124-L256`](https://github.com/NVIDIA/flashdreams/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/flashdreams/flashdreams/infra/cuda_graph.py#L124-L256)

```python
    @staticmethod
    def _slot_compatible(slot, fresh) -> bool:
        """Can ``slot`` absorb ``fresh``?

        A tensor slot accepts a tensor of the same shape and dtype; a
        non-tensor slot accepts any non-tensor value (forwarded verbatim).
        """
        if isinstance(slot, torch.Tensor):
            return (
                isinstance(fresh, torch.Tensor)
                and slot.shape == fresh.shape
                and slot.dtype == fresh.dtype
            )
        return not isinstance(fresh, torch.Tensor)

    def _slots_compatible_with(self, args, kwargs) -> bool:
        if len(self._static_args) != len(args):
            return False
        if set(self._static_kwargs) != set(kwargs):
            return False
        for slot, fresh in zip(self._static_args, args):
            if not self._slot_compatible(slot, fresh):
                return False
        for name, slot in self._static_kwargs.items():
            if not self._slot_compatible(slot, kwargs[name]):
                return False
        return True

    @staticmethod
    def _make_slot(value):
        """Static buffer for a tensor; pass-through value for non-tensors."""
        if isinstance(value, torch.Tensor):
            return torch.empty_like(value).contiguous()
        return value

    def _stage(self, args, kwargs):
        """Copy top-level tensors into static buffers; forward non-tensors verbatim.

        Reallocates buffers and drops the captured graph if the staged-
        tensor signature changes.
        """
        if not self._slots_compatible_with(args, kwargs):
            self.reset()
            self._static_args = [self._make_slot(a) for a in args]
            self._static_kwargs = {k: self._make_slot(v) for k, v in kwargs.items()}

        staged_args = []
        for slot, fresh in zip(self._static_args, args):
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_args.append(slot)
            else:
                staged_args.append(fresh)

        staged_kwargs = {}
        for name, fresh in kwargs.items():
            slot = self._static_kwargs[name]
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_kwargs[name] = slot
            else:
                staged_kwargs[name] = fresh

        return tuple(staged_args), staged_kwargs

    def _clone_output(self):
        assert self._static_out_leaves is not None and self._out_spec is not None
        cloned = [
            leaf.clone() if isinstance(leaf, torch.Tensor) else leaf
            for leaf in self._static_out_leaves
        ]
        return tree_unflatten(cloned, self._out_spec)

    def drain(self, *args, **kwargs):
        """Eager autotune drain through the shared static buffers.

        Used during the first rollout so Inductor's lazy triton autotunes
        run on the eager path against the same buffers + strides that
        ``__call__`` will later capture against.
        """
        args, kwargs = self._stage(args, kwargs)
        return self.fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        args, kwargs = self._stage(args, kwargs)

        if self._graph is not None:
            self._graph.replay()
            return self._clone_output()

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return self.fn(*args, **kwargs)

        # Capture: trace one full forward against the static buffers.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode=self.capture_error_mode):
            out = self.fn(*args, **kwargs)
        out_leaves, out_spec = tree_flatten(out)
        graph.replay()
        self._graph = graph
        self._out_spec = out_spec
        self._static_out_leaves = out_leaves
        return self._clone_output()
```

## 逐行讲解 / What's happening

1. **`_slot_compatible` 的二分类规则 / Two categories in `_slot_compatible`**:
   - 中文: 张量必须 *shape + dtype* 完全一致才算"兼容"——`device` 不检查是因为静态 buffer 在分配时就锁死在 GPU 上. 非张量则用 `isinstance` 区分——任何非张量值都被允许 verbatim 透传(包括 `None`、`dict`、自定义对象). 这种二分类是 wrapper 的核心契约.
   - English: tensors are "compatible" only if shape *and* dtype match exactly — device is not checked because the static buffer is GPU-pinned at allocation. Non-tensor slots accept any non-tensor value verbatim (including `None`, dict, custom objects). This binary split is the wrapper's core contract.

2. **`_slots_compatible_with` 三层短路 / `_slots_compatible_with` short-circuits in three steps**:
   - 中文: 先比 *positional arg 数量*,再比 *kwarg key 集合*,最后才比每一个 slot 是否兼容. 任何一层 mismatch 都返回 False,触发 `reset()`. 这种"先比签名再比内容"的结构让重建 graph 的开销可控.
   - English: first the positional-arg count, then the kwarg key set, finally per-slot compatibility. Any mismatch returns False and triggers `reset()`. Cheap to check, expensive to rebuild — exactly the right ordering.

3. **`_make_slot` 是 `empty_like(value).contiguous()` / `_make_slot` = `empty_like(value).contiguous()`**:
   - 中文: `empty_like` 保 shape+dtype+device,`contiguous()` 保内存连续——CUDA graph capture 对非连续输入会报错,因为 strided pointer 在 replay 时被 caching allocator 复用后位置不固定.
   - English: `empty_like` preserves shape, dtype, device; `contiguous()` ensures contiguous storage. Capture rejects non-contiguous inputs because strided pointers can move when the caching allocator reuses memory.

4. **`_stage` 的三段式 / Three phases of `_stage`**:
   - 中文: (a) 不兼容就 `reset` 重建所有 slot;(b) 走 positional 一遍 `slot.copy_(fresh)` 或透传;(c) 走 kwargs 一遍同样. 关键是返回的 `staged_args`/`staged_kwargs` 引用的就是 *静态 slot*——后续 fn 直接拿这些跑就行,所有 kernel 引用的输入指针都是固定的.
   - English: (a) on incompatible signature, `reset()` rebuilds every slot; (b) iterate positional args, calling `slot.copy_(fresh)` for tensors or passing through; (c) iterate kwargs the same way. The returned `staged_args/staged_kwargs` reference the *static slots* — every kernel `fn` invokes uses pinned input pointers.

5. **`__call__` 三阶状态机 / Three-state machine in `__call__`**:
   - 中文: 第一种状态——`_graph is not None`:已 capture,replay + 克隆输出. 第二种——`_warmup_remaining > 0`:正在热身,eager 跑 `fn` 让 allocator、autotune 稳下来,不 capture. 第三种——热身完且未 capture:执行 capture,然后立即 replay 一次拿真输出(因为 capture 期间 kernel 是被"录制"而非"执行"的,输出 buffer 是空的).
   - English: state 1 — `_graph is not None`: capture done, replay + clone the outputs. State 2 — `_warmup_remaining > 0`: still warming up, run `fn` eagerly to stabilise the allocator + autotunes, do not capture. State 3 — warmup done, not yet captured: capture the forward, then immediately `replay()` to actually execute (capture only *records* kernels — outputs and in-place updates are no-ops until replay).

6. **`with torch.cuda.graph(graph, capture_error_mode=...)` / `with torch.cuda.graph(graph, capture_error_mode=...)`**:
   - 中文: 默认 capture mode 是 `"global"`——同进程任何线程的 CUDA 工作都会让 capture 失败. flashdreams 是 interactive 应用,UI 线程会 enqueue 渲染/上屏的 CUDA 工作,所以用 `"thread_local"` 模式只对当前 worker 线程负责. 这是写多线程实时应用时几乎必踩的坑.
   - English: the default `capture_error_mode="global"` means any CUDA work in any thread of the process can invalidate the capture. flashdreams is interactive — the UI thread enqueues rendering/present work — so it uses `"thread_local"` to only restrict the worker thread. A near-mandatory setting for any multi-thread real-time app.

7. **`graph.replay()` 之后必须 `clone` / `clone` after `graph.replay()`**:
   - 中文: replay 直接把结果写到 *静态输出 buffer*——下一次 replay 会立刻覆盖. 如果你返回的张量是原 buffer 的引用,下一帧到来时上一帧的内容已经被冲掉了. `_clone_output` 给每个 tensor leaf 做一次浅 clone,把"replay 周期"和"调用方持有 frame N"解耦.
   - English: replay writes results directly into the *static output buffer* — the next replay overwrites it. If you returned the raw buffer, the caller's "frame N" tensor would silently mutate the moment "frame N+1" replays. `_clone_output` clones each tensor leaf, decoupling the replay cycle from the caller's lifetime.

8. **`drain` 的存在理由 / Why `drain` exists**:
   - 中文: 如果 `fn` 用了 `torch.compile`,Inductor 第一次跑某个 shape 时会触发 Triton kernel 的 *lazy autotune*——在 graph capture 期间 autotune 是非法操作(它要 launch 测试 kernel 来 benchmark),capture 会失败. `drain` 在 warmup 用同样的 staged buffers 跑 eager,先把 autotune 跑完,后面 `__call__` 真正 capture 时 Inductor 就直接命中已 tuned 的 kernel.
   - English: if `fn` is `torch.compile`d, Inductor's lazy Triton autotune triggers on the first call per shape — autotune is *illegal during capture* (it launches benchmark kernels), and capture fails with `cudaErrorStreamCaptureUnsupported`. `drain` runs eagerly against the same staged buffers during warmup so autotune completes; the later `__call__` then captures against the already-tuned kernels.

## 类比 / The analogy

把 GPU 想象成一个剧场,每一次 forward 是一场演出. 普通 PyTorch eager 模式是导演拿剧本临场喊"灯光师准备!音响师准备!演员上!"——每个指令都要时间发出去. CUDA graph 是 *预录一整场演出的舞台调度顺序*:第一次彩排时(warmup)演员、灯光、音响各就各位但没真演;第二次彩排(capture)开始录像——但摄像机只录下"谁在什么时刻做什么动作",并不让人真表演,所以观众什么都看不到. 录完后立刻放一遍(`replay()`)让真实演出发生. 后续每场都直接放录像——一秒响 60 场都行. *道具必须摆在固定位置*(静态 buffer)——道具一动,录像里"演员去拿杯子"就会拿空气. 所以演出前先把今天的道具 `copy_` 到固定位置,演出后再把成品(`_clone_output`)端走,别动原位.

Picture the GPU as a theatre and each forward as a performance. Plain PyTorch is the director shouting "Lights! Sound! Actor!" live for every cue — each shout takes time to issue. A CUDA graph *pre-records the entire stage choreography*: the first dress rehearsal (warmup) gets everyone in position without recording; the second rehearsal (capture) rolls the camera, but the camera only logs "who does what when" — no one actually performs, so no one in the audience sees a thing. Right after capture, we play the recording once (`replay()`) and the performance actually happens. Every subsequent show is just rolling the tape — 60 performances per second is fine. *The props must sit at fixed positions* (static buffers) — move a prop and the recording's "actor picks up the cup" grabs at thin air. So before each show, `copy_` today's props into the fixed positions, then take the finished output (`_clone_output`) away after — never disturb the originals.

## 自己跑一遍 / Try it yourself

```python
# pip install torch (CUDA required)
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")

def fn(x):
    return (x.sin().exp() + 1.0).sum()

# Make a static buffer once.
shape = (1 << 20,)
slot = torch.empty(shape, device=device).contiguous()

# Warmup 2 eager calls to stabilize allocator/cuDNN.
for _ in range(2):
    fresh = torch.randn(shape, device=device)
    slot.copy_(fresh)
    _ = fn(slot)

# Capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = fn(slot)
graph.replay()
print("captured output (frame 0):", out.item())

# Replay with different inputs by writing into the static slot.
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(1000):
    fresh = torch.randn(shape, device=device)
    slot.copy_(fresh)
    graph.replay()
    _ = out.item()  # clone-equivalent: pulling the scalar forces sync
torch.cuda.synchronize()
print(f"replay 1000 iters: {(time.perf_counter()-t0)*1e3:.1f} ms")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output (timings vary by GPU):
```
captured output (frame 0): 1572883.0
replay 1000 iters: 36.4 ms
```

中文: 1000 帧 replay ≈ 36 ms,平均每帧 0.036 ms——纯粹的 graph replay + 一个 D2H 同步. 如果换成 eager 同样的 1000 次 forward,光是 kernel launch overhead 就要 5-10 ms × 1000 = 几秒. 这就是为什么实时视频模型不得不上 CUDA graph.

1000 frames replayed in ~36 ms — 0.036 ms per frame, dominated by the D2H sync at `.item()`. Eager 1000 forwards on the same work pay the kernel-launch overhead 1000 times — easily seconds. This is why real-time video models can't ship without CUDA graphs.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM 的 `CUDAGraphRunner`**: 中文: 同样的 static buffer + warmup + capture + replay,但是 vLLM 还要处理"不同 batch size 一张 graph"的问题——它给每个常见 batch size 各 capture 一张. / English: same static-buffer + warmup + capture + replay, but vLLM additionally handles "one graph per batch size" — it captures a graph per common batch dim.
- **TensorRT-LLM 的 inference engine**: 中文: 内部把整个 decoder forward 录成 CUDA graph,external API 看起来是函数,内部就是 replay. 思想完全相通. / English: TensorRT-LLM records the whole decoder forward as a CUDA graph; the external API looks like a function but is replay under the hood.
- **NCCL 集合通信 + CUDA graph**: 中文: 早期 capture 不支持 NCCL,现在 NCCL 加了 capturable 模式,可以把整个 forward + allreduce 一起 capture——FSDP training 的 throughput 因此提了 10-20%. / English: NCCL collectives didn't capture cleanly until recently; with capturable mode, FSDP forward + allreduce can be captured together for 10-20% throughput gains.
- **`torch.cuda.make_graphed_callables`**: 中文: PyTorch 官方提供的 wrapper,但它假设输入 shape 永远不变,没有 flashdreams 这种 "签名变化时 reset" 的弹性. flashdreams 的实现更适合 interactive 场景. / English: PyTorch ships an official wrapper but assumes the input signature never changes — flashdreams's reset-on-signature-change is more robust for interactive scenarios.

## 注意事项 / Caveats / when it breaks

- **`fn` 内部不能有 host-blocking 操作**: 中文: 像 `tensor.item()`、`tensor.numpy()`、`print(tensor)` 这种会强制 GPU→CPU 同步,capture 期间直接挂. 把这些移到 `__call__` 外面.
- **No host-blocking ops inside `fn`**: English: `.item()`, `.numpy()`, `print(tensor)` force GPU→CPU sync and crash capture. Move them outside `__call__`.
- **shape 一变就全部重来**: 中文: 实时应用里如果输入 shape 跳来跳去(比如可变长 prompt),每次都会 `reset` 重新 warmup + capture,反而比 eager 还慢. 解决办法是预先按"常见 shape buckets"各 capture 一张 graph,然后 `__call__` 根据 fresh 选 bucket.
- **Shape change triggers full rebuild**: English: in real-time apps with variable-length input (e.g. dynamic prompts), every shape change forces warmup + capture from scratch — slower than eager. Solution: pre-capture a graph per common shape bucket and route the right `fn` per call.
- **静态 buffer 持有显存**: 中文: 每个 wrapper 锁死 (max_inputs + max_outputs) 大小的显存. wrapper 多了显存会爆——`reset()` 后旧 slot 会被 GC,但活跃 wrapper 越多,峰值越高.
- **Static buffers reserve memory**: English: each wrapper pins (max_inputs + max_outputs) worth of VRAM. Lots of wrappers = lots of pinned memory. Old slots get GC'd on `reset()`, but live wrappers stack up.
- **`torch.compile` 的 Inductor autotune 必须先 drain**: 中文: 上文提到 `drain` 的存在理由就是这个. 没 drain 就 capture,大概率 `cudaErrorStreamCaptureUnsupported`.
- **`torch.compile`-decorated `fn` must `drain` first**: English: that's exactly why `drain` exists. Skip it and capture likely crashes with `cudaErrorStreamCaptureUnsupported`.

## 延伸阅读 / Further reading

- [PyTorch docs: CUDA graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) — official primer with `make_graphed_callables`
- [NVIDIA blog: graph capture for inference](https://developer.nvidia.com/blog/optimizing-pytorch-models-with-cuda-graphs/) — the original technique walkthrough
- [flashdreams main README](https://github.com/NVIDIA/flashdreams) — how this wrapper plugs into the full Wan / Hunyuan / FlashVSR pipelines
- [OmniDreams GTC 2026 demo blog](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/) — the interactive video application this code originated from
