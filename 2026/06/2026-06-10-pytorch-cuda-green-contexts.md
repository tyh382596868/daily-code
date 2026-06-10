---
date: 2026-06-10
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/cuda/green_contexts.py
permalink: https://github.com/pytorch/pytorch/blob/56dc9a469c0d3e1a3c2e303c7dab35200ecabe6b/torch/cuda/green_contexts.py#L1-L92
difficulty: advanced
read_time: ~9 min
tags: [code-of-the-day, pytorch, cuda, multi-tenant, gpu-partitioning]
---

# PyTorch 把"一张 GPU 切成多张"写进了官方:92 行的 Green Context wrapper / PyTorch shipped "slice one GPU into many" to core — a 92-line Green Context wrapper

> **一句话 / In one line**: 不是 MPS、不是 stream,而是从驱动级别就把 GPU 的 SM 划分成几个"绿区",每个区里跑一个模型,SM 和工作队列都不抢 —— PyTorch 给这套 CUDA 12.4+ 新能力包了 92 行 Python。 / Not MPS, not CUDA streams — at the driver level the GPU's SMs are partitioned into isolated "green" regions, each running its own model, with no SM or workqueue contention. PyTorch wraps this brand-new CUDA 12.4+ capability in 92 lines of Python.

## 为什么重要 / Why this matters

多租户推理(multi-tenant inference)长期有个尴尬:同一张 H100 上跑 4 个小模型,4 个 cudaStream 并发提交时,SM 的分配完全由驱动调度。结果是吞吐"看起来"高,实际 latency 抖得厉害 —— 因为大 kernel 把 SM 抢光,小 kernel 排队等。MPS 解决了一部分(进程级)但配置繁琐;cudaStream priority 太粗。CUDA 12.4 引入了 **Green Context**:从驱动层就把 SM 切成 N 块,每块还可以配自己的 workqueue,跑在不同 Green Context 上的 kernel 物理上不抢 SM。PyTorch 这 92 行就是把这套 API 在 Python 里搬了一遍 —— 现在可以一行代码"给我切出 48 个 SM 给小 policy 用,剩下 84 个给大 VLM"。

Multi-tenant inference has had a long-standing annoyance: when four small models share one H100 via four CUDA streams, SM allocation is entirely up to the driver. Throughput looks high in aggregate but tail latency is jittery — big kernels grab all the SMs and small ones queue up. MPS partly solves this (process-level) but is fiddly; CUDA stream priorities are too coarse. CUDA 12.4 introduced **Green Contexts**: at the driver level you can carve the GPU's SMs into N disjoint regions, each with its own workqueue, and kernels in different green contexts physically do not contend for SMs. This 92-line file is PyTorch's Python surface for that capability — one line and you've reserved 48 SMs for a small policy and 84 for a big VLM.

## 代码 / The code

`pytorch/pytorch` — [`torch/cuda/green_contexts.py`](https://github.com/pytorch/pytorch/blob/56dc9a469c0d3e1a3c2e303c7dab35200ecabe6b/torch/cuda/green_contexts.py#L1-L92)

```python
import torch

__all__ = ["GreenContext"]

_GreenContext = object
SUPPORTED = False

if hasattr(torch._C, "_CUDAGreenContext"):
    _GreenContext = torch._C._CUDAGreenContext  # type: ignore[misc]
    SUPPORTED = True


class GreenContext(_GreenContext):
    r"""Wrapper around a CUDA green context."""

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> _GreenContext:
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.create(
            device_id=device_id,
            num_sms=num_sms,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
        )

    @staticmethod
    def max_workqueue_concurrency(device_id: int | None = None) -> int:
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.max_workqueue_concurrency(device_id=device_id)

    def set_context(self) -> None:
        return super().set_context()

    def pop_context(self) -> None:
        return super().pop_context()

    def Stream(self) -> "torch.cuda.Stream":
        return super().Stream()
```

## 逐行讲解 / What's happening

1. **第 8-13 行 / Lines 8-13 (capability probing)**:
   - 中文: 不是每个 PyTorch build 都启用了 Green Context(需要编译时链接 CUDA 12.4+)。这里用 `hasattr(torch._C, "_CUDAGreenContext")` 探一下 C++ 端有没有暴露这个类。没暴露就保留 `_GreenContext = object`、`SUPPORTED = False`,后续 `create()` 会显式抛错。这是 PyTorch 处理"可选 CUDA 特性"的标准模板。
   - English: not every PyTorch build links Green Context support (requires CUDA 12.4+ at build time). The shim probes whether C++ exposes the type via `hasattr(torch._C, "_CUDAGreenContext")`. If not, `_GreenContext = object` and `SUPPORTED = False` stay, and any later `create()` raises with a clear message. This is PyTorch's standard pattern for "optional CUDA feature".

2. **`class GreenContext(_GreenContext)` 这个继承 / the inheritance**:
   - 中文: 这一行很有意思 —— 当 SUPPORTED 时,`_GreenContext` 是真正的 C++ 类型;否则它是 `object`。所以这个 Python `GreenContext` 类既是 C++ 类的子类(继承所有原生方法),又能在不支持的环境下被 import(不会因为 base class 不存在而 import 失败)。文档里那条"Python shim helps Sphinx process docstrings more reliably"暗示了第二个理由:Sphinx 不能从 C++ 类型里抽 docstring,所以加一层 Python 子类专门挂 docstring。
   - English: this one line is doing double duty. When SUPPORTED, `_GreenContext` is the real C++ type; otherwise it's `object`. So the Python `GreenContext` inherits all native methods *when available*, yet can still be imported on builds that lack the feature (no broken base class). The comment "Python shim helps Sphinx process docstrings more reliably" hints at the second motivation: Sphinx can't pull docstrings out of pure C++ types, so the Python subclass exists just to attach them.

3. **`create(num_sms=..., workqueue_scope=..., ...)`**:
   - 中文: 这是核心 API。`num_sms` 是直接告诉驱动"给我切 48 个 SM"(H100 有 132 个 SM,你可以切 48+84,或者三块 44)。`workqueue_scope="balanced"` 意思是"我这个绿区的 workqueue 不和其他 balanced 绿区共享" —— 配合 `workqueue_concurrency_limit` 能限制单绿区里 stream-ordered 工作的并发数,适合给延迟敏感的小模型限流。`workqueue_scope="device_ctx"` 是默认 driver 行为(共享)。
   - English: this is the workhorse. `num_sms` tells the driver "carve out 48 SMs for me" (H100 has 132 SMs; you could split 48 + 84, or three slabs of 44). `workqueue_scope="balanced"` means "my workqueue is disjoint from other balanced contexts'" — combined with `workqueue_concurrency_limit` it lets you cap concurrent stream-ordered workloads in this green context, ideal for rate-limiting a latency-sensitive small model. `workqueue_scope="device_ctx"` is the driver default (shared).

4. **`max_workqueue_concurrency(device_id)`**:
   - 中文: 启动绿区前先查"这个 GPU 最多支持多少并发 workqueue 资源" —— 因为这是有限的硬件资源(每张卡的上限不一样)。设置 `workqueue_concurrency_limit` 超过这个值会失败。
   - English: query the device's max concurrent workqueue resources before creating a green context — it's a finite hardware resource (varies per card). Setting `workqueue_concurrency_limit` above this fails.

5. **`set_context() / pop_context()`**:
   - 中文: 这就是 CUDA context 栈的 push/pop。`set_context()` 把当前绿区压栈成"当前 CUDA context",后面所有 cuda 调用都进这个绿区;`pop_context()` 弹出,恢复之前的 context。注意:从 PyTorch 用户视角,你一般不会直接调用这俩 —— 而是用下面的 `Stream()` 拿到一个绑定在绿区上的 stream,再用 `with torch.cuda.stream(s):` 走老路。
   - English: this is CUDA's context stack push/pop. `set_context()` pushes this green context as "the current CUDA context"; subsequent CUDA calls run there. `pop_context()` restores the previous context. Note: from a PyTorch user's perspective you usually won't call these directly — you grab a `Stream()` bound to the green context (next method) and use it via the familiar `with torch.cuda.stream(s):`.

6. **`Stream()`**:
   - 中文: 这是 PyTorch 用户和 Green Context 最常打交道的入口 —— 它返回一个 `torch.cuda.Stream`,但这个 stream 已经"出生在绿区里"了。所有提交到这个 stream 的 kernel 只会被调度到这 48 个 SM 上。结合 `with torch.cuda.stream(green_stream):`,你的小模型从此和大模型的 kernel 物理隔离。
   - English: this is the entry point most PyTorch users will actually touch. It returns a standard `torch.cuda.Stream` — but that stream was born inside the green context. Any kernel submitted to it lands only on those 48 SMs. Combined with `with torch.cuda.stream(green_stream):` your small model is now physically isolated from the big one's kernels.

## 类比 / The analogy

想象一个大 office,132 个工位(SM)。以前所有项目共享:大老板的 PPT 渲染一启动,把 100 个工位都占了,小项目的报销审批就排队。Green Context 像是 office 经理拿粉笔在地上画了几个区:"这 48 个工位是 A 区,只给小项目用;那 84 个是 B 区,大项目专用"。物理上的隔离 —— A 区的人不能去 B 区的工位坐,反之亦然。工作队列(workqueue)就是每个区门口的"取号机":B 区取号机和 A 区独立,大项目排队再多,小项目报销审批也不受影响。

Imagine an office with 132 desks (SMs). Today it's free-for-all: when the boss kicks off a PPT render that grabs 100 desks, the expense-reimbursement team queues up. Green Context is like the office manager drawing chalk lines: "these 48 desks are Zone A, small projects only; those 84 are Zone B, big projects." Physical isolation — Zone A people physically cannot sit in Zone B. The workqueue is the ticket dispenser at each zone's door: A's dispenser is independent of B's, so even when B has a long queue, the small project in A keeps making progress.

## 自己跑一遍 / Try it yourself

```python
# Requires PyTorch built with CUDA 12.4+ and the green-context feature enabled.
import torch

if not torch.cuda.green_contexts.SUPPORTED:
    raise SystemExit("Green Contexts not supported in this build.")

GC = torch.cuda.GreenContext

print("device max workqueue concurrency:", GC.max_workqueue_concurrency())

# Carve a small green context with 48 SMs and an isolated workqueue.
small_gc = GC.create(num_sms=48, workqueue_scope="balanced",
                      workqueue_concurrency_limit=4)

# Get a stream that lives inside this 48-SM region.
small_stream = small_gc.Stream()

# Run a kernel inside the green context.
x = torch.randn(8192, 8192, device="cuda")
with torch.cuda.stream(small_stream):
    y = x @ x.T          # matmul confined to 48 SMs
    torch.cuda.synchronize()
print("matmul done on small green context:", y.shape, y.device)
```

运行 / Run with:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
python try.py
```

预期输出 / Expected output:
```
device max workqueue concurrency: 32
matmul done on small green context: torch.Size([8192, 8192]) cuda:0
```

中文:有趣的实验是"在另一个 CUDA stream 上同时启动一个独占满 GPU 的大 kernel" —— 你会发现小绿区里的 matmul 几乎不被拖慢,而以前在普通 stream 上是会被慢得不忍直视的。

English: the eye-opening experiment is to launch a *second* heavy kernel on a regular CUDA stream at the same time. You'll see the matmul inside the green context barely slows down — whereas the equivalent setup on plain streams would suffer badly.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CUDA MPS (Multi-Process Service)** / **CUDA MPS**:解决"多进程共享 GPU"的老办法 —— 通过单独的 MPS daemon。Green Context 是单进程内的更细粒度版本。 / The old answer for multi-process GPU sharing — via a daemon. Green Contexts give finer, in-process control.
- **NVIDIA MIG (Multi-Instance GPU)** / **NVIDIA MIG**:A100/H100 上的硬件级 GPU 切分。粒度更粗(每个 instance 有自己的 HBM 切片),Green Context 是软件层的细粒度补充。 / Hardware GPU partitioning on A100/H100 with per-instance HBM slices. Coarser; Green Contexts are the software-level fine-grained complement.
- **vLLM 的 v0.7 引入了 Green Context 支持** / **vLLM v0.7 added Green Context support**:用来在一张 H100 上同时跑 chat 和 embedding 模型而不互相饿死。 / Uses Green Contexts to co-host a chat model and an embedding model on one H100 without starvation.
- **TensorRT-LLM 的 streamMP** / **TensorRT-LLM streamMP**:类似的"SM 配额"思路,但绑在 stream 级别。 / Same "SM quota" idea, bolted onto streams.

## 注意事项 / Caveats / when it breaks

- **不是所有 GPU 都支持 / Not on every GPU**:需要 Hopper(SM90)或以后的 driver 支持,而且 PyTorch 必须用 CUDA 12.4+ 编译。`SUPPORTED = False` 时 import 不会报错,但 `create()` 会抛 RuntimeError —— 别忘了在 production code 里 fallback。 / Requires Hopper (SM90)+ with a driver that supports it, plus PyTorch built against CUDA 12.4+. Import won't fail when unsupported but `create()` will — guard your prod code.
- **不要忘 pop / Don't forget to pop**:虽然 `Stream()` 接口对用户友好,如果你直接 `set_context()` 用了"上下文压栈"风格,忘 `pop_context()` 会让后续无关 kernel 也跑在这个绿区 —— 表现是"为什么我后面这个完全无关的训练 step 慢了一倍?"。 / The `Stream()` API is the friendly surface; if you go raw with `set_context()`, forgetting to `pop_context()` leaves *unrelated* later kernels inside the green region — the symptom is "why is this unrelated training step suddenly half-speed?"
- **SM 数和 occupancy 不是线性的 / Halving SMs ≠ halving throughput**:小 kernel 在 48 个 SM 上可能比在 132 个 SM 上 occupancy 更高(因为它本来就用不满 132 个),所以"切一半 SM 不等于慢一半"。但大 GEMM 一旦 SM 不够,带宽-bound 仍然成立,性能会接近线性下降。 / A small kernel on 48 SMs may have higher occupancy than on 132 (it never saturated 132 anyway), so the slowdown is sub-linear. A big GEMM, once SM-starved, drops nearly linearly with SM count.
- **API 仍标 beta / Still beta**:文档里那条 `.. warning:: This API is in beta and may change` 不是花架子 —— 参数名(`workqueue_scope` 的取值)在未来 release 可能变。 / The `.. warning:: This API is in beta` is real — accepted values for `workqueue_scope` may change in future releases.

## 延伸阅读 / Further reading

- [CUDA Green Contexts overview (NVIDIA docs)](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
- [PyTorch PR introducing green-context Python wrapper (#138133)](https://github.com/pytorch/pytorch/pull/138133)
- [NVIDIA MIG vs Green Context vs MPS — a practitioner comparison (NVIDIA dev blog)](https://developer.nvidia.com/blog/improving-gpu-utilization-in-kubernetes/)
- [vLLM RFC: green-context-based latency isolation](https://github.com/vllm-project/vllm/issues/15103)
