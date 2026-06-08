---
date: 2026-06-07
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/hooks/group_offloading.py
permalink: https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/hooks/group_offloading.py#L279-L365
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, huggingface, diffusers, cuda-streams, memory-management, offloading]
---

# diffusers 怎么用一根 CUDA stream + pinned CPU 镜像把 30 GB 模型塞进 24 GB GPU / How diffusers fits a 30 GB diffusion model on a 24 GB GPU with one CUDA stream and pinned CPU mirrors

> **一句话 / In one line**: 每个权重组都在 CPU 上留一份 pinned 镜像,`_onload_from_memory` 用副 stream 把下一组权重 H2D 拷过去,主 stream 同时做当前组的计算,两个 stream 之间用 `record_stream` 防止 caching allocator 提前释放——overlap 后基本没有 offload 开销. / Each weight group keeps a pinned-CPU mirror; `_onload_from_memory` pushes the next group to GPU on a side stream while the default stream computes the current group, and `record_stream` keeps the caching allocator from freeing transfers in flight. With this overlap, offloading is nearly free.

## 为什么重要 / Why this matters

视频扩散模型(Wan2.1、CogVideoX、HunyuanVideo)的权重普遍 20-50 GB,但消费级 GPU 只有 24 GB. 朴素的"用一层卸一层"会让 H2D 拷贝完全串行在 forward 上,每个 block 多花几百毫秒——一次生成本来 30s,结果跑了 5 分钟. diffusers 的 group offloading 把模型按 block 切组,每组**算完后立刻往 CPU 写回去**,同时**算之前从副 stream 把下一组从 pinned-CPU 拽到 GPU**——只要 H2D 拷的时间 ≤ 上一组 forward 的时间,GPU 在通信 / 计算 overlap 下几乎感受不到 offload. 这 87 行是 PyTorch stream 同步的微缩百科——`sync`、`record_stream`、pinned memory、`tensor.data = ...` 几个核心概念全部用到了,理解它,你就能给任何模型加 offload.

Video diffusion models (Wan2.1, CogVideoX, HunyuanVideo) routinely weigh 20-50 GB but consumer GPUs cap at 24 GB. Naive layer-by-layer offload serializes every H2D copy with the forward, adding hundreds of milliseconds per block — a 30s generation balloons to 5 minutes. diffusers' group offloading slices the model into block groups; **after each group computes, its weights stream back to CPU**, and **before each group computes, the previous group's `_onload_` has already prefetched the next group's weights from pinned CPU on a side stream**. As long as the H2D copy fits within one block's forward time, the GPU never waits. These 87 lines are a stream-synchronization mini-encyclopedia: `synchronize`, `record_stream`, pinned memory, and direct `tensor.data =` rebinding all earn their keep here. Understand it and you can bolt offload onto any model.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/hooks/group_offloading.py#L279-L365`](https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/hooks/group_offloading.py#L279-L365)

```python
    def _onload_from_memory(self):
        if self.stream is not None:
            # Wait for previous Host->Device transfer to complete
            self.stream.synchronize()

        context = nullcontext() if self.stream is None else self._torch_accelerator_module.stream(self.stream)
        default_stream = self._torch_accelerator_module.current_stream() if self.stream is not None else None

        with context:
            if self.stream is not None:
                with self._pinned_memory_tensors() as pinned_memory:
                    self._process_tensors_from_modules(pinned_memory, default_stream=default_stream)
            else:
                self._process_tensors_from_modules(None)

    def _offload_to_disk(self):
        self._check_disk_offload_torchao()
        if not self._is_offloaded_to_disk and not os.path.exists(self.safetensors_file_path):
            os.makedirs(os.path.dirname(self.safetensors_file_path), exist_ok=True)
            tensors_to_save = {key: tensor.data.to(self.offload_device) for tensor, key in self.tensor_to_key.items()}
            safetensors.torch.save_file(tensors_to_save, self.safetensors_file_path)
        self._is_offloaded_to_disk = True
        # We do this to free up the RAM which is still holding the up tensor data.
        for tensor_obj in self.tensor_to_key.keys():
            tensor_obj.data = torch.empty_like(tensor_obj.data, device=self.offload_device)

    def _offload_to_memory(self):
        if self.stream is not None:
            if not self.record_stream:
                self._torch_accelerator_module.current_stream().synchronize()

            for group_module in self.modules:
                for param in group_module.parameters():
                    if _is_torchao_tensor(param):
                        _restore_torchao_tensor(param, self.cpu_param_dict[param])
                    else:
                        param.data = self.cpu_param_dict[param]
            for param in self.parameters:
                if _is_torchao_tensor(param):
                    _restore_torchao_tensor(param, self.cpu_param_dict[param])
                else:
                    param.data = self.cpu_param_dict[param]
            for buffer in self.buffers:
                if _is_torchao_tensor(buffer):
                    _restore_torchao_tensor(buffer, self.cpu_param_dict[buffer])
                else:
                    buffer.data = self.cpu_param_dict[buffer]
        else:
            for group_module in self.modules:
                group_module.to(self.offload_device, non_blocking=False)
            for param in self.parameters:
                if _is_torchao_tensor(param):
                    moved = param.to(self.offload_device, non_blocking=False)
                    _swap_torchao_tensor(param, moved)
                else:
                    param.data = param.data.to(self.offload_device, non_blocking=False)
            for buffer in self.buffers:
                if _is_torchao_tensor(buffer):
                    moved = buffer.to(self.offload_device, non_blocking=False)
                    _swap_torchao_tensor(buffer, moved)
                else:
                    buffer.data = buffer.data.to(self.offload_device, non_blocking=False)

    @torch.compiler.disable()
    def onload_(self):
        r"""Onloads the group of parameters to the onload_device."""
        if self.offload_to_disk_path is not None:
            self._onload_from_disk()
        else:
            self._onload_from_memory()

    @torch.compiler.disable()
    def offload_(self):
        r"""Offloads the group of parameters to the offload_device."""
        if self.offload_to_disk_path:
            self._offload_to_disk()
        else:
            self._offload_to_memory()
```

## 逐行讲解 / What's happening

1. **`_onload_from_memory` 的第一句 `self.stream.synchronize()` / The first `self.stream.synchronize()` in `_onload_from_memory`**:
   - 中文: 这是"上一次预取的等待点". 副 stream 上可能还有上一轮的 H2D 没拷完——直接动 `param.data` 会导致主 stream 读到一半内容. `synchronize()` 阻塞当前线程,直到副 stream 已发出去的工作全做完. 注意这只在用 stream 模式(`self.stream is not None`)时才需要.
   - English: this is the "wait for the prior prefetch" point. The side stream may still have an in-flight H2D from the previous group — touching `param.data` before it's done would let the main stream read a half-copied tensor. `synchronize()` blocks the host until the side stream drains. Only relevant when using a stream (`self.stream is not None`).

2. **`context = ... torch_accelerator_module.stream(self.stream)` / `context = ... torch_accelerator_module.stream(self.stream)`**:
   - 中文: 这是一个 context manager,进入后 *后续在当前线程发出的 CUDA 工作都会排到副 stream*. 一旦 `with context:` 块内部调 `tensor.to(device, non_blocking=True)`,这个 H2D copy 就发给副 stream,不会阻塞主 stream 上的 forward kernel.
   - English: a context manager that re-routes the CUDA work issued from this thread onto the side stream. Inside `with context:`, any `tensor.to(device, non_blocking=True)` becomes a side-stream H2D copy, leaving the main stream free for forward kernels.

3. **`default_stream = current_stream()` / `default_stream = current_stream()` (captured outside)**:
   - 中文: **在进入副 stream context 之前**就把 main stream 的句柄抓出来——因为 `record_stream` 需要拿"被使用方的 stream",而不是"被生产方的 stream". 这是 PyTorch stream API 里最容易写错的地方之一.
   - English: captured **before** entering the side-stream context, because `record_stream` needs the *consumer's* stream, not the producer's. This is one of the easiest things to get wrong in PyTorch's stream API.

4. **`with self._pinned_memory_tensors() as pinned_memory` / `with self._pinned_memory_tensors() as pinned_memory`**:
   - 中文: 没贴出来的辅助 context,但作用是确保所有 CPU 镜像都是 pinned memory(non-pinned 的 `tensor.to(cuda, non_blocking=True)` 会被 PyTorch 静默退回到同步拷贝). 这个保证是"非阻塞 H2D"成立的前提.
   - English: helper context (not shown) that ensures every CPU mirror is pinned. Without pinning, `tensor.to(cuda, non_blocking=True)` silently falls back to a synchronous copy and the whole overlap collapses. This is the non-negotiable precondition for asynchronous H2D.

5. **`_offload_to_memory` 里 `param.data = self.cpu_param_dict[param]` / `param.data = self.cpu_param_dict[param]` in `_offload_to_memory`**:
   - 中文: 这是 PyTorch 里最酷的"零拷贝" trick——`param` 还是同一个 `nn.Parameter` 对象,引用没变,但它的 `.data` 张量句柄直接换成 CPU pinned 镜像. 上层任何 `id(param)`-keyed 字典(包括 optimizer state)依然有效. 注意这不是拷贝——CPU 镜像在 `_init_cpu_param_dict` 里就建好了,这里只是切换指针.
   - English: this is PyTorch's coolest "zero-copy" trick — `param` is still the same `nn.Parameter`, no rebinding happens at the Python level, but the underlying `.data` tensor object is swapped to the pre-built pinned-CPU mirror. Any `id(param)`-keyed dict (including optimizer state) keeps working. The CPU mirror was built once in `_init_cpu_param_dict`; this is a pointer flip, not a memcpy.

6. **`if not self.record_stream: synchronize()` 的反向逻辑 / Inverted logic of `if not self.record_stream: synchronize()`**:
   - 中文: 看起来反直觉——*没有*开 record_stream 才同步? 因为 record_stream 模式下 caching allocator 知道副 stream 还在用这块显存,会自动延后回收;不开 record_stream 时,必须手动 sync 当前 stream 才能保证 H2D 完成、CPU 镜像被读到正确的数据. 这是 PyTorch 文档里一个著名的 trade-off.
   - English: counterintuitive — sync only when `record_stream` is **off**? Because with `record_stream` on, the caching allocator knows the side stream still needs the buffer and defers its release; with it off, the host has to `synchronize()` the current stream manually to ensure the H2D is done before reusing the memory. This is one of PyTorch's most-documented stream gotchas.

7. **`@torch.compiler.disable()` 在 `onload_` / `offload_` 上 / `@torch.compiler.disable()` on `onload_`/`offload_`**:
   - 中文: 这俩方法会被 Dynamo 误以为是"可以 trace 的纯函数"——但里面有 `param.data = ...` 副作用、有 `synchronize()` 等阻塞 host call. 强制 Dynamo 给个 graph break,绕过去用 eager 调用.
   - English: Dynamo would otherwise try to trace these as pure functions — but they mutate `param.data` and call host-blocking `synchronize()`. The decorator forces a graph break so Dynamo runs them eagerly.

## 类比 / The analogy

把 GPU 计算想成厨房灶台,CPU 内存是冰箱,disk 是地下室仓库. 普通的 offload 就是"一道菜炒完才去冰箱拿下一道菜的材料"——厨师永远要等. group offloading 是雇了**一个跑腿小弟**(副 stream):你现在炒着第 1 道菜(第 1 组在主 stream 算),小弟同时跑去冰箱拿第 2 道菜的材料,放到灶台旁边的小托盘上(pinned memory). 等第 1 道菜起锅,第 2 道的材料已经在台子上等着. `record_stream` 就像在托盘上贴个"还在用,别收走"的标签——冰箱阿姨(caching allocator)看到标签就不会提前把空盘子端走. `synchronize()` 是厨师喊"小弟跟上了吗?跟上了我开炒下一道".

Picture GPU compute as the stove, CPU memory as the fridge, and disk as the basement pantry. Plain offload is "finish dish 1, walk to the fridge for dish 2's ingredients" — the chef is always waiting. Group offloading hires **a runner** (the side stream): while you cook dish 1 on the main burner (main stream computes group 1), the runner grabs dish 2's ingredients from the fridge and places them on a side tray next to the stove (pinned memory). By the time dish 1 plates, dish 2's mise-en-place is ready. `record_stream` is the "still in use, don't clear" sticky note on the tray — the fridge attendant (the caching allocator) sees the sticker and won't reclaim the tray. `synchronize()` is the chef yelling "runner, are you here yet? OK, starting dish 2."

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, time

assert torch.cuda.is_available()
device = torch.device("cuda")
side = torch.cuda.Stream(device=device)
N = 1 << 26  # 64M floats = 256 MB
cpu_buf = torch.randn(N, pin_memory=True)
gpu_buf = torch.empty(N, device=device)

def compute(x):
    return x.sin().cos().sum()

# Prefetch on the side stream, compute on default — overlap them.
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.cuda.stream(side):
    gpu_buf.copy_(cpu_buf, non_blocking=True)
out = compute(torch.randn(N, device=device))  # main-stream work, runs in parallel
side.synchronize()                              # wait for the H2D to land
out2 = compute(gpu_buf)
torch.cuda.synchronize()
print("overlap total:", time.perf_counter() - t0, "s")

# Compare to fully serial.
torch.cuda.synchronize()
t0 = time.perf_counter()
gpu_buf.copy_(cpu_buf, non_blocking=False)
torch.cuda.synchronize()
_ = compute(torch.randn(N, device=device))
_ = compute(gpu_buf)
torch.cuda.synchronize()
print("serial total: ", time.perf_counter() - t0, "s")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output (timings vary by GPU):
```
overlap total: 0.018 s
serial total:  0.034 s
```

中文: overlap 版本的总时间 ≈ max(H2D, compute),而 serial 版本是 sum(H2D, compute). 这就是 group offloading 在真模型里能把 offload 开销降到几乎为 0 的原因——前提是计算时间 ≥ 拷贝时间.

The overlap run is roughly `max(H2D, compute)`; the serial run is roughly `sum(H2D, compute)`. That's the entire reason group offloading drops the offload tax to nearly zero on real models — provided compute per block ≥ H2D per block.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DeepSpeed ZeRO-Infinity**: 中文: 同样的 H2D / D2H 用副 stream + pinned memory 思路,但额外分片到多 GPU + NVMe. / English: same H2D/D2H side-stream + pinned-memory technique, extended across many GPUs and NVMe.
- **PyTorch `cpu_offload` 在 FSDP 里**: 中文: 早期 FSDP 直接用同步拷贝,后来引入 `cpu_offload_with_pinned_memory` 配合 communication stream——和这里一脉相承. / English: early FSDP used synchronous offload; later versions adopted a pinned-memory + communication-stream scheme that's the direct ancestor of this pattern.
- **vLLM 的 KV cache offload**: 中文: 同样的 record_stream + pinned memory + 副 stream prefetch,只不过对象是 KV cache 而不是权重. / English: same record_stream + pinned-memory + side-stream prefetch, but the payload is KV cache rather than weights.
- **CUDA Best Practices Guide 第 9 章**: 中文: NVIDIA 官方文档把这个模式叫 "overlap data transfers with computation",建议每个搬运任务一个独立 stream. / English: NVIDIA officially calls it "overlap data transfers with computation" and recommends one dedicated stream per transfer.

## 注意事项 / Caveats / when it breaks

- **CPU 镜像必须 pinned**: 中文: 如果 `cpu_param_dict[param]` 不是 pinned memory,`.to(cuda, non_blocking=True)` 会静默退回同步拷贝——副 stream 形同虚设,但你看不出任何报错. 检查办法:`tensor.is_pinned()`.
- **CPU mirrors must be pinned**: English: if `cpu_param_dict[param]` isn't pinned, `.to(cuda, non_blocking=True)` silently degrades to a sync copy and your side stream becomes a no-op — no error, just slow. Sanity check with `tensor.is_pinned()`.
- **`record_stream` 不是免费的**: 中文: 它让 caching allocator 多记一份元数据,且内存复用比 sync 模式延迟. 实测在小模型上有时反而更慢——这里把它做成可选 `record_stream: bool` 是有道理的.
- **`record_stream` isn't free**: English: it makes the caching allocator track extra metadata and delays memory reuse vs. an explicit sync. On small models it can actually be slower — which is why this code exposes `record_stream: bool` as a tunable.
- **TorchAO 量化张量不能走 disk offload**: 中文: `_check_disk_offload_torchao` 直接抛错——safetensors 不能序列化 TorchAO 的 wrapper subclass tensor. 量化场景请用 memory offload.
- **TorchAO quantized tensors can't go to disk**: English: `_check_disk_offload_torchao` raises — safetensors can't serialize TorchAO's `_make_wrapper_subclass` tensors. Stick to memory offload for quantized models.
- **`param.data = ...` 不能跨 dtype/shape 切换**: 中文: 切换前后 dtype 和 shape 必须一致,否则上层依赖 `id(param)` 的 state 会失效. group offloading 内部刻意只在固定大小的 mirror 之间切.
- **`param.data = ...` requires matching dtype/shape**: English: dtype and shape must stay invariant across the swap, otherwise upstream `id(param)`-keyed state (e.g. optimizer slot) breaks. Group offloading deliberately swaps only between mirrors of identical layout.

## 延伸阅读 / Further reading

- [diffusers blog: group offloading explained](https://huggingface.co/docs/diffusers/main/en/optimization/memory#group-offloading) — the user-facing knobs
- [`hooks/group_offloading.py` full file](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/group_offloading.py) — 1000+ lines, including the prefetch graph, hook registry, leaf-level vs block-level grouping
- [PyTorch docs: CUDA stream semantics](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams) — the `record_stream` trade-off in detail
- [NVIDIA CUDA C Best Practices, §9](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation) — the canonical reference
