---
date: 2026-09-06
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/_utils/pin_memory.py
permalink: https://github.com/pytorch/pytorch/blob/e6fbf0c760281dcbe5bd38ea7a153bd56a2012d8/torch/utils/data/_utils/pin_memory.py#L439-L586
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader, pin-memory, recursion]
---

# PyTorch pin_memory：递归搬 batch，但保住容器形状 / PyTorch pin_memory: Move a Batch Recursively, Keep the Container Shape

> **一句话 / In one line**: 这个 worker thread 一边循环取 batch，一边递归把 tensor pin 到 page-locked memory，容器结构原样保留。 / This worker thread loops over batches and recursively pins tensors into page-locked memory while preserving container structure.

## 为什么重要 / Why this matters

中文：DataLoader 的 CPU 侧优化不只是“开多线程”这么简单。真正的关键是把传给 GPU 的 batch 变成 pinned memory，同时别破坏 dict、list、namedtuple 这些嵌套结构。PyTorch 在这里把线程启动、递归处理和异常退出都收进一个 worker。

English: DataLoader-side optimization is more than “just use more threads.” The key is to move GPU-bound batches into pinned memory without breaking nested dicts, lists, or namedtuples. PyTorch bundles worker startup, recursive handling, and clean shutdown into one place.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/_utils/pin_memory.py`](https://github.com/pytorch/pytorch/blob/e6fbf0c760281dcbe5bd38ea7a153bd56a2012d8/torch/utils/data/_utils/pin_memory.py#L439-L586)

```python
def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    torch.set_num_threads(1)
    torch.multiprocessing._set_thread_name("pt_data_pin")

    if device == "cuda":
        torch.cuda.set_device(device_id)

    def do_one_step():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set():
            data = pin_memory(data, device)
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    while not done_event.is_set():
        do_one_step()


def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    if isinstance(data, (str, bytes)):
        return data
    if isinstance(data, collections.abc.Mapping):
        try:
            return type(data)({k: pin_memory(v, device) for k, v in data.items()})
        except TypeError:
            return {k: pin_memory(v, device) for k, v in data.items()}
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(pin_memory(x, device) for x in data))
    if isinstance(data, collections.abc.Sequence):
        try:
            return type(data)([pin_memory(x, device) for x in data])
        except TypeError:
            return [pin_memory(x, device) for x in data]
    return data
```

## 逐行讲解 / What's happening

1. **第 439-445 行 / Lines 439-445**:
   - 中文: worker 先把线程数压到 1，再按目标设备设线程名和 CUDA 设备。
   - English: The worker first limits itself to one CPU thread, then names the thread and selects the CUDA device.
2. **第 447-494 行 / Lines 447-494**:
   - 中文: 每次只处理一个队列项；如果没结束，就 pin 完再塞回输出队列。
   - English: Each step handles one queue item; if the worker is still alive, it pins the batch and puts it on the output queue.
3. **第 496-586 行 / Lines 496-586**:
   - 中文: `pin_memory()` 递归处理 tensor、mapping、tuple、sequence，核心是保结构不保引用。
   - English: `pin_memory()` recursively handles tensors, mappings, tuples, and sequences; the key is preserving structure, not identity.

## 类比 / The analogy

中文：像搬家时把家具逐件抬上车，但房间编号、抽屉层级和箱子标签都不改。

English: It is like moving furniture piece by piece while keeping room numbers, drawer hierarchies, and box labels intact.

## 自己跑一遍 / Try it yourself

```python
def pin(x):
    if isinstance(x, dict):
        return {k: pin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [pin(v) for v in x]
    return f"pinned({x})"

batch = {"obs": [1, 2], "meta": {"step": 3}}
print(pin(batch))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'obs': ['pinned(1)', 'pinned(2)'], 'meta': {'step': 'pinned(3)'}}
```

中文：pin memory 的价值不是更快地搬一个 tensor，而是让整批数据更顺畅地进入 GPU 拷贝链路。

English: The value of pin memory is not moving one tensor faster; it is making the whole batch flow more smoothly into the GPU copy path.
