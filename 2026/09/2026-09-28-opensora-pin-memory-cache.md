---
date: 2026-09-28
topic: diffusion
source: tracked
repo: hpcaitech/Open-Sora
file: opensora/datasets/pin_memory_cache.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/datasets/pin_memory_cache.py#L7-L76
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, pinned-memory]
---

# Open-Sora pinned memory cache：CPU buffer 也要复用 / Open-Sora Pinned Memory Cache: Reuse CPU Buffers Too

> **一句话 / In one line**: pin-memory 不是每次都重新分配，而是把足够大的 pinned CPU tensor 当池子借出再归还。 / Pinned memory is not reallocated every time; large enough CPU buffers are borrowed from a small pool and released back.

## 为什么重要 / Why this matters

视频扩散训练常常要把大批 latent、帧和条件从 CPU 送进 GPU。`pin_memory=True` 能让拷贝更快，但频繁分配 pinned memory 本身也贵。这段代码把 pinned tensor 做成缓存池：命中时切出一个 view，没命中时才新建。

Video diffusion training moves large latents, frames, and conditions from CPU to GPU. Pinned memory speeds up those transfers, but allocating pinned buffers repeatedly is itself expensive. This cache turns pinned tensors into a small reusable pool: slice a view on hit, allocate only on miss.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/datasets/pin_memory_cache.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/datasets/pin_memory_cache.py#L7-L76)

```python
class PinMemoryCache:
    force_dtype: Optional[torch.dtype] = None
    min_cache_numel: int = 0
    pre_alloc_numels: List[int] = []

    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}
        self.output_to_cache: Dict[int, int] = {}
        self.cache_to_output: Dict[int, int] = {}
        self.lock = threading.Lock()
        self.total_cnt = 0
        self.hit_cnt = 0

        if len(self.pre_alloc_numels) > 0 and self.force_dtype is not None:
            for n in self.pre_alloc_numels:
                cache_tensor = torch.empty(n, dtype=self.force_dtype, device="cpu", pin_memory=True)
                with self.lock:
                    self.cache[id(cache_tensor)] = cache_tensor

    def get(self, tensor: torch.Tensor) -> torch.Tensor:
        """Receive a cpu tensor and return the corresponding pinned tensor. Note that this only manage memory allocation, doesn't copy content.

        Args:
            tensor (torch.Tensor): The tensor to be pinned.

        Returns:
            torch.Tensor: The pinned tensor.
        """
        self.total_cnt += 1
        with self.lock:
            # find free cache
            for cache_id, cache_tensor in self.cache.items():
                if cache_id not in self.cache_to_output and cache_tensor.numel() >= tensor.numel():
                    target_cache_tensor = cache_tensor[: tensor.numel()].view(tensor.shape)
                    out_id = id(target_cache_tensor)
                    self.output_to_cache[out_id] = cache_id
                    self.cache_to_output[cache_id] = out_id
                    self.hit_cnt += 1
                    return target_cache_tensor
        # no free cache, create a new one
        dtype = self.force_dtype if self.force_dtype is not None else tensor.dtype
        cache_numel = max(tensor.numel(), self.min_cache_numel)
        cache_tensor = torch.empty(cache_numel, dtype=dtype, device="cpu", pin_memory=True)
        target_cache_tensor = cache_tensor[: tensor.numel()].view(tensor.shape)
        out_id = id(target_cache_tensor)
        with self.lock:
            self.cache[id(cache_tensor)] = cache_tensor
            self.output_to_cache[out_id] = id(cache_tensor)
            self.cache_to_output[id(cache_tensor)] = out_id
        return target_cache_tensor

    def remove(self, output_tensor: torch.Tensor) -> None:
        """Release corresponding cache tensor.

        Args:
            output_tensor (torch.Tensor): The tensor to be released.
        """
        out_id = id(output_tensor)
        with self.lock:
            if out_id not in self.output_to_cache:
                raise ValueError("Tensor not found in cache.")
            cache_id = self.output_to_cache.pop(out_id)
            del self.cache_to_output[cache_id]

    def __str__(self):
        with self.lock:
            num_cached = len(self.cache)
            num_used = len(self.output_to_cache)
            total_cache_size = sum([v.numel() * v.element_size() for v in self.cache.values()])
        return f"PinMemoryCache(num_cached={num_cached}, num_used={num_used}, total_cache_size={total_cache_size / 1024**3:.2f} GB, hit rate={self.hit_cnt / self.total_cnt:.2f})"
```

## 逐行讲解 / What's happening

1. **第 13-16 行 / Lines 13-16**:
   - 中文: 三张表维护 cache、借出的 output view，以及二者的反向关系；lock 让 dataloader 多线程下也能更新。
   - English: Three maps track cached buffers, borrowed output views, and the reverse relation; the lock keeps updates safe under dataloader threads.
1. **第 38-45 行 / Lines 38-45**:
   - 中文: 只要某个空闲 cache 足够大，就切成目标 shape 的 view 返回，不复制内容。
   - English: If a free cached tensor is large enough, it returns a shaped view without copying data.
1. **第 47-55 行 / Lines 47-55**:
   - 中文: 没有可用块时才分配新的 pinned CPU tensor，并用最小缓存尺寸避免小张量造成碎片。
   - English: Only misses allocate a new pinned CPU tensor, and the minimum cache size avoids tiny fragmented blocks.
1. **第 64-69 行 / Lines 64-69**:
   - 中文: 调用方用完 output view 后释放映射，底层 cache tensor 仍留在池里等待下次复用。
   - English: After the caller releases the output view, the backing cache tensor stays in the pool for reuse.

## 类比 / The analogy

这像摄影棚里的灯架。拍完一个镜头不会把灯架拆掉扔了，而是松开夹子，下个镜头换个角度继续用。

It is like light stands in a studio. After one shot, you do not throw them away; you loosen the clamp and reuse the same stand for the next setup.

## 自己跑一遍 / Try it yourself

```python
class Cache:
    def __init__(self): self.free=[]
    def get(self, n):
        for i, buf in enumerate(self.free):
            if len(buf) >= n:
                return self.free.pop(i)[:n]
        return [0] * n
    def put(self, buf): self.free.append(buf)

c=Cache(); a=c.get(4); c.put(a); b=c.get(2)
print(len(a), len(b), len(c.free))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
4 2 0
```

注意第二次请求只要 2 个元素，但可以从之前的 4 元素 buffer 切 view。真实代码里这个 buffer 是 pinned tensor。

The second request needs only two elements but can be served from the previous four-element buffer. In the real code, that buffer is pinned memory.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch DataLoader pin_memory** / **PyTorch DataLoader also moves batches through pinned host memory before GPU transfer.**
- **vLLM workspace buffers** / **vLLM uses preallocated workspaces for repeated temporary tensors.**
- **CUDA graph warmup buffers** / **Inference systems often warm up and keep buffers stable before capture.**

## 注意事项 / Caveats / when it breaks

- **必须成对释放** / **Callers must call `remove`; otherwise the cache thinks the block is still checked out.**
- **只管理内存，不复制数据** / **`get` returns storage; copying the original tensor into it is still the caller's job.**
- **dtype 策略要一致** / **`force_dtype` can be useful, but mixing expected dtypes silently can confuse downstream code.**

## 延伸阅读 / Further reading

- Open-Sora source permalink above
- PyTorch pinned memory docs
