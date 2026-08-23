---
date: 2026-08-23
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/utils/memory.py
permalink: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py#L93-L174
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, accelerate, oom-retry]
---

# Accelerate executable batch size：OOM 后自动退一步 / Accelerate Executable Batch Size: Step Down After OOM

> **一句话 / In one line**: Accelerate 用一个装饰器捕获 OOM 类错误，清理设备 cache，然后用更小的 batch size 重试训练函数。 / Accelerate wraps a training function, catches OOM-like failures, clears device caches, and retries with a smaller batch size.

## 为什么重要 / Why this matters

训练脚本最烦人的失败之一是 batch size 稍微大了一点。手动二分会浪费时间，而且不同设备的 OOM 报错字符串并不一样。这个工具把“试、失败、清理、缩小、再试”做成一个可复用入口。

One of the most annoying training failures is a batch size that is just slightly too large. Manual search wastes time, and OOM errors look different across devices. This utility packages the loop: try, fail, clear memory, reduce, and retry.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/utils/memory.py`](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py#L93-L174)

```python
def should_reduce_batch_size(exception: Exception) -> bool:
    """Checks if `exception` relates to CUDA out-of-memory, XPU out-of-memory, CUDNN not supported, or CPU out-of-memory"""
    _statements = [
        " out of memory.",  # OOM for CUDA, HIP, XPU
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
        "FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed",  # HPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

def find_executable_batch_size(
    function: Optional[callable] = None,
    starting_batch_size: int = 128,
    reduce_batch_size_fn: Optional[callable] = None,
):
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size
    if reduce_batch_size_fn is None:
        def reduce_batch_size_fn():
            nonlocal batch_size
            batch_size = int(batch_size * 0.9)
            return batch_size
    def decorator(*args, **kwargs):
        nonlocal batch_size
        clear_device_cache(garbage_collection=True)
        params = list(inspect.signature(function).parameters.keys())
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    clear_device_cache(garbage_collection=True)
                    batch_size = reduce_batch_size_fn()
                else:
                    raise
    return decorator
```

## 逐行讲解 / What's happening

1. **第 93-108 行 / Lines 93-108**:
   - 中文: `should_reduce_batch_size` 不吞所有异常，只认几类内存相关的 `RuntimeError` 字符串。
   - English: `should_reduce_batch_size` does not swallow every exception; it matches only known memory-related `RuntimeError` strings.
2. **第 143-145 行 / Lines 143-145**:
   - 中文: 没传函数时返回 partial，所以它既能当普通函数用，也能当装饰器用。
   - English: Without a function, it returns a partial, allowing decorator-style use.
3. **第 146-151 行 / Lines 146-151**:
   - 中文: 默认缩小策略是每次乘 0.9，且用 `nonlocal` 记住当前 batch size。
   - English: The default reducer multiplies by 0.9 and keeps the current batch size in a `nonlocal` variable.
4. **第 163-173 行 / Lines 163-173**:
   - 中文: 真正的循环只在 OOM 类异常时缩小；业务错误会原样抛出。
   - English: The retry loop reduces only for OOM-like errors. Regular bugs are re-raised unchanged.

## 类比 / The analogy

这像往电梯里搬箱子。第一次塞 128 个箱子，超重报警；你不是把电梯拆了，而是先清空，再少塞 10%，直到电梯能动。

It is like loading boxes into an elevator. You try 128 boxes, the overload alarm rings, so you empty it and try 10 percent fewer boxes until the elevator can move.

## 自己跑一遍 / Try it yourself

```python
def should_reduce(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc)

def run_with_batch(fn, batch_size):
    while batch_size:
        try:
            return fn(batch_size)
        except RuntimeError as exc:
            if not should_reduce(exc):
                raise
            batch_size = int(batch_size * 0.9)
    raise RuntimeError("No executable batch size found")

def train(batch):
    if batch > 72:
        raise RuntimeError("CUDA out of memory.")
    return f"ok at batch={batch}"

print(run_with_batch(train, 100))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
ok at batch=72
```

最关键的是只对内存错误重试；普通 bug 不应该被 batch-size 搜索掩盖。

The key point is selective retry. Ordinary bugs should not be hidden behind batch-size search.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **auto batch finder** / **auto batch finder**: 许多训练框架会先探测最大可行 batch。 / Many training frameworks probe the largest feasible batch.
- **HTTP retry policy** / **HTTP retry policy**: 只重试临时错误，不重试 4xx 业务错误。 / Good HTTP clients retry transient failures but not business errors.
- **gradient accumulation tuning** / **gradient accumulation tuning**: batch 太小时用累积步数补全有效 batch。 / When the per-step batch is small, accumulation restores the effective batch.

## 注意事项 / Caveats / when it breaks

- **字符串匹配不完美** / **String matching is imperfect**: 新硬件或新后端可能换错误文案。 / New backends may use new error messages.
- **不是二分搜索** / **This is not binary search**: 默认 0.9 退让稳健但不一定最快。 / The default 0.9 reduction is robust but not fastest.
- **状态函数要可重试** / **The wrapped function must be retryable**: 失败后如果留下半初始化状态，下一次可能仍然坏。 / If a failed run leaves partial state behind, the next retry may still fail.

## 延伸阅读 / Further reading

- Accelerate memory utilities: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py
- Accelerate docs: https://huggingface.co/docs/accelerate/
