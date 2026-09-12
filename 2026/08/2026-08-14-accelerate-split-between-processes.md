---
date: 2026-08-14
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/state.py
permalink: https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/state.py#L426-L512
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, huggingface, distributed-inference]
---

# Accelerate split_between_processes：把提示词公平发给每张卡 / Accelerate split_between_processes: Deal Prompts Fairly Across Processes

> **一句话 / In one line**: `split_between_processes` 用 `divmod` 算每个进程的切片区间，并可通过重复最后一个样本把各进程补到同样长度。 / `split_between_processes` computes per-process slices with `divmod` and can pad by repeating the last sample so all processes return equal lengths.

## 为什么重要 / Why this matters

分布式推理经常不是训练 batch，而是一堆 prompts、图片路径或 dataset rows。这个 context manager 把输入按 rank 切开，还保留 list、tuple、tensor、dict、Dataset 的容器形状，避免用户自己写一堆 rank 分支。

Distributed inference often starts from prompts, image paths, or dataset rows rather than a training batch. This context manager splits inputs by rank while preserving lists, tuples, tensors, dicts, and `Dataset` objects, so users do not need rank-specific branching code.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/state.py`](https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/state.py#L426-L512)

```python
@contextmanager
def split_between_processes(self, inputs: list | tuple | dict | torch.Tensor, apply_padding: bool = False):
    if self.num_processes == 1:
        yield inputs
        return
    length = len(inputs)
    if isinstance(inputs, dict):
        length = len(inputs[list(inputs.keys())[0]])
        if not all(len(v) == length for v in inputs.values()):
            raise ValueError("All values in the dictionary must have the same length")
    num_samples_per_process, num_extras = divmod(length, self.num_processes)
    start_index = self.process_index * num_samples_per_process + min(self.process_index, num_extras)
    end_index = start_index + num_samples_per_process + (1 if self.process_index < num_extras else 0)

    def _split_values(inputs, start_index, end_index):
        if isinstance(inputs, (list, tuple, torch.Tensor)):
            result = inputs[start_index:end_index]
            if apply_padding:
                if isinstance(result, torch.Tensor):
                    from accelerate.utils import pad_across_processes, send_to_device

                    tensorized_result = send_to_device(result, self.device)
                    result = pad_across_processes(
                        tensorized_result, pad_index=send_to_device(inputs[-1], self.device)
                    )
                else:
                    result += [inputs[-1]] * (num_samples_per_process + (1 if num_extras > 0 else 0) - len(result))
            return result
        elif isinstance(inputs, dict):
            for key in inputs.keys():
                inputs[key] = _split_values(inputs[key], start_index, end_index)
            return inputs
        else:
            if is_datasets_available():
                from datasets import Dataset

                if isinstance(inputs, Dataset):
                    clamped_start = min(start_index, len(inputs))
                    clamped_end = min(end_index, len(inputs))
                    result_idcs = list(range(clamped_start, clamped_end))
                    if apply_padding:
                        result_idcs += [len(inputs) - 1] * (
                            num_samples_per_process + (1 if num_extras > 0 else 0) - len(result_idcs)
                        )
                    return inputs.select(result_idcs)
            return inputs

    yield _split_values(inputs, start_index, end_index)
```

## 逐行讲解 / What's happening

1. **第 464-472 行 / Lines 464-472**: 中文: 单进程直接返回；dict 输入要求每个 key 的长度一致，否则无法同步切片。 / English: Single-process execution returns immediately; dictionary values must share a length so each key can be sliced consistently.
2. **第 473-475 行 / Lines 473-475**: 中文: `divmod` 把余数优先分给前几个 rank，保证负载最多只差 1。 / English: `divmod` assigns the remainder to early ranks, keeping workloads at most one sample apart.
3. **第 477-491 行 / Lines 477-491**: 中文: list、tuple、tensor 走普通切片；padding 时非 tensor 重复最后一项，tensor 交给跨进程 pad 工具。 / English: Lists, tuples, and tensors use slicing; padding repeats the last non-tensor item or delegates tensor padding to distributed helpers.
4. **第 492-511 行 / Lines 492-511**: 中文: dict 递归切每个 value；HF `Dataset` 则用 `select` 取 row index。 / English: Dictionaries recurse over values, while HF `Dataset` uses `select` over row indices.

## 类比 / The analogy

像发扑克牌：7 张牌发给 3 个人时，前两个人拿 2 张，最后一个拿 1 张；如果游戏要求每人 2 张，就把最后一张复制一张当占位牌。

It is like dealing cards: seven cards over three players gives two, two, and one. If the game requires equal hand sizes, the final card is copied as a placeholder.

## 自己跑一遍 / Try it yourself

```python
def split(inputs, rank, world, pad=False):
    n, extra = divmod(len(inputs), world)
    start = rank * n + min(rank, extra)
    end = start + n + (1 if rank < extra else 0)
    out = inputs[start:end]
    if pad and len(out) < n + (1 if extra else 0):
        out += [inputs[-1]] * (n + (1 if extra else 0) - len(out))
    return out

for rank in range(3):
    print(rank, split(["A", "B", "C", "D", "E"], rank, 3, pad=True))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 ['A', 'B']
1 ['C', 'D']
2 ['E', 'E']
```

最后一个 rank 的真实样本只有一个；padding 只是为了后续 gather 形状对齐。

The last rank has only one real sample; padding is only there to make later gathering shape-compatible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch DistributedSampler** / **PyTorch DistributedSampler**: 同样用 rank 和 world size 把样本切给不同进程。 / It also uses rank and world size to assign samples to processes.
- **HF Datasets sharding** / **HF Datasets sharding**: dataset row 切分通常保留 dataset 容器，而不是先转成 list。 / Dataset row splitting often preserves the dataset container rather than converting to a list.

## 注意事项 / Caveats / when it breaks

- **padding 后要丢弃假样本** / **Drop padded samples afterward**: 重复的最后一项不能算进最终指标。 / The repeated last item must not be counted in final metrics.
- **dict 长度必须一致** / **Dictionary lengths must match**: prompt 和 image list 长度不同会直接报错。 / Mismatched prompt and image-list lengths raise an error immediately.

## 延伸阅读 / Further reading

- [huggingface/accelerate source](https://github.com/huggingface/accelerate/blob/16cb6eb80dd9aa8b4df1a63ef57863e455d53b83/src/accelerate/state.py#L426-L512)
