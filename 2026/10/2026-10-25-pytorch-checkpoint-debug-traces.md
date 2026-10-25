---
date: 2026-10-25
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/checkpoint.py
permalink: https://github.com/pytorch/pytorch/blob/c9b5004c45376c912422325d032b2d74b4828d5c/torch/utils/checkpoint.py#L1184-L1242
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, activation-checkpointing, debugging]
---

# PyTorch checkpoint 调试：把两次执行的证据并排摆出 / PyTorch Checkpoint Debugging: Put Both Executions Side by Side

> **一句话 / In one line**: checkpoint 重算出错时，PyTorch 同时记录原始 forward 与重算 forward 的算子和调用栈。 / When checkpoint recomputation diverges, PyTorch records both the original forward and the replayed forward, including operators and traces.

## 为什么重要 / Why this matters

activation checkpoint 以计算换显存，反向时会重新跑一段 forward。难点在于重算路径一旦悄悄不同，错误位置往往离根因很远；这段代码把两份“黑匣子记录”做成可比对的诊断报告。

Activation checkpointing trades memory for compute by replaying part of forward during backward. If the replay quietly follows a different path, the visible error can be far from its cause; this code turns both runs into comparable flight recorders.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/checkpoint.py`](https://github.com/pytorch/pytorch/blob/c9b5004c45376c912422325d032b2d74b4828d5c/torch/utils/checkpoint.py#L1184-L1242)

```python
capture_logs_fwd = CaptureLogs()
capture_logs_recompute = CaptureLogs()

def unpack_error_cb(e: CheckpointError) -> NoReturn:
    def get_str_tb(label, capture_logs):
        out = ""
        for i, (log, tb) in enumerate(zip(capture_logs.logs, capture_logs.tbs, strict=False)):
            out += f"{log}   ({i + 1} of {len(capture_logs.logs)} in {label})\n\n"
            found_torch_dispatch = False
            for line in tb:
                is_torch_dispatch = line['name'] == '__torch_dispatch__'
                if not found_torch_dispatch and not is_torch_dispatch:
                    continue
                if is_torch_dispatch:
                    found_torch_dispatch = True
                    continue
                out += f"{line['filename']}:{line['line']}:{line['name']}\n"
        return out
    raise CheckpointError(_checkpoint_error_template.format(
        forward_traces=get_str_tb("original", capture_logs_fwd),
        recompute_traces=get_str_tb("recompute", capture_logs_recompute))) from e
```

## 逐行讲解 / What's happening

1. **第 1205-1206 行 / Lines 1205-1206**：forward 和 replay 各有独立记录器，比较从数据结构上就被保证。 / Forward and replay get separate recorders, making comparison structurally unavoidable.
2. **第 1212-1224 行 / Lines 1212-1224**：报告只保留 `__torch_dispatch__` 之后的用户相关栈，避免框架噪声淹没问题。 / The report retains the user-relevant stack after `__torch_dispatch__`, avoiding a wall of framework noise.
3. **第 1230-1237 行 / Lines 1230-1237**：同一个异常模板承载两份轨迹和算子序列，直接暴露第一次分叉。 / One exception template carries both traces and operator streams, exposing the first divergence.

## 类比 / The analogy

像把同一架飞机的原始飞行和模拟重飞的黑匣子放在一起，逐秒找第一次仪表不一致。 / It is like placing the black boxes from an original flight and a simulator replay side by side, then locating the first mismatched instrument reading.

## 自己跑一遍 / Try it yourself

```python
original = ["matmul", "relu", "sum"]
replay = ["matmul", "dropout", "sum"]
for i, (a, b) in enumerate(zip(original, replay), 1):
    if a != b:
        print(f"diverged at {i}: original={a}, replay={b}")
        break
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `diverged at 2: original=relu, replay=dropout`。

对 checkpoint 来说，首次分叉比最终的 shape error 更有诊断价值。 / For checkpointing, the first divergence is more useful than the final shape error.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **编译器差分测试 / Compiler differential testing**：比较解释执行与编译执行。 / Compare interpreted and compiled executions.
- **分布式重放 / Distributed replay**：记录输入和事件序列以复现罕见故障。 / Record inputs and event order to reproduce rare failures.

## 注意事项 / Caveats / when it breaks

- **调试有开销 / Debugging costs overhead**：捕获每个算子和栈不应长期打开。 / Capturing every operator and stack should not remain enabled in production.
- **随机性也要受控 / Randomness must be controlled**：未同步的 RNG 会制造无意义的差异。 / Unsynchronized RNG can create meaningless differences.

## 延伸阅读 / Further reading

- [PyTorch checkpoint source](https://github.com/pytorch/pytorch/blob/c9b5004c45376c912422325d032b2d74b4828d5c/torch/utils/checkpoint.py)
