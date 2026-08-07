---
date: 2026-08-07
topic: infrastructure
source: tracked
repo: karpathy/nanoGPT
file: train.py
permalink: https://github.com/karpathy/nanoGPT/blob/master/train.py#L275-L288
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, ddp, gradient-accumulation]
---

# nanoGPT DDP 累积梯度：只在最后一个 micro-step 同步 / nanoGPT DDP Accumulation: Sync Only on the Last Micro-Step

> **一句话 / In one line**: nanoGPT 用 `require_backward_grad_sync` 把多次 backward 变成一次跨卡 all-reduce。 / nanoGPT uses `require_backward_grad_sync` to turn many backward calls into one cross-rank all-reduce.

## 为什么重要 / Why this matters

大 batch 训练常靠 gradient accumulation：显存里一次只跑小 micro-batch，多个 micro-batch 的梯度加起来再更新参数。DDP 默认每次 backward 都同步梯度；如果不关掉中间同步，就会把通信成本放大好几倍。

Large-batch training often uses gradient accumulation: each micro-batch fits in memory, gradients accumulate, and the optimizer steps once. DDP normally synchronizes gradients on every backward; if intermediate syncs are not disabled, communication cost multiplies.

## 代码 / The code

`karpathy/nanoGPT` — [`train.py`](https://github.com/karpathy/nanoGPT/blob/master/train.py#L275-L288)

```python
for micro_step in range(gradient_accumulation_steps):
    if ddp:
        model.require_backward_grad_sync = (
            micro_step == gradient_accumulation_steps - 1
        )
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch("train")
    scaler.scale(loss).backward()
```

## 逐行讲解 / What's happening

1. **第 275 行 / Line 275 (micro-batches)**:
   - 中文: 外层一次 optimizer step，内层跑多个 micro-step。
   - English: One outer optimizer step contains several inner micro-steps.
2. **第 276-281 行 / Lines 276-281 (DDP sync gate)**:
   - 中文: 只有最后一个 micro-step 才让 DDP 同步梯度。
   - English: Only the final micro-step allows DDP to synchronize gradients.
3. **第 282-284 行 / Lines 282-284 (loss scaling)**:
   - 中文: 每个 micro-batch 的 loss 先除以累积步数，这样总梯度尺度不变。
   - English: Each micro-batch loss is divided by the accumulation count, keeping the final gradient scale unchanged.
4. **第 285-288 行 / Lines 285-288 (overlap)**:
   - 中文: 下一批数据立刻预取，当前 loss 继续做 backward。
   - English: The next batch is prefetched immediately while the current loss enters backward.

## 类比 / The analogy

像团队写日报：每个人先在本地草稿里补几条，最后统一发一次群消息。每写一条就群发，会让通信噪声淹没真正的工作。

It is like a team status update: everyone drafts several local notes, then sends one shared message at the end. Broadcasting after every sentence wastes the channel.

## 自己跑一遍 / Try it yourself

```python
accum_steps = 4
for micro_step in range(accum_steps):
    sync = micro_step == accum_steps - 1
    print(micro_step, "sync" if sync else "local")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 local
1 local
2 local
3 sync
```

前三步只累计本地梯度，最后一步才跨卡对齐。

The first three steps accumulate locally; the last one aligns gradients across ranks.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `no_sync()`** / **PyTorch `no_sync()`**: 官方上下文管理器也是为了跳过中间 all-reduce。 / The official context manager also skips intermediate all-reduce calls.
- **Accelerate accumulation plugin** / **Accelerate accumulation plugin**: HF Accelerate 把同一件事包装成训练框架级状态。 / HF Accelerate wraps the same idea as framework-level training state.

## 注意事项 / Caveats / when it breaks

- **loss 必须缩放** / **Scale the loss**: 不除以累积步数会把有效学习率放大。 / Without division, the effective learning rate grows.
- **最后一步必须同步** / **The final step must sync**: 忘记打开同步会让各 rank 参数漂移。 / Forgetting the final sync lets ranks drift apart.

## 延伸阅读 / Further reading

- [nanoGPT training loop](https://github.com/karpathy/nanoGPT/blob/master/train.py#L275-L288)

