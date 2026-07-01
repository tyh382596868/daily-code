---
date: 2026-07-01
topic: infrastructure
source: tracked
repo: karpathy/nanoGPT
file: model.py
permalink: https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L170-L193
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, infrastructure, inference-optimization]
---

# nanoGPT 推理优化：只给最后一个 token 做 lm_head / nanoGPT Inference Optimization: Run lm_head Only on the Last Token

> **一句话 / In one line**: 训练时需要所有位置的 logits；采样时只需要最后一个位置的 logits。 / Training needs logits for every position; sampling only needs logits for the final position.

## 为什么重要 / Why this matters

这段 `forward` 把同一个 GPT 主干分成训练路径和推理路径：有 `targets` 时投影整段序列并计算交叉熵，没有 `targets` 时只取 `x[:, [-1], :]` 送进 `lm_head`。对大词表模型来说，少做 `T-1` 个位置的 vocab 投影，能省下明显的显存和时间。

This `forward` splits one GPT backbone into a training path and an inference path. With `targets`, it projects every token and computes cross-entropy; without `targets`, it sends only `x[:, [-1], :]` through `lm_head`. For large vocabularies, skipping `T-1` vocabulary projections saves meaningful memory and time.

## 代码 / The code

`karpathy/nanoGPT` — [`model.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L170-L193)

```python
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```

## 逐行讲解 / What's happening

1. **第 170-183 行 / Lines 170-183**: 中文: token embedding、position embedding、Transformer blocks 和 final norm 始终完整运行，因为最后一个 token 仍需要看完整上下文。 / English: Token embeddings, position embeddings, transformer blocks, and final norm still run for the whole prefix because the last token needs the full context.
2. **第 184-188 行 / Lines 184-188**: 中文: 训练路径保留所有时间步 logits，才能把 `(B,T,V)` 展平成 `(B*T,V)` 和每个 target 对齐。 / English: The training path keeps logits at every timestep so `(B,T,V)` can flatten to `(B*T,V)` and align with every target.
3. **第 189-191 行 / Lines 189-191**: 中文: 推理路径用列表索引 `[-1]` 保留时间维，输出仍是 `(B,1,V)`，下游采样代码不需要特殊处理 rank。 / English: The inference path uses list indexing `[-1]` to preserve the time dimension, so the output remains `(B,1,V)` and downstream sampling code does not need rank special-cases.

## 类比 / The analogy

像餐厅结账：训练时你要核对每一道菜的价格；真正买单时，只需要最后的总额。`lm_head` 就是那台昂贵的收银机。

It is like a restaurant bill: during auditing you check every item, but at checkout you only need the final total. `lm_head` is the expensive cash register.


## 自己跑一遍 / Try it yourself

```python
import torch
B, T, C, V = 2, 5, 4, 10
x = torch.randn(B, T, C)
lm_head = torch.nn.Linear(C, V)
train_logits = lm_head(x)
infer_logits = lm_head(x[:, [-1], :])
print(train_logits.shape)
print(infer_logits.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
torch.Size([2, 5, 10])
torch.Size([2, 1, 10])
```

中文: 这个小例子保留了源码里的关键控制流，但把依赖压到最低，便于你直接观察形状、索引或状态变化。

English: The miniature keeps the original control-flow idea while stripping dependencies down so the shape, index, or state change is visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers generation** / **Transformers generation**: 中文: 自回归解码每步通常只消费最后一个 hidden state。 / English: Autoregressive decoding usually consumes only the last hidden state at each step.
- **KV-cache 解码 / KV-cache decoding**: 中文: attention 复用历史 KV，输出也只推进新 token。 / English: Attention reuses historical KV and advances only the new token.

## 注意事项 / Caveats / when it breaks

- **不能跳过主干 / Do not skip the backbone**: 中文: 只能少做 `lm_head`，不能只跑最后一个 token 的 Transformer block。 / English: You can skip extra `lm_head` projections, not the transformer context computation.
- **训练必须全量 / Training needs all positions**: 中文: next-token loss 要监督每个位置，不能用这个推理分支。 / English: Next-token loss supervises every position, so the inference branch is wrong for training.

## 延伸阅读 / Further reading

- Source permalink above.
- Project repository linked from the frontmatter.
