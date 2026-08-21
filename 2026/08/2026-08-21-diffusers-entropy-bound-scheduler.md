---
date: 2026-08-21
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_entropy_bound.py
permalink: https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/schedulers/scheduling_entropy_bound.py#L87-L182
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, scheduler, entropy]
---

# Diffusers EntropyBound：低不确定 token 先落定 / Diffusers EntropyBound: Accept Low-Uncertainty Tokens First

> **一句话 / In one line**: `EntropyBoundScheduler.step` 给 logits 降温后采样，按 token entropy 从低到高累计，只接受总不确定性没越界的位置，其余位置重新噪声化。 / `EntropyBoundScheduler.step` cools logits, samples candidates, accepts the lowest-entropy positions under a cumulative bound, and renoises the rest.

## 为什么重要 / Why this matters

离散扩散或 masked token 生成里，不是每个位置都应该同时定稿。模型很确信的位置可以先固定；还不确定的位置继续保留随机性。这个 scheduler 把“是否接受一个 token”写成 entropy budget，而不是固定接受 top-k 个位置。

In discrete diffusion or masked token generation, not every position should be finalized at the same time. Confident positions can lock in first; uncertain positions should stay random. This scheduler expresses acceptance as an entropy budget instead of a fixed top-k count.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_entropy_bound.py`](https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/schedulers/scheduling_entropy_bound.py#L87-L182)

```python
def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
    if num_inference_steps <= 0:
        raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
    self.num_inference_steps = num_inference_steps
    self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.long)

...

fraction = (self.num_inference_steps - int(timestep)) / self.num_inference_steps
temperature = self.config.t_min + (self.config.t_max - self.config.t_min) * fraction
model_output = model_output / temperature
sampled_tokens, sampled_probs = self._sample_from_logits(model_output, temperature=1.0, generator=generator)

token_entropy = torch.distributions.Categorical(logits=model_output).entropy()
sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)

sorted_accepted = cumulative_entropy - sorted_token_entropy <= entropy_bound
accepted_index = torch.scatter(
    input=torch.zeros_like(sorted_accepted), dim=-1, index=sorted_indices, src=sorted_accepted
)

random_tokens = torch.randint(
    low=0, high=model_output.shape[-1], size=sample.shape, device=sample.device, generator=generator
)
prev_sample = torch.where(accepted_index, sampled_tokens, random_tokens)
```

## 逐行讲解 / What's happening

1. **第 87-91 行 / Lines 87-91 (timesteps)**:
   - 中文: scheduler 保存推理步数，并生成简单的 step index 序列。
   - English: The scheduler stores inference length and creates a simple sequence of step indices.
2. **第 151-156 行 / Lines 151-156 (temperature anneal)**:
   - 中文: 温度从 `t_max` 走向 `t_min`，并在采样和 entropy 测量前统一缩放 logits。
   - English: Temperature moves from `t_max` toward `t_min`, and logits are scaled once before both sampling and entropy measurement.
3. **第 158-164 行 / Lines 158-164 (entropy budget)**:
   - 中文: 每个位置的 entropy 从小到大排序，累计 entropy 没超过预算的位置才被接受。
   - English: Position entropies are sorted from low to high, and positions are accepted while cumulative entropy stays within budget.
4. **第 165-172 行 / Lines 165-172 (scatter back and renoise)**:
   - 中文: 接受 mask 被 scatter 回原位置；未接受位置用随机 token 替换，继续保留噪声。
   - English: The acceptance mask is scattered back to original positions; rejected positions are replaced with random tokens.

## 类比 / The analogy

像批改填空题。你先填最确定的空，直到不确定性预算用完；剩下的空不硬猜，留到下一轮再看上下文。

It is like filling blanks in a worksheet. You commit the answers you are most confident about until the uncertainty budget is spent; the rest wait for another pass.

## 自己跑一遍 / Try it yourself

```python
entropies = [0.05, 0.7, 0.2, 0.4]
budget = 0.5

order = sorted(range(len(entropies)), key=lambda i: entropies[i])
accepted = [False] * len(entropies)
running = 0.0
for i in order:
    if running <= budget:
        accepted[i] = True
    running += entropies[i]

print(accepted)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[True, False, True, True]
```

中文: 低 entropy 的位置先被接受；高 entropy 的位置被留给下一轮。

English: Low-entropy positions are accepted first; high-entropy positions are left for the next round.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MaskGIT-style decoding** / **MaskGIT-style decoding**: 逐轮固定高置信 token，低置信位置继续 mask。
- **Speculative decoding accept/reject** / **Speculative decoding accept/reject**: 候选 token 先被提出，再按置信或概率规则接受。

## 注意事项 / Caveats / when it breaks

- **entropy 不是正确性的保证** / **Entropy is not correctness**: 低 entropy 只表示模型很确定，不表示一定对。
- **预算过小会拖慢收敛** / **Too small a bound slows convergence**: 每轮接受太少位置会增加推理步数。

## 延伸阅读 / Further reading

- [Diffusers `EntropyBoundScheduler`](https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/schedulers/scheduling_entropy_bound.py#L87-L182)
- [Diffusers repository](https://github.com/huggingface/diffusers)
