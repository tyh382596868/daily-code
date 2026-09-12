---
date: 2026-07-15
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py
permalink: https://github.com/vllm-project/vllm/blob/fdf2cf66d3aa34800aaf1f4382fb2028886f0bae/vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py#L185-L303
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, lora, moe, flashinfer]
---

# vLLM LoRA MoE：W13 融进 kernel，W2 在外面补差量 / vLLM LoRA MoE: Fuse W13 into the Kernel, Add W2 Outside

> **一句话 / In one line**: vLLM 把 MoE 的第一层 LoRA delta 交给 FlashInfer routed kernel，同时把第二层 delta 在 kernel 外恢复并相加。 / vLLM sends the first MoE LoRA delta into the FlashInfer routed kernel, then reconstructs and adds the second delta outside the kernel.

## 为什么重要 / Why this matters

MoE 推理里，token 会先按 expert 重排，再做 gate/up/down 两层专家 MLP。LoRA 又给每个 expert 增加低秩增量。如果所有东西都在 Python 里摊开，会丢掉 fused kernel 的吞吐；如果硬塞进一个 kernel，又会受限于 kernel 支持的中间结果。这里的折中很干净：W13 delta 在 SwiGLU 之前像 bias 一样进 kernel；W2 delta 需要 post-activation，所以拿 kernel 返回的中间激活，在外面用 LoRA 路径补上。

MoE inference permutes tokens by expert and runs a gate/up/down expert MLP. LoRA adds low-rank deltas per expert. Expanding all of that in Python loses the fused-kernel benefit, but forcing everything into one kernel depends on what intermediates the kernel can expose. This implementation splits the problem cleanly: W13 LoRA enters the routed kernel before SwiGLU, while W2 LoRA is computed from the returned post-activation and added afterward.

## 代码 / The code

`vllm-project/vllm` — [`vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py`](https://github.com/vllm-project/vllm/blob/fdf2cf66d3aa34800aaf1f4382fb2028886f0bae/vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py#L185-L303)

```python
# ---- 1) W13 LoRA delta -> gemm1_lora_delta (bf16, [T, top_k, 2I]) ----
gemm1_lora_delta = None
w13_meta = (None, None, None, None)

gemm1_lora_delta = torch.zeros(
    num_tokens,
    top_k,
    2 * intermediate_size,
    dtype=torch.bfloat16,
    device=hidden_states.device,
)

lora_x = hidden_states
if not self.expects_unquantized_inputs:
    orig = lora_context.original_hidden_states
    assert orig is not None and orig.shape[0] == hidden_states.shape[0], (
        "quantized trtllm LoRA path requires original_hidden_states"
    )
    lora_x = orig
# add_inputs=False: write the pure delta only (the base is fused in
# by the kernel) and do NOT multiply by the routing weight (it is a
# pre-SwiGLU bias).
w13_meta = self.apply_w13_lora(
    lora_context,
    y=gemm1_lora_delta,
    x=lora_x,
    topk_ids=topk_ids,
    topk_weights=topk_weights,
    expert_map=expert_map,
    w1=w1_cfg,
    w2=w2_cfg,
    num_tokens=num_tokens,
    top_k_num=top_k,
    add_inputs=False,
)

# apply_w13_lora writes the delta in vLLM's w13 order (gate=w1 first,
# up=w3 second), but FlashInfer's gemm1_lora_delta expects the halves
# in [up, gate] order. Swap them so the delta lands on the matching
# SwiGLU branch.
gemm1_lora_delta = torch.cat(
    [
        gemm1_lora_delta[..., intermediate_size:],
        gemm1_lora_delta[..., :intermediate_size],
    ],
    dim=-1,
)

# ---- 2) Call the routed flashinfer kernel ----
ret = self.invoke_routed_moe(
    hidden_states=hidden_states,
    w1=w1,
    w2=w2,
    packed_topk_ids=packed_topk_ids,
    gemm1_lora_delta=gemm1_lora_delta,
    global_num_experts=global_num_experts,
    a1q_scale=a1q_scale,
    output=output,
)
# ---- 3) W2 LoRA (computed out of kernel) ----
expanded_idx_to_permuted_idx = ret[1]
gemm1_act_permuted = ret[2]  # [max_padded, I], post-act
act = self._unpermute_activation(
    gemm1_act_permuted,
    expanded_idx_to_permuted_idx,
    num_tokens,
    top_k,
    intermediate_size,
)  # (T*top_k, I) -- same layout as the triton path's intermediate_cache2

(
    sorted_token_ids_lora,
    expert_ids_lora,
    num_tokens_post_padded_lora,
    token_lora_mapping,
) = w13_meta

w2_delta = torch.zeros(
    num_tokens,
    top_k,
    K,
    dtype=output.dtype,
    device=output.device,
)
self.apply_w2_lora(
    lora_context,
    y=w2_delta,
    x=act,
    topk_weights=topk_weights,
    sorted_token_ids_lora=sorted_token_ids_lora,
    expert_ids_lora=expert_ids_lora,
    num_tokens_post_padded_lora=num_tokens_post_padded_lora,
    token_lora_mapping=token_lora_mapping,
    num_tokens=num_tokens,
    w1=w1_cfg,
    w2=w2_cfg,
    top_k_num=top_k,
    add_inputs=False,
)
output.add_(w2_delta.sum(dim=1))
```

## 逐行讲解 / What's happening

1. **`gemm1_lora_delta`**:
   - 中文: 先建一个 `[tokens, top_k, 2 * intermediate]` 的 BF16 buffer，表示每个 token 路由到每个 expert 后，gate/up 投影要加的 LoRA 增量。
   - English: It allocates a BF16 buffer shaped `[tokens, top_k, 2 * intermediate]`, one LoRA delta per token, routed expert, and gate/up channel.
2. **`add_inputs=False`**:
   - 中文: 这里写入的是纯 delta，不把 base projection 加进去；base 权重仍由 FlashInfer kernel 自己处理。
   - English: This writes only the LoRA delta. The base projection remains inside the FlashInfer kernel.
3. **half swap**:
   - 中文: vLLM 内部 W13 的顺序和 FlashInfer 期望的顺序不同，所以用一次 `torch.cat` 交换 gate/up 两半。
   - English: vLLM and FlashInfer disagree on the gate/up half order, so one `torch.cat` swaps the two halves before entering the kernel.
4. **`ret[1]` / `ret[2]`**:
   - 中文: routed kernel 已经做完 base MoE 输出，同时返还 token 重排映射和 post-activation 激活，让 W2 LoRA 能在外面复原。
   - English: The routed kernel returns the finalized base output plus the permutation map and post-activation tensor needed to compute W2 LoRA outside.
5. **`output.add_(w2_delta.sum(dim=1))`**:
   - 中文: W2 delta 已经带 top-k 路由权重，所以只需要沿 expert 维求和，加回最终输出。
   - English: The W2 delta is already routing-weighted, so summing over top-k and adding to the output completes the correction.

## 类比 / The analogy

这像一家餐厅有中央厨房和外卖加料台。能在中央厨房一起炒的调料，直接进大锅；必须等出锅后才能撒的芝士，就在加料台补上。关键是外卖单号不能乱，`expanded_idx_to_permuted_idx` 就是保证每份菜回到原订单的取餐号。

Think of a restaurant with a central kitchen and a finishing counter. Ingredients that can be cooked with the dish go into the main wok; toppings that must be added after cooking are handled at the counter. The permutation map is the ticket number that makes sure every topping returns to the right order.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

tokens, top_k, hidden, inter = 2, 2, 3, 4
base_out = np.zeros((tokens, hidden))
w13_delta = np.arange(tokens * top_k * 2 * inter).reshape(tokens, top_k, 2 * inter)

# vLLM order: [gate, up], kernel order: [up, gate]
kernel_delta = np.concatenate([w13_delta[..., inter:], w13_delta[..., :inter]], axis=-1)

# pretend the kernel returns post-activation in permuted order
idx_map = np.array([1, 0, 3, 2])
act_permuted = np.arange(tokens * top_k * inter).reshape(tokens * top_k, inter)
act = act_permuted[np.clip(idx_map, 0, None)]

w2_delta = act.reshape(tokens, top_k, inter).sum(axis=-1, keepdims=True)
out = base_out + np.repeat(w2_delta.sum(axis=1), hidden, axis=1)
print(kernel_delta.shape, act.shape, out)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(2, 2, 8) (4, 4) [[28. 28. 28.]
 [92. 92. 92.]]
```

中文: 这个 toy 例子保留了两个核心动作：进入 kernel 前换顺序，离开 kernel 后按映射恢复中间激活。

English: The toy keeps the two important moves: reorder before the kernel, then recover the intermediate activation after the kernel.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FlashAttention 的 `return_softmax` 调试路径** / **FlashAttention debug-return paths**: kernel 主路径返回最终输出，额外返回中间统计给外层校验或补算。 / The kernel returns the primary output but can expose extra intermediates for outer-layer checks or corrections.
- **量化 GEMM 的 scale 外置补偿** / **Out-of-kernel scale correction for quantized GEMM**: kernel 做最贵的矩阵乘，Python/CUDA 外围处理无法或不值得融合的小修正。 / The kernel handles the expensive matmul while surrounding code applies corrections that are hard or not worth fusing.

## 注意事项 / Caveats / when it breaks

- **顺序必须一致 / Ordering must match**: gate/up 顺序换错，SwiGLU 的两半会串位，数值看起来不是 NaN 但语义全错。 / If the gate/up halves are swapped incorrectly, values may remain finite while the activation semantics are wrong.
- **依赖 kernel 返回中间量 / Depends on returned intermediates**: 如果 routed kernel 不返回 post-activation 和 permutation map，W2 LoRA 无法在外面补。 / Without post-activation and permutation outputs, the external W2 correction cannot be reconstructed.
- **硬件和 dtype 受限 / Hardware and dtype constrained**: 这个路径明确面向 BF16、Blackwell/SM100+ 和特定 FlashInfer API。 / This path is gated for BF16, Blackwell/SM100+, and a specific FlashInfer API.

## 延伸阅读 / Further reading

- [vLLM `trtllm_lora_moe.py`](https://github.com/vllm-project/vllm/blob/fdf2cf66d3aa34800aaf1f4382fb2028886f0bae/vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py)
- [FlashInfer project](https://github.com/flashinfer-ai/flashinfer)
