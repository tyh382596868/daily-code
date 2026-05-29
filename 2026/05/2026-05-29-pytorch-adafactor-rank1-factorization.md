---
date: 2026-05-29
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/_adafactor.py
permalink: https://github.com/pytorch/pytorch/blob/516f64b797cf7645a973e20d856d3e0ddec79948/torch/optim/_adafactor.py#L330-L416
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, pytorch, optimizer, low-memory, adafactor]
---

# Adafactor 的两条向量代替整张 V / Adafactor: a row vector and a column vector replace the full second-moment matrix

> **一句话 / In one line**: Adafactor 把 Adam 里 `O(m*n)` 的二阶动量 V 换成两条只占 `O(m + n)` 的统计量 `row_var`、`col_var`，再用它们的外积近似回完整 V。 / Adafactor replaces Adam's `O(m*n)` second-moment matrix V with a row-variance vector and a column-variance vector — `O(m + n)` state — and reconstructs V on the fly as their outer product.

## 为什么重要 / Why this matters

训练 7B-级别的 VLA 或 WAM 时，Adam 那张和参数等大的 V 经常是显存爆炸的真凶 —— bf16 参数 14 GB，对应 V 又是一份。Adafactor 的核心洞察是：对于矩阵参数，二阶动量的"有用部分"基本是低秩的，所以只存两条边际的均方就够了。这个 single-tensor reference 实现把所有数学全摊在 90 行里，没有 foreach、没有 fused kernel、没有 CUDA graph trick，可以当成"二阶动量低秩近似"的最干净教学。下次你看 Shampoo、Galore、Muon 这类后续低秩优化器，可以拿这段做对照。

When you fine-tune a 7B VLA or WAM, Adam's second-moment buffer V is one of the most common causes of OOM — bf16 weights are 14 GB and V duplicates that footprint. Adafactor's insight is that the *useful* part of V for a matrix parameter is approximately low-rank, so storing only the row mean-square and column mean-square is enough. This single-tensor reference path lays out the entire math in ~90 lines without `foreach`, fused kernels, or CUDA graph hacks — the cleanest possible teaching version of "low-rank optimizer state". Read this once and Shampoo, GaLore, Muon all look like variations on the same theme.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/_adafactor.py`](https://github.com/pytorch/pytorch/blob/516f64b797cf7645a973e20d856d3e0ddec79948/torch/optim/_adafactor.py#L330-L416)

```python
def _single_tensor_adafactor(
    params, grads, row_vars, col_vars, variances, state_steps,
    grad_scale, found_inf, *,
    d, lr, beta2_decay, weight_decay, eps1, eps2, maximize, has_complex,
):
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Grad scaling should occur outside of optimizer.step()")
    lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        step_t = state_steps[i]
        row_var, col_var, variance = row_vars[i], col_vars[i], variances[i]
        if eps1 is None:
            eps1 = torch.finfo(param.dtype).eps

        # update step
        step_t += 1
        step_float = step_t.item()

        one_minus_beta2_t = step_float**beta2_decay
        rho_t = min(lr, 1 / (step_float**0.5))
        alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

        # Perform stepweight decay
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        if grad.dim() > 1:
            # same as (g * g).mean(dim=-1) w/o materializing an intermediate size g
            row_mean = (
                torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1))
            )
            row_var.lerp_(row_mean, one_minus_beta2_t)
            # same as (g * g).mean(dim=-2) w/o materializing an intermediate size g
            col_mean = (
                torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2))
            )
            col_var.lerp_(col_mean, one_minus_beta2_t)
            var_estimate = row_var @ col_var
            var_estimate.div_(row_var.mean(dim=-2, keepdim=True).clamp_(min=eps1))
        else:
            grad_squared = grad * grad
            variance.lerp_(grad_squared, one_minus_beta2_t)
            var_estimate = variance.clone()

        # square the eps1 as we sqrt after to keep eps1's magnitude
        update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_()
        update.mul_(grad)
        denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * d))
        param.add_(update, alpha=-alpha / denom)
```

## 逐行讲解 / What's happening

1. **学习率的相对衰减 / Relative learning rate (`rho_t`, `alpha`)**:
   - 中文：`rho_t = min(lr, 1/sqrt(t))` 是 Adafactor 自带的 warmup-then-decay 学习率，不需要外面再挂 scheduler；`alpha` 进一步用参数本身的 RMS 缩放 —— 越大的参数允许越大的更新，所以同一个 lr 适用于不同尺度的层。
   - English: `rho_t = min(lr, 1/sqrt(t))` is Adafactor's built-in warmup+decay schedule — you do not need an external scheduler. `alpha` then scales by the param's RMS so larger weights get larger updates, which is what lets one `lr` work across layers of very different scales.

2. **行/列方差更新 / Row and column variance updates**:
   - 中文：对矩阵梯度 `g` 形状 `[m, n]`，`row_mean = (g**2).mean(-1)` 得到 `[m, 1]`，`col_mean = (g**2).mean(-2)` 得到 `[1, n]`。注意它用 `norm(...).square_().div_(...)` 写法是为了不显式 materialize `g*g` 这个 `[m, n]` 的中间 tensor —— 在大模型里能省一倍峰值显存。
   - English: for a matrix grad `g` of shape `[m, n]`, `row_mean = (g**2).mean(-1)` produces `[m, 1]` and `col_mean = (g**2).mean(-2)` produces `[1, n]`. The `norm(...).square_().div_(...)` chain avoids ever materializing the `[m, n]` square — important when `g` is gigabytes.

3. **`lerp_` = EMA 更新 / `lerp_` is the EMA update**:
   - 中文：`row_var.lerp_(row_mean, one_minus_beta2_t)` 等价于 `row_var = (1 - w) * row_var + w * row_mean`，原地写回。`one_minus_beta2_t = t ** beta2_decay`（默认 `-0.8`），随训练逐步减弱，对应 Adafactor 论文里的"逐步固定二阶估计"。
   - English: `lerp_` is in-place `row_var = (1 - w) * row_var + w * row_mean`. The weight `t ** beta2_decay` (default `-0.8`) decays as training progresses, matching the paper's "freeze the second-moment estimate over time" idea.

4. **从两条向量重建 V / Reconstructing V from two vectors**:
   - 中文：`var_estimate = row_var @ col_var`，形状 `[m, 1] @ [1, n] = [m, n]`，得到一个秩 1 的 V 估计；再除以 `row_var.mean(-2)` 做归一化，避免数值放大。这就是 Adafactor 的核心一行：低秩外积近似完整的二阶动量。
   - English: `row_var @ col_var` of shapes `[m, 1] @ [1, n]` yields a rank-1 V estimate over the full `[m, n]` grid. Dividing by `row_var.mean(-2)` keeps the magnitude in the same ballpark as the true V. That single matmul is the whole point of Adafactor.

5. **向量参数走 fallback / Vectors fall back to Adam-like V**:
   - 中文：bias 这种 1D 参数没法分行列，所以直接走 `variance.lerp_(grad**2, ...)`，等价于 Adam 的 V。Adafactor 只对矩阵生效。
   - English: 1-D params (biases, LayerNorm γ) cannot be factored, so they keep a full per-element `variance` — effectively Adam's V. Adafactor only saves memory on matrices.

6. **最后的 RMS 裁剪 / Final RMS clip**:
   - 中文：`update.rsqrt_().mul_(grad)` 就是 `grad / sqrt(V_est)`；`denom = max(1, ||update|| / sqrt(N) / d)` 是 Adafactor 的 update RMS clipping，让单步更新的 RMS 不超过阈值 `d`（默认 1.0），相当于自带的梯度裁剪。
   - English: `update.rsqrt_().mul_(grad)` is the canonical Adam-style `grad / sqrt(V)`. The `denom = max(1, ||update|| / sqrt(N) / d)` is Adafactor's per-step update-RMS clipping; it caps the RMS of one update at `d` (default 1.0), giving you a free, parameter-scale-aware gradient clip.

## 类比 / The analogy

想象你要记一张巨大的 Excel 表里每个格子的平均访问频率：Adam 选择把整张表 `[m, n]` 存下来；Adafactor 说，"每行的平均访问 + 每列的平均访问基本就够了"，于是只存两条边沿。需要某个格子的频率时，用"行均值 × 列均值 / 总均值"近似回去。绝大多数 Excel 表都是低秩的，所以这种近似几乎没误差，但内存只占 `m + n`。

Picture you need to remember the access frequency of every cell in a giant spreadsheet. Adam stores the full `[m, n]` grid. Adafactor argues that the row marginals and the column marginals are nearly all the useful information, so it stores just those two strips. Whenever you need a cell's frequency, you reconstruct it as `(row marginal) * (col marginal) / (overall mean)`. Most real spreadsheets are low rank enough that the approximation is essentially free — but memory drops from `m * n` to `m + n`.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch

torch.manual_seed(0)
W = torch.randn(1024, 1024, requires_grad=True)
opt = torch.optim.Adafactor([W], lr=1e-2)

# fake objective: drive W toward a known target
target = torch.randn_like(W)
for step in range(50):
    loss = (W - target).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step % 10 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")

state = opt.state[W]
print("row_var shape :", state["row_var"].shape)   # [1024, 1]
print("col_var shape :", state["col_var"].shape)   # [1,    1024]
print("variance      :", state.get("variance"))    # None — 2D param, factored path
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step   0  loss=2.0072
step  10  loss=0.4...
step  20  loss=0.0...
...
row_var shape : torch.Size([1024, 1])
col_var shape : torch.Size([1, 1024])
variance      : None
```

中文：注意 state 里 `variance` 是 None，因为参数是 2D 的，走了 factored 路径；同样大小的 Adam state 会多出一张 `[1024, 1024]` 的 V。

English: notice that `variance` is `None` because the parameter is 2-D and went down the factored path. An equivalently-sized Adam would carry an extra `[1024, 1024]` V buffer per parameter.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GaLore** / **GaLore**: 中文 — 把"低秩"从二阶动量推到一阶梯度本身，每隔 N 步重投影梯度到低秩子空间。 / English — pushes the low-rank idea up one level, projecting the *gradient itself* to a low-rank subspace every N steps.
- **Shampoo / Distributed Shampoo** / **Shampoo / Distributed Shampoo**: 中文 — 反过来把"完整 Kronecker 二阶动量"作为目标，跟 Adafactor 的一阶外积近似形成对比。 / English — goes in the opposite direction, building a full Kronecker-factored preconditioner instead of approximating it.
- **8-bit Adam** / **8-bit Adam**: 中文 — 不动 V 的形状，只压缩存储精度。和 Adafactor 是正交的两种省显存策略。 / English — keeps V's shape but quantizes its storage, an orthogonal axis of memory savings.

## 注意事项 / Caveats / when it breaks

- **真的低秩才有效** / **Only works when V is actually low rank**: 中文 — Adafactor 假设矩阵参数的二阶动量近似低秩。如果你的层模式不规则（比如稀疏 embedding 矩阵），近似会变差，需要回退到 Adam 或者 fp32 V。 / English — the rank-1 approximation can be poor for highly heterogeneous weights (e.g. sparse embeddings). Watch for divergent loss in the first few thousand steps and roll back to Adam if you see it.
- **自带的 lr schedule 会干扰外部 scheduler** / **The built-in lr schedule conflicts with external schedulers**: 中文 — `rho_t = min(lr, 1/sqrt(t))` 是硬编码的；如果你在外面叠 warmup，结果会被里面的 `min` 截断。生产用法通常把外部 scheduler 关掉。 / English — the `min(lr, 1/sqrt(t))` clamp is baked in. External warmup schedulers get silently capped; most production recipes disable them when using Adafactor.
- **`step` 是 Tensor 但要 `.item()`** / **`step_t` is a Tensor but called with `.item()`**: 中文 — 注意第 377 行 `step_float = step_t.item()` 会和 CUDA Graph、`torch.compile` 不太友好，因为它强制同步。生产里通常走 `_multi_tensor_adafactor` 这条路径，把 step 留在 GPU 上。 / English — `step_t.item()` forces a device sync and breaks CUDA Graphs / `torch.compile` capture. The `_multi_tensor` path is preferred in production for that reason.

## 延伸阅读 / Further reading

- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (Shazeer & Stern, 2018)](https://arxiv.org/abs/1804.04235)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- [PyTorch Adafactor docs](https://pytorch.org/docs/stable/generated/torch.optim.Adafactor.html)
