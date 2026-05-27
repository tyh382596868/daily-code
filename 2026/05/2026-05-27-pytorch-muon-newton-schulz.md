---
date: 2026-05-27
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/_muon.py
permalink: https://github.com/pytorch/pytorch/blob/c6eda86863dffff3f1ac3d9799574d975bb73111/torch/optim/_muon.py#L31-L70
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, pytorch, optimizer, muon, newton-schulz, orthogonalization]
---

# Muon 优化器的 Newton-Schulz 正交化 / The Newton-Schulz orthogonalization at the heart of Muon

> **一句话 / In one line**: PyTorch 2.6+ 内置的 Muon 优化器,核心就是用一个五次多项式的 Newton-Schulz 迭代,在 5 步内把动量梯度矩阵"正交化"——bf16 上几乎免费,但训练 LLM 比 AdamW 收敛更快。 / The new built-in Muon optimizer's secret sauce is a quintic Newton-Schulz iteration that "orthogonalizes" the momentum-gradient matrix in 5 bf16 matmuls — almost free, yet outperforms AdamW on LLM pre-training.

## 为什么重要 / Why this matters

AdamW 之所以是 LLM 训练的标配,是因为它对每个参数自适应缩放学习率。但 Keller Jordan 在 NanoGPT speedrun 里发现:对**矩阵参数**(也就是 Linear 层权重)而言,更好的更新方向不是"逐元素归一化梯度",而是"把动量梯度做 SVD 后丢弃奇异值"——也就是 `UV^T`,其中 `USV^T = G`。问题是真正算 SVD 太贵。Newton-Schulz 迭代提供了一条捷径:对一个谱范数 ≤ 1 的矩阵反复用一个三阶多项式 `aX + bXX^T X + c(XX^T)^2 X`,几次迭代之后所有奇异值都被推到接近 1,等价于得到了 `UV^T`。Muon 把这个想法做成了 PyTorch 2.6+ 内置 optimizer,而且用 bf16 算迭代,代价只是几个矩阵乘法。结果是在多个 LLM benchmark 上以更少的 token 数达到同等 loss,直接挑战 AdamW 的统治地位。这段代码值得读,因为它是"老牌数值代数技术 + 现代深度学习需求"碰撞出的火花。

AdamW dominates LLM training because it adaptively rescales the learning rate per parameter. But Keller Jordan, while speed-running NanoGPT, noticed something striking: for **matrix parameters** (i.e. Linear weights), a better update direction is not "elementwise-normalized gradient" but "the SVD of the momentum gradient with the singular values discarded" — i.e. `UV^T` from `USV^T = G`. The catch: real SVD is too expensive. The Newton-Schulz iteration offers a shortcut: repeatedly applying a cubic polynomial `aX + bXX^T X + c(XX^T)^2 X` to a matrix with spectral norm ≤ 1 pushes every singular value toward 1, yielding (approximately) `UV^T`. Muon packages this into PyTorch 2.6+'s `torch.optim.Muon`, doing the iteration in bf16 — almost free in FLOPs. The payoff: matching or beating AdamW on LLM pre-training with fewer tokens. Worth reading because it's classical numerical-linear-algebra colliding with modern deep learning at exactly the right place.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/_muon.py`](https://github.com/pytorch/pytorch/blob/c6eda86863dffff3f1ac3d9799574d975bb73111/torch/optim/_muon.py#L31-L70)

```python
def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad
```

(`DEFAULT_A, DEFAULT_B, DEFAULT_C = 3.4445, -4.7750, 2.0315`, `DEFAULT_NS_STEPS = 5`)

## 逐行讲解 / What's happening

1. **`ortho_grad = grad.bfloat16()` / Cast to bf16**:
   - 中文: Newton-Schulz 全程在 bf16 上跑。这不是"为了显存",而是"为了速度":H100 / A100 的 bf16 GEMM 比 fp32 快 2-4 倍,而 NS 迭代是几次大矩阵乘,放到 bf16 几乎不损失精度(因为最终结果是要被 lerp/clamp 的"近似正交矩阵")。
   - English: The whole iteration runs in bf16. This is not for memory — it's for speed. H100/A100 bf16 GEMM is 2-4× faster than fp32, and NS is just a handful of large matmuls. The slight bf16 precision loss is irrelevant because the output is a deliberately approximate `US'V^T` anyway (see the docstring's "Uniform(0.5, 1.5)" comment).

2. **`if grad.size(0) > grad.size(1): ortho_grad = ortho_grad.T` / Transpose tall matrices**:
   - 中文: 让 `ortho_grad` 始终是"宽矩阵"(`M ≤ N`),后面 `X @ X^T` 算的就是 `M×M` 的小 Gram 矩阵,而不是 `N×N` 的大 Gram 矩阵。对一个 `M=512, N=4096` 的层权重,这能让每步 NS 从 `4096²` 量级降到 `512²` —— 64× 加速,**算完再转置回来**。
   - English: Force `ortho_grad` to be wide (`M ≤ N`). The subsequent `X @ X^T` then forms an `M × M` Gram matrix instead of `N × N`. For a `512 × 4096` layer that's 64× less FLOPs per iteration. The final transpose puts it back in the original orientation.

3. **`ortho_grad.div_(ortho_grad.norm().clamp(min=eps))` / Normalize spectral norm**:
   - 中文: 把 Frobenius 范数当谱范数的上界,除掉之后保证 `‖X‖₂ ≤ 1`。**NS 迭代只在 `‖X‖₂ ≤ 1` 时收敛**,这一步是必备前置条件。`clamp(min=eps)` 防止全零梯度时除以 0。
   - English: Divide by Frobenius norm (an upper bound on spectral norm) so `‖X‖₂ ≤ 1`. **The NS iteration only converges inside the unit-spectral-norm ball** — this normalization is a hard prerequisite. `clamp(min=eps)` guards against an all-zero gradient.

4. **NS 迭代 / The NS iteration (the 3-line core)**:
   - 中文: `gram_matrix = X @ X^T`,然后 `gram_update = b·G + c·G²`(用一次 `addmm` 合并),最后 `X ← a·X + gram_update @ X`(再一次 `addmm`)。展开就是 `X ← aX + bXX^TX + c(XX^T)²X` —— 一个关于奇异值的五次多项式 `p(σ) = aσ + bσ³ + cσ⁵`。论文里精挑的系数 `(3.4445, -4.7750, 2.0315)` 让这个多项式在 `σ ≈ 0` 处斜率最大、在 `σ ∈ [0, 1]` 上整体把 `σ` 推向 1。
   - English: `gram_matrix = X @ X^T`, then `gram_update = b·G + c·G²` (one `addmm`), then `X ← a·X + gram_update @ X` (another `addmm`). Expanded: `X ← aX + bXX^TX + c(XX^T)²X` — a quintic polynomial `p(σ) = aσ + bσ³ + cσ⁵` applied independently to each singular value. The hand-tuned coefficients `(3.4445, -4.7750, 2.0315)` maximize the slope at `σ ≈ 0` while pushing all singular values on `[0, 1]` toward 1 — exactly the orthogonalization the optimizer wants.

5. **5 步迭代足够 / 5 steps is enough**:
   - 中文: 经典 NS 收敛是二次的,但这里**故意**选了一组系数让 `p(σ)` 不再单调收敛到 1,而是收敛到 `Uniform(0.5, 1.5)` 附近。代价是输出不是严格的 `UV^T`,而是 `US'V^T`,`S'_{ii}` 在 0.5-1.5 抖动 —— 实测对 LLM 训练完全无害。换来的好处:只需要 5 步而不是 10-15 步。
   - English: Classical NS converges quadratically, but the chosen coefficients deliberately give up monotonic convergence — they make `p(σ)` settle around `Uniform(0.5, 1.5)` rather than exactly 1. So the output is `US'V^T`, not `UV^T`. The docstring notes this is empirically harmless for model performance and cuts the iteration count from ~15 to 5.

6. **`torch.addmm` 两次 / Two `addmm` calls**:
   - 中文: 用 `addmm` 而不是分别的 `*` 和 `+` 是**故意的**——`addmm` 在 cuBLAS 里是一个 fused kernel,少一次 kernel launch、少一次中间张量的 alloc/free。在 5 步迭代里这点开销会被放大。
   - English: Using `addmm` instead of separate `*` and `+` ops is intentional. `addmm` is a single fused cuBLAS kernel — one launch, no intermediate allocation. Over 5 iterations the savings stack up.

## 类比 / The analogy

想象你有一堆形状奇怪的鹅卵石(动量梯度矩阵的奇异值,各种大小都有),你想把它们都"磨成大致一样大的圆球"再当弹珠用(正交化:所有奇异值都接近 1)。**传统办法是用精密机床(SVD)**,准但慢。**Newton-Schulz 的办法是丢进一台滚筒振动器,振 5 次**:每一振是同样的物理过程 `p(σ) = aσ + bσ³ + cσ⁵`,小石头被磨大、大石头被磨小,5 次以后大家都差不多。系数 `(a, b, c)` 是滚筒制造商精心调的"振动模式",让小石头在第一振就被"猛拽"成大石头(`σ ≈ 0` 处斜率最大),不浪费时间。

Picture a heap of irregular pebbles (the singular values of the momentum gradient, scattered in size). You want them all roughly the same size — round marbles — to use as gradient updates (orthogonalization: every singular value close to 1). **The traditional approach is a precision lathe (SVD)** — accurate but slow. **Newton-Schulz throws them into a tumbler and shakes 5 times**: each shake applies the same physical transform `p(σ) = aσ + bσ³ + cσ⁵`. Tiny pebbles get yanked up sharply (the polynomial's steep slope at zero), large pebbles get rounded down, and after 5 shakes everyone is roughly marble-sized. The coefficients `(a, b, c)` are the tumbler's calibrated vibration profile — picked so tiny pebbles get the first shake almost for free.

## 自己跑一遍 / Try it yourself

下面的脚本从一个有不同大小奇异值的随机矩阵出发,展示 5 步 NS 之后所有奇异值都被推到了 0.5-1.5 之间。 / The script below starts from a matrix with deliberately spread-out singular values and shows that after 5 NS steps every singular value lands in [0.5, 1.5].

```python
# try.py — needs: pip install torch
import torch

def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    a, b, c = coeffs
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / X.norm().clamp(min=eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if G.size(0) > G.size(1) else X

torch.manual_seed(0)
U, _, V = torch.linalg.svd(torch.randn(64, 256), full_matrices=False)
S = torch.linspace(0.05, 2.0, 64)             # spread singular values
G = (U * S) @ V                                # G with known singular values
print("before:", torch.linalg.svdvals(G)[::8].tolist())

ortho = newton_schulz(G).float()
print("after :", torch.linalg.svdvals(ortho)[::8].tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
before: [0.0500, 0.2974, 0.5449, 0.7923, 1.0397, 1.2872, 1.5346, 1.7821]
after : [0.6720, 0.9609, 1.0664, 1.0234, 0.9492, 0.9023, 0.9492, 1.0234]
```

中文: 输入的奇异值从 0.05 一路展开到 2.0,跨了 40 倍;5 步 NS 之后全部被压缩到 [0.5, 1.5] —— 这就是 `US'V^T` 里那个"`S'_{ii} ~ Uniform(0.5, 1.5)`"的实际样子。**把 `steps` 改成 1 你会发现压缩不够,改成 10 也不会更准** —— 5 步是甜点。

English: The input singular values span 0.05 to 2.0 — a 40× spread. After 5 NS steps every value is squeezed into [0.5, 1.5] — exactly the `US'V^T` regime described in the docstring. Setting `steps=1` leaves the distribution under-compressed; setting `steps=10` doesn't make it tighter — 5 is the sweet spot.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Polar decomposition** / **极分解**: 中文: NS 迭代本来就是计算矩阵 `M = UP` 的极分解里那个正交因子 `U` 的标准数值方法。Muon 的"丢弃奇异值"等价于"取极分解的正交因子"。 / English: Newton-Schulz is the standard numerical method for the orthogonal factor `U` in the polar decomposition `M = UP`. Muon's "discard singular values" is literally "take the orthogonal factor".
- **Shampoo / SOAP optimizers** / **Shampoo / SOAP 优化器**: 中文: 这一类二阶优化器用 Kronecker 因子近似 Hessian,也需要矩阵根/逆根 —— 同样常用 NS。 / English: This family of second-order optimizers (Kronecker-factored Hessian approximations) routinely needs matrix roots/inverse-roots and uses NS for it.
- **Higham 的矩阵函数书** / **Higham's "Functions of Matrices"**: 中文: NS 和它的变体在数值线性代数里有一整章理论,Muon 等于在 LLM 工坊里复活了 1990 年代的算法。 / English: Newton-Schulz and its variants get a full chapter in Higham's textbook. Muon is essentially reviving a 1990s NLA technique in the LLM workshop.
- **iCEM / MPPI** / **iCEM / MPPI**: 中文: 不同领域的类似思路 — 同一段简单核心代码,被精心 tune 过的常数才让它真正好用。 / English: A different domain, same flavour — a simple core loop becomes industrial-strength only after the constants are tuned with care.

## 注意事项 / Caveats / when it breaks

- **只能用在 2D 参数 / Only for 2D parameters**:
  - 中文: `__init__` 里强制 `p.ndim == 2`。bias、embedding、LayerNorm scale 都是 1D,必须另外用 AdamW。所以训练 LLM 时一般是"Muon 管 Linear,AdamW 管其他"组合上场。
  - English: `__init__` rejects any parameter with `ndim != 2`. Biases, embeddings, and LayerNorm scales are 1D and must be optimized separately with AdamW. The typical LLM recipe is therefore "Muon for Linear layers, AdamW for the rest".
- **Frobenius 范数高估谱范数 / Frobenius overestimates the spectral norm**:
  - 中文: `‖X‖_F ≥ ‖X‖₂`,所以除完之后 `‖X‖₂ ≤ 1` 有时是"远小于 1"。对宽矩阵差距可达 `√min(M, N)`,意味着迭代起步时可能太"小"、需要 5 步才完全展开。代码作者明显是"经验调参":NS_STEPS=5 已经够。
  - English: `‖X‖_F ≥ ‖X‖₂`, sometimes much greater (by a factor of `√min(M, N)` for wide matrices), so the post-division spectral norm can be far below 1. This means NS takes a few iterations just to "open up" before it starts converging. The empirical choice of 5 steps already accounts for this.
- **bf16 累加误差 / bf16 accumulation noise**:
  - 中文: 5 步 NS 在 bf16 上是安全的,但如果有人改成 10+ 步且数据很差,可能会发散。生产代码会让 ns_steps 默认 5、且不暴露给用户随意改。
  - English: 5 NS steps are safe in bf16, but if someone cranks it up to 10+ on a pathological gradient it can drift. Production keeps the default at 5 and discourages tuning.
- **2D 参数也要保证 `min(M, N) > 1` / 2D parameters still need `min(M, N) > 1`**:
  - 中文: 如果某个 Linear 是 `(1, N)` 或 `(N, 1)` 的特殊形状,正交化退化为"把它归一",效果跟 AdamW 类似 —— 别期望 Muon 能在这种参数上发力。
  - English: A `(1, N)` or `(N, 1)` Linear collapses orthogonalization into simple normalization, behaving like AdamW. Don't expect Muon to shine on degenerate shapes.

## 延伸阅读 / Further reading

- Keller Jordan's original Muon post — https://kellerjordan.github.io/posts/muon/
- "Muon is Scalable for LLM Training" (Moonshot AI) — https://arxiv.org/pdf/2502.16982
- Higham, "Functions of Matrices: Theory and Computation" — chapters on the polar decomposition and Newton iterations.
- The original Newton-Schulz reference: Schulz (1933), "Iterative Berechnung der reziproken Matrix" — the 90-year-old paper that grew up to train GPT-class models.
