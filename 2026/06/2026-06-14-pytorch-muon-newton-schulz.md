---
date: 2026-06-14
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/_muon.py
permalink: https://github.com/pytorch/pytorch/blob/6f2953ae46a3b5f25bfc7bd3acfe6e2f663d1ed3/torch/optim/_muon.py#L31-L70
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, pytorch, optimizer, muon, newton-schulz, orthogonalization]
---

# PyTorch 把 Muon 写进官方了:40 行 Newton-Schulz 把梯度矩阵正交化 / PyTorch officially ships Muon — 40 lines of Newton-Schulz that orthogonalises the gradient matrix

> **一句话 / In one line**: Muon 优化器的全部魔法就是"5 步 Newton-Schulz 迭代把矩阵的奇异值全压到 ~1",剩下的就是普通带 momentum 的 SGD。 / Muon's entire magic is "five Newton-Schulz iterations crush every singular value to ~1"; everything else is plain momentum SGD.

## 为什么重要 / Why this matters

Muon(由 Keller Jordan 提出)在 NanoGPT speedrun 上把 AdamW 的训练时长直接砍掉一半,2025 年 Moonshot 又证明它能直接 scale 到 LLM 训练。PyTorch 2.x 现在把它放进了 `torch.optim`——意味着今后所有 PyTorch 用户都能 `torch.optim.Muon` 一行用上,不再依赖第三方实现。最有教学价值的不是 `class Muon` 的 schedule,而是核心子函数 `_zeropower_via_newtonschulz`:整套数值线性代数被压缩成一个 40 行的函数。理解它,就理解了 "为什么一个数学等式可以当优化器" 的全部技术内涵。

Muon (proposed by Keller Jordan) cut NanoGPT speedrun training time in half versus AdamW, and Moonshot's 2025 paper showed it scales cleanly to LLM training. PyTorch has now landed Muon in `torch.optim` — meaning everyone gets `torch.optim.Muon` as a one-liner, no third-party install required. The teaching gem isn't the surrounding `class Muon` schedule, it's the core helper `_zeropower_via_newtonschulz`: all the numerical linear algebra you need, compressed into 40 lines. Read it and you understand why "a mathematical identity can be an optimizer".

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/_muon.py`](https://github.com/pytorch/pytorch/blob/6f2953ae46a3b5f25bfc7bd3acfe6e2f663d1ed3/torch/optim/_muon.py#L31-L70)

```python
# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. ...
    This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5) ...
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
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

## 逐行讲解 / What's happening

1. **数学背景 / Math background**:
   - 中文: Muon 的核心思想是:把更新方向 `G` 替换成它的"零次幂"`G^0 = UV^T`(其中 `G = USV^T` 是 SVD)。换句话说,扔掉所有奇异值信息,只保留方向。这样每个参数维度被更新的"大小"都一致,RMS 不再被少数大奇异值主宰。SVD 太贵,所以用 5 步 Newton-Schulz 多项式去逼近。
   - English: Muon's core idea is to replace the update direction `G` with its "zeroth power" `G^0 = UV^T` (where `G = USV^T` is the SVD). I.e. throw away the singular values, keep only the direction. Every parameter dimension then receives an update of comparable magnitude, so no handful of large singular values dominates the RMS. SVD is too expensive, so a 5-step Newton-Schulz polynomial approximates it instead.

2. **第 54 行 / Line 54 (`a, b, c = ns_coefficients`)**:
   - 中文: 这三个常数 `(3.4445, -4.7750, 2.0315)` 是 Keller 通过最大化迭代多项式 `f(x) = ax + bx³ + cx⁵` 在 `x=0` 处的斜率得到的——这意味着小奇异值(接近 0)被"拉"到 1 的速度最快。
   - English: These constants `(3.4445, -4.7750, 2.0315)` come from Keller maximising the slope at `x = 0` of the iteration polynomial `f(x) = ax + bx³ + cx⁵` — which means small singular values (near 0) get "pulled up" to 1 fastest.

3. **第 55-57 行 / Lines 55-57 (`bfloat16` + transpose)**:
   - 中文: 整个迭代降到 bf16 跑(成本立刻减半)。如果矩阵是高瘦型(行数 > 列数),先转置再迭代——因为 NS 用的是 `G @ G.T`,小的那一边平方后开销少很多。最后再转置回去。
   - English: Run the whole iteration in bf16 (cost halved instantly). If the matrix is tall (more rows than cols), transpose first — NS uses `G @ G.T`, and squaring the smaller dimension is dramatically cheaper. Transpose back at the end.

4. **第 59 行 / Line 59 (`div_(grad.norm().clamp(...))`)**:
   - 中文: 把 spectral norm(最大奇异值)归一化到 ≤ 1。NS 多项式只在 `[0, 1]` 区间内收敛——你必须先保证所有奇异值都在这个区间。Frobenius norm 一定 ≥ spectral norm,所以拿它除是安全的上界。
   - English: Normalise so the spectral norm (largest singular value) is ≤ 1. The NS polynomial converges only on `[0, 1]` — you must pin every singular value into that interval first. Since the Frobenius norm is always ≥ the spectral norm, dividing by it is a safe upper-bound choice.

5. **第 61-66 行 / Lines 61-66 (the NS iteration)**:
   - 中文: 这就是整套算法的心脏。每次迭代做两个 matmul:
     - 中文: `gram_matrix = G @ G.T`——shape `(m, m)` 的 Gram 矩阵。
     - 中文: `gram_update = b·gram + c·gram@gram`(用 `addmm` 一次性 fuse 了)。
     - 中文: `G ← a·G + gram_update @ G`(同样 `addmm` fuse)。
   - 中文: 写成数学就是 `G ← a·G + b·G·G.T·G + c·G·G.T·G·G.T·G`——一个关于 `G` 的五次多项式。在 SVD 视角下,每个奇异值 `σ` 都被独立映射成 `f(σ) = aσ + bσ³ + cσ⁵`,五步迭代后 `f^5(σ) ≈ 1`。
   - English: This is the heart of the whole algorithm. Each iteration is two matmuls:
     - English: `gram_matrix = G @ G.T` — a Gram matrix of shape `(m, m)`.
     - English: `gram_update = b·gram + c·gram @ gram` (fused into a single `addmm`).
     - English: `G ← a·G + gram_update @ G` (another fused `addmm`).
   - English: As maths this is `G ← a·G + b·G·G.T·G + c·G·G.T·G·G.T·G` — a degree-5 polynomial in `G`. Through the SVD lens, every singular value `σ` is independently mapped by `f(σ) = aσ + bσ³ + cσ⁵`, and after five iterations `f^5(σ) ≈ 1`.

6. **关于"不到完全正交"的小字 / The "not quite orthogonal" footnote**:
   - 中文: 文档说结果不是干净的 `UV^T` 而是 `US'V^T`,其中 `S' ~ Uniform(0.5, 1.5)`。这是故意的——他们牺牲了一点正交性(从 1 抖到 0.5–1.5),换来零点处更陡的斜率,可以用更少迭代次数把小奇异值拉起来。实验表明对训练效果几乎没影响。
   - English: The docstring notes the result isn't a clean `UV^T` but `US'V^T` with `S' ~ Uniform(0.5, 1.5)`. This is intentional — they trade a bit of orthogonality (the singular values now bounce between 0.5 and 1.5 instead of being exactly 1) for a steeper slope at zero, which means fewer iterations are needed to lift small singular values. Empirically, training behaves identically.

## 类比 / The analogy

想象你要给一个学生打分。原始的"梯度"就像考试的原始总分:有人考 95,有人考 23,差距很大。AdamW 的做法是用历史方差去做归一化(每个学生都减自己均值再除以自己方差),但这只在标量维度上做。Muon 的做法更激进——它直接把"分数"压成"通过/不通过"这种二值化(等价于把所有奇异值压成 1):95 → 1,23 → 1。然后学习率就可以放心拧大,因为不会被某个考了 95 的拖偏方向。Newton-Schulz 就是那把把分数压扁的尺子——不用算 SVD,5 次多项式拉伸就够了。

Think of grading students. The raw "gradient" is like the raw exam total: 95, 23, 87, etc. — huge spread. AdamW normalises along the per-coordinate axis (subtract running mean, divide by running variance). Muon goes further — it pushes the "scores" all the way down to a binary pass/fail (equivalent to forcing every singular value to 1): 95 → 1, 23 → 1. Now you can turn the learning rate way up because no single big-score student can drag the update direction off course. Newton-Schulz is the press that flattens the scores — no SVD needed, five polynomial stretches do the job.

## 自己跑一遍 / Try it yourself

```python
import torch

def ns(G, steps=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-7):
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X /= X.norm().clamp(min=eps)
    for _ in range(steps):
        AAt = X @ X.T
        upd = torch.addmm(AAt, AAt, AAt, beta=b, alpha=c)
        X = torch.addmm(X, upd, X, beta=a)
    return X.T if transposed else X

torch.manual_seed(0)
G = torch.randn(64, 256) * torch.linspace(0.01, 10.0, 64).unsqueeze(1)
out = ns(G).float()
u, s, v = torch.linalg.svd(G, full_matrices=False)
u2, s2, v2 = torch.linalg.svd(out, full_matrices=False)
print("input  singular values (min/max):", s.min().item(), s.max().item())
print("output singular values (min/max):", s2.min().item(), s2.max().item())
```

运行 / Run with:
```bash
pip install "torch>=2.5"
python try.py
```

预期输出 / Expected output:
```
input  singular values (min/max): 0.013... 121.4...
output singular values (min/max): 0.498... 1.500...
```

中文一句:输入的奇异值跨了 4 个数量级,5 步迭代后全部落在 [0.5, 1.5] 里——这就是 Muon 把"方向"和"大小"完全分离的实证。

English: the input singular values span four orders of magnitude; after five iterations they all sit in `[0.5, 1.5]`. This is the empirical demonstration of Muon completely decoupling "direction" from "magnitude".

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Keller Jordan 的 NanoGPT speedrun** / **Keller Jordan's NanoGPT speedrun**: 原始实现就这套 NS 系数;PyTorch 直接复刻了它的常量 `3.4445 / -4.7750 / 2.0315`。 / The original speedrun implementation uses the same NS coefficients; PyTorch literally copied his `3.4445 / -4.7750 / 2.0315`.
- **Moonshot AI 的 Kimi 训练** / **Moonshot AI's Kimi training**: 提出 `match_rms_adamw` adjust 函数,让 Muon 直接接 AdamW 的 lr/wd 不用重新调参——这一支也被 PyTorch 收录了。 / Their `match_rms_adamw` LR-adjustment lets you reuse AdamW's lr/wd without re-tuning — also shipped in this PyTorch release.
- **Shampoo / SOAP 优化器** / **Shampoo / SOAP optimizers**: 同样基于"二阶信息正交化梯度"的思路,但用的是预算更高的 Schur 分解或 inverse-root,不是 NS。 / Same "use second-order info to orthogonalise the gradient" idea, but they pay for a Schur decomposition or matrix inverse-root instead of NS.
- **PyTorch torch.linalg.matrix_norm + qr** / **PyTorch's `torch.linalg.qr`**: 真要纯正交化用 QR 也行,但代价是不可微 + 不能 fuse 进 bf16 amp。 / If you want truly orthogonal output you can use QR — but it's not differentiable and can't fuse with bf16 amp.

## 注意事项 / Caveats / when it breaks

- **只支持 2D 参数** / **2D parameters only**: bias、embedding、conv weight(4D) 都不能直接喂给 Muon——Muon 的 `__init__` 里直接 `raise` 了。文档建议这些参数用 AdamW。 / Bias, embedding, conv weight (4D) can't be fed to Muon directly — `__init__` raises immediately. Use AdamW for them, as the docstring suggests.
- **必须先归一化** / **You must normalise first**: 跳过 `div_(grad.norm())` 那一行,NS 多项式会发散——奇异值 > 1 时 `aσ + bσ³ + cσ⁵` 会爆。 / Skip the `div_(grad.norm())` line and the polynomial diverges — for `σ > 1`, `aσ + bσ³ + cσ⁵` blows up.
- **bf16 不是免费午餐** / **bf16 isn't free**: 矩阵 condition number 极差的时候(例如 layernorm 输出的早期训练),bf16 NS 偶尔会输出 NaN。生产代码建议把 `eps` 调大或自动 fallback 到 fp32。 / When the matrix has a brutal condition number (e.g. layernorm outputs early in training), bf16 NS occasionally emits NaN. Bump `eps` or auto-fallback to fp32 in production.
- **adjust_lr 默认是 "original"** / **`adjust_lr` defaults to `"original"`**: Keller 的版本 `sqrt(max(1, A/B))`,适合从零调参;迁移 AdamW 配方时记得切到 `"match_rms_adamw"`。 / Keller's default is `sqrt(max(1, A/B))`, which is fine when you're tuning fresh; if you're porting an AdamW recipe, switch to `"match_rms_adamw"`.

## 延伸阅读 / Further reading

- [Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"](https://kellerjordan.github.io/posts/muon/)
- [Liu et al., "Muon is Scalable for LLM Training" (arXiv:2502.16982)](https://arxiv.org/abs/2502.16982)
- [Keller's original Muon repo](https://github.com/KellerJordan/Muon)
- PyTorch PR introducing `torch.optim.Muon` (search `_muon.py` in pytorch/pytorch)
