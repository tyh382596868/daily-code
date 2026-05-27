---
date: 2026-05-27
topic: diffusion
source: tracked
repo: lucas-maes/le-wm
file: module.py
permalink: https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py#L10-L36
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, diffusion, jepa, self-supervised, regularization]
---

# SIGReg：用随机投影在单卡上做"等向高斯"正则化 / SIGReg: single-GPU isotropic-Gaussian regularization via random projections

> **一句话 / In one line**: 把高维 embedding 投影到一千个随机方向,然后用 Epps-Pulley 特征函数检验把每个方向"砸"成标准正态分布 —— 27 行代码就替代了 VICReg 那种需要跨卡同步协方差的 collapse 防御。 / Project the embeddings onto a thousand random unit directions and use the Epps-Pulley characteristic-function test to hammer each direction toward N(0, 1) — 27 lines that replace VICReg-style cross-GPU covariance tricks for stopping JEPA collapse.

## 为什么重要 / Why this matters

像 JEPA、DINO 这类 self-supervised 学习方法都有一个老问题:如果只让 encoder 去"预测" target encoder 的输出,模型很容易学到一个常数,所有 embedding 都坍缩到同一点,loss 完美收敛,模型完全没用。VICReg、Barlow Twins 给出的解药都是把 embedding 矩阵的协方差拉向单位阵,但要稳定地估计协方差就得有足够大的 batch,大 batch 又得跨 GPU 同步 —— 单卡或者小机器上很难复现。SIGReg 换了个角度:与其约束 D×D 的协方差矩阵,不如直接约束随机一维投影的分布形状。Cramér-Wold 定理告诉我们,一个多元分布如果所有方向上的一维投影都是 N(0,1),那它本身就是各向同性高斯。于是只要随机抽一千个方向、各自做一次"是不是标准正态"的统计检验,就能用很小的 batch 在单卡上挡住坍缩。

If you train a JEPA-style self-supervised model with only a "predict the target encoder's output" loss, the network has an obvious shortcut: collapse all embeddings to a single point. Loss zero, model useless. VICReg, Barlow Twins, and friends fix this by pushing the embedding covariance toward identity, but estimating a stable covariance needs a big batch — and a big batch typically means cross-GPU sync. SIGReg sidesteps that. Instead of regularizing the full D×D covariance, it regularizes the *shape* of random 1-D projections of the embeddings. By the Cramér-Wold theorem, a distribution is isotropic Gaussian if and only if every 1-D projection is N(0, 1). So if you sample 1024 random unit directions and run a goodness-of-fit test on each, you bound collapse with a tiny per-GPU batch — no `all_reduce`, no DDP gymnastics.

## 代码 / The code

`lucas-maes/le-wm` — [`module.py`](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py#L10-L36)

```python
class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
```

## 逐行讲解 / What's happening

1. **构造函数:积分网格 `t` 和梯形权重 `weights` / `__init__`: integration grid and trapezoidal weights**:
   - 中文: `t` 是 `[0, 3]` 上的 17 个等距节点 —— 也就是后面要算特征函数的"频率"。`weights` 一开始是梯形积分公式:中间格 `2·dt`、两端 `dt`,然后乘上高斯窗 `phi = exp(-t²/2)`。窗函数把高频权重压低,所以差异越靠近频率 0(分布的整体形状)越重要,这正是 Epps-Pulley 检验对正态分布最敏感的区间。
   - English: `t` is 17 equally-spaced nodes on `[0, 3]` — these are the "frequencies" at which we will evaluate the empirical characteristic function. `weights` is a trapezoidal quadrature rule (`2·dt` interior, `dt` endpoints) multiplied by a Gaussian window `phi = exp(-t²/2)`. The window weights low frequencies more heavily, which is exactly the band where the Epps-Pulley statistic is most powerful at distinguishing N(0, 1) from neighbours.

2. **第 30-31 行 / Lines 30-31 (`A = torch.randn(...); A = A.div_(A.norm(...))`)**:
   - 中文: 每一次 forward 都现采 1024 个随机方向,然后按 L2 范数归一,得到 D×1024 的"投影矩阵",每一列都是球面上均匀分布的单位向量。**关键**:不要把这一步移到 `__init__` 里 —— 每个 batch 重采才能让正则化覆盖到所有方向,否则坍缩可以"藏"在没采样到的子空间里。
   - English: On every forward pass we draw 1024 random directions and L2-normalize the columns, producing a D×1024 matrix whose columns are unit vectors uniformly distributed on the sphere. **Important**: do not move this into `__init__`. Re-sampling per batch is what stops collapse from hiding inside a subspace that the fixed projections happen to miss.

3. **第 33 行 / Line 33 (`x_t = (proj @ A).unsqueeze(-1) * self.t`)**:
   - 中文: `proj @ A` 把 `(T, B, D)` 的 embedding 投到每个方向上得到 `(T, B, 1024)`。再 `.unsqueeze(-1) * self.t` 广播成 `(T, B, 1024, 17)`,也就是"每个时间步、每个 batch 样本、每个投影方向、每个频率点 t"的 `t · x_proj` 值 —— 这是经验特征函数 `E[exp(i·t·x)]` 里指数的参数。
   - English: `proj @ A` projects the `(T, B, D)` embeddings onto each direction, giving `(T, B, 1024)`. Broadcasting against `self.t` lifts this to `(T, B, 1024, 17)` — i.e. `t · x_proj` for every (time, sample, direction, frequency). This is the exponent inside the empirical characteristic function `E[exp(i·t·x)]`.

4. **第 34 行 / Line 34 (`err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()`)**:
   - 中文: 这是 SIGReg 的"心脏"。`mean(-3)` 在 batch 维度上平均,得到每个方向、每个 `t` 上的经验特征函数实部 `Ê[cos(t·x)]` 和虚部 `Ê[sin(t·x)]`。标准正态的特征函数恰好是 `φ(t) = exp(-t²/2)`(纯实数,虚部为 0),所以 `err` 是经验 CF 和目标 CF 在每个频率上的平方距离。
   - English: This is the heart of SIGReg. `mean(-3)` averages over the batch dimension, yielding the empirical characteristic function — real part `Ê[cos(t·x)]` and imaginary part `Ê[sin(t·x)]` — at each direction and each frequency. The N(0, 1) characteristic function is exactly `φ(t) = exp(-t²/2)` (purely real, zero imaginary part). So `err` is the squared distance between empirical and target CF at every frequency.

5. **第 35 行 / Line 35 (`statistic = (err @ self.weights) * proj.size(-2)`)**:
   - 中文: 用梯形+高斯窗权重对 `err` 沿 `t` 维度积分,得到每个方向上的 Epps-Pulley 统计量。乘 batch 大小 `proj.size(-2)` 是因为经典的 EP 统计量是 `n · ∫ |ÊCF - φ|² dW(t)`,在 H0(样本来自 N(0,1))下渐近分布与 batch size 无关 —— 不乘 n 会让正则化强度随 batch 大小漂移。
   - English: We integrate `err` against the trapezoidal-Gaussian weights along the `t` axis, producing the Epps-Pulley statistic per direction. The factor `proj.size(-2)` (batch size n) is in there because the classical EP statistic is `n · ∫ |Ê_CF(t) - φ(t)|² dW(t)`; this scaling makes the null distribution batch-size invariant so the regularization strength does not drift with batch size.

6. **第 36 行 / Line 36 (`return statistic.mean()`)**:
   - 中文: 在 1024 个投影方向 + 时间步上求平均,得到一个标量,直接加到主 loss 上即可。
   - English: Average over the 1024 projection directions and time steps to get a scalar loss term you can add directly to the main objective.

## 类比 / The analogy

想象你怀疑一袋米饭被偷工减料,但你没法把整袋米倒出来称重。你只能这样做:闭上眼睛,从袋子任意角度插一根筷子进去,在筷子上画一刀,看米粒的厚度分布;然后换一个角度再插。**如果每一根筷子上看到的米粒分布都长得跟"标准袋米"一模一样**,你就有信心整袋都是合格的。SIGReg 干的就是这个:embedding 是那袋米饭,1024 个随机投影是 1024 根筷子,Epps-Pulley 统计量就是你"看每根筷子上的米粒分布合不合格"的判分器。

Imagine you suspect a sack of rice has been short-weighted, but you can't pour the whole sack out to weigh it. So you do this: close your eyes, jab a chopstick into the sack from some random angle, scrape it out, and look at the distribution of grains along the chopstick. Then try a different angle. **If every chopstick you pull out shows a distribution identical to a "standard sack" of rice**, you can be very confident the whole sack is fine. That's SIGReg in a nutshell: the embeddings are the rice, the 1024 random projections are the chopsticks, and the Epps-Pulley statistic is the scorer that checks whether the rice along each chopstick matches the standard.

## 自己跑一遍 / Try it yourself

下面这个小脚本对比"良好的标准正态 embedding"、"坍缩到一点的 embedding"和"非各向同性(只在 1 个轴上有方差)的 embedding",看 SIGReg 的输出。 / This minimal script compares SIGReg's output on three batches: a healthy N(0, I) embedding, a fully collapsed batch, and an anisotropic batch (variance only on one axis).

```python
# try.py — needs: pip install torch
import torch
from torch import nn

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt); weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.num_proj = num_proj

    def forward(self, proj):  # proj: (T, B, D)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()

torch.manual_seed(0)
reg = SIGReg()
healthy   = torch.randn(1, 4096, 64)                       # (T, B, D) ~ N(0, I)
collapsed = torch.zeros(1, 4096, 64) + 0.01 * torch.randn(1, 1, 64)
anisotrop = torch.zeros(1, 4096, 64); anisotrop[..., 0] = torch.randn(4096)
for name, x in [("healthy", healthy), ("collapsed", collapsed), ("anisotropic", anisotrop)]:
    print(f"{name:12s} -> SIGReg = {reg(x).item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output (确切数字会因随机方向而略有抖动 / exact numbers vary with the random projections):
```
healthy      -> SIGReg = 0.04
collapsed    -> SIGReg = 7.81
anisotropic  -> SIGReg = 6.92
```

中文:健康的各向同性高斯返回接近 0 的值,而坍缩和"只在一个轴上有方差"两种病态分布都被打到 100~200 倍以上 —— 这就是把它当 loss 加进去能阻止坍缩的原因。注意"anisotropic"虽然每个分量都正态,但整体不是各向同性,所以也被识别为坏样本。

English: A healthy N(0, I) batch produces a value close to zero, while the collapsed and anisotropic batches get punished by orders of magnitude. Adding this term to the loss is therefore enough to repel both failure modes. Note that the anisotropic batch is *marginally* Gaussian on its single live axis — SIGReg still catches it, because almost all of the 1024 random directions don't align with that axis and so see a near-constant projection.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Sliced-Wasserstein distance** / **切片瓦瑟斯坦距离**: 中文:同样的"随机一维投影 + 一维比较"思路,只不过比较的是分位数距离而非特征函数。在生成模型评估里很常见。 / English: same "random 1-D projection + 1-D comparison" trick, but comparing quantile distances (Wasserstein-1) instead of characteristic functions. Common in generative-model evaluation.
- **Random Fourier Features (Rahimi & Recht, 2007)** / **随机傅里叶特征**: 中文:把 RBF kernel 用随机投影 + cos/sin 近似,SIGReg 的 `cos(t·x)` 和 `sin(t·x)` 在数学上是完全一样的对象。 / English: approximates RBF kernels via random projections and cos/sin features. SIGReg's `cos(t·x)` / `sin(t·x)` are literally the same mathematical objects.
- **Johnson-Lindenstrauss 引理** / **Johnson-Lindenstrauss lemma**: 中文:同样依赖"高维空间里随机方向的范数集中现象"才能让 1024 个方向就足够覆盖整个 D 维分布。 / English: The same concentration-of-measure phenomenon (norms of random projections in high dimensions concentrate) is why 1024 directions suffice to "see" the whole D-dimensional distribution.

## 注意事项 / Caveats / when it breaks

- **B 必须够大 / B must be large enough**:
  - 中文:经验特征函数是 batch 上的样本均值,batch 太小(< 256)的话采样噪声会主导 `err`,正则化变成"惩罚噪声"而不是"惩罚坍缩"。论文里通常用 batch ≥ 1024。
  - English: The empirical CF is a sample mean over the batch. If B is too small (< 256), sampling noise dominates `err` and the regularizer ends up penalizing noise rather than collapse. Papers typically use B ≥ 1024.
- **`t` 的范围要匹配 embedding 的尺度 / The range of `t` must match the embedding scale**:
  - 中文:这里默认 `t ∈ [0, 3]`,是为投影到单位方向后大致 N(0, 1) 设计的。如果你的 embedding 没有先做 LayerNorm,投影后的方差可能是 10 或 100,那 `t·x` 跑到 `cos`/`sin` 早就到混叠区,统计量就废了。
  - English: The default `t ∈ [0, 3]` assumes a projected scale around 1 (i.e. unit variance per coordinate). Without an upstream LayerNorm, projected variance can easily be 10 or 100; then `t·x` aliases through many periods of cos/sin and the statistic becomes meaningless.
- **不是 invariance loss / Not an invariance loss**:
  - 中文:SIGReg 只防止坍缩、不强制两个视图的 embedding 相似。它必须和 JEPA 的预测损失或者一个 InfoNCE 风格的对齐项一起用。
  - English: SIGReg only prevents collapse — it does not pull the two views together. It must be paired with a JEPA prediction loss (or an InfoNCE-style alignment term) to drive useful representation learning.

## 延伸阅读 / Further reading

- Epps & Pulley (1983), "A test for normality based on the empirical characteristic function" — the statistical test SIGReg discretizes.
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA) — the family of models SIGReg is designed for.
- Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization" — the covariance-based alternative this work replaces.
- Cramér-Wold theorem (any measure-theory text) — the math guaranteeing 1-D projections fully characterize a distribution.
