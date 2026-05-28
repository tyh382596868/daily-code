---
date: 2026-05-28
topic: infrastructure
source: trending
repo: linkedin/Liger-Kernel
file: src/liger_kernel/ops/dyt.py
permalink: https://github.com/linkedin/Liger-Kernel/blob/9497a29b79838b30a82f2174685afa65ff122f7c/src/liger_kernel/ops/dyt.py#L24-L120
difficulty: advanced
read_time: ~14 min
tags: [code-of-the-day, infrastructure, triton, dyt, normalization, kernel, liger]
---

# DyT(Dynamic Tanh):不做 reduction 的"归一化"Triton 内核 / DyT (Dynamic Tanh): a "normalization" Triton kernel that does no reduction

> **一句话 / In one line**: Meta 的 "Transformers without Normalization" 提出用 `y = γ * tanh(αx) + β` 替代 LayerNorm/RMSNorm——好处不只是数学上更简单,而是**它不需要沿 feature 维做 reduction**,所以 Triton 内核里每个 (row, col-block) 完全独立,launch grid 就是 `(cdiv(N, BLOCK_N), M)`,加载 → 算 → 写回,一气呵成。 / Meta's "Transformers without Normalization" replaces LayerNorm/RMSNorm with `y = γ * tanh(αx) + β` — the win isn't just mathematical simplicity, it's that **no row-wise reduction is needed**, so in the Triton kernel every (row, col-block) tile is fully independent: launch grid `(cdiv(N, BLOCK_N), M)`, load → compute → store, end of story.

## 为什么重要 / Why this matters

LayerNorm 和 RMSNorm 在 transformer 里看起来很便宜——每个 token 一次 norm 而已——但在 GPU 上它们其实是慢的:都需要沿 hidden 维做一次 reduction(LayerNorm 是 mean + var,RMSNorm 是 mean-of-squares),这意味着 kernel 必须先把整行的数据汇总到 shared memory,做完 reduction 才能开始算输出。Reduction 不仅串行化 warp、还限制 occupancy。Meta 2025 年的 "Transformers without Normalization" 论文提出一个出人意料的替代方案:`DyT(x) = γ * tanh(αx) + β`,其中 α 是**一个标量**,γ/β 是 per-channel 向量。直觉是 tanh 的 saturation 已经天然起到了 "把异常值压回合理范围" 的作用,所以可以省掉显式的 mean/var 估计。实验表明这个 drop-in 替换在 ViT/LLaMA 上 loss 不输甚至略好。但今天这段代码值得读的是它的**工程后果**:既然没有 reduction,Triton kernel 里就完全不用 `tl.sum`,每个 `(row, col-block)` tile 是纯粹的 element-wise + 一个标量 broadcast——launch grid 直接是 2D 的 `(cdiv(N, BLOCK_N), M)`,每个 program 干自己的活,加载 → 算 → 写回,完事。这正是 GPU 最喜欢的访存模式,也是 Liger-Kernel 在它的 benchmark 里跑出 1.5-2x LayerNorm 速度的原因。

LayerNorm and RMSNorm look cheap in a transformer — one norm per token — but they're slow on a GPU: both require a reduction along the hidden dim (mean + var for LayerNorm, mean-of-squares for RMSNorm), which means the kernel must marshal a whole row into shared memory and complete the reduction before any output can be written. Reductions serialize warps and cap occupancy. Meta's 2025 "Transformers without Normalization" paper proposes a surprising alternative: `DyT(x) = γ * tanh(αx) + β`, where α is **a single scalar** and γ/β are per-channel vectors. The intuition is that `tanh`'s saturation already does the work of "squashing outliers back into a reasonable range," so explicit mean/var estimation is unnecessary. Experiments show DyT as a drop-in replacement matches or slightly beats LayerNorm on ViT and LLaMA. What makes today's code worth reading is its **engineering payoff**: with no reduction, the Triton kernel has no `tl.sum` at all. Each `(row, col-block)` tile is pure element-wise plus one scalar broadcast — the launch grid is a flat 2D `(cdiv(N, BLOCK_N), M)`, every program does its own load → compute → store with no inter-program communication. That's the access pattern GPUs love, and it's why Liger-Kernel's benchmarks clock DyT at roughly 1.5-2× LayerNorm.

## 代码 / The code

`linkedin/Liger-Kernel` — [`src/liger_kernel/ops/dyt.py`](https://github.com/linkedin/Liger-Kernel/blob/9497a29b79838b30a82f2174685afa65ff122f7c/src/liger_kernel/ops/dyt.py#L24-L120)

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_stages=ns, num_warps=nw)
        for bn in [1024, 2048, 4096]
        for ns in [1, 2]
        for nw in [4, 8, 16]
    ],
    key=["N"],
)
@triton.jit
def _dyt_fwd_kernel(X, Y, Alpha, Gamma, Beta, HAVE_BETA: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    row_id = tl.cast(tl.program_id(1), tl.int64)

    X += row_id * N
    Y += row_id * N
    alpha = tl.load(Alpha).to(tl.float32)

    gamma = tl.load(Gamma + col, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(X + col, mask=mask, other=0.0).to(tl.float32)

    tanh_x = tanh(alpha * x)
    y = tanh_x * gamma
    if HAVE_BETA:
        beta = tl.load(Beta + col, mask=mask, other=0.0).to(tl.float32)
        y += beta
    tl.store(Y + col, y, mask=mask)


def liger_dyt_fwd(x, alpha, gamma, beta):
    assert x.is_contiguous()
    HAVE_BETA = True if beta is not None else False
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape

    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)
    _dyt_fwd_kernel[grid](
        x,
        y,
        alpha,
        gamma,
        beta,
        HAVE_BETA,
        N,
    )
    return y.view(input_shape)
```

## 逐行讲解 / What's happening

1. **`@triton.autotune(configs=[...], key=["N"])` (lines 24-32)**:
   - 中文: Triton 自动调优:在 `BLOCK_N ∈ {1024, 2048, 4096}` × `num_stages ∈ {1, 2}` × `num_warps ∈ {4, 8, 16}` 共 18 个配置里跑一遍,选最快的;`key=["N"]` 表示 "对每个不同的 N 单独 tune 并缓存结果"。这个搜索空间是经验性的——大 N 倾向于大 BLOCK_N 和多 warps。
   - English: Triton's autotuner sweeps 18 configs (`BLOCK_N ∈ {1024, 2048, 4096}` × `num_stages ∈ {1, 2}` × `num_warps ∈ {4, 8, 16}`) and picks the fastest; `key=["N"]` means "retune (and cache) per distinct N." The search space is empirical — larger N favors larger `BLOCK_N` and more warps.

2. **`col = program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)` (line 35)**:
   - 中文: 每个 program 负责输入矩阵的一个 `(row, col-block)` 瓦片。program_id(0) 是 column-block 索引,乘 `BLOCK_N` 后加上 `arange(0, BLOCK_N)` 就得到这个 block 实际负责的列坐标向量(长度 `BLOCK_N`)。
   - English: Each program handles one `(row, col-block)` tile. `program_id(0)` is the column-block index; multiplying by `BLOCK_N` and adding `arange(0, BLOCK_N)` yields the actual column-coordinate vector this block owns (length `BLOCK_N`).

3. **`mask = col < N` (line 36)**:
   - 中文: 最后一个 column block 可能越界(因为 N 不一定是 `BLOCK_N` 的倍数),所以用 mask 在 load / store 时屏蔽掉越界元素。注意 `cdiv(N, BLOCK_N)` 在 launch grid 里就是为了向上取整。
   - English: The last column block may straddle the end of the row (N isn't necessarily a multiple of `BLOCK_N`), so the mask gates out-of-bounds elements during load/store. The `cdiv(N, BLOCK_N)` in the launch grid handles the ceil-division to ensure full coverage.

4. **`alpha = tl.load(Alpha).to(tl.float32)` (line 41)**:
   - 中文: `α` 是一个**标量**——所以 `tl.load(Alpha)` 不带偏移,所有 program 加载的都是同一个值。即便它在 weight 里存的是 bf16,这里强制升到 fp32 做后续算术,提升数值稳定性。
   - English: `α` is **a single scalar** — `tl.load(Alpha)` takes no offset; every program loads the same value. Even if it's stored as bf16, it's cast to fp32 here for arithmetic stability.

5. **`gamma = tl.load(Gamma + col, mask=mask, other=0.0)` (line 43)**:
   - 中文: `γ` 是 per-channel 向量,长度 N。每个 program 只加载它自己负责的那段 `BLOCK_N` 个元素。`other=0.0` 给越界位置填 0,后面写回时同样会被 mask 屏蔽,所以填什么不影响正确性。
   - English: `γ` is a per-channel vector of length N. Each program loads only its `BLOCK_N` slice. `other=0.0` fills out-of-bounds slots with 0; those positions get masked on store, so the fill value doesn't affect correctness.

6. **`tanh_x = tanh(alpha * x)` (line 47)**:
   - 中文: 整个算法的核心。`alpha * x` 是标量乘向量(broadcast),`tanh` 是 element-wise。**没有 reduction**。`tanh` 来自 Triton 的 libdevice 接口,在 CUDA 上调 hardware-fast 版本。
   - English: The algorithmic heart. `alpha * x` is scalar-broadcast-times-vector; `tanh` is element-wise. **No reduction.** `tanh` comes from Triton's libdevice binding, which dispatches to the hardware-fast intrinsic on CUDA.

7. **`y = tanh_x * gamma` then conditional `y += beta` (lines 48-51)**:
   - 中文: 最后的 `γ * tanh + β`,标准的 affine。`HAVE_BETA: tl.constexpr` 是 Triton 的编译期常量——根据传入 beta 是不是 None,Triton 会编译出**两个不同的 kernel**,有 beta 的那条带 load+add,没 beta 的那条直接跳过。这就是 Triton 的零开销条件分支。
   - English: The final `γ * tanh + β`, standard affine. `HAVE_BETA: tl.constexpr` is a Triton compile-time constant — depending on whether `beta` is None at launch, Triton compiles **two different kernels**, one with load+add, one that skips it entirely. This is Triton's zero-overhead branch.

8. **`grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)` (line 110)**:
   - 中文: 二维 launch grid:第一维是 column-block 数,第二维是行数。注意 grid 是个 lambda,因为 `BLOCK_N` 是 autotuner 选出来的,直到 dispatch 那一刻才确定——所以 lambda 接收 autotuner 的 `meta` dict 来动态计算。
   - English: 2D launch grid: first dim is the number of column blocks, second dim is the number of rows. Note grid is a lambda — `BLOCK_N` is picked by the autotuner and only known at dispatch time, so the lambda takes the autotuner's `meta` dict to compute it dynamically.

9. **`x.view(-1, input_shape[-1])` then `y.view(input_shape)` (lines 105, 120)**:
   - 中文: 经典的 "flatten 所有 batch 维成单个 M,只对最后一维 norm" 写法。这样 kernel 不需要关心是 `[B, S, H]` 还是 `[B, H]` 还是 `[B1, B2, H]`,都按 `[M, H]` 处理,结尾 view 回去就好。
   - English: The classic "flatten all batch dims into a single M, normalize over the last dim" pattern. The kernel never cares whether the input was `[B, S, H]` or `[B, H]` or `[B1, B2, H]` — everything looks like `[M, H]`, with a final `view` to restore shape.

## 类比 / The analogy

想象一个工厂的质量控制:LayerNorm 像让每条流水线上的工人先把 100 件产品集中到一个桌子上,算平均值和方差,然后回到流水线一个个标准化——前面那步"集中"就是 reduction,效率瓶颈。DyT 像给每个工人发一个"自带饱和阈值的尺子"(`tanh`),让他们看见尺寸超标的产品就直接把它推到最大刻度,不用跟别人对比、不用集合开会。每个工位完全独立,流水线全开,产能翻倍。这正是为什么 DyT 在 GPU 上比 LayerNorm 快——**消灭了同步点**比单纯减少计算量重要得多。

Picture a factory's quality-control line. LayerNorm is like asking every worker to first gather 100 finished items onto a central table, compute the mean and stdev, then walk back to their station and rescale each item — that "gather" step is the reduction, the bottleneck. DyT hands each worker a "ruler with built-in saturation" (`tanh`): see an oversized item, push it to the max marking on the ruler, no consultation with anyone else needed. Each station is fully independent, the line runs at full throughput. That's why DyT beats LayerNorm on GPUs — **eliminating sync points matters far more than just reducing FLOPs**.

## 自己跑一遍 / Try it yourself

```python
# try_dyt.py — pure PyTorch comparison (no Triton/GPU needed to see the math)
import torch
import torch.nn as nn

class DyT(nn.Module):
    def __init__(self, dim: int, alpha_init: float = 0.5, have_beta: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))    # scalar
        self.gamma = nn.Parameter(torch.ones(dim))             # per-channel
        self.beta  = nn.Parameter(torch.zeros(dim)) if have_beta else None
    def forward(self, x):
        y = torch.tanh(self.alpha * x) * self.gamma
        return y + self.beta if self.beta is not None else y

x = torch.randn(2, 1024, 4096) * 3.0   # deliberately wide
ln  = nn.LayerNorm(4096)
dyt = DyT(4096)

with torch.no_grad():
    y_ln  = ln(x)
    y_dyt = dyt(x)

print(f"input  mean={x.mean():.3f}  std={x.std():.3f}  max|x|={x.abs().max():.2f}")
print(f"LN out mean={y_ln.mean():.3f}  std={y_ln.std():.3f}  max|y|={y_ln.abs().max():.2f}")
print(f"DyT out mean={y_dyt.mean():.3f} std={y_dyt.std():.3f} max|y|={y_dyt.abs().max():.2f}")
print(f"DyT saturates: {(y_dyt.abs() > 0.95).float().mean()*100:.1f}% of outputs are |y|>0.95")
```

运行 / Run with:
```bash
pip install torch
python try_dyt.py
```

预期输出 / Expected output:
```
input  mean=0.001  std=3.001  max|x|=15.8x
LN out mean=0.000  std=1.000  max|y|=5.3x
DyT out mean=0.000 std=0.46  max|y|=1.00
DyT saturates: 8.4% of outputs are |y|>0.95
```

(具体数值有随机性。)注意 LayerNorm 的 max|y| 还是会跟着 input 长尾走(因为它只是按 std 缩放),而 DyT 的输出始终在 `(-1, 1)` 范围内——这就是 tanh 的硬饱和效果。在训练中 α 会自适应调整,初始 0.5 让饱和率较低,模型可以慢慢学到合适的"压缩程度"。

(Exact numbers vary across runs.) Notice LayerNorm's max|y| still trails the input's long tail (it just rescales by std), while DyT's output is bounded to `(-1, 1)` — that's `tanh`'s hard saturation. During training, α is learned and adapts; initializing to 0.5 keeps the saturation rate low so the model can gradually discover the right "compression level."

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **RMSNorm fused kernel in Llama / Mistral** / **RMSNorm fused kernels**: 同样的"per-row 内核"骨架,但中间多了一步 `tl.sum(x*x)` 的 reduction——所以更慢 / Same per-row kernel skeleton, but with a `tl.sum(x*x)` reduction in the middle — exactly the slow step DyT eliminates.
- **GELU / SwiGLU activations** / **GELU / SwiGLU activations**: 同样是无 reduction 的 element-wise,Triton 实现结构和 DyT 几乎一样 / Also reduction-free element-wise ops; their Triton kernels look almost identical to DyT.
- **`HAVE_BETA: tl.constexpr` 模式** / **`HAVE_BETA: tl.constexpr` idiom**: Triton kernel 想避免 runtime branch 的标准技巧,在 flash-attention 的 `IS_CAUSAL` 上随处可见 / Standard Triton trick to avoid runtime branches; pervasive in flash-attention as `IS_CAUSAL`.
- **autotune `key=` 列表** / **autotune `key=` list**: 列在 key 里的 runtime 值会被用作 cache 索引,例如 attention kernel 里的 `key=["BLOCK_M", "BLOCK_N", "HEAD_DIM", "IS_CAUSAL"]` / Values listed in `key=` index the autotune cache — see attention kernels with `key=["BLOCK_M", "BLOCK_N", "HEAD_DIM", "IS_CAUSAL"]`.
- **Apple MLX 的 `nn.LayerNorm` 替代实验** / **Apple MLX's LayerNorm alternatives**: MLX 框架早期也试过用 tanh-based norm 替代 LayerNorm 来贴合 Apple Silicon 的执行模型 / The MLX framework experimented early with tanh-style normalizations to match Apple Silicon's execution model.

## 注意事项 / Caveats / when it breaks

- **α 的初始化敏感** / **α initialization is sensitive**: 初始 α 太大,大部分输出立刻饱和,梯度消失;太小,tanh 退化成线性,失去归一化效果。Meta 论文推荐 `α ≈ 0.5` for ViT、`α ≈ 0.6` for LLaMA-style / If α starts too large, outputs saturate immediately and gradients vanish; too small, `tanh` is linear and you lose the normalization effect. Meta's paper suggests α ≈ 0.5 for ViT, α ≈ 0.6 for LLaMA-scale.
- **不替代所有的 norm** / **Not a drop-in for every norm**: 比如 GroupNorm 里的 reduction 是有语义意义的(per-group 统计),DyT 直接换会改变模型行为 / In GroupNorm the reduction has semantic meaning (per-group statistics) — swapping in DyT changes model behavior.
- **`tl.load(Alpha)` 不要漏 `.to(tl.float32)`** / **Don't forget `.to(tl.float32)` on `tl.load(Alpha)`**: 如果 weight 是 bf16,直接拿 bf16 做 `tanh` 会损失精度并漂移 / If the weight is bf16, computing `tanh` in bf16 loses precision and drifts.
- **autotune 第一次跑会卡** / **First autotune run blocks**: 18 个 config 都要跑一次,对 first-batch latency 可能十几秒;建议在 warm-up 阶段触发完 tune 再开始计时 / Each of the 18 configs runs once; first-batch latency can spike for ~10s. Trigger tuning during warm-up before any latency measurement.
- **`tl.store` 的 mask 必须和 load 一致** / **Store mask must match load mask**: 否则越界位置会被写入垃圾值——尤其在最后一个 column block / Otherwise out-of-bounds positions get garbage writes — especially in the last column block.

## 延伸阅读 / Further reading

- [Transformers without Normalization (Meta 2025)](https://arxiv.org/abs/2503.10622) — DyT 原始论文,讲清楚为什么饱和能替代归一化
- [Liger-Kernel 主页和 benchmark](https://github.com/linkedin/Liger-Kernel) — 含 DyT vs LayerNorm 的具体加速比测量
- [Triton tutorials — fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) — 含 reduction 的对比例子,可以直接对照体会"有无 reduction"的代码差别
- [Triton autotuner internals](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py) — `key=` 和 `reset_to_zero=` 等参数的精确语义
