---
date: 2026-06-04
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/swa_utils.py
permalink: https://github.com/pytorch/pytorch/blob/3348c74c01b3448d82baba8379eb8a63017ba064/torch/optim/swa_utils.py#L42-L82
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, pytorch, ema, swa, foreach, multi-tensor]
---

# PyTorch 用一个 fused lerp_ 写完了 EMA / PyTorch's EMA is one fused lerp_ over the whole parameter list

> **一句话 / In one line**: `get_ema_multi_avg_fn` 把 EMA 公式 `W_ema = decay*W_ema + (1-decay)*W_model` 重写成等价的 `torch._foreach_lerp_(W_ema, W_model, 1-decay)` —— 全部参数一个 CUDA kernel 搞定,跳过 Python loop。 / `get_ema_multi_avg_fn` rewrites the EMA update `W_ema = decay*W_ema + (1-decay)*W_model` as the algebraically identical `torch._foreach_lerp_(W_ema, W_model, 1-decay)` — all parameters in a single fused CUDA kernel, no Python loop.

## 为什么重要 / Why this matters

EMA 在扩散模型、VLA、对比学习里都是"另一个模型"——大家在训练 student,但用于 eval / sampling 的是 EMA 副本。问题是参数表少则上千张张量,多则上万张(7B 模型 ≈ 4000 张),一张张更新会让 Python overhead 比实际算术还多。PyTorch 的标准做法是 multi-tensor APIs:把整个 list 喂给一个 `_foreach_*` kernel,在 GPU 上一次性处理。而 EMA 这件事的关键观察是 —— 它的数学等价于 `lerp`(linear interpolation)。PyTorch 直接调用现成的 `_foreach_lerp_`,EMA 实现就只剩 40 行,而且免费拿到 fused 性能。这段代码值得每个写训练循环的人看一眼,因为它把"代数化简 → 复用已有 fused kernel"这个思维范式展示得很干净。

EMA shows up everywhere: diffusion training, VLAs, contrastive SSL — the model you actually evaluate or sample with is the EMA copy. The problem is that there are anywhere from a thousand to ten thousand parameter tensors per model (a 7B model is ~4000), and a per-tensor update loop spends more time in Python than doing arithmetic. PyTorch's standard answer is multi-tensor APIs: hand the full list to a single `_foreach_*` kernel and let the GPU process them as one batched operation. The clever observation that powers EMA in particular is that its math is algebraically identical to `lerp` (linear interpolation), so PyTorch can reuse the existing `_foreach_lerp_`. The whole implementation collapses to 40 lines and inherits fused-kernel performance for free. Worth a read because it's a clean demo of the "rewrite to reuse an existing fused kernel" mindset that you should be doing in your own training code.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/swa_utils.py`](https://github.com/pytorch/pytorch/blob/3348c74c01b3448d82baba8379eb8a63017ba064/torch/optim/swa_utils.py#L42-L82)

```python
def get_ema_multi_avg_fn(decay=0.999):
    """Get the function applying exponential moving average (EMA) across multiple params.

    The EMA is computed as:

    .. math::
        W_0^{\\text{EMA}} = W_0^{\\text{model}}

    .. math::
        W_{t+1}^{\\text{EMA}} = \\text{decay} \\times W_t^{\\text{EMA}} + (1 - \\text{decay}) \\times W_{t+1}^{\\text{model}}

    where :math:`W_t^{\\text{EMA}}` is the EMA parameter at step :math:`t`,
    :math:`W_t^{\\text{model}}` is the model parameter at step :math:`t`,
    and :math:`\\text{decay}` is the decay rate (default: 0.999).

    Args:
        decay (float): Decay rate for EMA. Must be in the range [0, 1]. Default: 0.999

    Returns:
        Callable: A function that updates EMA parameters given current model parameters
    """

    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @torch.no_grad()
    def ema_update(
        ema_param_list: PARAM_LIST, current_param_list: PARAM_LIST, _
    ) -> None:
        # foreach lerp only handles float and complex
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(
            ema_param_list[0]
        ):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list, strict=True):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))

    return ema_update
```

## 逐行讲解 / What's happening

1. **`get_ema_multi_avg_fn(decay)` 是一个工厂函数**:
   - 中文: 它不直接做 EMA,而是返回一个 closure。这样 `AveragedModel` 就可以把 EMA / SWA / 你自定义的平均策略当成一个参数传进去,内部统一用同一个调用接口。
   - English: It doesn't do EMA itself — it returns a closure. That lets `AveragedModel` accept EMA, SWA, or your own custom averaging strategy as a parameter and dispatch through one uniform interface.
2. **`torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)`**:
   - 中文: 这是整段代码的关键一行。`torch.lerp(a, b, w) = a + w * (b - a) = (1-w)*a + w*b`。代入 `w = 1-decay`,正好等于 `decay*a + (1-decay)*b` —— 这就是 EMA。`_foreach_lerp_` 是 multi-tensor 版本,所有参数张量塞进同一个 CUDA kernel 一次性算完,inplace 写回 `ema_param_list`。
   - English: The pivot line. Algebraically `torch.lerp(a, b, w) = a + w * (b - a) = (1-w)*a + w*b`. Plug in `w = 1 - decay` and you get `decay*a + (1-decay)*b` exactly — the EMA update. `_foreach_lerp_` is the multi-tensor variant: it batches every parameter tensor into a single fused CUDA kernel and writes back in place.
3. **`torch.is_floating_point(...) or torch.is_complex(...)` 的判断**:
   - 中文: 整数 buffer(比如 `num_batches_tracked`)不能 lerp,所以走 Python fallback。日常 EMA 用例 99% 都是浮点,这个分支几乎不进。
   - English: Integer buffers (e.g. `num_batches_tracked` in `BatchNorm`) don't support `lerp`, so a Python fallback handles them. In real EMA workloads this branch is essentially never taken.
4. **`@torch.no_grad()` 装饰器**:
   - 中文: EMA 更新本身不应该进 autograd 图,否则反向传播会把它当成可导操作往 EMA 上回传梯度。
   - English: The EMA update must not be tracked by autograd; otherwise backward would treat it as a differentiable op and try to push gradients into the EMA copy.

## 类比 / The analogy

你在调一壶饮料,大锅里是"过去 10000 步平均的样本"(EMA),小杯子里是"刚做出来的最新一步"(current)。两种做法:
- 笨办法:一勺一勺,从小杯子取 0.001 倒进大锅、再从大锅取 0.999 倒回去 —— 一万个参数张量做一万次。
- 聪明办法:把大锅和小杯子都放进一台"自动混合机"(`_foreach_lerp_`),告诉它"目标比例 1:999",一次性完成混合。

每个 step 后 EMA 模型都在向当前模型方向"漂"一点点,但 99.9% 仍是过去的平均 —— 所以它对最新一步的噪声极其不敏感,这就是为什么用它来 eval / sampling 比直接用 student 稳很多。

You're maintaining a punch bowl: the bowl is the "average over the last 10 000 samples" (EMA) and a small cup contains "the freshest pour" (current). Two ways:
- Naïve: ladle-by-ladle, take 0.001 from the cup into the bowl, take 0.999 from the bowl back — repeat ten thousand times.
- Smart: drop both vessels into an automatic mixer (`_foreach_lerp_`), tell it "blend at ratio 1:999", and the whole pour is done in one shot.

After each step the EMA model drifts a tiny bit toward the current model, but 99.9% of it is still the historical average — which is exactly why it's the stable copy you eval or sample from.

## 自己跑一遍 / Try it yourself

```python
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 2))
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.99))

optim = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(32, 8), torch.randn(32, 2)
for step in range(20):
    loss = ((model(x) - y) ** 2).mean()
    optim.zero_grad(); loss.backward(); optim.step()
    ema.update_parameters(model)

with torch.no_grad():
    w_model = model[0].weight
    w_ema   = ema.module[0].weight
    print("model first weight:", w_model.flatten()[:3].tolist())
    print("ema   first weight:", w_ema.flatten()[:3].tolist())
    print("ratio |ema-init|/|model-init|:",
          (w_ema - w_ema.detach()).abs().mean().item(),
          (w_model - w_model.detach()).abs().mean().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
model first weight: [...]    <- drifted noticeably
ema   first weight: [...]    <- drifted, but ~100x less
```

中文重点:训练 20 步后,model 参数已经显著偏离初值,而 EMA 参数只挪动了 model 漂移量的约 1% —— 这正是 `1 - decay = 0.01` 的混合比例。

After 20 steps the model weights have drifted noticeably; the EMA weights have moved roughly 1% as far — exactly the `1 - decay = 0.01` blend ratio.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **diffusers `EMAModel`** / **HuggingFace `EMAModel`**: 几乎是同一个公式,但还多了 inverse-gamma 的"动态 decay"(早期参数还没稳定时 decay 自适应变小)。 / Same formula but adds inverse-gamma "dynamic decay" so decay shrinks adaptively while parameters are still fluctuating early on.
- **MoCo / BYOL 的 target encoder 更新** / **MoCo / BYOL target encoder update**: 整个对比 SSL 框架的"target network"就是 online network 的 EMA。 / The entire target network in contrastive SSL is just an EMA of the online network.
- **FSDP EMA shadow copy** / **FSDP EMA shadow copy**: 分布式训练里 EMA 副本通常和参数同 shard,`_foreach_lerp_` 在每个 rank 本地各自跑。 / In FSDP the EMA shadow is sharded the same way as the parameters, and `_foreach_lerp_` runs locally on each rank.
- **`torch._foreach_*` 家族还有 `_addcmul_`、`_lerp_`、`_div_`** / **The `torch._foreach_*` family**: AdamW、AdaFactor、Muon 的"多张量"代码路径都在用同样的范式 —— 把整个参数 list 压成一个 fused kernel。 / The AdamW, AdaFactor, and Muon multi-tensor codepaths all use the same pattern — collapse the parameter list into a single fused kernel.

## 注意事项 / Caveats / when it breaks

- **decay 不能在训练前期一直 0.999** / **don't keep decay at 0.999 from step 0**: 初值是随机的,前 1000 步直接用 0.999 会让 EMA 长时间停留在初始权重附近。常见做法是 `decay_t = min(decay, (1 + t) / (10 + t))`(diffusers 的 EMA)。 / Parameters start out random; a hard 0.999 will pin EMA near the initialization for thousands of steps. The common fix is `decay_t = min(decay, (1 + t) / (10 + t))` (used by diffusers).
- **buffer 默认不被 EMA** / **buffers default to non-averaged**: `AveragedModel(use_buffers=False)` 时 `BatchNorm` 的 running mean/var 不走 EMA,而是直接跟着 model 同步 —— 这通常更对,因为 BN 统计本身就有滑动均值。 / With `use_buffers=False`, `BatchNorm` running stats are copied verbatim from the model rather than EMA'd. Usually correct: BN already keeps its own moving stats.
- **`@torch.no_grad()` 不是装饰用** / **`@torch.no_grad()` is not cosmetic**: 漏掉它会让 EMA copy 进 autograd 图,反向时多一段无意义的计算,严重时显存爆掉。 / Forgetting it lets the EMA copy into the autograd graph; backward then traces meaningless extra computation and you may OOM.

## 延伸阅读 / Further reading

- PyTorch `swa_utils` source: <https://github.com/pytorch/pytorch/tree/main/torch/optim>
- "Polyak averaging" — the statistical origin of EMA in stochastic optimization
- "Improved Denoising Diffusion Probabilistic Models" — the paper that popularized using EMA for diffusion sampling
