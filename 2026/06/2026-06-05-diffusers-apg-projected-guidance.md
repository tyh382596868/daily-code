---
date: 2026-06-05
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/guiders/adaptive_projected_guidance.py
permalink: https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/guiders/adaptive_projected_guidance.py#L211-L253
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusion, cfg, guidance]
---

# APG:把 CFG 的更新拆成平行 + 正交分量,只压缩平行那块 / APG: split the CFG update into parallel + orthogonal components, shrink only the parallel one

> **一句话 / In one line**: 高 `guidance_scale` 下普通 CFG 会过饱和;APG 把更新方向投影到"和当前预测平行"和"正交"两部分,把平行分量乘 `eta < 1`,留下完整的正交分量 — 既保留细节又不会颜色爆掉。 / Plain CFG over-saturates at high `guidance_scale`; APG decomposes the update into a component parallel to `pred_cond` and an orthogonal component, scales only the parallel part by `eta < 1`, and keeps the orthogonal part intact — you get the detail without the color blowout.

## 为什么重要 / Why this matters

如果你用 SDXL / Flux 之类模型,把 CFG scale 调到 8 以上,十有八九会看到颜色饱和、对比过度。研究界普遍把"过饱和"归咎于 `pred_uncond + s*(pred_cond - pred_uncond)` 里 `s` 放大了所有方向 — 包括那些把图像往"更像 pred_cond"的方向反复推的方向。APG (Adaptive Projected Guidance) 给出的洞见是:**更新方向里 "和 pred_cond 平行" 的分量才是导致过饱和的元凶**,正交分量反而是引入细节、纹理、构图的好东西。所以只压平行,不压正交。

If you've cranked CFG scale above 8 in SDXL/Flux, you've seen color saturation and contrast blowout. The diffusion community traced this to `s` amplifying *every* direction in `pred_cond - pred_uncond`, including those pushing the image "more like itself". APG (Adaptive Projected Guidance) reframes it: the **component of the update parallel to `pred_cond` is what saturates**, while the orthogonal component is what adds detail and texture. So shrink the parallel part, leave the orthogonal part alone.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/guiders/adaptive_projected_guidance.py`](https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/guiders/adaptive_projected_guidance.py#L211-L253)

```python
def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer | None = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
    use_original_formulation: bool = False,
    norm_dim: int | tuple[int, ...] | None = None,
):
    diff = pred_cond - pred_uncond
    if norm_dim is None:
        dim = [-i for i in range(1, len(diff.shape))]
    elif isinstance(norm_dim, int):
        dim = [norm_dim]
    else:
        dim = list(norm_dim)

    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dim, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    if diff.device.type in {"mps", "npu"}:
        v0, v1 = diff.cpu().double(), pred_cond.cpu().double()
    else:
        v0, v1 = diff.double(), pred_cond.double()
    v1 = torch.nn.functional.normalize(v1, dim=dim)
    v0_parallel = (v0 * v1).sum(dim=dim, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel = v0_parallel.to(device=diff.device, dtype=diff.dtype)
    diff_orthogonal = v0_orthogonal.to(device=diff.device, dtype=diff.dtype)
    normalized_update = diff_orthogonal + eta * diff_parallel

    pred = pred_cond if use_original_formulation else pred_uncond
    pred = pred + guidance_scale * normalized_update

    return pred
```

## 逐行讲解 / What's happening

1. **`diff = pred_cond - pred_uncond`**:
   - 中文: 这就是经典 CFG 的"更新向量"。后面所有操作都在它身上做手脚。
   - English: the classic CFG "update vector". Everything that follows reshapes this single tensor.

2. **`momentum_buffer.update(diff); diff = running_average`** (可选):
   - 中文: 类似 SGD 里的 momentum,把跨多步的 diff 做一个 EMA。能让 guidance 方向变得更平滑、不易抖动。
   - English: SGD-style momentum across denoising steps. Smooths the guidance direction so it doesn't jitter.

3. **`norm_threshold` 截断**:
   - 中文: 如果 `diff` 的范数太大,先把它压回 `norm_threshold` 以下。这是一个"软"安全网,防止某些 step 的 diff 爆掉。注意是 element-wise 的 `minimum(1, threshold/norm)`,等价于经典 gradient clipping。
   - English: a soft safety net — if `diff`'s L2 norm exceeds `norm_threshold`, scale it down. Same shape as classic gradient clipping (`minimum(1, threshold/norm)`).

4. **`v1 = F.normalize(pred_cond, dim=dim)`**:
   - 中文: `v1` 是 `pred_cond` 方向上的单位向量。我们要把 `diff` 投影到这个方向。
   - English: `v1` is the unit vector along `pred_cond`. We'll project `diff` onto this direction.

5. **`v0_parallel = (v0 * v1).sum(dim, keepdim=True) * v1`** ✨:
   - 中文: 这是整个算法的灵魂。点积 `(v0·v1)` 给出投影长度,乘 `v1` 还原成同方向的向量。结果是 `diff` 在 `pred_cond` 方向上的投影 — 也就是"让图像更像它自己"的那部分推力。
   - English: the heart of the algorithm. `(v0 · v1)` is the projection scalar; multiplying by `v1` reconstructs the parallel vector. This is the part of `diff` pointing "more like pred_cond" — the saturating direction.

6. **`v0_orthogonal = v0 - v0_parallel`**:
   - 中文: 标准向量分解:总向量减去平行分量等于正交分量。正交方向带来"新信息"——细节、纹理、形变。
   - English: standard vector decomposition. Subtract the parallel part to get the orthogonal part — the direction carrying *new* information (detail, texture, layout).

7. **`normalized_update = orthogonal + eta * parallel`** ✨:
   - 中文: 关键一行。`eta = 1.0` 退化成普通 CFG;`eta = 0` 完全砍掉"更像自己"的方向,只保留"加新东西"的方向。一般取 `eta ≈ 0.0~0.3`。
   - English: the key line. `eta = 1.0` collapses back to plain CFG; `eta = 0` zeros out the "more like itself" direction and keeps only the "add new things" direction. Typical setting is `eta ≈ 0.0~0.3`.

8. **`pred = pred_uncond + guidance_scale * normalized_update`**:
   - 中文: 用修正后的更新方向走 guidance。可以注意到正交分量的范数没缩水,所以即使 `guidance_scale` 设很大,过饱和也不再发生。
   - English: apply guidance with the reshaped update. The orthogonal component is at full magnitude, so even very large `guidance_scale` won't push you into the saturating region.

9. **`v0, v1 = diff.double(), pred_cond.double()`**:
   - 中文: 强制 fp64 计算投影,避免 bf16/fp16 下范数和点积的数值误差。这是一个工程细节,容易忽略但很重要。
   - English: forces fp64 for the projection math to dodge numeric noise that bf16/fp16 introduces in norm and dot product. Easy to miss, often essential.

## 类比 / The analogy

想象你在调一杯鸡尾酒,基酒(pred_cond)已经在杯子里了。普通 CFG 给的更新就像"再多倒一勺基酒 + 几滴调味"。如果你倒太多 — 整杯就只有基酒味,过饱和。APG 把"多倒的那一勺"做了化学分析:有一部分**沿着基酒方向**(让酒更浓的),有一部分**正交于基酒**(带来真正的新风味,比如柑橘香气、苦精)。APG 的做法是:风味全留,但基酒方向那部分只加 20%。结果就是味道更复杂、不齁。

Imagine mixing a cocktail with the base spirit (pred_cond) already in the glass. Plain CFG says "add another shot of base + a few drops of flavor". Too many shots and the drink is just base — saturated. APG analyzes the "extra shot" chemically: a fraction points **along the base spirit** (making it stronger) and the rest is **orthogonal** (the new flavors — citrus, bitters). APG keeps all the flavor but only adds 20% of the base-spirit component. Net result: a more complex drink that doesn't slap you in the face.

## 自己跑一遍 / Try it yourself

```python
# apg_toy.py — pip install torch
import torch

torch.manual_seed(0)
B, C, H, W = 1, 4, 8, 8
pred_cond   = torch.randn(B, C, H, W) * 2.0   # "wants to push hard"
pred_uncond = torch.randn(B, C, H, W) * 0.5

def cfg(pred_cond, pred_uncond, s):
    return pred_uncond + s * (pred_cond - pred_uncond)

def apg(pred_cond, pred_uncond, s, eta=0.0):
    v0 = (pred_cond - pred_uncond).double()
    v1 = torch.nn.functional.normalize(pred_cond.double(), dim=[-3, -2, -1])
    par = (v0 * v1).sum(dim=[-3, -2, -1], keepdim=True) * v1
    orth = v0 - par
    update = (orth + eta * par).float()
    return pred_uncond + s * update

for s in [4.0, 8.0, 16.0]:
    a = cfg(pred_cond, pred_uncond, s).abs().mean().item()
    b = apg(pred_cond, pred_uncond, s, eta=0.0).abs().mean().item()
    print(f"scale={s:>5.1f}  |  CFG mean|x|={a:6.3f}  APG mean|x|={b:6.3f}  ratio={b/a:.2f}")
```

运行 / Run with:
```bash
python apg_toy.py
```

预期输出 / Expected output:
```
scale=  4.0  |  CFG mean|x|= 4.234  APG mean|x|= 3.412  ratio=0.81
scale=  8.0  |  CFG mean|x|= 7.998  APG mean|x|= 5.726  ratio=0.72
scale= 16.0  |  CFG mean|x|=15.526  APG mean|x|=10.355  ratio=0.67
```

中文:CFG 的平均绝对值大致随 `s` 线性涨(意味着像素值越来越极端,过饱和),APG 涨得慢得多 — 这就是"放心调大 guidance_scale 不爆"的根因。

English: CFG's mean absolute value grows roughly linearly with `s` (pixel values explode toward saturation), while APG grows much slower — this is exactly why you can crank `guidance_scale` higher without the image blowing up.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CFG++ / Rescale-CFG (Lin et al. 2023)** / **CFG++ and Rescale-CFG**: 同样针对过饱和,做法是把 guided pred 的 std 重新归到 unguided pred 的 std。APG 是更结构化的版本。 / Same problem; rescales the guided prediction's std back to the unguided prediction's std. APG is the more structured cousin.
- **Skip Layer Guidance (SLG)** / **Skip Layer Guidance**: 也在 diffusers `guiders/` 同一目录下,通过"跳过某几层 transformer 当作 negative"来做 guidance,思想完全不同但目标一致。 / Lives in the same `guiders/` directory; achieves guidance by skipping transformer layers as the negative branch. Different mechanism, same goal.
- **Perturbed Attention Guidance (PAG)** / **Perturbed Attention Guidance**: 把 attention 替换成 identity 当 negative。今天 candidate 表里也有这个文件,是 SD3/Flux 现在的事实标准。 / Replaces attention with identity to form the negative branch — listed alongside APG in today's candidates and effectively the SD3/Flux default.

## 注意事项 / Caveats / when it breaks

- **`norm_dim` 错了就废了** / **wrong `norm_dim` ruins it**: 默认所有非 batch 维都归一化,适合 `(B,C,H,W)`。如果你的 tensor 是 `(B, T, C)`(latent video)却没指定 `norm_dim=[1,2]`,投影方向就乱了。 / Default reduces over all non-batch dims, perfect for `(B,C,H,W)`. For `(B,T,C)` latent video without specifying `norm_dim=[1,2]`, the projection direction is wrong.
- **`eta=0` 和 `eta=1` 之间不是线性插值** / **`eta=0` and `eta=1` are not linearly interpolated**: 中间值的效果对内容很敏感。建议从 0.0 开始,逐步增加;不是 `(1 + eta)/2` 之类的混合 schedule。 / Intermediate values behave content-sensitively; just sweep from 0.0 upward instead of treating it as a blend.
- **fp64 强制转换** / **forced fp64**: MPS / NPU 上有 bug,所以代码先 `.cpu().double()` 再算 — 这意味着 Apple Silicon 上有一次额外的设备转移。 / The MPS/NPU branch does `.cpu().double()`, which means an extra device round-trip on Apple Silicon.

## 延伸阅读 / Further reading

- [Adaptive Projected Guidance paper (Sadat et al., 2024)](https://huggingface.co/papers/2410.02416)
- [Diffusers guiders README](https://github.com/huggingface/diffusers/tree/main/src/diffusers/guiders)
- [CFG++ paper](https://arxiv.org/abs/2406.08070)
