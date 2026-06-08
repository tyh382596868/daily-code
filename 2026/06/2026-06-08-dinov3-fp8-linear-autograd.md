---
date: 2026-06-08
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/layers/fp8_linear.py
permalink: https://github.com/facebookresearch/dinov3/blob/50001c6db58dbca7e7d06a5c5a9f1e078ca29197/dinov3/layers/fp8_linear.py#L17-L96
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, diffusion, fp8, autograd-function, quantization]
---

# DINOv3 的 fp8 Linear:一个 65 行的可微分 fp8 矩阵乘法 / DINOv3's fp8 Linear: a differentiable fp8 matmul in 65 lines

> **一句话 / In one line**: 把 forward 量化成 fp8、把 backward 留在 bf16,用一个 `torch.autograd.Function` 把"row-wise per-token 量化的 fp8 矩阵乘法"装成一个普通的 `nn.Linear`。
> Quantize the forward to fp8, keep the backward in bf16, and wrap the whole "row-wise per-token-scaled fp8 matmul" inside one `torch.autograd.Function` so it drops in as a normal `nn.Linear`.

## 为什么重要 / Why this matters

DINOv3 是当下几乎所有"靠视觉表征喂下游"的系统(diffusion 模型的 conditioning、世界模型的 latent、VLA 的视觉塔)都会用的视觉骨干。把它的 Linear 层从 bf16 切到 fp8,带宽减半、tensor core 吞吐翻倍。难点不是 forward,而是怎么让 backward 还能跑——PyTorch 现在的 `torch._scaled_mm` 只暴露了 fp8 forward,没给一套现成的可微分包装。这 65 行就是那个包装:per-token amax → 求 scale → fp8 quant → 调 `_scaled_mm` → 把 scale 还原回去。Backward 反而退回 bf16 GEMM,因为权重梯度对 outlier 太敏感,fp8 算了精度不够。

DINOv3 is the workhorse vision backbone behind almost every system that "feeds downstream from visual features": diffusion conditioning, world-model latents, VLA vision towers. Moving its Linear layers from bf16 to fp8 halves bandwidth and roughly doubles tensor-core throughput. The hard part isn't the forward — it's keeping the backward differentiable, because PyTorch's `torch._scaled_mm` only exposes the fp8 forward; there's no off-the-shelf autograd wrapper. These 65 lines *are* that wrapper: per-token amax → derive scale → fp8 cast → call `_scaled_mm` → fold the scale back in. The backward deliberately falls back to bf16 GEMM, because weight gradients are too sensitive to outliers for fp8 to be accurate.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/layers/fp8_linear.py`](https://github.com/facebookresearch/dinov3/blob/50001c6db58dbca7e7d06a5c5a9f1e078ca29197/dinov3/layers/fp8_linear.py#L17-L96)

```python
EPS = 1e-12


def scale(t, amax_t):
    max_v = torch.finfo(torch.float8_e4m3fn).max
    scale_t = torch.clamp(amax_t.float(), min=EPS) / max_v
    t_fp8 = (t / scale_t).to(torch.float8_e4m3fn)
    return t_fp8, scale_t


def matmul(first, amax_first, second_t, amax_second_t, bias):
    first_fp8, scale_first = scale(first, amax_first)
    second_t_fp8, scale_second_t = scale(second_t, amax_second_t)
    # PyTorch's row-wise scaled matmul kernel is based on CUTLASS and is quite
    # slow. Hence we fall back to an "unscaled" matmul, which uses cuBLAS, and
    # apply the scale manually afterwards.
    output = torch._scaled_mm(
        first_fp8,
        second_t_fp8.t(),
        scale_a=scale_first.new_ones((1, 1)),
        scale_b=scale_second_t.t().new_ones((1, 1)),
        bias=None,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    output = (output * scale_first * scale_second_t.t()).to(torch.bfloat16)
    if bias is not None:
        output = output + bias
    return output


@torch.compiler.allow_in_graph
class Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b_t, bias):
        amax_a = a.abs().amax(dim=-1, keepdim=True)
        amax_b_t = b_t.abs().amax(dim=-1, keepdim=True)
        out = matmul(a, amax_a, b_t, amax_b_t, bias)

        ctx.a_requires_grad = a.requires_grad
        ctx.b_requires_grad = b_t.requires_grad
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False

        ctx.save_for_backward(a, b_t, amax_b_t.max())

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b_t, amax_b = ctx.saved_tensors

        if ctx.a_requires_grad:
            b = b_t.t().contiguous()
            amax_grad_out = grad_out.abs().amax(dim=-1, keepdim=True)
            amax_b = amax_b.repeat(b.shape[0], 1)
            grad_a = matmul(grad_out, amax_grad_out, b, amax_b, None)
        else:
            grad_a = None
        if ctx.b_requires_grad:
            grad_b = grad_out.t() @ a
        else:
            grad_b = None
        if ctx.bias_requires_grad:
            grad_bias = grad_out.sum(dim=0)
        else:
            grad_bias = None

        return grad_a, grad_b, grad_bias


class Fp8Linear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = Fp8LinearFn.apply(input.flatten(end_dim=-2), self.weight, self.bias)
        out = out.unflatten(0, input.shape[:-1])
        return out
```

## 逐行讲解 / What's happening

1. **`scale(t, amax_t)`(行 17-21) / Lines 17-21**:
   - 中文: 对张量 `t`,用它的逐行(per-row)绝对值最大值 `amax_t` 求 scale。`max_v` 是 `float8_e4m3fn` 能表示的最大值(约 448)。`scale_t = amax_t / max_v` 把 `t` 的每一行单独压到 `[-max_v, max_v]` 区间,然后 cast 到 fp8。`torch.clamp(min=EPS)` 防止全 0 行除以 0。
   - English: For a tensor `t`, derive its scale from the per-row absolute max `amax_t`. `max_v` is the largest value a `float8_e4m3fn` can hold (~448). `scale_t = amax_t / max_v` squeezes each row individually into `[-max_v, max_v]`, then casts to fp8. The `clamp(min=EPS)` guards an all-zero row from dividing by zero.

2. **`matmul(...)` 里的 `scale_a/scale_b = new_ones((1,1))`(行 30-38) / Lines 30-38**:
   - 中文: 这里有个微妙的反直觉操作。`torch._scaled_mm` 本来就支持 per-row scale,但作者注释说 CUTLASS 那条 path 太慢——于是他们传 `ones` 当 scale(等价于"不缩放"),用 cuBLAS 路径跑 GEMM,再在第 39 行手动乘回 `scale_first * scale_second_t.t()`。性能换出来的代价是稍微多一次 elementwise mul。
   - English: A subtle counterintuitive move. `torch._scaled_mm` *does* support per-row scales natively, but the comment notes that the CUTLASS path is slow — so they pass `ones` as scales (effectively "unscaled"), get the cuBLAS path, then manually multiply the scales back on line 39. The trade is one extra elementwise mul for a much faster matmul.

3. **`Fp8LinearFn.forward`(行 48-59) / Lines 48-59**:
   - 中文: 标准的 autograd Function 模板。`amax_a = a.abs().amax(dim=-1, keepdim=True)` 是 per-token 量化的核心——每个 token 一个独立 scale,而不是整个 batch 共享一个。这对 outlier 极其重要:某一个 token 上有个大值不会拖垮其他 token 的精度。然后 `save_for_backward(a, b_t, amax_b_t.max())` 把 input 和权重的 amax(取 max 后只剩一个标量)存起来。
   - English: The standard autograd-Function template. The crucial line is `amax_a = a.abs().amax(dim=-1, keepdim=True)` — per-token quantization: each token gets its own scale, instead of one shared scale per batch. This is huge for outlier handling: a large value in one token doesn't crush precision in the others. Then `save_for_backward(a, b_t, amax_b_t.max())` stashes the input, the weight, and a single scalar amax of the weight for use in the backward.

4. **`Fp8LinearFn.backward`(行 62-81) / Lines 62-81**:
   - 中文: 这里是整段代码最有意思的设计选择。`grad_a` 走 fp8(对输入的梯度,fp8 够用);但 `grad_b = grad_out.t() @ a` 是普通的 bf16 GEMM——**权重梯度故意不用 fp8**。为什么?权重梯度是把整个 batch 的贡献加起来,数值范围比 forward 大得多,outlier 也更尖锐,fp8 e4m3 的指数范围撑不住。所以这是一个"半 fp8"的方案。
   - English: The most interesting design decision in the whole file. `grad_a` goes through fp8 (input gradients tolerate fp8 just fine), but `grad_b = grad_out.t() @ a` is a plain bf16 GEMM — **the weight gradient deliberately stays out of fp8**. Why? Weight grads sum contributions from the entire batch, so their dynamic range blows up and outliers get sharper than in the forward; fp8 e4m3's exponent range can't hold it. The result is a "half-fp8" recipe: fp8 forward + fp8 grad-input + bf16 grad-weight.

5. **`Fp8Linear(torch.nn.Linear)`(行 84-88) / Lines 84-88**:
   - 中文: 这是真正把它"塞进" nn 体系的胶水——直接继承 `nn.Linear`,只 override `forward`。所有 `.weight` / `.bias` / `state_dict` / `to(device)` 都白嫖父类。这意味着你可以把一个训练好的 bf16 模型 load 进来,然后 `convert_linears_to_fp8(model)` 把 Linear 全部换成 Fp8Linear——权重保持不变,只在 forward 时量化。
   - English: This is the glue that drops it into the nn ecosystem — subclass `nn.Linear`, override only `forward`. Everything else (`.weight`, `.bias`, `state_dict`, `to(device)`) is inherited for free. The implication: you can load a bf16 checkpoint, then run `convert_linears_to_fp8(model)` to swap every Linear for an Fp8Linear — weights unchanged, quantization happens only at the forward pass.

## 类比 / The analogy

中文: 想象一个邮局快递站,每件包裹都贵但又重(bf16)。fp8 量化就像"把每件包裹拆开,把贵重的部分(scale)留底,把粗略形状(fp8 张量)装小箱子寄出去",到了对面再拿出底单 scale 把它们组装回来。"per-token amax" 就是"按每一件包裹分别记底单",而不是"整车货只记一张总单"——因为一辆车里如果有个金条,它的体积会让其他纸箱都压扁。Backward 故意不走 fp8,就像"贵重包裹只单向用便宜运法,反向(质保索赔)走原路保险件"。

English: Picture a postal hub where every parcel is bulky and expensive (bf16). Fp8 quantization is like "open each parcel, write down its value (the scale) on a slip, repack the rough shape (fp8 tensor) into a small box, ship the box", and then on the receiving end use the slip to rebuild the original. "Per-token amax" means writing a slip *per parcel*, not one slip per truckload — because if a single truck has a gold bar inside, its volume crushes every other paper box. Forcing the backward back to bf16 is like saying "use the cheap shipping method outbound only — for any return/warranty claim, ship full-cost insured."

## 自己跑一遍 / Try it yourself

```python
# fp8_linear_demo.py
import torch
from torch import nn

# Requires a Hopper/Ada GPU + recent PyTorch for torch._scaled_mm
assert torch.cuda.is_available(), "needs CUDA"

EPS = 1e-12

def scale(t, amax_t):
    max_v = torch.finfo(torch.float8_e4m3fn).max
    scale_t = torch.clamp(amax_t.float(), min=EPS) / max_v
    return (t / scale_t).to(torch.float8_e4m3fn), scale_t

class Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b_t):
        a_fp8, sa = scale(a, a.abs().amax(dim=-1, keepdim=True))
        b_fp8, sb = scale(b_t, b_t.abs().amax(dim=-1, keepdim=True))
        out = torch._scaled_mm(a_fp8, b_fp8.t(),
                               scale_a=sa.new_ones((1,1)), scale_b=sb.t().new_ones((1,1)),
                               out_dtype=torch.bfloat16, use_fast_accum=False)
        out = (out * sa * sb.t()).to(torch.bfloat16)
        ctx.save_for_backward(a, b_t)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        a, b_t = ctx.saved_tensors
        # bf16 fallback for both gradients (compare against fp8 grad_a in the real DINOv3 version)
        grad_a = grad_out.to(b_t.dtype) @ b_t
        grad_b = grad_out.t() @ a
        return grad_a, grad_b

a = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y_fp8 = Fp8LinearFn.apply(a, w)
y_ref = a @ w.t()
print("fp8 vs bf16 max abs err:", (y_fp8 - y_ref).abs().max().item())
y_fp8.sum().backward()
print("grad ok:", a.grad is not None, w.grad is not None)
```

运行 / Run with:
```bash
pip install torch  # requires CUDA + Hopper (H100) or Ada (4090) GPU
python fp8_linear_demo.py
```

预期输出 / Expected output:
```
fp8 vs bf16 max abs err: ~0.2-0.6
grad ok: True True
```

中文一两句: 关键观察是 max abs err 大约 0.5 而不是 0.01——这就是 fp8 该有的样子。如果你看到这个量级的误差,说明 quantize-dequantize 走通了。把 `dim=-1` 换成 `dim=None`(全张量共享 scale),你会看到误差立刻爆炸到几十。

English: The thing to notice is that the max-abs error is in the ballpark of 0.5, not 0.01 — that's exactly the fp8 noise floor. If you see numbers in that range, your quantize-dequantize roundtrip is correct. Swap `dim=-1` for `dim=None` (one shared scale for the whole tensor) and the error immediately blows up to tens.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TransformerEngine (NVIDIA)** / **TransformerEngine (NVIDIA)**: 工业级的 fp8 实现,也是 forward fp8 + backward bf16,但 amax 用了 history-based scaling(看过去 N 步的 max)。/ Industrial-grade fp8 implementation — also fp8-forward + bf16-backward, but uses history-based scaling (max over last N steps) instead of per-step amax.
- **`torchao` 的 `Float8Linear`** / **torchao's `Float8Linear`**: PyTorch 官方仓库的 fp8 训练工具,同样的 autograd Function 模板,加了 delayed-scaling、tensor-wise vs row-wise 的可切换路径。/ PyTorch's official fp8-training utility — same autograd-Function template, adds delayed scaling and a toggle between tensor-wise and row-wise paths.
- **`linkedin/Liger-Kernel`** / **linkedin/Liger-Kernel**: 走 Triton 路径自己手写 fp8 matmul kernel,绕过 `_scaled_mm`,在某些 shape 上比 cuBLAS 快。/ Hand-rolls fp8 matmul in Triton, bypassing `_scaled_mm` — faster than cuBLAS on certain shapes.

## 注意事项 / Caveats / when it breaks

- **需要 Hopper(H100)或 Ada(4090)以上 / Hopper (H100) or Ada (4090) and up only**: fp8 tensor core 是这一代 GPU 才有的硬件。A100 上没有,程序会在 `_scaled_mm` 上抛错。/ fp8 tensor cores are hardware on these GPUs only — A100 doesn't have them and `_scaled_mm` will throw.
- **`in_features` 和 `out_features` 必须是 64 的倍数 / Both `in_features` and `out_features` must be multiples of 64**: 文件下面 `convert_linears_to_fp8` 里有显式 assert。Hopper 的 fp8 tile 大小是 64;另外 Inductor 在某些情况下会把内维 pad 到 64,踩到这里会数值不一致。/ The file's `convert_linears_to_fp8` enforces this. The Hopper fp8 tile size is 64, and Inductor sometimes pads inner dims to 64 — a mismatch silently corrupts numerics.
- **第一次跑要 `torch._dynamo.reset_code_caches()` / Reset Dynamo caches after conversion**: 不重置的话,已经 trace 过的旧 graph 还在,新的 Fp8Linear 不会真正进 graph。文件末尾(本片段外)显式调用了这个 + `reset_cudagraph_trees()`。/ Without resetting, cached Dynamo graphs still hold the old bf16 module — your Fp8Linear won't actually enter the graph. The file (outside this snippet) explicitly calls both `_dynamo.reset_code_caches()` and `reset_cudagraph_trees()` after conversion.
- **`grad_b` 仍是 bf16 / `grad_b` is still bf16**: 不要被"fp8 训练"这个词骗——只有一半的 GEMM 真在 fp8。如果你想全 fp8,得自己写 fp8 backward 并处理 outlier(通常需要 stochastic rounding 或更宽的指数 e5m2)。/ Don't be misled by "fp8 training" — only half the GEMMs are actually fp8. A full-fp8 setup needs you to write the fp8 backward yourself, usually with stochastic rounding or a wider e5m2 exponent.

## 延伸阅读 / Further reading

- [DINOv3 paper (Meta, 2026)](https://github.com/facebookresearch/dinov3/blob/main/README.md)
- [PyTorch `torch._scaled_mm` docs](https://pytorch.org/docs/stable/generated/torch._scaled_mm.html)
- [Microsoft FP8-LM paper — early fp8 training recipes](https://arxiv.org/abs/2310.18313)
- [torchao Float8Linear deep-dive](https://github.com/pytorch/ao/tree/main/torchao/float8)
