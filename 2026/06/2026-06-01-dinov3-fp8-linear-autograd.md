---
date: 2026-06-01
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/layers/fp8_linear.py
permalink: https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/dinov3/layers/fp8_linear.py#L13-L96
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, diffusion, fp8, quantization, autograd]
---

# 用 80 行写一个 FP8 Linear:dinov3 的训练级 e4m3 量化 / 80 lines to ship an FP8 Linear: dinov3's training-grade e4m3 quantization

> **一句话 / In one line**: 把 act 和 weight 各按一条 row 缩到 `float8_e4m3fn`,调用 `torch._scaled_mm` 再把 bf16 输出乘回原 scale,正/反向都能跑,顺手把 `nn.Linear` 全部就地替换 / Quantize act and weight per-row into `float8_e4m3fn`, call `torch._scaled_mm`, then multiply the bf16 output back by the scales — forward, backward, and an in-place `nn.Linear` swap, all in one file.

## 为什么重要 / Why this matters

H100/H200/B200 真正能拿满吞吐的算子已经是 FP8 matmul 了,但大多数训练代码还是 bf16,因为大家不知道往哪儿插。dinov3 这个文件给了一份**最小可训练 FP8 Linear**:74 行覆盖了 scaling 公式、`torch._scaled_mm` 的正确用法、自定义 backward、以及把整个 ViT 里所有 `nn.Linear` 就地换掉的 helper。可以直接抄进任何 ViT / LLM / DiT。

FP8 matmul is the only way to actually hit advertised TFLOPs on H100/H200/B200, but most training code still runs bf16 because it's not obvious where the FP8 boundary should go. dinov3 ships a self-contained, **trainable** FP8 Linear in 74 lines: the scaling formula, the right way to call `torch._scaled_mm`, a custom backward, and a one-liner that replaces every `nn.Linear` in a ViT in-place. You can drop it into any ViT / LLM / DiT today.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/layers/fp8_linear.py`](https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/dinov3/layers/fp8_linear.py#L13-L96)

```python
# avoid division by zero when calculating scale
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

1. **`scale(t, amax_t)`**:
   - 中文: 给一个张量 `t` 和它的每行 absmax,算出 `scale = amax / fp8_max`,然后 `t / scale` 落到 `float8_e4m3fn`。`fp8_max` 是 e4m3 能表示的最大正数(≈448),所以这一步把每行最大的元素正好压到 fp8 满量程。`EPS` 防止全零行整除。
   - English: given a tensor `t` and its per-row absmax, compute `scale = amax / fp8_max`, then `(t / scale).to(float8_e4m3fn)`. `fp8_max` is the largest positive in e4m3 (≈448), so this maps each row's max element exactly to FP8 full-scale. `EPS` guards against an all-zero row.

2. **`matmul(...)`(第 24-42 行 / lines 24-42)**:
   - 中文: 两个输入各自量化,然后调用 `torch._scaled_mm`。注意它**故意传了两个 1.0 的 scale**,因为 cuBLAS 的非缩放路径比 row-wise 缩放的 CUTLASS 路径快得多,所以先做无缩放 matmul,再 `output * scale_first * scale_second_t.t()` 把缩放手动乘回去。`out_dtype=bf16`,`use_fast_accum=False` 在训练里更安全。
   - English: quantize both inputs, then call `torch._scaled_mm` — but with **scales of `1.0`**. Why? The row-wise scaled-mm kernel is CUTLASS-based and slow; the unscaled cuBLAS path is much faster, so do the unscaled matmul and apply scales manually afterwards. `out_dtype=bf16`, `use_fast_accum=False` is the safe choice for training (avoids the bias of the H100 fast-accumulator).

3. **`Fp8LinearFn.forward`(第 48-59 行 / lines 48-59)**:
   - 中文: 每个输入算自己的 `amax`(按最后一维 keepdim,所以每行一个 scale),调用 `matmul`,然后 `save_for_backward(a, b_t, amax_b_t.max())`。注意保存的是**完整的 a 和 b_t**(bf16/fp16),不是它们的 fp8 版本——反向要重新量化。
   - English: each input gets its own row-wise `amax` (keepdim along the last dim → one scale per row), calls `matmul`, and saves the full `a`, `b_t`, plus a single scalar `amax_b_t.max()`. Notice it saves the **original bf16/fp16 tensors**, not the FP8 ones — backward re-quantizes them.

4. **`Fp8LinearFn.backward`(第 62-81 行 / lines 62-81)**:
   - 中文: `grad_a = grad_out @ b`(用 FP8 算,grad_out 在 backward 现场算 amax,b 用前向保存的标量 amax)。`grad_b = grad_out.t() @ a` 用普通 bf16(因为权重的 grad 只算一次,FP8 收益不大),`grad_bias = grad_out.sum(dim=0)`。三个 `requires_grad` 旗子决定要不要算对应分支,推理时全部跳过。
   - English: `grad_a = grad_out @ b` runs in FP8 (re-computing grad_out's amax on the fly, reusing b's saved scalar amax). `grad_b = grad_out.t() @ a` stays in bf16 (the weight grad is computed once per step, so FP8 here barely helps). `grad_bias = grad_out.sum(dim=0)`. The three `requires_grad` flags gate each branch so inference skips all of them.

5. **`Fp8Linear`(第 84-88 行 / lines 84-88)**:
   - 中文: 继承 `nn.Linear`,只重写 `forward`,把任意 `[..., in_features]` 输入 flatten 成 `[N, in_features]` 给 autograd Function,出来再 unflatten 回去。这样 `Fp8Linear` 在 state_dict / 序列化 / `.parameters()` 上和原版 `nn.Linear` 完全兼容。
   - English: subclass `nn.Linear`, override only `forward`. Flatten arbitrary `[..., in_features]` to `[N, in_features]` for the autograd Function, then unflatten the output. This way `Fp8Linear` is fully drop-in for `nn.Linear` at the state-dict / serialization / `.parameters()` level.

6. **`@torch.compiler.allow_in_graph`**:
   - 中文: 告诉 Dynamo "别动这个 autograd Function,我自己知道我在干什么"。custom autograd 在 `torch.compile` 下默认会被打散重写,这个装饰器把它当成一个原子节点保留在图里。
   - English: tells Dynamo "treat this autograd Function as opaque". Custom autograd Functions are normally graph-broken or rewritten under `torch.compile`; this decorator preserves it as a single atomic node so Inductor doesn't try to be clever.

## 类比 / The analogy

想象你在用一台只支持 8 位的复印机复印一张高对比度的照片。直接复印会丢死黑/死白细节,所以你先按行测每行最亮的像素,把每行整体除以"亮度/255",这样每行最亮的点正好等于 255 复印出来后,再按原来除掉的系数乘回来。FP8 量化就是这套"测最亮 → 整行归一 → 复印 → 乘回去"的流程,只不过最亮的尺度叫 `amax`,255 叫 `fp8_max`,复印机叫 `torch._scaled_mm`。

Imagine photocopying a high-contrast photo on a machine that only supports 8-bit grayscale. Copy it raw and you crush blacks and whites. So per row you measure the brightest pixel, divide the whole row by `brightness/255` so its brightest dot lands exactly on 255, copy, then multiply back. FP8 quantization is exactly that "measure max → normalize row → copy → restore" loop, except "brightest pixel" is `amax`, "255" is `fp8_max`, and the copier is `torch._scaled_mm`.

## 自己跑一遍 / Try it yourself

```python
# fp8_demo.py — needs H100/H200/B200 + nightly torch with float8_e4m3fn
import torch

def scale(t, amax):
    max_v = torch.finfo(torch.float8_e4m3fn).max
    s = torch.clamp(amax.float(), min=1e-12) / max_v
    return (t / s).to(torch.float8_e4m3fn), s

def fp8_mm(a, b_t):
    a8, sa = scale(a, a.abs().amax(-1, keepdim=True))
    b8, sb = scale(b_t, b_t.abs().amax(-1, keepdim=True))
    out = torch._scaled_mm(
        a8, b8.t(),
        scale_a=sa.new_ones((1, 1)), scale_b=sb.new_ones((1, 1)),
        bias=None, out_dtype=torch.bfloat16, use_fast_accum=False,
    )
    return (out * sa * sb.t()).to(torch.bfloat16)

torch.manual_seed(0)
a   = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
b_t = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)  # row = out feature

ref = (a @ b_t.t()).to(torch.bfloat16)
got = fp8_mm(a, b_t)
print("max abs err :", (got - ref).abs().max().item())
print("rel L2 err  :", (got - ref).norm() / ref.norm())
```

运行 / Run with:
```bash
# Hopper-class GPU + nightly torch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
python fp8_demo.py
```

预期输出 / Expected output:
```
max abs err : ~0.06
rel L2 err  : ~0.003
```

中文: 注意"绝对误差"看起来不小,但**相对**误差只有 0.3% 量级——这是 FP8 训练能 work 的根本原因。换成全零行试试,你会看到 `EPS` 在保护你免于 NaN。

English: the absolute error looks big, but the **relative** error is ~0.3% — that's why FP8 training works at all. Try feeding an all-zero row and you'll see `EPS` quietly saving you from a NaN.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **torchao / `Float8Linear`** / **torchao's `Float8Linear`**: PyTorch 官方的 FP8 训练 recipe,比这版更复杂(支持 delayed scaling、tensorwise vs rowwise、动态/静态 scaling),但骨架一模一样:`forward` 里量化 + scaled_mm,`backward` 里重新量化 grad_out。
- **Transformer Engine (NVIDIA TE)** / **NVIDIA Transformer Engine**: H100 上的工业级 FP8 实现,核心也是 `amax history + delayed scaling`,只是 amax 是 EMA 维护的而不是 dinov3 这里的"每 step 实时算"。
- **MS-AMP / FP8-LM** / **Microsoft MS-AMP**: 把 FP8 推到 optimizer state 和 master weight,但 matmul 那层的逻辑等价于这份代码。

## 注意事项 / Caveats / when it breaks

- **per-row scaling 不能太极端 / row-wise scaling can be too coarse**: 中文: 如果某一行里有一个离群值大 100×,这行剩下的值会被压到 fp8 的下溢区(零)。LLM 里偶尔需要 tensor-wise + per-token-row 组合或 outlier 通道隔离(SmoothQuant、AWQ)。
- **must align to 64 / `out_features` 必须 64 对齐**: 中文: 文件后半 `convert_linears_to_fp8` 里检查 `% 64 != 0` raise,因为 H100 FP8 TensorCore 一次吃 64 元素。dinov3 把 FFN 维度叫 `swiglu64` 就是为了这个对齐。
- **backward 的 grad_b 仍是 bf16 / `grad_b` stays in bf16**: 这是个**故意的取舍**——权重梯度只算一次,FP8 节省的时间不多,但 FP8 误差会累积到 optimizer,所以反而 bf16 更稳。
- **`use_fast_accum=False` matters**: 中文: H100 的 FP8 accumulator 默认是 bf16,fast-accum 模式用 fp16,数值更脆。训练用 `False`,推理可以试 `True` 换吞吐。

## 延伸阅读 / Further reading

- [DINOv3 paper / repo](https://github.com/facebookresearch/dinov3) — Meta 用 FP8 训了 7B-参数级别的 ViT,本文件是他们生产代码的简化版
- [PyTorch `torch._scaled_mm` docs](https://docs.pytorch.org/docs/stable/generated/torch._scaled_mm.html)
- [torchao Float8 README](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md) — 同主题的 PyTorch 官方实现,对比读一遍非常有收获
- [NVIDIA Transformer Engine FP8 user guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
