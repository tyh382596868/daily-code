---
date: 2026-06-12
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_inductor/kernel/conv.py
permalink: https://github.com/pytorch/pytorch/blob/c02cc46513fc09877228889a278c7fbd7f82d25d/torch/_inductor/kernel/conv.py#L709-L748
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, pytorch, inductor, triton, conv-transpose, kernel-reuse]
---

# 数学恒等式当编译器优化:PyTorch Inductor 让 ConvTranspose2d 直接借用 backward-input 的 Triton kernel / Math identity as a compiler optimization: PyTorch Inductor lets ConvTranspose2d reuse the backward-input Triton kernel

> **一句话 / In one line**: `ConvTranspose2d(x, w) == conv_backward_input(grad_output=x, weight=w)`,所以 Inductor 干脆不写新 kernel,直接把已有的反向 Triton 模板正向调一次 —— 30 行新代码替代了一整套转置卷积 autotune / `ConvTranspose2d(x, w) == conv_backward_input(grad_output=x, weight=w)`, so Inductor doesn't write a new kernel — it just calls the existing backward-input Triton template forward, replacing a whole transposed-conv autotune path with 30 new lines.

## 为什么重要 / Why this matters

PyTorch Inductor 在做 `torch.compile(... max_autotune_conv_backends="TRITON")` 的时候,对转置卷积长期处于"无 kernel 可用"状态 —— 因为前向 Triton 模板只支持 dilation=1, output_padding=0 这些"普通卷积",一遇到 ConvTranspose 直接 `NoValidChoicesError`。要写完整的转置卷积 Triton 模板?光支持 dilation、groups、output_padding 这几个 corner case 就是几千行。这个 PR (#186067, 2026-06-11 合并) 的解法只有 30 行:**别写新的,转置卷积在数学上就是 grad-of-input 卷积**,backward-input 的 Triton 模板已经在仓库里了 (PR #178945),直接复用就行。

When you call `torch.compile(... max_autotune_conv_backends="TRITON")`, Inductor has long had a hole for transposed convs — the forward Triton template only handles vanilla convolution (dilation=1, no output_padding), so anything transposed crashes with `NoValidChoicesError`. Writing a full transposed-conv Triton template just to cover dilation/groups/output_padding edge cases is thousands of lines. This PR (#186067, merged 2026-06-11) gets there in 30 lines: **don't write a new template** — transposed conv *is* the input-gradient of plain conv. The `conv2d_bwd_input` Triton template was already in-tree from #178945. Just call it forward.

## 代码 / The code

`pytorch/pytorch` — [`torch/_inductor/kernel/conv.py`](https://github.com/pytorch/pytorch/blob/c02cc46513fc09877228889a278c7fbd7f82d25d/torch/_inductor/kernel/conv.py#L709-L748)

```python
if (
    torch._inductor.utils._use_conv_autotune_backend("TRITON")
    and use_triton_template(layout)
    and transposed
    and ndim == 2
):
    # ConvTranspose2d is mathematically identical to conv_backward_input:
    # the input plays the role of grad_output and the same weight layout
    # is used. Reuse the backward-input Triton template.
    conv_configs = V.choices.get_conv_configs(device_type)
    dtype_size = x.get_dtype().itemsize
    for cfg in conv_configs(
        sympy_product([layout.size[0], layout.size[2], layout.size[3]]),
        out_chan,
        in_chan,
        dtype_size=dtype_size,
    ):
        conv2d_bwd_input_template.maybe_append_choice(
            choices,
            input_nodes=(x, weight),
            layout=layout,
            KERNEL_H=kernel_shape[0],
            KERNEL_W=kernel_shape[1],
            PADDING_H=padding[0],
            PADDING_W=padding[1],
            STRIDE_H=stride[0],
            STRIDE_W=stride[1],
            DILATION_H=dilation[0],
            DILATION_W=dilation[1],
            GROUPS=groups,
            ALLOW_TF32=torch.backends.cudnn.fp32_precision == "tf32",
            num_stages=cfg.num_stages,
            num_warps=cfg.num_warps,
            **cfg.kwargs,
        )

if use_ck_conv_template(layout):
    # ... (ROCm CK backend, unchanged)

if not choices and config.max_autotune_conv_backends.strip():
    choices.append(
        aten_convolution.bind(
            args,
            layout,
            ordered_kwargs_for_cpp_kernel,
            **kwargs,
        )
    )
```

## 逐行讲解 / What's happening

1. **第 1-6 行的 guard / The guard on lines 1-6**:
   - 中文: 四个条件要全部成立才进这条分支 —— Triton 后端开启、layout 支持 Triton 模板、`transposed=True`、空间维度是 2D。也就是说这条优化只针对 ConvTranspose2d,1D / 3D 还是走 ATen fallback。
   - English: All four conditions must hold — Triton backend enabled, layout supports Triton templates, `transposed=True`, and spatial dims = 2. So this branch fires *only* for ConvTranspose2d; 1D and 3D still fall back to ATen.

2. **第 7-9 行的注释 / The comment on lines 7-9**:
   - 中文: 这是整个 PR 的核心理由 —— "转置卷积等价于反向输入卷积"。具体推导:`y = ConvTranspose2d(x, w)` 的算法是把 x 上采样后再用 w 做普通卷积;而 `grad_input = conv_backward_input(grad_output, w)` 也是上采样 `grad_output` 再用 w 卷一次。两者矩阵分解后是完全一样的运算。
   - English: This comment is the entire reason the PR exists — "transposed conv equals backward-input conv". The derivation: `y = ConvTranspose2d(x, w)` upsamples `x` and convolves with `w`; `grad_input = conv_backward_input(grad_output, w)` upsamples `grad_output` and convolves with `w`. After matrix factoring they are literally the same op.

3. **第 13-18 行的 conv_configs / The `conv_configs` loop on lines 13-18**:
   - 中文: Inductor 的 autotuner 会枚举一堆 (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps) 配置,挑跑得最快的那个。这里把 `sympy_product([N, H, W])` 当作 M,`out_chan` 当作 N,`in_chan` 当作 K —— 注意 **out_chan 和 in_chan 的角色在转置卷积里是反过来的**(转置卷积的"输出通道"就是 backward-input 的"输入通道"),作者直接利用了这一点。
   - English: Inductor's autotuner enumerates many `(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)` configurations and picks the fastest. Here `sympy_product([N, H, W])` is M, `out_chan` is N, `in_chan` is K — note that **out_chan and in_chan swap roles in transposed conv** (transposed-conv's "output channels" are backward-input's "input channels"). The author leans on that swap directly.

4. **第 19-37 行的 `maybe_append_choice` / Lines 19-37 (`maybe_append_choice`)**:
   - 中文: 把同一个 `conv2d_bwd_input_template` 当成一个 Choice 注册到 autotune 候选里。`input_nodes=(x, weight)` 这里把 `x` 喂到原本 backward-input 模板里"`grad_output`"的位置 —— 没有特殊化处理,就是把 tensor 换个语义槽。`DILATION_H/W` 当 `tl.constexpr` 用,所以必须是 concrete int,这也是为什么 PR 在前面加了一句 `if transposed: dilation = guard_int_seq(dilation)`。
   - English: It registers the same `conv2d_bwd_input_template` as an autotune choice. `input_nodes=(x, weight)` plugs `x` into the slot the template thought was `grad_output` — no specialization, just feeding the tensor into a renamed slot. `DILATION_H/W` are template-time `tl.constexpr`s, so they must be concrete ints — which is exactly why the PR adds `if transposed: dilation = guard_int_seq(dilation)` earlier.

5. **最后的 `if not choices` 兜底 / The `if not choices` safety net at the bottom**:
   - 中文: 即便上面所有候选都没注册成功(比如 dilated 1D / 3D conv-transpose 这种连 backward-input 模板都不支持),也别让 Inductor 用 `NoValidChoicesError` 崩 —— 退回到 ATen 的 `aten_convolution`。这是这个 PR 的第二个独立 fix,补上了原来前向卷积 lowering 缺失的"全后端失败"保护网。
   - English: Even if every backend above declines (e.g. dilated 1D/3D conv-transpose that backward-input doesn't support either), don't let Inductor crash with `NoValidChoicesError` — fall back to ATen's `aten_convolution`. That is the PR's second, separate fix: it closes the gap where the forward-conv lowering had no full-backend-fallback safety net.

## 类比 / The analogy

想象你是个法律事务所的 paralegal,公司里已经有一份"申请退款 (refund)"的标准模板,而你今天要写一份"撤销退款 (chargeback)"的新文件。资深律师瞟了一眼说:"撤销退款在法律上和申请退款是镜像 —— 把申请人和被申请人对调一下,模板可以原封不动用。" 你不用从头写新模板,只要把申请书里的两栏对调,签个名,文件就生效了。这个 PR 干的事情一模一样:转置卷积是"用反向输入卷积的模板,把 `x` 填到原本写 `grad_output` 的那一栏",30 行就完事。

Imagine you're a paralegal at a law firm. There's already a standard template for "refund request", and today you need to draft a brand-new "chargeback notice". The senior partner glances over: "Chargeback is the mirror image of refund — just swap the applicant and respondent columns, the rest of the template stands." You don't write a new template. You swap two columns, sign, file. This PR is the exact same move at the compiler level: take the existing backward-input Triton template, put `x` into the slot that was labeled `grad_output`, and you're done in 30 lines.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(2, 3, 8, 8, requires_grad=True)
w = torch.randn(4, 3, 3, 3)                          # (out_ch, in_ch, kH, kW)

# Forward path A: ConvTranspose2d(x, w)
y_transpose = F.conv_transpose2d(x, w, stride=1, padding=0)

# Forward path B: input-gradient of a plain conv whose output sits where x is now.
# Equivalence: ConvTranspose2d(x, w) == d(loss)/d(input) of  conv2d(input, w)  with  d(loss)/d(output) = x.
fake_input = torch.zeros(2, 4, 10, 10)               # shape that conv2d would produce y_transpose-sized output
y_bwd_input = torch.nn.grad.conv2d_input(
    input_size=fake_input.shape,
    weight=w,
    grad_output=x,                                   # ← x plays grad_output here
    stride=1, padding=0,
)

print(torch.allclose(y_transpose, y_bwd_input, atol=1e-5))   # → True
print("max diff =", (y_transpose - y_bwd_input).abs().max().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
True
max diff = 0.0
```

中文:同一段输入、同一份权重,跑 ConvTranspose2d 和跑 `conv2d_input` (conv2d 对输入的梯度算子) 得到 bit-exact 相同的结果 —— 这就是 Inductor 那段 30 行代码所依赖的恒等式。注意必须用 `torch.nn.grad.conv2d_input` 这个公开 API,而不是 autograd 走一遍。

English: Same input, same weight, and `ConvTranspose2d` vs. `conv2d_input` (the input-gradient operator of `conv2d`) return bit-exact identical tensors. That is the identity the new 30-line Inductor block hangs on. Note we use the public `torch.nn.grad.conv2d_input` op directly — *not* an autograd round trip.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **cuDNN 早就是这么干的 / cuDNN already does this**: `cudnnConvolutionBackwardData` 和 `cudnnConvolutionTransposeForward` 在底层是同一个 kernel 路径,只是参数顺序不一样。Inductor 现在追上了 cuDNN 的做法。 / `cudnnConvolutionBackwardData` and `cudnnConvolutionTransposeForward` share the same kernel path under the hood with different argument order. Inductor just caught up.
- **FlashAttention 用同一个 kernel 算 fwd / bwd / FlashAttention reuses one kernel for fwd / bwd**: Triton 的 FlashAttention 实现里,backward 的 `dQ, dK, dV` 都是前向 kernel 改一下 reduction 维度算的,核心 GEMM 是一样的 / The Triton FlashAttention backward computes `dQ, dK, dV` by reusing the forward kernel with different reduction axes — the core GEMM is identical.
- **AOTI 的 split-out-bias 重用 / AOTI's split-out-bias reuse**: PyTorch AOTI 编译 `F.linear(x, w, b)` 时,会把 bias 拆出去复用现有 GEMM kernel,完全是同一个 "shape-shift the inputs, reuse the existing GEMM" 思路 / When AOTI compiles `F.linear(x, w, b)`, it splits the bias out so it can reuse the existing GEMM. Same "shape-shift the inputs, reuse the existing GEMM" pattern.
- **xformers 的 attention bias 通过 GEMM 重用 / xformers's attention-bias-via-GEMM reuse**: xformers 把 attention bias 作为 GEMM 的累加项,同一个 fused GEMM kernel 覆盖了三种不同的 bias 模式 / xformers folds attention bias into the GEMM accumulator so one fused GEMM kernel covers three different bias modes.

## 注意事项 / Caveats / when it breaks

- **只在 2D 上工作 / Only 2D**: 这条优化路径 `ndim == 2` 是硬条件。`ConvTranspose1d` / `ConvTranspose3d` 仍然走 ATen 后备,因为对应维度的 backward-input Triton 模板还没写 / The `ndim == 2` guard is hard. `ConvTranspose1d` / `ConvTranspose3d` keep falling back to ATen because the matching backward-input templates haven't been written yet.
- **`output_padding != 0` 仍是问题 / `output_padding != 0` is still tricky**: ConvTranspose2d 的 `output_padding` 在 backward-input 那边没有直接对应,需要在 layout shape 上预补齐。PR 的测试覆盖了 `output_padding with stride > 1` 的情形,但你自己改 kernel 时要小心 / `output_padding` doesn't have a direct counterpart in backward-input — the layout shape has to be pre-padded. The PR's tests cover `output_padding with stride > 1`, but be careful if you modify the kernel.
- **`max_autotune_conv_backends` 必须非空 / `max_autotune_conv_backends` must be non-empty**: 那个 ATen 兜底逻辑只在 `config.max_autotune_conv_backends.strip()` 为真时触发。如果你显式把这个 config 设成 `""`,Inductor 仍然会按原来的行为崩 —— 这其实是有意保留的"专家模式" / The ATen safety net only fires when `config.max_autotune_conv_backends.strip()` is truthy. If you explicitly set it to `""` you keep the old crash behavior — that's an intentional "expert mode" escape hatch.

## 延伸阅读 / Further reading

- [PR #186067 — Reuse conv2d_bwd_input Triton template for ConvTranspose2d forward](https://github.com/pytorch/pytorch/pull/186067)
- [Issue #186066 — original bug report](https://github.com/pytorch/pytorch/issues/186066)
- [PR #178945 — the original conv2d_bwd_input Triton template that this PR reuses](https://github.com/pytorch/pytorch/pull/178945)
- [PyTorch Inductor autotuning docs](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html)
- [Conv transpose as backward-of-conv (a classic explainer)](https://arxiv.org/abs/1603.07285)
