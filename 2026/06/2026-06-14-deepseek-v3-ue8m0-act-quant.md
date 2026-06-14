---
date: 2026-06-14
topic: infrastructure
source: tracked
repo: deepseek-ai/DeepSeek-V3
file: inference/kernel.py
permalink: https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/kernel.py#L9-L57
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, fp8, triton, quantization, blackwell]
---

# DeepSeek-V3 的 ue8m0 act_quant:一句 ceil(log2(s)) 就把 FP8 scale 变成 Blackwell 原生 / DeepSeek-V3's ue8m0 act_quant: one `ceil(log2(s))` makes the FP8 scale Blackwell-native

> **一句话 / In one line**: 把 amax/448 这个浮点 scale 强行 round 到最近的 2 的整数次幂,scale 就变成 Blackwell TensorCore 直接认得的 ue8m0 格式,省掉一次 scale-级 rounding。 / Round the `amax / 448` scale up to the nearest power of two and the scale itself becomes Blackwell-native ue8m0, removing one rounding step from the MXFP8 GEMM.

## 为什么重要 / Why this matters

DeepSeek-V3 用 block-wise FP8 (e4m3) 跑训练:每 128 个连续元素共用一个 scale。原始版本里 scale 是 fp32,Triton 在做反量化时还得多读一个 fp32。Blackwell 上的 MXFP8 TensorCore 已经内置支持一种叫 **ue8m0** 的 scale 格式:8 位无符号指数 + 0 位尾数,本质就是 `2^k`。把 scale 强行 round 到 2 的幂,硬件可以直接 fused 进 GEMM,不再需要单独读 scale。这次提交(2025-08-27)正是这条物理学和硬件的同步线:`s = 2^ceil(log2(amax/448))`。

DeepSeek-V3 trains with block-wise FP8 (e4m3): every 128 contiguous elements share one scale. The original scale was fp32, which means the TensorCore has to load an extra fp32 per block. Blackwell's MXFP8 TensorCore natively supports a scale format called **ue8m0** — 8-bit unsigned exponent, zero mantissa, i.e. literally `2^k`. If you round the scale to a power of two, the hardware can fuse it into the GEMM and skip the scale-reading step entirely. The recent 2025-08-27 patch is exactly that: `s = 2^ceil(log2(amax/448))`.

## 代码 / The code

`deepseek-ai/DeepSeek-V3` — [`inference/kernel.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/kernel.py#L9-L57)

```python
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x)) # reduction
    amax = tl.maximum(amax, 1e-4) # clamp to 1e-4
    s = amax / 448.
    if scale_fmt == "ue8m0":
        exp = tl.math.ceil(tl.math.log2(s))
        s = tl.math.exp2(exp)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s
```

## 逐行讲解 / What's happening

1. **第 22-24 行 / Lines 22-24 (`pid` + `offs` + `tl.load`)**:
   - 中文: 每个 Triton program 处理一个 BLOCK_SIZE(默认 128)的元素块,在原 tensor 里对应 `[pid*128, pid*128+128)` 这个连续切片。`x.to(tl.float32)` 一定要做——后面要算 `log2`,fp16 精度不够。
   - English: Each Triton program instance owns one `BLOCK_SIZE` (= 128) slice `[pid*128, pid*128+128)`. The `to(tl.float32)` cast is non-negotiable — we are about to take `log2(s)` and fp16 doesn't have enough exponent range.

2. **第 26-27 行 / Lines 26-27 (`amax`)**:
   - 中文: 取这一块里绝对值最大的元素。`tl.maximum(amax, 1e-4)` 是为了防止整个 block 全是 0:如果不 clamp,后面 `s = amax/448` 会得到 0,再 `log2(0)` 直接 nan。
   - English: Take the block's max absolute value. The `tl.maximum(amax, 1e-4)` clamp guards against an all-zeros block — without it, `s = amax/448` would be 0 and `log2(0)` blows up to NaN.

3. **第 28 行 / Line 28 (`s = amax / 448.`)**:
   - 中文: 448 是 fp8 e4m3 能表示的最大有限值。把 scale 取成 `amax/448`,意思是:量化后最大那个 e4m3 值正好落在 ±448 上,把这个 block 整段动态范围吃满。
   - English: 448 is the largest finite value representable in fp8 e4m3. Picking `s = amax / 448` means the post-quant max e4m3 value lands exactly at ±448 — the block consumes its entire dynamic range, no clipping, no wasted bits.

4. **第 29-31 行 / Lines 29-31 (the ue8m0 branch)**:
   - 中文: 关键 3 行——`ceil(log2(s))` 把 scale 强行往上 round 到一个整数指数,然后 `exp2(exp)` 把它恢复成 `2^exp`。结果:scale 永远是 2 的幂。Blackwell 的 MXFP8 TensorCore 直接读这 8 个指数 bit,不再读 fp32。代价是 scale 略微变大、量化值略微变小,但避免了一次 scale 级 rounding 和一个 4× memory 占用。
   - English: The three magic lines. `ceil(log2(s))` rounds the scale **up** to an integer exponent, then `exp2(exp)` materialises it as `2^exp`. The scale is now always a power of two. Blackwell's MXFP8 TensorCore reads only those 8 exponent bits — no more fp32 scale read. The trade-off: the scale becomes slightly larger (so quantised magnitudes shrink a bit), but you skip a rounding step and quarter the scale-memory footprint.

5. **第 32-35 行 / Lines 32-35 (divide, cast, store)**:
   - 中文: `y = x / s` 把 block 里的元素归一化到 ±448,然后 `.to(y_ptr.dtype.element_ty)` 把它 cast 成 e4m3。注意 cast 这一步硬件还会做一次 round-to-nearest,这是不可避免的 element 级 rounding;但 scale 级的 rounding 通过 ue8m0 已经省掉了。
   - English: `y = x / s` normalises the block into ±448, then `.to(y_ptr.dtype.element_ty)` casts to e4m3 — the hardware does a round-to-nearest here, which is the unavoidable element-level rounding. The scale-level rounding, however, is gone thanks to ue8m0.

6. **第 51-56 行 / Lines 51-56 (Python launcher)**:
   - 中文: 输出 tensor 直接分配 `torch.float8_e4m3fn`,scale tensor 的 shape 是 `(..., last_dim // block_size)`——每 128 个元素一个 scale。Triton grid 是 1D,正好等于 block 的个数。
   - English: The output tensor is allocated directly as `torch.float8_e4m3fn`, and the scale tensor has shape `(..., last_dim // block_size)` — one scale per 128-element block. The Triton grid is 1D, matching the block count exactly.

## 类比 / The analogy

想象你要把一沓尺子(不同最大长度)塞进同一个固定 5 厘米刻度的盒子。原本你为每根尺子算一个精确缩小比例(比如 6.37 倍),写在便签上贴在盒边——读尺子时还要先念这个比例。**ue8m0 模式**说:"不要 6.37,改用 8。"刻度被压得稍小,但你只要写一个指数 3(因为 8=2³),盒子盖上印有指数尺,扫一眼就知道几倍,完全不用再读便签。便签那张纸就被你省下来了——这就是为什么 Blackwell 把 scale 内嵌到 GEMM 后吞吐能上去。

Picture stuffing a stack of rulers (each with a different maximum length) into a fixed 5 cm slot. The old way: compute a precise scaling ratio for each (say 6.37×), write it on a sticky note, attach it to the slot edge — and every time you read the ruler you also have to read the sticky note. The **ue8m0 mode** says: "don't use 6.37, round up to 8." The ruler is squeezed a touch tighter, but you only need to write the exponent `3` (since 8 = 2³) and the slot lid already has a printed exponent gauge — one glance tells you the multiplier, no sticky note needed. That sticky note is what Blackwell saves by fusing the scale into the GEMM.

## 自己跑一遍 / Try it yourself

```python
import torch

def py_act_quant(x, block_size=128, use_ue8m0=False):
    assert x.size(-1) % block_size == 0
    x32 = x.float().view(-1, block_size)
    amax = x32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    s = amax / 448.0
    if use_ue8m0:
        s = torch.pow(2.0, torch.ceil(torch.log2(s)))
    y = (x32 / s).to(torch.float8_e4m3fn)
    return y.view_as(x), s.view(*x.shape[:-1], -1)

torch.manual_seed(0)
x = torch.randn(2, 128, dtype=torch.bfloat16)
y_a, s_a = py_act_quant(x, use_ue8m0=False)
y_b, s_b = py_act_quant(x, use_ue8m0=True)
print("plain scale  :", s_a.flatten().tolist())
print("ue8m0 scale  :", s_b.flatten().tolist())
print("dequant err A:", (y_a.float() * s_a - x.float()).abs().mean().item())
print("dequant err B:", (y_b.float() * s_b - x.float()).abs().mean().item())
```

运行 / Run with:
```bash
pip install "torch>=2.5"
python try.py
```

预期输出 / Expected output:
```
plain scale  : [0.0072..., 0.0070...]
ue8m0 scale  : [0.0078125, 0.0078125]    # 都是 2^-7
dequant err A: 0.0014...
dequant err B: 0.0015...                 # 误差略大,但 scale 变成了 2 的幂
```

中文一句:ue8m0 模式的反量化误差通常只比普通模式高 5-10%,但 GEMM 吞吐能多出 1.5-2× 在 Blackwell 上——这就是用一点点精度换硬件原生支持的典型场景。

English: the ue8m0 dequant error is usually only 5–10 % worse than the plain version, but you gain 1.5–2× GEMM throughput on Blackwell hardware. A classic "trade a sliver of precision for native hardware support" deal.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NVIDIA Transformer Engine** / **NVIDIA Transformer Engine**: TE 的 MXFP8 amax 历史记录维护正是为了在 GEMM 前把 fp32 scale 替换成 ue8m0(`mxfp8_scale_factor` API)。 / TE's MXFP8 amax history is maintained precisely so that the fp32 scale can be replaced with ue8m0 (`mxfp8_scale_factor` API) right before the GEMM.
- **vLLM / SGLang 的 FP8 KV cache** / **vLLM / SGLang FP8 KV cache**: 同样的 `amax → scale → cast` 三段式,只是 block 大小变成 64 或 16(一个 head)。 / Same three-step `amax → scale → cast` recipe; the block is just 64 or 16 (one head). 
- **Open-source `TorchAO` int8/fp8 utilities** / **Open-source `TorchAO` int8/fp8 utilities**: TorchAO 的 `tensor_quant` 也是先取 amax、再除最大可表示值,DeepSeek 在它之上加了 ue8m0 这条快路。 / TorchAO's `tensor_quant` follows the same amax-then-divide-max recipe; DeepSeek's contribution is the ue8m0 fast path on top.

## 注意事项 / Caveats / when it breaks

- **必须 clamp amax** / **You must clamp amax**: 没有 `tl.maximum(amax, 1e-4)`,空块会让 `log2(0) = -inf` 把后面所有数学击穿。 / Without `tl.maximum(amax, 1e-4)`, an all-zeros block would hit `log2(0) = -inf` and detonate every downstream math op.
- **scale_fmt 必须是 tl.constexpr** / **`scale_fmt` must be `tl.constexpr`**: Triton 在编译时根据这个分支生成不同的 kernel——运行时再传字符串会强制走慢路径甚至 fail。 / Triton specialises the kernel at compile time on this branch. Passing the string at runtime forces the slow path or fails outright.
- **ue8m0 只在 Blackwell 上变快** / **ue8m0 only goes fast on Blackwell**: H100/H800 没有原生 ue8m0 scale 支持,这条分支在那里是纯精度损失没有性能收益。 / Hopper has no native ue8m0 support; on H100/H800 this branch is pure precision loss with zero throughput gain.
- **ceil 而不是 round** / **`ceil`, not `round`**: 上取整保证 scale ≥ 真实 scale,绝不会让任何元素超出 ±448——下取整就会 clip。 / Rounding up guarantees the scale ≥ the true scale, so no element can exceed ±448 — rounding down would clip.

## 延伸阅读 / Further reading

- [OpenAI cookbook: Mixed precision with FP8](https://github.com/openai/triton/blob/main/python/tutorials/14-fp8-matmul.py)
- [Microscaling Formats (MX) v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA Blackwell architecture whitepaper § MXFP8](https://resources.nvidia.com/en-us-blackwell-architecture)
- DeepSeek-V3 paper, Appendix B (FP8 training recipe)
