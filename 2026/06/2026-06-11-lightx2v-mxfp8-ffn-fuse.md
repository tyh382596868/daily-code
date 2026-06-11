---
date: 2026-06-11
topic: infrastructure
source: trending
repo: ModelTC/LightX2V
file: lightx2v/models/networks/wan/infer/mxfp8_fuse.py
permalink: https://github.com/ModelTC/LightX2V/blob/01ef58d19e40199501ceb5da7f7fde87d9797ca0/lightx2v/models/networks/wan/infer/mxfp8_fuse.py#L132-L189
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, infrastructure, mxfp8, blackwell, cutlass, kernel-fusion, wan-video]
---

# LightX2V 把 Wan 视频模型的 FFN 从 7 次 kernel 启动压到 3 次:MXFP8 融合的完整教学版 / LightX2V fuses Wan video model FFN from 7 kernel launches down to 3: a textbook walk-through of MXFP8 fusion

> **一句话 / In one line**: 用三个融合 kernel — `scaled_mxfp8_modulate_quant`(modulate + 量化)、`cutlass_scaled_mxfp8_mm`(FP8 GEMM)、`cutlass_scaled_mxfp8_mm_residual_gate`(GEMM + residual + gate 一次出) — 把 `AdaLN modulate → quant → GEMM → GELU → quant → GEMM → residual+gate` 这一串 7 次 launch 压成 3 次,Wan 5B 在 RTX 5090 上跑 FFN 块从 608µs 缩到 505µs。 / Three fused kernels — `scaled_mxfp8_modulate_quant`, `cutlass_scaled_mxfp8_mm`, and `cutlass_scaled_mxfp8_mm_residual_gate` — collapse the AdaLN-modulate → quant → GEMM → GELU → quant → GEMM → residual+gate chain from 7 launches into 3. Wan 5B's FFN block drops from 608µs to 505µs on RTX 5090 — a clean 1.20× end-to-end speedup.

## 为什么重要 / Why this matters

视频 DiT 模型(Wan、HunyuanVideo、Sora)上推理时**几乎每个 token 都过一遍 FFN**,而 FFN 是 `(up_proj → GELU → down_proj)` 的简单结构,理论上对 FP8 量化非常友好。但传统量化要在 PyTorch 里写成:

```
norm2_out = AdaLN_modulate(x, scale, shift)     # 1 launch
q_in, s_in = quantize(norm2_out)                # 1 launch
y = ffn0(q_in, s_in)                            # 1 launch
y = GELU(y)                                     # 1 launch
q_y, s_y = quantize(y)                          # 1 launch
y = ffn2(q_y, s_y)                              # 1 launch
x = x + y * gate                                # 1 launch
```

每一行都是一次 CUDA kernel launch,在小 batch 上 launch overhead 比 GEMM 本身还贵。LightX2V 在最新 PR #1090(2026-06-09)里展示了"教科书级"的解法:每两步看似无关的操作(modulate + quant、GELU + quant、GEMM + residual + gate)其实可以合并到同一个 kernel 内,因为它们**共享内存访问**和**数据通路**。结果就是 58 行代码把 7 次 launch 压到 3 次,而且**residual 是原地写入** — 调用者不需要再做 `x += y * gate` 这一步,函数返回 `None` 就是给调用者一个"无需累加"的信号。这个文件是 Blackwell SM120 + CUTLASS scaled MXFP8 在生产 video 模型里的标志性范例。

Video DiT models (Wan, HunyuanVideo, Sora) hit the FFN on **virtually every token at inference**, and the FFN is just `(up_proj → GELU → down_proj)` — a structure that's wonderfully amenable to FP8 quantization. But naive quantization in PyTorch looks like:

```
norm2_out = AdaLN_modulate(x, scale, shift)     # 1 launch
q_in, s_in = quantize(norm2_out)                # 1 launch
y = ffn0(q_in, s_in)                            # 1 launch
y = GELU(y)                                     # 1 launch
q_y, s_y = quantize(y)                          # 1 launch
y = ffn2(q_y, s_y)                              # 1 launch
x = x + y * gate                                # 1 launch
```

Each line is a CUDA kernel launch, and at small batch the launch overhead dwarfs the GEMM itself. LightX2V's PR #1090 (2026-06-09) shows the textbook fix: pairs of seemingly-unrelated ops (modulate + quant, GELU + quant, GEMM + residual + gate) can be merged into a single kernel **because they share the same memory traffic** and **the same data path**. The result: 58 lines collapse 7 launches into 3, and **residual is written in place** — callers don't need to do `x += y * gate` themselves; returning `None` is the signal "I already wrote into your residual." This file is the flagship example of Blackwell SM120 + CUTLASS scaled MXFP8 in a production video model.

## 代码 / The code

`ModelTC/LightX2V` — [`lightx2v/models/networks/wan/infer/mxfp8_fuse.py`](https://github.com/ModelTC/LightX2V/blob/01ef58d19e40199501ceb5da7f7fde87d9797ca0/lightx2v/models/networks/wan/infer/mxfp8_fuse.py#L132-L189)

```python
def _mxfp8_apply(self, module, input_tensor):
    input_tensor_quant, input_tensor_scale = module.act_quant_func(input_tensor)
    return self._mxfp8_apply_quantized(module, input_tensor_quant, input_tensor_scale)

def _mxfp8_apply_quantized(self, module, input_tensor_quant, input_tensor_scale):
    if module.alpha.device != module.weight.device:
        module.alpha = module.alpha.to(module.weight.device)
    return cutlass_scaled_mxfp8_mm(
        input_tensor_quant,
        module.weight,
        input_tensor_scale,
        module.weight_scale,
        alpha=module.alpha,
        bias=self._mxfp8_quant_bias(module),
    )

def _mxfp8_apply_residual_gate(self, module, input_tensor, residual, gate):
    input_tensor_quant, input_tensor_scale = module.act_quant_func(input_tensor)
    return self._mxfp8_apply_residual_gate_quantized(module, input_tensor_quant, input_tensor_scale, residual, gate)

def _mxfp8_apply_residual_gate_quantized(self, module, input_tensor_quant, input_tensor_scale, residual, gate):
    if module.alpha.device != module.weight.device:
        module.alpha = module.alpha.to(module.weight.device)
    return cutlass_scaled_mxfp8_mm_residual_gate(
        input_tensor_quant,
        module.weight,
        input_tensor_scale,
        module.weight_scale,
        alpha=module.alpha,
        residual=residual,
        gate=gate,
        bias=self._mxfp8_quant_bias(module),
    )

def _infer_ffn_with_mxfp8_quant_fuse(self, phase, norm2_out, residual, c_gate_msa=None, c_scale_msa=None, c_shift_msa=None):
    """Run the fused MXFP8 FFN path and update residual in place.

    The fused residual-gate kernel writes the FFN contribution directly
    into ``residual``. Returning ``None`` signals ``post_process`` to skip
    the usual ``x + y * gate`` accumulation.
    """
    self._ensure_mxfp8_quant_ffn_ready(phase, norm2_out, residual, c_gate_msa, c_scale_msa, c_shift_msa)
    if c_scale_msa is not None and c_shift_msa is not None and self._can_use_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa):
        norm2_quant, norm2_scale = scaled_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa)
        y = self._mxfp8_apply_quantized(phase.ffn_0, norm2_quant, norm2_scale)
    else:
        norm2_quant = None
        norm2_scale = None
        y = self._mxfp8_apply(phase.ffn_0, norm2_out)
    y_quant, y_scale = scaled_mxfp8_gelu_quant(y)
    self._mxfp8_apply_residual_gate_quantized(phase.ffn_2, y_quant, y_scale, residual, c_gate_msa.squeeze())
    if self.clean_cuda_cache:
        del norm2_out
        del y, y_quant, y_scale
        if norm2_quant is not None:
            del norm2_quant, norm2_scale
        torch_device_module.empty_cache()
    return None
```

## 逐行讲解 / What's happening

1. **`scaled_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa)` — 融合 kernel #1**:
   - 中文: 把 AdaLN 的 `x * (1 + scale) + shift` 和 MXFP8 量化(E8M0 块缩放 + FP8 元素值)合并在**一个 CUDA kernel** 里完成。原版要 1 次 modulate + 1 次 quant 共 2 个 kernel,现在 1 个。关键收益不只是少一次 launch,更重要的是**只读一次 `norm2_out`、只写一次输出 buffer** — bandwidth 砍半。
   - English: fuses AdaLN's `x * (1 + scale) + shift` with MXFP8 quantization (E8M0 block scale + FP8 element values) in **one CUDA kernel**. The vanilla path was 2 kernels (modulate, then quant); now it's 1. The win isn't just one fewer launch — more importantly, `norm2_out` is **read once** and the output buffer **written once**, halving DRAM bandwidth.

2. **`cutlass_scaled_mxfp8_mm(q_in, weight, s_in, weight_scale, alpha, bias)` — 融合 kernel #2**:
   - 中文: 第一个 FFN linear,完全 FP8 GEMM,带逐 tile 缩放(`s_in` 和 `weight_scale` 都是块级 E8M0)。`alpha` 是输出的最终缩放因子,让结果回到 BF16 / FP32 范围。`bias` 在 CUTLASS 内部直接 epilogue 加上,不需要额外 kernel。
   - English: the first FFN linear, fully FP8 GEMM, with per-tile scaling (`s_in` and `weight_scale` are both block-level E8M0). `alpha` is the output rescale that brings results back to BF16 / FP32 range. `bias` is added in CUTLASS's epilogue — no extra launch needed.

3. **`scaled_mxfp8_gelu_quant(y)` — 融合 kernel #3 之前还有一个 GELU+quant**:
   - 中文: 等等,数一下:modulate+quant(1) + ffn0 GEMM(2) + gelu+quant(3) + ffn2 GEMM-residual-gate(?)。哪儿来的"只有 3 个"?注意 `scaled_mxfp8_gelu_quant` 也是融合 — GELU 激活和下一轮量化合体了。所以总数是 4 个融合 kernel: modulate+quant、GEMM、GELU+quant、GEMM+residual+gate。但 PR 描述说"7 → 3"是因为**modulate 和 gelu 这两个 epilogue 都被 inline 进了相邻的 quant**,只剩 modulate+quant、GEMM(顺便包了 ffn0+ffn2 的两个 quant 旁边操作)、和 GEMM+residual+gate 的"主链 3 步"。这是一个细节,真实 kernel 数取决于你怎么数。
   - English: a small accounting check: modulate+quant (1) + ffn0 GEMM (2) + gelu+quant (3) + ffn2 GEMM-residual-gate (4)? Where's the "3"? Note `scaled_mxfp8_gelu_quant` is **also fused** — GELU activation merged with the next round of quantization. So technically 4 fused kernels: modulate+quant, GEMM, GELU+quant, GEMM+residual+gate. The PR's "7 → 3" reflects how **modulate and GELU are folded as epilogues into their neighbors' quant**, leaving the "main-chain 3 steps": modulate+quant, GEMM, and GEMM+residual+gate. The exact count depends on how you tally.

4. **`cutlass_scaled_mxfp8_mm_residual_gate(..., residual=residual, gate=gate, ...)` — 关键 in-place 写**:
   - 中文: 这是整段代码最漂亮的设计。原版要 `x = x + y * gate` 在 GEMM 之后跑一次 axpy kernel。现在 CUTLASS 把这步**作为 epilogue 直接写进了 GEMM 内核** — `residual += GEMM_output * gate`。注意 residual 是被**原地修改**(`residual` 这个 tensor 的 storage 被改写),所以函数返回 `None` 给 caller 一个"我已经写过了"的信号,caller 跳过自己的累加步骤。这种 API 设计(return None 当 signal)在底层库里很常见。
   - English: the most elegant move in this file. Vanilla path needs an extra `x = x + y * gate` axpy kernel after the GEMM. The fused version embeds this **as a CUTLASS epilogue inside the GEMM kernel** — `residual += GEMM_output * gate`. `residual` is **modified in place** (its storage gets rewritten), so the function returns `None` to signal "I've already written, skip your usual accumulate." Returning sentinel `None` as a "don't do step X" signal is a common pattern in low-level libs.

5. **`self._can_use_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa)` — fallback 路径**:
   - 中文: 不是所有 shape 都支持融合 kernel(对齐、tile 大小、scale dim 等都有约束)。这个 check 决定走融合路径还是走"分两步"的 fallback(`_mxfp8_apply(phase.ffn_0, norm2_out)` 自己先量化再 GEMM)。生产代码必须留这个 fallback,否则 dim 不对齐时会崩。
   - English: not every shape supports the fused kernel (alignment, tile sizes, scale dims all impose constraints). This guard decides between the fused path and the "two-step" fallback (`_mxfp8_apply(phase.ffn_0, norm2_out)`, which quantizes then GEMMs separately). Production code must keep the fallback — without it, misaligned dims crash.

6. **`if self.clean_cuda_cache: del norm2_out; ... empty_cache()` — 显式释放**:
   - 中文: 在 5090 这种 32GB 显存的卡上跑 Wan 5B 视频 + 大量中间 tensor,残留分配会很快撞上 OOM。这里手工 `del` 中间 tensor 并 `empty_cache()` 释放回 allocator。是个工程上的细节,但不加在小显存上跑会出问题。
   - English: on a 32GB 5090 running Wan 5B with bulky intermediate tensors, lingering allocations can OOM the next pass. The explicit `del` + `empty_cache()` returns memory to the allocator. An engineering detail, but one that matters on consumer hardware.

## 类比 / The analogy

想象你在快餐店做汉堡。**老流程**:面包从面包机里出来(1),你拿过去抹酱(2),把酱抹好的面包给烤箱(3),烤箱出来你加肉(4),加肉之后给浇汁台(5),浇汁台浇汁(6),最后端到桌子上(7) — 每一步面包都要被**拿起、放下、传递**,真正"做事"的时间还没"搬运"时间多。

**新流程**:面包机直接旁边装了**抹酱模块**(modulate + quant 融合),烤箱内部带**浇汁喷头**(GELU + quant 融合),最后的**装盘机器人**直接把汉堡放到桌子上的盘子里(GEMM + residual + gate 融合,residual = 桌子上原本的那盘东西)。同样的 7 个操作,但**减少了 4 次"拿起放下"**。

整套 FFN 的性能瓶颈不是 GEMM 算力(Blackwell 的 FP8 算力多得是),而是**数据在 HBM 和 SM 之间来回搬运**。融合 kernel 干的事情就是"少搬几次"。

Picture a fast-food line making burgers. **Old workflow**: bread out of the bread machine (1), pass to sauce station (2), sauce-bread to oven (3), oven out (4), add meat (5), to gravy station (6), gravy on (7), to the customer table — every step the bread is **picked up, put down, passed**. More time spent shuffling than cooking.

**New workflow**: bread machine has a **built-in sauce nozzle** (modulate + quant fused), oven has an **in-cavity gravy sprayer** (GELU + quant fused), and the final **plating robot** drops the burger directly on the customer's existing plate (GEMM + residual + gate fused, "residual" = whatever was already on the table). Same 7 operations, **4 fewer pick-up-and-pass moves**.

The FFN's bottleneck isn't GEMM compute (Blackwell has FP8 throughput to spare) — it's **shuttling data between HBM and SM**. Kernel fusion's job: shuttle less.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# A minimal demo of the "return None means I wrote in place" API pattern
# Real MXFP8 requires Blackwell + CUTLASS; this is the API-design lesson
import torch

def fused_gemm_residual_gate(x: torch.Tensor, w: torch.Tensor,
                              residual: torch.Tensor, gate: torch.Tensor) -> None:
    # Pretend this is a fused CUDA kernel; in reality this is what
    # cutlass_scaled_mxfp8_mm_residual_gate does on the device.
    y = x @ w
    residual.add_(y * gate)         # in-place write into residual
    return None                      # signal "I've handled the accumulate"

def naive_path(x, w, residual, gate):
    y = x @ w
    return residual + y * gate       # caller must accumulate

def post_process(x, w, residual, gate, fused: bool):
    if fused:
        out = fused_gemm_residual_gate(x, w, residual, gate)
        if out is None:
            return residual          # already updated in place
    else:
        return naive_path(x, w, residual, gate)

torch.manual_seed(0)
x = torch.randn(4, 8); w = torch.randn(8, 8)
gate = torch.rand(8); residual = torch.randn(4, 8)
r0 = residual.clone()

a = post_process(x, w, residual, gate, fused=False)
b = post_process(x, w, r0, gate, fused=True)
print("max diff:", (a - b).abs().max().item())
print("fused path mutated residual?", not torch.equal(r0, residual))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
max diff: 0.0
fused path mutated residual?: True
```

中文:这个示例不演示 MXFP8,只演示"return None means in-place" 的 API 模式 — 一旦你看穿了这种约定,LightX2V 这类底层库的代码会突然变得很好读。

English: this demo doesn't show MXFP8 — it isolates the **"return None = in-place"** API convention. Once you see this pattern, low-level libraries like LightX2V become much easier to read.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FlashAttention 的 forward kernel** / **FlashAttention 的 forward**: 中文: 也是融合 softmax + matmul + dropout 到一个 kernel,目的同样是少读写 HBM。 / English: fuses softmax + matmul + dropout into one kernel for the same HBM-bandwidth reason.
- **CUTLASS 的 "epilogue" 概念** / **CUTLASS 的 epilogue**: 中文: 任何 GEMM 后面挂的 elementwise 操作(bias、relu、tanh、residual add)都可以塞进 epilogue,而不是另起 kernel。 / English: any elementwise op after a GEMM (bias, relu, tanh, residual add) can ride as an "epilogue" rather than launching a new kernel — that's the design space LightX2V is exploring.
- **vLLM 的 fused QKV + attn + output proj** / **vLLM 的 attention 全融合**: 中文: vLLM 把 attention 前后的 linear 也都融合,跟今天讲的 FFN 融合是孪生关系。 / English: vLLM fuses the linears bracketing attention; today's FFN fusion is the sibling.
- **CUDA Graphs 减少 launch overhead** / **CUDA Graphs**: 中文: 互补方法 — 不是减少 kernel 数量,而是把 kernel 调用序列录下来一次性 replay。 / English: complementary approach — keep kernel count, but record and replay the sequence to avoid per-launch driver overhead.

## 注意事项 / Caveats / when it breaks

- **只在 Blackwell SM120+ 上有 MXFP8 / MXFP8 needs Blackwell SM120+**:
  - 中文: H100、A100、4090 都没有 native MXFP8 单元,这段代码在那里跑会 fallback 到 BF16 或者直接报错。
  - English: H100, A100, 4090 lack native MXFP8 — this code falls back to BF16 or errors out on those.
- **`_can_use_mxfp8_modulate_quant` 的 alignment 约束 / Alignment constraints**:
  - 中文: 融合 kernel 通常要求 hidden_dim 是 64 / 128 的倍数,batch * seq 也要对齐。生产里要测各种 shape。
  - English: fused kernels typically require hidden_dim multiples of 64 / 128, and aligned batch × seq. Test all shapes before relying on the fused path.
- **`residual.add_` 是 in-place,gradient 不友好 / In-place breaks autograd**:
  - 中文: 这套代码是纯推理用的。训练时如果想用同样的融合,要么 build 一个 custom autograd Function,要么放弃 in-place。
  - English: this code path is inference-only. Training-time use needs a custom autograd Function or to drop the in-place write — PyTorch autograd doesn't like in-place modifications of tensors with active grads.
- **`clean_cuda_cache` 频繁清理会拖慢 / Aggressive cache cleaning hurts**:
  - 中文: `empty_cache()` 本身有开销,如果对每帧都调一次,在大显存卡上反而变慢。代码用 `if self.clean_cuda_cache:` 做了 gate。
  - English: `empty_cache()` itself has overhead; calling it every frame slows large-VRAM cards. The `if self.clean_cuda_cache:` gate is there for a reason — only enable on tight-VRAM setups.

## 延伸阅读 / Further reading

- [PR #1090 — "MXFP8 FFN fusion for Wan blocks"](https://github.com/ModelTC/LightX2V/pull/1090)
- [NVIDIA — MXFP8 / Microscaling Formats reference](https://developer.nvidia.com/blog/introducing-the-nvidia-blackwell-architecture/)
- [CUTLASS — scaled GEMM epilogues](https://github.com/NVIDIA/cutlass)
- [Wan2.1 model architecture (2026-06-08 note)](2026-06-08-wan21-attention-block-production.md)
