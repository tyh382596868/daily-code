---
date: 2026-06-10
topic: diffusion
source: trending
repo: PKU-YuanGroup/Helios
file: helios/modules/helios_kernels/attention_dispatch.py
permalink: https://github.com/PKU-YuanGroup/Helios/blob/8f2a2faab3298c8a7630a2c73aea37c01b5bab01/helios/modules/helios_kernels/attention_dispatch.py#L1-L167
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, attention, flash-attention, dispatcher]
---

# Helios 的 attention 派发器:在 GPU 动物园里活下来的 167 行 / Helios's attention dispatcher: 167 lines that survive the GPU zoo

> **一句话 / In one line**: 从 PKU 刚发的实时长视频世界模型 Helios 里抠出来一段 import 时 FA3→FA2→SageAttn→xformers→SDPA 优雅降级、运行时把 fixed-len 和 NaViT 风格 varlen 用同一个 API 路由出去 —— 这就是任何要在 H100/A100/4090 上一起跑的扩散世界模型最后都要写的那个文件。 / Lifted from PKU-YuanGroup's brand-new real-time long-video world model Helios: import-time FA3 → FA2 → SageAttn → xformers → SDPA graceful fallback, with a runtime API that routes both fixed-length and NaViT-style varlen through one function. Every diffusion world model that has to run across H100 / A100 / consumer cards eventually writes this exact file.

## 为什么重要 / Why this matters

要让一个 video diffusion 世界模型跑在用户真实硬件上,你必须面对的事实是:研究院里训练用的 H100 上有 FA3,GA 测试用的 A100 上只能跑 FA2,用户的 4090 可能只有 SageAttn 或者 xformers,而 partner 客户的 V100 只剩 PyTorch 自带的 SDPA。再加上长视频生成里因为不同 sample 历史帧数不同,经常用 NaViT-style varlen 来 batch 不等长序列 —— 这又是另一套 API。Helios 这 167 行就是 PKU 团队踩遍坑后的"production attention 派发器":import 时一路 try-except 选最优后端,runtime 一个 `attn_varlen_func` 统一对外暴露,内部自动判断走 fixed 还是 varlen 路径。读完它,你就知道任何 GPU-异质环境里 attention 调用层应该长什么样。

To ship a video-diffusion world model into the wild you must accept this: training was on H100 with FA3, your QA cluster only has FA2 on A100, the user's 4090 may have just SageAttn or xformers, and a partner's V100 falls back to PyTorch SDPA. Add to that NaViT-style varlen attention — different samples have different history lengths, so you batch ragged sequences — which uses a *different* API again. Helios's 167 lines are PKU's post-mortem "production attention dispatcher": at import time it try-excepts down a chain to the best available backend; at runtime, a single `attn_varlen_func` is exposed and internally routes to fixed-length or varlen kernels. Read it and you'll know exactly what the attention layer of any heterogeneous-GPU deployment should look like.

## 代码 / The code

`PKU-YuanGroup/Helios` — [`helios/modules/helios_kernels/attention_dispatch.py`](https://github.com/PKU-YuanGroup/Helios/blob/8f2a2faab3298c8a7630a2c73aea37c01b5bab01/helios/modules/helios_kernels/attention_dispatch.py#L1-L167)

```python
import torch
from kernels import get_kernel


try:
    # FA3 Only support Hopper (SM90, H100/H800)
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError("FA3 requires Hopper (SM90+), current GPU not supported")
    flash_attn3 = get_kernel("kernels-community/flash-attn3")
    flash_attn_func = flash_attn3.flash_attn_func
    flash_attn_varlen_func = flash_attn3.flash_attn_varlen_func
    print("Flash Attn 3 is installed!")
except (ImportError, RuntimeError):
    try:
        flash_attn2 = get_kernel("kernels-community/flash-attn2")
        flash_attn_func = flash_attn2.flash_attn_func
        flash_attn_varlen_func = flash_attn2.flash_attn_varlen_func
        print("Flash Attn 2 is installed!")
    except ImportError:
        print("Flash Attn 2 / 3 is not installed!")
        flash_attn_varlen_func = None
        flash_attn_func = None

try:
    from sageattention import sageattn, sageattn_varlen
    print("Sage Attn is installed!")
except ImportError:
    print("Sage Attn is not installed!")
    sageattn_varlen = None
    sageattn = None

try:
    from xformers.ops import memory_efficient_attention as xformers_attn_func
    print("Xformers is installed!")
except ImportError:
    print("Xformers is not installed!")
    xformers_attn_func = None


@torch.compiler.disable
def _flash_attn_wrapper(q, k, v):
    return flash_attn_func(q, k, v)

@torch.compiler.disable
def _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)


def attn_varlen_func(q, k, v, attention_mask=None):
    if attention_mask is None:
        if flash_attn_func is not None:
            x = _flash_attn_wrapper(q, k, v)
            return x

        if sageattn is not None:
            x = sageattn(q, k, v, tensor_layout="NHD")
            return x

        if xformers_attn_func is not None:
            x = xformers_attn_func(q, k, v)
            return x

        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return x

    B, L, H, C = q.shape

    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask
    if flash_attn_varlen_func is not None:
        x = _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    elif sageattn_varlen is not None:
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    else:
        raise NotImplementedError("No Attn Installed!")

    x = x.unflatten(0, (B, L))

    return x
```

## 逐行讲解 / What's happening

1. **第 5-13 行 / Lines 5-13 (FA3 import gate)**:
   - 中文: FA3 是 NVIDIA Hopper 专属(SM90+,H100/H800)。所以 import 之前先 `torch.cuda.get_device_capability()` 查 compute capability,major < 9 就主动 raise RuntimeError 转入 except 分支 —— 这是关键的一步,因为 FA3 的 Python 包在 SM80 卡上也能 `import flash_attn_3`,但 kernel 一调用就 illegal memory access。这种"软支持"是最坑的,所以先用硬件 check 兜底。
   - English: FA3 is Hopper-only (SM90+, H100/H800). Before importing, we check `torch.cuda.get_device_capability()` and raise RuntimeError on `major < 9` — *critical*, because the FA3 Python package can be imported on SM80 cards but errors out with illegal memory access on first kernel call. Silent breakage like that is the worst — so we hardware-gate before the import.

2. **`from kernels import get_kernel`**:
   - 中文: 这是个 HF 的小包 —— 它会从 hub 下载预编译的 CUDA 算子(`kernels-community/flash-attn3`)。比传统 `pip install flash-attn` 体验好太多,后者要在你本地编译 CUDA 代码,经常 build 失败。
   - English: an HF micro-package that fetches prebuilt CUDA ops from the hub (`kernels-community/flash-attn3`). Massively better UX than `pip install flash-attn`, which compiles CUDA on your machine and frequently fails.

3. **第 14-23 行 / Lines 14-23 (FA2 fallback)**:
   - 中文: 嵌套的 try/except —— 既捕获 FA3 包没装(`ImportError`),也捕获了硬件不支持的主动 raise(`RuntimeError`)。退到 FA2 后再尝试一次同样的 import。如果连 FA2 都没有,两个变量都设 None —— **后面 runtime 会通过 `is not None` 判断走哪条路径**。
   - English: nested try/except catches both "FA3 not installed" (`ImportError`) and our hardware-gate `RuntimeError`. Falls back to FA2 by trying the parallel import. If even FA2 is missing, both vars are set to None — *runtime path selection later keys off `is not None` checks*.

4. **第 25-32 行 / Lines 25-32 (Sage)** + **第 34-39 行 / Lines 34-39 (xformers)**:
   - 中文: 互相独立的 try/except 块。注意 Sage 和 xformers 都不是 FA 系列的"竞争品",而是补充 —— Sage 在消费级 GPU 上的 int8/int4 量化 attention 上比 FA 快,xformers 是历史悠久的稳定 fallback。装一套也行,装多套更稳。
   - English: independent try/except blocks. Sage and xformers aren't competitors to FA — they complement it. Sage's int8/int4 quantized attention is faster on consumer GPUs than FA. xformers is the venerable stable fallback. Having any one is enough, having all is safest.

5. **`@torch.compiler.disable`**:
   - 中文: 极其关键的一行。Flash Attn 的 forward 是 C++/CUDA 算子,`torch.compile` 进去之后 Dynamo 会尝试 trace,但是 trace 不进 CUDA kernel,就会报错或者无声地慢。`disable` 告诉 compile "这个函数是黑盒,你别碰它,直接调原 Python 实现"。
   - English: critical. Flash Attn's forward is a C++/CUDA op; `torch.compile`'s Dynamo tries to trace through it but cannot enter CUDA kernels — either it errors or silently slows. `disable` tells compile "this function is a black box, leave it alone, call the original Python".

6. **`attn_varlen_func`,第 47-50 行 / Lines 47-50 (fixed-len path entry)**:
   - 中文: 这个函数有点反直觉的命名 —— 它叫 `varlen`,但内部根据 `attention_mask is None` 二分:`None` 就走 fixed-len 路径(传统 attention),否则走 varlen 路径。这种"统一对外 API + 内部分发"是 production 系统降低调用方负担的标配。
   - English: counterintuitively named — it's called `varlen` but bifurcates on `attention_mask is None`: None means fixed-len (regular attention), else varlen. This "one unified API, internal dispatch" is a production hallmark that lowers caller burden.

7. **第 49-67 行 / Lines 49-67 (fixed-len fallback chain)**:
   - 中文: 当 `attention_mask=None` 时,按 FA → Sage → xformers → SDPA 的顺序尝试 —— 每个都先 `is not None` 检查再调用。注意 SDPA 调用时 `q.transpose(1, 2)` 是因为 PyTorch SDPA 接受 `(B, H, L, D)` 而 FA 接受 `(B, L, H, D)`。最后转回去保证输出形状一致。
   - English: when `attention_mask=None`, try FA → Sage → xformers → SDPA. Each branch first checks `is not None`. SDPA needs `q.transpose(1, 2)` because PyTorch SDPA wants `(B, H, L, D)` while FA uses `(B, L, H, D)`. Transposes back at the end to keep output shape consistent.

8. **第 69-83 行 / Lines 69-83 (varlen path)**:
   - 中文: varlen 模式 attention_mask 是个 4-tuple:`cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv` —— FlashAttn 的 varlen API 标准接口。`cu_seqlens` 是"累积序列长度"的前缀和,`[0, L1, L1+L2, L1+L2+L3, ...]`,让 kernel 知道一条 flat 张量里每个子序列的边界。`q.flatten(0, 1)` 把 `(B, L, H, D)` 压成 `(B*L, H, D)`,再 unflatten 回来。varlen 里只有 FA 和 Sage 提供,否则直接 NotImplementedError —— 因为 xformers 和 SDPA 没有等价 API。
   - English: in varlen mode, `attention_mask` is a 4-tuple — `cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv` — FlashAttn's standard varlen API. `cu_seqlens` is a prefix-sum of sequence lengths, `[0, L1, L1+L2, L1+L2+L3, ...]`, telling the kernel where each subsequence starts/ends in the flat tensor. `q.flatten(0, 1)` collapses `(B, L, H, D)` → `(B*L, H, D)`, then unflatten back at the end. Only FA and Sage offer varlen; otherwise raise NotImplementedError because xformers and SDPA have no equivalent API.

## 类比 / The analogy

这就像一个连锁咖啡店要在全球开店。在东京你有最好的咖啡豆(FA3),在上海退而求其次(FA2),在拉萨可能只能用茶叶代替(Sage),在火星上只有 Tang 粉(SDPA)。但顾客的菜单是一样的 —— "来一杯拿铁",店里店外都一样地点。后厨那台叫"attention dispatcher"的机器自动判断:这家店有什么原料?能做拿铁吗?做不了就用最接近的替代品。客人感觉到的"还行"的体验,背后是这家分店的厨房在选不同的原料组合 —— 你的代码只调一个 `attn_varlen_func`,具体跑的是哪个 backend,完全不用关心。

It's like a global coffee chain. In Tokyo you have the best beans (FA3). In Shanghai you settle for second-tier (FA2). In Lhasa you might have to use tea (Sage). On Mars you only have Tang powder (SDPA). But the menu the customer reads is identical — "one latte please" — whether they're at the Tokyo store or the Mars outpost. A back-kitchen machine called "attention dispatcher" automatically picks the closest match from whatever the local store has. Your code calls one `attn_varlen_func`; which backend actually runs is invisible — and that's the point.

## 自己跑一遍 / Try it yourself

```python
# Minimal dispatcher pattern — same idea, runs CPU-only on any machine.
import torch
import torch.nn.functional as F

# Pretend backends — register what's available
flash_attn = None
sage_attn  = None
try:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    sdpa_ok = True
except ImportError:
    sdpa_ok = False

def attn_dispatch(q, k, v):
    """One API, fallback chain inside."""
    if flash_attn is not None:
        print("→ FA")
        return flash_attn(q, k, v)
    if sage_attn is not None:
        print("→ SageAttn")
        return sage_attn(q, k, v)
    if sdpa_ok:
        print("→ SDPA (PyTorch fallback)")
        # SDPA expects (B, H, L, D); our convention is (B, L, H, D)
        return sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    raise NotImplementedError("No attention backend available!")

B, L, H, D = 2, 64, 4, 32
q = torch.randn(B, L, H, D)
k = torch.randn(B, L, H, D)
v = torch.randn(B, L, H, D)
out = attn_dispatch(q, k, v)
print("output shape:", out.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
→ SDPA (PyTorch fallback)
output shape: torch.Size([2, 64, 4, 32])
```

中文:把 `sdpa_ok = False` 改成 `False`,看到 NotImplementedError —— 这就是 production 系统遇到"什么后端都没装"时的明确失败信号,不会假装能跑。

English: flip `sdpa_ok = False` manually and watch NotImplementedError fire — that's the production signal "no backend available", a loud failure beats silent slowness.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **HuggingFace transformers `_flash_attention_forward`** / **HuggingFace transformers `_flash_attention_forward`**:同样的 import-time FA detection + runtime 路径选择,但 transformers 内部用一个 enum 而不是 None-check。 / Same import-time FA detection + runtime path selection, but transformers uses an enum internally instead of None-checks.
- **vLLM 的 attention_selector** / **vLLM's `attention_selector`**:更复杂(因为还涉及 paged-KV 后端 vs vanilla 后端),但骨架完全相同。 / More complex because it also dispatches paged-KV vs vanilla backends, but the skeleton is identical.
- **diffusers 的 `AttnProcessor` 多态** / **diffusers's `AttnProcessor` polymorphism**:走的是面向对象路线 —— 每个后端一个 Processor class,模型在 `__init__` 时根据可用性挑一个。形状不同但语义一样。 / OOP route — one Processor class per backend, picked at `__init__` based on availability. Different shape, same semantic.
- **PyTorch SDPA backend registry**(昨天讲过 / covered earlier) / **PyTorch SDPA backend registry**:PyTorch 内置版本的"派发器",FA3/FA2 通过 `priority_order()` 注册进 SDPA 调度器。 / PyTorch's built-in dispatcher; FA3/FA2 register into the SDPA scheduler via `priority_order()`.

## 注意事项 / Caveats / when it breaks

- **`@torch.compiler.disable` 不能忘** / **don't forget `@torch.compiler.disable`**:不加这行,`torch.compile` 进 FlashAttn 会出现"莫名 graph break + 性能反而下降"。 / Without it, `torch.compile` over FlashAttn produces graph breaks and unexpected slowdowns.
- **import 顺序很重要** / **import order matters**:这套 try-except 是在 *module-level* 跑的 —— `import helios.modules.helios_kernels.attention_dispatch` 那一刻就在选后端。要切换后端只能 `os.environ` 调控或者重启进程。 / The try-except runs at *module level* — backend selection happens at import time. To switch backends, set an env var before import or restart the process.
- **FA3 + bf16 + head_dim=128 是甜区** / **FA3 + bf16 + head_dim=128 is the sweet spot**:head_dim=64 时 FA3 比 FA2 快不多;head_dim=256 时反而可能慢。所以"装上 FA3"≠ "无脑加速",还要看模型形状。 / FA3 only big-wins at head_dim=128. At head_dim=64 it's marginally faster than FA2; at head_dim=256 it can be slower. Installing FA3 ≠ automatic speedup.
- **Sage 在某些 head_dim 上输出会和 FA 有数值差异** / **Sage's output diverges from FA on certain head_dims**:int8 量化的代价 —— 训练时不要混用 backend,推理时要做 numerical regression test。 / int8-quant cost — never mix backends during training; do a numerical regression test in inference.
- **varlen 模式下 batch 假象消失** / **varlen mode hides batches**:`q.flatten(0, 1)` 之后看起来是一条 batch=1 的序列,但 `cu_seqlens` 编码了真实 batch 结构。kernel 内部按 `cu_seqlens` 分块算 attention,各子序列之间不互相 attend。 / `q.flatten(0, 1)` makes everything look batch=1 but `cu_seqlens` encodes the real batch structure. Kernels chunk attention along `cu_seqlens`; subsequences don't attend to each other.

## 延伸阅读 / Further reading

- [Helios paper / blog (PKU-YuanGroup, real-time long video diffusion)](https://github.com/PKU-YuanGroup/Helios)
- [FlashAttention 3 paper — "Asynchrony and low-precision"](https://arxiv.org/abs/2407.08608)
- [SageAttention paper — quantized attention](https://arxiv.org/abs/2410.02367)
- [HuggingFace `kernels` library — prebuilt CUDA op fetching](https://huggingface.co/docs/kernels/index)
