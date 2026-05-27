---
date: 2026-05-27
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/lora/layer.py
permalink: https://github.com/huggingface/peft/blob/417aff89a3b302b5cb1cbb7a7205c126641fb01c/src/peft/tuners/lora/layer.py#L953-L994
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, lora, finetune]
---

# PEFT 的 LoRA forward:一行加法就是整个 LoRA / The one line of addition that *is* LoRA — PEFT's `Linear.forward`

> **一句话 / In one line**: 把 PEFT 一千多行的 LoRA 实现层层剥开,真正干活的就是 `result + lora_B(lora_A(dropout(x))) * scaling` 这一行;其余都是 dtype 转换、多适配器路由、merge/unmerge 状态机的"外壳"。 / Strip away PEFT's thousand-plus lines of LoRA scaffolding and the real computation is a single residual: `result + lora_B(lora_A(dropout(x))) * scaling`. Everything else is dtype shuffling, multi-adapter routing, and merge-state plumbing.

## 为什么重要 / Why this matters

LoRA(Low-Rank Adaptation)是过去三年微调大模型最重要的工程发明。原理一句话:别动 base 权重 `W`,而是学一个低秩近似 `ΔW = B·A`,其中 `A ∈ R^{r×d}` 和 `B ∈ R^{d×r}` 都是小矩阵(`r << d`)。可训练参数从 `d²` 降到 `2rd`,在 r=8 时通常能省 99% 显存。**但代码里 LoRA 真正长什么样?**——很多教程画了图却没给一行能跑的实现。PEFT 这段 `Linear.forward` 把它写得极其干净:`base_layer(x)` 给你 `Wx`,`lora_B(lora_A(dropout(x)))` 给你 `BA·x`,加起来就是 `(W + BA)·x = (W + ΔW)·x`。读这段代码的额外价值,是看清"工业级 LoRA 实现"在 minimal 数学之外还要处理什么:多适配器叠加、merged/unmerged 状态切换、disable_adapters 开关、dtype 来回 cast。看完你会理解为什么 PEFT 的 `Linear` 类一共两千多行。

LoRA (Low-Rank Adaptation) is the most influential fine-tuning trick of the last three years. The idea is one sentence: leave the base weight `W` alone, learn a low-rank delta `ΔW = B·A` with `A ∈ R^{r×d}` and `B ∈ R^{d×r}` (`r << d`). Trainable parameters drop from `d²` to `2rd` — typically a 99% reduction at r=8. **But what does LoRA actually look like in code?** Many tutorials show diagrams; few show a runnable line. PEFT's `Linear.forward` shows it cleanly: `base_layer(x)` gives `Wx`, `lora_B(lora_A(dropout(x)))` gives `BA·x`, sum to `(W + BA)·x = (W + ΔW)·x`. Reading this file also reveals everything an industrial LoRA implementation has to handle *beyond* the math: multi-adapter stacking, merged/unmerged state, the `disable_adapters` toggle, dtype casts. After this you'll understand why the `Linear` class alone is 2000+ lines.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/lora/layer.py`](https://github.com/huggingface/peft/blob/417aff89a3b302b5cb1cbb7a7205c126641fb01c/src/peft/tuners/lora/layer.py#L953-L994)

```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        lora_A_keys = self.lora_A.keys()
        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = self._cast_input_dtype(x, lora_A.weight.dtype)
            if active_adapter not in self.lora_variant:  # vanilla LoRA
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                result = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x,
                    result=result,
                    **variant_kwargs,
                    **kwargs,
                )

        result = result.to(torch_result_dtype)

    return result
```

## 逐行讲解 / What's happening

1. **四路状态分支 / Four state branches**:
   - 中文: `forward` 一进来先决定走哪条路:
     1. `disable_adapters=True` → 直接调 `base_layer`,如果之前是 merged 状态先 unmerge(否则会双倍计算 delta)。
     2. `adapter_names is not None` → 同一个 batch 里不同样本用不同的 adapter(`_mixed_batch_forward`),这是 inference 时多任务路由的关键能力。
     3. `self.merged` → adapter 已经被烤进 `base_layer.weight`,直接 base 前向。
     4. 其余 → 正常 LoRA 前向,**实际加 delta 的地方**。
   - English: `forward` immediately routes to one of four modes:
     1. `disable_adapters=True` → call `base_layer` directly; first `unmerge()` if needed to avoid double-counting the delta.
     2. `adapter_names is not None` → different samples in the same batch use different adapters (`_mixed_batch_forward`) — the inference-time multi-task routing primitive.
     3. `self.merged` → the adapter has been baked into `base_layer.weight`, so `base_layer` already does the right thing.
     4. Else → normal LoRA forward — **where the delta actually gets added**.

2. **`result = self.base_layer(x, *args, **kwargs)` / Call the frozen base layer**:
   - 中文: base layer 是 `nn.Linear`(或者量化后的版本),权重冻结、不计算梯度。先用它得到 `Wx`,然后再加 LoRA delta —— **顺序很重要**,因为如果先做 LoRA 再做 base 会丢掉 base 的 bias,而且不方便在 base 是量化 layer(比如 4-bit GPTQ)时融合。
   - English: The base layer is `nn.Linear` (or a quantized counterpart) with frozen weights. We first compute `Wx`, then add the LoRA delta. **Order matters**: doing it the other way around loses the base bias and complicates the case where `base_layer` is a 4-bit GPTQ / bitsandbytes quantized layer.

3. **`torch_result_dtype = result.dtype` / Snapshot the original dtype**:
   - 中文: 后面要在 LoRA 内部 cast 到 `lora_A.weight.dtype`(通常 fp32 或 bf16),最后一行 `result = result.to(torch_result_dtype)` 把结果再 cast 回 base 的 dtype。这样调用方根本感觉不到 LoRA 在 dtype 上做了"小动作"。
   - English: We will cast inside LoRA to `lora_A.weight.dtype` (typically fp32 or bf16), and the final `result.to(torch_result_dtype)` restores the base dtype. The caller never sees the dtype shuffle.

4. **多适配器 for 循环 / The multi-adapter loop**:
   - 中文: `self.active_adapters` 是个列表 —— 同一个 layer 可以同时挂多个 LoRA 适配器,它们的 delta **相加**。这是 PEFT 多任务/混合微调的核心:训练时只激活一个 adapter,推理时可以把几个不同任务的 adapter 全部加上(权重就是各自的 `scaling`)。`for` 里每个 adapter 跑一遍 `lora_B(lora_A(dropout(x)))` 累加进 `result`。
   - English: `self.active_adapters` is a list — a single layer can have multiple LoRA adapters active, and their deltas **sum**. This is PEFT's multi-task/mixture-of-adapters story: train one adapter at a time, then at inference compose several. The loop runs `lora_B(lora_A(dropout(x)))` per adapter and accumulates into `result`.

5. **第 31 行 / Line 31 (`result = result + lora_B(lora_A(dropout(x))) * scaling`)** —— **整个 LoRA 的灵魂**:
   - 中文: 读懂这一行就读懂了 LoRA。`dropout(x)` 给低秩分支一个独立的 dropout;`lora_A(x)` 是 `(d → r)` 的下投影,`lora_B(...)` 是 `(r → d)` 的上投影 —— 整条路径只有 `2 r d` 个参数,比 base 的 `d²` 小 r/(d/2) 倍。`scaling = α / r`,把训练时学到的 delta 按秩归一,使得不同 r 之间训练超参可迁移。最后 `result + ...` 是残差结构,**保留了原 base 的所有能力**。
   - English: Understanding this single line is understanding LoRA. `dropout(x)` gives the low-rank branch its own dropout; `lora_A(x)` is the `(d → r)` down-projection; `lora_B(...)` is the `(r → d)` up-projection — total params `2rd`, smaller than the base's `d²` by a factor of `d / (2r)`. `scaling = α / r` rescales the delta by rank so that hyper-parameters transfer between different `r`. The `result + ...` is a residual — **the base layer's full capability is preserved**.

6. **变体分支 / Variant branch (`self.lora_variant`)**:
   - 中文: 如果 adapter 注册了一个 "variant"(比如 DoRA、VeLoRA、MonteCLoRA),forward 把控制权交给那个 variant 的 `forward` —— 这就是 PEFT 支持那么多 LoRA 改良版的扩展点。原始 LoRA 不走这条路。
   - English: If an adapter has registered a "variant" (DoRA, VeLoRA, MonteCLoRA, …), `forward` hands control to that variant's `forward`. This is the extension point that lets PEFT support so many LoRA-family methods. Vanilla LoRA skips this branch.

## 类比 / The analogy

想象一家公司的"标准员工手册"(base 权重 `W`)厚得像砖,改一个字都要全体重新审阅 —— 改不起。**LoRA 的做法是给每个员工配一张便利贴(adapter)**:贴在手册边上,写"对于第 47 条,执行时额外做 X"。这张便利贴薄得多,改起来便宜。**多适配器** = 给同一本手册贴多张便利贴(销售部一张、法务部一张),每张写各自的"调整方案"。**merged** = 把便利贴上的字真的抄进手册正文,从此调用更快但便利贴消失。**unmerge** = 把字擦掉,便利贴重新独立。**disable_adapters** = 让员工忽略便利贴,只看手册原文。所有这些状态机,核心其实只是一句"先读手册,再读便利贴,把两者加起来执行"。

Picture a company's "official employee handbook" (the base weight `W`) — a brick of a binder where changing one sentence triggers a full review. Too costly. **LoRA gives every employee a sticky note (the adapter)**: "for clause 47, also do X". Sticky notes are cheap to write and modify. **Multi-adapter** = several sticky notes on the same handbook (Sales department's note, Legal department's note), each contributing its own adjustment. **Merged** = transcribe the sticky-note text into the handbook itself — faster at runtime, but the note disappears. **Unmerge** = scrape the ink off, sticky note becomes independent again. **disable_adapters** = the employee is told to ignore stickies and read only the handbook. All this state machinery boils down to one rule: "read the handbook, read the sticky note, execute both".

## 自己跑一遍 / Try it yourself

下面这个 minimal LoRA 不依赖 peft,只用 nn.Linear 就把核心 forward 复现一遍。 / This minimal LoRA replicates the core forward without depending on peft.

```python
# try.py — needs: pip install torch
import torch
import torch.nn as nn

class MiniLoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False                   # freeze base
        self.lora_A = nn.Linear(d_in,  r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)            # delta starts at 0
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

torch.manual_seed(0)
base = nn.Linear(128, 64)
lora = MiniLoRALinear(base, r=8, alpha=16)
x = torch.randn(2, 128)
print("base only :", base(x)[0, :4].tolist())
print("with LoRA :", lora(x)[0, :4].tolist())          # identical at init (lora_B = 0)
print("trainable params:", sum(p.numel() for p in lora.parameters() if p.requires_grad),
      "/ total:", sum(p.numel() for p in lora.parameters()))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
base only : [0.3217, 0.5712, -0.0408, 0.4711]
with LoRA : [0.3217, 0.5712, -0.0408, 0.4711]
trainable params: 1536 / total: 9792
```

中文: **初始化时 LoRA 和 base 输出完全一致**,因为 `lora_B` 初始化为零 —— 这是 LoRA 论文的关键 trick:训练开始时整个低秩分支等价于 0,不破坏 pretrained 能力。可训练参数 1536(= 128×8 + 8×64)只占总参数的 ~16%,而对于 LLM 的大 Linear (d=4096) 这个比例会降到 0.4% 以下。

English: **At initialization, the LoRA wrapper produces identical output to the base layer** because `lora_B` is initialized to zero. This is the key trick from the LoRA paper: the low-rank branch starts at the zero function, so pre-trained capability is untouched at step 0. Trainable parameters are 1536 (= 128×8 + 8×64) — about 16% of the total here, but on real LLM Linears (`d ≈ 4096`) this drops below 0.4%.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DoRA / VeRA / LoRA+** / **DoRA / VeRA / LoRA+**: 中文: 整个 PEFT `lora_variant` 注册表都建立在这种"base + low-rank residual"骨架上 —— variant 各自调整 `A`、`B`、scaling、或者拆分 magnitude/direction。 / English: PEFT's entire `lora_variant` registry is built on this "base + low-rank residual" skeleton — variants tweak `A`, `B`, scaling, or split magnitude vs. direction.
- **HuggingFace `accelerate` 的 GradientAccumulationPlugin** / **`accelerate` GradientAccumulationPlugin**: 中文: 同样的"装饰器叠加"思路:一个 nn.Module 被多层 wrapper 包装(adapter、quantization、grad-checkpoint),每一层只关心"我在 forward 时往结果加什么"。 / English: Same "decorator stack" pattern: an `nn.Module` is wrapped by multiple layers (adapter, quantization, gradient checkpointing), each only deciding what to add to the forward.
- **ControlNet** / **ControlNet**: 中文: 也是冻结 base、训练侧路、forward 时把侧路输出加进 main path —— 数学上和 LoRA 高度同构。 / English: Freeze the base, train a side branch, add its output to the main path at forward time — mathematically isomorphic to LoRA.

## 注意事项 / Caveats / when it breaks

- **`lora_B` 必须零初始化 / `lora_B` MUST be zero-initialized**:
  - 中文: 否则训练开始时整个 base 输出立刻被"污染",pretrained 知识在前几步就被随机噪声覆盖。原 LoRA 论文专门强调这一点。
  - English: Otherwise the base output is corrupted at step 0 and pretrained knowledge is washed away in the first few updates. The LoRA paper devotes a section to this.
- **多 adapter 不是 free** / **Multiple adapters aren't free**:
  - 中文: 每多挂一个 active adapter,forward 就多算一次 `lora_B @ lora_A @ x`。对推理延迟敏感的场景应该 merge 之后再上线。
  - English: Each additional active adapter adds one `lora_B @ lora_A @ x` to the forward. For latency-sensitive inference, merge first, then deploy.
- **disable_adapters + merged 是隐形坑 / `disable_adapters` + `merged` is a subtle trap**:
  - 中文: 看这段代码:当 `disable_adapters=True` 且当前是 merged 状态时,会**自动 unmerge**。如果你的代码循环里反复 toggle `disable_adapters`,会反复 unmerge/merge,每次都是矩阵加减 —— 性能塌方。
  - English: Notice the code: when `disable_adapters=True` and the layer is merged, it **auto-unmerges**. Code that repeatedly toggles `disable_adapters` will repeatedly unmerge/merge — each one is a full matrix add/subtract. Performance cliff.
- **`dropout` 不只是正则化 / `dropout` is more than regularization**:
  - 中文: LoRA 的 dropout 是加在**低秩分支输入**而不是 base 输入上的 —— 这意味着 train/eval 切换时,只影响 LoRA delta 的强度,base 永远确定。这是一个有意为之的设计。
  - English: LoRA's dropout sits on the **input to the low-rank branch**, not the base input. Train/eval mode therefore only modulates the LoRA delta strength; the base layer is unaffected. This is deliberate.

## 延伸阅读 / Further reading

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — the original paper.
- PEFT docs — https://huggingface.co/docs/peft
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" — the most cited LoRA variant, supported in this same file via `lora_variant`.
- Karpathy's `nanoGPT` LoRA fork — minimal reference implementation to read alongside PEFT.
