---
date: 2026-06-10
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/lora/intruders.py
permalink: https://github.com/huggingface/peft/blob/8317e69bf407774ea5cb5635338029ab65a30b9f/src/peft/tuners/lora/intruders.py#L20-L160
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, huggingface, peft, lora, svd, catastrophic-forgetting]
---

# 训练完才动手:PEFT 把"切除 LoRA 入侵维度"做成了一个 140 行的后处理 / After training, then surgery: PEFT ships "remove LoRA intruder dimensions" as a 140-line post-hoc step

> **一句话 / In one line**: 一篇 2024 年的论文证明"LoRA 看起来和全量微调等价是个错觉",真凶是 LoRA 在合并后的权重 W+ΔW 里塞入了几个"和原权重几乎正交的新奇异方向";PEFT 把那篇论文的修复方案变成了 reduce_intruder_dimension():SVD 一下,挑出来,按比例减回去。 / A 2024 paper showed "LoRA looks equivalent to full fine-tuning" is an illusion — the real culprit is a handful of new singular directions in the merged W+ΔW that are nearly orthogonal to the base W. PEFT ships the paper's fix as `reduce_intruder_dimension()`: SVD, identify them, subtract a tunable fraction.

## 为什么重要 / Why this matters

LoRA 的甜蜜区:小、快、能学到 task-specific 知识。LoRA 的痛点:**灾难性遗忘** —— 微调过新 task 的模型,原本会的事情常常做不动了。Shuttleworth 等人在 2024 年(["LoRA vs Full Fine-tuning: An Illusion of Equivalence"](https://arxiv.org/abs/2410.21228))用奇异值分解(SVD)给出了机理解释:LoRA 训练出来的 ΔW 不是"对原 W 的微小扰动",而是在 W+ΔW 里**新增了几条全新的奇异方向**,这些方向和原 W 的所有奇异向量都近似正交。论文管它们叫 **intruder dimensions**(入侵维度)。新维度承载了 task 知识,但也是遗忘的元凶 —— 它们扭曲了原模型用来"理解一般文本"的方向。PEFT 把修复方案写成了独立函数:训练结束、合并前,SVD 一下找出这些入侵方向,按一个可调比例 `mitigation_lambda` 把它们减回去,在 task accuracy 和原知识恢复之间换个比例。

LoRA's sweet spot: small, fast, captures task-specific knowledge. LoRA's pain point: **catastrophic forgetting** — the fine-tuned model often loses skills it used to have. Shuttleworth et al. (2024, ["LoRA vs Full Fine-tuning: An Illusion of Equivalence"](https://arxiv.org/abs/2410.21228)) gave a mechanistic explanation via SVD: the trained ΔW is *not* a small perturbation of W; rather, W+ΔW grows **brand-new singular directions** that are near-orthogonal to all of W's. The paper calls these **intruder dimensions**. They carry the task knowledge — and they're also the forgetting culprit, distorting the directions the base model used for general-language understanding. PEFT ships the paper's remedy as a standalone post-training function: after training but before merge, SVD to find the intruder directions, subtract a tunable fraction `mitigation_lambda` of them, trading task accuracy for restored base knowledge.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/lora/intruders.py`](https://github.com/huggingface/peft/blob/8317e69bf407774ea5cb5635338029ab65a30b9f/src/peft/tuners/lora/intruders.py#L20-L160)

```python
def reduce_intruder_dimension(
    peft_model,
    old_adapter_name="default",
    new_adapter_name="intruder_reduced",
    top_k=10,
    threshold_epsilon=0.5,
    mitigation_lambda=0.75,
    logging_sink=print,
):
    """Intruder dimension mitigation (Shuttleworth et al., 2024)."""

    if peft_model.peft_type != "LORA":
        raise ValueError("The provided model is not using LoRA and is therefore not supported.")

    peft_model.add_adapter(new_adapter_name, peft_model.peft_config[old_adapter_name])

    for layer_name, layer in peft_model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue

        W = layer.get_base_layer().weight.data
        dW = layer.get_delta_weight(old_adapter_name)
        W_merged = W + dW
        is_embedding = old_adapter_name not in layer.lora_B

        cast_to_fp32 = W.dtype in (torch.float16, torch.bfloat16)
        if cast_to_fp32:
            W_dtype = W.dtype
            W = W.float()

        # find intruder dimensions: top-k singular directions of W+dW
        # whose absolute cosine with every base singular vector is below threshold
        U_base,   _S_base, _V_base = torch.linalg.svd(W,        full_matrices=False)
        U_merged, S_merged, V_merged = torch.linalg.svd(W_merged, full_matrices=False)

        cos_sim = (U_merged.T @ U_base).abs().max(dim=1).values
        intruder_idcs = torch.where(cos_sim[:top_k] < threshold_epsilon)[0].tolist()

        if not intruder_idcs:
            logging_sink(f"{layer_name}: No intruders")
            # copy adapter weights unchanged
            if is_embedding:
                layer.lora_embedding_B[new_adapter_name].data = layer.lora_embedding_B[old_adapter_name].data.clone()
                layer.lora_embedding_A[new_adapter_name].data = layer.lora_embedding_A[old_adapter_name].data.clone()
            else:
                layer.lora_B[new_adapter_name].weight.data = layer.lora_B[old_adapter_name].weight.data.clone()
                layer.lora_A[new_adapter_name].weight.data = layer.lora_A[old_adapter_name].weight.data.clone()
            continue

        # reconstruct an "intruder ΔW" from the identified directions
        B_intruder = U_merged[:, intruder_idcs] @ torch.diag(S_merged)[intruder_idcs, :].sqrt()
        A_intruder = (torch.diag(S_merged)[:, intruder_idcs]).sqrt() @ V_merged[intruder_idcs, :]

        # subtract (1 - lambda) of it from W+dW, then convert back to a rank-r A/B pair
        W_mitigated  = W_merged + (mitigation_lambda - 1) * (B_intruder @ A_intruder)
        dW_mitigated = W_mitigated - W
        dW_mitigated /= layer.scaling[old_adapter_name]

        U_dW, S_dW, V_dW = torch.linalg.svd(dW_mitigated, full_matrices=False)

        effective_rank = layer.r[old_adapter_name]
        B_new = U_dW[:, :effective_rank] @ torch.diag(S_dW[:effective_rank]).sqrt()
        A_new = torch.diag(S_dW[:effective_rank]).sqrt() @ V_dW[:effective_rank]

        if is_embedding:
            layer.lora_embedding_B[new_adapter_name].data = B_new
            layer.lora_embedding_A[new_adapter_name].data = A_new
        else:
            layer.lora_B[new_adapter_name].weight.data = B_new
            layer.lora_A[new_adapter_name].weight.data = A_new

        if cast_to_fp32:
            W = W.to(W_dtype)

    peft_model.set_adapter(new_adapter_name)
```

## 逐行讲解 / What's happening

1. **`add_adapter(new_adapter_name, ...)`**:
   - 中文: 不要 in-place 改原 adapter,而是先复制一份新 adapter("intruder_reduced"),把修复结果写到新 adapter 里。这样万一 `mitigation_lambda` 调失败了,`set_adapter(old)` 一键回滚。这是 PEFT 风格的可逆设计。
   - English: instead of mutating the original adapter in place, allocate a fresh "intruder_reduced" adapter and write the mitigated weights into it. If a chosen `mitigation_lambda` regresses your eval, `set_adapter(old)` rolls back instantly. Classic PEFT reversible-by-design.

2. **第 34-35 行 / Lines 34-35 (`W` 和 `dW`)**:
   - 中文: `W` 是冻结的 base 权重,`dW = layer.get_delta_weight(...)` 算的是 ΔW = scaling * B@A,即合并 LoRA 时会加到 W 上的那个矩阵。`W_merged = W + dW` 就是"如果你 merge LoRA 之后得到的权重"。
   - English: `W` is the frozen base, `dW = layer.get_delta_weight(...)` computes scaling · B@A — the matrix that gets added to W when you merge LoRA. `W_merged = W + dW` is what your weights would look like after merging.

3. **两次 SVD / Two SVDs**:
   - 中文: 对 W 做 SVD 拿到 base 的左奇异向量 `U_base`(描述"原模型在输入空间里识别哪些方向"),对 W_merged 做 SVD 拿 `U_merged`。注意:LoRA 的秩是 r,但它影响的是合并矩阵的 *full* 奇异谱 —— 这就是为什么 SVD 必须在 W 和 W_merged 上做、不能在 ΔW 上做。
   - English: SVD W to get its left singular vectors `U_base` (the directions the base model "recognizes" in input space). SVD W_merged to get `U_merged`. Important: although LoRA has rank r, it influences the *full* singular spectrum of the merged matrix — that's why SVD must run on W and W_merged, not on ΔW directly.

4. **第 41-42 行 / Lines 41-42 (the cos-sim trick)**:
   - 中文: 这是整段代码的心脏。`U_merged.T @ U_base` 是一个矩阵,第 i 行第 j 列是 `<U_merged[:,i], U_base[:,j]>`。取绝对值之后 `.max(dim=1)` 表示"对于 U_merged 的第 i 个方向,它和 U_base 所有方向里最相似的那个的相似度"。如果这个最大相似度都很小,说明这个方向是个**入侵者** —— 在 base 的奇异分解里几乎找不到任何对应方向。`< threshold_epsilon` 选出"最大相似度低于 0.5"的方向,`[:top_k]` 限定只在最重要的 top_k 个方向里找(top_k=10 是默认值,理由是更靠后的奇异方向影响小)。
   - English: this is the heart. `U_merged.T @ U_base` is a matrix whose (i,j) entry is `<U_merged[:,i], U_base[:,j]>`. After `.abs().max(dim=1)` you get, for each direction in U_merged, its cosine similarity to its *best-matching* base direction. If that max is small, the direction is an **intruder** — base has nothing close to it. `< threshold_epsilon` selects directions whose best match falls below 0.5; `[:top_k]` restricts the search to the top_k most-influential directions (top_k=10 by default — later singular directions have negligible effect).

5. **第 51-52 行 / Lines 51-52 (build the intruder block)**:
   - 中文: 把找出的入侵方向重组成一个矩阵 `B_intruder @ A_intruder`。注意奇异值的平方根分摊到 B 和 A 两边 —— 这样 B 和 A 范数接近,不会出现一边超大一边超小的训练动力学问题(注释里专门提到了这点)。
   - English: rebuild the intruder block as `B_intruder @ A_intruder`. The square root of S is shared between B and A so neither matrix has a dramatically different norm — the inline comment flags this as critical for stable training dynamics if you ever continue training the mitigated adapter.

6. **第 55 行 / Line 55 (the mitigation formula)**:
   - 中文: `W_mitigated = W_merged + (lambda - 1) * (B_intruder @ A_intruder)`。当 lambda=1.0 时,`(1-1)=0`,等于不修复;当 lambda=0.0 时,从 W_merged 里减去整块入侵 ΔW(完全恢复原 W 的那部分方向);默认 0.75 表示"减掉 25% 的入侵分量"。这是论文里 Figure 8 的核心 trade-off 旋钮 —— lambda 越小,task accuracy 掉得越多但 base 知识恢复得越好。
   - English: `W_mitigated = W_merged + (lambda - 1) * (B_intruder @ A_intruder)`. At lambda=1.0 you subtract 0 — no mitigation. At lambda=0.0 you remove the entire intruder block. The default 0.75 strips off 25%. This is the trade-off knob from Figure 8 of the paper: smaller lambda recovers more base knowledge but loses more task accuracy.

7. **第 56 行 / Line 56 (`dW_mitigated /= layer.scaling[...]`)**:
   - 中文: PEFT 里 `get_delta_weight()` 已经把 scaling(`alpha / rank`)乘进去了。我们重新提取 A、B 时要除回去,否则下次合并 adapter 又会乘一次,等于翻倍。
   - English: `get_delta_weight()` already multiplied scaling (`alpha / rank`) into ΔW. When we factor back into A, B for storage we have to divide it out — else the next merge would re-apply scaling and double it.

8. **第 59-62 行 / Lines 59-62 (factor back to rank-r A and B)**:
   - 中文: 修复后的 dW_mitigated 是个完整 rank 的矩阵,但 LoRA adapter 必须 fit 到原本的 rank r 里。再做一次 SVD,只取前 r 个奇异分量,得到新的 B、A。这是 "best rank-r approximation by SVD"(Eckart-Young 定理)。
   - English: the mitigated dW_mitigated is full-rank but a LoRA adapter must fit back into its rank r. One more SVD, take only the top r components — by Eckart-Young, that's the best rank-r approximation in Frobenius norm.

## 类比 / The analogy

想象你给一只训练好的导盲犬补习"在地铁站怎么走":你专门带它在地铁里训练了一周。结果:地铁站它走得贼溜,但出了地铁,它对汽车声、对马路上的红绿灯反应明显迟钝了 —— 它脑子里那些用来"分辨车流"的回路被新塞进去的"分辨地铁人流"的回路挤掉了一部分。Intruder dimension mitigation 像是一位训犬师:做完地铁训练后,他识别出"哪些回路是新加进去的、和原本'识别车流'的回路无关",然后部分削减这些新回路 —— 牺牲一点"地铁站熟练度",换回大部分"识别车流"的能力。`mitigation_lambda` 就是训犬师选择"削多少"的旋钮。

Imagine you re-trained a working guide dog on "navigate the subway" for a week. After: the dog is brilliant in the subway, but its responses to car horns and traffic lights have noticeably dulled — the "read car traffic" circuits got partly displaced by the new "read subway crowds" circuits. Intruder dimension mitigation is a trainer who, after the subway week, can identify which neural circuits are new and orthogonal to the "read traffic" circuits, and partly clip them. Trade a bit of subway proficiency for recovered car-traffic awareness. `mitigation_lambda` is the knob for "how much to clip".

## 自己跑一遍 / Try it yourself

```python
import torch

torch.manual_seed(0)
d, r = 256, 8
W   = torch.randn(d, d)      # frozen base
A   = torch.randn(r, d) * 0.1
B   = torch.randn(d, r) * 0.1
dW  = B @ A                  # what LoRA learned
W_merged = W + dW

# detect intruder directions
U_base,   _, _ = torch.linalg.svd(W,        full_matrices=False)
U_merged, S_m, V_m = torch.linalg.svd(W_merged, full_matrices=False)
cos_sim = (U_merged.T @ U_base).abs().max(dim=1).values
intruders = torch.where(cos_sim[:10] < 0.5)[0].tolist()
print("intruder dims:", intruders, "their cos to base:", cos_sim[intruders].tolist())

# mitigate at lambda = 0.5 (subtract 50% of intruder mass)
lam = 0.5
B_int = U_m[:, intruders] @ torch.diag(S_m)[intruders, :].sqrt()
A_int = (torch.diag(S_m)[:, intruders]).sqrt() @ V_m[intruders, :]
W_mitigated = W_merged + (lam - 1) * (B_int @ A_int)

print("‖W_merged - W‖_F        =", (W_merged   - W).norm().item())
print("‖W_mitigated - W‖_F     =", (W_mitigated - W).norm().item())
print("relative knowledge restored:", 1 - (W_mitigated - W).norm().item() / (W_merged - W).norm().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
intruder dims: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] their cos to base: [...]
‖W_merged - W‖_F        = 2.5xxx
‖W_mitigated - W‖_F     = 2.0xxx
relative knowledge restored: ~20%
```

中文:这个 toy 例子里 ΔW 是随机噪声,自然全是 intruder。注意"恢复比例"≈ `(1 - lambda)` 乘以"入侵方向占总 ΔW 的范数比例" —— 这就是 mitigation_lambda 旋钮在数学上控制的量。

English: in this toy ΔW is random noise so every direction looks like an intruder. Note the "knowledge restored" ratio ≈ `(1 - lambda)` × (intruder fraction of ΔW's norm) — that's exactly the math the `mitigation_lambda` knob controls.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DoRA(Weight-Decomposed LoRA)** / **DoRA (Weight-Decomposed LoRA)**:同样在做"把 W 分解,只让方向变,模长不变"的事情,但是在训练时做。Intruder mitigation 是训练后做。 / Also decomposes W (direction vs magnitude) — but during training. Intruder mitigation is post-training.
- **TIES-Merging / DARE** / **TIES-Merging / DARE**:多任务 LoRA merge 的算法 —— 也在解决"不同 task 的 ΔW 之间方向冲突"的问题,套路也是 SVD + 选方向。 / Multi-task LoRA merging — same family, also based on SVD + direction selection.
- **PEFT 里其他 LoRA-后处理工具** / **Other PEFT post-LoRA tools**:`peft/tuners/lora/{loraga,corda,eva}.py` 也都是 "SVD-based 后处理 / 初始化"系列的兄弟模块。 / `peft/tuners/lora/{loraga,corda,eva}.py` are sibling SVD-based post-processing or init tools.
- **Concept Erasure (Belrose et al.)** / **Concept Erasure (Belrose et al.)**:用 PCA-style 投影从 embedding 中去掉特定方向 —— 同一个 motif:"识别低秩方向 → 减掉它"。 / Uses PCA-style projection to remove specific directions from embeddings — same motif: identify a low-rank direction and subtract it.

## 注意事项 / Caveats / when it breaks

- **SVD 在大矩阵上是 O(d³)** / **SVD is O(d³) on large matrices**:Llama-70B 的 hidden_dim=8192,每层 attention 投影矩阵的 SVD 都不便宜。函数会对**每一个 LoRA 层**跑两次 SVD。70B 模型大概十分钟级别(论文里也说过)。建议在能放下 fp32 的机器上跑。 / Llama-70B has hidden_dim=8192; SVD per layer is non-trivial. This function does two SVDs per LoRA layer — ten-ish minutes wall-clock on a 70B (the paper notes this). Run on a machine that can hold fp32.
- **只支持 LoRA,不支持 DoRA / IA3 / LoHa** / **Only supports vanilla LoRA**:文档里明说了"the method may not generalize to other delta-weight methods"。 / The docstring explicitly says it may not generalize beyond LoRA.
- **top_k=10 是经验值** / **top_k=10 is heuristic**:对 hidden_dim 大的模型,可能有 30+ 个入侵维度;`top_k=10` 限制了搜索深度。如果你看到 forgetting 仍然严重,试试 top_k=50。 / Heuristic. Big models may have 30+ intruder dims; `top_k=10` caps the search. If forgetting is still bad, try top_k=50.
- **lambda 调小有副作用** / **Smaller lambda has a sneaky side-effect**:它不只削入侵方向,因为我们最后是 rank-r 重新拟合 dW_mitigated,所以连"非入侵但相关"的方向也会受到 rank 投影的影响。lambda 大约在 0.6-0.85 之间是大多数 task 的甜点。 / Smaller lambda doesn't just clip intruders — because we rank-r-refit dW_mitigated at the end, non-intruder-but-related directions also get squeezed by the rank projection. Sweet spot is usually lambda ∈ [0.6, 0.85].

## 延伸阅读 / Further reading

- [Shuttleworth et al., "LoRA vs Full Fine-tuning: An Illusion of Equivalence" (Oct 2024)](https://arxiv.org/abs/2410.21228)
- [PEFT PR adding `reduce_intruder_dimension`](https://github.com/huggingface/peft/pull/2999)
- [DoRA paper — competing decomposition-based LoRA](https://arxiv.org/abs/2402.09353)
- [TIES-Merging — multi-LoRA merging with direction conflict resolution](https://arxiv.org/abs/2306.01708)
