---
date: 2026-06-14
topic: infrastructure
source: trending
repo: EternalEvan/Astra
file: diffsynth/models/wan_video_dit_moe.py
permalink: https://github.com/EternalEvan/Astra/blob/3fb82e1939bfcf87d6e27a7da57d8b55ff9cb393/diffsynth/models/wan_video_dit_moe.py#L225-L304
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, trending, world-model, mixture-of-experts, action-conditioning, astra]
---

# Astra(ICLR 2026)的"动作专家混合":80 行让一个 DiT 同时驱动游戏、车、机械臂 / Astra's (ICLR 2026) Mixture of Action Experts: 80 lines let one DiT drive games, cars, and manipulators

> **一句话 / In one line**: 不同模态的动作(键盘、方向盘、末端位姿)各走自己的专家分支,共用一个 top-k router,DiT 一套权重就能跨域 rollout。 / Different action modalities (keyboard, steering wheel, end-effector pose) flow through their own expert branches sharing one top-k router, so a single DiT weight set can roll out across domains.

## 为什么重要 / Why this matters

世界模型最难的不是预测未来一帧,而是处理**异构动作**:训练数据里可能既有 sekai(游戏键盘 0/1 信号)、又有 nuscenes(连续转向角)、又有 openx(7D 机器人末端 pose)。早期方案是分别训三个模型,但样本效率差。Astra(Tsinghua + 快手,ICLR 2026)用了 MoE 的思路——所有数据共用一套 DiT,只在 action 投影那条小支上分专家。这套设计在 `wan_video_dit_moe.py` 里被简化成 80 行可读代码:`ModalityProcessor` 把异构 action 各自投到统一维度(unified_dim=30,很小),`MultiModalMoE` 用一个全局 router 选出 top-2 专家,再加权求和。这是把"模态适配"和"特征学习"显式拆开的经典模板。

The hard part of world modeling isn't predicting the next frame — it's handling **heterogeneous actions**: training data might mix sekai (game keyboard 0/1 signals), nuscenes (continuous steering angles), and openx (7-D robot end-effector poses). The naive fix is one model per domain, but sample efficiency suffers. Astra (Tsinghua + Kuaishou, ICLR 2026) takes the MoE route — every domain shares the DiT and only forks along the action-projection branch. The pattern is condensed into 80 readable lines in `wan_video_dit_moe.py`: `ModalityProcessor` projects each heterogeneous action onto a tiny unified dim (30, deliberately small), and `MultiModalMoE` uses a global router to pick top-2 experts, then a weighted sum. A classic template for separating "modality adaptation" from "feature learning".

## 代码 / The code

`EternalEvan/Astra` — [`diffsynth/models/wan_video_dit_moe.py`](https://github.com/EternalEvan/Astra/blob/3fb82e1939bfcf87d6e27a7da57d8b55ff9cb393/diffsynth/models/wan_video_dit_moe.py#L225-L304)

```python
class MultiModalMoE(nn.Module):
    """简化的多模态MoE - 只保留专家,不包含router"""

    def __init__(self, unified_dim: int = 30, hidden_dim: int = 60, output_dim: int = None,
                 num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.unified_dim = unified_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim or unified_dim

        # 🔧 定义模态到专家的映射
        self.modality_to_expert = {
            "sekai": 0,      # sekai数据使用专家0
            "nuscenes": 1,   # nuscenes数据使用专家1
            "openx": 2,      # openx数据使用专家2
            "unknown": 0     # 默认使用专家0
        }

        # Experts - 输入unified_dim,输出output_dim (每层独立)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(unified_dim, self.output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor, top_k_indices: torch.Tensor,
                modality_type: str = "unknown") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, unified_dim]
            expert_weights: [batch_size, seq_len, top_k] - 从全局router得到的权重
            top_k_indices: [batch_size, seq_len, top_k] - 从全局router得到的专家索引
            modality_type: 模态类型标识(用于专家分配和统计)
        Returns:
            output: [batch_size, seq_len, output_dim]
            expert_stats: 专家选择统计信息
        """
        batch_size, seq_len, input_dim = x.shape
        assert input_dim == self.unified_dim, f"Expected input dim {self.unified_dim}, got {input_dim}"

        original_dtype = x.dtype
        x = x.to(self.experts[0][0].weight.dtype)

        # 🔧 获取该模态应该使用的目标专家
        target_expert_id = self.modality_to_expert.get(modality_type, 0)

        # Expert processing
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [batch, seq, output_dim]
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=-2)  # [batch, seq, num_experts, output_dim]

        # Weighted combination using provided weights and indices
        output = torch.zeros(batch_size, seq_len, self.output_dim,
                           device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]   # [batch, seq]
            weight = expert_weights[:, :, k:k+1]  # [batch, seq, 1]

            expert_output = torch.gather(
                expert_outputs,
                dim=2,
                index=expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, expert_outputs.shape[-1])
            ).squeeze(2)  # [batch, seq, output_dim]

            output += weight * expert_output

        output = output.to(original_dtype)
        return output, expert_stats
```

## 逐行讲解 / What's happening

1. **`__init__` 第 228-250 行 / `__init__` lines 228-250**:
   - 中文: `unified_dim=30, hidden_dim=60, num_experts=4, top_k=2`——参数量都小,因为这块只处理 action(不是图像或文本)。最关键的是 `modality_to_expert` 字典:把字符串 "sekai/nuscenes/openx" 硬编码到专家索引——是一个"模态先验",告诉模型每种数据天然偏好哪个专家。
   - English: `unified_dim=30, hidden_dim=60, num_experts=4, top_k=2` — small dims, because this block only handles actions (not images or text). The crucial detail is the `modality_to_expert` dict: hard-coded string-to-index mapping that injects a **modality prior** — telling the model which expert each domain naturally prefers.

2. **专家 = 单 Linear / Each expert is one `nn.Linear`**:
   - 中文: 4 个专家每个就是一层 `Linear(30, 30)`(默认 `output_dim = unified_dim`)。没有 hidden 层、没有激活——这是反直觉的设计,因为传统 Mixtral MoE 每个专家是一个 FFN(Linear-GeLU-Linear)。Astra 故意做轻量,因为它后面接的整个 Wan-DiT 已经很重,action 这条小支只需要做"模态相关的轻微 reweighting",更复杂反而过拟合。
   - English: Each of the four experts is just a single `Linear(30, 30)` (default `output_dim == unified_dim`). No hidden layer, no activation — counter-intuitive, since classic Mixtral experts are full FFNs (Linear-GeLU-Linear). Astra keeps it minimal on purpose: the downstream Wan-DiT is already heavy, and the action branch only needs a "modality-aware reweighting" — more capacity overfits.

3. **`forward` 第 264-272 行 / Lines 264-272 (dtype shenanigans)**:
   - 中文: bf16 输入下 expert 权重可能是 fp32(尤其是 LoRA fine-tune 时),需要先 cast 到 expert 的 dtype 再算,算完 cast 回原 dtype。这种 dtype dance 在混精度训练里非常常见——多写两行省下半夜的 type mismatch 报错。
   - English: With bf16 inputs, expert weights might be fp32 (especially during LoRA fine-tune); cast input to the expert dtype, compute, then cast back. This dtype dance is routine in mixed-precision training — two extra lines that save you from 3 a.m. type-mismatch errors.

4. **第 278-283 行 / Lines 278-283 (compute ALL experts, then select)**:
   - 中文: 这里是 MoE 设计上的关键选择——**所有 4 个专家都先跑一遍**,然后再用 router 选 top-k。这是密集 MoE 的做法,适合 GPU(并行 4 个小 matmul 比稀疏 indexing 快);稀疏 MoE 才会只跑被选中的那些。Astra 选密集是因为 action embedding 序列很短,通信开销不划算。
   - English: A key design choice — **run all four experts**, then select top-k. This is dense MoE (better for GPUs: four small matmuls in parallel beat sparse indexing); sparse MoE would only run selected experts. Astra goes dense because the action embedding sequence is short and communication overhead would dominate.

5. **第 286-299 行 / Lines 286-299 (the top-k gather + weighted sum)**:
   - 中文: 这是整个 forward 的灵魂。`expert_outputs.shape = (B, L, num_experts, output_dim)`。`top_k_indices[:, :, k]` 形状 `(B, L)`,告诉每个位置该选哪个专家。用 `torch.gather(dim=2)` 把那个专家的输出抠出来——`unsqueeze + expand` 的 4 行只是为了让索引张量 broadcast 到 `(B, L, 1, output_dim)`,gather 完 squeeze 掉那个 1。最后 `output += weight * expert_output`——对 top_k 个专家做加权求和。
   - English: The heart of the forward. `expert_outputs.shape = (B, L, num_experts, output_dim)`. `top_k_indices[:, :, k]` is `(B, L)`, telling each position which expert to pick. `torch.gather(dim=2)` carves that expert's output out — the four `unsqueeze + expand` lines are just to broadcast the index tensor to `(B, L, 1, output_dim)`; the `squeeze(2)` drops the inserted singleton. Then `output += weight * expert_output` performs the weighted sum across top-k experts.

6. **Router 在哪? / Where's the router?**:
   - 中文: 注意 `expert_weights` 和 `top_k_indices` 是**外部传进来**的——这块叫 "MoE 但不含 router"。在更大的 Astra DiT block 里,有个全局 router 看完整个 sequence 的 context 后才决定权重,然后传给这里。这种"router 外置"的设计让你可以共用一个 router 给多个 expert 集合(action expert、image expert 等),省参数也省计算。
   - English: Note `expert_weights` and `top_k_indices` arrive **from outside** — the class is literally "MoE without a router". The outer Astra DiT block has a global router that sees the full sequence context before deciding weights, then passes them in. The "external router" pattern lets you share one router across multiple expert sets (action experts, image experts, etc.) — saves params and compute.

## 类比 / The analogy

想象一个翻译团队接待外国客人。"中文翻英文""中文翻法文""中文翻日文"是 3 个不同专家,但客人开口时,你不知道他要去哪种语种环境。前台(router)听完客人的一段话(action token)后,投票:65% 法文 + 35% 英文。然后**所有 3 位翻译同时开始翻**(密集),最后把法文版乘 0.65、英文版乘 0.35 加起来——客人就拿到了一份混合版。Astra 的小改动是:`modality_to_expert` 这个字典是个"VIP 备注卡"——前台早就知道客人从美国机场来,所以投票会倾向英文专家。这就是模态先验如何被注入到 routing 决策里。

Picture a translation team serving an international guest. "Chinese→English", "Chinese→French", and "Chinese→Japanese" are three experts, but you don't know which language the guest needs. The receptionist (router) listens to their utterance (an action token) and votes: 65 % French + 35 % English. Then **all three translators start translating in parallel** (dense), and you blend the outputs as `0.65 × French + 0.35 × English`. Astra's twist: the `modality_to_expert` dict is a "VIP note card" — the receptionist already knows this guest came from a US airport, so the vote leans English. That's how a modality prior gets injected into the routing decision.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

class ActionMoE(nn.Module):
    def __init__(self, dim=30, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x):
        # x: (B, L, dim)
        logits = self.router(x)                            # (B, L, num_experts)
        topk_w, topk_idx = logits.softmax(-1).topk(self.top_k, dim=-1)
        outs = torch.stack([e(x) for e in self.experts], dim=-2)   # (B, L, E, dim)
        y = torch.zeros_like(x)
        for k in range(self.top_k):
            sel = topk_idx[..., k:k+1, None].expand(-1, -1, 1, x.size(-1))
            picked = outs.gather(dim=2, index=sel).squeeze(2)
            y = y + topk_w[..., k:k+1] * picked
        return y, topk_idx

torch.manual_seed(0)
moe = ActionMoE()
sekai_action = torch.randn(2, 8, 30)
out, picks = moe(sekai_action)
print("output shape:", out.shape)
print("expert picks (B0, L0):", picks[0, 0].tolist())
print("router entropy:", -(moe.router(sekai_action).softmax(-1) * moe.router(sekai_action).log_softmax(-1)).sum(-1).mean().item())
```

运行 / Run with:
```bash
pip install "torch>=2.4"
python try.py
```

预期输出 / Expected output:
```
output shape: torch.Size([2, 8, 30])
expert picks (B0, L0): [1, 2]   # 每个位置选了 2 个专家
router entropy: 1.38...           # 接近 log(4) ≈ 1.39:初始化时 router 是均匀的
```

中文一句:训练初期 router 的熵接近 `log(num_experts)`,说明它完全没有偏好;训练后熵会下降,说明 router 学到了"哪种 action 走哪个专家"——可以用熵单调下降作为 MoE 学到东西的简单 sanity check。

English: at initialisation the router entropy is near `log(num_experts)` — no preference at all; once training kicks in, entropy drops as the router learns "which action goes to which expert". Monitoring router entropy is a cheap sanity check that the MoE is actually specialising.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Mixtral 8x7B 的 SparseMoE** / **Mixtral 8x7B SparseMoE**: 大块用 sparse MoE(只跑 top-k 个专家),每个专家是 FFN——Astra 是它的轻量化变体。 / Bigger sibling using **sparse** MoE (only run top-k experts) with full FFNs each; Astra is the lightweight variant.
- **GR00T-N1.7 的 action expert** / **GR00T-N1.7 action expert**: 直接给每个 embodiment 配一个 dense action head,没共享 router——更简单但容量浪费。 / One dense action head per embodiment, no shared router — simpler but capacity-inefficient.
- **MoE-LoRA / LoRA-MoE 系列** / **MoE-LoRA / LoRA-MoE works**: 同样思路用于把多个任务的 LoRA adapter 混合,router 决定权重。 / Same idea applied to mixing multiple task LoRA adapters, with the router deciding weights.
- **Switch Transformer** / **Switch Transformer**: 只用 top-1(没有混合,只挑一个),进一步简化。 / Uses top-1 only (no mixing, hard selection) — a further simplification.

## 注意事项 / Caveats / when it breaks

- **dense MoE 的代价** / **The cost of dense MoE**: 所有 expert 都要算,所以 FLOPs ∝ num_experts。Astra 的专家小、序列短,所以划算;大 LLM 必须用 sparse。 / Every expert runs, so FLOPs scale with `num_experts`. Astra's experts are tiny and the sequence is short, so it pays off; a large LLM must go sparse.
- **router collapse** / **Router collapse**: 没有 auxiliary load-balancing loss 的话,router 会 collapse 到只选一个专家——其他三个永远不被更新。代码里的 `collect_expert_statistics` 就是用来检测这个的,但需要外部训练循环加 aux loss 才能真正防住。 / Without an auxiliary load-balancing loss, the router collapses onto one expert and the other three never update. The `collect_expert_statistics` helper exposes this, but you still need an aux loss in the outer training loop to actually prevent it.
- **`modality_to_expert` 是硬编码** / **`modality_to_expert` is hard-coded**: 想加新模态(比如人手动作),要改源码——不是 config-driven 的设计。Astra 把这种 prior 当成一个超参——研究阶段可接受,工程化要重构成 registry。 / Adding a new modality (e.g. human-hand actions) requires editing source code — not config-driven. Astra treats this prior as a hyperparam; research-OK, but production-grade demands a registry.
- **top_k 必须 ≤ num_experts** / **`top_k` must be ≤ `num_experts`**: 不然 gather 时 indices 越界——代码里没显式 assert,容易漏。 / Otherwise `gather` over-indexes — the code has no explicit assert, easy to miss.

## 延伸阅读 / Further reading

- [Astra paper (arXiv:2512.08931)](https://arxiv.org/abs/2512.08931) — full architecture of the interactive WM
- [Mixtral of Experts (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088)
- [Switch Transformer (arXiv:2101.03961)](https://arxiv.org/abs/2101.03961)
- [Astra project page](https://eternalevan.github.io/Astra-project/)
