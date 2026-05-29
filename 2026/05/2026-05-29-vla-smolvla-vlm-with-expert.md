---
date: 2026-05-29
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/smolvla/smolvlm_with_expert.py
permalink: https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L65-L146
difficulty: intermediate
read_time: ~14 min
tags: [code-of-the-day, vla, smolvla, vlm-with-expert, cross-attention]
build_role: VLM backbone wiring + action expert head
---

# SmolVLA 怎么把一个 VLM 和一个"小专家"绑在一起 / How SmolVLA stitches a frozen VLM onto a slim action expert

> **一句话 / In one line**: `SmolVLMWithExpertModel.__init__` 用 80 行干完三件事：加载冻结的 SmolVLM2、深拷贝它的 text config 再缩小一半作为 action expert、最后把 expert 每层的 k_proj/v_proj 改成跨过来读 VLM 的 KV。 / In 80 lines, this constructor loads SmolVLM2 (frozen perceiver), deep-copies its text config and shrinks it by `expert_width_multiplier` to become the trainable action expert, and finally rewires each expert layer's `k_proj`/`v_proj` so it can cross-attend into the bigger VLM.

## 为什么重要 / Why this matters

"VLM + 小专家"是 2025-2026 这一波 VLA 的标准架构：π0、π0.5、π0-FAST、Helix、SmolVLA、GR00T 全是这个套路 —— 大基座只做感知 + 语义理解，永远冻着；旁边挂一个又小又快的 transformer 专门吐 action token，可以 flow matching、可以 BC、可以 discrete bin。问题是这个"挂在旁边"具体怎么挂？层数怎么对齐？hidden size 不一样怎么 cross-attention？层共享 KV 还是层独立 KV？SmolVLA 把这套接线方式用一个构造函数原原本本写出来了 —— 你做 nanoVLA 时基本可以照搬。

The "frozen VLM + slim action expert" pattern is the dominant VLA architecture in 2025-2026 — π0, π0.5, π0-FAST, Helix, SmolVLA, GR00T all share it. The big VLM acts purely as a perception/semantics tower (frozen) while a small fast transformer next door produces action tokens via flow matching, BC, or discrete bins. The hard part is the wiring: how many expert layers, how to align them with VLM layers, how to cross-attend when the hidden sizes differ, whether the expert shares the VLM's KV cache. SmolVLA's constructor lays the entire recipe out in one readable block — copy-paste-ready for your own nanoVLA.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/smolvla/smolvlm_with_expert.py`](https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L65-L146)

```python
def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
    ):
        super().__init__()
        require_package("transformers", extra="smolvla")
        if load_vlm_weights:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype="bfloat16", low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)   # default: hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None
        ...
        self.set_requires_grad()
```

## 逐行讲解 / What's happening

1. **加载并可裁剪 VLM / Load + optionally truncate VLM (lines 88-102)**:
   - 中文：`from_pretrained` 用 bf16 + `low_cpu_mem_usage=True` 把 SmolVLM2-500M 加载进来。如果 `num_vlm_layers > 0`，直接把 `text_model.layers` 切片 —— SmolVLA 默认只保留前 16 层而不是 24 层，因为后几层对视觉语言对齐没用，但会拖慢推理。
   - English: load SmolVLM2 in bf16 with low-mem init. If `num_vlm_layers` is set, the text decoder is truncated by Python slicing — SmolVLA's default keeps only the early layers since late layers are tuned for generation, not perception.

2. **复制配置，缩小一半 / Deep-copy config, shrink by `expert_width_multiplier` (lines 106-110)**:
   - 中文：`copy.deepcopy(config.text_config)` 是关键 —— expert 跟 VLM 在 **架构上是同款**（同样的 norm、同样的 RoPE 配置、同样的 head_dim），但 `hidden_size` 被乘了 0.5，`intermediate_size` 重新算（用 LLaMA 那套 `2/3 * 4 * d`），`num_hidden_layers` 默认等于 VLM 层数 —— 这样 expert 和 VLM 可以一层一层对齐做 cross-attention。
   - English: `copy.deepcopy(config.text_config)` is critical — the expert inherits the VLM's architecture (norm, RoPE, head_dim) but its `hidden_size` is multiplied by 0.5 and the `intermediate_size` is recomputed via the LLaMA `2/3 * 4 * d` rule. `num_hidden_layers` defaults to the VLM's layer count so the two stacks align 1:1 for cross-attention.

3. **从空 config 造 expert / Build expert from config, no weights (line 116)**:
   - 中文：`AutoModel.from_config(lm_expert_config)` 只构造网络结构，所有参数随机初始化 —— 因为 expert 是要训的，VLM 是要冻的。
   - English: `AutoModel.from_config` builds the architecture with random init — exactly what you want, because the expert is the only thing being trained.

4. **改写 k_proj / v_proj 做 cross-attention / Rewrite k_proj/v_proj for cross-attention (lines 120-134)**:
   - 中文：这是最妙的一步。expert 原本的 self-attention，k_proj 输入维度是 `expert.hidden_size`，但我们要让它读 VLM 的 hidden state，所以把 `k_proj` 和 `v_proj` 重新换成 `nn.Linear(vlm_hidden, expert_hidden)`。这样 expert 的每个 query 可以"伸手"去拉 VLM 在同一层 layer_idx 上的 hidden state。`q_proj` 保留，因为 query 还是 expert 自己的。
   - English: this is the prettiest step. Each expert layer's `k_proj`/`v_proj` is **replaced** so its input dimension equals the VLM's hidden size (not the expert's). That way, every expert layer can pull KV from the VLM at the same layer index. `q_proj` is untouched because the query stays expert-local. The `self_attn_every_n_layers` knob skips the rewrite on certain layers so they revert to ordinary self-attention.

5. **去掉 expert 的 token embedding / Strip the expert's `embed_tokens` (line 136)**:
   - 中文：expert 不会从 token id 出发，它的输入是 action token / observation 的 embedding，所以 `embed_tokens` 直接置 None，省掉这块参数。
   - English: the expert never sees token IDs — its inputs are action/observation embeddings produced upstream — so `embed_tokens` is nulled out and removed from the parameter count.

6. **`get_intermediate_size` 是 LLaMA 公式 / The classic LLaMA FFN width rule**:
   - 中文：`int(2/3 * 4 * d)` 然后向上取整到 256 的倍数 —— LLaMA 用这个公式来让 FFN 大约比 hidden 大 8/3 倍，同时对齐到硬件友好的整数倍。expert 缩小时要保持这个比例。
   - English: the LLaMA convention — `int(2/3 * 4 * d)` rounded up to a multiple of 256 — ensures the FFN width stays ~8/3× hidden and is hardware-aligned. The expert keeps the same ratio after shrinking.

## 类比 / The analogy

把 VLM 想成一位资历很深的视觉顾问，专门看图、读说明书、给出结论但不动手；action expert 是一位敏捷的实习生，反应快、动作敏捷，但什么都不懂。每次执行任务前实习生（query）跑去顾问的工位（VLM 的 hidden state）问一句"你看到啥了？怎么干？"，顾问把答案直接交给他（KV），然后实习生根据答案立刻做出动作。`expert_width_multiplier=0.5` 是说"实习生不必跟顾问一样宽肩膀"，但他俩的工位编号是一一对应的（层数一致），方便实习生每层都能找到对应的顾问。

Imagine the VLM as a senior visual consultant — slow, thoughtful, and never picks up a tool. The action expert is a nimble intern: fast hands but no domain knowledge. Before each move the intern (query) sprints to the consultant's desk (VLM hidden state) and asks "what do you see, what should I do?". The consultant hands over a note (KV) and the intern acts immediately. The `expert_width_multiplier=0.5` is the intern wearing a slimmer jacket than the consultant; the matching layer count is the requirement that the intern always knows which consultant's desk to visit at each step.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文：在 nanoVLA 里这就是 `VLMWithExpert.__init__` 那一档 —— 是整个仓库的"骨架"。上游是 vision tower + tokenizer（昨天讲的 nanoVLM modality projector 就插在 VLM 前面），下游是 action head（discrete bin 或 flow-matching head）。输入是 `pixel_values + input_ids + action_token_embeddings`，输出是 expert 每一层的 hidden state，最后一层送进 action head。如果省掉 expert、直接拿 VLM 输出最后一层 hidden 接 action head 也能跑，但你会发现：(1) 不能冻 VLM 否则模型不学新动作；(2) 冻了 VLM 又会破坏它已有的视觉语言能力。expert 这一层中间件就是为了同时获得这两点。生产级实现还要加上：`set_requires_grad()` 里的精细 freeze（最后两层 VLM 也要解冻一点，否则 cross-attention 在新动作上对不齐）、bf16 + fp32 master-weight、KV cache 共享（VLM 的 KV 在多个 action chunk 上复用，是延迟从 80ms 降到 15ms 的关键）。

English: in nanoVLA this is the file that holds the whole architecture together. Upstream sit a vision tower and tokenizer (yesterday's nanoVLM pixel-shuffle projector slots in just before the VLM); downstream sits an action head (discrete bins or a flow-matching head). The inputs are `pixel_values + input_ids + action_token_embeddings`; the outputs are per-layer hidden states from the expert, the last of which feeds the action head. If you skipped the expert and fed the VLM's last hidden state directly into an action head, you would discover two problems: (a) you cannot freeze the VLM, because then the model learns nothing about actions, and (b) if you unfreeze the VLM, you destroy the visual-language capability you paid for. The expert exists precisely so you can keep both. A production version layers on a thoughtful `set_requires_grad` (the last two VLM layers usually need a sliver of training so the new cross-attention can align), bf16 with fp32 master weights, and — crucially — KV cache sharing so VLM KV can be reused across multiple action chunks; that is the trick that takes inference from 80 ms to 15 ms per step.

## 自己跑一遍 / Try it yourself

```python
# pip install torch transformers
# minimal "VLM + expert" wiring against a tiny dummy VLM
import copy
import torch
import torch.nn as nn

class DummyVLM(nn.Module):
    """Stand-in for SmolVLM: a stack of TransformerEncoderLayer."""
    def __init__(self, hidden=512, layers=4, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden, heads, batch_first=True)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(layers)])
        self.hidden = hidden

def build_expert(vlm: DummyVLM, width_mult: float = 0.5):
    """Mirror SmolVLA's __init__ at toy scale."""
    expert_hidden = int(vlm.hidden * width_mult)
    expert_layer = nn.TransformerDecoderLayer(expert_hidden, nhead=4, batch_first=True)
    expert = nn.ModuleList([copy.deepcopy(expert_layer) for _ in range(len(vlm.layers))])
    # rewrite k_proj/v_proj to take VLM hidden as input
    for blk in expert:
        blk.multihead_attn.kdim = vlm.hidden
        blk.multihead_attn.vdim = vlm.hidden
        blk.multihead_attn.k_proj_weight = nn.Parameter(torch.randn(expert_hidden, vlm.hidden) * 0.02)
        blk.multihead_attn.v_proj_weight = nn.Parameter(torch.randn(expert_hidden, vlm.hidden) * 0.02)
    return expert, expert_hidden

vlm = DummyVLM(hidden=512, layers=4)
for p in vlm.parameters():       # freeze VLM
    p.requires_grad = False
expert, h = build_expert(vlm)
print(f"VLM hidden={vlm.hidden}, expert hidden={h}")
print(f"trainable params: {sum(p.numel() for p in expert.parameters() if p.requires_grad)}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
VLM hidden=512, expert hidden=256
trainable params: ~3M
```

中文：注意"可训参数"只计 expert 的 —— VLM 的 ~30M 参数全部冻结，这是 SmolVLA 默认设置。

English: only the expert counts as trainable — the VLM's ~30 M parameters stay frozen, which is SmolVLA's default `train_expert_only=True`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi (π₀)** / **openpi (π₀)**: 中文 — π₀ 也是"PaliGemma + flow-matching expert"，思路完全一致，只是 expert 用 flow matching 出连续 action（昨天讲过 openpi 的 flow matching loss）。 / English — π₀ pairs a frozen PaliGemma with a flow-matching expert; identical wiring philosophy, but the expert produces continuous actions via flow matching rather than discrete tokens.
- **openvla (原版)** / **openvla (the original)**: 中文 — 反向案例：openvla 没有 expert，直接 fine-tune 整个 7B VLM 输出 discrete action token。效果好但训练/推理都重。SmolVLA 是为了把这条路压成消费级 GPU 而提出的。 / English — the original OpenVLA skipped the expert and fine-tuned the whole 7 B VLM to output discrete action tokens. Great quality, but heavy. SmolVLA exists specifically to compress that into a consumer-GPU workload.
- **GR00T N1** / **GR00T N1**: 中文 — Isaac-GR00T 也是 VLM + expert（DiT 风格的 flow-matching head），加上昨天讲的 `CategorySpecificLinear` 做多本体。 / English — Isaac-GR00T runs the same VLM+expert split but its expert is a DiT-style flow-matching head, and it adds today's `CategorySpecificLinear` for multi-embodiment routing.
- **lerobot/policies/groot/** / **lerobot/policies/groot/**: 中文 — 同一个仓库里另一份实现，可以对比着读，结构高度一致。 / English — the same lerobot repo has a Groot port; reading it side by side is the fastest way to see the family resemblance.

## 注意事项 / Caveats / when it breaks

- **expert 层数必须整除 VLM 层数** / **`num_expert_layers` must divide `num_vlm_layers`**: 中文 — 否则 cross-attention 对不齐 layer index。`assert ... % num_expert_layers == 0` 就是干这事。 / English — otherwise cross-attention has no canonical VLM layer to attend to. The `assert` enforces it.
- **冻 VLM 但解冻最后几层是常见配方** / **Freeze the VLM but thaw the last 1-2 layers**: 中文 — `set_requires_grad` 默认会解冻 VLM 的最后一两层，因为 expert 跨过来读它们的 KV，完全冻死会让对齐永远学不到新动作。 / English — `set_requires_grad` (called at the end of init) explicitly unfreezes the final 1-2 VLM layers when `train_expert_only=False`, because the expert reads their KV; leaving them stone-cold blocks any new-action alignment from forming.
- **bf16 全程，但 grad scaler 慎用** / **bf16 throughout, no GradScaler**: 中文 — 整个网络是 bf16 加载的，bf16 自带足够的动态范围，不要再叠 fp16 的 GradScaler，否则 LR 会被打成 0。 / English — the whole stack is bf16; bf16's exponent range removes the need for a `GradScaler`. Adding one (out of fp16 habit) zeros your LR.

## 延伸阅读 / Further reading

- [SmolVLA blog post](https://huggingface.co/blog/smolvla)
- [π₀ paper — Physical Intelligence's foundation policy](https://arxiv.org/abs/2410.24164)
- [Helix — the Figure VLM-with-expert layout](https://www.figure.ai/news/helix)
- [lerobot policies index](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies)
