---
date: 2026-06-25
topic: infrastructure
source: trending
repo: physical-superintelligence-lab/Psi0
file: src/psi/models/psi0.py
permalink: https://github.com/physical-superintelligence-lab/Psi0/blob/c7bdd8421f517f003dc51bafb8a8d4ab64b3fb73/src/psi/models/psi0.py#L308-L420
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, vla, humanoid, joint-attention, sd3, diffusion-transformer, psi0]
---

# Psi0 的 SD3 风格联合自注意力：动作 token 和 VLA token 共享一个注意力块 / Psi0's SD3-Style Joint Self-Attention: Action Tokens and VLA Tokens Share One Attention Block

> **一句话 / In one line**: `JointVLAAttnProcessor` 把动作流（noisy actions）和 VLA 流（视觉语言 token）的 Q/K/V 分别投影后在序列维度拼接，一次 `scaled_dot_product_attention` 调用完成联合注意力，再按原始长度拆分输出——这是 SD3 的联合注意力思路在机器人 VLA 上的直接移植。 / `JointVLAAttnProcessor` projects Q/K/V for both the action stream (noisy actions) and the VLA stream (vision-language tokens) separately, concatenates them along the sequence axis, runs one joint `scaled_dot_product_attention`, then splits the output back by original stream lengths — SD3's joint attention design ported directly to robot VLA.

## 为什么重要 / Why this matters

Psi0 是 RSS 2026 发布的人形机器人 VLA（2663 stars，Apache 2.0），架构是：Qwen3-VL-2B（视觉语言主干，冻结）+ 约 500M 参数的多模态扩散 Transformer 动作头。动作头的核心设计挑战是：如何让带噪动作 token 高效地"看到"语言和视觉上下文？

标准做法是独立的交叉注意力层（动作 tokens 作为 query，VLA tokens 作为 key/value）。Psi0 选择了 SD3 论文（Stable Diffusion 3）中提出的**联合自注意力**：在同一个注意力块里，两个流各自投影出 Q/K/V，然后在序列维度拼接（动作 Q ++ VLA Q，动作 K ++ VLA K，动作 V ++ VLA V），做一次联合 SDPA，再把输出切回两个流。这样动作 token 可以在单次 forward 中双向关注 VLA 上下文（而不只是 query to key-value），且不需要额外的交叉注意力参数。作者表明这只需 80 条机器人演示就能微调成功——联合注意力让小数据量的 VLA 微调变得高效。

Psi0 (RSS 2026, 2.7k stars) is a humanoid robot VLA built on a frozen Qwen3-VL-2B backbone and a ~500M-param diffusion transformer action head. The `JointVLAAttnProcessor` is the architectural core: instead of a separate cross-attention layer (action queries → VLA keys/values), it implements SD3-style joint attention where both streams project their own Q/K/V, concatenate along the sequence axis, run one SDPA call, then split. This bidirectional coupling lets action tokens attend to language/vision context and vice versa in a single block, without additional cross-attention parameters. The design enables fine-tuning on as few as 80 robot demonstrations.

## 代码 / The code

`physical-superintelligence-lab/Psi0` — [`src/psi/models/psi0.py`](https://github.com/physical-superintelligence-lab/Psi0/blob/c7bdd8421f517f003dc51bafb8a8d4ab64b3fb73/src/psi/models/psi0.py#L308-L420)

```python
class JointVLAAttnProcessor:
    """Attention processor for SD3-like joint self-attention for VLA action heads."""

    def __call__(
        self,
        attn,                                          # Attention module (holds projections)
        hidden_states,                                 # action stream: [B, S_action, D]
        encoder_hidden_states=None,                    # VLA stream:    [B, S_vla, D]
        attention_mask=None,                           # optional bool mask over VLA tokens
        *args, **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # --- Action stream projections ---
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        # Reshape to multi-head format: [B, heads, S, head_dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key  .view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key   = attn.norm_k(key)

        # --- VLA stream projections (separate Q/K/V projections: add_q/k/v_proj) ---
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            # --- Joint concatenation along sequence axis ---
            query = torch.cat([query, enc_q], dim=2)   # [B, heads, S_action+S_vla, head_dim]
            key   = torch.cat([key,   enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

            # Build attention mask: action tokens are always valid; VLA tokens use provided mask
            if attention_mask is not None:
                assert attention_mask.dtype == torch.float32
                attn_mask = torch.cat([
                    torch.ones(batch_size, 1, 1, hidden_states.shape[1],
                               device=attention_mask.device, dtype=torch.bool),  # action tokens
                    (attention_mask == 1)[:, None, None, :],                      # VLA tokens
                ], dim=-1)
            else:
                attn_mask = None

        # --- Single joint SDPA call ---
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # --- Split outputs back by original stream lengths ---
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],    # action stream output
            hidden_states[:, residual.shape[1] :],    # VLA stream output
        )
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Linear projection + dropout for action stream
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, encoder_hidden_states
```

## 逐行讲解 / What's happening

1. **两套独立投影（`to_q/k/v` 和 `add_q/k/v_proj`）：两个流各自的 QKV / Two independent projection sets**
   - 中文：动作流用 `attn.to_q/k/v`（标准注意力投影），VLA 流用 `attn.add_q/k/v_proj`（额外投影，来自 SD3 的"added" 命名规范）。两套投影参数相互独立——这比共享 K/V（MQA 风格）表达能力更强，两个流可以在不同的子空间里处理信息。
   - English: The action stream uses `attn.to_q/k/v` (standard attention projections) and the VLA stream uses `attn.add_q/k/v_proj` (separate projections, following SD3's "added context" naming). These are independent parameters — stronger than shared K/V (MQA style), each stream operates in its own subspace. The "added" naming comes from Stability AI's diffusers library SD3 implementation.

2. **`torch.cat([query, enc_q], dim=2)`：序列维度联合 / Joint concatenation along sequence dim**
   - 中文：拼接发生在第 2 维（sequence 维度，即 `[B, heads, S, head_dim]` 的 S 维）。拼接后，Q/K/V 的序列长度是 `S_action + S_vla`，一次 SDPA 可以让动作 token 关注 VLA token，同时 VLA token 也能关注动作 token（双向）。这比交叉注意力（单向：action queries only attend to VLA keys）信息流更丰富。
   - English: Concatenation happens along dim=2 (the S axis in `[B, heads, S, head_dim]`). After concatenation, each query token can attend to all `S_action + S_vla` key tokens in a single SDPA call — action tokens see VLA context, and VLA tokens see action states. This bidirectional flow distinguishes joint attention from one-directional cross-attention.

3. **注意力掩码构建 / Attention mask construction**
   - 中文：动作 token 侧永远用全 True（无 padding，动作序列等长），VLA 侧用传入的 `attention_mask`（VLA token 可能有 padding）。两者拼接成 `[B, 1, 1, S_action+S_vla]` 的 bool mask，传给 SDPA 的 `attn_mask` 参数。
   - English: Action tokens get an all-True mask (no padding in the action sequence). VLA tokens use the provided `attention_mask` (which may mask out padding). Both are concatenated into a `[B, 1, 1, S_action+S_vla]` bool tensor. SDPA interprets `True` as "attend" and `False` as "mask out."

4. **`hidden_states[:, :residual.shape[1]]` 拆分 / Output splitting by original length**
   - 中文：SDPA 输出形状是 `[B, S_action+S_vla, D]`，按动作流原始长度 `residual.shape[1]`（即 `S_action`）在序列维度切分，前半段返回给动作流，后半段（VLA 流输出）视情况经过 `to_add_out` 投影。`context_pre_only=True` 时 VLA 流不需要输出投影（只用于 context，不更新 VLA 表示）。
   - English: The SDPA output is split at index `residual.shape[1]` (= `S_action`): the first slice is the updated action stream, the second is the updated VLA stream. When `context_pre_only=True`, the VLA stream output is not projected (it's context-only, not updated). This follows SD3's design where the first few blocks of the model may treat the context stream as read-only.

## 类比 / The analogy

标准交叉注意力像一个单向窗口：动作 token 通过窗口（Q）看 VLA 书架（K/V），但 VLA 书架不知道动作 token 的存在。联合自注意力像一个圆桌会议：动作 token 和 VLA token 都坐在桌子旁，每个人都能直接和所有人交流（联合 Q/K/V 拼接）。会议结束后，动作 token 的座位方向（前 `S_action` 列）和 VLA token 的座位方向（后 `S_vla` 列）分别切出来，各自汇报自己的输出。

Standard cross-attention is a one-way observation window: action tokens (query) look through the window at the VLA bookshelf (keys/values), but the bookshelf is unaware of the action tokens. Joint self-attention is a round-table meeting: action tokens and VLA tokens all sit at the table and can speak to anyone directly (joint Q/K/V concatenation). After the meeting, the action side of the table and the VLA side are separated back out — `hidden_states[:, :S_action]` and `hidden_states[:, S_action:]` — and each side reports its updated state.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn.functional as F

def joint_attn(q_a, k_a, v_a, q_c, k_c, v_c, heads):
    """Minimal SD3-style joint attention: action stream + context stream."""
    B, S_a, D = q_a.shape
    S_c = q_c.shape[1]
    head_dim = D // heads
    def reshape(x): return x.view(B, -1, heads, head_dim).transpose(1, 2)
    Q = torch.cat([reshape(q_a), reshape(q_c)], dim=2)
    K = torch.cat([reshape(k_a), reshape(k_c)], dim=2)
    V = torch.cat([reshape(v_a), reshape(v_c)], dim=2)
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    out = out.transpose(1, 2).reshape(B, S_a + S_c, D)
    return out[:, :S_a], out[:, S_a:]  # split back

B, S_a, S_c, D, H = 2, 10, 20, 64, 4
act_tokens = torch.randn(B, S_a, D);  vla_tokens = torch.randn(B, S_c, D)
proj = lambda x: x  # identity for demo
out_act, out_vla = joint_attn(act_tokens, act_tokens, act_tokens,
                               vla_tokens, vla_tokens, vla_tokens, heads=H)
print(f"action out: {out_act.shape}, vla out: {out_vla.shape}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
action out: torch.Size([2, 10, 64]), vla out: torch.Size([2, 20, 64])
```

输出形状与输入完全一致，但每个 token 现在已经融合了来自另一个流的信息——这就是联合注意力的核心效果。

The output shapes match the inputs, but each token now carries information from the other stream — that is the fundamental effect of joint attention.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion 3（原始来源）** / **Stable Diffusion 3 (original)**: `JointAttnProcessor2_0` 在 `huggingface/diffusers` 的 SD3 实现里，完全相同的拼接 → SDPA → 拆分逻辑，只是用于图文联合而非动作 VLA。
- **FLUX 模型的 double stream blocks** / **FLUX double stream blocks**: 用类似的双流联合注意力处理图像 token 和文本 token，也是各自独立投影 QKV 再拼接。
- **SmolVLA 今日 VLA 笔记**的 prefix/suffix 设计 / **SmolVLA's prefix/suffix design (today's VLA note)**: prefix（VLM tokens）和 suffix（action tokens）在 Transformer 内部通过 attention mask 实现部分类似的效果，但 SmolVLA 用的是 causal mask，Psi0 用的是 joint bidirectional attention——两种思路的权衡。

## 注意事项 / Caveats / when it breaks

- **内存随序列长度平方增长** / **Memory scales quadratically**: 联合注意力的序列长度是 `S_action + S_vla`，FlashAttention 的 `O(N)` 内存需求仍然基于这个总长度。当 VLA token 非常长（多帧视频输入）时，显存压力会显著增加。
- **`context_pre_only=True` 的含义** / **`context_pre_only=True` semantics**: 当设为 True 时，VLA 流的输出不做 `to_add_out` 投影也不更新——意味着这个 block 只让动作 token 受益于 VLA 上下文，VLA 表示不被动作流影响。通常用在靠近输出端的 block（不想让动作噪声污染 VLA 表示）。
- **与 LoRA 的兼容性** / **LoRA compatibility**: `add_q/k/v_proj` 这些额外投影是专门为联合注意力新增的参数，不在原始 VLM 的参数空间内，因此 LoRA 微调时需要显式把这些投影加入可训练参数集合，否则它们始终是随机初始化的。

## 延伸阅读 / Further reading

- [Psi0 技术报告（RSS 2026）](https://arxiv.org/abs/2506.10780)
- [Stable Diffusion 3 原始论文（joint attention 来源）](https://arxiv.org/abs/2403.03206)
- [Psi0 项目主页](https://github.com/physical-superintelligence-lab/Psi0)
