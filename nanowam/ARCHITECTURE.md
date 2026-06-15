# nanoWAM Architecture

> 设计大图。具体决策见 `decisions/*.md`。具体进度见 `PROGRESS.md`。

## 一句话

**用 Wan2.1-T2V-1.3B 当 video 主干,加一层"action token 序列 + 双 head + teacher-forcing causal mask",做成一个能在 robot 数据上联合预测 video 和 action 的 World-Action Model。**

## 最终架构(Stage 5 完成时长这样)

```
text  ──┐
        ├─→ T5(冻结) ──→ text context
        │
video ──→ VAE encode ─┐
                      ├─→ noisy_latent (query) ─┐
clean video latent ───┤                          │
                      └─→ cond_latent  (KV ref) ─┤
                                                  │
action ──→ Linear ────┐                           ├─→ cat 4 段 ──→ Wan DiT ×L
                      ├─→ noisy_action (query) ──┤    (单流共享 blocks)
                      │                           │    + FlexAttn 3 规则 mask
clean action ─────────┤                           │    + AdaLN per-token
                      └─→ cond_action  (KV ref) ──┘    + cross-attn 到 text
                                                       │
                                              split + drop cond/pad
                                                       │
                                       ┌───────────────┴───────────────┐
                                       │                               │
                              proj_out (Linear)                action_proj_out
                              + unpatchify                      (Linear)
                                       │                               │
                                  pred_video                      pred_action
```

(数据流图详见 `docs/datafloow_v3.png`,等 stage 4 完成后产出。)

## 各模态的接口约定

### Video latent

- **形状**:`[B, 16, F, H', W']`(z_dim=16 是 Wan2.1 VAE,决策 002)
- **进 transformer 前**:patchify(p1, p2, p3) + `patch_embedding_mlp(Linear)` → `[1, B·(F'·H'·W'), D]`
- **D = inner_dim**:Wan2.1-T2V-1.3B 的 `num_heads × head_dim`,具体数字 stage 0 之后填进 `configs/wan21_1_3B.yaml`

### Action

User 实际 action 是 `[B, horizon, action_dim]`。我们**塞进 5D 壳子 `[B, action_dim, F, τ, 1]`**:

- `C = action_dim`(关节数等)
- `F` = 跟 video 同步的帧数
- `H = τ` = `action_per_frame`(每 video 帧配 τ 个高频子动作)
- `W = 1`(占位哑维)

这样 action 能直接复用 video 那套 `_input_embed / _add_noise / RoPE` 管线。详见 lingbot 的同名机制(`lingbot_va/wan_va/wan_va_server.py:456-462`)和我们的决策 005。

### Text

- T5 (`umt5-xxl`,~4.7B,冻结)预编码出 `[B, L, 4096]`
- 进 transformer 前 `text_embedder = Linear(4096 → D)` 投到模型维
- 通过 **cross-attention** 进 transformer(不参与 self-attn)

## Transformer 块结构(参考 lingbot `WanTransformerBlock`)

每层做四件事,顺序固定:

1. **AdaLN seed**:`scale_shift_table + per-token timestep` → chunk(6) → 6 个 modulation
2. **Self-attention(带 rotary,带 FlexAttn block_mask)**:
   - `norm1·(1+scale_msa) + shift_msa`
   - `to_q/k/v(同源)` → RMSNorm → `apply_rotary(Q,K)`
   - `flex_attention(Q, K, V, block_mask)` → `to_out`
   - `x = x + attn·gate_msa`
3. **Cross-attention(无 rotary,K/V 来自 text)**:
   - `norm2(x)` → `to_q(x)` + `to_k/to_v(text_emb)`
   - `flex_attention` 只做 batch isolation mask
   - `x = x + attn`(没有 gate)
4. **FFN(gelu-approximate)**:
   - `norm3·(1+c_scale) + c_shift`
   - `FeedForward`
   - `x = x + ffn·c_gate`

(注意 lingbot 用 `c_*` 前缀给 FFN,容易误以为是给 cross-attn 的,**实际不是**)

## FlexAttn 三规则 mask(Stage 4 才上)

按 token 的 `noise_id ∈ {0=noisy, 1=cond}` 和 `frame_id`(video chunk·2 偶,action chunk·2+1 奇)分情况:

| 规则 | mask |
|---|---|
| ① clean→clean | `block_causal_mask` |
| ② noisy→clean | `block_causal_mask_exclude_self`(query 看 cond 历史,不看自己对应的 cond,防泄题) |
| ③ noisy→noisy | `block_self_mask`(只看自己当前 chunk) |
| AND | `seq_mask`(batch isolation)+ `block_window_mask`(window_size) |

代码参考 `/tmp/daily_code_cache/lingbot_va/wan_va/modules/model.py:154-201`。

## 训练 loss

每段 noisy token 算自己的 flow-matching velocity MSE:

- video: `L_dyn = ||v_θ(z^{(s)}, s, cond, …) − (z − ε)||²`
- action: `L_inv = ||v_ψ(a^{(s)}, s, cond, …) − (a − ε)||²`
- 总 loss: `L = L_dyn + λ · L_inv`(λ ≈ 0.1 起步,根据 loss 量级调)

cond 是 50% 概率 clean / 50% 概率 noise 加噪到 `s_aug ∈ [0.5, 1]`(video 端)或 100% clean(action 端)。 详见 lingbot 论文 Eq.10-12。

## 推理(Stage 5)

按 LingBot-VA Algorithm 1:

```
z_0 ← E(o_0); C ← {z_0}
loop:
    # video chunk:只去噪到 s=0.5 就停(提速 2x)
    z̃_{t+1:t+K} ← ε + ∫_0^{0.5} v_θ(... | C) ds
    # action chunk:用半噪 video 当 context,完整去噪到 s=1
    a_{t:t+K-1} ← ε + ∫_0^1 v_ψ(... | z̃, C) ds
    for i in [t, t+K):
        execute a_i; receive o_{i+1}; z_{i+1} ← E(o_{i+1})
    C ← C ∪ {z_{t+1:t+K}, a_{t:t+K-1}}  # 注意 z 是真实观测,不是生成的 z̃
    t += K
```

参考 `wan_va_server.py:572-604`(`_compute_kv_cache`)。
