---
date: 2026-05-29
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L93-L185
difficulty: advanced
read_time: ~13 min
tags: [code-of-the-day, wam, action, rope, register-tokens, multi-embodiment]
build_role: Action registers — action and state tokens as register tokens with their own 1D RoPE inside the video DiT
---

# dreamzero 把 action 和 state 当 register token 插进 video 序列 / dreamzero appends action and state as register tokens inside the video sequence

> **一句话 / In one line**: dreamzero 在 video 序列尾部插入 "action register" 和 "state register" token,给它们各自一份独立的 1-D RoPE(`freqs_action`、`freqs_state`),通过单一 `causal_rope_action_apply` 把三段 RoPE(video 的 3-D + action 的 1-D + state 的 1-D)拼成一个 RoPE 序列。 / dreamzero appends "action register" and "state register" tokens to the video sequence, each with its own 1-D RoPE (`freqs_action`, `freqs_state`). A single `causal_rope_action_apply` function concatenates all three RoPE bundles — video's 3-D, action's 1-D, state's 1-D — into one freq sequence.

## 为什么重要 / Why this matters

lingbot-va 把 action 和 video 共用一个 attention sequence(用 mask 区分),FastWAM 把 action 单独搞一个 DiT,dreamzero 选了第三条路:**register token 路径**。Register token 是 Vision Transformer 文献里的概念 —— 在 image patch 序列后面再挂几个"非画面"的 learnable token,让模型用它们存全局信息。dreamzero 把这个 idea 推到 WAM:在 video patch 后面挂"动作 register"和"机器人状态 register",每个都有独立的 1-D RoPE 区分时间步,**通过 RoPE 而非 mask 让 attention 知道它们是不同种 token**。这是目前最贴近"统一 multi-modal transformer"理念的设计 —— 一次 attention 同时学到 video / action / state 三者关系。

lingbot-va shares the attention sequence (separating by mask); FastWAM uses a separate DiT entirely; dreamzero picks a third path: **register tokens**. The register-token idea comes from Vision Transformers — append a few "non-pixel" learnable tokens to image patches so the model can stash global state there. dreamzero adapts this to WAM: append "action register" and "state register" tokens to video patches, each with its own 1-D RoPE, **letting RoPE — not the mask — encode which kind of token each one is**. This is the closest to a unified multi-modal transformer: one attention layer simultaneously learns video / action / state relationships.

## 代码 / The code

`dreamzero0/dreamzero` — [`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L100-L151)

```python
def causal_rope_action_apply_no_polar(
    x,                          # [B, seq_len, n_heads, head_dim], the q/k tensor
    freqs,                      # [seq_video,  head_dim/2, 2]  — video 3-D RoPE
    freqs_action,               # [seq_action, head_dim/2, 2]  — action 1-D RoPE
    freqs_state,                # [seq_state,  head_dim/2, 2]  — state  1-D RoPE
    action_register_length,     # = num_action_per_block + num_state_per_block
    num_action_per_block,
    num_state_per_block,
    action_state_index,         # which "chunk" we're processing
):
    B, seq_len, n, d = x.shape

    # ---- View x as pairs of reals (real, imag) ----
    x = x.reshape(B, seq_len, n, -1, 2)
    x_real, x_imag = x[..., 0], x[..., 1]

    # ---- Split video freqs into cos / sin ----
    freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
    freqs_cos, freqs_sin = freqs[..., 0], freqs[..., 1]

    # ---- Append action and state register freqs to the end of the sequence ----
    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)

        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block]
        freqs_state_slice = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block]

        # Concatenate action + state freqs for this chunk
        freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(
            action_register_length, 1, -1, 2)
        freqs_cos_1d, freqs_sin_1d = freqs_1d[..., 0], freqs_1d[..., 1]

        # Append to the video-freq sequence ⇒ unified RoPE freq table
        freqs_cos = torch.cat([freqs_cos[0], freqs_cos_1d], dim=0).unsqueeze(0)
        freqs_sin = torch.cat([freqs_sin[0], freqs_sin_1d], dim=0).unsqueeze(0)

    # ---- The rotary multiply: real·cos − imag·sin, real·sin + imag·cos ----
    x_real_rotated = x_real * freqs_cos - x_imag * freqs_sin
    x_imag_rotated = x_real * freqs_sin + x_imag * freqs_cos
    return torch.stack((x_real_rotated, x_imag_rotated), dim=-1).flatten(3)
```

## 逐行讲解 / What's happening

1. **`freqs`、`freqs_action`、`freqs_state` 三套独立 RoPE / Three independent RoPE tables**:
   - 中文:video 用 3-D RoPE(沿 t, h, w 各编一段,前几天讲过);action register 用独立 1-D RoPE,`freqs_action[i]` 表示"第 i 个 action 帧的位置编码";state register 也是独立 1-D。三套 RoPE 是 register 设计的**关键** —— 仅靠 mask 区分还不够,attention 需要能从位置编码识别"这是 video 还是 action 还是 state"。
   - English: video uses 3-D RoPE (per-axis freq for t/h/w as in earlier notes). Action registers use an independent 1-D RoPE: `freqs_action[i]` is "frame i of the action register". State registers use their own 1-D too. Three RoPE tables is the **load-bearing** design decision — masks alone don't suffice; attention also needs to read "video / action / state" from the positional encoding.

2. **`action_state_index` 是 chunk 游标 / Chunk cursor**:
   - 中文:dreamzero 做 self-forcing 流式推理时,序列分块处理。`action_state_index` 标识"当前在处理第几个 chunk",对应的 action / state freqs 切片 `[idx * num_per_block : (idx+1) * num_per_block]` 取出来用。下一个 chunk 来时 `action_state_index += 1`,自动取下一段位置编码 —— 不会重复也不会跳过。
   - English: dreamzero processes sequences in chunks for self-forcing inference. `action_state_index` tracks the current chunk; the corresponding action/state freq slices `[idx * num_per_block : (idx+1) * num_per_block]` are pulled out. Next chunk increments the index — never repeating, never skipping positions.

3. **`assert action_register_length == num_action_per_block + num_state_per_block` / The invariant**:
   - 中文:这条 assert 是设计的硬约束 —— 每个 chunk 里 action register 和 state register 数量是固定的。这让 attention 能预编译 mask、shape 完全静态。
   - English: this assert is a hard design invariant — each chunk holds a fixed count of action registers and state registers. Static shapes let the kernel pre-compile and let masks be cached.

4. **`torch.cat([freqs_action_slice, freqs_state_slice], dim=0)` / Pack action then state**:
   - 中文:action 在前、state 在后,顺序固定。`freqs_1d.view(register_length, 1, -1, 2)` 把它们当成"一段串行 token 的 RoPE"。
   - English: action first, state next, fixed order. The reshape treats them as one continuous register block.

5. **拼到 video freqs 尾部 / Append to the end of the video freqs**:
   - 中文:`freqs_cos = torch.cat([freqs_cos[0], freqs_cos_1d], dim=0)`。结果是一条 `seq_video + register_length` 长的 RoPE 序列。然后这条序列直接喂给 attention 的 q/k 做旋转 —— attention 就把 video 和 register 一视同仁,但它们的 RoPE 来源不同。
   - English: `torch.cat([freqs_cos[0], freqs_cos_1d], dim=0)` produces a single `seq_video + register_length`-long RoPE table. Attention rotates q/k against this combined table, treating video and registers uniformly — but their RoPE origins differ.

6. **复数乘法做旋转 / Complex multiply as rotation**:
   - 中文:`x_real * cos - x_imag * sin` + `x_real * sin + x_imag * cos` 就是 `(real + i imag) * (cos + i sin)` 的展开 —— 2D 旋转矩阵 R(θ) 作用在每对 (real, imag) 上。这跟 video 的 RoPE 数学完全一致,只是位置坐标不同。
   - English: `x_real * cos - x_imag * sin` + `x_real * sin + x_imag * cos` is the expansion of `(real + i imag)(cos + i sin)` — i.e. 2-D rotation R(θ) per pair. The math is identical to video RoPE; only the position coordinate differs.

7. **`no_polar` 版本是 TRT 友好版本 / The `_no_polar` variant is TRT-friendly**:
   - 中文:文件里同时有 `_polar` 和 `_no_polar` 两个版本。前者用 `torch.view_as_complex` 和 `torch.polar`,数学上等价但 TensorRT 不支持复数 op;后者把复数拆成显式的 (real, imag) 张量,可以编译。这是部署侧的工程妥协,值得在 nanoWAM 里抄 —— 一份代码两种 backend。
   - English: the file ships both `_polar` and `_no_polar` versions. The polar form uses `torch.view_as_complex` and `torch.polar` (equivalent maths) but TensorRT cannot lower complex ops. The `_no_polar` form expresses the rotation in real-tensor arithmetic. nanoWAM should keep both — one code path per backend.

## 类比 / The analogy

像在一本相册(video patch)的末尾贴几张便签:一张写"这一组照片对应的动作"(action register),一张写"机器人当时的关节读数"(state register)。便签和照片在书架上(attention 序列)排在一起,但便签自己用了不同的编号系统(action 用红色编号,state 用蓝色编号,照片用三维坐标 XYZ)。读相册的人(attention)看编号就知道哪页是照片、哪页是动作便签、哪页是状态便签,不会混淆。

Think of pasting two sticky notes at the end of a photo album (video patches): one says "the actions taken while these photos were captured" (action register), the other "the robot's joint readings at the time" (state register). The notes and photos sit on the same shelf (attention sequence), but the notes use their own numbering systems (red for actions, blue for state, XYZ coordinates for photos). The reader (attention) can tell which page is a photo vs an action note vs a state note by the numbering alone — no mixing.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 里这是 `nano/wam/blocks/rope_with_registers.py` —— 介于 `rope_3d.py`(video 那段)和 attention 之间的胶水。上游:video 的 3-D RoPE 表(前面 patchify-positional 笔记)+ 你新加的 `freqs_action` 和 `freqs_state` 两个 1-D RoPE 表;下游:attention 的 q/k 旋转。如果省掉这套 register 设计、改用 lingbot-va 风格的 mask:可以工作,但 (1) attention 难以学到精细的"video token 和第 5 帧 action 的对齐"—— RoPE 提供的相对距离信号没了;(2) 流式推理时分块/缓存逻辑更复杂(mask 要按 chunk 重建,RoPE 直接 slice)。生产实现还要补:(1) **multi-embodiment register**(每种机器人 state 维度不同,需要 `CategorySpecificLinear` 投到统一 register 大小);(2) **register 数量自适应**(短任务少 register、长任务多 register,而不是固定 num_per_block);(3) **register pretraining**(很多 SOTA 流派会先用 mask language modeling 风格预训练 register,让它学到"如何写小抄给 attention 看"的能力)。

English: in nanoWAM this is `nano/wam/blocks/rope_with_registers.py` — the glue between video's 3-D RoPE (earlier note) and attention. Upstream: video 3-D RoPE table + your new `freqs_action` and `freqs_state` 1-D RoPE tables. Downstream: q/k rotation inside attention. Skipping registers (using lingbot-style masks instead): works, but (1) attention loses fine-grained alignment between "video token X" and "action frame Y" — the relative-distance signal RoPE gives is gone, and (2) chunked / cached streaming inference is harder (masks have to be rebuilt per chunk, RoPE can just be sliced). Production extensions: (1) **multi-embodiment registers** via `CategorySpecificLinear` to project varying state dims to a common register size, (2) **adaptive register count** (short tasks fewer registers, long tasks more, instead of fixed per-block), (3) **register pretraining** — some SOTA recipes pre-train registers with masked LM objectives so they learn how to "write notes for attention".

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Demonstrate: video tokens + action registers + state registers, all rotated by one fused RoPE.
import torch

torch.manual_seed(0)
B, n_heads, head_dim = 1, 2, 8
seq_video, n_action, n_state = 10, 3, 2

def make_rope(length, dim, base=1):
    pos = torch.arange(length, dtype=torch.float32) * base
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    angles = pos[:, None] * freqs[None, :]
    return torch.stack([angles.cos(), angles.sin()], dim=-1)   # [L, dim/2, 2]

freqs_video  = make_rope(seq_video, head_dim, base=1)
freqs_action = make_rope(n_action, head_dim, base=10)         # 'action time' counted in larger jumps
freqs_state  = make_rope(n_state,  head_dim, base=100)        # 'state time' counted differently

x_video  = torch.randn(B, seq_video, n_heads, head_dim)
x_action = torch.randn(B, n_action, n_heads, head_dim)
x_state  = torch.randn(B, n_state,  n_heads, head_dim)
x = torch.cat([x_video, x_action, x_state], dim=1)            # [B, 15, n_heads, head_dim]

def rotate(x, freqs_video, freqs_action, freqs_state):
    B, L, n, d = x.shape
    x = x.reshape(B, L, n, d // 2, 2)
    xr, xi = x[..., 0], x[..., 1]
    freqs = torch.cat([freqs_video, freqs_action, freqs_state], dim=0)   # [L, d/2, 2]
    fc, fs = freqs[..., 0].view(1, L, 1, -1), freqs[..., 1].view(1, L, 1, -1)
    out_r = xr * fc - xi * fs
    out_i = xr * fs + xi * fc
    return torch.stack([out_r, out_i], dim=-1).flatten(3)

y = rotate(x, freqs_video, freqs_action, freqs_state)
print("input shape :", x.shape)             # [1, 15, 2, 8]
print("output shape:", y.shape)             # [1, 15, 2, 8]
print("magnitudes preserved:",
      torch.allclose(x.flatten(3).norm(dim=-1), y.norm(dim=-1), atol=1e-4))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
input shape : torch.Size([1, 15, 2, 8])
output shape: torch.Size([1, 15, 2, 8])
magnitudes preserved: True
```

中文:三种 token 各自旋转角度由各自的 RoPE 表决定,但物理上拼成一个 15-token 序列输出 —— attention 就能"一次同时看"。

English: each token type rotates by its own RoPE table, but physically they form one 15-token sequence — attention sees them in a single pass.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ViT register tokens (Darcet et al., 2024)** / **ViT register tokens (Darcet et al., 2024)**: 中文 — 视觉 transformer 的 register token 文献,dreamzero 把它推广到 (video, action, state) 三模态。 / English — the original "registers in ViT" paper that dreamzero generalises to (video, action, state) three-modality.
- **Sora 的 patch-based unified transformer** / **Sora's patch-based unified transformer**: 中文 — Sora 把不同分辨率/不同时长 video 统一编成 patch 序列,跟 dreamzero 把 video+action+state 统一编序列同源 —— 都是"一个 transformer 吃异质 token"。 / English — Sora encodes variable-resolution / variable-length video into a unified patch sequence; dreamzero generalises that to heterogeneous (video / action / state) tokens — same philosophical move.
- **GR00T MultiEmbodimentActionEncoder** / **GR00T `MultiEmbodimentActionEncoder`**: 中文 — 上游的 action encoder,dreamzero 同名文件里也有(就在 `wan_video_dit_action_casual_chunk.py` 顶端),完全照搬 Isaac-GR00T。说明 register token + multi-embodiment encoder 是配套出现的。 / English — the upstream action encoder, copied verbatim from Isaac-GR00T in the same file (top of `wan_video_dit_action_casual_chunk.py`). Register tokens + multi-embodiment encoders tend to ship together.

## 注意事项 / Caveats / when it breaks

- **三套 RoPE 必须按相同 `head_dim` 划分** / **All three RoPE tables must split `head_dim` consistently**: 中文 — `freqs_video`、`freqs_action`、`freqs_state` 的最后一维必须都是 `head_dim`(或 `head_dim/2`,看是 cos/sin 还是复数表示),否则拼接形状错。 / English — `freqs_video`, `freqs_action`, `freqs_state` must all use the same `head_dim` (or `head_dim/2`) for their last dim; otherwise the cat fails.
- **TRT 部署用 `_no_polar` 版本** / **Use `_no_polar` for TRT deployment**: 中文 — `view_as_complex` + `polar` 在 TensorRT 下不能 lower。生产部署时必须切到 `_no_polar`,数学等价但全实数运算。 / English — `view_as_complex` + `polar` cannot be lowered to TRT. Switch to `_no_polar` for deployment; identical math in real-tensor form.
- **`num_action_per_block` 不能动态变** / **`num_action_per_block` must be static**: 中文 — assert 会挂,且 attention 编译图依赖固定 shape。如果 episode 长度变化,要 pad 到固定 register length。 / English — the assert blows up and the attention compile graph depends on the static shape. Pad short episodes up to a fixed register count.
- **register 数量过多会拖垮 attention** / **Too many registers slow attention quadratically**: 中文 — register 加进 attention sequence 后,QK 计算复杂度按 `(N_video + N_reg)^2` 涨。视频已经几万 token 时,register 控制在几十到几百是合理范围。 / English — registers extend the attention sequence, blowing up QK to `(N_video + N_reg)^2`. With video already at 30 k tokens, keep registers in the tens-to-hundreds range.

## 延伸阅读 / Further reading

- [Vision Transformers Need Registers (Darcet et al., 2024)](https://arxiv.org/abs/2309.16588)
- [Today's lingbot-va action note (mask-based single stream)](./2026-05-29-lingbot-action-embedder.md)
- [Today's FastWAM ActionDiT note (parallel decoupled DiT)](./2026-05-29-fastwam-action-dit.md)
- [Yesterday's Isaac-GR00T CategorySpecificLinear note (multi-embodiment encoder)](../../2026/05/2026-05-29-isaac-groot-category-specific-linear.md)
