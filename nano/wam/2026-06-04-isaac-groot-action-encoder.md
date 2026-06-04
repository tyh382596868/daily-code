---
date: 2026-06-04
topic: wam
source: wam
repo: NVIDIA/Isaac-GR00T
file: gr00t/model/modules/flowmatching_modules.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/model/modules/flowmatching_modules.py#L25-L113
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, flow-matching, sinusoidal, gr00t]
build_role: action-conditioning module — fuses action chunk + flow-matching time into transformer-ready tokens
---

# GR00T 把"动作 + flow 时间"塞进一个 MLP / GR00T fuses action and flow-time into one small MLP

> **一句话 / In one line**: `ActionEncoder` 把动作 `a_{1:T}` 和 flow-matching 标量时间 `tau` 通过 W1 投影 + sinusoidal time + W2 拼接 + swish + W3 三层 MLP,得到 T 个 hidden-size 的 token,直接当成"动作 token"喂给 DiT。 / `ActionEncoder` runs the action chunk `a_{1:T}` and the flow-matching scalar time `tau` through a 3-layer MLP — W1 projects actions, sinusoidal encoding lifts tau, W2 concatenates and swishes, W3 finalizes — producing T tokens of `hidden_size` that the DiT consumes as ordinary action tokens.

## 为什么重要 / Why this matters

WAM 的 action-conditioning 怎么做?上周(5-29)我们看了 lingbot 用 FlexAttention mask 把动作和视频 latent 拼在一起当一条 token 流,通过注意力 mask 控制谁能看谁。今天看一个**完全不同的解法**:GR00T 在每个 flow-matching 步把"当前 noise 时间 tau"和"动作向量"在 MLP 内部融合,得到的 token 本身就携带了"我是 flow 时间 tau 下、chunk 第 t 步的动作"这条信息,然后这些 token 平等地走进 DiT 自注意力 —— 不需要特别 mask。这两种思路都对应 curriculum 的 `action-conditioning` 项,但生产代码里你会同时看到 —— 谁更适合你的 nanoWAM 取决于:你的 DiT 是否已经支持 BlockMask、你是否要让动作 token 跨时间共享 KV。

How do you do action-conditioning in a WAM? Last week (5-29) we looked at lingbot, which used a FlexAttention mask to splice actions and video latents into one token stream and let attention masks decide who sees whom. Today's pick takes a **completely different route**: GR00T fuses "current flow-matching time tau" and "action vector" inside a small MLP at every step, so the resulting tokens already carry "I am the action at chunk step t under flow time tau" in their features — they then enter the DiT self-attention on equal footing with everything else, no special mask required. Both solve the same `action-conditioning` curriculum slot, but real production code uses both. The right one for your nanoWAM depends on whether your DiT already has FlexAttention BlockMask support and whether you want action tokens to share KV across timesteps.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/model/modules/flowmatching_modules.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/model/modules/flowmatching_modules.py#L25-L113)

```python
def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = nn.Linear(action_dim, hidden_size)         # (d -> w)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)    # (2w -> w)
        self.W3 = nn.Linear(hidden_size, hidden_size)        # (w -> w)

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps -> (B, T)
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step -> (B, T, w)
        a_emb = self.W1(actions)

        # 3) Sinusoidal encoding of tau -> (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat last dim -> (B, T, 2w), then W2 -> (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x))

        # 5) Finally W3 -> (B, T, w)
        x = self.W3(x)
        return x
```

## 逐行讲解 / What's happening

1. **`timesteps.unsqueeze(1).expand(-1, T)`**:
   - 中文: flow-matching 训练时,每个样本只采一个 tau ∈ [0, 1]。但动作 chunk 有 T 步,所以把同一个 tau broadcast 到 T 步的每个位置 —— 意思是"在 tau 这个 noise level 下,整个 chunk 都要同时去噪"。
   - English: At flow-matching training time, each sample draws one scalar tau ∈ [0, 1]. But the action chunk has T steps, so the same tau is broadcast to all T positions — meaning "at this noise level, the whole chunk is being denoised together."
2. **`a_emb = self.W1(actions)`**:
   - 中文: 把每个 timestep 的动作向量(假设是 `action_dim = 14`,人形机器人的关节命令)直接 Linear 投影到 hidden_size。动作语义被压进一个 w 维向量。
   - English: Project each timestep's action vector (e.g. 14 joint commands for a humanoid) directly via Linear into hidden_size. Action semantics compressed into a w-dim vector.
3. **`SinusoidalPositionalEncoding`**:
   - 中文: 经典的 sin/cos 编码,但这里编码的不是位置而是 tau。`exponent.exp()` 生成 [1, ..., 1/10000] 区间的频率,使得不同 tau 在不同尺度上展开,模型能区分"接近完全 noise (tau≈1)"和"接近真实动作 (tau≈0)"。
   - English: Classic sin/cos encoding, but here it encodes tau rather than position. `exponent.exp()` produces frequencies from 1 down to 1/10000, expanding different tau values across different scales so the model can tell apart "almost-pure-noise (tau≈1)" from "almost-clean action (tau≈0)".
4. **`torch.cat([a_emb, tau_emb], dim=-1)`** —— 关键融合点:
   - 中文: 这是这段代码的核心思路。把"动作 token"和"时间 token"沿 hidden 维拼起来(变成 2w),然后让 W2 自己学怎么混合它们,而不是用 `a + t` 这种简单加法。MLP 比加法更灵活,可以学非线性混合(比如"tau 大时压低动作幅度")。
   - English: The central design choice. Concatenate action and time tokens along the hidden axis (giving 2w) and let W2 learn the mixing, instead of `a + t` addition. The MLP can learn nonlinear interactions (e.g. "when tau is large, scale down the action magnitude").
5. **`swish(self.W2(x))`** + **`W3(x)`**:
   - 中文: 一个小型 2 层 MLP,激活函数 swish (= x * sigmoid(x)),输出 hidden_size 维 token。这些 token 直接成为 DiT 的输入序列的一部分。
   - English: A small 2-layer MLP with swish activation (= x * sigmoid(x)) emits hidden_size-dim tokens. These tokens drop directly into the DiT's input sequence.

## 类比 / The analogy

想象你是一个调音师,你拿到一段乐谱(动作 chunk,T 个音符)和一个旋钮(tau,0 到 1 之间,表示这段乐谱被多少噪音盖住了)。普通做法是"音符弹得响一点抵消噪声"——就是加法。GR00T 的做法是把音符和旋钮值一起塞进一台"信号处理器"(W1 + W2 + W3 MLP),让它学:tau 接近 0 时输出干净的音符 token,tau 接近 1 时输出"被噪音稀释"的 token。每个 T 位置都过同一台处理器,所以处理器自然知道"这个位置在 chunk 里是第几个"——靠的不是位置编码,而是每个时间步独立处理 + DiT 后面自己加位置编码。

You're a sound engineer holding a score (an action chunk of T notes) and a knob (tau ∈ [0, 1], how noisy this version is). A naïve fix is "play the notes louder to drown the noise" — that's plain addition. GR00T instead feeds notes and the knob value into a small signal processor (the W1 + W2 + W3 MLP) and lets it learn: when tau ≈ 0, output clean note tokens; when tau ≈ 1, output noise-diluted tokens. Each of the T positions passes through the same processor independently — chunk position is encoded later by the DiT, not here.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

> 这是 `action-conditioning` 课程项的**第二种实现**(跨仓库变体),5-29 我们已经在 lingbot 看过用 FlexAttention mask 的版本。

**中文**: 在你从零搭的 nanoWAM 里,这是连接"动作模态"和"视频 latent 模态"的桥梁。完整数据流:`actions (B, T, d) + tau (B,) -> ActionEncoder -> (B, T, w)` 这 T 个 token 与视频 latent token(已经通过 VAE 5-29 编码 + patchify 5-29 切块 + 3D RoPE 5-29 加位置)拼接 `torch.cat`,一起喂进 DiT block(5-25)。flow 时间 tau 同时还要走另一条路径喂给 DiT 的 adaLN-Zero(用于全局 modulation)—— 但**这里**的 tau 是 token 级 conditioning,不是全局 conditioning。两条路径各司其职:全局 modulation 告诉每层"现在是 tau 这个 noise level",而 ActionEncoder 输出的 tokens 告诉 self-attention "在这个 noise level 下,这是 chunk 第 t 步的动作"。

省掉这个组件会怎样?动作 token 没法表达"我是 chunk 第几步、在 tau 这个 noise 下应该是什么样",生成的 trajectory 会丢失 chunk 内部的时序结构。生产级 nanoWAM 还要补:(1) action shape 变化(双手机器人 26+ DoF vs 单臂 7 DoF)时的可配置 input_dim;(2) classifier-free guidance(已覆盖 5-29)时,batch 一半要把 actions 置 0;(3) 训练初期 tau 接近 1 时 sinusoidal 频率撞高会饱和,有些实现会改用 `RandomFourierFeatures` 让频率可学。

**English**: In your from-scratch nanoWAM, this is the bridge between the action modality and the video-latent modality. Full data flow: `actions (B, T, d) + tau (B,) -> ActionEncoder -> (B, T, w)`. These T tokens are concatenated (`torch.cat`) with video-latent tokens — already encoded by the VAE (5-29), patchified (5-29), and tagged with 3D RoPE (5-29) — and the joint sequence enters the DiT block (5-25). The flow time tau **also** flows through a separate path into the DiT's adaLN-Zero for global modulation, but **here** tau is *token-level* conditioning rather than global. The two paths divide labor: global modulation tells every layer "we're at noise level tau"; ActionEncoder tokens tell self-attention "at this noise level, this is the chunk's t-th action."

Skipping this means action tokens have no way to express "I'm step t of the chunk at noise level tau," and generated trajectories lose intra-chunk temporal structure. A production nanoWAM additionally needs: (1) configurable input_dim for different robot embodiments (a bimanual robot has 26+ DoF, a single arm has 7); (2) classifier-free guidance (5-29) requires zeroing actions for half the batch; (3) high-frequency sinusoidal terms saturate when tau ≈ 1 early in training — some implementations swap in learnable `RandomFourierFeatures`.

**依赖关系 / Dependencies**: 依赖 `dit-block` (5-25 已覆盖)。是 `action-conditioning` 的跨仓库变体,5-29 lingbot FlexAttention mask 是另一种解法。 / Depends on `dit-block` (covered 5-25). This is a cross-repo variant of `action-conditioning`; the 5-29 lingbot FlexAttention mask is the other approach.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn, torch.nn.functional as F

def swish(x): return x * torch.sigmoid(x)

class SinPosEnc(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):  # t: (B, T)
        half = self.dim // 2
        exp = -torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.)) / half)
        f = t.float().unsqueeze(-1) * exp.exp()
        return torch.cat([torch.sin(f), torch.cos(f)], -1)

class ActionEncoder(nn.Module):
    def __init__(self, action_dim, w):
        super().__init__()
        self.W1 = nn.Linear(action_dim, w)
        self.W2 = nn.Linear(2 * w, w)
        self.W3 = nn.Linear(w, w)
        self.pe = SinPosEnc(w)
    def forward(self, a, tau):
        B, T, _ = a.shape
        tau = tau.unsqueeze(1).expand(-1, T)        # (B, T)
        a_emb = self.W1(a)                           # (B, T, w)
        t_emb = self.pe(tau).to(a_emb.dtype)         # (B, T, w)
        x = swish(self.W2(torch.cat([a_emb, t_emb], -1)))
        return self.W3(x)                            # (B, T, w)

enc = ActionEncoder(action_dim=14, w=128)
a   = torch.randn(2, 8, 14)                          # 2 samples, 8-step chunk, 14-DoF
tau = torch.rand(2)                                  # one tau per sample
tok = enc(a, tau)
print("action tokens:", tok.shape)                   # (2, 8, 128)
# Show that tau actually affects the output
tok_low  = enc(a, torch.zeros(2))
tok_high = enc(a, torch.ones(2))
print("delta(tau=0 vs tau=1):", (tok_low - tok_high).abs().mean().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
action tokens: torch.Size([2, 8, 128])
delta(tau=0 vs tau=1): ~0.X     <- non-zero, tau really gets mixed into the output
```

中文重点:把同一个 action 喂入两个不同 tau,输出 token 不同 —— 这就是 conditioning 起作用的证据。如果 delta 是零,说明 W2 还没学到 mixing。

The key thing to notice: feeding the same actions with two different tau values produces different tokens — that's the proof conditioning is active. If delta were zero, W2 hasn't learned to mix yet.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lingbot-va 的 FlexAttention mask 方案** / **lingbot-va FlexAttention mask approach**: 同一个 curriculum 项的另一种解法 —— 不在 MLP 里融合,而在 attention 里用 mask 控制谁看谁。 / The other solution to the same curriculum slot — instead of MLP fusion, control who sees whom via attention masks.
- **π₀ flow-matching action head** / **π₀ flow-matching action head**: 几乎一样的 `concat(action_emb, time_emb) -> MLP`,但 π0 是把"上下文 token"直接 cross-attention 到 VLM,而 GR00T 用 self-attention。 / Almost the same `concat(action_emb, time_emb) -> MLP`, but π₀ cross-attends "context tokens" into the VLM whereas GR00T uses self-attention.
- **DiT 的 timestep embedder** / **DiT timestep embedder**: 同样用 sinusoidal,但只编码全局 t,然后通过 adaLN-Zero 调制每层 LayerNorm —— 是全局 conditioning,不是 token 级。 / Same sinusoidal trick but only encodes a global t and modulates LayerNorm via adaLN-Zero — global conditioning, not token-level.
- **VAR / MAGVIT 的 time injection** / **VAR / MAGVIT time injection**: 类似,但因为是离散 token 模型,time embedding 加在 codebook 之后而非动作向量。 / Similar but, being discrete-token models, the time embedding is added after the codebook rather than to an action vector.

## 注意事项 / Caveats / when it breaks

- **每个 sample 必须只有一个 tau** / **one tau per sample only**: 这个实现假设 `timesteps.shape == (B,)`。如果你想让 chunk 内不同 t 用不同 tau(罕见但合法),要把 expand 那一行去掉,直接传 `(B, T)` 的 tau。 / The implementation assumes `timesteps.shape == (B,)`. If you want a different tau per chunk position (rare but legal), remove the expand and pass `(B, T)`-shaped tau directly.
- **swish vs SiLU** / **swish vs SiLU**: `x * sigmoid(x)` 就是 PyTorch 内置 `F.silu(x)`,但自己写一遍可以更明确,也避免某些老版本 SiLU 在 bf16 上数值精度问题。 / `x * sigmoid(x)` is identical to PyTorch's built-in `F.silu(x)`. Hand-rolling it makes the intent explicit and dodges old-version SiLU bf16 precision quirks.
- **dtype 转换 `.to(dtype=a_emb.dtype)`** / **explicit `.to(dtype=a_emb.dtype)`**: sinusoidal 默认 fp32,而 a_emb 在混合精度训练里可能是 bf16 —— 不转 dtype 在 cat 时就报错。 / Sinusoidal defaults to fp32 while a_emb may be bf16 under mixed precision. Without the explicit dtype cast, `cat` raises.
- **`log(10000) / half_dim`** / **`log(10000) / half_dim`**: 频率范围是经验值。如果你的 tau 不是 [0, 1] 而是 [0, 1000],原始公式的高频段会过密而低频段过稀。可以把 1000 0 这个常数也参数化。 / The 10000 constant assumes tau ∈ [0, 1]. If your tau is on [0, 1000] the high-frequency band gets oversampled and the low-frequency band undersampled. Parameterize the constant.

## 延伸阅读 / Further reading

- GR00T N1 / N1.5 paper: <https://arxiv.org/abs/2503.14734>
- "Flow Matching for Generative Modeling" (Lipman et al.): <https://arxiv.org/abs/2210.02747>
- π₀ technical report — for the cross-attention alternative wiring: <https://www.physicalintelligence.company/research/pi0>
