---
date: 2026-06-13
topic: diffusion
source: tracked
repo: lucas-maes/le-wm
file: jepa.py
permalink: https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py#L11-L110
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, jepa, world-model, latent-rollout]
---

# 100 行写完一个 JEPA 世界模型 —— 完整的 encode → predict → rollout 合约 / 100 lines for a complete JEPA world model — the full encode → predict → rollout contract

> **一句话 / In one line**: 一份只有 100 行的 PyTorch 类把"图像编码器 + 动作编码器 + 预测器"按 JEPA 的方式拼起来,直接得到一个可以在 latent 空间里做"想象 rollout"的世界模型 / A single 100-line PyTorch class stitches "image encoder + action encoder + predictor" JEPA-style into a world model that can imagine rollouts entirely in latent space.

## 为什么重要 / Why this matters

最近一年的"世界模型"热潮 —— Yann LeCun 的 V-JEPA 2、dino_wm、le-wm —— 共同的核心思路其实只有一句话:**不要在像素空间预测下一帧,在 latent 空间预测下一帧的 embedding,用 MSE 监督**。这条思路最干净的实现就在 `lucas-maes/le-wm` 这份 153 行的 `jepa.py` 里。它没有 diffusion、没有 LLM、没有 tokenizer —— 一个 nn.Module 同时承担"训练时算 loss"和"推理时 rollout 出整段未来"两个角色,而且用 `einops.rearrange` 把所有 (B, S, T) 张量的来回展平写得清清楚楚。

The last twelve months of "world model" hype — Yann LeCun's V-JEPA 2, dino_wm, le-wm — all reduce to one core idea: **don't predict the next frame in pixel space; predict the next frame's embedding in latent space, supervised with MSE**. The cleanest implementation of that idea lives in this 153-line `jepa.py` from `lucas-maes/le-wm`. No diffusion, no LLM, no tokenizer — one `nn.Module` plays both roles, training-time loss computation and inference-time multi-step rollout, with `einops.rearrange` making every (B, S, T) flatten/unflatten boundary explicit.

## 代码 / The code

`lucas-maes/le-wm` — [`jepa.py`](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py#L11-L110)

```python
class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings."""
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding. emb: (B, T, D), act_emb: (B, T, A_emb)."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W), action_sequence: (B, S, T, action_dim).
        S is the number of action plan samples, T is the time horizon."""
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]                                # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]                            # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]   # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)                 # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # one extra step to predict the terminal state
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info
```

## 逐行讲解 / What's happening

1. **第 11-27 行 / Lines 11-27 (`__init__`)**:
   - 中文: 模块只持有 5 个子组件 —— 一个视觉编码器、一个动作编码器、一个预测器,以及两个可选的 `projector` (放在 encoder 输出后) 和 `pred_proj` (放在 predictor 输出后)。注意这里只是"接线",并没有规定 encoder 必须是 DINO,predictor 必须是 transformer —— 任何满足"输入 (B*T, ...) 出 `last_hidden_state` " 接口的 encoder 都能塞进来。
   - English: The class holds just five sub-modules — a vision encoder, an action encoder, a predictor, and two optional `projector` (after the encoder) and `pred_proj` (after the predictor). Note this is pure *wiring* — nothing forces the encoder to be DINO or the predictor to be a transformer. Anything that takes `(B*T, ...)` and returns a `last_hidden_state` slot in works.

2. **第 29-45 行 / Lines 29-45 (`encode`)**:
   - 中文: 把 `(B, T, C, H, W)` 的视频压成 `(B*T, C, H, W)` 喂给 encoder,取出 CLS token,然后用 `einops` 还原回 `(B, T, D)`。如果传入了 `action`,顺手把它也编码成 `act_emb`。**注意整个函数是"in-place 写回 info dict"** —— 上下游约定靠一个共享的字典传递,后面的 `rollout` 会反复利用这一点。
   - English: Squash `(B, T, C, H, W)` video into `(B*T, C, H, W)`, push it through the encoder, pluck the CLS token, then `einops` it back to `(B, T, D)`. If `action` is present, encode it into `act_emb` too. **The function writes back into the `info` dict in place** — upstream and downstream all communicate through one shared dict, and `rollout` exploits this aggressively.

3. **第 47-55 行 / Lines 47-55 (`predict`)**:
   - 中文: 预测器吃 `(B, T, D)` 的历史 embedding 和 `(B, T, A_emb)` 的历史动作 embedding,给出下一时刻每个时间步对应的预测 embedding。`pred_proj` 是一个可选的线性投影 —— 比如想把预测维度从 768 投到 256 时用。
   - English: Predictor takes a history of `(B, T, D)` state embeddings and `(B, T, A_emb)` action embeddings, returns predicted next-step embeddings at every time step. `pred_proj` is an optional linear that projects the prediction dim — e.g. 768 → 256 — if you want.

4. **第 71-74 行 / Lines 71-74 (拆分历史与未来 / split history vs future)**:
   - 中文: 这是整个 `rollout` 最关键的一行 —— 把 `action_sequence` 在时间轴上劈成两半:`act_0` 是已经发生过的 `H` 步动作 (history),`act_future` 是接下来要"想象"执行的 `T - H` 步动作。前者用来给 encoder 看真实历史,后者用来"自回归地展开未来"。
   - English: This is the most important line in `rollout` — it splits `action_sequence` along the time axis: `act_0` is the `H` past actions (history) and `act_future` is the `T - H` actions to *imagine* executing. The first half feeds real history to the encoder; the second half drives the autoregressive unroll of the future.

5. **第 77-79 行 / Lines 77-79 (encode the first sample, broadcast)**:
   - 中文: 每个候选 action plan (S 个) 共享同一段真实历史观测 —— 所以只 encode 一次,然后 `unsqueeze(1).expand(B, S, ...)` 把它复制到所有 S 个候选上,**这个 expand 是 view 不是 copy,内存零成本**。
   - English: All `S` candidate action plans share the same real observation history — encode it once and `unsqueeze(1).expand(B, S, ...)` broadcasts it across all `S` candidates. **This expand is a view, not a copy — zero memory cost.**

6. **第 83-85 行 / Lines 83-85 (flatten batch × samples)**:
   - 中文: 一句 `rearrange("b s ... -> (b s) ...")` 把 batch 和 samples 两个维度合一,接下来 predictor 看到的是普通的 `(B*S, T, D)`,意识不到候选维度的存在。这是写"batched MPC / batched CEM"风格规划器最常用的小技巧。
   - English: One `rearrange("b s ... -> (b s) ...")` collapses batch and samples into one axis, so the predictor sees plain `(B*S, T, D)` and never has to know about the candidate dim. Standard trick when writing batched-MPC / batched-CEM planners.

7. **第 88-97 行 / Lines 88-97 (the autoregressive loop)**:
   - 中文: 滑窗大小 `HS = history_size` (默认 3)。每一步只把 emb 和 act 的最后 `HS` 帧切出来喂给 predictor,取出 `[:, -1:]` 这唯一一帧预测,append 到 emb 序列后面。下一次循环时新的 emb 末尾就多了一帧"想象出来的"状态。**经典的 GPT 风格自回归,只不过 token 是 latent embedding**。
   - English: Sliding-window size `HS = history_size` (default 3). Each step slices the last `HS` frames of emb + act, feeds them to the predictor, takes the single predicted frame `[:, -1:]`, and appends it to the running emb sequence. Next iteration sees one more imagined frame at the end. **GPT-style autoregression, except the "tokens" are latent embeddings.**

8. **第 107-110 行 / Lines 107-110 (unflatten and return)**:
   - 中文: `rearrange("(b s) ... -> b s ...")` 把 (B*S) 还原成 (B, S),写回 `info["predicted_emb"]`。后面就交给 `criterion` / `get_cost` 跟 `goal_emb` 算 MSE,得到每个 candidate plan 的成本 —— 这就是为什么 le-wm 能直接用来当 MPC 控制器的"想象引擎"。
   - English: `rearrange("(b s) ... -> b s ...")` reconstitutes `(B, S)` and writes back into `info["predicted_emb"]`. Downstream `criterion` / `get_cost` will MSE that against `goal_emb` to produce a per-candidate cost — which is exactly why le-wm can plug directly into an MPC controller as its "imagination engine".

## 类比 / The analogy

中文: 把这个模型想象成一个**会下盲棋的国际象棋大师**。他不需要看到真实棋盘,只需要听你报"白马 e4 → 黑兵 e5 → 白象 c4 → …" 这一串动作,脑海里就能逐步推演出"现在棋盘大致是什么样"的抽象状态 —— 这就是 latent rollout。预测器在他脑中的位置就是那个"听到一步走法就把脑内棋盘更新一格"的认知模块,history_size=3 相当于他只看最近 3 步上下文。S 个候选 plan 就好比他在脑内同时模拟 32 条"如果我下一步走 X / Y / Z 会怎样"的并行思路。

English: Picture this model as a **chess grandmaster playing blindfold**. He doesn't need to see the real board — you just call out the move sequence "white knight e4 → black pawn e5 → white bishop c4 → ..." and he rolls forward an abstract "what the position roughly looks like now" in his head. That's the latent rollout. The predictor is the cognitive module that updates the mental board one notch per move; `history_size=3` is him only attending to the last three plies of context. The `S` candidate plans are the 32 parallel "if my next move is X / Y / Z, what happens?" branches he keeps simultaneously in his head.

## 自己跑一遍 / Try it yourself

```python
# minimal_jepa_rollout.py — distil the JEPA contract into ~25 lines
import torch
import torch.nn as nn
from einops import rearrange

class TinyJEPA(nn.Module):
    def __init__(self, d=16, a=4, history=3):
        super().__init__()
        self.encoder = nn.Linear(8, d)               # fake "vision" encoder
        self.action_encoder = nn.Linear(a, d)
        # predictor: concat(state, action) -> next state
        self.predictor = nn.Sequential(nn.Linear(2*d, d), nn.Tanh(), nn.Linear(d, d))
        self.history = history

    def rollout(self, obs0, actions):                # obs0: (B, 8); actions: (B, T, a)
        emb = self.encoder(obs0).unsqueeze(1)        # (B, 1, d)
        for t in range(actions.shape[1]):
            a_emb = self.action_encoder(actions[:, max(0,t-self.history+1):t+1])
            s_emb = emb[:, -self.history:]
            x = torch.cat([s_emb[:, -1], a_emb[:, -1]], dim=-1)
            next_emb = self.predictor(x).unsqueeze(1)
            emb = torch.cat([emb, next_emb], dim=1)
        return emb                                    # (B, 1+T, d)

m = TinyJEPA()
obs0 = torch.randn(2, 8); acts = torch.randn(2, 5, 4)
rollout = m.rollout(obs0, acts)
print("rollout shape:", rollout.shape)               # (2, 6, 16)
```

运行 / Run with:
```bash
pip install torch einops
python minimal_jepa_rollout.py
```

预期输出 / Expected output:
```
rollout shape: torch.Size([2, 6, 16])
```

中文: 最值得注意的是 `rollout` 里 **没有一行 detach / no_grad** —— 整段 latent rollout 是可微的。也就是说真要做 planning,可以直接对 `actions` 求导,梯度会一路穿过整个想象序列传回来 —— 这就是为什么 le-wm 用 CEM 也行、用 gradient-based MPC 也行。

English: The thing to notice is that `rollout` contains **no detach / no_grad** — the entire latent rollout is differentiable. If you want gradient-based planning you can backprop through `actions` directly and the gradient flows back through the whole imagined sequence — which is why le-wm can drive either CEM-style or gradient-based MPC out of the box.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **dino_wm** / **dino_wm**: 同样的 (B, S, T) 候选维度展平 + sliding-window predictor。CEM planner 直接消费 `predicted_emb` 算 cost。/ The same (B, S, T) flatten-then-rollout pattern; CEM planner consumes `predicted_emb` directly to compute a cost.
- **V-JEPA 2 (`facebookresearch/jepa`)** / **V-JEPA 2 (`facebookresearch/jepa`)**: 也是 encoder + predictor,但 predictor 是个完整 transformer,且训练时用 EMA target encoder 防止 collapse。Inference 时的 rollout 结构与本文几乎一致。/ Also an encoder + predictor pair, but the predictor is a full transformer trained against an EMA target encoder to prevent collapse. The inference-time rollout structure is almost identical.
- **MuZero** / **MuZero**: 不同社区的同一思路 —— "representation → dynamics → prediction"。MuZero 的 `dynamics_network` 就是这里的 `predictor`。/ Same idea, different community lineage — "representation → dynamics → prediction". MuZero's `dynamics_network` is exactly the role of `predictor` here.

## 注意事项 / Caveats / when it breaks

- **训练 collapse 风险 / Collapse risk during training**:
  - 中文: 纯 MSE 监督一个 latent 预测器很容易学到 trivial constant —— predictor 永远输出零向量,loss 也很低。le-wm 通过 ImageBind / DINO 的预训练 encoder + 不更新 encoder 来回避这个问题 (源码里 encoder 通常 frozen),V-JEPA 用 EMA target。如果你自己写 from-scratch 版,**记得让 encoder 来自一个固定的预训练模型,或者用 EMA target**。
  - English: Pure-MSE supervision of a latent predictor collapses to the trivial constant — predictor outputs zero, loss stays low. le-wm sidesteps that by using a frozen ImageBind / DINO encoder (the source typically freezes the encoder); V-JEPA uses an EMA target. If you build a from-scratch version, **freeze the encoder against a pretrained checkpoint, or wrap it in an EMA target**.
- **history_size 决定了能 rollout 多远 / `history_size` caps the horizon you can imagine**:
  - 中文: 滑窗只看最近 3 帧,所以 predictor 必须把所有长期信息压进当前 embedding。如果环境需要更长上下文 (比如导航类),要么增大 `history_size`,要么换一个本身就有 memory 的 predictor (Transformer + KV cache)。
  - English: The sliding window only sees the last three frames, so the predictor must compress all long-term info into the current embedding. If your env needs longer context (navigation-y tasks), bump `history_size` or swap in a stateful predictor (Transformer + KV cache).
- **CLS token 假设 / The CLS-token assumption**:
  - 中文: `pixels_emb = output.last_hidden_state[:, 0]` 默认 encoder 把整张图汇总到 `[CLS]` 位置 —— 这对 DINO / ViT 成立,对 ConvNet 不成立。换 backbone 时这一行要改 (e.g. global pool)。
  - English: `pixels_emb = output.last_hidden_state[:, 0]` assumes the encoder summarises the whole image into a `[CLS]` slot — fine for DINO / ViT, false for a ConvNet. Swap that line for a global-pool when you swap backbone.

## 延伸阅读 / Further reading

- [le-wm README — Lucas Maes' minimalist JEPA world model for control](https://github.com/lucas-maes/le-wm)
- [V-JEPA 2 paper — "Video Joint Embedding Predictive Architectures"](https://arxiv.org/abs/2506.09985)
- [DINO-WM paper — "World Models on Pretrained DINO embeddings"](https://arxiv.org/abs/2411.04983)
- [Yann LeCun's "A Path Towards Autonomous Machine Intelligence" — the JEPA manifesto](https://openreview.net/forum?id=BZ5a1r-kVsf)
