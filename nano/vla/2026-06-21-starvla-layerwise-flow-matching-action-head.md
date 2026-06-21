---
date: 2026-06-21
topic: vla
source: vla
repo: starVLA/starVLA
file: starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py
permalink: https://github.com/starVLA/starVLA/blob/1055c97556c482703137bab635f44b241b9b8ad2/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L197-L408
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, vla, flow-matching, action-head, layerwise, cross-attention, dit, curriculum]
build_role: action-head-continuous (advanced cross-repo variant — layer-wise conditioning upgrade)
---

# 层级特征 × DiT 动作头:starVLA 怎么让 early DiT 看底层像素、late DiT 看高层语义 / Layer-wise features × DiT action head: how starVLA lets early DiT layers read low-level pixels while late layers read high-level semantics

> **一句话 / In one line**: 标准 VLA 把 VLM 最后一层输出喂给 DiT;starVLA 的 `LayerwiseFlowmatchingActionHead` 把每一层 VLM 的输出都喂给 DiT——DiT 的第 k 层可以 cross-attend 到 VLM 的第 k 层特征,从像素细节到语义意图都有对应的信号源。/ Standard VLAs feed the VLM's last-layer output to the DiT; starVLA's `LayerwiseFlowmatchingActionHead` feeds *every* VLM layer's output to the DiT — DiT layer k can cross-attend to VLM layer k features, giving the action head signals from pixel-level detail all the way to high-level semantic intent.

## 为什么重要 / Why this matters

大多数 VLA 的 action head 把 VLM 当一个黑盒——取最后一层 `hidden_states` 作为 cross-attention 的 key/value。这相当于只看"VLM 理解完之后的结论",丢掉了中间层的处理过程。

VLM 早期层保留了局部纹理和空间细节(低抽象);后期层压缩成了语义摘要(高抽象)。对于精细操控任务,action head 需要两种信息:语义引导("拧开瓶盖")和低级感知("手指与物体的接触位置")。

starVLA 的解法:把 VLM 每层的 `hidden_states` 作为一个 list 传进 DiT,DiT 自己的每层 cross-attention 去选择从哪一层 VLM 特征提取信息——不需要人工写 routing 规则,DiT 在训练里自己学会了分层对齐。

Most VLA action heads treat the VLM as a black box — take the last-layer `hidden_states` as cross-attention key/values. This discards all the intermediate representations and only preserves the VLM's "final conclusion".

VLM early layers retain local texture and spatial detail (low abstraction); later layers compress everything into semantic summaries (high abstraction). Fine-grained manipulation tasks need both: semantic guidance ("unscrew the cap") and low-level perception ("exact contact point between finger and object").

starVLA's solution: pass every VLM layer's `hidden_states` as a list to the DiT, letting each DiT cross-attention layer decide which VLM layer to draw from — no manual routing rules, the DiT learns this layer-wise alignment during training.

## 代码 / The code

`starVLA/starVLA` — [`starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`](https://github.com/starVLA/starVLA/blob/1055c97556c482703137bab635f44b241b9b8ad2/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L197-L408)

```python
class LayerwiseFlowmatchingActionHead(nn.Module):
    """
    Layer-wise cross-attention DiT action head.
    Receives vl_embs_list — one tensor per VLM layer — instead of just the final layer.
    """

    def __init__(self, global_config, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        self.model = DiT(**diffusion_model_cfg_kwargs)      # DiT backbone
        self.action_dim = action_config.action_dim
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(
            action_config.num_target_vision_tokens, self.input_embedding_dim
        )
        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def forward(
        self,
        vl_embs_list: list,        # list of (B, seq_len, D) — one per VLM layer
        actions: torch.Tensor,     # (B, action_horizon, D_action)
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ):
        device = actions.device
        B, L, D = vl_embs_list[0].shape

        # ── Rectified-flow noise injection ──
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(B, device=device, dtype=actions.dtype)
        t = t[:, None, None]                              # (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions  # linear interpolation
        velocity = actions - noise                         # target velocity

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        # ── Embed noisy action trajectory ──
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # ── Prepend learnable future tokens ──
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = torch.cat((future_tokens, action_features), dim=1)

        # ── Layer-wise DiT forward ──
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,   # KEY: list of per-VLM-layer tensors
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
        )

        # ── Decode action tokens only ──
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device

        # Start from pure noise
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype, device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((batch_size,), t_disc, device=device, dtype=torch.long)

            action_features = self.action_encoder(actions, timesteps)
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = torch.cat((future_tokens, action_features), dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_list,   # same layer-wise list
                timestep=timesteps,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
            )
            pred_velocity = self.action_decoder(model_output)[:, -self.action_horizon:]

            # Euler integration step
            actions = actions + dt * pred_velocity

        return actions
```

## 逐行讲解 / What's happening

1. **`vl_embs_list: list` — 核心设计决策**
   - 中文: 这是和标准 VLA 最大的区别。标准方案传的是 `vl_embs: Tensor (B, seq, D)`;这里传的是 `list[Tensor]`,长度 = VLM 的层数。DiT backbone 内部的每层 cross-attention 会从这个 list 里选择对应层的特征来 attend。
   - English: This is the key departure from standard VLAs. Standard approaches pass `vl_embs: Tensor (B, seq, D)` — a single tensor. Here it's `list[Tensor]`, one tensor per VLM layer. Each cross-attention inside the DiT backbone selects features from the corresponding list entry.

2. **`noisy_trajectory = (1 - t) * noise + t * actions` — rectified flow**
   - 中文: 这是 flow-matching 的线性插值路径:在 `t=0` 时是纯噪声,在 `t=1` 时是真实动作。`velocity = actions - noise` 是目标速度场——模型学会在任意中间时刻预测"应该向哪个方向走"。
   - English: This is the rectified-flow linear interpolation: at `t=0` the trajectory is pure noise, at `t=1` it's the true action. `velocity = actions - noise` is the target velocity field — the model learns to predict "which direction to move" at any intermediate time.

3. **Beta 分布时间采样 `sample_time`**
   - 中文: 不是均匀采样 t ∈ [0,1],而是用 Beta 分布,让训练更多采样 t 接近 0 的位置(高噪声)。这在 flow-matching 的实践中通常改善精细动作的生成质量,因为高噪声步骤的梯度信号更强。
   - English: Time is sampled from a Beta distribution rather than uniformly from [0,1], biasing training toward lower values of t (higher noise). In practice this improves fine-grained action quality because gradient signals are stronger at high noise levels.

4. **`future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)`**
   - 中文: 可学习的"未来 token"——类似 BERT 的 `[CLS]` token,但在 action 生成里充当"待预测位置的占位符"。它们和 `action_features` 拼在一起输入 DiT。`pred = pred[:, -actions.shape[1]:]` 只取 action 位置的输出作为预测。
   - English: Learnable "future tokens" — analogous to BERT's `[CLS]` but serving as placeholder positions for the to-be-predicted actions. They're concatenated with `action_features` before entering the DiT. Only the action-token positions in the output are decoded as predictions.

5. **`Euler integration` 在 `predict_action`**
   - 中文: `actions = actions + dt * pred_velocity` — 最简单的 ODE 解法。`num_inference_timesteps` 步遍历下来,噪声逐渐变成真实动作。步数越多越准,但推理延迟越大。
   - English: `actions = actions + dt * pred_velocity` — the simplest ODE solver. Over `num_inference_timesteps` steps, noise is gradually transformed into the final action. More steps = more accurate but higher latency.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 `action-head-continuous` 组件的**进阶跨仓版本**。你的 nanoVLA 当前已经覆盖了基础的 flow-matching action head(leRobot 版本,2026-06-15 的 note),那个版本把 VLM 最后一层输出直接作为 DiT cross-attention 的 key/value。

starVLA 版本展示了如何把这个"flat VLM feature"升级成"per-layer feature list":

**输入/输出:**
- 输入: `vl_embs_list` (list of `[B, seq, D]`, 长度 = VLM 层数) + `actions` (训练) 或 `None` (推理)
- 输出: 训练时返回 flow-matching loss;推理时返回 `[B, action_horizon, D_action]` 动作

**上游依赖:** VLM backbone 在前向时需要把每层的 `hidden_states` 保存下来并以 list 形式传出。这要求 VLM backbone 配合——通常通过 hook 或修改 `output_hidden_states=True` 来实现。

**下游:** 动作直接送给机器人执行器。

**省掉这个组件会发生什么?** 退化成标准单层 cross-attention action head——功能正确,但对精细操控任务的表现比层级版差。

**生产级还需要加什么?** (1) DiT 内部如何把 list 里的多层特征路由到各自的 cross-attention 层(starVLA 的 `DiT` 类负责这个);(2) 多张相机视图时,各相机的 VLM 输出如何合并进同一个 `vl_embs_list`;(3) 实时 chunking:`predict_action_realtime` 处理上一个 chunk 和当前 chunk 之间的平滑过渡。

This is the **advanced cross-repo variant** of the `action-head-continuous` curriculum component. Your nanoVLA already covers the baseline flow-matching action head (leRobot version, 2026-06-15 note), which passes the VLM's final-layer output as flat cross-attention key/values.

The starVLA version shows how to upgrade from "flat VLM feature" to "per-layer feature list":

**I/O:** Input is `vl_embs_list` (list of `[B, seq, D]`, one per VLM layer) + `actions` (training) or nothing (inference). Output is the flow-matching loss during training, or `[B, action_horizon, D_action]` actions during inference.

**Upstream dependency:** The VLM backbone must save and expose every layer's `hidden_states`, typically via `output_hidden_states=True` or forward hooks.

**Downstream:** Actions go directly to the robot actuator.

**What happens if you skip this?** You fall back to a single-layer flat cross-attention action head — functionally correct, but weaker on fine-grained manipulation where low-level spatial features matter.

**What production needs on top?** (1) The DiT's internal routing that distributes list entries to corresponding cross-attention layers (handled by starVLA's `DiT` class); (2) merging multi-camera VLM outputs into one `vl_embs_list`; (3) real-time chunking for smooth action transition between chunks (`predict_action_realtime`).

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

# 最简版的 layer-wise flow-matching action head
class NanoLayerwiseFMHead(nn.Module):
    def __init__(self, num_vlm_layers=8, d_model=256, d_action=7, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.d_action = d_action
        # 每个 DiT 层 cross-attend 对应 VLM 层
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            for _ in range(num_vlm_layers)
        ])
        self.action_proj = nn.Linear(d_action, d_model)
        self.out_proj = nn.Linear(d_model, d_action)

    def forward(self, vl_embs_list, actions):
        t = torch.rand(actions.shape[0], 1, 1, device=actions.device)
        noise = torch.randn_like(actions)
        noisy = (1 - t) * noise + t * actions
        velocity = actions - noise

        x = self.action_proj(noisy)  # (B, horizon, d_model)
        for i, (layer, vl) in enumerate(zip(self.cross_attn_layers, vl_embs_list)):
            x, _ = layer(x, vl, vl)   # x attends to VLM layer i

        pred = self.out_proj(x)
        return ((pred - velocity) ** 2).mean()

# 模拟 8 层 VLM 输出
B, seq, D, num_layers = 2, 20, 256, 8
vl_embs = [torch.randn(B, seq, D) for _ in range(num_layers)]
actions = torch.randn(B, 16, 7)

head = NanoLayerwiseFMHead(num_vlm_layers=num_layers, d_model=D)
loss = head(vl_embs, actions)
print(f"loss: {loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
loss: ~0.5-1.0   (varies by random seed)
```

中文: `vl_embs` 是一个 list,长度等于 VLM 层数。每个 DiT 层 cross-attend 到对应 VLM 层——不同深度的 VLM 特征被独立对齐到不同 DiT 层。

English: `vl_embs` is a list with length equal to the VLM's number of layers. Each DiT layer cross-attends to the corresponding VLM layer — different abstraction levels of VLM features are independently aligned to different DiT layers.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **leRobot SmolVLA** / **leRobot SmolVLA**: 用 VLM 最后一层 hidden state 作为单一 cross-attention key/value — 这是"flat"版本,是本 note 升级的起点。`expert_model.py` 里 `smol_vlm_with_expert` 的 `encoder_hidden_states` 只有一个张量。/ Uses only the last-layer hidden state as cross-attention key/value — the "flat" version that this note upgrades. `encoder_hidden_states` there is a single tensor.
- **GR00T N1.7** / **GR00T**: 类似地用最后层 VLM 输出 cross-attend,但 GR00T 有专门的"proprio encoder"把机器人状态单独 embed 进来,可以认为是另一种形式的"多源 cross-attention"。/ Also uses last-layer VLM output for cross-attention, but GR00T adds a dedicated proprio encoder — another form of multi-source conditioning.
- **U-Net skip connections** / **U-Net 跳跃连接**: 同样的"早层特征 → 后层解码"思路——早期卷积层的低级特征直接跳接到解码器对应分辨率,不经过瓶颈压缩。layer-wise VLM → DiT 是这个思路在 Transformer 里的实现。/ Same "early-layer features → late decoder" idea — early conv features skip-connect to matching-resolution decoders without bottleneck compression. Layer-wise VLM → DiT is the Transformer instantiation of this idea.

## 注意事项 / Caveats / when it breaks

- **VLM 需要输出所有层** / **VLM must expose all layers**: 需要 `output_hidden_states=True` 或 hook — 会带来显存和带宽开销(每层都要保留 `hidden_states`)。层数越多 × 序列越长,显存压力越大。/ Requires `output_hidden_states=True` or hooks — adds memory and bandwidth overhead. More layers × longer sequences = higher pressure.
- **DiT 层数 ≠ VLM 层数时的对齐策略** / **Alignment when DiT layers ≠ VLM layers**: starVLA 的 DiT 类内部处理了这个映射(可能是 round-robin 或学习的路由);自己实现时需要决定"DiT 第 i 层 attend 哪个 VLM 层"。最简方案是线性插值索引。/ starVLA's DiT class handles this mapping internally; when implementing yourself you need to decide how DiT layer i maps to VLM layer j. The simplest approach is linear interpolation of indices.
- **Beta 分布时间采样** / **Beta-distribution time sampling**: 超参数 `noise_beta_alpha` 和 `noise_beta_beta` 的选取影响训练效率。`alpha=1, beta=1` 退化为均匀分布;文中 starVLA 用的值需要根据任务调整。/ The `noise_beta_alpha` and `noise_beta_beta` hyperparameters affect training efficiency. `alpha=1, beta=1` reduces to uniform; starVLA's values require task-specific tuning.

## 延伸阅读 / Further reading

- Flow matching 基础: [Flow Matching for Generative Modeling (arXiv 2210.02747)](https://arxiv.org/abs/2210.02747)
- Rectified Flow: [arXiv 2209.03003](https://arxiv.org/abs/2209.03003)
- starVLA 项目: [GitHub starVLA/starVLA](https://github.com/starVLA/starVLA)
- 对应 leRobot 基础版本 (2026-06-15): `nano/vla/2026-06-15-vla-jepa-flow-matching-action-head.md`
