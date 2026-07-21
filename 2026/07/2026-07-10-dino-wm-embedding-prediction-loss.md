---
date: 2026-07-10
topic: diffusion
source: tracked
repo: gaoyuezhou/dino_wm
file: models/visual_world_model.py
permalink: https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py#L189-L271
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, embedding-prediction]
---

# dino_wm VWorldModel：在 embedding 空间预测未来 / dino_wm VWorldModel: Predict the Future in Embedding Space

> **一句话 / In one line**: dino_wm 先把图像、proprio 和动作编码成 token，再预测未来 embedding，并把 action 维度从预测损失里排除。 / dino_wm encodes images, proprioception, and actions into tokens, predicts future embeddings, and excludes action dimensions from the prediction loss.

## 为什么重要 / Why this matters

世界模型不一定要直接预测像素。这里的 `VWorldModel.forward` 在 DINO/VQ-VAE 风格的 latent 空间里学习动力学：模型看历史 latent 和动作，输出未来 latent；decoder 只是辅助检查“这个 latent 能不能还原成画面”。

A world model does not have to predict pixels directly. This `VWorldModel.forward` learns dynamics in a DINO/VQ-VAE-style latent space: history latents plus actions go in, future latents come out; the decoder is only an auxiliary check that the latent still reconstructs images.

## 代码 / The code

`gaoyuezhou/dino_wm` — [`models/visual_world_model.py`](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py#L189-L271)

```python
def forward(self, obs, act):
    """
    input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
            act: (b, num_frames, action_dim)
    output: z_pred: (b, num_hist, num_patches, emb_dim)
            visual_pred: (b, num_hist, 3, img_size, img_size)
            visual_reconstructed: (b, num_frames, 3, img_size, img_size)
    """
    loss = 0
    loss_components = {}
    z = self.encode(obs, act)
    z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
    z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
    visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

    if self.predictor is not None:
        z_pred = self.predict(z_src)
        if self.decoder is not None:
            obs_pred, diff_pred = self.decode(
                z_pred.detach()
            )  # recon loss should only affect decoder
            visual_pred = obs_pred['visual']
            recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
            decoder_loss_pred = (
                recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
            )
            loss_components["decoder_recon_loss_pred"] = recon_loss_pred
            loss_components["decoder_vq_loss_pred"] = diff_pred
            loss_components["decoder_loss_pred"] = decoder_loss_pred
        else:
            visual_pred = None

        # Compute loss for visual, proprio dims (i.e. exclude action dims)
        if self.concat_dim == 0:
            z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
            z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
            z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
        elif self.concat_dim == 1:
            z_visual_loss = self.emb_criterion(
                z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)],
                z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
            )
            z_proprio_loss = self.emb_criterion(
                z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim],
                z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
            )
            z_loss = self.emb_criterion(
                z_pred[:, :, :, :-self.action_dim],
                z_tgt[:, :, :, :-self.action_dim].detach()
            )

        loss = loss + z_loss
        loss_components["z_loss"] = z_loss
        loss_components["z_visual_loss"] = z_visual_loss
        loss_components["z_proprio_loss"] = z_proprio_loss
    else:
        visual_pred = None
        z_pred = None

    loss_components["loss"] = loss
    return z_pred, visual_pred, visual_reconstructed, loss, loss_components
```

## 逐行讲解 / What's happening

1. **第 199-203 行 / Lines 199-203 (`encode`, `z_src`, `z_tgt`)**:
   - 中文: `encode` 把视觉、状态和动作放到同一个 latent 张量里；`z_src` 是历史窗口，`z_tgt` 是要预测的未来窗口。
   - English: `encode` puts vision, state, and action into one latent tensor; `z_src` is the history window and `z_tgt` is the future window to match.
2. **第 205-218 行 / Lines 205-218 (predict then decode detached)**:
   - 中文: predictor 学 latent 动力学；`z_pred.detach()` 让 decoder reconstruction loss 不反向改 predictor。
   - English: The predictor learns latent dynamics; `z_pred.detach()` keeps reconstruction loss from updating the predictor path.
3. **第 222-239 行 / Lines 222-239 (exclude action dims)**:
   - 中文: 目标是预测世界状态，不是复读已知动作，所以 action 维度被排除在 `z_loss` 外。
   - English: The target is future world state, not copying the known action, so action dimensions are excluded from `z_loss`.
4. **第 241-244 行 / Lines 241-244 (loss components)**:
   - 中文: 总损失之外还记录 visual/proprio 子项，便于发现“画面预测好但状态漂了”的问题。
   - English: Visual and proprio sub-losses are logged separately, making it easier to catch cases where images look good but state drifts.

## 类比 / The analogy

这像学开车时先预测仪表盘和路况摘要，而不是每一粒柏油的纹理。摘要预测准了，规划器就能工作；画面还原只是确认摘要没有丢掉关键细节。

It is like learning to drive by predicting the dashboard and road summary instead of every asphalt grain. If the summary is right, the planner can work; reconstruction only checks that the summary did not discard important detail.

## 自己跑一遍 / Try it yourself

```python
def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)

# token = [visual1, visual2, proprio, action]
z_pred = [[1.0, 2.0, 0.5, 9.0], [2.0, 3.0, 0.8, 8.0]]
z_tgt = [[1.2, 1.9, 0.4, 0.0], [2.1, 2.9, 1.0, 0.0]]

losses = []
for pred, tgt in zip(z_pred, z_tgt):
    losses.append(mse(pred[:-1], tgt[:-1]))
print(round(sum(losses) / len(losses), 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.02
```

最后一维 action 没有进入 loss，所以动作值再大也不会逼模型“预测动作本身”。

The last action dimension is not part of the loss, so large action values do not force the model to predict the action itself.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **JEPA world models** / **JEPA world models**: 也在 representation 空间做未来预测，而不是直接回归像素。 / They also predict in representation space instead of directly regressing pixels.
- **Diffusion Policy latent objectives** / **Diffusion Policy latent objectives**: 常把条件输入和预测目标切开，避免模型学习捷径。 / They often separate conditioning inputs from prediction targets to avoid shortcuts.

## 注意事项 / Caveats / when it breaks

- **detach 边界要清楚** / **Detach boundaries matter**: 如果 decoder loss 反向进 predictor，模型可能为了重建画面牺牲动力学预测。 / If decoder loss updates the predictor, the model may trade dynamics accuracy for prettier reconstructions.
- **concat layout 是合约** / **Concat layout is a contract**: `:-1`、`:-2` 这些切片依赖 action/proprio token 的固定位置。 / Slices like `:-1` and `:-2` depend on fixed action/proprio token positions.

## 延伸阅读 / Further reading

- [dino_wm `VWorldModel.forward`](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py#L189-L271)
