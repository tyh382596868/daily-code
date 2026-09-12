---
date: 2026-08-20
topic: wam
source: wam
repo: Robbyant/lingbot-va
file: wan_va/train.py
permalink: https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/train.py#L234-L299
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, training-loop, video-action-loss]
build_role: training-loop advanced variant
---

# LingBot-VA loss：视频和动作分开算，再一起反传 / LingBot-VA Loss: Score Video and Action Separately, Backprop Together

> **一句话 / In one line**: 这段训练 step 同时计算视频 latent loss 和 action latent loss，分别按 scheduler 权重和 mask 归一化，再合成一个反向传播目标。 / This training step computes video-latent and action-latent losses separately, normalizes them with scheduler weights and masks, then backpropagates their sum.

## 为什么重要 / Why this matters

WAM 如果同时预测未来画面和动作，loss 不能简单把所有元素平均。视频 latent 有空间维度，动作 latent 有有效维度 mask，二者的 timestep 权重也可能不同。LingBot-VA 把这两路 loss 分开处理，最后才汇总，让训练目标更接近“既要想象世界，也要输出可执行动作”。

If a WAM predicts both future video and actions, the loss cannot be a blind mean over all elements. Video latents have spatial structure; action latents have validity masks; their timestep weights may differ. LingBot-VA handles the two losses separately and combines them at the end, matching the goal of imagining the world while producing executable actions.

## 代码 / The code

`Robbyant/lingbot-va` — [`wan_va/train.py`](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/train.py#L234-L299)

```python
def compute_loss(self,
    input_dict,
    pred
):
    latent_pred, action_pred = pred
    action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
    latent_pred = data_seq_to_patch(
                    self.patch_size, latent_pred,
                    input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                    input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
    Bn, Fn = input_dict['latent_dict']['timesteps'].shape
    latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
    action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)
    # Frame-wise video loss calculation
    latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
    latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
    # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
    latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
    latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
    # Sum per frame and compute mask per frame
    latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
    latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
    latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()
    # Frame-wise action loss calculation
    action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
    action_loss = action_loss * action_loss_weight[:, None, :, None, None]
    action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
    # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
    action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
    action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
    action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
    action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
    # Sum per frame and normalize by mask per frame
    action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
    action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
    action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()
    return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

def _train_step(self, batch, batch_idx):
    """Train a single batch, returns losses for logging."""
    batch = self.convert_input_format(batch)
    input_dict = self._prepare_input_dict(batch)

    should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

    if not should_sync:
        self.transformer.set_requires_gradient_sync(False)
    else:
        self.transformer.set_requires_gradient_sync(True)
    output = self.transformer(input_dict, train_mode=True)
    latent_loss, action_loss = self.compute_loss(input_dict, output)
    loss = latent_loss + action_loss

    loss.backward()
    losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}

    # Only update weights after accumulating gradients
    if should_sync:
        total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        losses['total_norm'] = total_norm
        losses['should_log'] = True
    else:
        losses['should_log'] = False
    return losses
```

## 逐行讲解 / What's happening

1. **第 238-246 行 / Lines 238-246 (shape restoration and weights)**:
   - 中文: action 和 video 预测先还原成各自 target 的形状，再按 latent/action scheduler 取训练权重。
   - English: Action and video predictions are restored to target shapes, then weighted by their separate schedulers.
2. **第 247-255 行 / Lines 247-255 (video loss)**:
   - 中文: 视频 loss 按帧展开，先对每帧元素求和，再除以该帧元素数，避免帧尺寸影响 loss 标度。
   - English: Video loss is flattened per frame, summed, and divided by element count so frame size does not change scale.
3. **第 257-268 行 / Lines 257-268 (action loss)**:
   - 中文: 动作 loss 额外乘 `actions_mask`，只让有效动作维度参与平均。
   - English: Action loss is multiplied by `actions_mask`, so only valid action dimensions contribute.
4. **第 275-299 行 / Lines 275-299 (gradient accumulation)**:
   - 中文: 非同步 micro-step 关闭梯度同步，只有累计边界才 clip、step、scheduler step 和 zero_grad。
   - English: Non-final micro-steps disable gradient sync; clipping, optimizer step, scheduler step, and zeroing happen only at the accumulation boundary.

## 类比 / The analogy

像给一份驾驶考试打分。路线规划和方向盘动作都要评分，但路线图有很多像素点，方向盘只有几个有效控制量；先各自按规则归一化，再合成总分才公平。

It is like grading a driving exam. Route prediction and steering actions both matter, but the route map has many pixels while steering has only a few valid controls. Each part needs its own normalization before the total score is fair.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoWAM 里，这是 `training-loop` 的高级版本。上游是加噪后的 video/action latent 和各自 target，下游是 optimizer step。它告诉你训练循环不能只写 `mse(pred, target).mean()`；需要区分视频和动作的形状、mask、timestep 权重和梯度累积。生产级还要补 mixed precision loss scaling、异常 batch 跳过、分布式指标聚合和 checkpoint resume。

In a nanoWAM, this is an advanced `training-loop` component. Upstream is noisy video/action latent plus targets; downstream is the optimizer step. It shows why the training loop cannot be just `mse(pred, target).mean()`: video and action need separate shapes, masks, timestep weights, and gradient accumulation behavior. A production version also needs mixed-precision loss scaling, bad-batch skipping, distributed metric aggregation, and checkpoint resume.

## 自己跑一遍 / Try it yourself

```python
video_errors = [[2, 2, 2, 2], [4, 4, 4, 4]]
action_errors = [[1, 9, 9], [2, 2, 9]]
action_masks = [[1, 0, 0], [1, 1, 0]]

video_loss = sum(sum(f) / len(f) for f in video_errors) / len(video_errors)
action_loss = 0
for errs, mask in zip(action_errors, action_masks):
    action_loss += sum(e * m for e, m in zip(errs, mask)) / (sum(mask) + 1e-6)
action_loss /= len(action_errors)
print(round(video_loss, 2), round(action_loss, 2), round(video_loss + action_loss, 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
3.0 1.5 4.5
```

中文: 被 mask 掉的 `9` 不会污染 action loss，这就是动作维度 mask 的价值。

English: The masked-out `9` values do not pollute action loss, which is the point of action-dimension masks.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM training loss** / **FastWAM training loss**: 也把视频和动作分成两路加噪和监督。 / It also separates video and action corruption/supervision paths.
- **Diffusion policy loss masks** / **Diffusion policy loss masks**: 机器人动作常有 padding 或无效自由度，loss 必须按 mask 归一化。 / Robot actions often contain padding or invalid degrees of freedom, so loss must be mask-normalized.

## 注意事项 / Caveats / when it breaks

- **mask 全 0 风险** / **All-zero mask risk**: 如果某帧动作 mask 全 0，`1e-6` 只能防 NaN，指标解释仍要小心。 / If one frame has an all-zero action mask, `1e-6` avoids NaN but the metric still needs care.
- **两个 scheduler 不能混用** / **Schedulers should not be mixed**: 视频和动作的 timestep/weight 可能不同，复用一个权重会改变训练目标。 / Video and action timestep weights may differ; sharing one weight changes the objective.

## 延伸阅读 / Further reading

- [LingBot-VA `train.py`](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/train.py#L234-L299)
- [LingBot-VA repository](https://github.com/Robbyant/lingbot-va)
