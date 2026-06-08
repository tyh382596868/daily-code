---
date: 2026-06-05
topic: diffusion
source: tracked
repo: facebookresearch/jepa
file: src/masks/multiblock3d.py
permalink: https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/masks/multiblock3d.py#L138-L203
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, masking, jepa]
---

# V-JEPA 的 3D 块状 mask:乘起来取交集、补集留给 encoder / V-JEPA's 3D block mask: multiply for intersection, complement feeds the encoder

> **一句话 / In one line**: 采样若干个 (T, H, W) 三维方块,把它们的 mask 相乘得到 predictor 区域,剩下的部分就是 encoder 能看到的 context。 / Sample a handful of (T, H, W) 3D blocks, multiply their masks to get the predictor region, and what's left is the context the encoder is allowed to see.

## 为什么重要 / Why this matters

V-JEPA、DINO-WM 这类"非生成式"世界模型,核心训练信号都不是"重建像素",而是"在 latent 空间里预测被遮住的那部分"。这个文件回答了一个看似简单实则关键的问题:**到底怎么挑要遮的区域?** 它不是随机丢 token,而是采样一组 3D 方块,把它们当作"未来 / 未观察"的区域交给 predictor,把剩下的索引给 encoder。这种做法让 encoder 必须学到时空一致的表示,而不是死记单帧。

V-JEPA-style world models train by predicting *latent* representations of masked regions, not by reconstructing pixels. The critical-but-easily-glossed-over design choice is: **how do you pick the regions to mask?** Random token dropout is too easy. The trick used here is to sample a small set of 3D blocks, hand them to the predictor as "future / unobserved" territory, and hand the encoder everything else. This pushes the encoder toward spatio-temporally consistent representations instead of frame-local memorization.

## 代码 / The code

`facebookresearch/jepa` — [`src/masks/multiblock3d.py`](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/masks/multiblock3d.py#L138-L203)

```python
def _sample_block_mask(self, b_size):
    t, h, w = b_size
    top = torch.randint(0, self.height - h + 1, (1,))
    left = torch.randint(0, self.width - w + 1, (1,))
    start = torch.randint(0, self.duration - t + 1, (1,))

    mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
    mask[start:start+t, top:top+h, left:left+w] = 0

    # Context mask will only span the first X frames
    # (X=self.max_context_frames)
    if self.max_context_duration < self.duration:
        mask[self.max_context_duration:, :, :] = 0

    # --
    return mask

def __call__(self, batch_size):
    """
    Create encoder and predictor masks when collating imgs into a batch
    # 1. sample pred block size using seed
    # 2. sample several pred block locations for each image (w/o seed)
    # 3. return pred masks and complement (enc mask)
    """
    seed = self.step()
    g = torch.Generator()
    g.manual_seed(seed)
    p_size = self._sample_block_size(
        generator=g,
        temporal_scale=self.temporal_pred_mask_scale,
        spatial_scale=self.spatial_pred_mask_scale,
        aspect_ratio_scale=self.aspect_ratio,
    )

    collated_masks_pred, collated_masks_enc = [], []
    min_keep_enc = min_keep_pred = self.duration * self.height * self.width
    for _ in range(batch_size):

        empty_context = True
        while empty_context:

            mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
            for _ in range(self.npred):
                mask_e *= self._sample_block_mask(p_size)
            mask_e = mask_e.flatten()

            mask_p = torch.argwhere(mask_e == 0).squeeze()
            mask_e = torch.nonzero(mask_e).squeeze()

            empty_context = len(mask_e) == 0
            if not empty_context:
                min_keep_pred = min(min_keep_pred, len(mask_p))
                min_keep_enc = min(min_keep_enc, len(mask_e))
                collated_masks_pred.append(mask_p)
                collated_masks_enc.append(mask_e)

    if self.max_keep is not None:
        min_keep_enc = min(min_keep_enc, self.max_keep)

    collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
    collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
    # --
    collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
    collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

    return collated_masks_enc, collated_masks_pred
```

## 逐行讲解 / What's happening

1. **`_sample_block_mask` 头三行 / first three randints**:
   - 中文: 随机挑一个起点 `(start, top, left)`,这就是这个 3D 方块在 video patch 网格里的左上前角。
   - English: pick a random anchor `(start, top, left)` — the upper-left-front corner of this 3D block in the video patch grid.

2. **`mask[start:start+t, top:top+h, left:left+w] = 0`**:
   - 中文: 注意"1 = keep / 看得见,0 = 被挡住"。这一刀把一个 `t×h×w` 的 3D 长方体清零,标记为"要预测的区域"。
   - English: convention is "1 = keep / visible, 0 = masked". This carves out a `t×h×w` cuboid as the to-predict region.

3. **`mask[max_context_duration:] = 0`**:
   - 中文: 这是一个容易看漏但很有意思的细节 — encoder 只允许看 video 的前 X 帧。后面所有时间步都强制设为 0,等于全部交给 predictor 去预测未来。
   - English: an easy-to-miss but important detail — the encoder is forbidden from looking past the first X frames. All later time steps are forced to 0, i.e. handed wholesale to the predictor. This is the "past-only context, future-only prediction" prior.

4. **`for _ in range(self.npred): mask_e *= self._sample_block_mask(p_size)`**:
   - 中文: 在 `__call__` 里关键的一行。采 `npred` 个独立方块,把它们的 mask **逐元素相乘**。由于 mask 只有 0/1,相乘 = 取交集,所以"全都没遮到"的 token 才会保留 1。换句话说:任何一个方块遮到的位置都进入 predictor 集合。
   - English: the meatiest line in `__call__`. Sample `npred` independent blocks and **element-wise multiply** their masks. Since masks are 0/1, multiplication = intersection of "keep" regions, so a token survives as encoder context only if no block touched it. Equivalently: union of all `npred` block regions becomes the predictor set.

5. **`mask_p = torch.argwhere(mask_e == 0)` / `mask_e = torch.nonzero(mask_e)`**:
   - 中文: 把布尔 mask 变成两个一维 index 张量 — 用 `gather` 直接挑 token 就行,不需要全量保留 mask。Transformer 输入侧因此能"丢掉"未观察的 token,节省内存。
   - English: convert the boolean mask into two flat index tensors. Downstream you `gather` from a packed sequence instead of materializing a full mask — predictor and encoder process disjoint sets of tokens, saving compute.

6. **`while empty_context`**:
   - 中文: 万一倒霉,`npred` 个方块把整个 video 全盖了,encoder 没东西看,就重采一次。生产代码常见的 "retry until non-degenerate" 模式。
   - English: if you're unlucky and the `npred` blocks cover everything, the encoder has nothing left — just resample. A common "retry until non-degenerate" pattern in masking code.

7. **`min_keep_enc / min_keep_pred` + `[:min_keep_x]` 截断**:
   - 中文: 不同样本被遮的格子数量不一样,但 batch collate 要求等长。最朴素的办法是按 batch 里的最小值截断 — 简单粗暴但 work。
   - English: each sample has a different number of masked tokens, but `default_collate` needs equal lengths. The simplest fix: truncate everything to the per-batch minimum. Crude but it works.

## 类比 / The analogy

想象你在一片麦田上空丢几块烟雾弹。烟雾(predictor 区域)是不规则的、可以重叠的,凡是被烟雾盖到一寸的地方,你都得"猜"。剩下没被烟雾盖到的麦穗就是 encoder 看到的"线索"。**乘法**就是"只有这一寸完全没烟雾才算可见"。这种做法逼着你学习麦田的"结构"——根据可见的麦穗推断烟雾下的麦穗长什么样,而不是死记单根麦穗。

Picture dropping a few smoke grenades over a wheat field. The smoke (predictor region) is irregular and overlappable; any inch touched by smoke is something you must guess. The remaining unobscured wheat is the encoder's evidence. **Multiplying masks** is the "only count as visible if no smoke touched this inch" rule. The training pressure forces the model to learn *field structure* — infer hidden wheat from visible wheat — instead of memorizing individual stalks.

## 自己跑一遍 / Try it yourself

```python
# minimal_jepa_mask.py — pip install torch
import torch

T, H, W = 8, 14, 14  # video grid: 8 frames, 14x14 patches each
npred, t, h, w = 3, 2, 5, 5  # 3 predictor blocks, each 2x5x5

def sample_block_mask():
    top = torch.randint(0, H - h + 1, (1,)).item()
    left = torch.randint(0, W - w + 1, (1,)).item()
    start = torch.randint(0, T - t + 1, (1,)).item()
    m = torch.ones(T, H, W, dtype=torch.int32)
    m[start:start+t, top:top+h, left:left+w] = 0
    return m

mask_e = torch.ones(T, H, W, dtype=torch.int32)
for _ in range(npred):
    mask_e *= sample_block_mask()         # intersect keep-regions
mask_e_flat = mask_e.flatten()
pred_idx = torch.nonzero(mask_e_flat == 0).squeeze()
enc_idx  = torch.nonzero(mask_e_flat == 1).squeeze()

total = T * H * W
print(f"total tokens = {total}")
print(f"encoder sees = {len(enc_idx)}  ({len(enc_idx)/total:.1%})")
print(f"predictor    = {len(pred_idx)} ({len(pred_idx)/total:.1%})")
print(f"sum = {len(enc_idx) + len(pred_idx)} (must equal total)")
```

运行 / Run with:
```bash
python minimal_jepa_mask.py
```

预期输出 / Expected output:
```
total tokens = 1568
encoder sees = 1278  (81.5%)
predictor    = 290 (18.5%)
sum = 1568 (must equal total)
```

中文:多跑几次,会看到 predictor 集合大小在变 — 因为方块位置随机、可能重叠也可能不重叠。这正是 V-JEPA 训练数据 augmentation 的一部分。

English: rerun a few times — the predictor size fluctuates because the blocks land randomly and may or may not overlap. That stochasticity is itself part of V-JEPA's data augmentation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MAE (He et al. 2021)** / **MAE**: 早期就是纯随机 token mask,但比例固定。V-JEPA 升级到 3D block 形式,因为视频里相邻 token 太冗余,随机 mask 太容易了。 / MAE uses pure random token masking at a fixed ratio. V-JEPA upgraded to 3D block masking because adjacent video tokens are too redundant — random masking is too easy a task.
- **DINO-WM** / **DINO-WM**: 用 DINOv2 做 encoder,用类似的 3D 块状预测 loss。同一个 mask 思路,不同的 backbone。 / DINO-WM uses DINOv2 as encoder with similar 3D block prediction loss. Same masking idea, different backbone.
- **SimVP / VideoMAE** / **SimVP, VideoMAE**: 视频自监督的"未来帧整块预测"也是这种思路的特例。 / "Whole future-frame prediction" in video SSL is a special case of this same design (block fills the entire spatial dim, only time varies).

## 注意事项 / Caveats / when it breaks

- **`min_keep` 截断会丢信息** / **`min_keep` truncation drops information**: batch 里最难的样本(被遮得最多)决定了所有人的可用 token 数。如果你 batch 很大,会损失大量上下文。生产实现会用 padding + attention mask 而不是截断。 / The hardest sample in the batch determines the available token count for everyone. With large batches you lose a lot of context. Production code uses padding + attention mask instead of truncation.
- **`while empty_context` 可能死循环** / **`while empty_context` can spin forever**: 如果你把 `spatial_pred_mask_scale` 调到 0.9+ 且 `npred` 大,几乎每次采样都会盖满。要保证参数组合留有 encoder 空间。 / If you crank `spatial_pred_mask_scale` to 0.9+ with large `npred`, almost every draw covers everything. Always leave headroom for the encoder.
- **`Value('i', -1)` 是 multiprocessing 共享内存** / **`Value('i', -1)` is shared multiprocessing memory**: DataLoader worker 之间共享同一个 seed counter,所以同一个 batch 内不同样本看到的方块大小相同(由 seed 决定),但位置不同。这是 V-JEPA 论文里强调的"批内一致的尺度"。 / Shared between DataLoader workers so all samples in a batch see the same block *size* (seeded) but different *positions*. This is the "batch-consistent scale" trick the paper highlights.

## 延伸阅读 / Further reading

- [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video (Meta AI, 2024)](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)
- [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377)
- [DINO-WM: World Models on Pre-trained Visual Features](https://arxiv.org/abs/2411.04983)
