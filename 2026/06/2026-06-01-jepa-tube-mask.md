---
date: 2026-06-01
topic: diffusion
source: tracked
repo: facebookresearch/jepa
file: src/masks/random_tube.py
permalink: https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/masks/random_tube.py#L60-L117
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, diffusion, self-supervised, video, masking]
---

# V-JEPA 的"管子"遮罩:同一片像素在所有帧上一起被挡 / V-JEPA's "tube" mask: occlude the same pixels across every frame

> **一句话 / In one line**: 先在二维平面上抽一张随机遮罩,再沿时间轴把同一张遮罩平铺到所有帧上,模型就没法靠"隔壁帧同一位置"偷答案。 / Sample one 2D mask, tile it along the time axis so the *same* spatial patches are dropped on every frame — and the model can no longer cheat by peeking at a neighboring frame at the same `(x, y)` location.

## 为什么重要 / Why this matters

视频自监督最容易翻车的地方在于:如果每一帧的遮罩是独立随机抽的,模型只要在时间上做一次轻微插值就能把缺失部分"补"回来——根本学不到东西。V-JEPA 的解决方案极其朴素:抽一张二维遮罩,然后 `np.tile` 沿时间维复制 `duration` 份。这让"被遮的位置"在整段视频里穿成一根**管子(tube)**,模型必须真的去理解物体形状与时序运动,而不是做帧间复制粘贴。

The classic failure mode in video self-supervised learning is that an independent random mask per frame leaks information — the model can recover any missing patch by interpolating from the same `(x, y)` in an adjacent frame, and ends up learning almost nothing. V-JEPA fixes this with one of the simplest possible tweaks: sample a single 2D mask, then `np.tile` it across the time axis so the occluded set forms a continuous **tube** through the video. To fill in those patches the encoder has to actually understand object shape and temporal motion, not just copy from a neighbor.

## 代码 / The code

`facebookresearch/jepa` — [`src/masks/random_tube.py`](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/masks/random_tube.py#L60-L117)

```python
class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        ratio=0.9,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_patches_spatial = self.height*self.width

        self.ratio = ratio

        self.num_keep_spatial = int(self.num_patches_spatial*(1.-self.ratio))
        self.num_keep = self.num_keep_spatial * self.duration

        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def __call__(self, batch_size):
        def sample_mask():
            mask = np.hstack([
                np.zeros(self.num_patches_spatial - self.num_keep_spatial),
                np.ones(self.num_keep_spatial),
            ])
            np.random.shuffle(mask)
            mask = torch.tensor(np.tile(mask, (self.duration, 1)))
            mask = mask.flatten()
            mask_p = torch.argwhere(mask == 0).squeeze()
            mask_e = torch.nonzero(mask).squeeze()
            return mask_e, mask_p

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(batch_size):
            mask_e, mask_p = sample_mask()
            collated_masks_enc.append(mask_e)
            collated_masks_pred.append(mask_p)

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_masks_enc, collated_masks_pred
```

## 逐行讲解 / What's happening

1. **`__init__` 中的尺寸推导 / Geometry inside `__init__`**:
   - 中文: 把 224×224 的图按 16×16 切成 `14×14 = 196` 个 patch,16 帧按时间 patch=2 压成 `duration=8` 个时间 token。`ratio=0.9` 表示要遮掉 90% 的空间 patch,所以 `num_keep_spatial = 196 * 0.1 ≈ 19`。
   - English: Geometry: a 224×224 image with patch=16 gives `14×14 = 196` spatial patches; 16 frames with `temporal_patch_size=2` collapse to `duration=8` time tokens. `ratio=0.9` means we drop 90% of the spatial patches, so `num_keep_spatial = 196 * 0.1 ≈ 19`.

2. **`sample_mask` 的 hstack + shuffle / The hstack-then-shuffle in `sample_mask`**:
   - 中文: 先拼出一个固定长度为 196 的 0/1 向量(0 占 177 个、1 占 19 个),再 shuffle。这比直接 `np.random.choice` 更高效,也保证遮挡比例**精确**等于配置值。
   - English: First build a length-196 vector of 0s and 1s with the *exact* counts (177 zeros, 19 ones), then shuffle. This is faster than `np.random.choice` and guarantees the masking ratio matches the config *exactly*, not just in expectation.

3. **`np.tile(mask, (duration, 1))` —— 管子在这里成形 / The tube takes shape here**:
   - 中文: 这一行是整个算法的灵魂。把同一张 `(14·14,)` 的二维遮罩复制 8 次得到 `(8, 196)`,然后 `flatten` 成长度 1568 的一维 token 序列。被挑中的 19 个空间位置在所有 8 个时间步上**同步**被遮——这就是"tube"。
   - English: This single line is the heart of the algorithm. The `(196,)` 2D mask gets tiled 8 times into `(8, 196)`, then flattened into a 1568-long token sequence. The 19 chosen spatial positions are dropped **synchronously** across all 8 time steps — that is the "tube".

4. **`mask_e` / `mask_p` 的拆分 / Splitting into `mask_e` and `mask_p`**:
   - 中文: `mask_e` (encoder) 是要喂给 encoder 的"可见"token 的索引;`mask_p` (predictor) 是要让 predictor 还原的"被挡"token 的索引。两者互补且不重叠,V-JEPA 训练目标就是让 predictor 从 `mask_e` 的表示里预测出 `mask_p` 位置的表示。
   - English: `mask_e` indexes the visible tokens fed to the encoder, while `mask_p` indexes the hidden tokens the predictor has to reconstruct. They're complementary and disjoint. V-JEPA's training objective is exactly: given the encoder representations at `mask_e`, predict the representations at `mask_p`.

5. **`_itr_counter = Value('i', -1)`**:
   - 中文: 这是个跨进程共享的整数计数器,用于让多个 DataLoader worker 之间协调 mask 的随机种子推进,避免不同 worker 抽到完全一样的 mask。
   - English: A cross-process shared integer counter used to coordinate the random seed across DataLoader workers, so each worker doesn't accidentally produce the same mask.

## 类比 / The analogy

想象拍一段 8 帧的延时摄影,然后用一块**布满 19 个小孔的卡纸**挡在镜头前——不管你拍哪一帧,布上的 19 个孔位都在同一个 `(x, y)` 上。如果你想猜被挡住的那块到底是什么,你不能去翻"旁边那帧",因为那张帧的同一个位置也被同样的孔挡住了;你只能去理解"这只猫在前 8 帧里大概朝哪个方向跑"。这才是 V-JEPA 希望模型学到的能力。

Imagine shooting an 8-frame time-lapse and holding a **stencil with 19 punched holes** in front of the lens — every frame is blocked at the same `(x, y)` positions. If you want to guess what's inside a hole, you *can't* peek at "the next frame" because that frame has the same holes in the same places. You're forced to reason about object identity and motion ("the cat was moving left, so its head must be in those positions now"). That reasoning is exactly what V-JEPA wants the encoder to learn.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
import torch

H_PATCHES, W_PATCHES, DURATION = 14, 14, 8
num_patches_spatial = H_PATCHES * W_PATCHES
ratio = 0.9
num_keep = int(num_patches_spatial * (1 - ratio))

mask = np.hstack([
    np.zeros(num_patches_spatial - num_keep),
    np.ones(num_keep),
])
np.random.shuffle(mask)

tube = np.tile(mask, (DURATION, 1))  # shape (8, 196)
print("kept per frame:", int(mask.sum()))
print("frames identical?", all(np.array_equal(tube[0], tube[t]) for t in range(DURATION)))
print("tube shape:", tube.shape, "total kept:", int(tube.sum()))
```

运行 / Run with:
```bash
pip install numpy torch
python try.py
```

预期输出 / Expected output:
```
kept per frame: 19
frames identical? True
tube shape: (8, 196) total kept: 152
```

中文一两句:注意 `frames identical?` 输出 `True`——这就是"tube"成立的关键证据:同一个空间遮罩被复制到每一帧。

Notice that `frames identical?` is `True` — the proof that the tube is intact. The same spatial mask is reused on every frame, so the encoder cannot solve the prediction task by copying across the time axis.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MaskFeat、VideoMAE v2** / **MaskFeat, VideoMAE v2**: 中文:同样使用 tube masking,只是把 ratio 抬到了 0.95,迫使模型在更稀疏的视觉证据下做预测。 / English: Both use tube masking with even higher ratios (~0.95), pushing the model to reason from sparser visual evidence.
- **MAE-ST**: 中文:Facebook 的早期 spatial-temporal MAE 工作,提出了"agnostic" / "tube" / "frame" 三种 mask 策略,实验显示 tube 在视频上明显优于 agnostic。 / English: Meta's earlier spatial-temporal MAE compared "agnostic", "tube", and "frame" masking — tube wins on video by a clear margin, motivating V-JEPA's choice.
- **LeRobot 中的 multi-camera masking** / **multi-camera masking in LeRobot**: 中文:多相机版本会沿相机轴而不是时间轴 tile,但思路一致——让模型不能从"另一台相机的同一像素"偷答案。 / English: Multi-camera variants tile along the camera axis instead of time — same idea, different axis: prevent the model from copying across viewpoints.

## 注意事项 / Caveats / when it breaks

- **`num_keep_spatial` 取整 / `num_keep_spatial` rounding**: 中文:`int(196 * 0.1) = 19`,实际保留比例是 9.69% 而非 10%。当 `num_patches_spatial` 较小时舍入误差会被放大,需要单独验证。 / English: `int(196 * 0.1) = 19` is 9.69%, not 10%. When `num_patches_spatial` is small, the rounding error matters and you should verify the actual ratio.
- **batch 内不同样本的 mask 也不同 / Different samples in a batch get different masks**: 中文:外层 `for _ in range(batch_size)` 保证每个样本独立采样,只在样本**内部**的时间维上共享 mask。如果你希望 batch 内共享(更严格的训练约束),要把 `sample_mask` 调用挪到循环外面。 / English: The outer `for _ in range(batch_size)` resamples per example — sharing happens **inside** a sample across time, not **across** samples. If you want batch-shared masks (stronger consistency constraint), move `sample_mask` outside the loop.
- **temporal_patch_size 必须整除 num_frames / `temporal_patch_size` must divide `num_frames`**: 中文:`duration = num_frames // temporal_patch_size`,如果不能整除会悄悄丢弃尾帧。 / English: `duration = num_frames // temporal_patch_size` silently drops trailing frames when it doesn't divide evenly — easy to miss.

## 延伸阅读 / Further reading

- [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://ai.meta.com/research/publications/v-jepa-latent-video-prediction-for-visual-representation-learning/)
- [VideoMAE v2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727) — explores tube vs. random masking head-to-head
- [Masked Autoencoders As Spatiotemporal Learners (MAE-ST)](https://arxiv.org/abs/2205.09113) — the paper that named the "tube" variant
