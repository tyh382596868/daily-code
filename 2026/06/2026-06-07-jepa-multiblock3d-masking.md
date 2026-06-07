---
date: 2026-06-07
topic: diffusion
source: tracked
repo: facebookresearch/jepa
file: src/masks/multiblock3d.py
permalink: https://github.com/facebookresearch/jepa/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/src/masks/multiblock3d.py#L66-L203
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, masking, video, self-supervised, v-jepa]
---

# V-JEPA 的 3D 块掩码:140 行 torch 就能逼模型学"视频物理" / V-JEPA's 3D block masking: 140 lines of plain torch that force the model to learn video physics

> **一句话 / In one line**: V-JEPA 不是随机扣单个 patch,而是在 `(T/tubelet, H/patch, W/patch)` 的三维网格上扣一整块连续 tubes,然后用补集当 encoder 输入——`argwhere(mask == 0)` 一句话搞定 token 索引. / V-JEPA doesn't mask random patches — it carves coherent spatio-temporal *tubes* out of a `(T/tubelet, H/patch, W/patch)` grid, then takes the complement as the encoder's input — and `argwhere(mask == 0)` is the whole token-extraction trick.

## 为什么重要 / Why this matters

随机扣 patch 的 MAE/MIM 在静态图像上能跑,但放到视频上很快变成"插帧任务":相邻帧补一帧,模型只要做最近邻就能猜对,根本学不到运动. V-JEPA-2 / VideoMAE 系列改成扣 3D 块——同一空间位置、连续若干帧一起被遮——这样补全任务就强迫模型预测"接下来这块区域会变成什么",变成真正的视频理解. 这 140 行代码就是这个机制的全部:三维 patch 网格上做拒绝采样,采几个 `(t, h, w)` 块求交集得到 encoder mask,补集就是 predictor mask. 没有 CNN、没有 transformer,只有 torch 张量索引——但它是 V-JEPA 整个预训练任务的"出题人".

Random per-patch MAE/MIM works on still images, but on video it collapses into frame interpolation: mask a frame and the model just copies the neighbours. V-JEPA-2 / VideoMAE-style methods mask **3D blocks** instead — the same spatial region is hidden across several consecutive frames — so reconstructing it forces the model to predict *how the region will evolve*. That's video understanding, not pixel inpainting. The 140 lines here are the entire mechanism: rejection-sample a few `(t, h, w)` blocks on a 3D patch grid, intersect them to get the encoder mask, take the complement as the predictor mask. No CNNs, no transformers — just torch index gymnastics — but this is the "exam writer" for the whole V-JEPA pretraining task.

## 代码 / The code

`facebookresearch/jepa` — [`src/masks/multiblock3d.py#L66-L203`](https://github.com/facebookresearch/jepa/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/src/masks/multiblock3d.py#L66-L203)

```python
class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start:start+t, top:top+h, left:left+w] = 0

        # Context mask will only span the first X frames
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

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
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
```

## 逐行讲解 / What's happening

1. **`__init__` 的网格设置 / Grid setup in `__init__`**:
   - 中文: `self.height, self.width = 224 // 16 = 14, 14`,`self.duration = 16 // 2 = 8`. 也就是说视频被 patch 化后变成 `8 × 14 × 14 = 1568` 个 token,后面所有的掩码都在这个三维网格上做.
   - English: `self.height, self.width = 224 // 16 = 14, 14` and `self.duration = 16 // 2 = 8`. After patchification the video lives on an `8 × 14 × 14 = 1568`-token grid, and every mask operation below happens on this grid.

2. **`Value('i', -1)` 多 worker 同步计数器 / Multi-worker counter (`Value('i', -1)`)**:
   - 中文: collator 会被 PyTorch DataLoader 的多个 worker 复制——但 `multiprocessing.Value` 是共享内存,加 `get_lock()` 保证全局唯一. 这样所有 worker 算出来的 batch,*predictor block size* 是确定性同步的(同一 step → 同一 seed),只有 *block 的位置*是各自随机的. 没有这个细节,跨 worker 的 mask 形状会对不上,collate 直接挂.
   - English: collators are forked across DataLoader workers, but `multiprocessing.Value` is shared memory and `get_lock()` makes increments atomic. The result is deterministic across workers: the same `step` value seeds the predictor *block size*, while the block *positions* are still independently random. Without this sync, mask shapes would disagree across workers and `default_collate` would explode.

3. **`_sample_block_size` 三连采样 / Three samples in `_sample_block_size`**:
   - 中文: 时间长度比例 `temporal_mask_scale`、空间面积比例 `spatial_mask_scale`、长宽比 `aspect_ratio` 各采一次,然后 `h = round(sqrt(num_keep * ar))`、`w = round(sqrt(num_keep / ar))`. 这是经典的"先定面积再定形状"技巧:面积控制难度,形状控制多样性.
   - English: it draws three independent uniform samples — temporal scale, spatial scale, aspect ratio — and converts area + aspect into `h, w` via `sqrt`. Classic "area first, shape second" trick: area controls task difficulty, aspect controls visual diversity.

4. **`_sample_block_mask` 在 3D 网格上掏一个洞 / Carving one hole in the 3D grid**:
   - 中文: 全 1 张量打底,然后 `mask[start:start+t, top:top+h, left:left+w] = 0`——一个连续的时空 cuboid 被置 0. 如果 `max_context_duration < self.duration`,后面的帧整段置 0(只看前 X 帧).
   - English: start from an all-ones tensor and zero out a contiguous `(t, h, w)` cuboid. If `max_context_duration < self.duration`, the trailing frames are also zeroed — context mask is restricted to the first X frames only.

5. **`__call__` 里的 `mask_e *= ...` 求交集 / Intersection via `mask_e *= ...`**:
   - 中文: 想扣多个 block 时,把每个 block 的"0 表示被扣"的张量直接乘起来——只要任一 block 在某位置是 0,乘积就是 0. 这就是 *npred 个 block 的并集被扣*. encoder 看到的是补集.
   - English: to mask multiple blocks, multiply per-block masks together. Any zero position in any block kills the product — that's the union of *npred* blocks being masked out, leaving the encoder the complement.

6. **`torch.argwhere(mask_e == 0)` vs `torch.nonzero(mask_e)`**:
   - 中文: 一句话把 (T, H, W) 的二值张量变成 1D token 索引列表. predictor 拿被扣位置的索引,encoder 拿保留位置的索引,后面拼回 token 序列特别快.
   - English: one line that turns a `(T, H, W)` binary tensor into 1-D token-index lists. Predictor gets the masked-position indices, encoder gets the kept-position indices — slot them back into a token sequence in O(N).

7. **`min_keep_enc` 取最小 / `min_keep_enc` = min over batch**:
   - 中文: 不同样本里的随机块面积不一样,最后 `default_collate` 要求形状一致——所以裁到 batch 内最短的长度. 这是 V-JEPA 实现里一个隐藏的"不对齐就裁短"约定,理解之后再去看 encoder 那边的 attention 处理就顺了.
   - English: per-sample block areas vary, but `default_collate` needs uniform shape, so each sample is truncated to the minimum over the batch. This "pad-to-min, not pad-to-max" convention is implicit but consequential — keep it in mind when you read the encoder-side attention code.

## 类比 / The analogy

把视频想象成一摞透明胶片(每一帧一张),所有胶片对齐叠在桌面上. 随机扣 patch 像是在每张胶片上用打孔器随机戳几个圆——你恢复一张图的某个洞,只需要看相邻几帧的同一坐标"复印过来". V-JEPA 的 3D 块掩码相当于用一个直角剪刀,**从顶端一路剪到底**——所有被剪到的胶片在那块矩形区域都是空的. 你想恢复它,就必须根据周围还剩的胶片"推理这一坨区域到底在发生什么运动",而不是抄旁边一帧.

Picture the video as a stack of transparent slides, one per frame, perfectly aligned on a desk. Random per-patch masking is like punching tiny holes in each slide at random spots — to reconstruct a hole, you just copy the same coordinate from the slide above or below. V-JEPA's 3D block mask is more like taking a rectangular cookie cutter and pressing it straight down through the *entire* stack — every slide loses the same rectangular region. Now there's no neighbouring slide to copy from; you have to *reason about what's happening in that volume of space-time* using only the remaining context. That's why this masking is the engine that makes V-JEPA learn motion instead of frame interpolation.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, math

torch.manual_seed(0)
T, H, W = 8, 14, 14          # 16 frames / tubelet=2, 224/16 patches
mask = torch.ones((T, H, W), dtype=torch.int32)

# carve one (t, h, w) block at a random position
t, h, w = 4, 6, 4
top  = torch.randint(0, H - h + 1, (1,)).item()
left = torch.randint(0, W - w + 1, (1,)).item()
start= torch.randint(0, T - t + 1, (1,)).item()
mask[start:start+t, top:top+h, left:left+w] = 0

flat = mask.flatten()
enc_idx  = torch.nonzero(flat).squeeze()
pred_idx = torch.argwhere(flat == 0).squeeze()
print("total tokens:", flat.numel())
print("encoder sees:", enc_idx.numel(), "tokens")
print("predictor target:", pred_idx.numel(), "tokens")
print("expected masked vol:", t*h*w)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
total tokens: 1568
encoder sees: 1472 tokens
predictor target: 96 tokens
expected masked vol: 96
```

注意 `pred_idx.numel()` 严格等于 `t * h * w = 4 * 6 * 4 = 96`——这是 3D 块掩码的"账"对得上的证据. encoder 看 1472 个,predictor 重建 96 个,加起来正好 1568,一个都没漏.

`pred_idx.numel()` is exactly `t * h * w = 96`, which is the bookkeeping check: a 3D block carves out exactly that many patches, the encoder sees the other 1472, and `1472 + 96 = 1568` covers every token in the grid. No off-by-ones.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **VideoMAE / VideoMAE-v2**: 用同样的 tubelet 思想,但块大小固定(75% mask ratio 是常数). V-JEPA 这里多了"块大小也是随机量"这一步,任务难度自适应.
- **VideoMAE / VideoMAE-v2**: same tubelet idea but with fixed block size (constant 75% mask ratio). V-JEPA generalizes by sampling block size too, so task difficulty self-adapts.
- **MAE for images** (`facebookresearch/mae`): 1D 随机 shuffle 索引取前 N 个当 keep,补集当 mask. 2D/3D 块在它基础上加了"几何连续性"约束.
- **MAE for images** (`facebookresearch/mae`): a 1-D random shuffle keeps the first N indices as the encoder set. The 3D-block variant here adds a *geometric continuity* constraint on top of that simple shuffle.
- **DiT / MM-DiT 视频训练**: 训练 video diffusion 时也可以用类似的 mask 当 condition——已知部分 latent,预测剩下的——这就是 image-to-video 任务的雏形.
- **DiT / MM-DiT video training**: video-diffusion finetuning can use exactly this kind of mask as a *condition* — keep some latents, predict the rest — which is the seed of image-to-video conditioning.

## 注意事项 / Caveats / when it breaks

- **`empty_context` while-loop 可能死循环**: 中文: 如果 `spatial_pred_mask_scale` 设到 `(0.95, 0.95)` 又 `npred=2`,encoder 几乎没有 token 可看——while 一直 reject. 论文里默认是 `(0.2, 0.8)` 加 `npred=1` 或 `npred=2` 时基本不会卡,但调参时要意识到这是个隐藏成本.
- **`empty_context` while-loop can spin**: English: cranking `spatial_pred_mask_scale` to `(0.95, 0.95)` with `npred=2` leaves almost no encoder tokens — the while-loop keeps rejecting. Default `(0.2, 0.8)` with `npred ∈ {1, 2}` is fine, but the rejection cost is invisible until you tune.
- **`min_keep_pred` 不是 padding,是裁剪**: 中文: batch 内最短的那条会拉低所有人. 如果你想保 token 多样性,加大 batch 或加大块面积下限.
- **`min_keep_pred` truncates, doesn't pad**: English: the shortest sample in a batch drags everyone down. If you need more tokens, raise the batch size or the lower bound of block area.
- **`tubelet_size` 必须能整除 `num_frames`**: 中文: 这是硬约束,代码不检查. 16 / 2 = 8 没事,17 / 2 就溢出了. 数据 pipeline 务必先做 frame-count 对齐.
- **`tubelet_size` must divide `num_frames`**: English: hard constraint, silently assumed by the code. 16/2 = 8 is fine; 17/2 truncates. Align your video pipeline first.

## 延伸阅读 / Further reading

- [V-JEPA-2 paper (Meta, 2025)](https://arxiv.org/abs/2402.10277) — the masking strategy is figure 2
- [VideoMAE (NeurIPS 2022)](https://arxiv.org/abs/2203.12602) — predecessor with fixed tubelet ratio
- [V-JEPA training entry point](https://github.com/facebookresearch/jepa/blob/main/src/train.py) — how the masks above feed into the encoder/predictor
- [Resampling rationale in I-JEPA](https://arxiv.org/abs/2301.08243) — same multi-block sampling, applied to single images
