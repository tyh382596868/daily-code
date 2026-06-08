---
date: 2026-06-08
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/loss/gram_loss.py
permalink: https://github.com/facebookresearch/dinov3/blob/50001c6db58dbca7e7d06a5c5a9f1e078ca29197/dinov3/loss/gram_loss.py#L11-L84
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, ssl, distillation, gram-matrix, dinov3]
---

# DINOv3 的 Gram Loss:不蒸特征,蒸"特征之间的关系" / DINOv3's Gram Loss: don't distill features — distill the *relationships between* features

> **一句话 / In one line**: 学生和教师不需要输出相同的 patch 特征,只要 patch-patch 的相似度矩阵(Gram 矩阵)一致即可,这样模型可以自由旋转表征空间却仍然保留几何结构。 / The student doesn't have to match the teacher's patch features — it only has to match the teacher's patch-patch similarity matrix (the Gram matrix), so the model is free to rotate its representation space while preserving geometric structure.

## 为什么重要 / Why this matters

特征级的 KD(MSE on features)有一个隐藏的坑:它要求学生在 *同一个坐标系* 下复现老师的输出。问题是这个坐标系本身是没有意义的——把所有特征同时乘一个旋转矩阵,下游任务表现完全一样,但 MSE 会爆。DINOv3 用 Gram loss 把"绝对位置"约束换成"相对关系"约束:学生只需要保持 patch 之间的相对夹角和距离不变,基底可以随便旋。这正是密集预测任务(分割、深度、检测)所真正依赖的东西——空间关系,不是绝对坐标。70 行代码,完整传达了这个想法。

Feature-level KD (MSE on features) has a subtle pitfall: it forces the student to reproduce the teacher's output *in the teacher's coordinate frame*. But that frame is arbitrary — multiply every feature by the same rotation matrix and downstream tasks are unchanged, yet the MSE blows up. DINOv3's Gram loss swaps an "absolute position" constraint for a "relative relationships" constraint: the student only has to preserve pairwise angles/distances between patches; the basis is free to rotate. That is precisely what dense-prediction tasks (segmentation, depth, detection) actually rely on — spatial relationships, not absolute coordinates. The whole idea lands in 70 lines.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/loss/gram_loss.py`](https://github.com/facebookresearch/dinov3/blob/50001c6db58dbca7e7d06a5c5a9f1e078ca29197/dinov3/loss/gram_loss.py#L11-L84)

```python
class GramLoss(nn.Module):
    """Implementation of the gram loss"""

    def __init__(
        self,
        apply_norm=True,
        img_level=True,
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()

        # Loss
        self.mse_loss = torch.nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):
        """Compute the MSE loss between the gram matrix of the input and target features."""

        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        # Float casting
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # SSL correlation
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            target_feats = target_feats.flatten(0, 1)

        # Compute similarities
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # Patch correlation
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            output_feats = output_feats.flatten(0, 1)

        # Compute similarities
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)
```

## 逐行讲解 / What's happening

1. **`F.normalize(feats, dim=-1)`**:
   - 中文: 把每个 patch 特征向量归一到单位长度,这样 `feats @ feats.T` 算出来的就是 cosine 相似度,而不是带模长的内积——模长会让大特征压制小特征,失去关系信息。
   - English: unit-normalise each patch vector so that `feats @ feats.T` is cosine similarity rather than a magnitude-weighted dot product — magnitudes would let large features dominate and wash out the relationships.

2. **`torch.matmul(feats, feats.transpose(-1, -2))`**:
   - 中文: 这一行就是 Gram 矩阵本体。形状 `(B, N, dim) @ (B, dim, N) → (B, N, N)`。每个 `(B, i, j)` 元素是 patch i 和 patch j 的 cosine 相似度。
   - English: this is the Gram matrix itself. Shape `(B, N, dim) @ (B, dim, N) → (B, N, N)`. Each `(B, i, j)` entry is the cosine similarity between patch i and patch j.

3. **`if not img_level: feats.flatten(0, 1)`**:
   - 中文: 把 batch 拍平到 patch 维,允许跨图比相似度。`img_level=True` 时只在图内比;`False` 时让一张图的 patch 也能和另一张图的 patch 形成关系,蒸馏信号更密。
   - English: flatten batch into the patch axis to allow cross-image similarity. With `img_level=True` you only compare patches within an image; with `False` patches from different images can also form relationships, giving a denser distillation signal.

4. **`target_sim[target_sim < 0] = 0.0`**:
   - 中文: 负相似度被裁成 0。意思是"老师认为这两个 patch 不像"就不要学生也精确复现这个负值——只学正相关。对噪声鲁棒,而且和很多 SSL 损失里"只用正样本"的思路一致。
   - English: negative similarities are clamped to 0. The intuition: if the teacher says "these two patches are dissimilar" don't force the student to reproduce that exact negative value — only learn the positive correlations. This is noise-robust and matches the "positive-only" stance of many SSL losses.

5. **`return self.mse_loss(student_sim, target_sim)`**:
   - 中文: 一条 MSE 把两张 `(N, N)` 相似度矩阵对齐。注意:学生的特征向量本身没有出现在 loss 里,只有它们 *之间的关系*。
   - English: one MSE aligns the two `(N, N)` similarity matrices. Note: the student's feature vectors themselves never appear in the loss — only the *relations between them*.

## 类比 / The analogy

想象你在教一个机器人画画。"特征蒸馏"等于你说:"把这只猫的眼睛精确画在 (320, 180)、鼻子画在 (340, 200)"——一旦机器人把画布旋转了 30 度,坐标全错,惩罚极大,可画的猫其实和老师一模一样。"Gram loss"则等于你只说:"鼻子离眼睛 30 像素,嘴在鼻子下方一个鼻子距离"——所有相对几何描述,画在哪都行,只要彼此关系对就给分。

Imagine teaching a robot to draw. "Feature distillation" tells it "put the cat's eyes at (320, 180), nose at (340, 200)" — rotate the canvas 30° and every coordinate is wrong, the loss explodes, even though the drawing is identical to the teacher's. "Gram loss" instead says "nose is 30 pixels from the eyes, mouth is one nose-length below the nose" — purely relational. Draw it anywhere, in any orientation; as long as the pairwise relations hold, full credit.

## 自己跑一遍 / Try it yourself

```python
# try.py
import torch, torch.nn as nn, torch.nn.functional as F

class GramLoss(nn.Module):
    def __init__(self): super().__init__(); self.mse = nn.MSELoss()
    def forward(self, s, t):
        s = F.normalize(s.float(), dim=-1); t = F.normalize(t.float(), dim=-1)
        ss = s @ s.transpose(-1, -2); ts = t @ t.transpose(-1, -2)
        ts[ts < 0] = 0; ss[ss < 0] = 0
        return self.mse(ss, ts)

torch.manual_seed(0)
teacher = torch.randn(2, 16, 64)                      # (B=2, N=16 patches, dim=64)
# Case A: student is a rotated copy of teacher → cosine relationships preserved
R = torch.linalg.qr(torch.randn(64, 64))[0]
student_rot = teacher @ R
# Case B: random unrelated student
student_rand = torch.randn(2, 16, 64)
loss = GramLoss()
print(f"loss(teacher, rotated-teacher)  = {loss(student_rot, teacher).item():.6f}")
print(f"loss(teacher, random-student)   = {loss(student_rand, teacher).item():.6f}")
print(f"feature-MSE rotated vs teacher  = {(student_rot - teacher).pow(2).mean().item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
loss(teacher, rotated-teacher)  = 0.000000
loss(teacher, random-student)   = ~0.06
feature-MSE rotated vs teacher  = ~2.0
```

旋转过的学生在普通 feature MSE 下损失巨大(≈2.0),但在 Gram loss 下损失精确为 0——这就是 Gram loss "旋转不变"的字面证明。

The rotated student has a huge feature MSE (~2.0) but Gram loss is exactly 0 — that's the literal proof of Gram loss's rotation invariance.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Style transfer 的 Gatys 损失 / Gatys' style loss**: 同样的 Gram 矩阵公式,只不过那里是用 conv 特征算 channel-channel Gram 来描述"风格" / Same Gram matrix formula, but there it's channel-channel Gram on conv features to encode "style".
- **CKA (Centered Kernel Alignment)**: 比较两个模型表征的常用指标,数学骨架等价于"先 Gram 再相似度" / Standard tool for comparing two models' representations; the math skeleton is "Gram first, then similarity".
- **DINOv2 / iBOT 的 patch-level 对比损失 / DINOv2 and iBOT's patch-level contrastive losses**: 也是在 patch 之间建关系,Gram loss 可以看成它们的"全密集"版本 / Also build patch-level relations; Gram loss can be read as a "fully dense" variant of them.

## 注意事项 / Caveats / when it breaks

- **N² 内存爆炸 / N² memory blow-up**: Gram 矩阵是 `(N, N)`,N=4096 个 patch 的话仅这一个 tensor 就 64 MB(fp32),反向再翻倍。要用大分辨率,先做下采样或 chunk / The Gram is `(N, N)` — with 4096 patches that single tensor is 64 MB in fp32 and doubles for backward. High-resolution training needs downsampling or chunked Gram.
- **`apply_norm=False` 会爆数值 / `apply_norm=False` explodes numerically**: 不归一化时 Gram 量级和 dim 成正比,fp16/bf16 下容易 NaN。代码默认开归一化是有道理的 / Without normalization the Gram scales with dim and easily NaNs in fp16/bf16 — the default is on for good reason.
- **裁负值会丢一部分信息 / Clamping negatives discards information**: 如果你的下游任务恰好依赖"反相关"信号(比如对比表征),`remove_neg=True` 会变成净损失。先做 ablation / If your downstream actually needs anti-correlation signal (e.g. contrastive uses), `remove_neg=True` becomes a net loss — ablate first.

## 延伸阅读 / Further reading

- DINOv3 paper: https://arxiv.org/abs/2508.10104 (the Gram loss is one of three new ingredients vs DINOv2)
- Gatys et al., "A Neural Algorithm of Artistic Style" — the original Gram-matrix-as-style paper
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (CKA)
