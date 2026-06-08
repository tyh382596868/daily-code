---
date: 2026-06-04
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/loss/gram_loss.py
permalink: https://github.com/facebookresearch/dinov3/blob/50001c6db58dbca7e7d06a5c5a9f1e078ca29197/dinov3/loss/gram_loss.py#L11-L84
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, dinov3, ssl, distillation, gram-matrix]
---

# DINOv3 用"两两相似度矩阵"作蒸馏目标 / DINOv3 distills the pairwise patch-similarity matrix, not the features

> **一句话 / In one line**: `GramLoss` 不蒸馏 patch 特征本身,而是蒸馏 patch 之间的内积矩阵 `X X^T` —— 这是一个对基底旋转不变的几何量,比直接对 features 做 MSE 更稳。 / `GramLoss` doesn't distill patch features directly; it distills the inner-product matrix `X X^T` between patches — a geometric quantity invariant to basis rotations and much more stable than feature-level MSE.

## 为什么重要 / Why this matters

DINOv3 把 dense features 的质量(分割、深度、跟踪这些下游任务能用的"逐像素"特征)又往前推了一大截,关键就是这个 Gram loss。直接拉近 student 和 teacher 的 patch 特征向量有个老问题:特征空间可以被任意正交矩阵旋转而不影响下游任务,但 MSE 会把它当成"错"。所以训练信号里有一大半是噪声。DINOv3 改成蒸馏 patch 两两之间的余弦相似度矩阵 —— 这个矩阵对正交变换是不变的,只关心"哪两个 patch 在 teacher 眼里像、在 student 眼里也得像"。代码只有 70 行,但这是 self-supervised 学习里近几年最大的设计转向之一。

DINOv3 made another big step forward on the quality of dense features (the per-patch features that downstream segmentation, depth, and tracking actually consume) and the headline ingredient is this Gram loss. Distilling patch features directly is fragile because the feature space can be rotated by any orthogonal matrix without changing downstream behavior — yet plain MSE punishes that rotation as if it were error. The training signal ends up half noise. DINOv3 instead distills the matrix of pairwise patch cosine similarities. That matrix is invariant under orthogonal transforms; it only cares "if two patches looked similar to the teacher, they should look similar to the student too." Seventy lines of code, but one of the biggest design shifts in self-supervised vision recently.

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

        self.mse_loss = torch.nn.MSELoss()

        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):
        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

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

        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)
```

## 逐行讲解 / What's happening

1. **`F.normalize(target_feats, dim=-1)` + `apply_norm` 分支**:
   - 中文: 先把每个 patch 特征向量 L2 归一化到单位球面。这一步是把"内积"自动等价为"余弦相似度",尺度被剔除掉,后面 `X X^T` 出来的就是 [-1, 1] 区间的相似度矩阵。
   - English: Each patch feature is L2-normalized onto the unit sphere first. This automatically makes the subsequent inner product behave as cosine similarity, so `X X^T` lives in [-1, 1] and is scale-invariant.
2. **`torch.matmul(target_feats, target_feats.transpose(-1, -2))`**:
   - 中文: 这一行是整个损失的灵魂 —— 形状 `(B, N, D) -> (B, N, N)`,每个 `(i, j)` 元素是图像里 patch i 和 patch j 之间的相似度。teacher 算一份,student 也算一份。
   - English: The one line that defines the whole loss. Shape `(B, N, D) -> (B, N, N)`; entry `(i, j)` is the similarity between patch i and patch j of the same image. Compute it for teacher, then again for student.
3. **`img_level=True` vs `flatten(0, 1)`**:
   - 中文: 默认 image-level,每张图独立算自己的 Gram。若改成 image-level=False,把 batch 整个 flatten 成 `(B*N, D)`,Gram 就变成"跨图 patch 对"也参与对比 —— 信号更密但容易把 batch 内的不同图像锁死成同一表示。
   - English: Default mode computes the Gram per image. If you switch off `img_level`, the batch is flattened to `(B*N, D)` and the Gram contains cross-image patch pairs too — denser signal but a risk of collapsing different images to the same representation.
4. **`remove_neg`**:
   - 中文: 把相似度 < 0 的项截到 0。直觉是负相似度多半是"两个无关 patch 的噪声",硬把 student 拉去匹配 teacher 的噪声反而有害。截掉只保留正相关。
   - English: Clip similarities below 0 to 0. Negative similarity is mostly noise from unrelated patches; forcing the student to match the teacher's noise is harmful. Keep only positive correlations.
5. **`mse_loss(student_sim, target_sim)`**:
   - 中文: 最后一行就是普通 MSE,但作用的对象从 `(B, N, D)` 特征矩阵变成了 `(B, N, N)` 相似度矩阵 —— 同样是 MSE,但梯度告诉 student 的是"几何关系怎么对齐",不是"分量值具体多少"。
   - English: A plain MSE, but applied to the `(B, N, N)` similarity matrix instead of the `(B, N, D)` features. Same loss function, completely different signal: the gradient tells the student how to match the *geometry* rather than the *coordinates*.

## 类比 / The analogy

想象你在教一个学生画地图。"普通 MSE"是:teacher 把每个城市的经纬度写下来,要求 student 一个数一个数对上 —— 但 teacher 用的可能是 UTM 坐标,student 用的是 WGS84,根本对不齐,但城市之间的相对位置其实一模一样。"Gram loss"是:teacher 不给坐标,只给"每两座城市之间的距离表"。无论你用哪套坐标系,只要城市之间的相对距离对得上,这张地图就是对的。

Imagine teaching a student to draw a map. Plain MSE is like the teacher writing down absolute latitude/longitude for every city and demanding the student match every number — but the teacher might use UTM and the student WGS84, so the numbers will never align even though the *map* is correct. Gram loss instead hands the student a table of pairwise city-to-city distances. Whatever coordinate system the student picks, as long as the relative distances match, the map is right.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn.functional as F

def gram_loss(student, teacher, remove_neg=True):
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.float(), dim=-1)
    s_sim = s @ s.transpose(-1, -2)
    t_sim = t @ t.transpose(-1, -2)
    if remove_neg:
        s_sim = s_sim.clamp(min=0)
        t_sim = t_sim.clamp(min=0)
    return F.mse_loss(s_sim, t_sim)

torch.manual_seed(0)
B, N, D = 2, 16, 64
teacher = torch.randn(B, N, D)

# student is teacher rotated by a random orthogonal matrix
Q, _ = torch.linalg.qr(torch.randn(D, D))
student_rot = teacher @ Q
print("MSE on features (rotated):", F.mse_loss(student_rot, teacher).item())
print("Gram loss   (rotated):", gram_loss(student_rot, teacher).item())

# now student is teacher + noise
student_noise = teacher + 0.5 * torch.randn_like(teacher)
print("MSE on features (noisy):", F.mse_loss(student_noise, teacher).item())
print("Gram loss   (noisy):", gram_loss(student_noise, teacher).item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
MSE on features (rotated): ~2.0      <- big!
Gram loss   (rotated): ~0.0          <- zero, as expected
MSE on features (noisy): ~0.25
Gram loss   (noisy): ~0.05           <- small but non-zero
```

中文重点:旋转 D 维特征后,feature-MSE 爆炸而 Gram loss 几乎为零 —— 这就是 DINOv3 想要的不变性。

The key thing to notice: after applying an arbitrary orthogonal rotation, feature-MSE explodes while Gram loss is essentially zero. This is exactly the invariance DINOv3 wanted.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Neural style transfer (Gatys et al.)** / **Neural style transfer**: 风格迁移的"风格"就是 VGG 特征的 Gram 矩阵 —— 比较 Gram 而不是特征值。 / The "style" in neural style transfer is precisely the Gram matrix of VGG features. Style is matched at the Gram level, not the feature level.
- **CKA (Centered Kernel Alignment)** / **CKA**: 神经网络层间相似度分析的标准工具,本质就是比较两层的 Gram。 / The standard tool for measuring similarity between neural network layers — also a Gram-matrix comparison.
- **Self-distillation in MoCo v3 / iBOT / EsViT** / **MoCo v3, iBOT, EsViT**: 早期版本都用 feature-level 蒸馏;DINOv3 走 Gram 路线后,后续的 V-JEPA2、SAM-3 也在跟进。 / Earlier versions used feature-level distillation. After DINOv3, V-JEPA2 and SAM-3 have started moving to Gram-style targets.

## 注意事项 / Caveats / when it breaks

- **`(B, N, N)` 矩阵在 N 大时很费显存** / **`(B, N, N)` blows up at high resolution**: 224 输入、patch=14 → N=256 还好,但放大到 16x16 = 1024 patch 时 Gram 矩阵就是 16M 元素 / batch 项。DINOv3 用 chunking 处理。 / At 224 input / patch=14 you get N=256 and the Gram is fine. Push resolution and N=1024 → 16M entries per batch item. DINOv3 chunks to keep memory in check.
- **`remove_neg` 不是无脑开** / **don't blindly enable `remove_neg`**: 关掉它会保留所有负相关信号,有些下游任务(细粒度区分、对比检索)反而更喜欢留着。所以才有两个独立开关 `remove_neg` 和 `remove_only_teacher_neg`。 / Turning it off keeps the negative-correlation signal, which can help fine-grained discrimination or retrieval. That's why there are two independent switches.
- **必须先 normalize** / **normalize is non-optional**: 不归一化的话 Gram 矩阵里既有"方向"又有"幅度",训练初期 student 的特征模长还在波动,MSE 会被幅度差主导,学不到几何。 / Without normalization the Gram entries carry both direction and magnitude. Early in training, the student's feature norms are still oscillating and MSE will be dominated by magnitude error, never learning the geometry.

## 延伸阅读 / Further reading

- DINOv3 paper: <https://arxiv.org/abs/2508.10104>
- Gatys et al., "A Neural Algorithm of Artistic Style" — origin of the Gram-matrix-as-style idea
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (CKA)
