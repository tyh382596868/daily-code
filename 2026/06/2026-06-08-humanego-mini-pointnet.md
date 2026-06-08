---
date: 2026-06-08
topic: diffusion
source: trending
repo: TX-Leo/HumanEgo
file: training/FlowMatchingModel.py
permalink: https://github.com/TX-Leo/HumanEgo/blob/1eece2f2a090ac30662d277ab358afed7c0bf2b8/training/FlowMatchingModel.py#L54-L81
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, trending, robotics, pointnet, 3d-geometry, flow-matching]
---

# 28 行 MiniPointNet:把 64 个点塞进一个 token / 28-line MiniPointNet: cram 64 points into one token

> **一句话 / In one line**: 两层 `Conv1d → MaxPool → Linear`,就能把每个物体的 64 点点云压成一个跟图像 token 对齐的向量,加几行代码 transformer 就懂"3D"了。 / Two `Conv1d → MaxPool → Linear` layers compress each object's 64-point cloud into one vector dimension-matched to the image tokens — a few lines of code teach a transformer "3D".

## 为什么重要 / Why this matters

HumanEgo 是这两周突然冒出来的一个很有意思的项目:它从 Meta Aria 眼镜的几分钟人类自我视角视频里学习机器人策略,完全 zero-shot——你只要戴眼镜做一遍任务,模型就能在真机器人上复刻。要做到这一点,模型必须同时拿到 2D 视觉 token、状态 token、以及**显式的 3D 几何**(物体在空间里的点云)。前两者大家都会,但"点云怎么塞进 transformer"才是这个项目最值得偷的小技巧。这 28 行的 MiniPointNet 给了一个标准答案——它把 PointNet 原论文里那个核心 idea 浓缩到了最朴素的形态,任何 policy 都能直接抄。

HumanEgo is a fresh project from the last couple of weeks: it learns robot policies from just a few minutes of human-egocentric Aria-glasses video — zero-shot. Wear the glasses, do the task once, and the model imitates it on a real robot. To pull that off the model has to consume 2D vision tokens, state tokens, **and explicit 3D geometry** (object point clouds) at once. Vision and state are well-trodden ground; the steal-worthy bit is "how do you cram a point cloud into a transformer?" This 28-line MiniPointNet is the canonical answer — the core PointNet idea reduced to its barest form, drop-in-able for any policy.

## 代码 / The code

`TX-Leo/HumanEgo` — [`training/FlowMatchingModel.py`](https://github.com/TX-Leo/HumanEgo/blob/1eece2f2a090ac30662d277ab358afed7c0bf2b8/training/FlowMatchingModel.py#L54-L81)

```python
class MiniPointNet(nn.Module):
    """ Encodes (64, 3) point cloud into a single feature vector per token. """
    def __init__(self, out_dim: int):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, T_ict, 64, 3)
        B, T, N, _ = pts.shape
        x = pts.view(B * T, N, 3).transpose(1, 2)  # (B*T, 3, 64)

        x = self.mlp1(x)               # (B*T, 128, 64)
        x = torch.max(x, dim=2)[0]     # Max Pooling -> (B*T, 128)

        x = self.mlp2(x)               # (B*T, out_dim)
        return x.view(B, T, -1)        # (B, T, out_dim)
```

## 逐行讲解 / What's happening

1. **`nn.Conv1d(3, 64, 1)` —— kernel=1 的卷积**:
   - 中文: kernel size = 1 的 1D Conv 数学上等价于"对每个点独立做一次 Linear(3 → 64)"——也就是说,每个点都被 *相同的* MLP 独立编码,点和点之间不交互。这是 PointNet 的精髓。
   - English: a kernel-1 1D-Conv is mathematically a per-point `Linear(3 → 64)` — every point is encoded independently by the *same* MLP, with no point-to-point interaction. That's the PointNet essence.

2. **`pts.view(B*T, N, 3).transpose(1, 2)`**:
   - 中文: 把 `(B, T_ict, 64, 3)` 拍平 batch×token 维到一起,然后把"点维"和"通道维"换位。Conv1d 想要 `(batch, channels, length)`,这里 channels=3(xyz),length=64(64 个点)。
   - English: flatten batch×token into one axis, then swap "points" and "channels". Conv1d expects `(batch, channels, length)`; here channels=3 (xyz) and length=64 (the 64 points).

3. **`torch.max(x, dim=2)[0]` —— 对称 max pooling**:
   - 中文: 这是 PointNet 里**最关键的一步**——按点取 max,得到一个 128 维向量。max 是对称函数,所以 64 个点不管顺序怎么打乱,输出都不变。这就解决了"点云没有天然顺序"的根本问题。
   - English: the **single most important step** in PointNet — element-wise max across points, yielding a 128-dim vector. Max is symmetric, so shuffle the 64 points however you like and the output is unchanged. That resolves the fundamental "point clouds have no canonical order" problem.

4. **`mlp2 = Linear(128) → Linear(out_dim)`**:
   - 中文: max pooling 后已经是"一个 token 一个向量"了,再过两层 MLP 把 128 维投影到目标维度,跟图像 token 对齐就能直接加。
   - English: after max pooling we already have "one vector per token"; two more MLP layers project from 128 to the target dim so it's dimension-aligned with image tokens and can be summed directly.

5. **`.view(B, T, -1)`**:
   - 中文: 还原回 `(B, T_ict, out_dim)`,正好可以在 `FlowMatchingModel` 里:`ict_tokens = ict_tokens + self.pcd_alpha * pcd_feats`,以一个可学习权重 `pcd_alpha` 加进 token 序列。
   - English: reshape back to `(B, T_ict, out_dim)`; in `FlowMatchingModel` this is added to the ICT tokens as `ict_tokens = ict_tokens + self.pcd_alpha * pcd_feats` with a learnable scalar `pcd_alpha`.

## 类比 / The analogy

想象一个袋子里装了 64 颗弹珠,每颗颜色和大小都不同——你想用一个标签描述这袋弹珠,但你绝不希望换换弹珠在袋里的顺序就改变标签内容。怎么办?对每颗弹珠先单独打一个性质描述卡(per-point MLP),然后从全部卡片里挑出"最红的"、"最大的"、"最重的"等极值(max pooling)——这堆极值合起来就是袋子的描述,跟摆放顺序完全无关。MiniPointNet 就是这种"独立打卡 + 取极值"的最小化实现。

Picture a bag of 64 marbles, each a different colour and size — you want a single label for the bag, but you absolutely do not want the label to change when you shake the bag and reorder the marbles. The trick: stamp a description card for each marble independently (per-point MLP), then pick the extreme values across all cards — the reddest, the biggest, the heaviest (max pooling). That set of extremes is the bag's label, order-independent. MiniPointNet is that "stamp-and-extreme" idea at its minimum size.

## 自己跑一遍 / Try it yourself

```python
# try.py
import torch, torch.nn as nn

class MiniPointNet(nn.Module):
    def __init__(self, out_dim=384):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )
    def forward(self, pts):                    # pts: (B, T, 64, 3)
        B, T, N, _ = pts.shape
        x = pts.view(B*T, N, 3).transpose(1, 2)
        x = self.mlp1(x)
        x = x.max(dim=2).values                # symmetric pooling
        return self.mlp2(x).view(B, T, -1)

torch.manual_seed(0)
net = MiniPointNet(out_dim=8).eval()
pts = torch.randn(1, 1, 64, 3)                 # one token, 64 points
shuffled = pts[:, :, torch.randperm(64), :]    # same points, different order

with torch.no_grad():
    feat_orig = net(pts)
    feat_shuf = net(shuffled)

print(f"feat (original order) : {feat_orig.flatten().tolist()}")
print(f"feat (shuffled order) : {feat_shuf.flatten().tolist()}")
print(f"max abs diff          : {(feat_orig - feat_shuf).abs().max().item()}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
feat (original order) : [..., ..., 8 numbers]
feat (shuffled order) : [same 8 numbers]
max abs diff          : 0.0
```

完全相等——这就是排列不变性。换成 `mean` 也对称,但论文实验表明 `max` 抓极端值更稳。

Bit-exact equality — that's permutation invariance in action. `mean` is also symmetric, but PointNet's experiments show `max` (which picks extremes) is more robust.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PointNet 原论文 (Qi et al., 2017)**: 这段代码的直接前身,只是去掉了 T-Net 对齐分支 / The direct ancestor of this code; only the T-Net alignment branch is dropped.
- **DeepSets / Set Transformer**: 同样的"先 per-element MLP,再对称 pool"框架被推广到任意集合 / The "per-element MLP then symmetric pool" pattern generalised to arbitrary sets.
- **3D Diffuser Actor / ChainedDiffuser**: 机器人 diffusion policy 也用 PointNet 级别的 encoder 处理本体感觉+3D 关键点 / Diffusion robot policies use PointNet-grade encoders for proprioception + 3D keypoints.
- **DP3 (Diffusion Policy 3D)**: 用 PointNet++ 多层级的 encoder,精度提升明显 / Uses PointNet++'s multi-level encoder for noticeable accuracy gains.

## 注意事项 / Caveats / when it breaks

- **`BatchNorm1d` 在 batch=1 时会塌** / **`BatchNorm1d` collapses at batch size 1**: 调到 eval 模式或换 `LayerNorm`,常见的真机推理坑 / Switch to eval mode or `LayerNorm` — a frequent real-robot-inference footgun.
- **没有平移/旋转不变性** / **No translation / rotation invariance**: MiniPointNet 只解决了顺序问题,几何变换不变性还得靠数据增强或显式坐标系对齐 / It only solves order; geometry-transform invariance needs augmentation or explicit frame alignment.
- **64 点对复杂物体太少** / **64 points may be too few**: 对薄壳几何(纸张、抹布)信号不够,要么涨点数(>256)、要么改用法向量+点 / Thin-shell geometry (paper, cloth) loses information; raise the point count (>256) or stack normals on top of coordinates.
- **`max` 对噪声敏感** / **`max` is noise-sensitive**: 一颗野点可能霸占整张 feature;数据要预先做 outlier filtering / A single outlier can dominate the feature vector — filter outliers first.

## 延伸阅读 / Further reading

- PointNet paper: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (Qi et al., 2017)
- PointNet++ paper (Qi et al., 2017) — hierarchical version, much stronger on dense geometry
- HumanEgo repo README — the broader context of how this feeds the Flow Matching policy
- Deep Sets paper (Zaheer et al., 2017) — the theoretical justification for "per-element MLP + symmetric pool"
