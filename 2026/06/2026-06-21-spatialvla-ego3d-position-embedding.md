---
date: 2026-06-21
topic: robotics
source: trending
repo: SpatialVLA/SpatialVLA
file: model/modeling_spatialvla.py
permalink: https://github.com/SpatialVLA/SpatialVLA/blob/18fd74b2a633ec8d9ec7aadcd803969555cc9fbd/model/modeling_spatialvla.py#L43-L100
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, vla, spatial-awareness, nerf, fourier-features, 3d-position-embedding]
---

# NeRF 风格的 3D 位置嵌入：让 VLM 真正知道每个视觉 patch 在空间中的位置 / NeRF-Style 3D Position Embeddings: Giving the VLM Actual Spatial Awareness

> **一句话 / In one line**: SpatialVLA 用 ZoeDepth 估计深度图，将每个 ViT patch 反投影到相机坐标系的 3D 点，再用 NeRF 同款的对数间隔正弦余弦特征编码这些 3D 坐标，把生成的位置嵌入加到 patch token 上——模型从此知道"这个 patch 距我 0.8 米、偏右 15 度"。 / SpatialVLA uses ZoeDepth to estimate a depth map, backprojects each ViT patch into 3D camera-space coordinates, encodes those 3D points with log-spaced sin/cos Fourier features (same as NeRF), and adds the resulting position embedding to patch tokens — so the model literally knows "this patch is 0.8 m away, 15 degrees to my right."

## 为什么重要 / Why this matters

标准 VLM（如 LLaVA、OpenVLA）把图像切成 patch 后，每个 patch 只有一个 2D 位置嵌入（行列索引），模型对"这两个 patch 对应的物体在三维空间中有多远"完全没有感知。这对桌面抓取这类任务不是大问题，但对需要精确 6-DoF 控制的操作（例如精密装配、穿孔定位）就很致命——模型不知道深度，就算看起来在同一位置，实际距离可能差 0.3 m。

SpatialVLA 的 `Ego3DPositionEmbeddingMLP` 把这个缺失的 3D 空间感知补上来：
1. ZoeDepth 给每个像素预测深度值。
2. `backproject_patch` 把 ViT patch 中心点用相机内参 K 反投影到 3D 相机坐标 (x, y, z)。
3. `Ego3DPositionEmbeddingMLP` 用 NeRF 同款的频率编码把 (x, y, z) 变成一个与 ViT token 同维度的嵌入向量，加到 patch token 上。

结果：LM backbone 在第一层就已经知道每个 patch 在 ego 坐标系下的 3D 位置——不是语义标签，而是真实的度量空间坐标。

Standard VLMs (like LLaVA, OpenVLA) give each image patch only a 2D position embedding (row/column index), leaving the model completely unaware of how far apart two patches are in 3D space. This is acceptable for simple grasping, but fatal for tasks requiring precise 6-DoF control (precision assembly, hole alignment) — without depth, two patches that look co-located could be 0.3 m apart in reality.

SpatialVLA's `Ego3DPositionEmbeddingMLP` fills this gap:
1. ZoeDepth predicts a per-pixel depth value.
2. `backproject_patch` uses camera intrinsics K to backproject each ViT patch center to 3D camera coordinates (x, y, z).
3. `Ego3DPositionEmbeddingMLP` encodes (x, y, z) with NeRF-style frequency features into a vector matching the ViT token dimension, then adds it to the patch token.

Result: the LM backbone already knows each patch's 3D position in ego-space from layer one — not a semantic label, but real metric-space coordinates.

## 代码 / The code

`SpatialVLA/SpatialVLA` — [`model/modeling_spatialvla.py`](https://github.com/SpatialVLA/SpatialVLA/blob/18fd74b2a633ec8d9ec7aadcd803969555cc9fbd/model/modeling_spatialvla.py#L43-L100)

```python
class Ego3DPositionEmbeddingMLP(nn.Module):
    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True):
        super(Ego3DPositionEmbeddingMLP, self).__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)

        center = torch.tensor([0., 0., 2.]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """init with small weights to maintain stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        x in [-2, 2], y in [-2, 2], z in [0., 4]
        """
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding
```

## 逐行讲解 / What's happening

1. **`freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)` — 对数间隔频率**:
   - 中文: 对数间隔的频率 [1, 2, 4, 8, ..., 2^(n_freqs-1)]，这与 NeRF 原论文完全相同。对数间隔比线性间隔更好：低频分量捕捉大尺度空间结构（房间级），高频分量捕捉精细几何（毫米级）。
   - English: Log-spaced frequencies [1, 2, 4, 8, ..., 2^(n_freqs-1)], identical to the original NeRF paper. Log-spacing is better than linear: low-frequency components capture coarse spatial structure (room-level), high-frequency components capture fine geometry (millimeter-level).

2. **`center = torch.tensor([0., 0., 2.])` — 归一化中心**:
   - 中文: 相机前方 2 m 处（z=2）作为深度中心，(x, y) 中心在 0。选 z=2 是因为机器人操作的典型工作距离在 0.5-3.5 m，以 2 m 为中心后归一化到 [-1, 1] 能均匀覆盖这个范围。
   - English: Centers depth at 2 m in front of the camera (z=2), with (x, y) centered at 0. z=2 is chosen because typical robot manipulation workspace is 0.5-3.5 m away; centering at 2 m then normalizing to [-1, 1] uniformly covers this range.

3. **`xyz_n = (xyz - self.center) / 2.0` — 归一化**:
   - 中文: 将原始 3D 坐标平移并缩放到大约 [-1, 1] 范围，使正弦余弦特征的采样更均匀，避免频率组件在极值区间的精度损失。
   - English: Shift and scale raw 3D coordinates to approximately [-1, 1], making sin/cos feature sampling more uniform and avoiding precision loss at extreme values.

4. **`xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands` 和 `sin/cos`**:
   - 中文: 对每个坐标分量（x, y, z）分别乘以所有频率 [1, 2, 4, ...]，再取 sin 和 cos，得到 `2×n_freqs` 个特征。加上原始归一化坐标本身（1 维），每个坐标轴贡献 `2*n_freqs + 1` 维特征，3 个坐标共 `3 × (2*8+1) = 51` 维。
   - English: Multiply each coordinate component (x, y, z) by all frequencies [1, 2, 4, ...], take sin and cos to get `2×n_freqs` features. Adding the normalized coordinate itself (1-dim), each axis contributes `2*n_freqs + 1` features; 3 axes total = `3 × (2*8+1) = 51` dimensions.

5. **`position_embedding_head` — 两层 MLP 投影**:
   - 中文: 把 51 维的频率特征投影到与 ViT patch token 相同的维度（默认 768）。中间加了 LayerNorm 和 ReLU，以及 `gain=0.01` 的 Xavier 初始化——极小的初始权重确保训练开始时 3D 位置嵌入只做微小扰动，不破坏预训练 ViT 特征。
   - English: Projects the 51-dim frequency features to match the ViT patch token dimension (default 768). LayerNorm and ReLU in between, plus `gain=0.01` Xavier init — tiny initial weights ensure the 3D position embedding starts as a small perturbation, preserving pretrained ViT features at the start of training.

## 类比 / The analogy

标准 2D patch 位置嵌入就像在一张地图上给每个街区标了"第 3 行第 5 列"。但地图不知道你在山上还是山谷里。SpatialVLA 的 3D 位置嵌入就像给每个街区加上了 GPS 高程信息，而 NeRF 的频率编码就像把这个高程用一首"音乐"表达出来——低频音符描述"我大概在哪个山头"，高频音符描述"我距离悬崖边缘精确几厘米"。

Standard 2D patch position embeddings are like labeling each city block "row 3, column 5" on a flat map. But the map doesn't know if you're on a hilltop or in a valley. SpatialVLA's 3D position embedding is like adding GPS elevation to each block, and the NeRF frequency encoding is like expressing that elevation as a piece of "music" — low-frequency notes say "I'm roughly on which hill," high-frequency notes say "I'm exactly N centimeters from the cliff edge."

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

class Ego3DPosEmb(nn.Module):
    def __init__(self, n_freqs=8, out_dim=768):
        super().__init__()
        freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        center = torch.tensor([0., 0., 2.])
        self.register_buffer("freq_bands", freq_bands)
        self.register_buffer("center", center)
        freq_out = 3 * (2 * n_freqs + 1)
        self.mlp = nn.Sequential(
            nn.Linear(freq_out, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def encode(self, xyz):
        n = (xyz - self.center) / 2.0
        f = n.unsqueeze(-1) * self.freq_bands
        enc = torch.cat([n.unsqueeze(-1), torch.sin(f), torch.cos(f)], -1)
        return enc.reshape(*xyz.shape[:2], -1)

    def forward(self, xyz):
        return self.mlp(self.encode(xyz))

# Simulate: B=2 images, N=196 patches (14x14 ViT), each with (x,y,z)
xyz = torch.randn(2, 196, 3)
xyz[..., 2] = xyz[..., 2].abs() + 0.5   # z must be positive (depth)

emb = Ego3DPosEmb(n_freqs=8, out_dim=768)
pos_embedding = emb(xyz)
print("position embedding shape:", pos_embedding.shape)  # (2, 196, 768)
print("encoding dim before MLP:", 3 * (2*8 + 1))         # 51
print("value range:", pos_embedding.min().item(), "to", pos_embedding.max().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
position embedding shape: torch.Size([2, 196, 768])
encoding dim before MLP: 51
value range: -0.02... to 0.02...
```

中文：注意输出值范围很小（约 ±0.02）——这是 `gain=0.01` Xavier 初始化的效果。把这个嵌入直接加到 ViT patch token（通常 norm 后范围约 [-1, 1]）上，初始时相当于加了微小噪声，不破坏预训练特征；训练过程中 MLP 权重逐渐增大，3D 空间感知越来越强。

English: Notice the small output range (≈ ±0.02) — this is the `gain=0.01` Xavier init effect. Adding this to ViT patch tokens (typically ≈ [-1, 1] after norm) starts as a tiny perturbation that preserves pretrained features; during training the MLP weights grow and 3D spatial awareness strengthens progressively.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NeRF 原论文的位置编码** / **NeRF positional encoding (Mildenhall et al., 2020)**: 用相同的 `(x, sin(2^k·x), cos(2^k·x))` 特征把 3D 坐标送进 MLP，让 MLP 拟合高频几何细节。SpatialVLA 借用了完全相同的频率设计。 / Uses the identical `(x, sin(2^k·x), cos(2^k·x))` features to feed 3D coordinates into an MLP for high-frequency geometry fitting. SpatialVLA borrows the exact same frequency design.
- **Deformable DETR / 3DETR 的 3D 位置编码** / **Deformable DETR / 3DETR**: 点云检测中用相同的 sin/cos 编码 3D 点，再加到 transformer 的 Q/K 上。区别在于这些方法直接用激光雷达深度，SpatialVLA 用 monocular depth estimation。 / Point-cloud detection encodes 3D points with identical sin/cos and adds them to transformer Q/K. The difference: these methods use LiDAR depth; SpatialVLA uses monocular depth estimation.
- **EgoVLM / EgoVideo** / **自我中心视频理解**: 自我中心视频分析里，使用相机内外参把视频帧投影到 3D 的思路逐渐普及，尤其在 AR/VR 和机器人场景。 / Egocentric video analysis increasingly uses camera intrinsics/extrinsics to project frames into 3D, especially in AR/VR and robotics contexts.

## 注意事项 / Caveats / when it breaks

- **依赖深度估计质量** / **Depends on depth estimation quality**: ZoeDepth 在纹理少（白墙、玻璃）或遮挡严重的区域深度估计误差大，3D 位置嵌入随之出错。如果机器人遭遇这类场景，3D 感知反而可能引入噪声。 / ZoeDepth has large depth estimation errors on textureless surfaces (white walls, glass) or heavy occlusion. In these scenes, the 3D position embedding may introduce noise rather than help.
- **训练时需要 ZoeDepth 推理** / **ZoeDepth inference at training time**: 每张训练图像都需要过一次 ZoeDepth，显著增加数据预处理时间。生产系统通常把深度图预计算并缓存，而不是 on-the-fly 推理。 / Every training image requires a ZoeDepth forward pass, significantly increasing data preprocessing time. Production systems typically precompute and cache depth maps rather than running on-the-fly.
- **相机内参必须已知** / **Camera intrinsics K must be known**: `backproject_patch` 需要相机焦距和主点坐标，从像素 (u, v, depth) 反投影到 3D (x, y, z)。若相机参数未校准或不同相机混用，反投影结果错误，3D 嵌入意义丧失。 / `backproject_patch` needs camera focal length and principal point to backproject from pixel (u, v, depth) to 3D (x, y, z). Uncalibrated cameras or mixed camera types will produce wrong backprojection and meaningless 3D embeddings.

## 延伸阅读 / Further reading

- [SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models](https://spatialvla.github.io/)
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288)
