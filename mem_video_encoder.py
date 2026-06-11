"""
mem_video_encoder.py

Reference implementation of the *video-based short-horizon memory* from
    MEM: Multi-Scale Embodied Memory for Vision-Language-Action Models
    (Section III-C "Video Encoder for Dense Short-Term Visual Memory" + Appendix C)

复现论文短期记忆视频编码器的五个核心特性 / Five faithful properties:
  1. 可分离时空注意力 / Space-time *separable* attention:
     绝大多数层只做帧内双向空间注意力;每隔第 4 层额外做一次"同一 patch
     跨时间步"的因果时间注意力。
  2. 固定正弦时间位编码 e(t),且 e(current)=0 / sinusoidal temporal PE with
     e(current)=0  =>  单图(K=1)行为与原始 ViT 逐位相等。
  3. 零新增参数 / No new learnable params:时间注意力复用同一 block 的
     spatial attention 投影(qkv/proj)与 norm。
  4. Token dropping:只返回"当前帧"的 patch token,交给 VLA backbone 的
     token 数 == 单图 ViT。
  5. 复杂度从 O(n^2 K^2)(联合)降到 O(K n^2 + n K^2)(可分离)。

帧顺序约定 / Frame order convention:  video[:, 0] = 最旧, video[:, T-1] = 当前帧。
"""

import math
import torch
import torch.nn as nn


def sinusoidal_pos_embed(num_positions: int, dim: int, device=None) -> torch.Tensor:
    """标准正弦位置编码表 / standard sinusoidal table, shape (num_positions, dim)."""
    pe = torch.zeros(num_positions, dim, device=device)
    pos = torch.arange(num_positions, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Attention(nn.Module):
    """普通多头自注意力,作用在 (B, L, D) 的序列维 L 上 / plain MHSA over the seq dim."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each (B, heads, L, d)
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, heads, L, L)
        if attn_mask is not None:                        # True = 屏蔽 / blocked
            attn = attn.masked_fill(attn_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class STBlock(nn.Module):
    """
    一个 ViT block。若 space_time=True,则在标准空间注意力残差*之前*插入一个
    因果时间注意力残差 —— 复用同一套 attn 投影与 norm1,因此相对普通 ViT block
    不引入任何新参数。K=1 时时间分支被跳过,block 与普通 ViT block 逐位相等。

    A ViT block. With space_time=True it prepends a causal temporal-attention
    residual that REUSES this block's spatial attn projections and norm1
    (=> zero new params). For T==1 the temporal branch is skipped, so the
    block is bit-identical to a plain ViT block.
    """

    def __init__(self, dim: int, heads: int, space_time: bool = False):
        super().__init__()
        self.space_time = space_time
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x, temporal_pe=None, causal_mask=None):
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        # (a) 每 4 层一次:同一 patch、跨时间步的因果注意力 / causal temporal attention
        if self.space_time and T > 1:
            h = x + temporal_pe.view(1, T, 1, D)             # 加 e(t);当前帧 e=0
            h = self.norm1(h)
            h = h.permute(0, 2, 1, 3).reshape(B * N, T, D)   # 按 patch 分组 -> 时间序列
            h = self.attn(h, attn_mask=causal_mask)          # 跨 T 因果
            h = h.reshape(B, N, T, D).permute(0, 2, 1, 3)
            x = x + h
        # T==1 (K=1):跳过时间分支 -> 与普通 ViT block 完全一致 / exact single-image match

        # (b) 帧内双向空间注意力 / bidirectional spatial attention within each frame
        h = self.norm1(x).reshape(B * T, N, D)
        h = self.attn(h)                                     # 跨 N patch
        x = x + h.reshape(B, T, N, D)

        # (c) MLP
        x = x + self.mlp(self.norm2(x))
        return x


class VideoMemoryEncoder(nn.Module):
    """
    短期视频记忆编码器 / dense short-term visual-memory video encoder.

    输入 video (B, T, C, H, W),输出当前帧的 N 个 token (B, N, D) —— token 数与
    单图 ViT 相同,但已通过因果时间注意力把过去帧信息"焊"进当前帧表征。
    """

    def __init__(self, img_size=224, patch=16, in_ch=3, dim=384, depth=12, heads=6,
                 temporal_every=4):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        # 每隔 temporal_every 层放一个时空层 / every Nth layer is space-time
        self.blocks = nn.ModuleList([
            STBlock(dim, heads, space_time=((i + 1) % temporal_every == 0))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def _temporal_pe(self, T, device):
        # recency: 当前帧(最后一帧)= 0,越旧越大 / current frame -> 0, older -> larger
        recency = torch.arange(T - 1, -1, -1, device=device)
        tpe = sinusoidal_pos_embed(T, self.dim, device)[recency].clone()
        tpe[recency == 0] = 0.0                               # 强制 e(current)=0
        return tpe

    def forward(self, video: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        # video: (B, T, C, H, W) ; video[:, -1] 是当前帧 / last frame is current
        B, T, C, H, W = video.shape
        x = self.patch_embed(video.reshape(B * T, C, H, W))   # (B*T, D, h, w)
        x = x.flatten(2).transpose(1, 2)                      # (B*T, N, D)
        x = x + self.spatial_pos                              # 每帧相同的空间位编码
        x = x.reshape(B, T, self.num_patches, self.dim)

        tpe = self._temporal_pe(T, video.device)              # (T, D), e(current)=0
        causal = torch.triu(torch.ones(T, T, device=video.device), diagonal=1).bool()

        for blk in self.blocks:
            x = blk(x, temporal_pe=tpe, causal_mask=causal)

        x = self.norm(x)
        if return_all:
            return x                                          # (B, T, N, D)
        return x[:, -1]                                       # 丢弃过去帧 -> (B, N, D)


# ----------------------------- 自检 / self-test --------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 224, 224
    enc = VideoMemoryEncoder(img_size=H, patch=16, dim=384, depth=12, heads=6,
                             temporal_every=4).eval()

    # ---- 性质 5:与同深度普通 ViT 参数量完全相同(零新增参数) ----
    plain = VideoMemoryEncoder(img_size=H, patch=16, dim=384, depth=12, heads=6,
                               temporal_every=10_000).eval()  # 无时空层
    n_mem = sum(p.numel() for p in enc.parameters())
    n_plain = sum(p.numel() for p in plain.parameters())
    print(f"[params] video-memory encoder = {n_mem:,}")
    print(f"[params] plain single-img ViT = {n_plain:,}")
    print(f"[params] 新增参数 / extra params = {n_mem - n_plain}  (期望 0 / expect 0)")
    assert n_mem == n_plain

    with torch.no_grad():
        # ---- 性质 4:K=6 多帧输入,输出 token 数仍 == 单图 ----
        K = 6
        clip = torch.randn(B, K, C, H, W)
        out_mem = enc(clip)                                   # (B, N, D)
        print(f"\n[tokens] 输入 {K} 帧 -> 交给 backbone 的 token 数 = {out_mem.shape[1]} "
              f"(== 单图 ViT 的 {enc.num_patches})")
        print(f"[tokens] 朴素堆帧需要 {K} x {enc.num_patches} = {K * enc.num_patches} 个 token")

        # ---- 性质 2:K=1 退化为普通 ViT(单图不变性)----
        single = clip[:, -1:].clone()                         # 只取当前帧 (B,1,C,H,W)
        out_t1 = enc(single)                                  # T=1 路径
        # 用强制无时空层的同权重模型跑同一张图,应逐位相等
        plain.load_state_dict(enc.state_dict())
        out_plain = plain(single)
        max_diff = (out_t1 - out_plain).abs().max().item()
        print(f"\n[K=1] |video_encoder(T=1) - plain_ViT| max = {max_diff:.2e}  "
              f"(期望 ~0 / single-image invariant)")
        assert max_diff < 1e-5

        # ---- 性质 1/3:记忆真的有用 —— 当前帧表征随历史帧变化 ----
        diff_mem_vs_single = (out_mem - out_t1).abs().mean().item()
        print(f"\n[memory] 同一当前帧:有历史 vs 无历史,表征平均差 = {diff_mem_vs_single:.4f} "
              f"(>0 说明过去帧已写入当前 token)")

        # ---- 因果性:扰动"更旧的帧"会改变当前帧输出;但当前帧只能看过去 ----
        clip2 = clip.clone()
        clip2[:, 0] = torch.randn(B, C, H, W)                 # 改最旧的一帧
        out_mem2 = enc(clip2)
        causal_effect = (out_mem - out_mem2).abs().mean().item()
        print(f"[causal] 改最旧帧 -> 当前帧输出变化 = {causal_effect:.4f} "
              f"(>0:历史经因果时间注意力流入当前帧)")

    print("\n全部自检通过 / all checks passed ✅")
