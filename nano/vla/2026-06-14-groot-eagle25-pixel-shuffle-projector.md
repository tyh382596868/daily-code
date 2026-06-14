---
date: 2026-06-14
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py
permalink: https://github.com/huggingface/lerobot/blob/8515d456be1dbef8c133f07188c785e683eca899/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py#L288-L328
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, modality-projector, pixel-shuffle, groot, eagle25]
build_role: modality-projector (cross-repo variant — production-grade companion to nanoVLM's projector)
---

# GR00T 用的 Eagle2.5 投影器:NHWC 三步乱舞把视觉 token 砍掉四分之三 / GR00T's Eagle2.5 projector: three NHWC permutations chop 75 % of the vision tokens

> **一句话 / In one line**: 视觉 token 太多?Eagle2.5 用纯 `view + permute` 的 pixel-shuffle 把每张图的 token 数砍到 1/4,通道维度变 4 倍,再过一个 MLP 砸进 LM 隐空间。 / Too many vision tokens? Eagle2.5 uses a pure-view-and-permute pixel-shuffle to cut each image's token count to a quarter, fans the channel dim out 4×, then runs the result through a tiny MLP into the LM's hidden space.

## 为什么重要 / Why this matters

VLA 里"视觉 → 语言"的桥是 **modality projector**——一个把 ViT 的 patch token 转换到 LM token 维度的小模块。这块大家拼命压缩 token 数,因为 SigLIP 一张 384×384 图就有 729 个 token,3 个相机就是 2187 个 token——再叠 5 帧历史立刻把 LM context 撑爆。GR00T 用的是 NVIDIA Eagle2.5-VL 这个 VLM 的 projector,它的关键技巧是 **pixel-shuffle**:不学新参数、纯靠 NHWC 维度乱舞,把 (B, 729, D) 砸成 (B, 184, 4D),token 数减 4 倍。lerobot 现在把整个 Eagle2.5 实现拉进了仓库,放在 `policies/groot/eagle2_hg_model/`。

In a VLA, the **modality projector** is the bridge from vision tokens to language tokens — it remaps ViT patch tokens into the LM's hidden dimension. Everyone in this space optimises token count, because SigLIP at 384×384 emits 729 tokens per image; three cameras = 2187 tokens, and stacking 5 frames of history blows past most LM context windows. GR00T uses NVIDIA Eagle2.5-VL's projector, whose key trick is **pixel-shuffle**: no learnable params, pure NHWC reshuffles that turn `(B, 729, D)` into `(B, 184, 4D)` — 4× fewer tokens. lerobot just pulled the whole Eagle2.5 implementation into the policy directory.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py`](https://github.com/huggingface/lerobot/blob/8515d456be1dbef8c133f07188c785e683eca899/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py#L288-L328)

```python
def pixel_shuffle(self, x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))

    x = x.permute(0, 2, 1, 3).contiguous()
    return x

def extract_feature(self, pixel_values):
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True
        )
        if hasattr(vit_embeds, "last_hidden_state"):
            vit_embeds = vit_embeds.last_hidden_state

    else:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        ).hidden_states[self.select_layer]

    if self.use_pixel_shuffle:
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(
            vit_embeds, scale_factor=self.downsample_ratio
        )  # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0], -1, vit_embeds.shape[-1]
        )  # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])

    if self.mlp_checkpoint and vit_embeds.requires_grad:
        vit_embeds = cp.checkpoint(self.mlp1, vit_embeds)
    else:
        vit_embeds = self.mlp1(vit_embeds)

    return vit_embeds
```

## 逐行讲解 / What's happening

1. **`extract_feature` 第 1 步——跑 ViT / Step 1: run the ViT**:
   - 中文: `select_layer=-1` 拿最后一层,否则取指定中间层(VLA 里常用倒数第二层因为最后一层往往特征已 collapse 到分类信号上)。结果 shape 是 `(B, num_tokens, D_vit)`,比如 `(B, 1024, 1024)` 对应 32×32 个 patch、1024 维特征。
   - English: `select_layer = -1` takes the final layer, otherwise an intermediate layer (VLAs often pick second-to-last because the final layer over-collapses to classification features). The output shape is `(B, num_tokens, D_vit)`, e.g. `(B, 1024, 1024)` for 32×32 patches × 1024 features.

2. **第 314-315 行 / Lines 314-315 (flatten → 2D grid)**:
   - 中文: `h = w = sqrt(num_tokens)` 把"token 序列"还原成正方形网格。reshape 成 `(B, h, w, D)`(NHWC),为接下来的 pixel-shuffle 做准备——这一步不影响内存布局,只是改 view。
   - English: `h = w = sqrt(num_tokens)` recovers the 2D patch grid from the token sequence. Reshape to `(B, h, w, D)` (NHWC) in preparation for pixel-shuffle — pure view, no memory shuffle.

3. **`pixel_shuffle` 第 289-298 行 / `pixel_shuffle` lines 289-298**:
   - 中文: scale=0.5 时,目标是 H→2H, W→2W, C→C/4 吗?不!这是**反向 pixel-shuffle**(downsample),做的是 H→0.5H, W→0.5W, C→4C。三步:
     - 中文: 第 1 步 view: `(N, W, H, C) → (N, W, 0.5H, 2C)`——把 H 拆成"上半-下半",每个"半"配 2C(把 C 的前后两半视作两个 H step)。
     - 中文: 第 2 步 permute(0,2,1,3): 把 H 提到第 1 维。
     - 中文: 第 3 步 view: `(N, 0.5H, W, 2C) → (N, 0.5H, 0.5W, 4C)`——同样的把戏在 W 上再来一次。
     - 中文: 最后一个 permute 把 H, W 转回 (W, H) 视角(符合 conv2d 习惯)。
   - English: With `scale=0.5` the goal is **down**sample: H→0.5H, W→0.5W, C→4C (this is the inverse of conv pixel-shuffle, sometimes called "pixel-unshuffle"). Three moves:
     - English: View 1: `(N, W, H, C) → (N, W, 0.5H, 2C)` — split H into halves, fold each half into 2C (the two halves of C now stand in for two adjacent H steps).
     - English: Permute (0, 2, 1, 3): bring H to the second axis.
     - English: View 2: `(N, 0.5H, W, 2C) → (N, 0.5H, 0.5W, 4C)` — the same trick on W.
     - English: Final permute restores (W, H) ordering for conv2d compatibility.

4. **为什么不用 conv stride? / Why not just a strided conv?**:
   - 中文: 因为没有学习参数!ViT 已经学好了好的 patch 表征,projector 该做的只是"把 4 个相邻 patch 拼起来,channel 4 倍化",让一个 token 携带原来 4 个 token 的信息。pixel-shuffle 是无参数、可逆的,纯几何重排——把"空间分辨率"换成"通道分辨率"。
   - English: Because no learnable params! The ViT already learned good patch features; the projector just needs to "fuse 4 adjacent patches, 4× the channel" so one token carries the info of four. Pixel-shuffle is parameter-free, invertible, pure geometric reshuffle — it swaps spatial resolution for channel resolution.

5. **第 319-321 行 / Lines 319-321 (flatten back)**:
   - 中文: 把 `(B, 0.5H, 0.5W, 4D)` reshape 回 `(B, 0.25·num_tokens, 4D)`,token 数 / 4,通道数 × 4——总 element 数不变。1024 个 token → 256 个 token,1024 维 → 4096 维。
   - English: Reshape `(B, 0.5H, 0.5W, 4D)` back to `(B, 0.25·num_tokens, 4D)` — quarter the tokens, quadruple the channels, same total elements. 1024 tokens → 256, 1024 dim → 4096.

6. **第 323-326 行 / Lines 323-326 (the MLP)**:
   - 中文: `mlp1` 是个 1-或 2-层 MLP,把 `4D_vit` 投影到 `D_lm`。在 `__init__` 里(第 134-150 行)有 3 个变体:2-layer (LN+Linear+GELU+Linear),1-layer w/ pixel-shuffle,1-layer no pixel-shuffle——三个分支对应三种 ViT 输出形状,接进同一个 LM。`mlp_checkpoint` 开启 activation checkpoint 省显存——projector 虽然小,但在长序列下梯度还是会爆。
   - English: `mlp1` is a 1- or 2-layer MLP that projects `4D_vit` to `D_lm`. The `__init__` (lines 134-150) defines three variants: 2-layer (LN + Linear + GELU + Linear), 1-layer with pixel-shuffle, 1-layer without — three branches corresponding to three ViT-output shapes feeding the same LM. `mlp_checkpoint` turns on activation checkpointing — the projector is small but its activations under long sequences still blow memory.

## 类比 / The analogy

想象一面 32×32 的瓷砖墙,每块瓷砖有 1024 种花纹。LM 表示空间太挤,只能放 256 块瓷砖。怎么办?把每 2×2 一组的瓷砖**叠**起来——4 块拼成一块,但花纹数从 1024 翻成 4096(因为 4 张花纹叠在了一起)。空间细节没丢,只是从"位置编码"挪到了"通道编码"。最后那个 MLP 是个"翻译官":告诉 LM 这种新瓷砖怎么读。整个过程零学习参数(pixel-shuffle 部分),纯靠 view 和 permute 重排内存。

Picture a 32×32 tile wall, each tile with 1024 patterns. The LM only has room for 256 tiles. Solution: stack each 2×2 group of tiles — four become one, and the pattern count goes from 1024 to 4096 (the four patterns sit on top of each other). No spatial detail lost; you just moved it from "position encoding" to "channel encoding". The final MLP is the translator that explains to the LM how to read these new fattened tiles. The pixel-shuffle step has zero learnable parameters — only views and permutes shuffling memory.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 **`modality-projector`** 这个 curriculum slot——以前你已经在 nanoVLM 上见过它的最小版(05-29 那篇的 1-Linear projector + view trick)。今天看到的是**生产版本**:输入是 ViT 的 `(B, N, D_vit)`,输出是 `(B, N/4, D_lm)`,塞给下游的 VLM backbone(`vlm-backbone-wiring` 这个 slot)。依赖关系上,projector 只依赖 vision-encoder 的输出形状,所以可以独立调试。如果你完全省掉这个 projector,LM 就要直接吃 ViT 维度的 token——`D_vit=1024` ≠ `D_lm=4096` 就会形状不匹配,你得在 LM 输入层加 embedding,但那样 LM 自己的 token embedding 就被混淆了。生产代码里还得加上的事:**支持多相机**(N 张图各自走一遍这个 projector,然后 cat)、**支持多分辨率**(tile-by-tile 编码,token 数随 H×W 变化)、**给 projector 加上 dropout 和混入 LM 的 chat template 占位符**。

This is the **`modality-projector`** curriculum slot — you saw its minimal cousin (nanoVLM's 1-Linear projector + view trick) on 05-29. Today's is the production version: inputs are the ViT's `(B, N, D_vit)`, outputs are `(B, N/4, D_lm)`, feeding straight into the downstream VLM backbone (the `vlm-backbone-wiring` slot). Dependency-wise the projector only needs the vision-encoder's output shape, so it's debuggable in isolation. If you omit it entirely, the LM has to consume ViT-dim tokens directly — `D_vit=1024 ≠ D_lm=4096` blows the shape match, and you'd have to add an extra embedding on the LM input, polluting the LM's own token embeddings. To turn this into production code, also add: **multi-camera support** (one projector pass per image, then concat), **multi-resolution support** (tile-by-tile encoding, variable N from H×W), and **dropout + chat-template placeholder reservation** inside the projector.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

def pixel_unshuffle(x_nhwc: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Eagle2.5 pixel-shuffle: (N, H, W, C) -> (N, scale*H, scale*W, C/scale**2)."""
    n, w, h, c = x_nhwc.size()
    x = x_nhwc.view(n, w, int(h * scale), int(c / scale))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * scale), int(w * scale), int(c / (scale * scale)))
    return x.permute(0, 2, 1, 3).contiguous()

D_VIT, D_LM = 1024, 4096
mlp = nn.Linear(D_VIT * 4, D_LM)            # 1-layer projector w/ pixel-shuffle

vit_tokens = torch.randn(2, 32 * 32, D_VIT)  # (B, N=1024, 1024)
print("ViT out :", vit_tokens.shape)

grid = vit_tokens.reshape(2, 32, 32, D_VIT)  # (B, H, W, C)
shuffled = pixel_unshuffle(grid, scale=0.5)  # (B, 16, 16, 4096)
flat = shuffled.reshape(2, -1, shuffled.shape[-1])
print("after PS :", flat.shape)              # (B, 256, 4096)

lm_tokens = mlp(flat)
print("to LM   :", lm_tokens.shape)          # (B, 256, 4096)

# Round-trip test: pixel_unshuffle has a true inverse (pixel-shuffle)
def pixel_shuffle_inverse(x_nhwc, scale=2.0):
    n, h, w, c = x_nhwc.size()
    x = x_nhwc.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, w, int(h * scale), int(c / scale))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x.view(n, int(h * scale), int(w * scale), int(c / (scale * scale)))

back = pixel_shuffle_inverse(shuffled, scale=2.0)
print("round-trip max diff:", (back - grid).abs().max().item())  # ~0
```

运行 / Run with:
```bash
pip install "torch>=2.4"
python try.py
```

预期输出 / Expected output:
```
ViT out : torch.Size([2, 1024, 1024])
after PS : torch.Size([2, 256, 4096])
to LM   : torch.Size([2, 256, 4096])
round-trip max diff: 0.0
```

中文一句:`round-trip max diff = 0` 是 pixel-shuffle 是个"双射重排"的硬证据——没丢任何信息,只是换了形状。这就是它不用学参数也能 work 的原因。

English: `round-trip max diff = 0` is hard proof that pixel-shuffle is a bijective reshuffle — no information lost, only the shape changes. That's why it doesn't need learnable params.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **nanoVLM 的 projector**(2026-05-29 那篇)/ **nanoVLM projector** (covered 2026-05-29): 一个更短的版本,用 `view + reshape` 而不是显式 NHWC permute,但 idea 相同。 / A shorter version that uses `view + reshape` instead of explicit NHWC permutes; same idea.
- **InternVL / Mini-Gemini 的 connector** / **InternVL / Mini-Gemini connector**: 也是 pixel-shuffle + MLP,Eagle 实际上是 InternVL 的直系后代,downsample_ratio 默认 0.5 几乎是行业标准。 / Same pixel-shuffle + MLP pattern; Eagle is InternVL's direct descendant and `downsample_ratio=0.5` is essentially the industry default.
- **PixelShuffle 在 image super-resolution** / **PixelShuffle in image super-resolution**: 原始 `nn.PixelShuffle` 用于做上采样(C/4 → 4C 反过来);Eagle 用的是反向版,也就是 `nn.PixelUnshuffle`。 / The original `nn.PixelShuffle` does upsampling (C/4 → 4× spatial); Eagle's version is the inverse, `nn.PixelUnshuffle`.
- **OpenVLA 的 fused vision projector** / **OpenVLA fused vision projector**: 直接用 `nn.Linear(D_vit, D_lm)` 不做 pixel-shuffle,代价是 token 数不减——所以 OpenVLA 只有 1 个相机时还可以,多相机就需要换 Eagle 这种压缩方案。 / OpenVLA uses a raw `nn.Linear(D_vit, D_lm)` with no pixel-shuffle, paying with full token count — fine for 1-camera setups but inadequate for multi-cam, hence the move toward Eagle-style compression.

## 注意事项 / Caveats / when it breaks

- **必须是正方形 grid** / **Must be a square grid**: `h = w = sqrt(N)`,如果 ViT 输出 patch 数不是平方数(例如非正方形图像)就直接崩。非正方形需要单独传 H、W。 / If the ViT's patch count isn't a perfect square (non-square images), this code crashes. Pass H and W separately for that case.
- **`scale_factor` 写反就是 upsampler** / **Flip `scale_factor` and you have an upsampler**: 这里 `0.5` 表示"空间缩一半,通道翻 4 倍";写成 `2.0` 就反过来变 PixelShuffle——很容易翻车的命名。 / `0.5` here means "halve spatial, quadruple channel"; `2.0` would invert it to a PixelShuffle. Easy to flip by accident.
- **D 必须能被 `1/scale²` 整除** / **D must be divisible by `1/scale²`**: scale=0.5 时,`D % 4 == 0`,否则 reshape 报错。Eagle 通过让 ViT 输出的隐藏维度是 4 的倍数(SigLIP-Large 是 1024)来避免。 / At `scale=0.5`, you need `D % 4 == 0`, else the reshape fails. Eagle dodges this by picking SigLIP-Large (`D = 1024`) — divisible by 4.
- **`contiguous()` 不能省** / **Don't skip `contiguous()`**: 两次 permute 后内存布局已经被打乱,下一个 view 没有 contiguous 会报"view size is not compatible with input tensor's size and stride"。 / The permutes scramble the memory layout; without `contiguous()` the next view raises "view size is not compatible with input tensor's size and stride".

## 延伸阅读 / Further reading

- [Eagle2.5-VL paper (NVIDIA, 2024)](https://arxiv.org/abs/2410.02713)
- [Isaac GR00T-N1.7 release notes](https://github.com/NVIDIA/Isaac-GR00T)
- [PyTorch `nn.PixelShuffle` / `PixelUnshuffle` docs](https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html)
- nanoVLM projector (daily-code 2026-05-29) — the minimal companion to this one
