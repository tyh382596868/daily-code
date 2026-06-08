---
date: 2026-06-08
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L43
difficulty: beginner
read_time: ~9 min
tags: [code-of-the-day, vla, vision-encoder, patch-embedding, nanovlm]
build_role: vision-encoder (the input stage of nanoVLA's vision tower — turn camera frames into token sequences)
---

# 37 行的 ViTPatchEmbeddings:一个 Conv2d 就是整个"图像分块" / 37 lines of ViTPatchEmbeddings: one Conv2d *is* the entire "patchify" step

> **一句话 / In one line**: `kernel_size = stride = patch_size` 的 Conv2d 一次性完成"切块"和"每块线性投影"两件事,所有 ViT(进而所有 VLA 的视觉塔)都靠这一招。
> A `Conv2d` with `kernel_size = stride = patch_size` does both "split into patches" and "linearly project each patch" in a single op — every ViT (and therefore every VLA's vision tower) hinges on this trick.

## 为什么重要 / Why this matters

VLA 的视觉塔通常是一个预训练好的 ViT (SigLIP / DINO / CLIP 等);从架构师角度,这个塔的"第一公里"——把 `(B, 3, H, W)` 的图像变成 `(B, num_patches, embed_dim)` 的 token 序列——其实就 37 行代码。理解这 37 行,你就明白了:为什么 ViT 不用 CNN backbone;为什么 patch_size 大就 token 少就快(但分辨率粗);为什么有 CLS token 和无 CLS token 是个独立开关;以及——这是关键——为什么你的 nanoVLA 不需要从头训练这一段,可以直接把 SigLIP 的权重 load 进来。每一个工业级 VLA(OpenVLA、π₀、Isaac GR00T)的视觉接入口都是这副长相。

The vision tower of a VLA is usually a pretrained ViT (SigLIP / DINO / CLIP). From an architect's view, the "first mile" of that tower — turning `(B, 3, H, W)` images into `(B, num_patches, embed_dim)` token sequences — is literally 37 lines of code. Understand these 37 lines and you understand: why ViTs don't need a CNN backbone; why bigger `patch_size` means fewer tokens which means faster (and coarser); why CLS-token is an independent switch from positional embedding; and — most importantly — why your nanoVLA does *not* need to train this stage from scratch and can just `load_state_dict` SigLIP weights into it. Every production VLA (OpenVLA, π₀, Isaac GR00T) has this same input stage.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L43)

```python
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim

        # Conv layer to extract the patches
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))


    def forward(self, x):
        x = self.conv(x)  # extract patches
        x = x.flatten(2)  # flatten the patches into a single dimension
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim)

        # Add CLS token (according to original ViT Paper) and position embeddings
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x
```

## 逐行讲解 / What's happening

1. **`kernel_size=self.patch_size, stride=self.patch_size`(行 21-22) / The Conv2d arguments**:
   - 中文: 这是全篇最关键的一行。一个 stride=patch_size 的 Conv2d 在图像上滑动时,**相邻窗口完全不重叠**——这等价于"把图像切成 patch_size × patch_size 的方块"。而 Conv2d 的内部计算 `out_c = sum(weight * patch)`,正好就是"对每个 patch 做一次线性投影到 embd_dim 维"。两件事一个 op 完成。

     如果换成两步——`F.unfold` 切块 + `nn.Linear` 投影——结果完全一样,但 Conv2d 路径在 cuDNN 里有专门的 kernel,速度快得多。这是 ViT 论文 (Dosovitskiy et al. 2021) 就用的实现技巧。
   - English: The whole point of the module is this line. A Conv2d with `stride = kernel_size = patch_size` slides over the image with **zero overlap** — equivalent to "chop the image into patch_size × patch_size tiles". And the Conv2d math `out_c = sum(weight * patch)` is exactly "linearly project each patch to `embd_dim`." One op does both jobs.

     The equivalent two-step formulation — `F.unfold` to extract patches + `nn.Linear` to project — gives the identical result, but Conv2d hits a specialized cuDNN kernel and is much faster. This is the canonical implementation trick from the original ViT paper.

2. **`num_patches = (img_size // patch_size) ** 2`(行 13) / Line 13**:
   - 中文: 简单的算术,但是 VLA 性能预算的关键变量。224 × 224 输入 + 16 patch_size = 196 token;改成 32 patch_size = 49 token。SigLIP-So400m 在 384 × 384 上跑出 729 token,nanoVLM 用 pixel shuffle (5/29 那期讲过) 砍到 64 token——为的就是让 token 数不爆 LM 上下文。
   - English: Trivial arithmetic, but this is the budget knob of any VLA. 224 × 224 input + 16 patch_size = 196 tokens; bump patch_size to 32 and it drops to 49. SigLIP-So400m at 384 × 384 produces 729 tokens; nanoVLM uses pixel shuffle (covered on 5/29) to compress that to 64 so the token count doesn't blow out the LM context.

3. **`flatten(2)` + `transpose(1, 2)`(行 35-37) / Lines 35-37**:
   - 中文: Conv2d 输出 `(B, embd_dim, H/P, W/P)`,要变成 sequence 格式 `(B, N, embd_dim)`。`flatten(2)` 把 `(H/P, W/P)` 两个空间维合成一个 `N = num_patches`;`transpose(1, 2)` 把 `embd_dim` 移到最后。这两行做了"图像格式 → sequence 格式"的转换,之后就是标准 transformer 序列。
   - English: Conv2d emits `(B, embd_dim, H/P, W/P)`; we need the transformer's sequence layout `(B, N, embd_dim)`. `flatten(2)` collapses the two spatial dims into a single `N = num_patches`; `transpose(1, 2)` moves `embd_dim` to last. Two lines do the "image format → sequence format" rewrite, after which it's a standard transformer input.

4. **`cls_flag` 分支(行 25-28, 39-41) / Lines 25-28, 39-41**:
   - 中文: 经典 ViT (Dosovitskiy 2021) 在 token 序列前面拼一个可学习的 `[CLS]` token,用它的最终特征做分类。SigLIP / DINOv2 / 大多数现代 ViT **不用 CLS**——直接对所有 patch token 做 mean pool 或者全保留。`cls_flag` 是个干净的二选一开关:`True` 拼 CLS 并把 position_embedding 加 1 维容纳它,`False` 就只有 patch tokens。VLA 视觉塔几乎都用 `False`——因为我们要把所有 patch 都喂给下游 projector。
   - English: The classic ViT (Dosovitskiy 2021) prepends a learned `[CLS]` token whose final hidden state does the classification. SigLIP / DINOv2 / most modern ViTs **drop CLS** — they either mean-pool the patch tokens or keep them all. `cls_flag` is a clean either-or switch: `True` prepends a CLS and widens position_embedding by one slot; `False` keeps only patch tokens. VLA vision towers almost universally use `False` — we want every patch fed to the downstream projector.

5. **`position_embedding` 是**学习出来的**而不是公式 RoPE(行 27, 29) / Lines 27, 29**:
   - 中文: 注意这里 position embedding 是一个直接 `nn.Parameter`——可学习的 lookup table。ViT 一直用这个,而不是 LM 现在流行的 RoPE。原因:patches 是 2D 网格、不是 1D 序列;RoPE 的 1D 性不太对得上 2D 空间。SigLIP / DINOv2 用 1D 可学习 position embedding;DINOv3 / Wan 等用了 2D RoPE(见 5/29 的 Wan 3D RoPE 笔记)。本文件这个简单版本就当 baseline 读。
   - English: Note that position_embedding here is a plain `nn.Parameter` — a learnable lookup table. ViTs historically use this rather than the RoPE that modern LMs prefer. Reason: patches form a 2D grid, not a 1D sequence; RoPE's 1D nature doesn't quite align with 2D space. SigLIP and DINOv2 use 1D learnable embeddings; DINOv3 and Wan introduce 2D / 3D RoPE variants (see the 5/29 Wan 3D RoPE note). Treat the lookup-table version here as the baseline.

## 类比 / The analogy

中文: 想象一个老式胶片冲印店。师傅拿到一张 24×36 的负片,他不会一眼看完整张——他会用一个放大镜,从左上角开始,**不重叠地一格一格扫**:第一个 4mm × 4mm 框是 patch 1,挪到隔壁是 patch 2……每扫一格,他记下这一格"主要颜色 + 颗粒度"等几个数字(投影到 embd_dim 维)。扫完所有格,他得到了一张"摘要笔记本",每页一格。Conv2d (stride=kernel) 就是这位师傅的放大镜——一台机器同时完成了"挪窗"和"做笔记"。Position embedding 就是给笔记本每一页加上"我是第几行第几列那格"的页码。

English: Picture an old-school film-developing shop. The technician takes a 24×36 mm negative and won't try to read the whole frame — they use a loupe, walking it across the negative **without overlap**, one cell at a time: the first 4×4 mm cell is patch 1, slide it over for patch 2, and so on. For each cell they jot down a few numbers — "dominant color + grain + contrast" — projecting the cell to `embd_dim` features. After all cells they have a notebook, one cell per page. A Conv2d with `stride=kernel` *is* that loupe — one machine doing both the "shift" and the "annotate" in one motion. The position embedding is the page number written on each notebook page so you remember which row/column it came from.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

> **Curriculum slot: `vision-encoder`**. Depends on: none (this is a leaf component — the very first stage of any vision tower).
> Already covered in this curriculum: `action-tokenizer` (5/10), `modality-projector` (5/29), `vlm-backbone-wiring` (5/29), `training-step` (5/28). This is the 5th covered component.

中文: 这是 nanoVLA 流水线的**第一个**模块。每张相机图像(腕部 + 基座相机各一张)都先经过 `ViTPatchEmbeddings`,得到一串 patch tokens;然后这串 tokens 进入 ViT 的若干 transformer block(file 后半段的 `ViTBlock`),输出最终视觉特征 `(B, N, embd_dim)`;接着送进 **modality-projector**(已覆盖,5/29 笔记)做下采样 + 维度对齐到 LM 空间;再喂给 **VLM backbone**(已覆盖,5/29 SmolVLA 笔记)和最终 **action head**。

如果省掉这个模块?那你只能用 CNN backbone(ResNet 等)出特征,然后 reshape——理论上可行,实际丢掉了 ViT 体系的所有预训练资源(SigLIP/DINO checkpoint 都用不上)。

生产级实现要加什么?
- **多相机融合**: 真实 VLA 有 2-4 个相机,通常每个相机独立 patchify,然后 token 序列拼接(或在 modality-projector 处合并)。Isaac GR00T 的 `embed_image` 函数就这么干。
- **可变分辨率**: SigLIP-So400m 支持 NaFlex(任意宽高比);你的 position_embedding 得改成可插值的(`F.interpolate` 一下)或用 2D RoPE。
- **预训练权重加载**: 不要从头训。文件下面的 `from_pretrained` 方法演示了"从 SigLIP safetensors → 自己的 state_dict"的 key remapping——关键是把 SigLIP 的 `q_proj/k_proj/v_proj` 拼成你这里合并的 `qkv_proj`。

English: This is the **first** module in the nanoVLA pipeline. Each camera frame (wrist + base) first goes through `ViTPatchEmbeddings` to produce patch tokens; those tokens then flow through the file's `ViTBlock`s, outputting visual features `(B, N, embd_dim)`; next a **modality-projector** (covered 5/29) downsamples + projects to LM space; the projected tokens feed the **VLM backbone** (covered 5/29 SmolVLA note) and ultimately the **action head**.

What if you omit it? You'd be stuck with CNN backbones (ResNet etc.) emitting features + a reshape — feasible in theory, but you give up every pretrained ViT checkpoint (SigLIP, DINOv2, DINOv3).

What does production need on top?
- **Multi-camera fusion**: real VLAs run 2-4 cameras; each camera typically gets patchified independently and the token sequences are concatenated (or merged in the modality-projector). Isaac GR00T's `embed_image` does exactly this.
- **Variable resolution**: SigLIP-So400m supports NaFlex (arbitrary aspect ratios); for that, your position_embedding must become interpolatable (`F.interpolate`) or you switch to 2D RoPE.
- **Pretrained weight loading**: do NOT train from scratch. The `from_pretrained` method below in the same file shows the key-remapping needed to convert SigLIP safetensors → your state_dict — the crucial step is concatenating SigLIP's separate `q_proj/k_proj/v_proj` into your combined `qkv_proj`.

## 自己跑一遍 / Try it yourself

```python
# patch_embed_demo.py
import torch
import torch.nn as nn

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embd_dim=768, use_cls=False):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.use_cls = use_cls
        self.conv = nn.Conv2d(3, embd_dim, kernel_size=patch_size,
                              stride=patch_size, padding="valid")
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, embd_dim))
            self.pos = nn.Parameter(torch.randn(1, self.num_patches + 1, embd_dim))
        else:
            self.pos = nn.Parameter(torch.randn(1, self.num_patches, embd_dim))

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)
        if self.use_cls:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.pos

pe = ViTPatchEmbeddings(img_size=224, patch_size=16, embd_dim=768, use_cls=False)
img = torch.randn(2, 3, 224, 224)  # batch of 2 RGB images
tokens = pe(img)
print(f"image {img.shape} -> tokens {tokens.shape}")
print(f"num_patches = ({224}/{16})**2 = {(224//16)**2}")

# Sanity check the equivalence: unfold + linear should match Conv2d
import torch.nn.functional as F
patches = F.unfold(img, kernel_size=16, stride=16)         # (B, 3*16*16, N)
W = pe.conv.weight.view(768, -1)                            # (768, 3*16*16)
manual = (W @ patches).transpose(1, 2) + pe.conv.bias       # (B, N, 768)
print("conv vs unfold+linear max err:", (manual - pe.conv(img).flatten(2).transpose(1,2)).abs().max().item())
```

运行 / Run with:
```bash
pip install torch
python patch_embed_demo.py
```

预期输出 / Expected output:
```
image torch.Size([2, 3, 224, 224]) -> tokens torch.Size([2, 196, 768])
num_patches = (224/16)**2 = 196
conv vs unfold+linear max err: < 1e-5
```

中文一两句: 注意最后一行——`F.unfold + matmul` 出来的结果跟 `Conv2d` 数值上完全一致(误差 < 1e-5)。这就是"一个 Conv2d 同时做切块和投影"的代数证明。

English: Pay attention to the last line — `F.unfold + matmul` matches `Conv2d` numerically to within 1e-5. That's the algebraic proof that "one Conv2d does both the patchify and the projection."

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot SmolVLA 的 `embed_image`** / **lerobot SmolVLA's `embed_image`**: 同样的 patch embed,但额外做了多相机 token 拼接。/ Same patch embed, plus multi-camera token concatenation.
- **Isaac GR00T 的 `gr00t/model/backbone/vit_patch.py`** / **Isaac GR00T `vit_patch.py`**: 支持可变分辨率(NaFlex 风格),用 sincos position embedding 而不是 learnable。/ Adds variable-resolution support (NaFlex-style), uses sincos position embeddings instead of learnable ones.
- **OpenVLA 用 SigLIP 整段当 backbone** / **OpenVLA uses SigLIP wholesale as backbone**: 它没自己写 PatchEmbeddings,直接 from `transformers` 加载 SigLIP——但 SigLIP 的内部实现就是这个文件 line 7 注释链接到的那一段。/ It doesn't write its own PatchEmbeddings; it loads SigLIP via `transformers` — and SigLIP's internals are exactly what the line-7 comment in this file links to.
- **OpenPI π₀ 用 PaliGemma backbone** / **OpenPI π₀ uses PaliGemma backbone**: PaliGemma 内部也是 SigLIP-So400m + patch embed。同一颗心。/ PaliGemma internally is also SigLIP-So400m + the same patch embed. Same heart.

## 注意事项 / Caveats / when it breaks

- **`padding="valid"` 意味着 img_size 必须是 patch_size 的整数倍 / `padding="valid"` requires img_size divisible by patch_size**: 224/16 = 14 ✓。如果你接的相机给 240 × 240,要么 resize 到 224 要么改 padding。/ 224/16 = 14 ✓. If your camera streams 240 × 240, either resize to 224 or change the padding scheme.
- **`position_embedding` 是固定 num_patches 的 / `position_embedding` is fixed-size**: 改输入分辨率就要插值。本文件的 `from_pretrained` 没处理这个——生产里你得加 `F.interpolate(pos, size=(new_h, new_w), mode='bilinear')`。/ Change input resolution and you must interpolate. The file's `from_pretrained` doesn't handle this — production code does `F.interpolate(pos, size=(new_h, new_w), mode='bilinear')`.
- **`cls_token = torch.zeros(...)` 初始为 0 / CLS init is zero**: 故意的——让训练自己学。BUT 如果你用 `nn.init.trunc_normal_(...)` 加点小噪声,有时收敛更快。Original ViT 论文用的就是 zeros。/ Intentional — let training learn it. But initializing with `nn.init.trunc_normal_` sometimes converges faster. The original ViT paper just used zeros.
- **3 channels 写死 / `in_channels=3` is hard-coded**: 如果你想用 depth 通道或 4-camera concatenated input,要在 patchify 前自己处理。/ If you want to fuse depth or 4-camera concatenated input, you have to handle it before patchify.

## 延伸阅读 / Further reading

- [ViT paper — Dosovitskiy et al. 2021, "An Image is Worth 16×16 Words"](https://arxiv.org/abs/2010.11929)
- [SigLIP paper — Zhai et al. 2023](https://arxiv.org/abs/2303.15343)
- [nanoVLM repo](https://github.com/huggingface/nanoVLM)
- [transformers SigLIP modeling reference (the line 7 comment link)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)
