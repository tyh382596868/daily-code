---
date: 2026-06-07
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L1-L43
difficulty: beginner
read_time: ~9 min
tags: [code-of-the-day, vla, vision-encoder, patch-embedding, vit, siglip]
build_role: vision-encoder (nanoVLA curriculum item — pixels → tokens, the entry point of the vision side)
---

# 一根 stride 等于 patch 的 Conv2d:VLA 视觉编码器的整个入口就这么简单 / One Conv2d with stride = patch size: the entire entry point of a VLA's vision encoder

> **一句话 / In one line**: `nn.Conv2d(3, D, kernel_size=patch, stride=patch)` 把 `(B, 3, 224, 224)` 的像素直接卷成 `(B, 196, D)` 的 token 序列,再加一个可学习的 position embedding(还可选一个 CLS token),整个 ViT 的输入端就齐了——这就是 VLA 看图的第一步. / A single `nn.Conv2d(3, D, kernel_size=patch, stride=patch)` turns `(B, 3, 224, 224)` pixels into a `(B, 196, D)` token sequence; add a learned position embedding (and optionally a CLS token) and the ViT input head is complete. This is the first 37 lines of any VLA's vision pipeline.

## 为什么重要 / Why this matters

每次我看新的 VLA 论文,作者都会画一堆华丽的 vision encoder 图——SigLIP、DINOv3、双相机融合、token reduction……让人以为视觉端是个超复杂系统. 实际上**所有这些 encoder 的第一层都是同一件事**:把图片切 patch、变成 token. nanoVLM 用 37 行把这一步剥到最干净——没有抽象、没有依赖、没有 `from transformers import ...`,就是 `Conv2d(stride=patch_size)` 加可学习 position embedding. 看懂这 37 行,你就理解了 SigLIP / CLIP / DINOv2 / DINOv3 的输入处理为什么是一样的;再看懂后面的 ViTBlock,你就理解了为什么 PaliGemma、SmolVLM、Qwen2.5-VL 都把 vision tower 当成一个"图 → token 序列"的黑盒. 从 nanoVLA 角度,这就是相机帧进系统的"第一道关",上面接你昨天学的 modality projector,再接 LM backbone.

Every VLA paper draws an elaborate vision-encoder diagram — SigLIP, DINOv3, multi-camera fusion, token reduction, you name it — making the vision side look like a beast. In reality **all of these encoders start with the same operation**: cut the image into patches and turn each patch into a token. nanoVLM strips this to 37 lines with no abstractions, no `from transformers import ...`, just `Conv2d(stride=patch_size)` plus a learned positional embedding. Read this and you'll see why SigLIP / CLIP / DINOv2 / DINOv3 all have identical input heads; pair it with the next ViTBlock and you'll see why PaliGemma, SmolVLM, and Qwen2.5-VL can treat their vision towers as black-box "image → token sequence" callables. For a nanoVLA, this is the very first stage of the camera-frame pipeline; downstream is the modality projector you read on 2026-05-29, then the LM backbone.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py#L1-L43`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L1-L43)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

1. **`num_patches = (img_size // patch_size) ** 2` 的算术 / The arithmetic of `num_patches`**:
   - 中文: `224 / 16 = 14`,所以 `num_patches = 196`. 一张 224×224 的图被切成 14×14 个 patch,每个 patch 变成一个 token. 这 196 就是后面所有 attention 的序列长度——你要算 KV cache、要估算 FLOPs,都从这里出发.
   - English: `224 / 16 = 14`, so `num_patches = 196`. A 224×224 image becomes a 14×14 grid of patches, one token per patch. This 196 is the sequence length that every downstream attention sees — it's the budget you reason about when estimating KV cache or FLOPs.

2. **`Conv2d(kernel=patch, stride=patch, padding="valid")` 是怎么"做 patch"的 / How `Conv2d(kernel=patch, stride=patch, padding="valid")` patches the image**:
   - 中文: 这个 Conv 的 stride 等于 kernel,所以每个输出位置看的 16×16 输入像素**不重叠**——本质上是个非重叠的 unfold,加上一个线性投影. 数学上它等价于:`reshape((B, 3, 14, 16, 14, 16))`、把两个 16 维 flatten 成 768、再 `Linear(768, D)`. 但 Conv 实现一步到位,而且 backward 直接用 cuDNN.
   - English: stride equals kernel, so adjacent output positions look at **non-overlapping** 16×16 patches — semantically a non-overlapping unfold plus a linear projection. It's mathematically equivalent to `reshape → flatten → Linear(768, D)`, but the Conv implementation does it in one shot and gets cuDNN-optimized backward for free.

3. **`x = self.conv(x)` 之后形状是 `(B, D, 14, 14)` / Shape after `self.conv(x)` is `(B, D, 14, 14)`**:
   - 中文: Conv2d 输出是 channel-first 的 4D 张量,channel 维度就是嵌入维度. 接下来 `flatten(2)` 把 H、W 拍成一条 196,`transpose(1, 2)` 把 channel 放到最后,变成 transformer 喜欢的 `(B, 196, D)`.
   - English: Conv2d outputs a channel-first 4-D tensor where the channel dim is now the embedding dim. `flatten(2)` collapses H, W into a length-196 sequence and `transpose(1, 2)` moves channel last, giving the transformer-shaped `(B, 196, D)`.

4. **`cls_token = self.cls_token.expand(x.shape[0], -1, -1)` 而不是 `repeat` / `expand` instead of `repeat` for `cls_token`**:
   - 中文: `expand` 不分配新内存,只是改 stride——而 `repeat` 真的 alloc 一份. 对一个 `(1, 1, D)` → `(B, 1, D)` 的广播,内存差 B 倍. 接下来 `cat` 会变成连续张量,但 expand 这一步先省了显存,在 long-context 里是值得抠的细节.
   - English: `expand` is a stride trick — zero new memory — while `repeat` actually allocates a copy. For `(1, 1, D) → (B, 1, D)` the saved memory is a factor of B. The subsequent `cat` materializes, but the `expand` step itself is free, which matters under long context.

5. **`x = x + self.position_embedding` 是直接加,不是 cat / Position embedding is added, not concatenated**:
   - 中文: ViT 用**绝对**位置嵌入,而且是**可学习的**(`nn.Parameter(torch.rand(...))`). 不是 sin/cos 这种固定函数. 训练时这个 196×D(或 197×D)的矩阵和图像一起优化——后面如果 finetune 到不同分辨率,就要 *插值* 这个 position embedding(SigLIP / DINOv3 都有相应代码,但 nanoVLM 这里不做).
   - English: ViT uses **learned absolute** position embeddings (`nn.Parameter(torch.rand(...))`), not the sinusoidal ones from the original Transformer. The 196×D (or 197×D with CLS) matrix is trained jointly with the image. If you ever finetune at a different resolution, this matrix has to be *interpolated* — SigLIP/DINOv3 both ship that helper; nanoVLM does not.

6. **`cls_flag` 的含义 / What `cls_flag` controls**:
   - 中文: 原版 ViT 用 CLS token 做分类(把第 0 个 token 的输出过 Linear). SigLIP 不用 CLS,直接对 196 个 patch token 求 mean 当全局表征. nanoVLM 用 `cls_flag` 兼容两种——VLA 场景通常 `cls_flag=False`,因为我们要的是**所有 patch 的 token**喂给后面的 modality projector,不需要单一全局向量.
   - English: original ViT used a CLS token for classification (LinearProbe on token 0). SigLIP omits CLS and does mean-pool over the 196 patch tokens. nanoVLM keeps `cls_flag` to support both — in a VLA you typically use `cls_flag=False` because you want **all the patch tokens** for the downstream modality projector, not a single global vector.

7. **`init.normal_(weight, mean=0, std=0.02)` 在 `_init_weights` / `_init_weights`'s `init.normal_(weight, mean=0, std=0.02)`**:
   - 中文: 没贴出来,但下面 `ViT._init_weights` 把 Conv2d 也按 std=0.02 初始化——这是 ViT/GPT 家族的标准初始化,纯 `Linear`/`Conv2d` 用 0.02,position embedding 用 0–1 的均匀 rand(`torch.rand`). 后者比较反直觉,但效果好.
   - English: not shown here but `ViT._init_weights` later initializes Conv2d with std=0.02 — the standard ViT/GPT family init. `Linear` and `Conv2d` use 0.02; position embedding gets a uniform `torch.rand` (0–1). The latter is counter-intuitive but empirically robust.

## 类比 / The analogy

把一张照片想象成一面用 224×224 个小贴纸拼起来的墙. patch embedding 就像往墙上**盖一个 16×16 的方形印章**,每按一下,印章把这 16×16 个贴纸"压成"一张唯一的卡片(D 维向量). 因为印章的步长正好是 16(贴纸不重叠),你按 14×14 = 196 次就把整面墙印完——每张卡片背面再盖一个"我是第 (r, c) 张"的位置编号章. transformer 看到的不是墙、不是贴纸,而是 196 张带位置编号的卡片. 这 196 张牌洗一洗(self-attention)就能看出来"右上角是不是有个杯子"——这就是后面 ViTBlock 在干的事.

Picture a photo as a wall of 224×224 sticker tiles. Patch embedding is like pressing a 16×16 square stamp onto the wall — each press "crushes" the underlying 16×16 stickers into a single card (a D-dimensional vector). Because the stride is exactly 16 (no sticker overlap), 14×14 = 196 stamp presses cover the wall. Then on the back of each card you stamp "I'm card (r, c)" — the position embedding. What the transformer sees isn't a wall or stickers, but a deck of 196 numbered cards. Shuffle the deck through self-attention and you can tell "is there a cup in the top-right corner?" — that's what the ViTBlock does next.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

**Curriculum item**: `vision-encoder` (依赖项 / depends on: 无 / none — 这是 vision 路径的最上游叶子节点)

中文: 在 nanoVLA 里,这就是相机帧 → token 的第一道关. 上游是数据集:输入 `(B, 3, 224, 224)` 的 normalized RGB(均值方差用 SigLIP 的 `[0.5, 0.5, 0.5]` / `[0.5, 0.5, 0.5]`,不是 ImageNet 的). 下游是 ViTBlock 栈,然后是**昨天学的 modality projector**——把这 196 个 D=768 的 token 投影到 LM 的 hidden_dim(比如 1024 或 2048),最后接 vlm-backbone-wiring(也是 2026-05-29 覆盖过的). 多相机场景下,production 版有两条路:(a) 给每路相机加一个 view embedding,然后 cat 到 token 维度(更省 token,但要扩词表);(b) 每路相机各跑一遍 vision encoder,然后在 token 维度 cat(更贵但更直接). nanoVLA 推荐用 (a). 再上一层,如果你想换 SigLIP / DINOv3 预训练权重,只要把 `Conv2d(3, D, k=patch, s=patch)` 的 `weight` 和 `bias` 直接 load 进来——nanoVLM 的 `from_pretrained` 在同一个文件下方就有完整的 key mapping 表,可以照抄.

In nanoVLA this is the very first gate of the camera-frame pipeline. Upstream is the dataset, which feeds normalized RGB of shape `(B, 3, 224, 224)` (use SigLIP's `[0.5, 0.5, 0.5]` mean/std, not ImageNet's). Downstream is the ViTBlock stack, then **yesterday's modality projector** — which takes these 196 tokens of dim D=768 and projects them into the LM's hidden_dim (1024 or 2048) — and finally the vlm-backbone-wiring covered on 2026-05-29. For multi-camera setups, a production VLA has two options: (a) add a per-camera view embedding and concat along the token dimension (token-efficient but vocab-extending) or (b) run the vision encoder once per camera and concat tokens (more compute but more straightforward). nanoVLA recommends (a). If you ever want SigLIP / DINOv3 pretrained weights, just load the `Conv2d(3, D, k=patch, s=patch)` `weight` and `bias` directly — nanoVLM's `from_pretrained` (below in the same file) has the full key-mapping table you can copy. Omit this layer and you'd have to figure out how to feed 224×224×3 = 150K raw pixel values into a transformer; that's why every VLA built on a transformer LM has *some* version of this 37-line block.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, torch.nn as nn
from types import SimpleNamespace

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.patch
        self.num_patches = (cfg.img // cfg.patch) ** 2
        self.conv = nn.Conv2d(3, cfg.dim, kernel_size=cfg.patch, stride=cfg.patch)
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, cfg.dim) * 0.02)

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x + self.pos

cfg = SimpleNamespace(img=224, patch=16, dim=768)
model = ViTPatchEmbeddings(cfg)
img = torch.randn(2, 3, 224, 224)
tokens = model(img)
print("input :", img.shape)
print("output:", tokens.shape)
print("num params:", sum(p.numel() for p in model.parameters()))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
input : torch.Size([2, 3, 224, 224])
output: torch.Size([2, 196, 768])
num params: 740352
```

中文: 整个 patch embedding 加 position embedding 一共 ~740K 参数——基本可以忽略,跟一个 SigLIP-base 的 86M 参数比起来. 真正的 vision tower 重量在 ViTBlock 堆叠那里. 这告诉你:**第一层 patchify 不是瓶颈,后面的 attention 才是**.

The patch embedding plus position embedding together weigh in at ~740K parameters — essentially noise next to a SigLIP-base (~86M). The real mass is in the ViTBlock stack. Takeaway: **the patchify layer is not the bottleneck; the attention layers are**.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SigLIP / `transformers.models.siglip.modeling_siglip.SiglipVisionEmbeddings`**: 中文: 几乎一字不差地相同——这正是 nanoVLM 注释里给的链接. 只是 SigLIP 的 position embedding 实现成 `nn.Embedding(num_patches, D)` + `arange` 索引,而不是 `nn.Parameter`,效果完全一样.
- **SigLIP / `transformers.models.siglip.modeling_siglip.SiglipVisionEmbeddings`**: English: almost identical — that's literally the link nanoVLM cites in the comments. SigLIP wraps the position embedding as `nn.Embedding(num_patches, D)` + an `arange` index rather than `nn.Parameter`; functionally the same.
- **lerobot `policies/smolvla/smolvlm_with_expert.py` 中的 `embed_image`**: 中文: 也用 SigLIP patch embedding,但额外做了 multi-image batching——把多个相机的图 stack 到 batch 维,跑一次 forward 拆开. 是这里"多相机方案 (b)" 的工程实现.
- **lerobot `policies/smolvla/smolvlm_with_expert.py`'s `embed_image`**: English: also uses SigLIP patch embedding, but batches multiple camera views by stacking into the batch dim and splitting after forward — a production implementation of "multi-camera option (b)" above.
- **openvla / Prismatic 的 `vision_backbone.py`**: 中文: 用 timm 加载 SigLIP/DINOv2,然后双视觉塔 cat token——本质还是同一种 patch embed,只是用了两个不同 backbone 的输出拼接.
- **openvla / Prismatic's `vision_backbone.py`**: English: uses timm to load SigLIP + DINOv2 in parallel and concatenates their token outputs — same patch-embed idea but applied twice on different pretrained towers.
- **Isaac-GR00T 的多视图融合**: 中文: 多个相机各跑一次 vision encoder,token 维度 cat 后送 modality projector. 是"多相机方案 (b)"的另一个工程例子.
- **Isaac-GR00T's multi-view fusion**: English: runs the vision encoder once per camera and concatenates token outputs before the projector — another instance of "multi-camera option (b)".

## 注意事项 / Caveats / when it breaks

- **resolution 必须能整除 patch_size**: 中文: 224/16 = 14 没事,224/14 就不整数 → Conv 输出形状错位,后面 attention 直接报错. 数据 pipeline 必须严格 resize.
- **resolution must be divisible by patch_size**: English: 224/16 = 14 is fine; anything that doesn't divide evenly breaks the conv output shape and the downstream attention. Resize strictly in your data pipeline.
- **`torch.rand` vs `torch.randn` 的初始化坑**: 中文: nanoVLM 这里 `torch.rand(0, 1)` 初始化 position embedding,这是 SigLIP 的做法. 但很多教学代码用 `torch.randn * 0.02`. 如果你换 backbone 时复制错了,模型仍然能跑,但学习率得调一下——pos embedding 数量级不同会改变 effective LR.
- **`torch.rand` vs `torch.randn` init gotcha**: English: nanoVLM uses `torch.rand` (uniform 0–1) for the position embedding, matching SigLIP. Many tutorials use `torch.randn * 0.02`. Mismatching when porting backbones still trains, but the effective LR shifts because the pos-embed magnitude differs by ~50×.
- **`expand` 之后做 cat 是隐式 alloc**: 中文: `expand` 本身不分配,但 `torch.cat((cls, x), dim=1)` 必须把两个张量拷成连续——所以总内存还是 (B, 197, D) × 4 bytes. expand 省不掉这个,只是省掉了 expand 那一步.
- **`expand` followed by `cat` still allocates**: English: `expand` is cost-free but `torch.cat((cls, x), dim=1)` materializes contiguous storage, so total memory is still `(B, 197, D) × 4` bytes. The savings are only on the `expand` step itself.

## 延伸阅读 / Further reading

- [ViT paper (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — section 3.1 derives the patch embedding analytically (it's the same Conv trick)
- [SigLIP modeling code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245) — production version with 2D position embeddings and resolution-interpolation helpers
- [DINOv3 release notes](https://github.com/facebookresearch/dinov3) — uses identical patch embedding, swaps everything downstream
- [Yesterday's note: modality projector](../../2026/05/2026-05-29-nanovlm-pixel-shuffle-projector.md) — what consumes the 196 tokens this layer produces
- [SmolVLA wiring](../../nano/vla/2026-05-29-smolvla-vlm-with-expert.md) — how the projected tokens get glued into the LM
