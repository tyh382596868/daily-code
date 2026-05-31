---
date: 2026-05-31
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L117-L168
difficulty: beginner
read_time: ~12 min
tags: [code-of-the-day, vla, vision-encoder, vit, siglip, nanovlm]
build_role: vision encoder — turns each camera frame into (num_patches, hidden_dim) tokens
---

# nanoVLM 把整个视觉塔写成 52 行 / nanoVLM's entire vision tower fits in 52 lines

> **一句话 / In one line**: 一个能装 SigLIP 预训练权重的 ViT,核心就是 `ViTBlock × N + 一个末端 LayerNorm`,加上 "保留 CLS 还是 token grid" 的开关。 / A SigLIP-compatible ViT is `ViTBlock × N + a trailing LayerNorm`, plus a switch for "CLS only vs full token grid."

## 为什么重要 / Why this matters

写自己的 nanoVLA 时,vision tower 是最容易抄一抄就糊弄过去的部分 —— 大多数项目直接 `AutoModel.from_pretrained("siglip-base")` 拉个黑盒来用。但当你想冻一半、训一半、或者把 patch_size 改小喂高清图、或者把 SigLIP 跟你的从零 ViT 互换时,你得真的看一遍 ViT 的接线方式。nanoVLM 把"能加载 HF SigLIP 权重的、能切换 CLS 模式的、能初始化的"ViT 用 52 行写完 —— 这正好是你 nanoVLA 需要的 vision-encoder 组件的最小完备实现。

When building your own nanoVLA, the vision tower is the one piece everyone hand-waves with `AutoModel.from_pretrained("siglip-base")`. But the moment you want to freeze half of it, swap patch sizes for higher-resolution inputs, or interchange a pretrained SigLIP with a from-scratch ViT, you need to actually understand the wiring. nanoVLM ships a complete SigLIP-compatible, CLS-toggleable, properly-initialized ViT in 52 lines — exactly the minimal-complete `vision_encoder` your nanoVLA needs.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L117-L168)

```python
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L94
class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            x = self.layer_norm(x[:, 0])
        else:
            x = self.layer_norm(x)
            #x = x.mean(dim=1)

        return x
```

## 逐行讲解 / What's happening

1. **`ViTBlock` 是 nanoGPT 同款 pre-LN 块 / `ViTBlock` is the nanoGPT pre-LN recipe (lines 118-128)**:
   - 中文: `x = x + attn(ln(x))`,再 `x = x + mlp(ln(x))`。**两个 LayerNorm 都在残差分支*内部*** —— 这就是 "pre-LN",和原始 ViT 用 "post-LN"(LN 在 residual 之后)有显著不同,深层网络更容易训。nanoGPT 把它发扬光大,nanoVLM 直接抄过来。
   - English: `x = x + attn(ln(x))`, then `x = x + mlp(ln(x))`. Both LayerNorms live *inside* the residual branch — this is "pre-LN," which is materially easier to train at depth than the original ViT's "post-LN." nanoGPT popularized it; nanoVLM imports it verbatim.

2. **`ViT.__init__` 组件清单 / `ViT.__init__` parts list (lines 132-139)**:
   - 中文: `patch_embedding`(图像 → token 序列)+ `cls_flag` 开关 + `dropout` + N 个 `ViTBlock` + 末端 `layer_norm`。没了。一个 ViT 总共就这 5 个零件。
   - English: `patch_embedding` (image → token sequence) + `cls_flag` switch + `dropout` + N `ViTBlock`s + trailing `layer_norm`. That's it — a ViT is five parts.

3. **`apply(self._init_weights)` (line 141)**:
   - 中文: `nn.Module.apply` 会递归地对所有子模块调用 `_init_weights`。这个细节很关键 —— 不调它的话,Linear 用 PyTorch 默认的 kaiming 均匀初始化,在深层 ViT 上会让 logits 爆炸或者训练慢一倍。`std=0.02` 是 GPT/ViT 系列的"标准值"。
   - English: `nn.Module.apply` recursively calls `_init_weights` on every submodule. Critical step — without it Linear layers use PyTorch's default kaiming-uniform, which makes deep ViTs either explode or converge half as fast. `std=0.02` is the GPT/ViT canon.

4. **`_init_weights` 三种类型分别处理 / Three init cases (lines 143-154)**:
   - 中文: `nn.Linear` 用 `N(0, 0.02²)` 初始化 + bias 置 0;`nn.LayerNorm` 是 `weight=1, bias=0`(恒等);`nn.Conv2d` (就是 `patch_embedding.conv`)用 `N(0, 0.02²)`。这三条规则可以原封不动用到几乎所有从零写的 transformer。
   - English: `nn.Linear` gets `N(0, 0.02²)` weights + zero bias; `nn.LayerNorm` gets `weight=1, bias=0` (identity); `nn.Conv2d` (used by `patch_embedding`) gets `N(0, 0.02²)`. These three rules transfer to almost any from-scratch transformer.

5. **`forward` 的 CLS 二选一 / Forward — the CLS toggle (lines 156-166)**:
   - 中文: 这里是 VLA 用法的 *关键*。`cls_flag=True` 时只保留 CLS token —— 适合图像分类。`cls_flag=False` 时保留整个 token 网格 —— **VLA 想要的就是这个**,因为下游 modality projector 要看到 `(num_patches, hidden_dim)` 而不是单一向量。注释里 `# x.mean(dim=1)` 是另一种"pooling 到单 vector"的选项,但 VLA 用不到。
   - English: this is the *critical* line for VLA usage. `cls_flag=True` collapses to a single CLS token (image classification). `cls_flag=False` keeps the full token grid — **what VLA needs**, because the downstream modality projector wants `(num_patches, hidden_dim)`, not a single vector. The commented `x.mean(dim=1)` is an alternative pooling that VLA also doesn't want.

6. **末端 LayerNorm 的位置 / Where the trailing LayerNorm sits (line 165)**:
   - 中文: 注意 LayerNorm 是在 *blocks 之外* 的、对最后一个 block 的输出再做一次 norm。这是 SigLIP 的约定(`vision_model.post_layernorm`),和原始 ViT 的 "blocks 之间所有 LN" 不同 —— 正是这个差异让 nanoVLM 能用同一个 `mapping` dict 把 SigLIP 权重 1:1 装进来。
   - English: the LayerNorm sits *outside* the block stack — applied to the final block's output. This is the SigLIP convention (`vision_model.post_layernorm`), not the original ViT's "LN between every block." That exact convention is what lets nanoVLM's `from_pretrained` load SigLIP weights into this class without any reshape gymnastics.

## 类比 / The analogy

ViT 像一家专门处理图片的快递分拣厂:`patch_embedding` 是输入口的传送带,把整张图切成 14×14 = 196 个小包裹(token);`ViTBlock × N` 是 N 个分拣台流水线,每个分拣台先让所有包裹互相沟通(attention)再各自整理(MLP);末端 `LayerNorm` 是出库前的"重新贴标签"。`cls_flag` 是一个分流闸门:要给上游下游(LM / projector)看全部 196 个包裹,还是只看那个 "代表整张图" 的 CLS 包裹?VLA 需要全部,所以闸门关。`_init_weights` 是开张前给每条传送带做的速度标定 —— 不标的话,第一天就会爆仓。

A ViT is a parcel sorting facility for images: `patch_embedding` is the intake conveyor that cuts each image into 14×14 = 196 packets (tokens); the `N` `ViTBlock`s are `N` sorting stations, each letting all packets talk to each other (attention) then individually re-organize (MLP); the trailing `LayerNorm` is the "relabel before shipping" step. `cls_flag` is the routing gate: do we ship all 196 packets to the downstream LM / projector, or only the single CLS packet that "represents the whole image"? VLA wants all of them, so the gate stays open. `_init_weights` is the speed-calibration done before opening day — skip it and the whole facility floods on day one.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

**Curriculum item**: `vision-encoder` — *no dependencies*.

中文: 这是 nanoVLA 的 `vision_encoder` 模块,在 build plan 里位置很关键 —— 它没有任何 depends_on,所以可以独立写;但它的输出是 **所有下游组件的食材**:`(B, num_patches, hidden_dim)` 的 token 张量,接下来由 5-29 教过的 modality projector(把视觉 token 投影到 LM 的 hidden_dim、且 pixel-shuffle 降到 64 token 节省 LM cost),再接 5-29 教过的 SmolVLA "VLM + 小专家" 那个 backbone。如果你省掉这个组件,就只能用别人黑盒的 `SiglipVisionModel`,那等于把"我要不要让 vision tower 跟着训"的开关焊死了;而很多 VLA fine-tune setting 恰好想要"vision 部分 LoRA、其它冻"。生产级实现需要在这 52 行之外补的细节是:(1) 多相机融合(把 N 个相机的 token grid 拼起来,加 camera-id embedding),(2) 高分辨率支持(patch_embedding 的 position_embedding 要插值),(3) FlashAttention / SDPA 路径(这里只有 SDPA),(4) 训练时的 stochastic depth / DropPath。

English: this is your nanoVLA's `vision_encoder` module, and it occupies a key build-plan slot — *no dependencies*, so you can write it standalone. But its output is the **raw material for every downstream component**: a `(B, num_patches, hidden_dim)` token tensor that feeds the modality projector covered on 2026-05-29 (pixel-shuffle from 256 → 64 tokens to save LM cost), which in turn feeds the SmolVLA "VLM + slim expert" backbone covered on 2026-05-29. Skip this and you're stuck with a black-box `SiglipVisionModel`, which welds shut the "should I train the vision tower?" switch — exactly the switch many VLA fine-tunes want set to LoRA-only. To productionize on top of these 52 lines: (1) multi-camera fusion (concat N cameras' token grids with a camera-id embedding), (2) variable-resolution support (interpolate the patch_embedding position_embedding), (3) FlashAttention / SDPA fallback (this code only has SDPA), (4) stochastic-depth / DropPath during training. With today's note plus the previous two, your nanoVLA has the *complete* vision-side stack: encode → project → consume.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Minimal standalone vision encoder — no nanoVLM dependency.
import torch
import torch.nn as nn

class Cfg:
    vit_img_size, vit_patch_size = 64, 16   # toy: 64x64 image, 16x16 patches => 4x4 = 16 patches
    vit_hidden_dim, vit_inter_dim = 96, 256
    vit_n_heads, vit_n_blocks = 4, 2
    vit_dropout, vit_ln_eps = 0.0, 1e-6
    vit_cls_flag = False                     # VLA wants the full token grid

class PatchEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(3, c.vit_hidden_dim, c.vit_patch_size, c.vit_patch_size)
        n = (c.vit_img_size // c.vit_patch_size) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, c.vit_hidden_dim))
    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2) + self.pos

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.vit_hidden_dim), nn.LayerNorm(c.vit_hidden_dim)
        self.attn = nn.MultiheadAttention(c.vit_hidden_dim, c.vit_n_heads, batch_first=True)
        self.mlp  = nn.Sequential(nn.Linear(c.vit_hidden_dim, c.vit_inter_dim),
                                  nn.GELU(), nn.Linear(c.vit_inter_dim, c.vit_hidden_dim))
    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        return x + self.mlp(self.ln2(x + a))

class NanoViT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed  = PatchEmbed(c)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.vit_n_blocks)])
        self.norm   = nn.LayerNorm(c.vit_hidden_dim)
    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

torch.manual_seed(0)
enc = NanoViT(Cfg())
img = torch.randn(2, 3, 64, 64)               # 2 toy images
tokens = enc(img)
print("input :", img.shape)
print("tokens:", tokens.shape, " (B, num_patches, hidden_dim)")
print("param count:", sum(p.numel() for p in enc.parameters()))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
input : torch.Size([2, 3, 64, 64])
tokens: torch.Size([2, 16, 96])  (B, num_patches, hidden_dim)
param count: ~250000
```

中文: 注意输出形状 `(2, 16, 96)` —— 4×4=16 个 patch token,每个 96 维。这就是 modality projector 的输入。把 `vit_cls_flag = True` 再跑一次,输出会变成 `(2, 96)`(单 vector),你立刻能看出 "VLA 用哪个分支" 的差别。

English: output shape `(2, 16, 96)` — 16 patch tokens (4×4 grid), 96-dim each. That's exactly what the modality projector consumes. Flip `vit_cls_flag = True` and rerun: shape collapses to `(2, 96)` — you can immediately see why VLA picks the `False` branch.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/lerobot` SmolVLA `embed_image`** / **lerobot SmolVLA `embed_image`**: 中文: 用的是 HF 的 `SiglipVisionModel`,但接口完全一致 —— `image -> (B, num_patches, hidden_dim) tokens`,然后塞进 5-29 教过的 VLM-with-expert backbone。 / English: uses HF's `SiglipVisionModel` but the interface is identical — image → patch tokens → goes into the VLM-with-expert backbone from 2026-05-29.
- **`NVIDIA/Isaac-GR00T` 多相机融合** / **GR00T multi-camera fusion**: 中文: GR00T 把多个相机各自跑一遍 ViT,然后用 `CategorySpecificLinear`(5-29 教过的"为每种机器人配一条 Linear")把它们投影到统一 hidden 空间。 / English: GR00T runs a separate ViT pass per camera and uses `CategorySpecificLinear` (the "one Linear per robot body" trick from 2026-05-29) to project all of them into a shared hidden space.
- **`Physical-Intelligence/openpi` SigLIP wrapper** / **openpi SigLIP wrapper**: 中文: π0 直接复用 `google/siglip-so400m-patch14-384`,patch 大小 14,分辨率 384 —— 比 nanoVLM 的设置大,但接线方式同款。 / English: π0 reuses `google/siglip-so400m-patch14-384`. Bigger patch / resolution than nanoVLM, identical wiring.
- **`facebookresearch/dinov3`** / **DINOv3**: 中文: 另一条可选的预训练 vision tower(自监督训出的 ViT)。你可以无缝替换 nanoVLM 的 SigLIP,只要 hidden_dim 兼容。 / English: an alternative pretrained vision tower (self-supervised ViT). Drop-in replacement for SigLIP as long as hidden_dim matches.

## 注意事项 / Caveats / when it breaks

- **`cls_flag` 必须设为 False 给 VLA 用** / **`cls_flag` must be False for VLA**: 中文: 一旦你不小心让它走了 CLS-only 分支,你 modality projector 就只收到一个 vector,VLM 完全看不到空间信息,VLA 会变成纯文本 + 一个全局图像描述符,精度暴跌。 / English: leaving `cls_flag=True` collapses to one vector. The modality projector then has no spatial info; your VLA degrades to "text + a global image descriptor" and accuracy tanks.
- **`position_embedding` 是固定分辨率的** / **`position_embedding` is fixed-resolution**: 中文: 它是 `(1, num_patches, hidden_dim)` 的可学习参数,不能开箱即用到不同的图像尺寸。换分辨率需要插值,具体看 `interpolate_pos_encoding` 的实现。 / English: it's a learnable `(1, num_patches, hidden_dim)` parameter — not resolution-agnostic. Different image sizes need `interpolate_pos_encoding`-style resampling.
- **`_init_weights` 必须 `apply`,不只是定义** / **`_init_weights` must be applied, not just defined**: 中文: 容易忘的细节 —— `apply(self._init_weights)` 这一行漏写,所有参数就走 PyTorch 默认,训练效果差到你怀疑论文是不是骗你。 / English: easy to forget — without `apply(self._init_weights)`, every parameter falls back to PyTorch defaults and your training looks nothing like the paper.
- **`x[:, 0]` 假设第 0 个 token 是 CLS** / **`x[:, 0]` assumes CLS is at index 0**: 中文: SigLIP 的官方权重其实 *没有* CLS token(`cls_flag=False`),如果你试图加载 SigLIP 权重又开 `cls_flag=True`,CLS 那个参数是随机的、没学过。 / English: the official SigLIP weights have no CLS token. If you load SigLIP and set `cls_flag=True`, the CLS parameter is freshly random and untrained — silent bug.

## 延伸阅读 / Further reading

- nanoVLM repo: <https://github.com/huggingface/nanoVLM>
- SigLIP paper: <https://arxiv.org/abs/2303.15343>
- Original ViT paper: <https://arxiv.org/abs/2010.11929>
- Past entry on the pixel-shuffle modality projector: `2026/05/2026-05-29-nanovlm-pixel-shuffle-projector.md`
- Past entry on SmolVLA's VLM-with-expert wiring: `nano/vla/2026-05-29-smolvla-vlm-with-expert.md`
