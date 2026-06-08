---
date: 2026-06-03
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L171-L251
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, vision-encoder, vit, weight-loading, qkv-fusion]
build_role: vision-encoder
---

# 把 SigLIP 的预训练权重灌进自己的"fused-QKV" ViT / Loading SigLIP's pretrained weights into your own fused-QKV ViT

> **一句话 / In one line**: HF SigLIP 把 Q/K/V 存成三个独立 Linear,你写的从零 ViT 把它们融成一个 Linear —— 80 行加载脚本 + 一行 `torch.cat` 就把权重正确搬过来,既保留了预训练能力,又快了 3×。 / HF SigLIP stores Q/K/V as three separate Linears; your from-scratch ViT fuses them into one — an 80-line loader plus one `torch.cat` migrates the weights correctly, keeping the pretrained knowledge *and* shaving 3× the matmul launches.

## 为什么重要 / Why this matters

搭 nanoVLA 时,**vision encoder** 这个槽位你有两个选择:(1) `transformers.AutoModel.from_pretrained("google/siglip-base-patch16-224")`,几行搞定但你失去了对架构和速度的控制;(2) 自己实现 ViT 但**用 SigLIP 的预训练权重启动**,因为视觉 backbone 从零训练太贵。难点是 (2):HF SigLIP 的 `Q/K/V` 是三个 `nn.Linear`,你的 ViT 通常融成一个 `qkv_proj`(快 3 倍,一次 matmul);两边的 state dict key 完全对不上。nanoVLM 给了一个**只有 80 行**的标准模板:逐层 build mapping、对 `position_embedding` 做 unsqueeze、对 Q/K/V 三个权重做 `torch.cat`、剩下的 layer norm 和 MLP 一对一搬。读懂这份代码,你就掌握了从 SigLIP / DINOv2 / CLIP 等任何 HF 视觉模型把权重灌进自己 from-scratch ViT 的通用方法 —— 这是 nanoVLA 的真正起点。

When you build a nanoVLA, the **vision encoder** slot gives you two choices: (1) `transformers.AutoModel.from_pretrained("google/siglip-base-patch16-224")` — three lines, but you give up control of architecture and speed; (2) write your own ViT and **bootstrap from SigLIP's pretrained weights**, because the vision backbone is too expensive to train from scratch. Option (2) is the hard one: HF SigLIP keeps `Q/K/V` as three separate `nn.Linear`s, while your from-scratch ViT typically fuses them into one `qkv_proj` (3× fewer matmul launches). The state-dict keys don't line up. nanoVLM ships an **80-line** standard template: build a per-layer mapping, `unsqueeze` the position embedding, `torch.cat` the three Q/K/V weight slabs, and copy the rest LN-by-LN and MLP-by-MLP. Master this file and you can bootstrap any HF vision model (SigLIP / DINOv2 / CLIP / DINOv3) into your own from-scratch ViT — the real starting point of a nanoVLA.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L171-L251)

```python
@classmethod
def from_pretrained(cls, cfg):
    from transformers import SiglipVisionConfig
    from huggingface_hub import hf_hub_download
    import safetensors

    hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
    # Copy HF hyper-params into our config so cls(cfg) builds the right shape.
    cfg.vit_dropout    = hf_config.attention_dropout
    cfg.vit_hidden_dim = hf_config.hidden_size
    cfg.vit_img_size   = hf_config.image_size
    cfg.vit_inter_dim  = hf_config.intermediate_size
    cfg.vit_ln_eps     = hf_config.layer_norm_eps
    cfg.vit_n_heads    = hf_config.num_attention_heads
    cfg.vit_n_blocks   = hf_config.num_hidden_layers
    cfg.vit_patch_size = hf_config.patch_size
    model = cls(cfg)
    safetensors_file = hf_hub_download(repo_id=cfg.vit_model_type, filename="model.safetensors")

    sd = model.state_dict()

    # 1) Embedding + post-norm names that are stable across all blocks.
    mapping = {
        'vision_model.embeddings.patch_embedding.weight': 'patch_embedding.conv.weight',
        'vision_model.embeddings.patch_embedding.bias':   'patch_embedding.conv.bias',
        'vision_model.embeddings.position_embedding.weight': 'patch_embedding.position_embedding',
        'vision_model.post_layernorm.weight': 'layer_norm.weight',
        'vision_model.post_layernorm.bias':   'layer_norm.bias',
    }

    # 2) Per-block: layer norms, MLP, attention OUT projection.  (Q/K/V handled below.)
    for i in range(cfg.vit_n_blocks):
        mapping[f'vision_model.encoder.layers.{i}.layer_norm1.weight'] = f'blocks.{i}.ln1.weight'
        mapping[f'vision_model.encoder.layers.{i}.layer_norm1.bias']   = f'blocks.{i}.ln1.bias'
        mapping[f'vision_model.encoder.layers.{i}.layer_norm2.weight'] = f'blocks.{i}.ln2.weight'
        mapping[f'vision_model.encoder.layers.{i}.layer_norm2.bias']   = f'blocks.{i}.ln2.bias'
        mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.weight'] = f'blocks.{i}.mlp.fc1.weight'
        mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.bias']   = f'blocks.{i}.mlp.fc1.bias'
        mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.weight'] = f'blocks.{i}.mlp.fc2.weight'
        mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.bias']   = f'blocks.{i}.mlp.fc2.bias'
        mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.weight'] = f'blocks.{i}.attn.out_proj.weight'
        mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.bias']   = f'blocks.{i}.attn.out_proj.bias'

    with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
        for hf_key, our_key in mapping.items():
            if hf_key in f.keys() and our_key in sd:
                tensor = f.get_tensor(hf_key)
                if tensor.shape == sd[our_key].shape:
                    sd[our_key].copy_(tensor)
                else:
                    # 3) position_embedding ships as [N, D] in HF, we store [1, N, D].
                    if 'position_embedding' in hf_key:
                        sd[our_key].copy_(tensor.unsqueeze(0))
                    else:
                        print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")

        # 4) THE KEY TRICK: fuse Q/K/V into one qkv_proj by concatenating along dim=0.
        for i in range(model.cfg.vit_n_blocks):
            q_w = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.weight')
            k_w = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.weight')
            v_w = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.weight')
            qkv_w = torch.cat((q_w, k_w, v_w), dim=0)
            sd[f'blocks.{i}.attn.qkv_proj.weight'].copy_(qkv_w)

            q_b = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.bias')
            k_b = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.bias')
            v_b = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.bias')
            qkv_b = torch.cat((q_b, k_b, v_b), dim=0)
            sd[f'blocks.{i}.attn.qkv_proj.bias'].copy_(qkv_b)

    model.load_state_dict(sd)
    return model
```

## 逐行讲解 / What's happening

1. **从 HF config 反推自己模型的超参 / Pull hyperparams from HF config**:
   - 中文: 第一步先 download `SiglipVisionConfig`,然后把 `hidden_size`、`num_attention_heads` 等八个字段写回自己的 `cfg`。这一步关键 —— 如果你写死 `cfg.vit_hidden_dim=768` 但下载的是 large(1024 dim),后面 `cls(cfg)` 构出来的模型 shape 就和权重对不上,bug 灾难。
   - English: download `SiglipVisionConfig` and copy eight fields (`hidden_size`, `num_attention_heads`, ...) back into your own `cfg`. Crucial — if you hard-coded `cfg.vit_hidden_dim=768` and the user downloaded the large variant (1024-dim), `cls(cfg)` builds a model whose shapes don't match the weights, and you'll be staring at a wall of "size mismatch" errors.

2. **静态 mapping(embeddings + post-norm)/ The static mapping (embeddings + post-norm)**:
   - 中文: `patch_embedding.weight` 和 `patch_embedding.bias` 直接对应 HF 的 Conv2d patch embed。注意 HF 用的 key 是 `'vision_model.embeddings.patch_embedding.weight'`,而我们叫 `'patch_embedding.conv.weight'` —— 命名差异是这类 loader 的全部困难。
   - English: `patch_embedding.weight` / `.bias` map straight to HF's Conv2d patch embed. Note HF's key is `'vision_model.embeddings.patch_embedding.weight'` while ours is `'patch_embedding.conv.weight'` — naming differences are the entire pain of this kind of loader.

3. **逐层 mapping (LN + MLP + out_proj) / Per-block mapping (LN + MLP + out_proj)**:
   - 中文: 这一块平凡 —— 12 个或 24 个 block,每层 10 个 key 对应。可以挑出来写个循环,代码量瞬间砍半。HF SigLIP 的层 norm 叫 `layer_norm1/2`,自家叫 `ln1/2`;MLP 叫 `mlp.fc1/2`,自家也叫 `mlp.fc1/2` —— 巧合。
   - English: trivial — 12 or 24 blocks, 10 keys each, all named in a regular pattern, so one loop kills it. HF SigLIP calls its layer norms `layer_norm1/2` and ours are `ln1/2`; both call the MLP layers `mlp.fc1/2` — that's a happy coincidence.

4. **`position_embedding` 形状的隐藏陷阱 / Hidden shape trap on `position_embedding`**:
   - 中文: HF 保存的是 `[N, D]`,但你的 `position_embedding` 是 `nn.Parameter(torch.rand(1, N, D))`(为了广播)。形状不同 → 走 `unsqueeze(0)` 分支。这是 loader 里**唯一**靠 try/except 的地方 —— 它先尝试直接 copy,失败了再按形状不匹配的方式处理。这个模式比 hard-code 维度 robust:加新 key 不会破。
   - English: HF stores it as `[N, D]`, but our `position_embedding` is `nn.Parameter(torch.rand(1, N, D))` (for broadcasting). Shape mismatch → take the `unsqueeze(0)` branch. This is the **only** place the loader uses try-then-fallback — it copies first, and only on shape-mismatch goes to the unsqueeze branch. More robust than hard-coding dims: adding new keys won't break it.

5. **核心 trick:Q/K/V `torch.cat(..., dim=0)` 融合 / The key trick: `torch.cat(..., dim=0)` to fuse Q/K/V**:
   - 中文: HF 的 SigLIP attention 是三次独立 matmul (`q_proj`, `k_proj`, `v_proj`),每次 `[D, D]` 权重。我们融成一个 `[3D, D]` 的 `qkv_proj`,一次 matmul,然后 `output.split(D, dim=2)` 切回 q/k/v。要让这个融合层和分开层**数值等价**,权重 cat 的顺序必须是 `[q; k; v]`(和 forward 里 split 的顺序一致),否则结果完全错乱。bias 同理。这就是为什么 cat 必须在 `dim=0`(out_channels 维)而不是别的维度。
   - English: HF's SigLIP attention runs three independent matmuls (`q_proj`, `k_proj`, `v_proj`), each `[D, D]`. We fuse them into one `[3D, D]` `qkv_proj`, one matmul, then `output.split(D, dim=2)` to recover q/k/v. For the fused version to be **numerically equivalent** to the split version, the weights must be cat-ed in `[q; k; v]` order (matching the forward's split order); any other order silently scrambles results. Same for bias. That's also why cat is on `dim=0` (the output-channel dim), nothing else.

6. **没有 try/except,只有 `if hf_key in f.keys() and our_key in sd` / No try/except — just `if hf_key in f.keys() and our_key in sd`**:
   - 中文: 这两个 guard 让你**可以增量地搭模型**:就算 mapping 里写了 50 个 key,只有 30 个匹配,剩下的 silently 跳过,你的 ViT 启动起来就是"混合状态" —— matched 部分是预训练,unmatched 部分是随机初始化。开发期非常有用。
   - English: these two guards let you **build the model incrementally**. Even if `mapping` has 50 keys and only 30 match, the rest are silently skipped, and your ViT boots in a *partially* pretrained state — matched layers carry weights, unmatched layers keep their random init. Very useful during development.

## 类比 / The analogy

中文:想象你买了一台**法国进口的洗衣机**(HF SigLIP),说明书写得很详细但全是法语,而且**水管接头是三孔的**(Q、K、V 三个 Linear)。你家(自家 ViT)墙上是**一个大三合一接头**(fused `qkv_proj`)。`from_pretrained` 这份代码就是那本《把法国洗衣机装到中式管道》的家装手册:第一步抄洗衣机型号(读 HF config),第二步把法语标签翻成中文(`mapping` 字典),第三步遇到形状不对的水管(`position_embedding`)就加个转接头(`unsqueeze`),最后**把法国三根管子绑成一根插进墙上**(`torch.cat(q, k, v)`)。装完试运行,如果哪根管子标签错位,水会喷一屋子(数值乱码),所以顺序绝不能错。

English: imagine you bought a **French-import washing machine** (HF SigLIP). Its manual is detailed but all in French, and its **water hookup has three separate pipes** (Q, K, V — three Linears). Your apartment (your from-scratch ViT) has **one fat three-in-one socket** (fused `qkv_proj`). `from_pretrained` is your "how to install a French washer in a Chinese plumbing system" handbook: step one, copy the machine's model number (read HF config); step two, translate French labels to local ones (`mapping` dict); step three, where a pipe shape doesn't match (`position_embedding`), add an adapter (`unsqueeze`); finally, **bundle the three French pipes together and plug into the wall socket** (`torch.cat(q, k, v)`). If any pipe gets mislabeled during the bundling, water sprays everywhere (numerical garbage) — order is non-negotiable.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:这是 nanoVLA 课程计划的 **`vision-encoder`** 槽位(`depends_on: []`)。它的输入是 RGB 图像 `[B, 3, H, W]`,输出是 patch token 序列 `[B, N, D]`(N = (H/P) × (W/P),D = hidden_dim)。**上游**是数据 pipeline(图像归一化、resize 到 224 或 384);**下游**是 `modality-projector`(已覆盖 2026-05-29 nanoVLM pixel-shuffle),它把 N 个 D 维 token 变成更少、更"胖"的 token,然后送进 LM。如果你跳过 vision encoder 这一步,模型就只能纯文本对话,这就不再是 VLA 而是 LM。生产级实现还要补:多图 / 多相机融合(把 N_cam 张图各自编码后拼接或加 view-embedding)、视频/temporal handling(LeRobot 把多帧拼成"camera×time"网格)、还有 LoRA 或 frozen flag(大部分 VLA 推理时把 vision encoder freeze 掉)。

English: this is the **`vision-encoder`** slot in the nanoVLA curriculum (`depends_on: []`). Inputs: RGB images `[B, 3, H, W]`; outputs: patch token sequence `[B, N, D]` where `N = (H/P) × (W/P)` and `D = hidden_dim`. **Upstream** is the data pipeline (normalization + resize to 224 or 384); **downstream** is the `modality-projector` (already covered 2026-05-29: nanoVLM pixel-shuffle) which compresses N small tokens into fewer fat tokens before they hit the LM. Skip the vision encoder and you don't have a VLA, you have an LM. A production implementation adds: multi-camera fusion (encode N_cam images then concat or add a view-embedding), temporal handling (LeRobot tiles frames into a "camera × time" grid), and a freeze / LoRA flag (most VLAs freeze the vision encoder at inference).

## 自己跑一遍 / Try it yourself

```python
# Minimal: fuse two separate Linears into one Linear via torch.cat — exact same math.
import torch, torch.nn as nn

torch.manual_seed(0)
D, N = 32, 8

# (a) the "HF SigLIP" style: separate q/k Linears
q_lin = nn.Linear(D, D, bias=True)
k_lin = nn.Linear(D, D, bias=True)

# (b) the "from-scratch" style: one fused qk Linear (we omit v for brevity)
qk_lin = nn.Linear(D, 2 * D, bias=True)

# Migrate weights using the SAME trick as nanoVLM:
with torch.no_grad():
    qk_lin.weight.copy_(torch.cat([q_lin.weight, k_lin.weight], dim=0))  # [2D, D]
    qk_lin.bias.copy_(torch.cat([q_lin.bias,   k_lin.bias],   dim=0))    # [2D]

x = torch.randn(N, D)
q_sep, k_sep = q_lin(x), k_lin(x)               # two separate matmuls
qk_fused = qk_lin(x)                            # one matmul, then split
q_fused, k_fused = qk_fused.split(D, dim=-1)

print("q max diff:", (q_sep - q_fused).abs().max().item())
print("k max diff:", (k_sep - k_fused).abs().max().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
q max diff: 0.0
k max diff: 0.0
```

中文:位级一致 —— 这就是为什么 nanoVLM 敢直接用预训练权重启动一个完全不同结构的 attention 层。把这个最小例子推广到 q/k/v 三个就是 nanoVLM 真实代码里那两行 `torch.cat`。

English: bit-identical — which is *why* nanoVLM can boot a structurally different attention layer from pretrained weights with zero retraining loss. Generalize this two-Linear toy to three (q/k/v) and you have exactly the two `torch.cat` lines from nanoVLM's real loader.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`lerobot` SmolVLA 的 `embed_image`** / **`lerobot` SmolVLA's `embed_image`**: 同一个 `vision-encoder` 槽位,但走 HF Transformers 的 `SiglipVisionModel.forward` 路线(option 1)。读 nanoVLM 这版看你失去了什么样的控制权。 / Same `vision-encoder` slot, but goes the `transformers.SiglipVisionModel.forward` route (option 1). Reading nanoVLM's version teaches you what control you give up.
- **`openpi` 的 `vit.py`** / **`openpi`'s `vit.py`**: 用 JAX 写的 ViT + LoRA 起点,QKV 一样融合但用 `jnp.concatenate`;同一思路换个框架。 / JAX ViT + LoRA bootstrap; same QKV fusion but with `jnp.concatenate`. Same idea, different framework.
- **`Isaac-GR00T` `eagle_backbone.py`** / **`Isaac-GR00T`'s `eagle_backbone.py`**: 多相机视觉编码器,先各自走 ViT 再拼接 view-embedding —— vision-encoder 的 "production" 版本。 / Multi-camera vision encoder: each camera through ViT then view-embedding concat — the "production" version of the vision-encoder slot.
- **karpathy `nanoGPT` 的 `c_attn`** / **karpathy nanoGPT's `c_attn`**: 同一个 fused-QKV trick,只不过他从零训练,不需要这套加载脚本 —— 是反过来理解 nanoVLM 的好对照。 / Same fused-QKV trick, but trained from scratch so no loader is needed — a useful contrast to read alongside nanoVLM.

## 注意事项 / Caveats / when it breaks

- **Cat 顺序必须和 split 顺序一致 / Cat order must match split order**: forward 里写 `q, k, v = qkv.split(D, dim=2)` 就必须 cat `[q; k; v]`。一旦顺序反了 (`[k; q; v]`),loss 第一步看起来还像样,因为权重还是预训练的,但模型语义全错,几百步训练后才暴露。**每次重构 attention 模块都要重新验证顺序**。 / If your forward does `q, k, v = qkv.split(D, dim=2)`, you must cat `[q; k; v]`. Reverse the order to `[k; q; v]` and the first-step loss still looks plausible (weights are pretrained), but semantics are scrambled and the bug only surfaces hundreds of training steps in. **Re-verify the order every time you refactor attention.**
- **HF 加 register_buffer 时你也要加 / Match HF's `register_buffer` usage**: 比如 SigLIP 的 `position_ids` buffer 不在权重文件里,但你的模型 forward 可能用它。漏一个 buffer 会让 `state_dict` 多/少 key,影响 EMA/teacher 复制。 / SigLIP keeps a `position_ids` buffer that's not in the weight file but is used at forward time. Missing or extra buffers will change your `state_dict` key set and break EMA/teacher copies.
- **`vit_dropout` 别从 `hf_config.attention_dropout` 拷 / Don't copy `vit_dropout` from `hf_config.attention_dropout`**: 这个值是 HF 默认 0.0,你训练时通常要自己设。这一行其实有点危险,我会建议显式覆盖。 / This HF value defaults to 0.0 and you typically set your own train-time dropout. The line is a soft footgun — I'd override it explicitly.
- **模型变体之间小心 / Variant mismatch**: SigLIP-base 和 SigLIP2-base 的 key 命名不一样(SigLIP2 加了 vision_model 嵌套层),`mapping` 字典换变体时要重看。 / SigLIP-base and SigLIP2-base name keys differently (SigLIP2 adds an extra nesting layer); the `mapping` dict must be re-audited per variant.

## 延伸阅读 / Further reading

- [HuggingFace SigLIP modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)
- [Existing daily-code entry: nanoVLM modality-projector (pixel shuffle)](../../2026/05/2026-05-29-nanovlm-pixel-shuffle-projector.md) — the next stage downstream of this vision encoder.
- [Existing daily-code entry: SmolVLA's `embed_image`](2026-05-29-smolvla-vlm-with-expert.md) — the same slot but consumed by an action expert.
- [karpathy nanoGPT's `c_attn`](https://github.com/karpathy/nanoGPT/blob/master/model.py) — for the fused-QKV pattern in a pure-LM setting.
