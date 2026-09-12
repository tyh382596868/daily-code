---
date: 2026-08-10
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L350-L369
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, wam, image-conditioning]
build_role: action-conditioning advanced variant, image embedding projection for first-last-frame conditioning
---

# Wan2.1 MLPProj：图像条件也要投影成上下文 token / Wan2.1 MLPProj: Image Conditions Become Context Tokens

> **一句话 / In one line**: `MLPProj` 用 LayerNorm-MLP-LayerNorm 把外部图像 embedding 变成 DiT 可消费的额外上下文 token。 / `MLPProj` converts external image embeddings into extra context tokens that the DiT can consume.

## 为什么重要 / Why this matters

World/action model 不只看噪声 latent 和文字，有时还要看首帧、尾帧或参考图。参考图通常来自 CLIP/vision encoder，维度和 DiT 内部 hidden size 不一致。这个小 projector 就是“接头”：先规范化，再用 MLP 换到模型需要的维度。

A world/action model may condition not only on noisy latents and text, but also on first frames, last frames, or reference images. Those image features often come from CLIP/vision encoders with a different width than the DiT hidden size, so this projector adapts them.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L350-L369)

```python
class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens
```

## 逐行讲解 / What's happening

1. **第 355-358 行 / Lines 355-358 (adapter MLP)**:
   - 中文: `LayerNorm -> Linear -> GELU -> Linear -> LayerNorm` 先稳定输入分布，再换到 `out_dim`。
   - English: `LayerNorm -> Linear -> GELU -> Linear -> LayerNorm` stabilizes input features and projects them to `out_dim`.
2. **第 359-361 行 / Lines 359-361 (first-last-frame mode)**:
   - 中文: `flf_pos_emb` 打开时加一组可学习位置 token，给首帧/尾帧条件提供顺序信息。
   - English: With `flf_pos_emb`, a learned positional tensor gives first/last-frame conditions ordering information.
3. **第 363-367 行 / Lines 363-367 (reshape and add position)**:
   - 中文: 代码把 batch 里的两组帧条件拼成 `2 * n` 个上下文 token，再加位置 embedding。
   - English: The code reshapes two frame-condition groups into `2 * n` context tokens and adds positional embeddings.
4. **第 368-369 行 / Lines 368-369 (return context)**:
   - 中文: 输出就是后续 cross-attention 或 conditioning 路径要吃的 extra context。
   - English: The output is the extra context consumed by later cross-attention or conditioning paths.

## 类比 / The analogy

像给不同规格的插头配转接头：参考图 embedding 是一种电压/接口，DiT hidden state 是另一种，MLPProj 负责把它们接上。

It is like using an adapter for different plugs: image embeddings and DiT hidden states have different interfaces, and `MLPProj` makes them fit.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，它属于 `action-conditioning` / image-conditioning 的输入适配层：`vision_encoder(image) -> projector -> context_tokens -> DiT blocks`。如果你做第一帧到视频、首尾帧到视频或动作条件视频预测，这个模块负责把外部条件变成统一 token 序列。

In a nanoWAM, this is an input adapter for `action-conditioning` or image-conditioning: `vision_encoder(image) -> projector -> context_tokens -> DiT blocks`. For image-to-video, first/last-frame-to-video, or action-conditioned prediction, it turns external conditions into a uniform token sequence.

## 自己跑一遍 / Try it yourself

```python
def project(x, weight, bias):
    return [sum(v * w for v, w in zip(x, col)) + b for col, b in zip(zip(*weight), bias)]

image_embed = [1.0, -1.0, 0.5]
weight = [[0.5, 0.0], [0.5, 1.0], [0.0, -1.0]]
bias = [0.0, 0.25]
print([round(v, 3) for v in project(image_embed, weight, bias)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.0, -1.25]
```

这个 toy projector 省略了 LayerNorm/GELU，但保留了核心形状变化：把 `in_dim` 特征换成 `out_dim` 上下文。

This toy projector skips LayerNorm/GELU but keeps the core shape change: convert `in_dim` features into `out_dim` context.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **VLA modality projector** / **VLA modality projector**: 视觉 token 投到 LM hidden size，本质上也是接口转换。 / Vision tokens are projected into LM hidden size, which is the same adapter pattern.
- **Text encoder bridge** / **Text encoder bridge**: T5/CLIP 文本 embedding 进入 DiT 前也常要投影和归一化。 / T5/CLIP text embeddings are often projected and normalized before entering a DiT.

## 注意事项 / Caveats / when it breaks

- **位置长度要匹配** / **Position length must match**: `FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER` 和 reshape 后 token 数不一致会直接形状报错。 / `FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER` must match the reshaped token count.
- **投影不是信息恢复** / **Projection does not recover information**: 如果上游 image encoder 丢了细节，MLP 只能重排特征，不能凭空补回。 / If the image encoder discarded detail, the MLP can only rearrange features, not recreate them.

## 延伸阅读 / Further reading

- [Wan2.1 `MLPProj`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L350-L369)
