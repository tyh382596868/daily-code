---
date: 2026-07-09
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L200-L229
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, cross-attention, image-conditioning]
build_role: text-conditioning advanced variant
---

# Wan2.1 I2V cross-attention：图像和文本分两路看，再相加 / Wan2.1 I2V Cross-Attention: Attend to Image and Text Separately, Then Add

> **一句话 / In one line**: I2V cross-attention 把 `context` 前半段当图像 token、后半段当 T5 文本 token，分别做 attention 后把结果相加。 / I2V cross-attention treats the first part of `context` as image tokens and the rest as T5 text tokens, attends to them separately, then adds the outputs.

## 为什么重要 / Why this matters

图生视频不是“多塞一张图”这么简单。图像条件和文本条件的 token 来源、长度和语义不同。Wan2.1 这里没有把两类 token 混进一套 K/V，而是给图像条件单独的 `k_img/v_img` 投影，让模型可以用不同参数读取参考图。

Image-to-video is not just "add one more image." Image and text conditions have different sources, lengths, and semantics. Wan2.1 does not mix them into one shared K/V projection; it gives image conditioning its own `k_img/v_img` projections so the model can read the reference image differently from text.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L200-L229)

```python
def forward(self, x, context, context_lens):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
        context_lens(Tensor): Shape [B]
    """
    image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
    context_img = context[:, :image_context_length]
    context = context[:, image_context_length:]
    b, n, d = x.size(0), self.num_heads, self.head_dim

    # compute query, key, value
    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)
    k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
    v_img = self.v_img(context_img).view(b, -1, n, d)
    img_x = flash_attention(q, k_img, v_img, k_lens=None)
    # compute attention
    x = flash_attention(q, k, v, k_lens=context_lens)

    # output
    x = x.flatten(2)
    img_x = img_x.flatten(2)
    x = x + img_x
    x = self.o(x)
    return x
```

## 逐行讲解 / What's happening

1. **第 208-210 行 / Lines 208-210 (split context)**:
   - 中文: 总 context 减掉固定的 T5 token 数，剩下的前缀就是图像上下文。
   - English: The total context length minus the fixed T5 token count gives the image-context prefix.
2. **第 214-218 行 / Lines 214-218 (separate projections)**:
   - 中文: 文本走 `k/v`，图像走 `k_img/v_img`；query 仍来自视频 latent `x`。
   - English: Text uses `k/v`, image uses `k_img/v_img`, while the query still comes from video latent `x`.
3. **第 219-221 行 / Lines 219-221 (two attention calls)**:
   - 中文: 图像 attention 不传 `context_lens`，文本 attention 才按文本长度 mask。
   - English: Image attention receives no `context_lens`; text attention uses the text-length mask.
4. **第 224-227 行 / Lines 224-227 (add then output)**:
   - 中文: 两路结果在 head flatten 后相加，再过同一个输出投影。
   - English: The two outputs are flattened, added, and passed through the shared output projection.

## 类比 / The analogy

这像写菜谱时同时看成品照片和文字步骤：照片告诉你形状和颜色，文字告诉你流程。你会分别读它们，最后在脑子里合成一个动作计划。

It is like cooking from both a finished dish photo and written instructions. The photo gives shape and color; the text gives process. You read them separately and combine them into one plan.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `text-conditioning` 的高级变体，也可以看作 `image-conditioning` 插件。上游把参考图编码成 image tokens，把 prompt 编成 T5 tokens；这里把两者接入 DiT block 的 cross-attention；下游 FFN/输出层继续更新视频 latent。如果 nanoWAM 只做 T2V，可以省掉 `k_img/v_img`；一旦做 I2V 或机器人观测条件，就需要类似分路。

This is an advanced `text-conditioning` variant, or an `image-conditioning` plugin. Upstream encodes the reference image into image tokens and the prompt into T5 tokens; this block injects both into DiT cross-attention; downstream FFN/output layers continue updating video latents. A T2V-only nanoWAM can omit `k_img/v_img`, but I2V or robot-observation conditioning needs a similar split.

## 自己跑一遍 / Try it yourself

```python
context = list(range(10))
t5_tokens = 4
image_context_length = len(context) - t5_tokens
context_img = context[:image_context_length]
context_txt = context[image_context_length:]
print(context_img)
print(context_txt)
print([i + t for i, t in zip([10, 20], [1, 2])])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0, 1, 2, 3, 4, 5]
[6, 7, 8, 9]
[11, 22]
```

前两行模拟 context 切分；最后一行模拟两路 attention 结果相加。

The first two lines simulate context splitting; the last line simulates adding the two attention outputs.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SD3 / MMDiT dual stream** / **SD3 / MMDiT dual stream**: 图像和文本先分开投影，再在 attention 汇合。 / Image and text are projected separately before meeting in attention.
- **VLA observation conditioning** / **VLA observation conditioning**: 机器人图像和语言指令也常需要不同投影头。 / Robot images and language instructions often need separate projection heads too.

## 注意事项 / Caveats / when it breaks

- **固定 T5 长度是假设** / **Fixed T5 length is an assumption**: `T5_CONTEXT_TOKEN_NUMBER` 必须和上游 padding/截断一致。 / `T5_CONTEXT_TOKEN_NUMBER` must match upstream padding/truncation.
- **相加会隐藏权重比例** / **Addition hides weighting**: 如果图像条件太强或太弱，可能需要 gate 或 scale。 / If image conditioning is too strong or weak, a gate or scale may be needed.

## 延伸阅读 / Further reading

- [Wan2.1 `WanI2VCrossAttention.forward`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L200-L229)
