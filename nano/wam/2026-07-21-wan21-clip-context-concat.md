---
date: 2026-07-21
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L2386-L2435
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, wan2.1, clip, text-conditioning, cross-attention]
component: text-conditioning
variant: advanced
---

# Wan2.1 I2V context：把 CLIP 图像 token 接到文本前面 / Wan2.1 I2V Context: Prepend CLIP Image Tokens Before Text

> **一句话 / In one line**: Wan2.1 在 forward 里先把文本 pad 到固定长度；如果有 `clip_fea`，再经过图像投影并拼到文本 context 前面，供每个 DiT block cross-attend。 / In Wan2.1 forward, text is padded to a fixed length; if `clip_fea` exists, it is projected and prepended before text context so every DiT block can cross-attend to both.

## 为什么重要 / Why this matters

图生视频不是只把首帧塞进 latent。生产模型通常同时给 DiT 两类条件：latent 里有空间起点，cross-attention context 里还有 CLIP 图像语义 token。Wan2.1 的这几行展示了“图像语义 + 文本语义”如何共享同一条 context 通道。

Image-to-video is not just putting the first frame into the latent. Production models often give DiT two kinds of conditioning: a spatial start in latent space and CLIP image semantic tokens in cross-attention context. These Wan2.1 lines show how image and text semantics share one context channel.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L2386-L2435)

```python
# Simplified teaching slice, not a verbatim copy.
context = text_embedding(stack([
    cat([tokens, zeros(text_len - len(tokens), dim)])
    for tokens in text_context
]))

if clip_fea is not None:
    context_clip = image_embedding(clip_fea)
    context = concat([context_clip, context], dim=1)

kwargs = {
    "e": time_modulation,
    "seq_lens": seq_lens,
    "grid_sizes": grid_sizes,
    "freqs": rope_freqs,
    "context": context,
}

for block in blocks:
    x = block(x, **kwargs)
x = head(x, timestep_embedding)
video = unpatchify(x, grid_sizes)
```

## 逐行讲解 / What's happening

1. **文本先定长 / Text is fixed-length first**: 中文: 每条 prompt 被 pad 到 `text_len`，batch 里 shape 一致。 English: each prompt is padded to `text_len`, giving the batch a stable shape.
2. **CLIP 特征过图像 embedding / CLIP features pass an image embedding**: 中文: 图像 token 被投到和文本 token 相同的 DiT hidden 维。 English: image tokens are projected into the same DiT hidden dimension as text tokens.
3. **图像放在文本前 / Image comes before text**: 中文: 拼接后 context 顺序是 `[image tokens, text tokens]`。 English: after concatenation, context order is `[image tokens, text tokens]`.
4. **所有 block 共用 context / All blocks share the context**: 中文: 每个 DiT block 都能 cross-attend 到同一份图文条件。 English: every DiT block can cross-attend to the same image-text condition.
5. **最后仍要 unpatchify / Output still unpatchifies**: 中文: context 只影响预测，输出 token 仍要折回视频 latent。 English: context conditions the prediction, but output tokens still fold back into video latents.

## 在 nanoWAM 中的位置 / Where this fits in nanoWAM

中文: 这是 `text-conditioning` 的高级变体：把“文本条件”升级成“图像语义 + 文本语义”的统一 cross-attention memory。nanoWAM 可以先只接文本，再增加一个可选图像 token 前缀。

English: This is an advanced `text-conditioning` variant: it upgrades text conditioning into a unified cross-attention memory containing image semantics plus text semantics. A nanoWAM can start with text only, then add an optional image-token prefix.

## 类比 / The analogy

像导演给动画师两份参考：先给一张概念图，再给文字分镜。每个制作环节都能同时翻看这两份资料。

It is like a director giving animators two references: a concept image first, then a written storyboard. Every production stage can consult both.

## 自己跑一遍 / Try it yourself

```python
def build_context(text_tokens, text_len, clip_tokens=None):
    padded_text = text_tokens + ["<pad>"] * (text_len - len(text_tokens))
    if clip_tokens is None:
        return padded_text
    return clip_tokens + padded_text

print(build_context(["a", "robot"], text_len=4))
print(build_context(["a", "robot"], text_len=4, clip_tokens=["img0", "img1"]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['a', 'robot', '<pad>', '<pad>']
['img0', 'img1', 'a', 'robot', '<pad>', '<pad>']
```

## 注意事项 / Caveats / when it breaks

- **顺序是隐式协议 / Order is an implicit contract**: 下游 block 不知道 token 类型，只能从顺序和训练分布中学习。 / downstream blocks do not know token type unless order and training distribution teach it.
- **context 长度会变 / Context length changes**: 加图像 token 会增加 cross-attention memory 和显存。 / image tokens increase cross-attention memory and VRAM.
- **空图像分支要稳定 / Missing image branch must be stable**: T2V 和 I2V 共用模型时，`clip_fea is None` 的路径必须同样可靠。 / when T2V and I2V share a model, the `clip_fea is None` path must remain reliable.

## 延伸阅读 / Further reading

- [Wan2.1 model.py context assembly](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L2386-L2435)
- [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1)
