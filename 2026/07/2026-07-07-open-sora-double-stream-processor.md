---
date: 2026-07-07
topic: diffusion
source: tracked
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/mmdit/layers.py#L194-L261
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, mmdit]
---

# Open-Sora DoubleStreamBlockProcessor：两条流，一次注意力 / Open-Sora DoubleStreamBlockProcessor: Two Streams, One Attention Call

> **一句话 / In one line**: 图像 token 和文本 token 各自归一化、各自产生 QKV，然后拼到同一次 attention 里交互。 / Image and text tokens are normalized and projected separately, then concatenated into one attention call.

## 为什么重要 / Why this matters

视频 DiT 里的“多模态融合”不一定要写成一个复杂模块。Open-Sora 的 MMDiT 把图像流和文本流保留为两套参数，但在 attention 那一步把 `q/k/v` 拼起来，让两边在同一个注意力池里交换信息。这种设计同时保留了模态专用投影和跨模态通信。

Multimodal fusion in a video DiT does not have to be a large separate subsystem. Open-Sora keeps image and text as separate parameter streams, but concatenates their `q/k/v` tensors for one shared attention call. That preserves modality-specific projections while still allowing cross-modal exchange.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py`](https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/mmdit/layers.py#L194-L261)

```python
class DoubleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
```

## 逐行讲解 / What's happening

1. **第 2-3 行 / Lines 2-3 (`img_mod`, `txt_mod`)**:
   - 中文: 同一个条件向量 `vec` 分别生成图像流和文本流的 shift/scale/gate。
   - English: the same conditioning vector `vec` produces shift/scale/gate values for image and text streams separately.
2. **第 5-16 行 / Lines 5-16 (separate QKV)**:
   - 中文: 两个模态各自做 LayerNorm、adaLN 调制、QKV 投影和 QK norm。
   - English: each modality gets its own LayerNorm, adaLN modulation, QKV projection, and QK normalization.
3. **第 18-20 行 / Lines 18-20 (`torch.cat`)**:
   - 中文: 真正的融合发生在序列维，把文本 token 放前面、图像 token 放后面。
   - English: fusion happens along the sequence dimension: text tokens first, image tokens after them.
4. **第 22-23 行 / Lines 22-23 (split back)**:
   - 中文: attention 输出再按原来的文本长度切回两条流。
   - English: the attention output is split back into text and image streams by the original text length.
5. **第 25-28 行 / Lines 25-28 (gated residuals)**:
   - 中文: gate 控制 attention 分支和 MLP 分支能改写多少原 token。
   - English: gates control how much the attention and MLP branches can rewrite the original tokens.

## 类比 / The analogy

像两个部门先各自准备材料：图片部门用自己的模板，文本部门也用自己的模板。开会时大家坐到同一张会议桌上讨论，会议结束后再回到各自部门执行。

It is like two teams preparing documents with their own templates, then joining one shared meeting table. After the meeting, each team takes its own notes back to its own workflow.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
rng = np.random.default_rng(0)
txt_q = rng.normal(size=(1, 2, 3, 4))
img_q = rng.normal(size=(1, 2, 5, 4))
q = np.concatenate([txt_q, img_q], axis=2)
attn_out = q  # pretend attention returned same shape
txt_out, img_out = attn_out[:, :, :txt_q.shape[2]], attn_out[:, :, txt_q.shape[2]:]
print(q.shape, txt_out.shape, img_out.shape)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
(1, 2, 8, 4) (1, 2, 3, 4) (1, 2, 5, 4)
```

关键点是：融合不改变每条流最终拥有的 token 数，只是在中间共享一次注意力上下文。

The key point is that fusion does not change how many tokens each stream owns; it only shares context during the attention call.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Flux / SD3 MMDiT** / **Flux / SD3 MMDiT**: 图像和文本用双流参数，在 attention 处合流。 / Image and text use dual-stream parameters and meet at attention.
- **Wan2.1 self-attention** / **Wan2.1 self-attention**: 视频 token 内部也常把不同空间时间区域拼到一个 attention kernel。 / Video tokens often concatenate different spatiotemporal regions into one attention kernel.

## 注意事项 / Caveats / when it breaks

- **序列长度切分必须准确** / **The split length must be exact**: `txt_q.shape[2]` 错了，图像和文本 token 会被切乱。 / If `txt_q.shape[2]` is wrong, image and text tokens are sliced incorrectly.
- **内存随拼接后长度增长** / **Memory follows the concatenated length**: 双流合并后 attention 是 `(L_txt + L_img)^2`。 / The attention cost scales with `(L_txt + L_img)^2`.

## 延伸阅读 / Further reading

- [Open-Sora `layers.py`](https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/mmdit/layers.py)
- [Open-Sora technical report](https://github.com/hpcaitech/Open-Sora)
