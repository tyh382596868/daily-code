---
date: 2026-08-12
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/smolvla/modeling_smolvla.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py#L503-L589
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, prefix-embedding]
build_role: vlm-backbone-wiring advanced variant, prefix assembly for image, language, and robot state tokens
---

# SmolVLA prefix embedding：把图像、语言、状态排成同一条前缀 / SmolVLA Prefix Embedding: Put Images, Language, and State into One Prefix

> **一句话 / In one line**: `embed_prefix()` 把多相机图像 token、语言 token、状态 token 拼成一条 prefix，并同时构造 padding mask 与 attention-role mask。 / `embed_prefix()` concatenates multi-camera image tokens, language tokens, and state tokens into one prefix while building padding and attention-role masks.

## 为什么重要 / Why this matters

VLA 的难点不是“有图像编码器”和“有语言模型”这么简单，而是这些 token 怎么在同一个 transformer 序列里排队。SmolVLA 在 prefix 侧完成三件事：图像经 SigLIP/VLM embedding，语言经词嵌入，机器人状态经线性投影；然后用 mask 告诉后面的 action expert 哪些位置是真 token、哪些位置属于 state/action 区域。

A VLA is not just "an image encoder plus a language model." The real contract is how those tokens line up in one transformer sequence. SmolVLA's prefix path embeds images through the VLM, language through token embeddings, and robot state through a projection, then builds masks that tell the action expert which slots are real and which belong to the state/action region.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/smolvla/modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py#L503-L589)

```python
def embed_prefix(
    self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed images with SigLIP and language tokens with embedding layer to prepare
    for SmolVLM transformer processing.
    """
    embs = []
    pad_masks = []
    att_masks = []
    for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
        if self.add_image_special_tokens:
            image_start_token = (
                self.vlm_with_expert.embed_language_tokens(
                    self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                )
                .unsqueeze(0)
                .expand(img.shape[0], -1, -1)
            )
            image_start_mask = torch.ones_like(
                image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
            )
            att_masks += [0] * (image_start_mask.shape[-1])
            embs.append(image_start_token)
            pad_masks.append(image_start_mask)
        img_emb = self.vlm_with_expert.embed_image(img)
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

        bsize, num_img_embs = img_emb.shape[:2]
        img_mask = img_mask[:, None].expand(bsize, num_img_embs)

        embs.append(img_emb)
        pad_masks.append(img_mask)
        att_masks += [0] * (num_img_embs)

    lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)
    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    att_masks += [0] * lang_emb.shape[1]

    state_emb = self.state_proj(state)
    state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
    embs.append(state_emb)
    state_mask = torch.ones(state_emb.shape[:2], dtype=torch.bool, device=state_emb.device)
    pad_masks.append(state_mask)
    att_masks += [1] * (state_emb.shape[1])

    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]
    if pad_masks.shape[1] < self.prefix_length:
        embs = pad_tensor(embs, self.prefix_length, pad_value=0)
        pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
        att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
    att_masks = att_masks.expand(state_emb.shape[0], -1)
    return embs, pad_masks, att_masks
```

## 逐行讲解 / What's happening

1. **图像循环 / Image loop**:
   - 中文: 每路相机先可选加 start token，再通过 `embed_image()` 变成视觉 token；`img_mask` 从一维相机有效性扩展到每个 patch token。
   - English: Each camera optionally gets a start token, then `embed_image()` turns it into vision tokens; `img_mask` expands from one camera flag to every patch token.
2. **尺度归一 / Embedding scale**:
   - 中文: 图像和语言 embedding 都乘以 hidden dim 的平方根，让它们进入 transformer 时量级接近。
   - English: Image and language embeddings are scaled by the square root of hidden size so their magnitudes are comparable when entering the transformer.
3. **状态 token / State token**:
   - 中文: 机器人状态不走 tokenizer，而是 `state_proj` 直接投影到同一 hidden size，并补成序列维。
   - English: Robot state does not go through a tokenizer; `state_proj` maps it directly into the shared hidden size and adds a sequence dimension.
4. **三张表一起返回 / Three tensors return together**:
   - 中文: `embs` 是实际 token，`pad_masks` 表示哪些位置有效，`att_masks` 表示哪些位置进入 state/action 注意力区域。
   - English: `embs` carries actual tokens, `pad_masks` marks valid positions, and `att_masks` marks the state/action attention region.

## 类比 / The analogy

像开会前排座位：摄像头报告坐前排，语言指令坐中间，机器人状态坐最后一排；座位表之外还要有签到表和分组颜色。

It is like seating people before a meeting: camera reports sit first, language instructions in the middle, robot state at the end. Besides the seating chart, you also need attendance marks and group colors.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `vlm-backbone-wiring` 的高级变体。你的 nanoVLA 里可以把它写成 `build_prefix(obs_images, text_tokens, state)`：输出 `prefix_emb`, `prefix_pad_mask`, `prefix_role_mask`。上游是视觉 encoder、tokenizer、状态 normalizer；下游是 VLM backbone 和 action head。如果省掉这层，action head 会拿不到统一序列，也无法区分图像/语言/state 的权限边界。

This is an advanced `vlm-backbone-wiring` variant. In a nanoVLA, this can be `build_prefix(obs_images, text_tokens, state)`, returning `prefix_emb`, `prefix_pad_mask`, and `prefix_role_mask`. Upstream are the vision encoder, tokenizer, and state normalizer; downstream are the VLM backbone and action head. If you omit this layer, the action head has no unified sequence and no boundary between image, language, and state privileges.

## 自己跑一遍 / Try it yourself

```python
def prefix(images, image_ok, words, state):
    embs, pad, role = [], [], []
    for img, ok in zip(images, image_ok):
        embs += [f"img:{img}:p0", f"img:{img}:p1"]
        pad += [ok, ok]
        role += [0, 0]
    embs += [f"tok:{w}" for w in words]
    pad += [True] * len(words)
    role += [0] * len(words)
    embs += [f"state:{x}" for x in state]
    pad += [True] * len(state)
    role += [1] * len(state)
    return embs, pad, role

print(prefix(["front", "wrist"], [True, False], ["pick", "cup"], [0.2]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(['img:front:p0', 'img:front:p1', 'img:wrist:p0', 'img:wrist:p1', 'tok:pick', 'tok:cup', 'state:0.2'], [True, True, False, False, True, True, True], [0, 0, 0, 0, 0, 0, 1])
```

这个小例子保留了源码的核心合同：token 序列、padding mask、role mask 必须同步增长。

This toy version keeps the source contract: token sequence, padding mask, and role mask must grow in sync.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi PaliGemma prefix** / **openpi PaliGemma prefix**: 也把图像/文本作为 prefix，把动作相关 token 放到 suffix。
- **OpenVLA predict_action** / **OpenVLA predict_action**: 语言和图像先走 VLM，动作在输出侧再解码和反归一化。

## 注意事项 / Caveats / when it breaks

- **mask 长度必须对齐** / **Mask length must align**: `embs`、`pad_masks`、`att_masks` 任意一个少一个 token，attention 就会错位。
- **多相机无效帧不能直接删** / **Invalid cameras should not just disappear**: 保持位置但用 mask 关闭，序列布局才稳定。

## 延伸阅读 / Further reading

- [SmolVLA `embed_prefix`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py#L503-L589)
