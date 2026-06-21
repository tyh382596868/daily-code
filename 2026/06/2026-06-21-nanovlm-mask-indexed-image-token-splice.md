---
date: 2026-06-21
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/vision_language_model.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L36-L80
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, huggingface, multimodal, vlm, token-splicing, boolean-mask]
---

# 30 行 multimodal fusion:一个布尔 mask 把图像 embedding 缝进 token 流 / 30 lines of multimodal fusion: one boolean mask splices image embeddings into the token stream

> **一句话 / In one line**: nanoVLM 先把所有 token(文字 + 图像占位符)一起 embed,再用 `input_ids == image_token_id` 的布尔 mask 一行原地替换图像位置,不用拼接也不用特殊 attention——30 行把多模态融合讲完了。/ nanoVLM embeds all tokens (text + image placeholders) together, then uses a single boolean mask `input_ids == image_token_id` to swap out the placeholder embeddings in-place — no concatenation, no special attention — multimodal fusion in 30 lines.

## 为什么重要 / Why this matters

初学者做多模态往往想到"先过视觉编码器,再过 LM,中间再来一个 cross-attention 把两路对齐"——这很复杂。nanoVLM 展示了一条更干净的路:把图像 token 的占位符先放进序列里,和文字 token 一起过 `token_embedding`,然后用一个布尔 mask `input_ids == IMAGE_TOKEN_ID` 把占位符 embedding 替换成 ViT 输出的图像 embedding。之后整条序列直接进 LM decoder——decoder 完全不知道哪些 token 来自图像,它只处理 embedding。这就是 LLaVA 系列模型的核心思想:用"替换占位符"代替"cross-attention"。

Beginners often reach for "vision encoder + LM + a cross-attention bridge" — complex and easy to get wrong. nanoVLM shows a cleaner path: put image-token placeholders into the sequence alongside text tokens, embed everything together, then use a single boolean mask to swap placeholder embeddings for ViT outputs. The fused sequence goes straight into the LM decoder, which never "knows" which tokens came from images — it just processes embeddings. This is the core idea behind LLaVA-family models: replace placeholders instead of adding cross-attention.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_language_model.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L36-L80)

```python
def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
    """
    Replace every image-token placeholder in `input_ids` with the corresponding
    slice from `image_embd`. Supports an arbitrary number of images per sample.
    """
    updated_token_embd = token_embd.clone()

    # Build a mask of all image-token positions: shape [B, T_seq]
    mask = (input_ids == self.tokenizer.image_token_id)
    # torch flattens before assigning: one-liner that does all images in all batch items
    updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)

    return updated_token_embd


def forward(self, input_ids, images, attention_mask=None, targets=None):
    images_tensor = self._process_images(images, input_ids.device)

    # Step 1: embed every token (text + image placeholders) as one sequence
    token_embd = self.decoder.token_embedding(input_ids)  # [B, T_seq, D_lm]

    if images_tensor is not None:
        # Step 2: encode images through ViT
        image_embd = self.vision_encoder(images_tensor)   # [N_imgs, T_img, D_vit]
        # Step 3: project ViT dim → LM dim
        image_embd = self.MP(image_embd)                  # [N_imgs, mp_len, D_lm]
        # Step 4: boolean-mask splice — the whole multimodal fusion
        token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

    # Step 5: LM decoder sees one unified embedding sequence
    logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

    loss = None
    if targets is not None:
        logits = self.decoder.head(logits)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
    return logits, loss
```

## 逐行讲解 / What's happening

1. **`token_embd = self.decoder.token_embedding(input_ids)` (Step 1)**
   - 中文: `input_ids` 里包含文字 token 的 id 和 `IMAGE_TOKEN_ID` 占位符。`token_embedding` 是普通的 `nn.Embedding`,对所有 id 统一 embed。占位符的 embedding 是随机初始化的,稍后会被覆盖。
   - English: `input_ids` contains regular token ids and `IMAGE_TOKEN_ID` placeholders. `token_embedding` is a plain `nn.Embedding` that embeds all ids uniformly. The placeholder embeddings are random initializations — they'll be overwritten next.

2. **`mask = (input_ids == self.tokenizer.image_token_id)` (Step 2 of `_replace_img_tokens_with_embd`)**
   - 中文: 布尔张量,形状 `[B, T_seq]`。每个 `True` 位置对应一个图像占位符 token。一个样本里可以有多张图、多个占位符,全被这个 mask 统一覆盖。
   - English: Boolean tensor of shape `[B, T_seq]`. Each `True` marks an image-placeholder token. One sample can have multiple images with multiple placeholder groups — the mask covers all of them uniformly.

3. **`updated_token_embd[mask] = image_embd.view(-1, D)` — 核心替换**
   - 中文: PyTorch 的高级索引:`tensor[bool_mask]` 返回所有 True 位置的行向量,赋值就是逐行替换。`image_embd.view(-1, D)` 把 `[N_imgs, mp_len, D_lm]` 展平成 `[N_total_img_tokens, D_lm]`——这要求 mask 里 True 的个数恰好等于所有图像的 token 总数。
   - English: PyTorch advanced indexing: `tensor[bool_mask]` gathers all rows at True positions; assignment replaces them row-by-row. `image_embd.view(-1, D)` flattens `[N_imgs, mp_len, D_lm]` to `[N_total_img_tokens, D_lm]` — this relies on the mask having exactly as many True entries as the total number of image tokens across the batch.

4. **`self.decoder(token_embd, ...)` (Step 5)**
   - 中文: LM decoder 收到的就是一条普通的 embedding 序列——它没有任何专门的多模态逻辑,不知道也不需要知道哪些 embedding 来自图像。多模态信息已经在 Step 4 被无缝地注入序列里了。
   - English: The LM decoder receives a plain embedding sequence — it has no special multimodal logic and doesn't need to know which embeddings came from images. The multimodal information has already been seamlessly injected in Step 4.

5. **`ignore_index=-100` 在 loss 里**
   - 中文: `targets` 里,图像 token 位置和 prompt 位置通常被设为 `-100`,只有回答部分的 token 参与 cross-entropy。这让模型只在回答 token 上收梯度。
   - English: In `targets`, image-token positions and prompt positions are typically set to `-100`, so only answer tokens contribute to the cross-entropy loss. This constrains gradients to the answer tokens only.

## 类比 / The analogy

想象一份填字游戏答题纸,答案格子先全填"?"占位。文字题目已经印好。然后你拿一张标注了"哪些格子是图片内容"的模板,把"?"全部替换成从图片里提取的关键词。最后把答题纸整张交给 LM 批改——LM 根本不知道这些内容哪些来自图片哪些来自文字,它只看格子里的内容。

Picture a fill-in-the-blank answer sheet where all image slots are pre-filled with "?". Text prompts are already printed. You then lay a stencil over the sheet that marks which slots are "image content", and replace every "?" with keywords extracted from the image. You hand the entire sheet to the LM to grade — it never sees the original images, only the completed embedding sequence.

## 自己跑一遍 / Try it yourself

```python
import torch

B, T_seq, D = 2, 10, 64
IMAGE_TOKEN_ID = 50257
NUM_IMG_TOKENS = 4  # 每张图占 4 个 token
N_imgs = 2          # batch 里共 2 张图

# 模拟 input_ids:每个样本里各有 4 个图像占位符
input_ids = torch.randint(0, 1000, (B, T_seq))
input_ids[0, 2:6] = IMAGE_TOKEN_ID   # 样本 0:位置 2-5 是图像
input_ids[1, 3:7] = IMAGE_TOKEN_ID   # 样本 1:位置 3-6 是图像

token_embd = torch.randn(B, T_seq, D)  # 随机 embedding
image_embd = torch.ones(N_imgs, NUM_IMG_TOKENS, D) * 99.0  # 图像 embedding 全是 99

mask = (input_ids == IMAGE_TOKEN_ID)
updated = token_embd.clone()
updated[mask] = image_embd.view(-1, D)

print("img positions replaced:", (updated[mask].mean(dim=-1) == 99).all())
print("non-img untouched:", torch.allclose(updated[~mask], token_embd[~mask]))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
img positions replaced: True
non-img untouched: True
```

中文: 两行输出证明替换是精确的——图像位置被覆盖,非图像位置完全没动。

English: Both lines confirm the splice is exact — image positions overwritten, non-image positions untouched.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LLaVA / LLaVA-1.5** / **LLaVA 系列**: `build_one_instance` 在 tokenize 阶段插入 `<image>` 占位符,然后 `prepare_inputs_labels_for_multimodal` 用 `torch.where` 做同样的替换。nanoVLM 是同一思路更干净的 100 行实现。/ `prepare_inputs_labels_for_multimodal` does the same boolean replacement. nanoVLM is the same idea in a cleaner 100-line implementation.
- **Qwen-VL / InternVL** / **Qwen-VL 系列**: 也用占位符 + boolean mask,区别在于他们支持动态分辨率,占位符数量随图像尺寸变化。/ Also uses placeholder + boolean mask, differing only in that they support dynamic resolution so the placeholder count varies.
- **nanoVLM 的 `generate`** / **nanoVLM's `generate` method**: 同样的 `_replace_img_tokens_with_embd` 在推理时也被调用——prefill 阶段用 image embedding 填充,KV cache 建立后后续 decode 步骤里就不再有图像 token 了。/ The same `_replace_img_tokens_with_embd` is called at inference too — image embeddings fill the prefill, then the KV cache takes over and no image tokens appear in subsequent decode steps.

## 注意事项 / Caveats / when it breaks

- **mask True 数必须等于图像 token 总数** / **Mask True count must equal total image tokens**: `image_embd.view(-1, D)` 的行数和 mask 里 True 的个数必须精确匹配。如果 `input_ids` 里的占位符数和实际图像 token 数不一致,赋值会 shape error 或静默截断。/ The number of rows in `image_embd.view(-1, D)` must exactly equal the count of True in the mask. A mismatch causes a shape error or silent truncation.
- **`token_embd.clone()` 是必须的** / **`token_embd.clone()` is required**: 直接在原 `token_embd` 上就地修改会导致 autograd 图被破坏。`clone()` 产生一个新张量,梯度可以正确回流到 `token_embedding` 参数。/ In-place modification on the original `token_embd` would corrupt the autograd graph. `clone()` creates a new tensor so gradients can correctly flow back through `token_embedding`.
- **批次内图像数量变化时** / **When image count varies across the batch**: 如果 batch 里样本 0 有 2 张图、样本 1 有 0 张图,需要确保 `input_ids` 里的占位符数正确,且 `image_embd` 里只包含真实图像的 embedding(不包含 padding)。nanoVLM 的 `_process_images` 处理了这种情况。/ If sample 0 has 2 images and sample 1 has none, the placeholder counts in `input_ids` must be correct, and `image_embd` must contain only real image embeddings (no padding). nanoVLM's `_process_images` handles this.

## 延伸阅读 / Further reading

- LLaVA 原论文 (同种思路): [arXiv 2304.08485](https://arxiv.org/abs/2304.08485)
- nanoVLM 项目: [GitHub huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)
- 模态投影器 (MP) 实现: `models/modality_projector.py` — 把 ViT 特征映射到 LM 维度的那个模块
