---
date: 2026-07-14
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/vision_language_model.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L31-L183
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, nanovlm, generation, kv-cache]
---

# nanoVLM generate：多模态预填一次，后面只解码新 token / nanoVLM generate: Multimodal Prefill Once, Then Decode New Tokens

> **一句话 / In one line**: nanoVLM 先把图像 token 替换进文本 embedding，做一次 multimodal prefill，然后用 KV cache 逐 token 解码。 / nanoVLM first replaces image placeholders with image embeddings, runs one multimodal prefill, then decodes token by token with a KV cache.

## 为什么重要 / Why this matters

VLM 推理慢的部分不是最后采样一个 token，而是“把图像和整段 prompt 一起过一遍”。这段代码把昂贵的图像编码和 prompt prefill 放在循环外，循环里只送入新生成的一个 token。小模型也用生产模型同样的推理形状。

The slow part of VLM inference is not sampling one token; it is running the image and whole prompt together. This code keeps image encoding and prompt prefill outside the loop, then feeds only the newly generated token inside the loop. A small model uses the same inference shape as production systems.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_language_model.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L31-L183)

```python
def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
    updated_token_embd = token_embd.clone()
    mask = (input_ids == self.tokenizer.image_token_id)
    updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)
    return updated_token_embd

@torch.inference_mode()
def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
    images_tensor = self._process_images(images, input_ids.device)
    token_embd = self.decoder.token_embedding(input_ids)

    if images_tensor is not None:
        image_embd = self.vision_encoder(images_tensor)
        image_embd = self.MP(image_embd)
        token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

    current_total_seq_len = token_embd.size(1)
    batch_size = input_ids.size(0)

    prefill_output, kv_cache_list = self.decoder(
        token_embd,
        attention_mask=attention_mask,
        kv_cache=None,
        start_pos=0,
    )
    current_logits = self.decoder.head(prefill_output[:, -1, :])

    newly_generated_ids_list = []
    for _ in range(max_new_tokens):
        filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filtered_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        newly_generated_ids_list.append(next_token_id)

        next_token_embed = self.decoder.token_embedding(next_token_id)
        current_token_start_pos = current_total_seq_len
        current_total_seq_len += 1

        if attention_mask is not None:
            attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

        decode_step_output, kv_cache_list = self.decoder(
            next_token_embed,
            attention_mask=attention_mask,
            kv_cache=kv_cache_list,
            start_pos=current_token_start_pos,
        )
        current_logits = self.decoder.head(decode_step_output[:, -1, :])
```

## 逐行讲解 / What's happening

1. **image placeholder 替换 / Image placeholder replacement**:
   - 中文: `input_ids == image_token_id` 找到图像占位符位置，把视觉投影后的 embedding 塞进去。
   - English: `input_ids == image_token_id` finds image placeholders and inserts projected vision embeddings.
2. **prefill 在循环外 / Prefill outside the loop**:
   - 中文: 整段多模态 prompt 只过一次 decoder，同时产出 `kv_cache_list`。
   - English: The full multimodal prompt runs through the decoder once and produces `kv_cache_list`.
3. **只喂新 token / Feed only the new token**:
   - 中文: 采样后只 embed `next_token_id`，不会重复计算旧 prompt。
   - English: After sampling, only `next_token_id` is embedded; the old prompt is not recomputed.
4. **`start_pos` 维护位置 / `start_pos` maintains position**:
   - 中文: cache 里已有多少 token，下一步 RoPE/position 就从哪里继续。
   - English: The next RoPE/position index continues from the number of tokens already cached.

## 类比 / The analogy

像开会先把投影和会议纪要读一遍，之后每个人发言只需要接着上一句话说，不必每次重放整场会议。

It is like reading the slides and meeting notes once; later each speaker continues from the previous sentence instead of replaying the whole meeting.

## 自己跑一遍 / Try it yourself

```python
prompt = ["<img>", "what", "is", "this"]
image_embedding = ["cat_patch"]
prefill = [image_embedding[0] if tok == "<img>" else tok for tok in prompt]
cache = list(prefill)
for token in ["a", "cat"]:
    cache.append(token)
    print("decode", token, "start_pos", len(cache) - 1)
print(cache)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
decode a start_pos 4
decode cat start_pos 5
['cat_patch', 'what', 'is', 'this', 'a', 'cat']
```

图像和 prompt 只进入 cache 一次，后续解码沿着 cache 继续。

The image and prompt enter the cache once; later decoding continues from that cache.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers generation** / **Transformers generation**: 大模型推理同样分 prefill 和 decode 两阶段。 / Large-model inference similarly separates prefill and decode.
- **VLA action decoding** / **VLA action decoding**: 机器人策略也常先预填观测/语言，再逐步解码动作 token。 / Robot policies often prefill observation/language, then decode action tokens step by step.

## 注意事项 / Caveats / when it breaks

- **图像 token 数要匹配 / Image token count must match**: 占位符数量和 `image_embd` 展平后的数量不一致会赋值失败或错位。 / Placeholder count must match flattened `image_embd`, or assignment fails or misaligns.
- **attention mask 要同步增长 / Attention mask must grow together**: 新 token 进入 cache 时，mask 也要追加一列。 / When a new token enters the cache, the mask needs one more column too.

## 延伸阅读 / Further reading

- [nanoVLM `generate`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L31-L183)
