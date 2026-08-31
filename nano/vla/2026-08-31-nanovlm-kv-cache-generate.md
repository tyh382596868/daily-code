---
date: 2026-08-31
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_language_model.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L82-L143
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, kv-cache, multimodal-prefill]
build_role: inference-loop advanced variant
---

# nanoVLM generate：多模态先 prefill，后面只解一个 token / nanoVLM Generate: Multimodal Prefill, Then Decode One Token at a Time

> **一句话 / In one line**: 图像和文本先合成一段 embedding 做 prefill，之后每步只把新 token 和 KV cache 送回 decoder。 / Image and text are fused into one embedding prefix for prefill; after that, each step sends only the new token plus KV cache back to the decoder.

## 为什么重要 / Why this matters

VLA 的推理 loop 也有同样结构：观测和语言指令是前缀，动作 token 或动作 latent 是逐步生成的后缀。如果每一步都重算图像和 prompt，延迟会爆。nanoVLM 的 `generate` 展示了最小可懂版本的前缀缓存推理。

A VLA inference loop has the same shape: observations and language form the prefix; action tokens or action latents are generated afterward. Recomputing images and prompts every step would destroy latency. nanoVLM's `generate` is a compact version of prefix-cached inference.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_language_model.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L82-L143)

```python
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
        start_pos=0
    )

    last_token_output_from_prefill = prefill_output[:, -1, :]
    current_logits = self.decoder.head(last_token_output_from_prefill)
    newly_generated_ids_list = []

    for _ in range(max_new_tokens):
        if greedy:
            next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
        else:
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
            start_pos=current_token_start_pos
        )
```

## 逐行讲解 / What's happening

1. **第 84-92 行 / Lines 84-92 (multimodal prefix)**:
   - 中文: 图像先过 vision encoder 和 projector，再替换 prompt 里的 image token 占位符。
   - English: Images pass through the vision encoder and projector, then replace image-token placeholders in the prompt.
2. **第 98-103 行 / Lines 98-103 (prefill)**:
   - 中文: 第一次 decoder 调用吃完整前缀，并产出 `kv_cache_list`。
   - English: The first decoder call consumes the whole prefix and returns `kv_cache_list`.
3. **第 116-123 行 / Lines 116-123 (sample next token)**:
   - 中文: 每一步只根据当前 logits 采样一个新 token。
   - English: Each step samples exactly one new token from the current logits.
4. **第 137-143 行 / Lines 137-143 (cached decode)**:
   - 中文: 后续 decoder 只吃新 token embedding，旧上下文靠 cache 补齐。
   - English: Later decoder calls consume only the new token embedding; prior context comes from the cache.

## 类比 / The analogy

像开会先读完整背景材料，之后每轮发言只补一句新信息。没人会每说一句话都从第一页重新念一遍。

It is like reading the full briefing once at the start of a meeting. After that, each turn adds one new sentence; nobody rereads the whole packet before speaking.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `inference-loop` 的 advanced variant。你的 nanoVLA 可以把相机图像、语言、机器人状态作为 prefill prefix，把动作查询或动作 token 作为 decode 后缀。上游是 observation encoder 和 tokenizer，下游是 action detokenizer 或连续动作 head。生产版还要加 action chunk 缓存、实时控制频率、超时保护和失败回退。

English: This is an advanced variant of the `inference-loop` curriculum item. In a nanoVLA, camera frames, language, and robot state can form the prefill prefix, while action queries or action tokens are decoded afterward. Upstream are the observation encoder and tokenizer; downstream is the action detokenizer or continuous action head. A production version adds action-chunk caching, real-time control cadence, timeout handling, and fallback behavior.

## 自己跑一遍 / Try it yourself

```python
cache = []
prefix = ["image", "instruction", "state"]

def decoder(tokens, kv_cache=None):
    kv_cache = list(kv_cache or [])
    kv_cache.extend(tokens)
    return f"logits_after_{tokens[-1]}", kv_cache

logits, cache = decoder(prefix)
for _ in range(3):
    next_token = "action"
    logits, cache = decoder([next_token], cache)
    print(logits, cache)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
three decode steps, with the cache growing each time
```

中文: 注意第二步以后 decoder 的输入只有一个 token，但 cache 保留了完整历史。

English: Notice that after the first step, the decoder input is one token while the cache carries the full history.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi sample_actions** / **openpi sample_actions**: 中文: 前缀编码一次，动作流匹配过程反复使用。 / English: The prefix is encoded once and reused during action flow matching.
- **LeRobot policy server** / **LeRobot policy server**: 中文: 部署时也会把观测处理和动作发送拆成不同阶段。 / English: Deployment also separates observation processing from action dispatch.

## 注意事项 / Caveats / when it breaks

- **cache 位置必须对齐 / Cache positions must align**: 中文: `start_pos` 错一位会让 RoPE 或位置编码串掉。 / English: A wrong `start_pos` can corrupt RoPE or positional encoding.
- **动作不是普通文本 / Actions are not ordinary text**: 中文: VLA 后缀常常需要连续 head 或专用 detokenizer。 / English: VLA suffixes often need a continuous head or a dedicated detokenizer.

## 延伸阅读 / Further reading

- nanoVLM generate source: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_language_model.py#L82-L143
- nanoVLA inference loop index: ../../topics/vla.md
