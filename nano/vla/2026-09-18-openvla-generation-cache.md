---
date: 2026-09-18
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/models/vlms/prismatic.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/models/vlms/prismatic.py#L488-L591
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, kv-cache, multimodal-generation]
build_role: inference-loop advanced variant
---

# OpenVLA generation cache：第一步看图，后面只看新 token / OpenVLA Generation Cache: See the Image Once, Then Read One New Token

> **一句话 / In one line**: OpenVLA 在第一次 generation step 注入图像 embedding，后续 step 只传最后一个 token 和 `past_key_values`，把多模态输入变成低延迟的自回归循环。 / OpenVLA injects image embeddings on the first generation step, then passes only the last token plus `past_key_values`, turning multimodal input into a low-latency autoregressive loop.

## 为什么重要 / Why this matters

VLA 的控制循环通常只有几十毫秒预算。每次生成动作 token 都重新编码图像和完整 prompt，会把视觉 backbone 和 prefix attention 的成本重复支付。OpenVLA 的 generation adapter 把 Hugging Face generation contract 接到自己的 multimodal `forward` 上，正好处理这个边界。

A VLA control loop may have only a few tens of milliseconds. Re-encoding the image and full prompt for every action token would repeatedly pay the vision and prefix-attention cost. OpenVLA's generation adapter bridges the Hugging Face generation contract to its multimodal `forward` and handles that boundary.

这段代码有两个层次：`prepare_inputs_for_generation` 决定什么时候使用 `inputs_embeds`、什么时候只保留最后一个 `input_id`；`generate_batch` 则负责把图像、文本、mixed precision 和可选的 token probability 组合起来。

The code has two layers. `prepare_inputs_for_generation` decides when to use `inputs_embeds` and when to keep only the last `input_id`; `generate_batch` wraps image/text preparation, mixed precision, and optional token-probability extraction.

## 代码 / The code

`openvla/openvla` — [`prismatic/models/vlms/prismatic.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/models/vlms/prismatic.py#L488-L591)

```python
def prepare_inputs_for_generation(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    **kwargs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
    if past_key_values:
        input_ids = input_ids[:, -1:]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
    )

    return model_inputs


@torch.inference_mode()
def generate_batch(
    self,
    pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
    texts: List[str],
    return_string_probabilities: Optional[List[str]] = None,
    **kwargs: str,
) -> Union[List[str], List[List[float]]]:
    tokenizer = self.llm_backbone.tokenizer

    batch_input_ids = [
        tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
    ]
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to(self.device)
    elif isinstance(pixel_values, dict):
        pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
    else:
        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    gen_texts, gen_probabilities = [], []

    autocast_dtype = self.llm_backbone.half_precision_dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
        for idx, input_ids in enumerate(batch_input_ids):
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values[idx]
            elif isinstance(pixel_values, dict):
                pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

            if return_string_probabilities is None:
                full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                gen_ids = full_out_ids[0, input_ids.shape[1] :]
                gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            else:
                full_out_dict = super().generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **kwargs,
                )
                gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]
                gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)
                slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                string_probs_unnormalized = token_probs[slice_idxs]
                string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                gen_probabilities.append(string_probs.cpu().numpy().tolist())

    return gen_texts if return_string_probabilities is None else gen_probabilities
```

## 逐行讲解 / What's happening

1. **第 499-500 行 / Lines 499-500 (cache-aware truncation)**:
   - 中文: 一旦已经有 `past_key_values`，完整 prefix 不再需要，`input_ids[:, -1:]` 只保留最新 token。
   - English: Once `past_key_values` exists, the full prefix is unnecessary, so `input_ids[:, -1:]` keeps only the newest token.
2. **第 502-506 行 / Lines 502-506 (first-step embeddings)**:
   - 中文: 第一次调用可以直接传 multimodal `inputs_embeds`；后续调用回到 token ids，避免每步重新拼图像 embedding。
   - English: The first call can pass multimodal `inputs_embeds`; later calls return to token ids and avoid rebuilding image embeddings every step.
3. **第 508-515 行 / Lines 508-515 (preserving image values)**:
   - 中文: 即使 `input_ids` 被截断，`pixel_values` 仍被保留在 model inputs 里，让自定义 `forward` 能在首步完成视觉融合。
   - English: Even when `input_ids` is shortened, `pixel_values` stays in the model inputs so the custom `forward` can perform visual fusion on the first step.
4. **第 531-540 行 / Lines 531-540 (input normalization)**:
   - 中文: 文本被逐条 tokenize，图像支持单 tensor 或多相机 dict；两种形式都加 batch 维。
   - English: Text is tokenized per sample, while images support either one tensor or a multi-camera dict; both forms receive a batch dimension.
5. **第 545-559 行 / Lines 545-559 (delegating generation)**:
   - 中文: adapter 不自己写 token loop，而是交给 `GenerationMixin`，后者会反复回调这个类的 `forward` 和 cache adapter。
   - English: The adapter does not reinvent the token loop; it delegates to `GenerationMixin`, which repeatedly calls this model's `forward` and cache adapter.
6. **第 564-589 行 / Lines 564-589 (probability probe)**:
   - 中文: 需要二分类或 stop token 置信度时，代码只取首个生成位置的 logits，再在感兴趣的 token 子集上重新归一化。
   - English: For binary or stop-token confidence, it reads the first generated position's logits and renormalizes only over the requested token subset.

## 类比 / The analogy

这像机器人第一次到仓库时拿一张完整地图，之后每走一步只报告“我现在在这里，下一步往哪走”。如果每一步都重新摊开整张地图，控制循环会被准备工作拖慢；KV cache 就是把已经读过的地图折好放在手边。

It is like a robot unfolding a full warehouse map once, then reporting only its current location and next move. Reopening the entire map every step would slow the controller; the KV cache is the folded map kept nearby.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `inference-loop` 的模型适配层，位于 observation encoder / VLM backbone 和 action tokenizer 或 continuous action head 之间。上游把当前相机帧、语言和 state 做成 prefix；这里首步运行完整 multimodal prefill，后续步消费 `past_key_values` 和最新 token；下游把生成 token 交给 `decode_action`，或把最后 hidden state 交给 action head。省掉它，nanoVLA 仍能生成，但每个控制 tick 都会重新算视觉 prefix，延迟会随 action horizon 线性膨胀。生产实现还要加入多 batch KV cache、显式 cache 生命周期、观测时间戳对齐、GPU stream 和 action chunk 调度。

English: This is the `inference-loop` adapter between the observation encoder/VLM backbone and an action tokenizer or continuous action head. The upstream side builds a prefix from camera frames, language, and state; the first step performs multimodal prefill, later steps consume `past_key_values` and the newest token; downstream code decodes action tokens or feeds the final hidden state to an action head. Without it, nanoVLA still works but recomputes the visual prefix every control tick, making latency grow with the action horizon. Production code adds batched KV-cache ownership, explicit cache lifetime, timestamp alignment, CUDA streams, and action-chunk scheduling.

## 自己跑一遍 / Try it yourself

```python
def prepare(input_ids, embeds, cache):
    if cache:
        input_ids = input_ids[-1:]
    model_inputs = {"input_ids": input_ids} if cache or embeds is None else {"inputs_embeds": embeds}
    model_inputs["past_key_values"] = cache
    return model_inputs

prefix = prepare([10, 11, 12], [0.1, 0.2, 0.3], None)
decode = prepare([10, 11, 12, 13], None, ["cached-prefix"])
print(prefix, decode)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'inputs_embeds': [0.1, 0.2, 0.3], 'past_key_values': None} {'input_ids': [13], 'past_key_values': ['cached-prefix']}
```

中文: 第一次调用携带完整 embedding，第二次调用只携带最后 token；这就是 prefix prefill 与 cached decode 的分界。  
English: The first call carries the full embedding sequence, while the second carries one token; that is the boundary between prefix prefill and cached decode.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **nanoVLM generate** / **nanoVLM generate**: 中文: 同样把图像和 prompt 放进第一次 forward，后续只用 KV cache 解码。 / English: It also performs image/prompt prefill once and decodes later tokens from the KV cache.
- **openpi policy server** / **openpi policy server**: 中文: 把模型推理封装成 observation-to-action 服务，但 cache 由 action sampler 和部署协议共同管理。 / English: It wraps inference as an observation-to-action service, with cache ownership shared by the sampler and deployment protocol.
- **OpenVLA-OFT** / **OpenVLA-OFT**: 中文: 通过并行 action positions 减少自回归次数，是同一个低延迟目标的非自回归路线。 / English: It targets the same latency problem through parallel action positions rather than autoregressive decoding.

## 注意事项 / Caveats / when it breaks

- **cache 不能跨 episode 误用** / **Do not leak cache across episodes**: 中文: 新图像、新任务或机器人 reset 后必须清空旧 cache，否则动作会混入上一段观察。 / English: New images, tasks, or robot resets must clear the old cache or actions will mix with stale observations.
- **pixel_values 的 batch 语义要稳定** / **Keep pixel batch semantics stable**: 中文: tensor 和 dict 两种输入都要明确每个维度对应 batch、camera 还是 channel。 / English: Tensor and dict inputs need explicit conventions for batch, camera, and channel dimensions.
- **第一步和后续步不可混用** / **Do not mix first-step and decode-step contracts**: 中文: `inputs_embeds` 与 `input_ids` 的切换必须和 `past_key_values` 是否为空一致。 / English: The switch between `inputs_embeds` and `input_ids` must agree with whether `past_key_values` is present.

## 延伸阅读 / Further reading

- [OpenVLA generation adapter](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/models/vlms/prismatic.py#L488-L591)
- [Transformers generation with cache](https://huggingface.co/docs/transformers/main/en/cache_explanation)
- [OpenVLA action prediction](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/extern/hf/modeling_prismatic.py#L492-L536)
