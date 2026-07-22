---
date: 2026-07-20
topic: diffusion
source: trending
repo: microsoft/World-R1
file: reward_server/reward_3d.py
permalink: https://github.com/microsoft/World-R1/blob/cf54603d4df179545bbf6ce292eadb0a7580697d/reward_server/reward_3d.py#L111-L210
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-r1, reward-model, logits]
---

# World-R1 reward：不生成文字，直接读 0-9 的 logits / World-R1 Reward: Read 0-9 Logits Instead of Generating Text

> **一句话 / In one line**: World-R1 的 3D reward worker 把视频/点图交给 Qwen3-VL，但评分时直接取最后位置上数字 token 的 logits，再算 0-9 的期望分。 / World-R1's 3D reward worker feeds video or pointmap evidence to Qwen3-VL, then scores by reading digit-token logits at the last position and computing the expected 0-9 score.

## 为什么重要 / Why this matters

用 VLM 当 reward model 时，最慢也最不稳定的部分常是“让模型生成一句评分”。World-R1 直接读 logits：不用采样、不用解析长文本，分数还是可微风格的 soft preference。这种技巧很适合视频扩散/RL 微调里的大批量 reward 评估。

When a VLM acts as a reward model, the slow and unstable part is often asking it to generate a rating sentence. World-R1 reads logits directly: no sampling, no long-text parsing, while still getting a soft preference score. This is useful for large-batch reward evaluation in video diffusion or RL fine-tuning.

## 代码 / The code

`microsoft/World-R1` — [`reward_server/reward_3d.py`](https://github.com/microsoft/World-R1/blob/cf54603d4df179545bbf6ce292eadb0a7580697d/reward_server/reward_3d.py#L111-L210)

```python
# Process
text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info([[message]])
batch_data = self.processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Move inputs to device
input_ids = batch_data['input_ids'].to(self.device)
attention_mask = batch_data['attention_mask'].to(self.device)
pixel_values = batch_data.get('pixel_values')
if pixel_values is not None:
    pixel_values = pixel_values.to(self.device)
pixel_values_videos = batch_data.get('pixel_values_videos')
if pixel_values_videos is not None:
    pixel_values_videos = pixel_values_videos.to(self.device)

cache_position = torch_local.arange(0, input_ids.shape[1], device=self.device)

with torch_local.no_grad():
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        cache_position=cache_position,
    )

logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
last_valid_indices = (attention_mask != 0).cumsum(dim=1).argmax(dim=1)
last_token_logits = logits[torch_local.arange(logits.size(0)), last_valid_indices, :]

# Token IDs: 15='0', 16='1', ..., 24='9'
digit_logits = last_token_logits[:, 15:25]
scores_range = torch_local.arange(0, 10, device=self.device).float()
probs = (digit_logits / self.temperature).softmax(dim=-1)
raw_score = (probs * scores_range).sum(dim=-1).item()
score = raw_score / 9.0
```

## 逐行讲解 / What's happening

1. **先按 chat 模板包装 / Wrap with the chat template**: 中文: prompt、视频帧或点图先被 processor 变成 VLM 输入。 English: the prompt plus frames or pointmap are converted into VLM inputs by the processor.
2. **显式搬到当前 GPU / Move tensors to the worker GPU**: 中文: worker 已经设置 `CUDA_VISIBLE_DEVICES`，所以输入也必须去同一个 device。 English: the worker has set `CUDA_VISIBLE_DEVICES`, so inputs must move to the same device.
3. **只 forward，不 generate / Forward, do not generate**: 中文: `no_grad` 下跑模型，拿 logits。 English: the model runs under `no_grad` and returns logits.
4. **找最后有效 token / Find the last valid token**: 中文: padding 后不能直接用最后一列，要用 `attention_mask` 找真实末尾。 English: after padding, the final column may be padding, so `attention_mask` finds the real end.
5. **切数字 token / Slice digit tokens**: 中文: 代码假设 token id 15 到 24 对应 0 到 9。 English: the code assumes token ids 15 through 24 correspond to digits 0 through 9.
6. **期望分归一化 / Normalize expected score**: 中文: softmax 后对 `0..9` 求期望，再除以 9 变成 `0..1` reward。 English: after softmax, it computes the expectation over `0..9` and divides by 9 to get a `0..1` reward.

## 类比 / The analogy

像让评委按 0 到 9 的按钮打分，而不是先写一段评语再从评语里猜分数。按钮每个都有概率，最后取加权平均。

It is like asking a judge to press one of the 0-to-9 buttons instead of writing a paragraph and later guessing the score from the paragraph. Each button has a probability, and the final score is the weighted average.

## 自己跑一遍 / Try it yourself

```python
import math
logits = [0.0, 0.2, 0.1, 0.0, 0.3, 1.5, 0.4, 0.2, 0.1, 0.0]
exp = [math.exp(x) for x in logits]
probs = [x / sum(exp) for x in exp]
raw = sum(i * p for i, p in enumerate(probs))
print(round(raw, 3))
print(round(raw / 9.0, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
4.646
0.516
```

logit 最大的数字会拉高分数，但其他数字的概率仍会参与期望。

The digit with the largest logit pulls the score upward, but all other digit probabilities still contribute to the expectation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Reward model preference heads** / **Reward model preference heads**: 直接读 scalar/logit，避免生成文本后处理。 / Read scalar or logits directly instead of post-processing generated text.
- **No-repeat ngram logits processor** / **No-repeat ngram logits processor**: 也是在 logits 层面控制输出，而不是等文本生成完再修。 / It also controls behavior at the logits level rather than fixing text afterward.

## 注意事项 / Caveats / when it breaks

- **token id 假设要校验 / Token-id assumptions must be verified**: 不同 tokenizer 未必让 15-24 对应数字。 / Another tokenizer may not map ids 15-24 to digits.
- **最后有效位置容易算错 / Last valid position is easy to get wrong**: padding 侧不同会改变索引逻辑。 / Different padding sides can change the index logic.
- **VLM reward 会继承 VLM 偏见 / VLM reward inherits VLM bias**: 它评的是模型眼里的 3D 质量，不是真实物理指标。 / It measures what the VLM perceives as 3D quality, not a physical metric.

## 延伸阅读 / Further reading

- [World-R1 reward_3d.py](https://github.com/microsoft/World-R1/blob/cf54603d4df179545bbf6ce292eadb0a7580697d/reward_server/reward_3d.py#L111-L210)
- [World-R1 repository](https://github.com/microsoft/World-R1)
