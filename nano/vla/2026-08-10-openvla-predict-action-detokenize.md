---
date: 2026-08-10
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/extern/hf/modeling_prismatic.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/extern/hf/modeling_prismatic.py#L492-L536
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-detokenization]
build_role: inference-loop advanced variant, token-to-continuous-action decoding
---

# OpenVLA action prediction：token 要还原成连续动作 / OpenVLA Action Prediction: Tokens Must Become Continuous Actions

> **一句话 / In one line**: `predict_action()` 把语言模型生成的动作 token 反查到动作分箱，再按数据集统计量反归一化成机器人动作。 / `predict_action()` turns generated action tokens into action-bin centers, then unnormalizes them with dataset statistics.

## 为什么重要 / Why this matters

离散动作 tokenizer 让 VLA 可以像生成文字一样生成动作，但机器人最后需要的是连续控制量。OpenVLA 的这个 wrapper 连接了两个世界：前半段调用 `generate()`，后半段把 token ids 解码成可发送给机器人的数值动作。

A discrete action tokenizer lets a VLA generate actions like text, but the robot eventually needs continuous controls. This wrapper bridges the two worlds: call `generate()`, decode token ids into numeric actions, then unnormalize them.

## 代码 / The code

`openvla/openvla` — [`prismatic/extern/hf/modeling_prismatic.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/extern/hf/modeling_prismatic.py#L492-L536)

```python
class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions
```

## 逐行讲解 / What's happening

1. **第 499-504 行 / Lines 499-504 (decode table)**:
   - 中文: 动作空间先被切成 `[-1, 1]` 上的分箱中心，同时记录真实文本 vocab size。
   - English: The action range is divided into bin centers over `[-1, 1]`, while the true text vocab size is recorded.
2. **第 512-518 行 / Lines 512-518 (prompt and generation)**:
   - 中文: 特殊空格 token 保持和训练时 prompt 形式一致，然后只生成 `action_dim` 个新 token。
   - English: A special space token keeps the prompt aligned with training, then exactly `action_dim` new tokens are generated.
3. **第 520-524 行 / Lines 520-524 (token to normalized action)**:
   - 中文: token id 通过 `vocab_size - id` 映射回动作分箱，再 `clip` 到合法范围。
   - English: Token ids map back to action bins through `vocab_size - id`, then get clipped to legal bins.
4. **第 526-536 行 / Lines 526-536 (unnormalize)**:
   - 中文: 被 mask 管理的维度按 `q01/q99` 还原到数据集动作范围，没 mask 的维度保留 normalized 值。
   - English: Masked dimensions are unnormalized with `q01/q99`; unmasked dimensions keep their normalized value.

## 类比 / The analogy

像把邮编翻译成街道地址：模型输出的是紧凑编号，投递前必须查表并换算到真实坐标。

It is like translating a zip code into a street address: the model emits compact IDs, but execution needs real coordinates.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这是 `inference-loop` 的最后一段：`prompt + image -> LM generate -> action ids -> normalized action -> robot action`。上游是 VLM/backbone 和 action tokenizer，下游是 action queue、插值器或机器人驱动。如果省掉反归一化，机器人收到的只是 `[-1, 1]` 范围里的抽象动作，不一定匹配真实关节/末端执行器尺度。

In a nanoVLA, this is the final part of the `inference-loop`: `prompt + image -> LM generate -> action ids -> normalized action -> robot action`. Upstream are the VLM backbone and action tokenizer; downstream are the action queue, interpolator, or robot driver.

## 自己跑一遍 / Try it yourself

```python
vocab_size = 100
bin_centers = [-0.75, -0.25, 0.25, 0.75]
token_ids = [99, 97, 96]
q01 = [0.0, -2.0, 10.0]
q99 = [1.0, 2.0, 20.0]
bins = [max(0, min(vocab_size - tid - 1, len(bin_centers) - 1)) for tid in token_ids]
norm = [bin_centers[i] for i in bins]
actions = [0.5 * (x + 1) * (hi - lo) + lo for x, lo, hi in zip(norm, q01, q99)]
print(bins)
print([round(x, 3) for x in actions])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0, 2, 3]
[0.125, 0.5, 18.75]
```

注意 token id 越接近 vocab 尾部，映射出来的动作分箱越靠前；这就是 OpenVLA action-tokenizer 的约定。

Notice that token ids near the end of the vocabulary map to earlier action bins; that is OpenVLA's action-tokenizer convention.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot ACT/Diffusion policies** / **LeRobot ACT/Diffusion policies**: 模型输出也要经过 processor/normalizer 才能落到机器人动作空间。 / Model outputs also pass through processors and normalizers before reaching robot action space.
- **openpi continuous heads** / **openpi continuous heads**: 不走离散 token，但同样需要把网络输出换算成物理动作。 / They skip discrete tokens, but still convert network outputs into physical actions.

## 注意事项 / Caveats / when it breaks

- **`unnorm_key` 必须匹配数据集** / **`unnorm_key` must match the dataset**: 多数据集训练时，用错统计量会让动作尺度错位。 / In multi-dataset training, the wrong stats produce the wrong action scale.
- **分箱有量化误差** / **Bins introduce quantization**: 动作越精细，需要越多 bin 或改用连续 head。 / Finer control needs more bins or a continuous head.

## 延伸阅读 / Further reading

- [OpenVLA `predict_action`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/extern/hf/modeling_prismatic.py#L492-L536)
