---
date: 2026-10-30
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/vla/action_tokenizer.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py#L29-L68
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer]
build_role: action-tokenizer
---

# OpenVLA 动作 token 的边界处理 / OpenVLA Action-Token Boundaries

> **一句话 / In one line**: 量化时接受两个端点，反量化时把多出来的最高索引收回最后一个 bin。 / Quantization accepts both endpoints; dequantization folds the extra top index back into the final bin.

## 为什么重要 / Why this matters

连续动作要借用语言模型的离散词表，但端点是最容易悄悄越界的地方。OpenVLA 明确把 `digitize` 的索引范围钉回可用的 bin center。

Continuous actions borrow an LLM’s discrete vocabulary, and endpoints are where silent out-of-range bugs hide. OpenVLA explicitly clamps `digitize` indices back to valid bin centers.

## 代码 / The code

`openvla/openvla` — [`action_tokenizer.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py#L55-L68)

```python
action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
discretized_action = np.digitize(action, self.bins)
return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))

discretized_actions = self.tokenizer.vocab_size - action_token_ids
discretized_actions = np.clip(discretized_actions - 1, a_min=0,
                              a_max=self.bin_centers.shape[0] - 1)
return self.bin_centers[discretized_actions]
```

## 逐行讲解 / What's happening

1. 中文：先裁剪到训练时承诺的动作范围。
   English: First clamp to the action range promised during training.
2. 中文：`digitize` 产生从 1 到 bin 数的索引，token id 从词表尾部向前映射。
   English: `digitize` yields 1-to-bin-count indices, mapped backward from the vocabulary tail.
3. 中文：解码先减一再裁剪，吸收最大端点带来的额外索引。
   English: Decoding subtracts one and clamps, absorbing the extra index introduced at the upper endpoint.

## 类比 / The analogy

像把音量旋钮标成 0 到 10：旋钮拧到最右仍属于第 10 格，而不是凭空出现第 11 格。

It is like a volume knob labeled 0 through 10: the far-right stop belongs to the last notch, not an imaginary eleventh one.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文：这是 nanoVLA 的 `action-tokenizer` 输入/输出边界：输入是归一化后的连续动作，输出是模型可预测的 token id；反向路径把 token 还原为控制量。上游必须先统一动作单位，下游必须加上机器人限幅。省略端点约束会让极值动作偶发地解码失败。

English: This is nanoVLA’s `action-tokenizer` boundary: normalized continuous actions enter and model-predictable token ids leave; the reverse path recovers controls. Upstream must standardize action units and downstream must impose robot safety limits. Without endpoint handling, extreme actions can intermittently fail to decode.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
bins = np.linspace(-1, 1, 4)
centers = (bins[:-1] + bins[1:]) / 2
x = np.array([-2., -1., 0., 1., 2.])
i = np.digitize(np.clip(x, -1, 1), bins)
print(centers[np.clip(i - 1, 0, len(centers) - 1)])
```

运行 / Run with: `pip install numpy && python try.py`

预期输出 / Expected output: 所有值都落入三个合法中心之一。 / Every value lands on one of three legal centers.

中文：这个极小版本故意展示 `+1` 端点。 English: This tiny version deliberately exercises the `+1` endpoint.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **量化器边界 / Quantizer boundaries**: 连续控制和图像量化都需要把理论端点变成有限码本。 / Continuous control and image quantization both turn mathematical endpoints into a finite codebook.

## 注意事项 / Caveats / when it breaks

- 中文：词表尾部 token 必须确实是可安全复用的 token。 / English: The vocabulary tail must truly be safe to repurpose.

## 延伸阅读 / Further reading

- [OpenVLA action tokenizer](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py#L29-L68)
