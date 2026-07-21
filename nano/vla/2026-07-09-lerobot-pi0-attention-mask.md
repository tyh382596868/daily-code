---
date: 2026-07-09
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/pi0/modeling_pi0.py
permalink: https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/pi0/modeling_pi0.py#L101-L137
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, attention-mask, prefix-lm]
build_role: vlm-backbone-wiring advanced variant
---

# LeRobot pi0 attention mask：用 cumulative mask 表达 prefix-LM / LeRobot pi0 Attention Mask: Prefix-LM from a Cumulative Mask

> **一句话 / In one line**: `make_att_2d_masks` 把一维 block 边界和 padding mask 扩成二维 attention mask，让 prefix token 双向可见、action token 因果可见。 / `make_att_2d_masks` expands one-dimensional block boundaries and padding into a 2D attention mask, giving prefix tokens bidirectional visibility and action tokens causal visibility.

## 为什么重要 / Why this matters

VLA 的 token 流通常混着图像、语言、状态和动作。它不是纯 causal LM，也不是纯 bidirectional encoder。pi0 用一个 `att_masks` 序列标出“哪些 token 开始一个新的因果段”，再用 cumulative sum 生成完整二维 mask。

A VLA token stream mixes images, language, state, and actions. It is neither a pure causal LM nor a pure bidirectional encoder. pi0 marks which tokens start a new causal segment with `att_masks`, then uses a cumulative sum to build the full 2D mask.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/pi0/modeling_pi0.py`](https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/pi0/modeling_pi0.py#L101-L137)

```python
def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
```

## 逐行讲解 / What's happening

1. **第 104-116 行 / Lines 104-116 (mask examples)**:
   - 中文: 注释直接给了三种模式：纯 causal、prefix-LM、分块 causal。
   - English: The comments show three modes directly: pure causal, prefix-LM, and block-causal.
2. **第 129 行 / Line 129 (`cumsum`)**:
   - 中文: 每遇到一个 `1`，当前位置之后进入新的 attention 段。
   - English: Each `1` advances the current attention segment.
3. **第 130 行 / Line 130 (`<=`)**:
   - 中文: query token 可以看所有 cumulative id 小于等于自己的 key token。
   - English: A query token can attend to key tokens whose cumulative id is less than or equal to its own.
4. **第 131-132 行 / Lines 131-132 (padding)**:
   - 中文: 最后再和二维 padding mask 相与，保证 padding 不参与注意力。
   - English: The final `&` removes padded query/key positions.

## 类比 / The analogy

这像会议记录权限：前面的背景材料所有人都能看，后面的行动记录只能看自己之前已经发生的部分；空白页永远不能引用。

It is like permissions in meeting notes. The background section is visible to everyone, action logs can only see what happened earlier, and blank pages cannot be cited.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `vlm-backbone-wiring` 的 attention policy。上游把图像、语言、状态、动作拼成 token；这个函数决定哪些 token 能互相看见；下游的 transformer 才能在同一个序列里同时支持 prefix 理解和 action 生成。如果省掉它，nanoVLA 只能退化成“全 causal”或“全双向”。

This is the attention policy for `vlm-backbone-wiring`. Upstream code packs image, language, state, and action tokens into one sequence; this function decides which tokens can see each other; the transformer can then support prefix understanding and action generation in one stream. Without it, nanoVLA falls back to all-causal or all-bidirectional behavior.

## 自己跑一遍 / Try it yourself

```python
def mask(att, pad):
    c = []
    total = 0
    for x in att:
        total += x
        c.append(total)
    return [[int(a <= b and pa and pb) for a, pa in zip(c, pad)] for b, pb in zip(c, pad)]

for row in mask([0, 0, 0, 1, 1], [1, 1, 1, 1, 1]):
    print(row)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 1, 1, 0, 0]
[1, 1, 1, 0, 0]
[1, 1, 1, 0, 0]
[1, 1, 1, 1, 0]
[1, 1, 1, 1, 1]
```

前三个 prefix token 彼此可见；后两个 token 开始因果解码。

The first three prefix tokens see each other; the last two tokens decode causally.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi pi0** / **openpi pi0**: LeRobot 注释说明这是从 openpi 复制的同一设计。 / The LeRobot comment says this is the same design copied from openpi.
- **OpenVLA action suffix** / **OpenVLA action suffix**: 也把视觉/语言放前缀，动作放后缀，只是 mask 形式不同。 / It also puts vision/language in the prefix and actions in the suffix, with a different mask implementation.

## 注意事项 / Caveats / when it breaks

- **`att_masks` 不是 padding** / **`att_masks` is not padding**: 它表达段边界；真正 padding 由 `pad_masks` 控制。 / It describes segment boundaries; `pad_masks` controls valid tokens.
- **方向别写反** / **Do not flip the inequality**: `key_cumsum <= query_cumsum` 才是“只能看过去”。 / `key_cumsum <= query_cumsum` is what means "can only see the past."

## 延伸阅读 / Further reading

- [LeRobot `make_att_2d_masks`](https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/pi0/modeling_pi0.py#L101-L137)
