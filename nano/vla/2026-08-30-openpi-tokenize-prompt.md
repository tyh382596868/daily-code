---
date: 2026-08-30
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L247-L266
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer, prompt-tokenization]
build_role: action-tokenizer advanced variant
---

# openpi TokenizePrompt：语言条件先变成模型前缀 / openpi TokenizePrompt: Turn Language Conditions into Model Prefixes

> **一句话 / In one line**: `TokenizePrompt` 把 `prompt` 从数据字典里取出，必要时连同离散状态一起 token 化，再写回 token 和 mask。 / `TokenizePrompt` removes `prompt` from the data dict, optionally tokenizes it with discrete state, then writes tokens and masks back.

## 为什么重要 / Why this matters

VLA 不是直接把自然语言字符串喂给动作 head。语言需要先被 tokenizer 变成前缀 token，mask 还要告诉模型哪些位置有效。openpi 把这个动作写成一个小 transform，方便和图像、状态、动作处理链组合。

A VLA does not feed raw natural-language strings into an action head. Language must become prefix tokens, and masks must tell the model which positions are valid. openpi packages this as a small transform, making it easy to compose with image, state, and action transforms.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L247-L266)

```python
@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
```

## 逐行讲解 / What's happening

1. **第 247-250 行 / Lines 247-250**:
   - 中文: transform 保存一个 `PaligemmaTokenizer`，并用 `discrete_state_input` 决定语言 tokenization 是否还要吃状态。
   - English: The transform owns a `PaligemmaTokenizer` and uses `discrete_state_input` to decide whether tokenization also consumes state.
2. **第 252-254 行 / Lines 252-254**:
   - 中文: `prompt` 被 `pop` 出来；如果缺失就立即报错，因为后面模型前缀无法构造。
   - English: `prompt` is popped out; if it is missing, the transform fails immediately because the model prefix cannot be built.
3. **第 256-260 行 / Lines 256-260**:
   - 中文: 离散状态模式下，状态是 tokenization 的输入之一；普通模式则显式传 `None`。
   - English: In discrete-state mode, state becomes part of tokenization; otherwise `None` is passed explicitly.
4. **第 262-266 行 / Lines 262-266**:
   - 中文: 非 Python 字符串会用 `.item()` 取出标量值，最后返回 token 和 token mask。
   - English: Non-Python string scalars are unwrapped with `.item()`, and the transform returns tokens plus token masks.

## 类比 / The analogy

这像进实验室前换证件：自然语言任务是身份证，机器人控制台不直接读身份证，而是把它换成门禁卡和权限表。

It is like exchanging an ID card before entering a lab: the natural-language task is the ID, but the robot console reads an access card and permission table instead.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这属于 `action-tokenizer` 的高级变体，也连接到 `vlm-backbone-wiring`。上游是 dataset 或 runtime 给出的 `prompt` 和可选 `state`，下游是 VLM prefix embedding。省掉这层，你的 action head 仍可预测动作，但它拿不到稳定的任务语言条件。

In a nanoVLA, this is an advanced variant of `action-tokenizer` and also connects to `vlm-backbone-wiring`. Upstream is the `prompt` and optional `state` from the dataset or runtime; downstream is the VLM prefix embedding. Without this layer, the action head may still predict actions, but it will not receive a stable task-language condition.

## 自己跑一遍 / Try it yourself

```python
class ToyTokenizer:
    def tokenize(self, prompt, state=None):
        words = prompt.lower().split()
        if state is not None:
            words += [f"s={state}"]
        return words, [1] * len(words)

def tokenize_prompt(data, discrete=False):
    prompt = data.pop("prompt", None)
    if prompt is None:
        raise ValueError("Prompt is required")
    state = data.get("state") if discrete else None
    tokens, mask = ToyTokenizer().tokenize(prompt, state)
    return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": mask}

print(tokenize_prompt({"prompt": "Pick Cup", "state": 7}, discrete=True))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'state': 7, 'tokenized_prompt': ['pick', 'cup', 's=7'], 'tokenized_prompt_mask': [1, 1, 1]}
```

这个最小版本展示了两个约定：原始 `prompt` 被消费掉，模型只看到 token 和 mask。

This minimal version shows two contracts: the raw `prompt` is consumed, and the model sees only tokens and masks.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi FAST tokenization** / **openpi FAST tokenization**: 把 prompt、state、actions 一起变成 token、AR mask 和 loss mask。 / It turns prompt, state, and actions into tokens, AR masks, and loss masks.
- **LeRobot SmolVLA prefix embedding** / **LeRobot SmolVLA prefix embedding**: 语言、图像和状态一起组成前缀 token。 / Language, image, and state form the prefix tokens together.
- **OpenVLA action tokenizer** / **OpenVLA action tokenizer**: 下游动作可以离散化，但上游语言也必须先稳定 token 化。 / Downstream actions may be discretized, but upstream language must also be tokenized reliably.

## 注意事项 / Caveats / when it breaks

- **`pop` 会改变输入字典** / **`pop` mutates the input dict**: 如果别的 transform 还要原始 `prompt`，顺序就必须调整。 / If another transform still needs raw `prompt`, ordering must be changed.
- **`.item()` 假设标量** / **`.item()` assumes a scalar**: batch 里如果是列表或数组，这里需要更明确的批处理逻辑。 / If a batch contains lists or arrays, this needs explicit batch handling.
- **mask 和 token 必须同长** / **Masks must match tokens**: 下游 attention 或 padding 会依赖这个长度契约。 / Downstream attention or padding relies on this length contract.

## 延伸阅读 / Further reading

- openpi transforms: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py
- PaliGemma overview: https://ai.google.dev/gemma/docs/paligemma
