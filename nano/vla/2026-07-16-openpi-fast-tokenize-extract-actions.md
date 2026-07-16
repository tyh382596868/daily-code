---
date: 2026-07-16
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L267-L299
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, openpi, action-tokenizer, fast]
build_role: action-tokenizer advanced variant
---

# openpi FAST transform：训练时 token 化，推理后还原动作 / openpi FAST Transform: Tokenize for Training, Extract Actions After Inference

> **一句话 / In one line**: openpi 把 FAST 的动作 tokenizer 包成两个 transform：输入侧产出 token/mask，输出侧把 token 还原成连续动作。 / openpi wraps the FAST action tokenizer in two transforms: the input side emits tokens/masks, and the output side extracts continuous actions from generated tokens.

## 为什么重要 / Why this matters

FAST 类 VLA 把动作变成语言模型 token。训练时，模型要看到 prompt、state 和 action token，还要知道哪些 token 是自回归、哪些 token 参与 loss。推理后，模型吐出的不是直接控制量，而是一串 token，必须按 horizon 和 action dim 解码回连续动作。

FAST-style VLAs turn actions into language-model tokens. During training, the model needs prompt, state, action tokens, autoregressive masks, and loss masks. After inference, the model output is not a control vector yet; it is a token sequence that must be decoded back into continuous actions using horizon and action dimension.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L267-L299)

```python
@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {**data, "actions": actions}
```

## 逐行讲解 / What's happening

1. **prompt 必填 / Prompt is required**: 中文: FAST token 流从语言任务开始，没有 prompt 就不能构造输入。 English: the FAST token stream starts from the language task, so prompt is mandatory.
2. **state 和 actions 一起 token 化 / State and actions tokenize together**: 中文: 训练时 `actions` 存在，tokenizer 能产出 action 监督 token。 English: during training, `actions` exists and the tokenizer can emit supervised action tokens.
3. **四个输出 / Four outputs**: 中文: token、token mask、AR mask、loss mask 分别服务 padding、自回归和监督区域。 English: tokens, token mask, AR mask, and loss mask separately handle padding, autoregression, and supervised regions.
4. **输出侧 pop token / Output side pops tokens**: 中文: 推理结果暂存在 `"actions"`，但语义其实是 token。 English: inference output is temporarily stored under `"actions"`, but semantically it is token IDs.
5. **用 horizon/dim 解码 / Decode with horizon and dimension**: 中文: token 序列必须知道要还原几步、每步几维。 English: token sequences need horizon and action dimension to become controls.

## 类比 / The analogy

这像把动作写成压缩电报。训练前要把“向右 2 cm，夹爪关闭”编码成电报码，还要标出哪些码要计分；执行前再把电报码翻译回电机命令。

It is like compressing actions into telegram code. Before training, "move right 2 cm, close gripper" becomes telegram tokens with scoring masks; before execution, the telegram is translated back into motor commands.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `action-tokenizer` 的高级变体，位于 dataset transform 和 LM forward 之间，同时也位于 model output 和 robot controller 之间。nanoVLA 如果采用离散动作 token，需要成对实现 `TokenizeActions` 和 `ExtractActions`，否则训练和推理会使用两套不一致的动作语义。

English: This is an advanced `action-tokenizer` component. It sits between dataset transforms and the LM forward, and also between model output and robot control. If a nanoVLA uses discrete action tokens, it needs paired `TokenizeActions` and `ExtractActions`; otherwise training and inference use inconsistent action semantics.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

def tokenize(prompt, state, actions):
    tokens = np.array([len(prompt), int(state[0] * 10), int(actions[0, 0] * 10)])
    return tokens, tokens != -1, np.array([1, 1, 1]), np.array([0, 0, 1])

def extract(tokens, horizon, dim):
    vals = tokens[-horizon * dim:].astype(np.float32) / 10.0
    return vals.reshape(horizon, dim)

tokens, token_mask, ar_mask, loss_mask = tokenize("pick", np.array([0.2]), np.array([[0.7]]))
print(tokens.tolist(), loss_mask.tolist())
print(extract(tokens, 1, 1).tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[4, 2, 7] [0, 0, 1]
[[0.699999988079071]]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA action tokenizer** / **OpenVLA action tokenizer**: 连续动作先离散成 token，再由 LM 预测。 / Continuous actions are discretized into tokens for LM prediction.
- **text tokenizer decode** / **text tokenizer decode**: encode/decode 必须共用同一词表和边界规则。 / encode and decode must share the same vocabulary and boundary rules.

## 注意事项 / Caveats / when it breaks

- **prompt 被 `pop` 掉 / Prompt is popped**: 下游 transform 如果还需要原始 prompt，必须提前复制。 / Downstream transforms needing raw prompt must copy it earlier.
- **action_horizon/action_dim 必须匹配 / Horizon and dimension must match**: 解码参数错了会把 token 边界切歪。 / Wrong decode parameters slice token boundaries incorrectly.
- **mask 不是装饰 / Masks are not decorative**: loss mask 错了，模型会学 prompt token 或 padding。 / A wrong loss mask trains on prompt tokens or padding.

## 延伸阅读 / Further reading

- [openpi `transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py)
- [openpi repository](https://github.com/Physical-Intelligence/openpi)
